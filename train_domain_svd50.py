#!/usr/bin/env python3
"""Train independent surface-domain regressors for absolute and residual XCO2."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import time

import h5py
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge
import torch

import two_route_training_reference as reference


DOMAINS = ("ND", "GL_OCEAN", "GL_LAND")
FEATURE_NAMES = (
    [f"PC_{band}_{component:02d}" for band in ("O2", "W", "S") for component in range(1, 31)]
    + ["sec_solar_zenith", "sec_sensor_zenith", "sin_AZA", "cos_AZA",
       "solar_distance", "solar_relative_velocity", "pressure_apriori_fph", "pressure_apriori_abp"]
)


def save_json(path: Path, value) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2), encoding="utf-8")
    os.replace(partial, path)


def secant_zenith(degrees: np.ndarray) -> np.ndarray:
    values = np.asarray(degrees, dtype="f8")
    valid = np.isfinite(values) & (values >= 0) & (values < 90)
    result = np.full(values.shape, np.nan, dtype="f8")
    result[valid] = 1.0 / np.cos(np.deg2rad(values[valid]))
    return result.astype("f4")


def load_year(root: Path, year: int, domain: str, limit: int | None = None,
              include_xco2_prior: bool = False) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(root / f"projected_{year}.h5") as source:
        assert source.attrs.get("status") == "complete"
        keep = np.flatnonzero(source["domain"][:] == domain.encode())
        if limit is not None:
            keep = keep[:limit]
        # Read contiguous domain blocks rather than scattered HDF5 point selections.
        def read(name):
            return source[name][:][keep]
        features = [read(f"pc_{band}").astype("f4") for band in ("O2", "W", "S")]
        assert all(value.shape == (len(keep), 30) for value in features)
        ids = read("sounding_id")
        assert len(ids) > 0 and len(np.unique(ids)) == len(ids)
        expected_mode = b"ND" if domain == "ND" else b"GL"
        assert np.all(read("mode") == expected_mode)
        fields = {name: read(name).astype("f4").reshape(-1) for name in (
            "solar_zenith", "sensor_zenith", "AZA", "retrieval_solar_distance",
            "retrieval_solar_relative_velocity", "surface_pressure_apriori_fph",
            "surface_pressure_apriori_abp",
        )}
        for values in fields.values():
            values[(~np.isfinite(values)) | (values <= -9999)] = np.nan
        extras = np.column_stack([
            secant_zenith(fields["solar_zenith"]),
            secant_zenith(fields["sensor_zenith"]),
            np.sin(np.deg2rad(fields["AZA"])), np.cos(np.deg2rad(fields["AZA"])),
            fields["retrieval_solar_distance"], fields["retrieval_solar_relative_velocity"],
            fields["surface_pressure_apriori_fph"], fields["surface_pressure_apriori_abp"],
        ])
        # Missing ancillary predictors are imputed only using training-year medians.
        extras[(~np.isfinite(extras)) | (extras < -9999)] = np.nan
        for column, field in ((6, "surface_pressure_apriori_fph"), (7, "surface_pressure_apriori_abp")):
            extras[fields[field] <= 0, column] = np.nan
        x = np.concatenate([*features, extras], axis=1).astype("f4")
        assert x.shape[1] == len(FEATURE_NAMES)
        assert np.all(np.isfinite(x[:, :90]))
        y = read("xco2").astype("f4").reshape(-1)
        prior = read("xco2_apriori").astype("f4").reshape(-1)
        assert np.all(np.isfinite(y)) and np.all(np.isfinite(prior))
        assert np.all(y > 0) and np.all(prior > 0)
        if include_xco2_prior:
            x = np.column_stack((x, prior)).astype("f4")
    return x, y, prior


def evaluate(model_id, route, family, seed, splits, predict, train_seconds, complexity,
             prediction_root=None, identities=None) -> list[dict]:
    rows = []
    for split in splits.values():
        start = time.perf_counter()
        native = predict(split.x)
        seconds = time.perf_counter() - start
        absolute = reference.absolute_prediction(route, native, split.apriori)
        assert native.shape == split.y.shape and np.all(np.isfinite(native))
        if prediction_root is not None:
            identity = identities[str(split.year)]
            np.savez_compressed(prediction_root / f"predictions_{split.year}.npz",
                                sounding_id=identity[0], original_index=identity[1],
                                reference_xco2=split.y, xco2_apriori=split.apriori,
                                predicted_target=native, predicted_xco2=absolute)
        target = reference.target_for(route, split.y, split.apriori)
        row = {
            "route": route, "family": family, "seed": seed, "model_id": model_id,
            "role": split.role, "year": split.year, "n": len(split.y),
            "train_seconds": train_seconds, "inference_seconds": seconds,
            "microseconds_per_row": seconds * 1e6 / len(split.y),
            "model_complexity": complexity,
        }
        row.update(reference.metrics(split.y, absolute))
        row["target_space_r2"] = reference.metrics(target, native)["r2"]
        rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--route", choices=reference.ROUTES)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    if (args.root / "SPECTRAL_ONLY.json").exists() and args.output_root is None:
        print("Model training is disabled: preserve fitted SVD bases and all saved PC scores.", flush=True)
        return
    torch.set_num_threads(4)
    root = args.root
    output_root = args.output_root or root
    if args.output_root is not None and args.output_root.resolve() == root.resolve():
        raise ValueError("Keep the model output separate from the completed spectral output")
    output = output_root / ("smoke_models" if args.smoke else "models") / args.domain
    if args.route:
        output /= args.route
    output.mkdir(parents=True, exist_ok=True)
    import fcntl
    lock = (output / "worker.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print(f"Already running: {output}", flush=True)
        return
    save_json(output / "worker.json", {"pid": os.getpid(), "domain": args.domain, "route": args.route})
    source_manifest = root / "spectral.complete.json"
    config = {"source_root": str(root.resolve()), "domain": args.domain,
              "route": args.route, "smoke": args.smoke,
              "feature_names": FEATURE_NAMES, "seeds": [17] if args.smoke else list(reference.SEEDS),
              "training_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "model_reference_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest(),
              "source_manifest_sha256": hashlib.sha256(source_manifest.read_bytes()).hexdigest()
              if source_manifest.exists() else "test-fixture"}
    config_path = output / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("Existing outputs use a different source or training configuration")
    save_json(config_path, config)
    if (output / "complete.json").exists():
        print(f"Already complete: {output}", flush=True)
        return
    print(f"Loading {args.domain}/{args.route or 'both'} training data", flush=True)
    limit = 300 if args.smoke else None
    parts = [load_year(root, year, args.domain, limit) for year in range(2015, 2020)]
    x_train = np.concatenate([part[0] for part in parts])
    y_train = np.concatenate([part[1] for part in parts])
    prior_train = np.concatenate([part[2] for part in parts])
    del parts
    save_json(output / "training_missingness.json", {
        name: int(count) for name, count in zip(FEATURE_NAMES, np.sum(~np.isfinite(x_train), axis=0))
    })
    median = np.nanmedian(x_train, axis=0)
    median[~np.isfinite(median)] = 0
    missing = ~np.isfinite(x_train)
    x_train[missing] = np.broadcast_to(median, x_train.shape)[missing]
    mean = x_train.mean(axis=0, dtype="f8").astype("f4")
    scale = x_train.std(axis=0, dtype="f8").astype("f4")
    scale[scale == 0] = 1
    x_train -= mean
    x_train /= scale
    np.savez_compressed(output / "training_only_scaler.npz", median=median, mean=mean, scale=scale)
    save_json(output / "feature_names.json", FEATURE_NAMES)
    splits = {"train": reference.Split(x_train, y_train, prior_train, "2015-2019", "train")}
    # Hold future-year arrays unopened until the 2020-selected models are frozen.
    for year in (2020,):
        x, y, prior = load_year(root, year, args.domain, limit)
        missing = ~np.isfinite(x)
        x[missing] = np.broadcast_to(median, x.shape)[missing]
        x -= mean
        x /= scale
        splits[str(year)] = reference.Split(x, y, prior, year, "validation" if year == 2020 else "test")
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    seeds = (17,) if args.smoke else reference.SEEDS
    rows = []
    routes = (args.route,) if args.route else reference.ROUTES
    fitted = []
    for route in routes:
        target_train = reference.target_for(route, y_train, prior_train)
        validation = splits["2020"]
        target_val = reference.target_for(route, validation.y, validation.apriori)
        runs = [("Ridge", 0)] + [(family, seed) for family in ("LightGBM", "MLP") for seed in seeds]
        for family, seed in runs:
            model_id = f"{route}_{family.lower()}_{seed}"
            run = output / model_id
            run.mkdir(exist_ok=True)
            completion = run / "complete.json"
            if completion.exists():
                rows.extend(json.loads(completion.read_text())["metrics"])
                fitted.append((route, family, seed, model_id, run))
                continue
            print(f"Training {args.domain}/{model_id}", flush=True)
            start = time.perf_counter()
            if family == "Ridge":
                best = None
                candidates = []
                # Correlated PC/geometry predictors need double-precision normal equations.
                ridge_train = x_train.astype("f8")
                ridge_validation = validation.x.astype("f8")
                for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
                    model = Ridge(alpha=alpha, solver="cholesky")
                    model.fit(ridge_train, target_train.astype("f8"))
                    error = float(np.sqrt(np.mean((model.predict(ridge_validation) - target_val) ** 2)))
                    candidates.append({"alpha": alpha, "validation_rmse": error})
                    if best is None or error < best[0]:
                        best = (error, model)
                model = best[1]
                joblib.dump(model, run / "model.joblib")
                save_json(run / "validation_search.json", candidates)
                predict = model.predict
                complexity = int(model.coef_.size + 1)
                del ridge_train, ridge_validation
            elif family == "LightGBM":
                model = lgb.LGBMRegressor(
                    objective="regression_l2", n_estimators=10 if args.smoke else 3000,
                    learning_rate=0.03, num_leaves=31, min_child_samples=100,
                    subsample=0.8, subsample_freq=1, colsample_bytree=0.9,
                    reg_lambda=1.0, random_state=seed, n_jobs=4, verbosity=-1,
                )
                model.fit(x_train, target_train, eval_set=[(validation.x, target_val)],
                          eval_metric="rmse", callbacks=[lgb.early_stopping(100, verbose=False)])
                model.booster_.save_model(str(run / "model.txt"))
                predict = model.predict
                complexity = int(sum(tree["num_leaves"] for tree in model.booster_.dump_model()["tree_info"]))
            else:
                model, info = reference.train_mlp(
                    x_train, target_train, validation.x, target_val, seed, device,
                    8192, 2 if args.smoke else 100, 12,
                )
                torch.save({"state_dict": model.state_dict(), "training": info,
                            "feature_count": len(FEATURE_NAMES), "route": route}, run / "model.pt")
                save_json(run / "training_history.json", info)
                predict = lambda x: reference.torch_predict(model, x, device, 16384) * info["target_std"] + info["target_mean"]
                complexity = sum(parameter.numel() for parameter in model.parameters())
            seconds = time.perf_counter() - start
            metrics = evaluate(model_id, route, family, seed, splits, predict, seconds, complexity)
            save_json(completion, {"status": "complete", "smoke": args.smoke, "metrics": metrics})
            rows.extend(metrics)
            fitted.append((route, family, seed, model_id, run))
            reference.save_metrics(rows, output / "metrics_runs.csv")
            print(f"Completed {args.domain}/{model_id} in {seconds:.1f}s", flush=True)
            del model
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    summary = reference.aggregate_metrics(rows, output / "validation_metrics_summary.csv")
    selected = {
        route: min([row for row in summary if row["route"] == route and row["role"] == "validation"], key=lambda row: row["rmse_mean"])
        for route in routes
    }
    save_json(output / "selection_frozen.json", {
        "selected_on": "2020 validation only", "validation_selected": selected,
        "components_per_band": 30, "feature_count": len(FEATURE_NAMES),
    })
    identities = {}
    for year in range(2021, 2026):
        x, y, prior = load_year(root, year, args.domain, limit)
        missing = ~np.isfinite(x)
        x[missing] = np.broadcast_to(median, x.shape)[missing]
        x -= mean
        x /= scale
        splits[str(year)] = reference.Split(x, y, prior, year, "test")
        with h5py.File(root / f"projected_{year}.h5") as source:
            keep = np.flatnonzero(source["domain"][:] == args.domain.encode())
            if limit is not None:
                keep = keep[:limit]
            identities[str(year)] = (source["sounding_id"][:][keep], source["rows"][:][keep])
    baseline = [{"year": split.year, "role": split.role, "n": len(split.y),
                 **reference.metrics(split.y, split.apriori)} for split in splits.values()]
    save_json(output / "prior_baseline.json", baseline)
    for route, family, seed, model_id, run in fitted:
        evaluation = run / "evaluation.complete.json"
        if evaluation.exists():
            rows.extend(json.loads(evaluation.read_text())["metrics"])
            continue
        trained = json.loads((run / "complete.json").read_text())
        info_row = trained["metrics"][0]
        if family == "Ridge":
            model = joblib.load(run / "model.joblib")
            predict = model.predict
        elif family == "LightGBM":
            model = lgb.Booster(model_file=str(run / "model.txt"))
            predict = lambda x: model.predict(x, num_threads=4)
        else:
            checkpoint = torch.load(run / "model.pt", map_location=device, weights_only=False)
            model = reference.MLP(len(FEATURE_NAMES)).to(device)
            model.load_state_dict(checkpoint["state_dict"])
            info = checkpoint["training"]
            predict = lambda x: reference.torch_predict(model, x, device, 16384) * info["target_std"] + info["target_mean"]
        metrics = evaluate(model_id, route, family, seed,
                           {key: value for key, value in splits.items() if value.role == "test"},
                           predict, info_row["train_seconds"], info_row["model_complexity"],
                           run, identities)
        save_json(evaluation, {"status": "complete", "metrics": metrics})
        rows.extend(metrics)
        del model
    reference.save_metrics(rows, output / "metrics_runs.csv")
    summary = reference.aggregate_metrics(rows, output / "metrics_summary.csv")
    save_json(output / "complete.json", {
        "status": "complete", "smoke": args.smoke, "domain": args.domain,
        "training_years": list(range(2015, 2020)), "validation_year": 2020,
        "test_years": list(range(2021, 2026)), "training_rows": len(y_train),
        "annual_eval_counts": {key: len(split.y) for key, split in splits.items()},
        "feature_count": len(FEATURE_NAMES), "spectral_features": 90,
        "components_per_band": 30, "source_root": str(root),
        "normalization_amplitudes_used_as_predictors": False,
        "models": len(set(row["model_id"] for row in rows)),
        "validation_selected": selected,
    })


if __name__ == "__main__":
    main()
