#!/usr/bin/env python3
"""CPU Ridge/LightGBM controls sharing the audited 99-input neural cohorts."""

import argparse
import fcntl
import json
import os
from pathlib import Path
import time

import h5py
import joblib
import lightgbm as lgb
import numpy as np
from sklearn.linear_model import Ridge

import search_mlp_secant as engine
import train_domain_svd50 as features
import two_route_training_reference as reference


def evaluation_data(args, data):
    if args.regional_spec:
        import east_asia_prior99 as regional
        return regional.evaluation_splits(args, data)
    splits, identities = {}, {}
    for year in range(2021, 2026):
        x, y, prior = features.load_year(args.source, year, args.domain, 40 if args.smoke else None, True)
        engine.standardize(x, data.median, data.mean, data.scale)
        splits[str(year)] = reference.Split(x, y, prior, year, "test")
        with h5py.File(args.source / f"projected_{year}.h5") as source:
            keep = np.flatnonzero(source["domain"][:] == args.domain.encode())
            if args.smoke:
                keep = keep[:40]
            identities[str(year)] = source["sounding_id"][:][keep], source["rows"][:][keep]
    return splits, identities


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="Shared neural/cohort output root")
    parser.add_argument("--tree-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--regional-spec", type=Path)
    parser.add_argument("--domain", choices=features.DOMAINS, required=True)
    parser.add_argument("--route", choices=reference.ROUTES, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.include_xco2_prior = True
    assert args.tree_root.resolve() != args.source.resolve()
    folder = args.tree_root / args.domain / args.route
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / "worker.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / "complete.json").exists():
            return
        engine.write_json(folder / "worker.json", {"pid": os.getpid(), "domain": args.domain, "route": args.route})
        data = engine.prepare(args, folder)
        assert data.x.shape[1] == data.val_x.shape[1] == 99
        validation_key = "historical_validation" if args.regional_spec else "2020"
        splits = {
            "train": reference.Split(data.x, data.y, data.prior,
                                      "2017-2019" if args.regional_spec else "2015-2019", "train"),
            validation_key: reference.Split(data.val_x, data.val_y, data.val_prior, validation_key, "validation"),
        }
        y = reference.target_for(args.route, data.y, data.prior)
        val_y = reference.target_for(args.route, data.val_y, data.val_prior)
        models = []
        for family, seed in [("Ridge", 0), *[("LightGBM", s) for s in ((17,) if args.smoke else engine.SEEDS)]]:
            run = folder / f"{family}_{seed}"
            run.mkdir(exist_ok=True)
            done = run / "trained.json"
            if done.exists():
                models.append(json.loads(done.read_text()))
                continue
            started = time.perf_counter()
            if family == "Ridge":
                candidates, best = [], None
                x64, val64 = data.x.astype("f8"), data.val_x.astype("f8")
                for alpha in (0.01, 0.1, 1., 10., 100.):
                    candidate = Ridge(alpha=alpha, solver="cholesky").fit(x64, y.astype("f8"))
                    error = float(np.sqrt(np.mean((candidate.predict(val64) - val_y) ** 2)))
                    candidates.append({"alpha": alpha, "validation_rmse": error})
                    if best is None or error < best[0]:
                        best = error, candidate
                model = best[1]
                joblib.dump(model, run / "model.joblib")
                engine.write_json(run / "validation_search.json", candidates)
                complexity = int(model.coef_.size + 1)
                del x64, val64
                validation_rmse = best[0]
            else:
                model = lgb.LGBMRegressor(objective="regression_l2", n_estimators=10 if args.smoke else 3000,
                    learning_rate=.03, num_leaves=31, min_child_samples=10 if args.smoke else 100,
                    subsample=.8, subsample_freq=1, colsample_bytree=.9,
                    reg_lambda=1., random_state=seed, n_jobs=4, verbosity=-1)
                model.fit(data.x, y, eval_set=[(data.val_x, val_y)], eval_metric="rmse",
                          callbacks=[lgb.early_stopping(100, verbose=False)])
                model.booster_.save_model(str(run / "model.txt"))
                validation_rmse = float(np.sqrt(np.mean((model.predict(data.val_x) - val_y) ** 2)))
                complexity = int(sum(t["num_leaves"] for t in model.booster_.dump_model()["tree_info"]))
            info = {"family": family, "seed": seed, "run": str(run), "parameters": complexity,
                    "train_seconds": time.perf_counter() - started, "validation_rmse": validation_rmse,
                    "feature_count": 99, "smoke": args.smoke}
            engine.write_json(done, info)
            models.append(info)
            print(f"Trained {args.domain}/{args.route}/{family}/{seed}: validation={validation_rmse:.5f}", flush=True)
            del model
        engine.write_json(folder / "selection_frozen.json", {
            "selected_on": validation_key, "test_metrics_used_for_selection": False, "models": models})
        # Regional future data must remain unopened until its neural searches are also frozen.
        if args.regional_spec:
            while not (args.output / "search_selection.complete.json").exists():
                if (args.output / "pipeline.error.json").exists():
                    raise RuntimeError("Regional neural search failed; preserve trained tree models")
                time.sleep(30)
        test_splits, identities = evaluation_data(args, data)
        splits.update(test_splits)
        rows = []
        for info in models:
            run = Path(info["run"])
            completed = run / "complete.json"
            if completed.exists():
                rows.extend(json.loads(completed.read_text())["metrics"])
                continue
            if info["family"] == "Ridge":
                model = joblib.load(run / "model.joblib")
                predict = model.predict
            else:
                model = lgb.Booster(model_file=str(run / "model.txt"))
                predict = lambda x: model.predict(x, num_threads=4)
            identity = dict(identities)
            metrics = features.evaluate(f"{args.route}_{info['family']}_{info['seed']}", args.route,
                info["family"], info["seed"], {k:v for k,v in splits.items() if v.role != "test"},
                predict, info["train_seconds"], info["parameters"])
            metrics += features.evaluate(f"{args.route}_{info['family']}_{info['seed']}", args.route,
                info["family"], info["seed"], test_splits, predict, info["train_seconds"], info["parameters"], run, identity)
            engine.write_json(completed, {"status": "complete", "metrics": metrics})
            rows.extend(metrics)
        engine.write_csv(folder / "metrics_runs.csv", rows)
        summary = reference.aggregate_metrics(rows, folder / "metrics_summary.csv")
        engine.write_json(folder / "prior_baseline.json", [
            {"year": s.year, "role": s.role, "n": len(s.y), **reference.metrics(s.y, s.apriori)} for s in splits.values()])
        engine.write_json(folder / "complete.json", {"status":"complete", "feature_count":99,
            "training_rows":len(data.y), "validation_rows":len(data.val_y), "models":len(models),
            "summary_rows":len(summary), "smoke":args.smoke})


if __name__ == "__main__":
    main()
