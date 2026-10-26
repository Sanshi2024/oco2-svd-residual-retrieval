#!/usr/bin/env python3
"""Train the two manuscript target routes on fast fixed-channel GL PCA data."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import h5py
import joblib
import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


TRAIN_YEARS = (2015, 2016, 2017, 2018, 2019)
EVAL_YEARS = (2020, 2021, 2022, 2023, 2024, 2025)
ROUTES = ("absolute", "delta")
SEEDS = (17, 29, 43)


@dataclass
class Split:
    x: np.ndarray
    y: np.ndarray
    apriori: np.ndarray
    year: int | str
    role: str


class MLP(nn.Module):
    def __init__(self, input_dim: int = 39) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)


def path_for(root: Path, year: int) -> Path:
    return root / f"GL_FIXED_PCA10_{year}_200k.h5"


def load_year(root: Path, year: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with h5py.File(path_for(root, year), "r") as source:
        x = source["features"][:].astype(np.float32, copy=False)
        y = source["xco2"][:].astype(np.float32, copy=False)
        apriori = source["xco2_apriori"][:].astype(np.float32, copy=False)
    if x.shape != (200000, 39):
        raise ValueError(f"Unexpected {year} feature shape: {x.shape}")
    if not np.all(np.isfinite(x)):
        raise ValueError(f"Non-finite features in {year}")
    return x, y, apriori


def target_for(route: str, y: np.ndarray, apriori: np.ndarray) -> np.ndarray:
    return y if route == "absolute" else y - apriori


def absolute_prediction(route: str, prediction: np.ndarray, apriori: np.ndarray) -> np.ndarray:
    return prediction if route == "absolute" else apriori + prediction


def metrics(y: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    error = prediction - y
    return {
        "rmse": float(math.sqrt(mean_squared_error(y, prediction))),
        "mae": float(mean_absolute_error(y, prediction)),
        "bias": float(np.mean(error)),
        "r2": float(r2_score(y, prediction)),
    }


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def torch_predict(model: nn.Module, x: np.ndarray, device: torch.device, batch_size: int) -> np.ndarray:
    model.eval()
    output = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            block = torch.from_numpy(x[start : start + batch_size]).to(device, non_blocking=True)
            output.append(model(block).float().cpu().numpy())
    return np.concatenate(output)


def train_mlp(
    x_train: np.ndarray,
    target_train: np.ndarray,
    x_val: np.ndarray,
    target_val: np.ndarray,
    seed: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
) -> tuple[MLP, dict]:
    set_seed(seed)
    target_mean = float(np.mean(target_train))
    target_std = float(np.std(target_train))
    if target_std == 0:
        raise ValueError("Training target has zero standard deviation")
    scaled_target = ((target_train - target_mean) / target_std).astype(np.float32)
    dataset = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(scaled_target))
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=device.type == "cuda",
        generator=generator,
    )
    model = MLP(x_train.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    best_rmse = float("inf")
    best_epoch = -1
    best_state = None
    waiting = 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        running = 0.0
        rows = 0
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(xb), yb)
            loss.backward()
            optimizer.step()
            running += float(loss.detach()) * len(xb)
            rows += len(xb)
        pred_scaled = torch_predict(model, x_val, device, batch_size * 2)
        pred = pred_scaled * target_std + target_mean
        val_rmse = float(np.sqrt(np.mean((pred - target_val) ** 2)))
        history.append({"epoch": epoch, "train_scaled_mse": running / rows, "val_rmse": val_rmse})
        print(f"MLP seed={seed} epoch={epoch} val_rmse={val_rmse:.5f}", flush=True)
        if val_rmse < best_rmse - 1e-5:
            best_rmse = val_rmse
            best_epoch = epoch
            best_state = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
            waiting = 0
        else:
            waiting += 1
            if waiting >= patience:
                break
    if best_state is None:
        raise RuntimeError("MLP training did not produce a checkpoint")
    model.load_state_dict(best_state)
    model.to(device)
    return model, {
        "target_mean": target_mean,
        "target_std": target_std,
        "best_epoch": best_epoch,
        "best_val_rmse": best_rmse,
        "history": history,
    }


def record_metrics(
    rows: list[dict],
    route: str,
    family: str,
    seed: int | str,
    model_id: str,
    split: Split,
    native_prediction: np.ndarray,
    train_seconds: float,
) -> None:
    prediction = absolute_prediction(route, native_prediction, split.apriori)
    row = {
        "route": route,
        "family": family,
        "seed": seed,
        "model_id": model_id,
        "role": split.role,
        "year": split.year,
        "n": len(split.y),
        "train_seconds": train_seconds,
    }
    row.update(metrics(split.y, prediction))
    rows.append(row)


def save_metrics(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate_metrics(rows: list[dict], path: Path) -> list[dict]:
    groups: dict[tuple, list[dict]] = {}
    for row in rows:
        key = (row["route"], row["family"], row["role"], row["year"])
        groups.setdefault(key, []).append(row)
    output = []
    for key, values in sorted(groups.items(), key=lambda item: tuple(map(str, item[0]))):
        route, family, role, year = key
        row = {
            "route": route,
            "family": family,
            "role": role,
            "year": year,
            "n": values[0]["n"],
            "runs": len(values),
        }
        for name in ("rmse", "mae", "bias", "r2", "train_seconds"):
            numbers = np.asarray([value[name] for value in values], dtype=float)
            row[f"{name}_mean"] = float(numbers.mean())
            row[f"{name}_std"] = float(numbers.std(ddof=0))
        output.append(row)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(output[0]))
        writer.writeheader()
        writer.writerows(output)
    return output


def save_rmse_figure(summary: list[dict], path: Path) -> None:
    colors = {"Ridge": "#0072B2", "LightGBM": "#009E73", "MLP": "#D55E00"}
    markers = {"Ridge": "o", "LightGBM": "s", "MLP": "^"}
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.2), sharey=True, constrained_layout=True)
    for ax, route in zip(axes, ROUTES):
        for family in colors:
            values = [
                row
                for row in summary
                if row["route"] == route
                and row["family"] == family
                and row["role"] in {"validation", "test"}
            ]
            values.sort(key=lambda row: int(row["year"]))
            ax.errorbar(
                [int(row["year"]) for row in values],
                [row["rmse_mean"] for row in values],
                yerr=[row["rmse_std"] for row in values],
                label=family,
                color=colors[family],
                marker=markers[family],
                linewidth=2,
                capsize=2,
            )
        ax.axvline(2020.5, color="#777777", linestyle="--", linewidth=1)
        ax.set_title("Direct XCO$_2$" if route == "absolute" else "Prior-referenced residual", fontsize=13)
        ax.set_xlabel("Evaluation year", fontsize=11)
        ax.set_xticks(EVAL_YEARS)
        ax.tick_params(labelsize=9)
        ax.grid(color="#DDDDDD", linewidth=0.7)
        ax.spines[["top", "right"]].set_visible(False)
    axes[0].set_ylabel("RMSE (ppm)", fontsize=11)
    axes[1].legend(frameon=False, fontsize=10)
    fig.savefig(path.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--projected-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=8192)
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=12)
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    complete_path = args.output_dir / "complete.json"
    if complete_path.exists():
        print(f"already complete: {complete_path}")
        return

    train_parts = [load_year(args.projected_root, year) for year in TRAIN_YEARS]
    x_train = np.concatenate([part[0] for part in train_parts], axis=0)
    y_train = np.concatenate([part[1] for part in train_parts])
    apriori_train = np.concatenate([part[2] for part in train_parts])

    feature_mean = x_train.mean(axis=0, dtype=np.float64).astype(np.float32)
    feature_std = x_train.std(axis=0, dtype=np.float64).astype(np.float32)
    feature_std[feature_std == 0] = 1.0
    x_train -= feature_mean
    x_train /= feature_std
    np.savez_compressed(args.output_dir / "feature_standard_scaler_1519.npz", mean=feature_mean, scale=feature_std)

    rng = np.random.default_rng(20261013)
    train_eval_index = np.sort(rng.choice(len(x_train), size=200000, replace=False))
    splits: dict[str, Split] = {
        "train": Split(
            x_train[train_eval_index],
            y_train[train_eval_index],
            apriori_train[train_eval_index],
            "2015-2019",
            "train",
        )
    }
    for year in EVAL_YEARS:
        x, y, apriori = load_year(args.projected_root, year)
        x -= feature_mean
        x /= feature_std
        splits[str(year)] = Split(x, y, apriori, year, "validation" if year == 2020 else "test")

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    rows: list[dict] = []
    model_root = args.output_dir / "models"
    model_root.mkdir(exist_ok=True)

    for route in ROUTES:
        target_train = target_for(route, y_train, apriori_train)
        target_val = target_for(route, splits["2020"].y, splits["2020"].apriori)

        ridge_start = time.perf_counter()
        ridge_candidates = []
        for alpha in (0.01, 0.1, 1.0, 10.0, 100.0):
            model = Ridge(alpha=alpha, solver="lsqr", tol=1e-5)
            model.fit(x_train, target_train)
            prediction = model.predict(splits["2020"].x)
            ridge_candidates.append((float(np.sqrt(np.mean((prediction - target_val) ** 2))), alpha, model))
        _, alpha, ridge = min(ridge_candidates, key=lambda item: item[0])
        ridge_seconds = time.perf_counter() - ridge_start
        ridge_id = f"{route}_ridge_alpha_{alpha:g}"
        joblib.dump(ridge, model_root / f"{ridge_id}.joblib")
        for split in splits.values():
            record_metrics(rows, route, "Ridge", 0, ridge_id, split, ridge.predict(split.x), ridge_seconds)

        for seed in SEEDS:
            lgb_start = time.perf_counter()
            booster = lgb.LGBMRegressor(
                objective="regression_l2",
                n_estimators=3000,
                learning_rate=0.03,
                num_leaves=31,
                max_depth=-1,
                min_child_samples=100,
                subsample=0.8,
                subsample_freq=1,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                random_state=seed,
                n_jobs=16,
                verbosity=-1,
            )
            booster.fit(
                x_train,
                target_train,
                eval_set=[(splits["2020"].x, target_val)],
                eval_metric="rmse",
                callbacks=[lgb.early_stopping(100, verbose=False)],
            )
            lgb_seconds = time.perf_counter() - lgb_start
            model_id = f"{route}_lightgbm_seed_{seed}"
            booster.booster_.save_model(str(model_root / f"{model_id}.txt"))
            for split in splits.values():
                prediction = booster.predict(split.x, num_iteration=booster.best_iteration_)
                record_metrics(rows, route, "LightGBM", seed, model_id, split, prediction, lgb_seconds)

        for seed in SEEDS:
            mlp_start = time.perf_counter()
            model, info = train_mlp(
                x_train,
                target_train,
                splits["2020"].x,
                target_val,
                seed,
                device,
                args.batch_size,
                args.max_epochs,
                args.patience,
            )
            mlp_seconds = time.perf_counter() - mlp_start
            model_id = f"{route}_mlp_seed_{seed}"
            torch.save(
                {"state_dict": model.state_dict(), "training": info, "route": route, "seed": seed},
                model_root / f"{model_id}.pt",
            )
            (model_root / f"{model_id}.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
            for split in splits.values():
                scaled = torch_predict(model, split.x, device, args.batch_size * 2)
                prediction = scaled * info["target_std"] + info["target_mean"]
                record_metrics(rows, route, "MLP", seed, model_id, split, prediction, mlp_seconds)
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

    save_metrics(rows, args.output_dir / "metrics_runs.csv")
    summary = aggregate_metrics(rows, args.output_dir / "metrics_summary.csv")
    save_rmse_figure(summary, args.output_dir / "annual_rmse")
    best = {}
    for route in ROUTES:
        candidates = [
            row for row in summary if row["route"] == route and row["role"] == "validation"
        ]
        best[route] = min(candidates, key=lambda row: row["rmse_mean"])
    complete = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "training_years": list(TRAIN_YEARS),
        "validation_year": 2020,
        "test_years": list(EVAL_YEARS[1:]),
        "rows_per_year": 200000,
        "feature_count": 39,
        "mlp_parameters": sum(parameter.numel() for parameter in MLP().parameters()),
        "routes": list(ROUTES),
        "families": ["Ridge", "LightGBM", "MLP"],
        "best_validation_family": best,
        "device": str(device),
    }
    complete_path.write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
