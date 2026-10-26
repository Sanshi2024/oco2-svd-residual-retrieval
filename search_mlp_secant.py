#!/usr/bin/env python3
"""Validation-only Optuna search for the frozen three-domain SVD30 features."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import fcntl
import hashlib
import json
import math
import os
import pickle
import sys
from pathlib import Path
import time

import h5py
import numpy as np
import optuna
from optuna.trial import TrialState
import torch
from torch import nn

import train_domain_svd50 as features
import two_route_training_reference as reference


ARCHITECTURES = ("64-64", "128-64", "256-128", "256-256-128", "512-256-128",
                 "512-512-256", "256-256-256-128")
SEEDS = (17, 29, 43)
BASELINE_HP = {"architecture": "128-64", "activation": "gelu", "normalization": "none",
               "dropout": 0.1, "lr": 0.001, "weight_decay": 0.0001, "batch_size": 8192,
               "scheduler": "constant", "warmup_epochs": 0}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    features.save_json(path, value)


def write_csv(path, rows):
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".partial")
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with temp.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temp, path)


def atomic_torch(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    torch.save(value, partial)
    os.replace(partial, path)


def hp_id(hp):
    return hashlib.sha256(json.dumps(hp, sort_keys=True).encode()).hexdigest()[:16]


class DenseMLP(nn.Module):
    def __init__(self, input_dim, hp):
        super().__init__()
        activation = {"relu": nn.ReLU, "gelu": nn.GELU, "silu": nn.SiLU}[hp["activation"]]
        layers = []
        widths = list(map(int, hp["architecture"].split("-")))
        for index, width in enumerate(widths):
            layers.append(nn.Linear(input_dim, width))
            if hp["normalization"] == "layernorm":
                layers.append(nn.LayerNorm(width))
            layers.append(activation())
            if hp["dropout"] and index < len(widths) - 1:
                layers.append(nn.Dropout(hp["dropout"]))
            input_dim = width
        layers.append(nn.Linear(input_dim, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x).squeeze(-1)


def suggest_hp(trial):
    return {
        "architecture": trial.suggest_categorical("architecture", list(ARCHITECTURES)),
        "activation": trial.suggest_categorical("activation", ["gelu", "relu", "silu"]),
        "normalization": trial.suggest_categorical("normalization", ["none", "layernorm"]),
        "dropout": trial.suggest_float("dropout", 0.0, 0.3, step=0.05),
        "lr": trial.suggest_float("lr", 3e-5, 3e-3, log=True),
        "weight_decay": trial.suggest_categorical("weight_decay", [0.0, 1e-6, 1e-5, 1e-4, 1e-3]),
        "batch_size": trial.suggest_categorical("batch_size", [1024, 2048, 4096, 8192, 16384]),
        "scheduler": trial.suggest_categorical("scheduler", ["constant", "plateau", "cosine"]),
        "warmup_epochs": trial.suggest_categorical("warmup_epochs", [0, 5]),
    }


def hidden_search_hp(trial, fixed):
    depth = trial.suggest_int("depth", 1, 4)
    widths = [trial.suggest_int(f"width_{index + 1}", 32, 640, step=4) for index in range(depth)]
    return {**fixed, "architecture": "-".join(map(str, widths))}


def architecture_params(architecture):
    widths = list(map(int, architecture.split("-")))
    assert 1 <= len(widths) <= 4
    assert all(32 <= width <= 640 and width % 4 == 0 for width in widths)
    return {"depth": len(widths), **{f"width_{i + 1}": width for i, width in enumerate(widths)}}


def fixed_hidden_recipe(path, domain, route):
    config = json.loads(path.read_text())
    assert config["selection_year"] in (2020, "2017-2019 random validation")
    records = [row for row in config["jobs"] if (row["domain"], row["route"]) == (domain, route)]
    assert len(records) == 1
    hp = records[0]["hp"]
    assert set(hp) == set(BASELINE_HP)
    architecture_params(hp["architecture"])
    return hp


def trial_hp(trial):
    return trial.user_attrs.get("hp", trial.params)


@dataclass
class Prepared:
    x: np.ndarray
    y: np.ndarray
    prior: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    val_prior: np.ndarray
    median: np.ndarray
    mean: np.ndarray
    scale: np.ndarray


def standardize(x, median, mean, scale):
    bad = ~np.isfinite(x)
    x[bad] = np.broadcast_to(median, x.shape)[bad]
    x -= mean
    x /= scale
    assert np.all(np.isfinite(x))
    return x


def prepare(args, folder):
    if args.regional_spec:
        import east_asia_prior99 as regional
        return regional.prepare(args, folder, sys.modules[__name__])
    limit = 40 if args.smoke else None
    parts = [features.load_year(args.source, year, args.domain, limit, args.include_xco2_prior)
             for year in range(2015, 2020)]
    x, y, prior = [np.concatenate([part[i] for part in parts]) for i in range(3)]
    del parts
    if args.baseline_root:
        old = args.baseline_root / "models" / args.domain / args.route
        assert json.loads((old / "feature_names.json").read_text()) == features.FEATURE_NAMES
        with np.load(old / "training_only_scaler.npz") as scaler:
            median, mean, scale = (scaler[name].copy() for name in ("median", "mean", "scale"))
        if args.include_xco2_prior:
            median = np.r_[median, np.median(prior)].astype("f4")
            mean = np.r_[mean, np.mean(prior, dtype="f8")].astype("f4")
            scale = np.r_[scale, max(float(np.std(prior, dtype="f8")), 1e-8)].astype("f4")
    else:
        assert args.smoke, "Formal runs must reuse the audited original training-only scaler"
        median = np.nanmedian(x, axis=0)
        median[~np.isfinite(median)] = 0
        filled = np.where(np.isfinite(x), x, median)
        mean = filled.mean(axis=0, dtype="f8").astype("f4")
        scale = filled.std(axis=0, dtype="f8").astype("f4")
        scale[scale == 0] = 1
    assert median.shape == mean.shape == scale.shape == (x.shape[1],)
    assert np.all(np.isfinite(scale)) and np.all(scale > 0)
    standardize(x, median, mean, scale)
    val_x, val_y, val_prior = features.load_year(args.source, 2020, args.domain, limit, args.include_xco2_prior)
    standardize(val_x, median, mean, scale)
    if not args.smoke:
        assert len(y) == 1000000 and len(val_y) == 200000
    np.savez_compressed(folder / "training_only_scaler.npz", median=median, mean=mean, scale=scale)
    write_json(folder / "feature_names.json", feature_names(args))
    return Prepared(x, y, prior, val_x, val_y, val_prior, median, mean, scale)


@torch.no_grad()
def gpu_rmse(model, x, native_target, target_mean, target_std):
    model.eval()
    error = torch.zeros((), dtype=torch.float64, device=x.device)
    for start in range(0, len(x), 16384):
        predicted = model(x[start:start + 16384]).double() * target_std + target_mean
        error += torch.sum((predicted - native_target[start:start + 16384].double()) ** 2)
    return float(torch.sqrt(error / len(x)).item())


class Trainer:
    def __init__(self, args, data, folder, device):
        self.args, self.folder, self.device = args, folder, device
        target = reference.target_for(args.route, data.y, data.prior)
        val_target = reference.target_for(args.route, data.val_y, data.val_prior)
        self.target_mean = float(np.mean(target, dtype="f8"))
        self.target_std = float(np.std(target, dtype="f8"))
        assert self.target_std > 0
        # Each candidate uses the full million-row training set resident on its own GPU.
        self.x = torch.as_tensor(data.x, device=device)
        self.target = torch.as_tensor(((target - self.target_mean) / self.target_std).astype("f4"), device=device)
        self.val_x = torch.as_tensor(data.val_x, device=device)
        self.val_target = torch.as_tensor(val_target, device=device)

    def fit(self, hp, seed, trial=None):
        run = self.folder / "candidates" / hp_id(hp) / f"seed_{seed}"
        run.mkdir(parents=True, exist_ok=True)
        done = run / "complete.json"
        if done.exists():
            info = json.loads(done.read_text())
            assert info["hp"] == hp and info["seed"] == seed
            if info["status"] == "pruned" and trial is not None:
                trial.report(info["best_validation_rmse"], info["epochs"])
                raise optuna.TrialPruned()
            if info["status"] == "complete":
                return info
        reference.set_seed(seed)
        model = DenseMLP(self.x.shape[1], hp).to(self.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=hp["lr"], weight_decay=hp["weight_decay"])
        if hp["scheduler"] == "plateau":
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                                  patience=7, min_lr=1e-6)
        elif hp["scheduler"] == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=max(1, self.args.max_epochs - hp["warmup_epochs"]), eta_min=1e-6)
        else:
            scheduler = None
        generator = torch.Generator(device=self.device).manual_seed(seed)
        history, best, best_epoch, waiting, start_epoch, elapsed = [], float("inf"), 0, 0, 1, 0.0
        resume = run / "resume.pt"
        if resume.exists():
            state = torch.load(resume, map_location=self.device, weights_only=False)
            assert state["hp"] == hp and state["seed"] == seed
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])
            if scheduler is not None:
                scheduler.load_state_dict(state["scheduler"])
            generator.set_state(state["generator"].cpu())
            torch.set_rng_state(state["cpu_rng"].cpu())
            if self.device.type == "cuda":
                torch.cuda.set_rng_state(state["cuda_rng"].cpu(), self.device)
            history, best, best_epoch = state["history"], state["best"], state["best_epoch"]
            waiting, start_epoch, elapsed = state["waiting"], state["epoch"] + 1, state["elapsed"]
            print(f"Resume {hp_id(hp)} seed={seed} epoch={start_epoch}", flush=True)
            if trial is not None:
                for row in history:
                    trial.report(row["validation_rmse"], row["epoch"])
        started = time.perf_counter()
        pruned = False
        try:
            for epoch in range(start_epoch, self.args.max_epochs + 1):
                if epoch <= hp["warmup_epochs"]:
                    for group in optimizer.param_groups:
                        group["lr"] = hp["lr"] * epoch / hp["warmup_epochs"]
                current_lr = optimizer.param_groups[0]["lr"]
                model.train()
                order = torch.randperm(len(self.x), generator=generator, device=self.device)
                running = torch.zeros((), device=self.device)
                for start in range(0, len(order), hp["batch_size"]):
                    ix = order[start:start + hp["batch_size"]]
                    xb, yb = self.x[ix], self.target[ix]
                    optimizer.zero_grad(set_to_none=True)
                    loss = torch.mean((model(xb) - yb) ** 2)
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
                    optimizer.step()
                    running += loss.detach() * len(ix)
                training_loss = float((running / len(self.x)).item())
                validation = gpu_rmse(model, self.val_x, self.val_target, self.target_mean, self.target_std)
                if not math.isfinite(training_loss) or not math.isfinite(validation):
                    raise FloatingPointError("Nonfinite objective; retain other completed candidates")
                history.append({"epoch": epoch, "train_scaled_mse_dropout_on": training_loss,
                                "validation_rmse": validation, "lr": current_lr})
                if validation < best - 1e-5:
                    best, best_epoch, waiting = validation, epoch, 0
                    atomic_torch(run / "best.pt", {
                        "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
                        "hp": hp, "seed": seed, "best_epoch": best_epoch,
                        "target_mean": self.target_mean, "target_std": self.target_std,
                    })
                else:
                    waiting += 1
                if scheduler is not None and epoch > hp["warmup_epochs"]:
                    scheduler.step(validation) if hp["scheduler"] == "plateau" else scheduler.step()
                if epoch == 1 or epoch % 5 == 0:
                    print(f"{hp_id(hp)} seed={seed} epoch={epoch} val={validation:.5f} best={best:.5f} lr={current_lr:.3g}", flush=True)
                if epoch % 5 == 0 or epoch == self.args.max_epochs:
                    atomic_torch(resume, {
                        "hp": hp, "seed": seed, "model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "scheduler": scheduler.state_dict() if scheduler is not None else None,
                        "generator": generator.get_state(), "cpu_rng": torch.get_rng_state(),
                        "cuda_rng": torch.cuda.get_rng_state(self.device) if self.device.type == "cuda" else None,
                        "history": history, "best": best, "best_epoch": best_epoch,
                        "waiting": waiting, "epoch": epoch, "elapsed": elapsed + time.perf_counter() - started,
                    })
                    write_json(run / "history.json", history)
                if trial is not None:
                    trial.report(validation, epoch)
                    if epoch >= self.args.min_epochs and trial.should_prune():
                        pruned = True
                        break
                if epoch >= self.args.min_epochs and waiting >= self.args.patience:
                    break
            assert history and (run / "best.pt").exists()
            info = {"status": "pruned" if pruned else "complete", "hp": hp, "seed": seed,
                    "best_validation_rmse": best, "best_epoch": best_epoch, "epochs": len(history),
                    "train_seconds": elapsed + time.perf_counter() - started,
                    "parameters": sum(p.numel() for p in model.parameters()),
                    "target_mean": self.target_mean, "target_std": self.target_std, "run": str(run)}
            write_json(run / "history.json", history)
            write_json(done, info)
            if pruned:
                raise optuna.TrialPruned()
            return info
        finally:
            del model, optimizer, scheduler
            if self.device.type == "cuda":
                torch.cuda.empty_cache()


def search(args, folder, data, device):
    if (folder / "search.complete.json").exists():
        print("Search already complete", folder, flush=True)
        return
    trainer = Trainer(args, data, folder, device)
    hidden_only = args.architecture_only_config is not None
    fixed = fixed_hidden_recipe(args.architecture_only_config, args.domain, args.route) if hidden_only else None
    sampler_file = folder / "sampler.pickle"
    sampler = pickle.loads(sampler_file.read_bytes()) if sampler_file.exists() else optuna.samplers.TPESampler(
        seed=20261018, n_startup_trials=24, n_ei_candidates=64,
        multivariate=True, group=hidden_only, warn_independent_sampling=False)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=12, n_warmup_steps=40,
                                        interval_steps=10, n_min_trials=6)
    study = optuna.create_study(study_name="validation_only", direction="minimize", sampler=sampler,
                               pruner=pruner, storage=f"sqlite:///{folder / 'study.sqlite3'}", load_if_exists=True)
    if not study.trials:
        if hidden_only:
            anchors = [fixed["architecture"], "180-100", "300-180", "180", "300", "512",
                       "128-64", "400-240", "640-320", "180-120-60", "300-180-100",
                       "512-300-180", "180-180-100-60", "300-240-180-100"]
            for architecture in dict.fromkeys(anchors):
                study.enqueue_trial(architecture_params(architecture))
        else:
            study.enqueue_trial(BASELINE_HP)
            study.enqueue_trial({**BASELINE_HP, "architecture": "256-128", "lr": 3e-4,
                                 "dropout": 0.0, "scheduler": "plateau", "warmup_epochs": 5})
            study.enqueue_trial({**BASELINE_HP, "architecture": "512-256-128", "lr": 3e-4,
                                 "dropout": 0.05, "weight_decay": 1e-5, "scheduler": "cosine"})
    for old in study.get_trials(states=(TrialState.RUNNING,)):
        study.tell(old.number, state=TrialState.FAIL)
        study.enqueue_trial(old.params)
    def finished_candidates():
        trials = study.get_trials(states=(TrialState.COMPLETE, TrialState.PRUNED))
        return {hp_id(trial_hp(t)) for t in trials if not t.user_attrs.get("duplicate")}

    while len(finished_candidates()) < args.trials:
        assert len(study.get_trials(states=(TrialState.FAIL,))) < 24, "Too many failed trials; inspect logs"
        assert len(study.trials) < args.trials * 10, "Repeated candidates exhausted search attempts"
        trial = study.ask()
        hp = hidden_search_hp(trial, fixed) if hidden_only else suggest_hp(trial)
        trial.set_user_attr("hp", hp)
        trial.set_user_attr("candidate_id", hp_id(hp))
        if hidden_only and hp_id(hp) in finished_candidates():
            trial.set_user_attr("duplicate", True)
            study.tell(trial, state=TrialState.PRUNED)
            continue
        print(f"Trial {trial.number} {args.domain}/{args.route}: {json.dumps(hp)}", flush=True)
        try:
            info = trainer.fit(hp, 17, trial)
            study.tell(trial, info["best_validation_rmse"])
        except optuna.TrialPruned:
            study.tell(trial, state=TrialState.PRUNED)
        except (FloatingPointError, torch.cuda.OutOfMemoryError) as exc:
            trial.set_user_attr("failure", str(exc))
            study.tell(trial, state=TrialState.FAIL)
            if device.type == "cuda":
                torch.cuda.empty_cache()
        trials = study.get_trials()
        write_json(folder / "progress.json", {
            "phase": "search", "target_trials": args.trials,
            "unique_finished_candidates": len(finished_candidates()),
            "complete": sum(t.state == TrialState.COMPLETE for t in trials),
            "pruned": sum(t.state == TrialState.PRUNED for t in trials),
            "failed": sum(t.state == TrialState.FAIL for t in trials),
            "best_validation_rmse": study.best_value if any(t.state == TrialState.COMPLETE for t in trials) else None,
        })
        write_csv(folder / "search_results.csv", [
            {"trial": t.number, "state": t.state.name, "validation_rmse": t.value,
             "candidate_id": t.user_attrs.get("candidate_id"), "duplicate": t.user_attrs.get("duplicate", False),
             **trial_hp(t), **t.params} for t in trials])
        partial = sampler_file.with_suffix(".partial")
        partial.write_bytes(pickle.dumps(sampler))
        os.replace(partial, sampler_file)
    ranked = sorted(study.get_trials(states=(TrialState.COMPLETE,)), key=lambda t: t.value)
    unique = []
    for trial in ranked:
        if hp_id(trial_hp(trial)) not in [hp_id(trial_hp(t)) for t in unique]:
            unique.append(trial)
    assert len(unique) >= args.top_k
    confirmation = []
    for trial in unique[:args.top_k]:
        hp = trial_hp(trial)
        runs = [trainer.fit(hp, seed) for seed in ((17,) if args.smoke else SEEDS)]
        scores = [r["best_validation_rmse"] for r in runs]
        confirmation.append({"candidate_id": hp_id(hp), "hp": hp, "runs": runs,
                             "validation_rmse_mean": float(np.mean(scores)),
                             "validation_rmse_std": float(np.std(scores)), "initial_trial": trial.number})
        write_json(folder / "confirmation.json", confirmation)
    selected = min(confirmation, key=lambda c: c["validation_rmse_mean"])
    selected_on = "2017-2019 random validation only" if args.regional_spec else "2020 validation only"
    frozen = {"domain": args.domain, "route": args.route, "selected_on": selected_on,
              "selection_statistic": "mean validation RMSE across three seeds" if not args.smoke else "smoke only",
              "selected": selected, "smoke": args.smoke, "feature_count": data.x.shape[1],
              "components_per_band": 30, "test_metrics_used_for_selection": False}
    write_json(folder / "selection_frozen.json", frozen)
    write_json(folder / "search.complete.json", {
        "status": "complete", "smoke": args.smoke, "trials": args.trials,
        "confirmed_configurations": len(confirmation), "training_rows": len(data.y),
        "unique_finished_candidates": len(finished_candidates()),
        "validation_rows": len(data.val_y), "selected": selected,
    })
    del trainer


def evaluate_selected(args, folder, data, device):
    assert (args.output / "search_selection.complete.json").exists(), "Freeze all six searches before test evaluation"
    if (folder / "complete.json").exists():
        return
    frozen = json.loads((folder / "selection_frozen.json").read_text())
    selected = frozen["selected"]
    training_years = "2017-2019" if args.regional_spec else "2015-2019"
    validation_key = "historical_validation" if args.regional_spec else "2020"
    splits = {"train": reference.Split(data.x, data.y, data.prior, training_years, "train"),
              validation_key: reference.Split(data.val_x, data.val_y, data.val_prior, validation_key, "validation")}
    identities = {}
    limit = 40 if args.smoke else None
    if args.regional_spec:
        import east_asia_prior99 as regional
        extra_splits, identities = regional.evaluation_splits(args, data)
        splits.update(extra_splits)
    for year in (() if args.regional_spec else range(2021, 2026)):
        x, y, prior = features.load_year(args.source, year, args.domain, limit, args.include_xco2_prior)
        standardize(x, data.median, data.mean, data.scale)
        splits[str(year)] = reference.Split(x, y, prior, year, "test")
        with h5py.File(args.source / f"projected_{year}.h5") as source:
            positions = np.flatnonzero(source["domain"][:] == args.domain.encode())
            if limit is not None:
                positions = positions[:limit]
            identities[str(year)] = (source["sounding_id"][:][positions], source["rows"][:][positions])
    rows = []
    for run_info in selected["runs"]:
        run = Path(run_info["run"])
        destination = folder / "selected_models" / f"seed_{run_info['seed']}"
        destination.mkdir(parents=True, exist_ok=True)
        finished = destination / "complete.json"
        if finished.exists():
            rows.extend(json.loads(finished.read_text())["metrics"])
            continue
        checkpoint = torch.load(run / "best.pt", map_location=device, weights_only=False)
        assert checkpoint["hp"] == selected["hp"]
        model = DenseMLP(data.x.shape[1], selected["hp"]).to(device)
        model.load_state_dict(checkpoint["state_dict"])
        atomic_torch(destination / "model.pt", checkpoint)
        def predict(x):
            return reference.torch_predict(model, x, device, 16384) * checkpoint["target_std"] + checkpoint["target_mean"]
        run_rows = features.evaluate(
            f"mlp_dense_{hp_id(selected['hp'])}_{run_info['seed']}", args.route, "MLP", run_info["seed"],
            {key: split for key, split in splits.items() if split.role != "test"}, predict,
            run_info["train_seconds"], run_info["parameters"])
        run_rows += features.evaluate(
            f"mlp_dense_{hp_id(selected['hp'])}_{run_info['seed']}", args.route, "MLP", run_info["seed"],
            {key: split for key, split in splits.items() if split.role == "test"}, predict,
            run_info["train_seconds"], run_info["parameters"], destination, identities)
        write_json(finished, {"status": "complete", "metrics": run_rows})
        rows.extend(run_rows)
        del model
    write_csv(folder / "metrics_runs.csv", rows)
    summary = reference.aggregate_metrics(rows, folder / "metrics_summary.csv")
    write_json(folder / "complete.json", {
        "status": "complete", "smoke": args.smoke, "domain": args.domain, "route": args.route,
        "seed_models": len(selected["runs"]), "summary_rows": len(summary), "training_rows": len(data.y),
        "feature_count": data.x.shape[1], "test_years": list(range(2020 if args.regional_spec else 2021, 2026)),
        "selected": selected,
    })


def feature_names(args):
    return features.FEATURE_NAMES + (["xco2_apriori"] if args.include_xco2_prior else [])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--architecture-only-config", type=Path)
    parser.add_argument("--include-xco2-prior", action="store_true")
    parser.add_argument("--regional-spec", type=Path)
    parser.add_argument("--domain", choices=features.DOMAINS, required=True)
    parser.add_argument("--route", choices=reference.ROUTES, required=True)
    parser.add_argument("--phase", choices=("search", "evaluate"), default="search")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--trials", type=int, default=128)
    parser.add_argument("--max-epochs", type=int, default=200)
    parser.add_argument("--min-epochs", type=int, default=40)
    parser.add_argument("--patience", type=int, default=30)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    assert args.output.resolve() not in (args.source.resolve(), args.baseline_root.resolve() if args.baseline_root else None)
    if args.smoke:
        args.trials, args.max_epochs, args.min_epochs, args.patience, args.top_k = 3, 2, 1, 2, 1
    else:
        assert args.trials == 128 and args.max_epochs == 200 and args.min_epochs == 40 and args.top_k == 3
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision("highest")
    device = torch.device(args.device)
    if device.type == "cuda":
        assert torch.cuda.is_available(), "Do not silently run formal searches on CPU"
    else:
        assert args.smoke, "Formal searches require an assigned GPU"
    folder = args.output / "studies" / args.domain / args.route
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / "worker.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    config = {"source": str(args.source.resolve()), "baseline_root": str(args.baseline_root) if args.baseline_root else None,
              "domain": args.domain, "route": args.route, "smoke": args.smoke,
              "trials": args.trials, "max_epochs": args.max_epochs, "min_epochs": args.min_epochs,
              "patience": args.patience, "top_k": args.top_k, "feature_names": feature_names(args),
              "search_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              "feature_script_sha256": hashlib.sha256(Path(features.__file__).read_bytes()).hexdigest(),
              "reference_script_sha256": hashlib.sha256(Path(reference.__file__).read_bytes()).hexdigest()}
    if args.architecture_only_config:
        config["architecture_only_config"] = json.loads(args.architecture_only_config.read_text())
    if args.regional_spec:
        import east_asia_prior99 as regional
        assert args.domain == "ND" and args.include_xco2_prior
        config["regional_spec"] = json.loads(args.regional_spec.read_text())
        config["regional_script_sha256"] = hashlib.sha256(Path(regional.__file__).read_bytes()).hexdigest()
    path = folder / "config.json"
    if path.exists():
        assert json.loads(path.read_text()) == config, "Never reuse a study with a different configuration"
    write_json(path, config)
    write_json(folder / "worker.json", {"pid": os.getpid(), "phase": args.phase,
                                        "physical_gpu": os.environ.get("CUDA_VISIBLE_DEVICES")})
    print(f"Start {args.domain}/{args.route} phase={args.phase} device={device}", flush=True)
    data = prepare(args, folder)
    if args.phase == "search":
        search(args, folder, data, device)
    else:
        evaluate_selected(args, folder, data, device)


if __name__ == "__main__":
    main()
