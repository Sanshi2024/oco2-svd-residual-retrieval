#!/usr/bin/env python3
"""Run six independent validation-only searches using only currently free GPUs."""

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import queue
import subprocess
import sys
import time


SOURCE = Path("/workspace_hdd/xianlei/OCO2/output/ML/Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test")
BASELINE = Path("/workspace_hdd/xianlei/OCO2/output/ML/Paper_ThreeDomain_Edge30_SVD30_Models_SecantGeometry_1519Train_2020Val_2021_2025Test")
OUTPUT = Path("/workspace_hdd/xianlei/OCO2/output/ML/Paper_ThreeDomain_Edge30_SVD30_MLPDenseSearch_20261018")
SCRIPTS = Path(__file__).resolve().parent
JOBS = [(domain, route) for domain in ("ND", "GL_OCEAN", "GL_LAND") for route in ("absolute", "delta")]
ARCHITECTURE_CONFIG = None
INCLUDE_XCO2_PRIOR = False
ELIGIBLE_GPUS = (0, 1, 2, 7)


def write_json(path, value):
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(value, indent=2))
    os.replace(partial, path)


def gpu_candidates():
    response = subprocess.check_output(["nvidia-smi", "--query-gpu=index,memory.used,utilization.gpu",
                                        "--format=csv,noheader,nounits"], text=True)
    records = [dict(zip(("gpu", "memory_mib", "utilization"), map(int, line.split(","))))
               for line in response.splitlines()]
    free = [r["gpu"] for r in records if r["gpu"] in ELIGIBLE_GPUS
            and r["memory_mib"] < 1024 and r["utilization"] < 15]
    return free, records


def preflight():
    assert (SOURCE / "spectral_pipeline.complete").exists()
    assert (BASELINE / "pipeline.complete").exists()
    assert json.loads((SOURCE / "spectral.complete.json").read_text())["components_per_band"] == 30
    assert json.loads((BASELINE / "experiment_protocol.json").read_text())["source_root"] == str(SOURCE)
    fingerprints = []
    for year in range(2015, 2026):
        path = SOURCE / f"projected_{year}.h5"
        stat = path.stat()
        fingerprints.append({"year": year, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
    config = {
        "source": str(SOURCE), "baseline": str(BASELINE), "jobs": JOBS,
        "training_years": list(range(2015, 2020)), "validation_year": 2020,
        "future_evaluation_years": list(range(2021, 2026)),
        "training_rows_per_domain": 1000000, "validation_rows_per_domain": 200000,
        "features": "90 full masked-projection PCs + 8 ancillary predictors; secant zeniths",
        "scalers": "Reuse unchanged original training-only scalers",
        "trials_per_job": 128, "trial_seed": 17,
        "confirmation": "Top 3 complete configurations, seeds 17/29/43; select mean 2020 RMSE",
        "max_epochs": 200, "min_epochs": 40, "patience": 30,
        "pruning": "Optuna median; 12 startup trials; 40-epoch grace; 10-epoch interval; min 6 trials",
        "architectures": ["64-64", "128-64", "256-128", "256-256-128", "512-256-128",
                          "512-512-256", "256-256-256-128"],
        "learning_rate": [3e-5, 3e-3], "learning_rate_sampling": "log-continuous TPE",
        "dropout": {"min": 0, "max": 0.3, "step": 0.05},
        "weight_decay": [0, 1e-6, 1e-5, 1e-4, 1e-3],
        "batch_size": [1024, 2048, 4096, 8192, 16384],
        "activations": ["gelu", "relu", "silu"], "normalization": ["none", "layernorm"],
        "schedulers": ["constant", "plateau", "cosine"], "warmup_epochs": [0, 5],
        "optimizer": "AdamW; gradient clip norm 5; float32 without mixed precision",
        "test_visibility": "2021-2025 were inspected in the previous experiment; not a new untouched confirmation set",
        "selection_guard": "No test targets used during search; freeze all six studies before any new test evaluation",
        "projected_file_fingerprints": fingerprints,
        "script_hashes": {name: hashlib.sha256((SCRIPTS / name).read_bytes()).hexdigest() for name in (
            "search_mlp_secant.py", "train_domain_svd50.py", "two_route_training_reference.py")},
    }
    if ARCHITECTURE_CONFIG is not None:
        for name in ("architectures", "learning_rate", "learning_rate_sampling", "dropout", "weight_decay",
                     "batch_size", "activations", "normalization", "schedulers", "warmup_epochs"):
            del config[name]
        config.update({"search_scope": "hidden layers only; 128 unique completed or pruned architectures",
                       "depth": [1, 4], "hidden_width": {"min": 32, "max": 640, "step": 4},
                       "sampler": "Conditional multivariate grouped TPE; 24 startup trials",
                       "fixed_nonarchitecture_recipes": json.loads(ARCHITECTURE_CONFIG.read_text())})
    if INCLUDE_XCO2_PRIOR:
        config["features"] = "90 full masked-projection PCs + 8 ancillary predictors + XCO2 apriori; secant zeniths"
        config["feature_count"] = 99
        config["xco2_prior_as_predictor"] = True
        config["scalers"] = "Original 98 training-only scales plus training-only mean/std for XCO2 apriori"
    # JSON normalization keeps tuple/list representations identical on resume.
    config = json.loads(json.dumps(config))
    path = OUTPUT / "experiment_protocol.json"
    if path.exists():
        assert json.loads(path.read_text()) == config, "Inputs or search specification changed"
    else:
        write_json(path, config)
    diagnostic = []
    for domain, route in JOBS:
        for seed in (17, 29, 43):
            path = BASELINE / "models" / domain / route / f"{route}_mlp_{seed}" / "training_history.json"
            old = json.loads(path.read_text())
            diagnostic.append({"domain": domain, "route": route, "seed": seed,
                               "best_epoch": old["best_epoch"], "epochs": len(old["history"]),
                               "best_validation_rmse": old["best_val_rmse"]})
    write_json(OUTPUT / "original_mlp_diagnostics.json", diagnostic)


def run_pool(phase, gpus):
    tasks = queue.Queue()
    for domain, route in JOBS:
        marker = OUTPUT / "studies" / domain / route / ("search.complete.json" if phase == "search" else "complete.json")
        if not marker.exists():
            tasks.put((domain, route))
    errors = []

    def worker(gpu):
        while True:
            try:
                domain, route = tasks.get_nowait()
            except queue.Empty:
                return
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu), "OMP_NUM_THREADS": "4",
                   "OPENBLAS_NUM_THREADS": "4", "MKL_NUM_THREADS": "4"}
            log_path = OUTPUT / "logs" / f"{domain}_{route}_{phase}.log"
            with log_path.open("a", buffering=1) as log:
                log.write(f"\nStart {datetime.now(timezone.utc).isoformat()} physical GPU {gpu}\n")
                command = [sys.executable, "-u", str(SCRIPTS / "search_mlp_secant.py"),
                           "--source", str(SOURCE), "--baseline-root", str(BASELINE),
                           "--output", str(OUTPUT), "--domain", domain, "--route", route,
                           "--phase", phase, "--device", "cuda:0"]
                if ARCHITECTURE_CONFIG is not None:
                    command += ["--architecture-only-config", str(ARCHITECTURE_CONFIG)]
                if INCLUDE_XCO2_PRIOR:
                    command += ["--include-xco2-prior"]
                process = subprocess.Popen(command, env=env, stdin=subprocess.DEVNULL,
                                           stdout=log, stderr=subprocess.STDOUT)
                print(f"{phase}: GPU {gpu}, {domain}/{route}, PID {process.pid}", flush=True)
                code = process.wait()
            if code:
                errors.append({"domain": domain, "route": route, "phase": phase,
                               "returncode": code, "log": str(log_path)})
            tasks.task_done()
    with ThreadPoolExecutor(max_workers=len(gpus)) as executor:
        list(executor.map(worker, gpus))
    if errors:
        write_json(OUTPUT / "pipeline.failed.json", errors)
        raise RuntimeError(f"{len(errors)} workers failed; preserve and resume their studies")


def freeze_searches():
    selected = []
    for domain, route in JOBS:
        folder = OUTPUT / "studies" / domain / route
        done = json.loads((folder / "search.complete.json").read_text())
        assert done["status"] == "complete" and done["smoke"] is False
        assert done["trials"] == 128 and done["confirmed_configurations"] == 3
        if ARCHITECTURE_CONFIG is not None:
            assert done["unique_finished_candidates"] == 128
        assert done["training_rows"] == 1000000 and done["validation_rows"] == 200000
        frozen = json.loads((folder / "selection_frozen.json").read_text())
        assert not frozen["test_metrics_used_for_selection"]
        assert len(frozen["selected"]["runs"]) == 3
        if INCLUDE_XCO2_PRIOR:
            assert frozen["feature_count"] == 99
        selected.append(frozen)
    write_json(OUTPUT / "search_selection.complete.json", {
        "status": "complete", "studies": 6, "selection_year": 2020,
        "utc": datetime.now(timezone.utc).isoformat(), "selected": selected,
    })


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "logs").mkdir(exist_ok=True)
    lock = (OUTPUT / "pipeline.lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        print("Dense MLP pipeline already running", flush=True)
        return
    if (OUTPUT / "pipeline.complete").exists():
        print("Dense MLP pipeline already complete", flush=True)
        return
    (OUTPUT / "pipeline.pid").write_text(str(os.getpid()))
    preflight()
    gpus, records = gpu_candidates()
    assert gpus, "No free eligible GPUs; do not touch other jobs"
    write_json(OUTPUT / "gpu_resources.json", {"assigned_gpus": gpus, "observed": records})
    env = {**os.environ, "OMP_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "MKL_NUM_THREADS": "2",
           "CUDA_VISIBLE_DEVICES": str(gpus[0])}
    smoke = OUTPUT / "logs" / "fixture_smoke.log"
    with smoke.open("w") as log:
        command = [sys.executable, "-u", str(SCRIPTS / "test_mlp_dense_search.py")]
        if ARCHITECTURE_CONFIG is not None:
            command += ["--hidden-only"]
        if INCLUDE_XCO2_PRIOR:
            command += ["--include-xco2-prior"]
        subprocess.run(command,
                       env=env, stdout=log, stderr=subprocess.STDOUT, check=True)
    run_pool("search", gpus)
    freeze_searches()
    evaluation_gpus, records = gpu_candidates()
    while not evaluation_gpus:
        print("Searches frozen; waiting for an idle eligible GPU for evaluation", flush=True)
        time.sleep(60)
        evaluation_gpus, records = gpu_candidates()
    write_json(OUTPUT / "evaluation_gpu_resources.json", {"assigned_gpus": evaluation_gpus, "observed": records})
    run_pool("evaluate", evaluation_gpus)
    subprocess.run([sys.executable, "-u", str(SCRIPTS / "summarize_mlp_dense_search.py"),
                    "--root", str(OUTPUT), "--baseline-root", str(BASELINE)], check=True)
    (OUTPUT / "pipeline.complete").touch()
    print("Dense MLP search and selected-model evaluation complete", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        OUTPUT.mkdir(parents=True, exist_ok=True)
        write_json(OUTPUT / "pipeline.error.json", {"error": repr(exc),
                                                    "utc": datetime.now(timezone.utc).isoformat()})
        raise
