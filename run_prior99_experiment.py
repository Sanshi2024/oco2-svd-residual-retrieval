#!/usr/bin/env python3
"""Separate global-three-domain and East-Asia-ND experiments with a prior input."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import run_mlp_dense_search as runner


SCRIPTS = Path(__file__).resolve().parent
PREVIOUS = Path("/workspace_hdd/xianlei/OCO2/output/ML/Paper_ThreeDomain_Edge30_SVD30_MLPDenseSearch_20261018")
GLOBAL = Path("/workspace_hdd/xianlei/OCO2/output/ML/Paper_ThreeDomain_Global_Prior99_1519Train_2020Val_2021_2025Test_20261018")
REGIONAL = Path("/workspace_hdd/xianlei/OCO2/output/ML/Paper_ND_EastAsia_SpTRegion_Prior99_1719RandomSplit_2020_2025Test_20261018")


def save(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        assert json.loads(path.read_text()) == payload, f"Changed immutable protocol: {path}"
    else:
        runner.write_json(path, payload)


def prepare_protocol(root, regional):
    mlp = root / "mlp_search"
    mlp.mkdir(parents=True, exist_ok=True)
    (root / "logs").mkdir(exist_ok=True)
    spec = {"latitude": [20, 45], "longitude": [110, 145], "bounds": "strict",
            "mode": "ND", "pool_years": [2017, 2018, 2019], "fractions": [.8, .1, .1], "split_seed": 1237,
            "source": str(runner.SOURCE), "svd_refitted": False,
            "paper_doi": "10.34133/remotesensing.0470", "paper_training_pool": 494887,
            "data_availability": "Authors explicitly do not distribute their large training dataset",
            "comparison_status": "Regional sampled-cohort comparison, NOT exact author-dataset reproduction",
            "differences": ["Our screened sampled ND pool is smaller than the paper's pool",
                "Same geographic bounds but source quality/land filters are not identical",
                "Frozen 2015-2019 ND SVD; no spectral refitting; random historical splits are not representation-independent",
                "Our 90 PC + 9 ancillary/prior inputs differ from SpT full radiance/SNR/polarization/detector inputs",
                "Future years have already been inspected in earlier experiments; this is exploratory"]}
    if regional:
        save(mlp / "regional_spec.json", spec)
        hp = {"architecture":"128-64", "activation":"gelu", "normalization":"none", "dropout":.1,
              "lr":.001, "weight_decay":.0001, "batch_size":1024, "scheduler":"cosine", "warmup_epochs":5}
        recipe = {"selection_year":"2017-2019 random validation", "source_search":"predeclared training recipe; no 2020 selection",
                  "jobs":[{"domain":"ND", "route":route, "hp":hp} for route in ("absolute","delta")]}
    else:
        assert (PREVIOUS / "pipeline.complete").exists()
        frozen = json.loads((PREVIOUS / "search_selection.complete.json").read_text())
        assert frozen["selection_year"] == 2020 and frozen["studies"] == 6
        jobs = []
        for record in frozen["selected"]:
            assert not record["test_metrics_used_for_selection"]
            jobs.append({"domain":record["domain"], "route":record["route"], "hp":record["selected"]["hp"]})
        assert {(r["domain"],r["route"]) for r in jobs} == set(runner.JOBS)
        recipe = {"selection_year":2020, "source_search":str(PREVIOUS), "jobs":jobs}
    save(mlp / "fixed_nonarchitecture_recipes.json", recipe)
    protocol = {"experiment":"east_asia_nd" if regional else "global_three_domains",
        "source":str(runner.SOURCE), "feature_count":99, "components_per_band":30,
        "features":"90 PCs + sec(SZA), sec(VZA), sin(AZA), cos(AZA), Sun distance/velocity, two pressure priors, XCO2 apriori",
        "xco2_prior_as_predictor":True, "routes":["absolute","delta"], "families":["Ridge","LightGBM","MLP"],
        "svd_refitted":False, "projection":"Reuse immutable existing domain-specific full-30-PC masked projections",
        "training_years":[2017,2018,2019] if regional else list(range(2015,2020)),
        "validation":"2017-2019 random 10%" if regional else 2020,
        "future_test_years":list(range(2020 if regional else 2021,2026)),
        "training_counts":"80% of 27,702 regional pooled rows" if regional else "1,000,000 per domain",
        "scale_fit":"Supervised training rows only; never use validation/future rows for XCO2 prior scaling",
        "hidden_search":{"depth":[1,4], "width":[32,640], "step":4, "unique_candidates":128,
                         "method":"TPE; nonarchitecture parameters fixed", "top_k":3, "confirmation_seeds":[17,29,43]},
        "regional_spec":spec if regional else None, "nonarchitecture_recipes":recipe,
        "input_files":[{"year":y,"size":(runner.SOURCE/f"projected_{y}.h5").stat().st_size,
                        "mtime_ns":(runner.SOURCE/f"projected_{y}.h5").stat().st_mtime_ns} for y in range(2015,2026)],
        "scripts":{p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in SCRIPTS.glob("*.py")},
        "test_visibility":"Previously inspected historical future years, not an independent untouched confirmation"}
    save(root / "experiment_protocol.json", protocol)
    return mlp


def tree_batch(root, mlp, regional):
    lock = (root / "tree_batch.lock").open("w")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    domains = ("ND",) if regional else ("ND","GL_OCEAN","GL_LAND")
    jobs = [(d,r) for d in domains for r in ("absolute","delta")]
    def work(job):
        domain, route = job
        command = [sys.executable,"-u",str(SCRIPTS/"train_prior99_trees.py"),
            "--source",str(runner.SOURCE),"--output",str(mlp),"--tree-root",str(root/"tree_models"),
            "--baseline-root",str(runner.BASELINE),"--domain",domain,"--route",route]
        if regional:
            command += ["--regional-spec",str(mlp/"regional_spec.json")]
        env = {**os.environ,"CUDA_VISIBLE_DEVICES":"","OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
        with (root/"logs"/f"trees_{domain}_{route}.log").open("a") as log:
            subprocess.run(command, env=env, stdin=subprocess.DEVNULL, stdout=log, stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=2 if regional else 3) as pool:
        list(pool.map(work,jobs))
    runner.write_json(root/"trees.complete.json",{"status":"complete","cohort_routes":len(jobs)})


def regional_neural(root, mlp):
    while True:
        free,_ = runner.gpu_candidates()
        if 7 in free:
            break
        print("Waiting for idle GPU 7 for regional ND experiment",flush=True)
        time.sleep(60)
    env = {**os.environ,"CUDA_VISIBLE_DEVICES":"7","OMP_NUM_THREADS":"4","OPENBLAS_NUM_THREADS":"4","MKL_NUM_THREADS":"4"}
    for phase in ("search","evaluate"):
        for route in ("absolute","delta"):
            command = [sys.executable,"-u",str(SCRIPTS/"search_mlp_secant.py"),
                "--source",str(runner.SOURCE),"--baseline-root",str(runner.BASELINE),"--output",str(mlp),
                "--domain","ND","--route",route,"--phase",phase,"--include-xco2-prior",
                "--architecture-only-config",str(mlp/"fixed_nonarchitecture_recipes.json"),
                "--regional-spec",str(mlp/"regional_spec.json")]
            with (root/"logs"/f"ND_{route}_{phase}.log").open("a") as log:
                process = subprocess.Popen(command,env=env,stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
                print(f"Regional {route}/{phase}: GPU 7, PID {process.pid}",flush=True)
                assert process.wait() == 0, f"Regional {route}/{phase} failed; inspect log"
        if phase == "search":
            selected = []
            for route in ("absolute","delta"):
                folder = mlp/"studies/ND"/route
                done = json.loads((folder/"search.complete.json").read_text())
                frozen = json.loads((folder/"selection_frozen.json").read_text())
                assert not done["smoke"] and done["unique_finished_candidates"] == 128
                assert frozen["feature_count"] == 99 and not frozen["test_metrics_used_for_selection"]
                assert len(done["selected"]["runs"]) == 3
                selected.append(frozen)
            runner.write_json(mlp/"search_selection.complete.json",{"status":"complete","studies":2,
                "selection_year":"2017-2019 random validation", "selected":selected})


def summarize(root, mlp, regional):
    import numpy as np
    import search_mlp_secant as engine
    records, runs = [], []
    domains = ("ND",) if regional else ("ND","GL_OCEAN","GL_LAND")
    expected_splits = 9 if regional else 7
    for domain in domains:
        for route in ("absolute","delta"):
            for folder in (mlp/"studies"/domain/route,root/"tree_models"/domain/route):
                done = json.loads((folder/"complete.json").read_text())
                assert not done["smoke"] and done["feature_count"] == 99
                assert json.loads((folder/"feature_names.json").read_text())[-1] == "xco2_apriori"
                with (folder/"metrics_summary.csv").open() as handle:
                    rows = list(csv.DictReader(handle))
                assert len(rows) in (expected_splits,2*expected_splits)
                for row in rows:
                    assert np.isfinite([float(row[k]) for k in ("rmse_mean","bias_mean","r2_mean")]).all()
                    assert int(row["runs"]) == (1 if row["family"] == "Ridge" else 3)
                    records.append({"domain":domain,**row})
                with (folder/"metrics_runs.csv").open() as handle:
                    runs.extend({"domain":domain,**row} for row in csv.DictReader(handle))
    expected_models = 14 if regional else 42
    assert len(records) == (54 if regional else 126)
    assert len(runs) == expected_models * expected_splits
    selected = []
    for domain in domains:
        for route in ("absolute","delta"):
            val = [r for r in records if r["domain"]==domain and r["route"]==route and r["role"]=="validation"]
            best = min(val,key=lambda r:float(r["rmse_mean"]))
            selected += [r for r in records if r["domain"]==domain and r["route"]==route and r["family"]==best["family"]]
    summary = root/"summary"
    summary.mkdir(exist_ok=True)
    engine.write_csv(summary/"metrics_summary_all_domains.csv",records)
    engine.write_csv(summary/"metrics_runs_all_domains.csv",runs)
    engine.write_csv(summary/"validation_selected_metrics.csv",selected)
    runner.write_json(summary/"complete.json",{"status":"complete","feature_count":99,"models":expected_models,
        "summary_rows":len(records),"metric_run_rows":len(runs),"regional":regional,"svd_refitted":False})
    (root/"pipeline.complete").touch()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment",choices=("global","east_asia"),required=True)
    parser.add_argument("--trees-only",action="store_true")
    args = parser.parse_args()
    regional = args.experiment == "east_asia"
    root = REGIONAL if regional else GLOBAL
    root.mkdir(parents=True,exist_ok=True)
    if args.trees_only:
        tree_batch(root,root/"mlp_search",regional)
        return
    lock = (root/"pipeline.lock").open("w")
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (root/"pipeline.complete").exists():
        return
    (root/"pipeline.pid").write_text(str(os.getpid()))
    mlp = prepare_protocol(root,regional)
    try:
        with (root/"logs/tree_batch.log").open("a") as log:
            trees = subprocess.Popen([sys.executable,"-u",str(Path(__file__).resolve()),"--experiment",args.experiment,"--trees-only"],
                stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT)
        if regional:
            regional_neural(root,mlp)
        else:
            runner.OUTPUT,runner.ARCHITECTURE_CONFIG = mlp,mlp/"fixed_nonarchitecture_recipes.json"
            runner.INCLUDE_XCO2_PRIOR,runner.ELIGIBLE_GPUS = True,(0,1,2)
            runner.main()
        assert trees.wait() == 0,"Tree-control batch failed; inspect tree logs"
        summarize(root,mlp,regional)
    except Exception as exc:
        error = {"error":repr(exc),"utc":datetime.now(timezone.utc).isoformat()}
        runner.write_json(root/"pipeline.error.json",error)
        runner.write_json(mlp/"pipeline.error.json",error)
        raise


if __name__ == "__main__":
    main()
