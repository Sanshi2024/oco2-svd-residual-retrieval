#!/usr/bin/env python3
"""Regional ND cohorts from existing PC projections; never refit spectral bases."""

import fcntl
import json
import os
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np

import train_domain_svd50 as features
import two_route_training_reference as reference


def load_cohort(args, year):
    spec = json.loads(args.regional_spec.read_text())
    cache = args.output / "regional_cohorts"
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / f"ND_EastAsia_{year}.npz"
    with (cache / f"{year}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if not target.exists():
            with h5py.File(args.source / f"projected_{year}.h5", "r") as source:
                assert source.attrs["status"] == "complete"
                positions = np.flatnonzero(source["domain"][:] == b"ND")
                if args.smoke:
                    positions = positions[:40]
                lat, lon = source["lat"][:][positions], source["lon"][:][positions]
                keep = ((lat > spec["latitude"][0]) & (lat < spec["latitude"][1])
                        & (lon > spec["longitude"][0]) & (lon < spec["longitude"][1]))
                selected = positions[keep]
                ids, rows = source["sounding_id"][:][selected], source["rows"][:][selected]
                assert np.all(source["mode"][:][selected] == b"ND")
            x, y, prior = features.load_year(args.source, year, "ND", 40 if args.smoke else None, True)
            x, y, prior = x[keep], y[keep], prior[keep]
            assert x.shape == (len(ids), 99) and len(ids) > 0 and len(np.unique(ids)) == len(ids)
            assert np.array_equal(x[:, -1], prior)
            partial = target.with_suffix(".partial.npz")
            np.savez(partial, x=x, y=y, prior=prior, sounding_id=ids, original_index=rows,
                     lat=lat[keep], lon=lon[keep], year=np.full(len(y), year, dtype="i2"))
            os.replace(partial, target)
            print(f"Cached East Asia ND {year}: {len(y):,} unique rows, 99 features", flush=True)
        with np.load(target) as saved:
            return {name: saved[name] for name in saved.files}


def historical_pool(args):
    cohorts = [load_cohort(args, year) for year in (2017, 2018, 2019)]
    pool = {name: np.concatenate([row[name] for row in cohorts]) for name in cohorts[0]}
    n = len(pool["y"])
    assert len(np.unique(pool["sounding_id"])) == n
    rng = np.random.default_rng(1237)
    order = rng.permutation(n)
    split = {"train": order[:int(n * .8)], "validation": order[int(n * .8):int(n * .9)],
             "historical_holdout": order[int(n * .9):]}
    assert np.array_equal(np.sort(np.concatenate(list(split.values()))), np.arange(n))
    path = args.output / "historical_split.npz"
    with (args.output / "historical_split.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        if path.exists():
            with np.load(path) as previous:
                assert np.array_equal(previous["sounding_id"], pool["sounding_id"])
                for key, value in split.items():
                    assert np.array_equal(previous[key], value)
        else:
            np.savez(path, sounding_id=pool["sounding_id"], original_index=pool["original_index"], **split)
    return pool, split


def prepare(args, folder, engine):
    pool, split = historical_pool(args)
    train, validation = split["train"], split["validation"]
    x, val_x = pool["x"][train].copy(), pool["x"][validation].copy()
    median = np.nanmedian(x, axis=0)
    median[~np.isfinite(median)] = 0
    filled = np.where(np.isfinite(x), x, median)
    mean = filled.mean(axis=0, dtype="f8").astype("f4")
    scale = filled.std(axis=0, dtype="f8").astype("f4")
    scale[scale == 0] = 1
    engine.standardize(x, median, mean, scale)
    engine.standardize(val_x, median, mean, scale)
    np.savez_compressed(folder / "training_only_scaler.npz", median=median, mean=mean, scale=scale)
    engine.write_json(folder / "feature_names.json", engine.feature_names(args))
    engine.write_json(folder / "regional_counts.json", {
        "annual_pool": {str(year): int(np.count_nonzero(pool["year"] == year)) for year in (2017, 2018, 2019)},
        "historical_pool": len(pool["y"]), "training_rows": len(train), "validation_rows": len(validation),
        "historical_holdout_rows": len(split["historical_holdout"]), "split_seed": 1237,
        "svd_refitted": False, "frozen_svd_fit_years": list(range(2015, 2020)),
        "historical_holdout_representation": "Frozen unsupervised SVD saw 2015-2019; not a fully untouched representation holdout",
        "screening": "Inherit existing screened annual samples; bounding-box ND, not author-identical land/quality cohort",
    })
    return engine.Prepared(x, pool["y"][train], pool["prior"][train], val_x,
                           pool["y"][validation], pool["prior"][validation], median, mean, scale)


def evaluation_splits(args, data):
    # This is called only after both regional searches are frozen by the global evaluation gate.
    assert (args.output / "search_selection.complete.json").exists()
    pool, split = historical_pool(args)
    holdout = split["historical_holdout"]
    cohorts = {"historical_holdout": {key: value[holdout] for key, value in pool.items()}}
    cohorts.update({str(year): load_cohort(args, year) for year in range(2020, 2026)})
    splits, identities = {}, {}
    for label, cohort in cohorts.items():
        x = cohort["x"].copy()
        bad = ~np.isfinite(x)
        x[bad] = np.broadcast_to(data.median, x.shape)[bad]
        x = (x - data.mean) / data.scale
        assert np.isfinite(x).all()
        splits[label] = reference.Split(x, cohort["y"], cohort["prior"], label, "test")
        identities[label] = (cohort["sounding_id"], cohort["original_index"])
    return splits, identities
