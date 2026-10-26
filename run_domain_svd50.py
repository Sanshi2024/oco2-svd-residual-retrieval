#!/usr/bin/env python3
"""Frozen common wavelength protocol with independent domain/band SVD models."""

from __future__ import annotations

import argparse
from collections import OrderedDict
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import h5py
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD

import run_three_domain_pipeline as base


DOMAINS = tuple(domain.decode() for domain in base.DOMAINS)
TRAIN_YEARS = tuple(range(2015, 2020))


def json_write(path: Path, payload: dict) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    partial.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(partial, path)


def load_sample(path: Path) -> dict:
    with np.load(path) as stored:
        return {key: stored[key] for key in stored.files}


def freeze_protocol(annual_root: Path, sample_root: Path, output: Path) -> tuple[dict, str]:
    protocol_path = output / "protocol.npz"
    ready = False
    if protocol_path.exists():
        with np.load(protocol_path) as stored:
            ready = all(f"{band}_continuum_indices" in stored.files for band in base.BANDS)
    if not ready:
        print("Freezing 30% edge rule and common wavelength grid from 2015-2019 granule metadata", flush=True)
        paths = {year: annual_root / f"OCO2_L1B_L2_{year}.h5" for year in TRAIN_YEARS}
        protocol = base.derive_protocol(paths, output, 0.30)
        previews = []
        for year in TRAIN_YEARS:
            sample = load_sample(sample_root / f"sample_{year}.npz")
            chosen = np.sort(np.concatenate([
                np.flatnonzero(sample["domain"] == domain)[:100] for domain in base.DOMAINS
            ]))
            spectra = {band: [] for band in base.BANDS}
            with h5py.File(paths[year], "r", locking=False) as source:
                metadata = source["spectral_metadata/granules"]
                for granule, fp, local in base.group_positions(
                    sample["granule_index"][chosen], sample["footprint_index"][chosen]
                ):
                    positions = chosen[local]
                    grid = base.wavelength(metadata["dispersion_coef_samp"][granule])
                    bad = metadata["bad_sample_list"][granule] != 0
                    for bi, band in enumerate(base.BANDS):
                        radiance = base.indexed(source[base.RADIANCE[band]], sample["rows"][positions])
                        left, right, weight, inside = base.interpolation_plan(grid[bi, fp], protocol[band]["target"])
                        values = radiance[:, left] * (1 - weight) + radiance[:, right] * weight
                        good = inside & ~bad[bi, fp, left] & ~bad[bi, fp, right]
                        values[:, ~good] = np.nan
                        scale = np.nanquantile(values, 0.95, axis=1)
                        scale[~np.isfinite(scale) | (scale <= 0)] = np.nan
                        spectra[band].append(values / scale[:, None])
            previews.append({band: np.concatenate(values) for band, values in spectra.items()})
        for band in base.BANDS:
            values = np.concatenate([preview[band] for preview in previews])
            count = 15
            valid_fraction = np.mean(np.isfinite(values) & (values > 0), axis=0)
            median = np.nanmedian(values, axis=0)
            candidates = [
                start for start in range(10, len(median) - count - 9)
                if np.all(valid_fraction[start:start + count] >= 0.90)
                and np.all(np.isfinite(median[start:start + count]))
            ]
            if not candidates:
                raise RuntimeError(f"No stable training-only continuum window for {band}")
            start = max(candidates, key=lambda value: float(np.mean(median[value:value + count])))
            protocol[band]["continuum"] = np.arange(start, start + count)
        with np.load(protocol_path) as saved:
            arrays = {key: saved[key] for key in saved.files}
        for band in base.BANDS:
            arrays[f"{band}_continuum_indices"] = protocol[band]["continuum"]
        np.savez_compressed(protocol_path, **arrays)
        summary = json.loads((output / "protocol.json").read_text())
        summary["continuum_selection"] = {
            "years": list(TRAIN_YEARS), "preview_rows_per_year_per_domain": 100,
            "window_points": 15, "minimum_preview_good_fraction": 0.90,
            "rule": "brightest median window in per-spectrum scaled training previews; exclude bad interpolation support",
        }
        for band in base.BANDS:
            summary["bands"][band]["continuum_wavelength_um"] = protocol[band]["target"][protocol[band]["continuum"]].tolist()
        json_write(output / "protocol.json", summary)
    with np.load(protocol_path) as stored:
        protocol = {band: {
            "target": stored[f"{band}_target_wavelength_um"],
            "continuum": stored[f"{band}_continuum_indices"],
        } for band in base.BANDS}
    summary = json.loads((output / "protocol.json").read_text())
    assert summary["edge_bad_frequency_threshold"] == 0.30
    digest = hashlib.sha256(protocol_path.read_bytes()).hexdigest()
    return protocol, digest


def resample(annual: Path, sample: dict, protocol: dict, output: Path, digest: str) -> None:
    if output.exists():
        with h5py.File(output) as stored:
            assert stored.attrs["protocol_sha256"] == digest
        return
    partial = output.with_suffix(".h5.partial")
    n = len(sample["rows"])
    order = np.argsort(sample["rows"])
    sorted_rows = sample["rows"][order]
    groups = list(base.group_positions(sample["granule_index"], sample["footprint_index"]))
    validity = np.ones(n, dtype=bool)
    with h5py.File(annual, "r", locking=False) as source, h5py.File(partial, "w") as target:
        for key in ("sounding_id", "rows", "granule_index", "footprint_index", "mode", "domain", *base.MODEL_SCALARS):
            target.create_dataset(key, data=sample[key], compression="lzf")
        metadata = source["spectral_metadata/granules"]
        dispersion = metadata["dispersion_coef_samp"][:]
        bad_all = metadata["bad_sample_list"][:] != 0
        metadata_good = (metadata["l1b_metadata_valid"][:] == 1) & (metadata["run_complete"][:] == 1)
        validity &= metadata_good[sample["granule_index"]]
        for bi, band in enumerate(base.BANDS):
            source_data = source[base.RADIANCE[band]]
            chunk_rows = source_data.chunks[0] if source_data.chunks else 4096
            unique, starts = np.unique(sorted_rows // chunk_rows, return_index=True)
            ends = np.r_[starts[1:], n]
            radiance = np.empty((n, base.NATIVE_CHANNELS), dtype="f4")
            print(f"{annual.name} {band}: reading {len(unique)} sequential chunks", flush=True)
            for number, (chunk, lo, hi) in enumerate(zip(unique, starts, ends), 1):
                start = int(chunk) * chunk_rows
                block = source_data[start:min(start + chunk_rows, len(source_data))]
                radiance[order[lo:hi]] = block[sorted_rows[lo:hi] - start]
                if number % 1000 == 0:
                    print(f"{annual.name} {band}: read {number}/{len(unique)}", flush=True)
            channels = len(protocol[band]["target"])
            spectra = np.zeros((n, channels), dtype="f4")
            masks = np.zeros((n, (channels + 7) // 8), dtype="u1")
            continuum_values = np.zeros(n, dtype="f8")
            for granule, fp, positions in groups:
                grid = base.wavelength(dispersion[granule])[bi, fp]
                left, right, weight, inside = base.interpolation_plan(grid, protocol[band]["target"])
                good = inside & ~bad_all[granule, bi, fp, left] & ~bad_all[granule, bi, fp, right]
                values = radiance[positions][:, left] * (1 - weight) + radiance[positions][:, right] * weight
                ci = protocol[band]["continuum"]
                ci = ci[good[ci]]
                continuum = np.mean(values[:, ci], axis=1, dtype="f8") if len(ci) >= 3 else np.zeros(len(positions))
                valid = np.isfinite(continuum) & (continuum > 0) & np.all(np.isfinite(values), axis=1)
                valid &= np.count_nonzero(good) > 50
                continuum_values[positions] = continuum
                validity[positions] &= valid
                normalized = np.zeros_like(values)
                normalized[valid] = values[valid] / continuum[valid, None]
                valid &= np.all(np.isfinite(normalized), axis=1)
                validity[positions] &= valid
                normalized[~valid] = 0
                spectra[positions] = normalized
                masks[positions] = np.packbits(good, bitorder="little")
            target.create_dataset(f"spectrum_{band}", data=spectra, chunks=(min(512, n), channels), compression="lzf")
            target.create_dataset(f"good_mask_{band}_packed", data=masks, compression="lzf")
            target.create_dataset(f"continuum_radiance_{band}", data=continuum_values, compression="lzf")
            target.create_dataset(f"target_wavelength_{band}_um", data=protocol[band]["target"])
            target.create_dataset(f"continuum_indices_{band}", data=protocol[band]["continuum"])
            del radiance, spectra, masks
            target.flush()
            print(f"{annual.name} {band}: resampling complete", flush=True)
        # Invalid radiance rows remain auditable; equal-size usable cohorts feed SVD and models.
        validity &= np.isfinite(sample["xco2"]) & (sample["xco2"] > 0)
        validity &= np.isfinite(sample["xco2_apriori"]) & (sample["xco2_apriori"] > 0)
        balanced = np.zeros(n, dtype=bool)
        pools = {domain: np.flatnonzero(validity & (sample["domain"] == domain.encode())) for domain in DOMAINS}
        budget = min(map(len, pools.values()))
        if budget < 100:
            raise RuntimeError("Insufficient usable rows after spectral validation")
        for domain, positions in pools.items():
            keys = base.splitmix64(sample["sounding_id"][positions], base.SEED)
            chosen = positions[np.argsort(keys, kind="stable")[:budget]]
            balanced[chosen] = True
        target.create_dataset("valid_spectrum", data=validity)
        target.create_dataset("selected_for_model", data=balanced)
        target.attrs["protocol_sha256"] = digest
        target.attrs["status"] = "complete"
        target.attrs["balanced_rows_per_domain"] = budget
        target.attrs["invalid_spectrum_rows"] = int(np.count_nonzero(~validity))
        target.attrs["normalization"] = "radiance divided by mean good points in frozen 15-point continuum window; minimum 3 good points"
    os.replace(partial, output)


def fit_domain_models(files: list[Path], output: Path, components: int, digest: str) -> dict:
    output.mkdir(exist_ok=True)
    models = {}
    summary = {}
    for domain in DOMAINS:
        models[domain] = {}
        domain_root = output / domain
        domain_root.mkdir(exist_ok=True)
        counts = []
        for path in files:
            with h5py.File(path) as source:
                keep = (source["domain"][:] == domain.encode()) & source["selected_for_model"][:]
                counts.append(int(keep.sum()))
        for band in base.BANDS:
            path = domain_root / f"svd_{band}.joblib"
            info_path = domain_root / f"svd_{band}.json"
            if path.exists() and info_path.exists():
                info = json.loads(info_path.read_text())
                assert info["protocol_sha256"] == digest and info["components"] == components
                models[domain][band] = joblib.load(path)
                summary[f"{domain}/{band}"] = info
                continue
            with h5py.File(files[0]) as source:
                channels = source[f"spectrum_{band}"].shape[1]
            mmap_path = domain_root / f"fit_{band}.mmap"
            pooled = np.memmap(mmap_path, mode="w+", dtype="f4", shape=(sum(counts), channels))
            cursor = 0
            for annual in files:
                with h5py.File(annual) as source:
                    keep = (source["domain"][:] == domain.encode()) & source["selected_for_model"][:]
                    for start in range(0, len(keep), 4096):
                        selected = keep[start:start + 4096]
                        if selected.any():
                            block = source[f"spectrum_{band}"][start:start + 4096][selected]
                            pooled[cursor:cursor + len(block)] = block
                            cursor += len(block)
            assert cursor == sum(counts)
            pooled.flush()
            print(f"SVD {domain}/{band}: fitting {pooled.shape}, components={components}", flush=True)
            model = TruncatedSVD(n_components=components, n_iter=7, random_state=base.SEED)
            model.fit(pooled)
            temporary = path.with_suffix(".joblib.partial")
            joblib.dump(model, temporary)
            os.replace(temporary, path)
            info = {
                "domain": domain, "band": band, "training_years": list(TRAIN_YEARS),
                "training_rows_per_year": counts, "shape": list(pooled.shape),
                "components": components, "mean_centered": False,
                "protocol_sha256": digest,
                "singular_values": model.singular_values_.tolist(),
                "explained_variance_ratio": model.explained_variance_ratio_.tolist(),
                "cumulative_remaining_variance": (1 - np.cumsum(model.explained_variance_ratio_)).tolist(),
            }
            json_write(info_path, info)
            models[domain][band] = model
            summary[f"{domain}/{band}"] = info
            del pooled
            mmap_path.unlink()
    refresh_variance_statistics(files, output, models, summary)
    joblib.dump(models, output / "domain_svd_models.joblib")
    json_write(output / "svd_summary.json", summary)
    return models


def merge_moments(previous: tuple | None, values: np.ndarray) -> tuple:
    values = np.asarray(values, dtype=np.float64)
    n = len(values)
    mean = values.mean(axis=0, dtype=np.float64)
    centered = values - mean
    m2 = np.sum(centered * centered, axis=0, dtype=np.float64)
    if previous is None:
        return n, mean, m2
    old_n, old_mean, old_m2 = previous
    total = old_n + n
    delta = mean - old_mean
    return total, old_mean + delta * (n / total), old_m2 + m2 + delta * delta * (old_n * n / total)


def refresh_variance_statistics(files: list[Path], output: Path, models: dict, summary: dict) -> None:
    method = "float64_streaming_centered_moments"
    if all(info.get("variance_statistics_method") == method for info in summary.values()):
        return
    print("Recomputing explained variance with float64 streaming moments; fitted bases remain unchanged", flush=True)
    moments = {(domain, band): [None, None] for domain in DOMAINS for band in base.BANDS}
    for path in files:
        with h5py.File(path) as source:
            domains = source["domain"][:]
            selected = source["selected_for_model"][:]
            for band in base.BANDS:
                for start in range(0, len(domains), 4096):
                    stop = min(start + 4096, len(domains))
                    values = source[f"spectrum_{band}"][start:stop].astype(np.float64)
                    for domain in DOMAINS:
                        keep = selected[start:stop] & (domains[start:stop] == domain.encode())
                        if not keep.any():
                            continue
                        block = values[keep]
                        scores = block @ models[domain][band].components_.astype(np.float64).T
                        state = moments[domain, band]
                        state[0] = merge_moments(state[0], block)
                        state[1] = merge_moments(state[1], scores)
                print(f"Stable variance moments: {path.name}/{band}", flush=True)
    for domain in DOMAINS:
        for band in base.BANDS:
            model = models[domain][band]
            info = summary[f"{domain}/{band}"]
            raw, score = moments[domain, band]
            assert raw[0] == score[0] == info["shape"][0]
            total_variance = float(raw[2].sum() / raw[0])
            explained = score[2] / score[0]
            ratio = explained / total_variance
            residual = 1.0 - np.cumsum(ratio)
            if not np.all(np.isfinite(ratio)) or residual[-1] < -1e-5:
                raise RuntimeError(f"Invalid float64 variance statistics: {domain}/{band}/{residual[-1]}")
            fingerprint = hashlib.sha256(model.components_.tobytes()).hexdigest()
            directory = output / domain
            for suffix in ("joblib", "json"):
                original = directory / f"svd_{band}.{suffix}"
                backup = original.with_name(original.name + ".before_float64_variance")
                if original.exists() and not backup.exists():
                    shutil.copy2(original, backup)
            info.setdefault("original_sklearn_explained_variance_ratio", info["explained_variance_ratio"])
            info["explained_variance_ratio"] = ratio.tolist()
            info["cumulative_remaining_variance"] = residual.tolist()
            info["variance_statistics_method"] = method
            info["basis_sha256"] = fingerprint
            model.explained_variance_ = explained
            model.explained_variance_ratio_ = ratio
            assert hashlib.sha256(model.components_.tobytes()).hexdigest() == fingerprint
            path = directory / f"svd_{band}.joblib"
            temporary = path.with_suffix(".joblib.partial")
            joblib.dump(model, temporary)
            os.replace(temporary, path)
            json_write(directory / f"svd_{band}.json", info)
            print(f"Stable variance {domain}/{band}: {100 * residual[-1]:.6f}% remaining", flush=True)


def projection_selection(source: h5py.File) -> tuple[np.ndarray, dict]:
    selected = np.flatnonzero(source["selected_for_model"][:])
    ids = source["sounding_id"][:]
    domains = source["domain"][:]
    unique, first, inverse, counts = np.unique(ids[selected], return_index=True, return_inverse=True, return_counts=True)
    duplicates = []
    payload_names = ["mode", "domain", "footprint_index", *base.MODEL_SCALARS]
    payload_names += [f"{prefix}_{band}" for band in base.BANDS for prefix in ("spectrum", "continuum_radiance")]
    payload_names += [f"good_mask_{band}_packed" for band in base.BANDS]
    for index in np.flatnonzero(counts > 1):
        positions = selected[inverse == index]
        for name in payload_names:
            if name not in source:
                continue
            values = base.indexed(source[name], positions)
            for row in values[1:]:
                equal = np.array_equal(row, values[0], equal_nan=True) if values.dtype.kind in "fc" else np.array_equal(row, values[0])
                if not equal:
                    raise RuntimeError(f"Conflicting duplicate sounding {int(unique[index])}: {name}")
        duplicates.append({"sounding_id": int(unique[index]), "resampled_indices": positions.tolist(),
                           "original_indices": base.indexed(source["rows"], positions).tolist() if "rows" in source else []})
    deduplicated = selected[np.sort(first)]
    counts_by_domain = {domain: int(np.count_nonzero(domains[deduplicated] == domain.encode())) for domain in DOMAINS}
    budget = min(counts_by_domain.values())
    chosen = []
    for domain in DOMAINS:
        candidates = deduplicated[domains[deduplicated] == domain.encode()]
        if len(candidates) > budget:
            keys = base.splitmix64(ids[candidates], base.SEED)
            candidates = candidates[np.argsort(keys, kind="stable")[:budget]]
        chosen.append(candidates)
    keep = np.sort(np.concatenate(chosen))
    return keep, {"selected_before_deduplication": len(selected), "duplicate_rows_removed": len(selected) - len(deduplicated),
                  "unique_counts_before_rebalancing": counts_by_domain, "final_rows_per_domain": budget,
                  "final_total_rows": len(keep), "duplicate_groups": duplicates}


def project(annual: Path, output: Path, models: dict, digest: str) -> None:
    ranks = {model.components_.shape[0] for bands in models.values() for model in bands.values()}
    if len(ranks) != 1:
        raise ValueError("All domain/band bases must use the same saved component count")
    rank = ranks.pop()
    if output.exists():
        with h5py.File(output) as source:
            assert source.attrs["protocol_sha256"] == digest
            assert source["pc_O2"].shape[1] == rank
        return
    partial = output.with_suffix(".h5.partial")
    if partial.exists():
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        backup = partial.with_name(partial.name + f".preserved_{stamp}")
        os.replace(partial, backup)
        print(f"Preserved earlier partial projection: {backup.name}", flush=True)
    with h5py.File(annual) as source, h5py.File(partial, "w") as target:
        selected, audit = projection_selection(source)
        json_write(output.with_suffix(".selection.json"), audit)
        if audit["duplicate_rows_removed"]:
            print(f"{output.name}: removed {audit['duplicate_rows_removed']} identical duplicate rows; {audit['final_rows_per_domain']} unique rows per domain", flush=True)
        n = len(selected)
        for key in ("sounding_id", "rows", "granule_index", "footprint_index", "mode", "domain", *base.MODEL_SCALARS):
            target.create_dataset(key, data=source[key][:][selected], compression="lzf")
        target.create_dataset("resampled_index", data=selected)
        for band in base.BANDS:
            target.create_dataset(f"pc_{band}", shape=(n, rank), dtype="f4", compression="lzf")
            target.create_dataset(f"valid_channels_{band}", shape=(n,), dtype="i2")
            target.create_dataset(f"reconstruction_rmse_{band}", shape=(n,), dtype="f4")
            target.create_dataset(f"continuum_radiance_{band}", data=source[f"continuum_radiance_{band}"][:][selected])
        cache = OrderedDict()
        domains = source["domain"][:]
        for start in range(0, n, 4096):
            positions = selected[start:start + 4096]
            for band in base.BANDS:
                spectra = base.indexed(source[f"spectrum_{band}"], positions).astype("f8")
                packed = base.indexed(source[f"good_mask_{band}_packed"], positions)
                scores = np.full((len(positions), rank), np.nan, dtype="f4")
                valid_count = np.zeros(len(positions), dtype="i2")
                reconstruction = np.zeros(len(positions), dtype="f4")
                for domain in DOMAINS:
                    local = np.flatnonzero(domains[positions] == domain.encode())
                    unique, inverse = np.unique(packed[local], axis=0, return_inverse=True)
                    components = models[domain][band].components_.astype("f8")
                    for mask_id, mask in enumerate(unique):
                        rows = local[inverse == mask_id]
                        key = (domain, band, mask.tobytes())
                        if key not in cache:
                            good = np.unpackbits(mask, bitorder="little")[:components.shape[1]].astype(bool)
                            design = components[:, good].T
                            if design.shape[0] <= rank or np.linalg.matrix_rank(design) < rank:
                                raise RuntimeError(f"Rank-deficient projection: {domain}/{band}")
                            cache[key] = (good, np.linalg.pinv(design))
                            if len(cache) > 1024:
                                cache.popitem(last=False)
                        good, pseudo = cache[key]
                        cache.move_to_end(key)
                        scores[rows] = spectra[rows][:, good] @ pseudo.T
                        valid_count[rows] = good.sum()
                        reconstructed = scores[rows].astype("f8") @ components[:, good]
                        reconstruction[rows] = np.sqrt(np.mean((reconstructed - spectra[rows][:, good]) ** 2, axis=1))
                if not np.all(np.isfinite(scores)):
                    raise RuntimeError(f"Non-finite scores: {annual.name}/{band}/{start}")
                target[f"pc_{band}"][start:start + len(positions)] = scores
                target[f"valid_channels_{band}"][start:start + len(positions)] = valid_count
                target[f"reconstruction_rmse_{band}"][start:start + len(positions)] = reconstruction
            if start % (4096 * 25) == 0:
                print(f"{output.name}: projected {start}/{n}", flush=True)
        assert len(np.unique(target["sounding_id"][:])) == n
        target.attrs["protocol_sha256"] = digest
        target.attrs["status"] = "complete"
        target.attrs["components_per_band"] = rank
        target.attrs["projection"] = "domain-specific masked least squares; bad interpolation support excluded"
    os.replace(partial, output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--sample-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--components", type=int, default=30)
    args = parser.parse_args()
    if args.components < 10:
        parser.error("Save at least ten components for the requested PC1-PC10 plots")
    args.output.mkdir(parents=True, exist_ok=True)
    protocol, digest = freeze_protocol(args.annual_root, args.sample_root, args.output)
    print(f"Configured SVD and full projection: {args.components} PCs per band; {3 * args.components} stored scores per sounding; regression training disabled", flush=True)
    print("Frozen protocol: " + ", ".join(f"{band}={len(protocol[band]['target'])} channels" for band in base.BANDS), flush=True)
    for year in range(2015, 2026):
        path = args.output / f"resampled_{year}.h5"
        sample = load_sample(args.sample_root / f"sample_{year}.npz")
        resample(args.annual_root / f"OCO2_L1B_L2_{year}.h5", sample, protocol, path, digest)
        if year == 2019:
            models = fit_domain_models(
                [args.output / f"resampled_{value}.h5" for value in TRAIN_YEARS],
                args.output / "svd", args.components, digest,
            )
            subprocess.run([sys.executable, str(Path(__file__).with_name("plot_domain_svd50.py")),
                            "--root", str(args.output)], check=True)
            for value in TRAIN_YEARS:
                project(args.output / f"resampled_{value}.h5", args.output / f"projected_{value}.h5", models, digest)
        elif year > 2019:
            project(path, args.output / f"projected_{year}.h5", models, digest)
    counts = {}
    for year in range(2015, 2026):
        with h5py.File(args.output / f"projected_{year}.h5") as source:
            names, values = np.unique(source["domain"][:], return_counts=True)
            counts[str(year)] = {name.decode(): int(value) for name, value in zip(names, values)}
            assert len(set(counts[str(year)].values())) == 1
    json_write(args.output / "spectral.complete.json", {
        "status": "complete", "domain_counts": counts, "protocol_sha256": digest,
        "components_per_band": args.components, "svd_training_years": list(TRAIN_YEARS),
    })


if __name__ == "__main__":
    main()
