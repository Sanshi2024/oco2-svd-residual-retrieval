#!/usr/bin/env python3
"""Formal balanced ND / GL-ocean / GL-land spectral preprocessing pipeline.

The fixed wavelength protocol is derived from 2015-2019 metadata. Each year
contributes an equal number of rows from the three retrieval domains. The
larger per-domain sample is resampled once for downstream modelling, while a
deterministic nested subset from 2015-2019 fits the common SVD basis.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import OrderedDict
from datetime import datetime, timezone
from pathlib import Path

import h5py
import joblib
import numpy as np
from sklearn.decomposition import TruncatedSVD


BANDS = ("O2", "W", "S")
BAND_LABELS = {"O2": "O2-A", "W": "Weak CO2", "S": "Strong CO2"}
RADIANCE = {"O2": "radiance_o2", "W": "radiance_weak_co2", "S": "radiance_strong_co2"}
NATIVE_CHANNELS = 1016
SEED = 20261014
DOMAINS = (b"ND", b"GL_OCEAN", b"GL_LAND")
LAND_FILL = -999999.0
MODEL_SCALARS = (
    "solar_zenith",
    "sensor_zenith",
    "AZA",
    "retrieval_solar_distance",
    "retrieval_solar_relative_velocity",
    "surface_pressure_apriori_fph",
    "surface_pressure_apriori_abp",
    "surface_pressure_fph",
    "surface_pressure_abp",
    "aod_total",
    "albedo_o2",
    "albedo_weak_co2",
    "albedo_strong_co2",
    "temperature_offset_fph",
    "wind_speed",
    "xco2",
    "xco2_apriori",
    "xco2_uncert",
    "lat",
    "lon",
)


def splitmix64(values: np.ndarray, seed: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.uint64) + np.uint64(seed)
    values += np.uint64(0x9E3779B97F4A7C15)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    return values ^ (values >> np.uint64(31))


def trim(rows: np.ndarray, keys: np.ndarray, size: int) -> tuple[np.ndarray, np.ndarray]:
    if len(rows) <= size:
        return rows, keys
    selected = np.argpartition(keys, size - 1)[:size]
    return rows[selected], keys[selected]


def indexed(dataset: h5py.Dataset, rows: np.ndarray) -> np.ndarray:
    order = np.argsort(rows, kind="stable")
    values = dataset[rows[order]]
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    return values[inverse]


def sequential_indexed(
    dataset: h5py.Dataset, rows: np.ndarray, block_rows: int = 500_000
) -> np.ndarray:
    """Gather sorted random rows through bounded contiguous HDF5 reads."""
    rows = np.asarray(rows, dtype=np.int64)
    if np.any(np.diff(rows) <= 0):
        raise RuntimeError("sequential_indexed requires strictly increasing rows")
    output = np.empty((len(rows),) + dataset.shape[1:], dtype=dataset.dtype)
    cursor = 0
    for start in range(0, len(dataset), block_rows):
        stop = min(start + block_rows, len(dataset))
        end = int(np.searchsorted(rows, stop, side="left"))
        if end == cursor:
            continue
        block = dataset[start:stop]
        output[cursor:end] = block[rows[cursor:end] - start]
        cursor = end
    if cursor != len(rows):
        raise RuntimeError(f"Gathered {cursor:,}/{len(rows):,} rows from {dataset.name}")
    return output


def classify_domains(mode: np.ndarray, wind_speed: np.ndarray) -> np.ndarray:
    domain = np.full(len(mode), b"OTHER", dtype="S10")
    domain[mode == b"ND"] = b"ND"
    gl = mode == b"GL"
    domain[gl & (wind_speed == LAND_FILL)] = b"GL_LAND"
    domain[gl & np.isfinite(wind_speed) & (wind_speed >= 0)] = b"GL_OCEAN"
    return domain


def sample_year(
    path: Path,
    rows_per_domain: int,
    svd_rows_per_domain: int,
    seed: int,
    chunk: int = 500_000,
) -> dict:
    if svd_rows_per_domain > rows_per_domain:
        raise ValueError("SVD subset cannot exceed the modelling sample")
    pools = {
        domain: {
            "rows": np.empty(0, dtype=np.int64),
            "keys": np.empty(0, dtype=np.uint64),
            "eligible": 0,
        }
        for domain in DOMAINS
    }
    with h5py.File(path, "r", locking=False) as source:
        meta = source["spectral_metadata/rows"]
        granules = source["spectral_metadata/granules"]
        valid_granules = (
            (granules["l1b_metadata_valid"][:] == 1)
            & (granules["run_complete"][:] == 1)
        )
        total = len(source["sounding_id"])
        for start in range(0, total, chunk):
            stop = min(start + chunk, total)
            mode = source["mode"][start:stop]
            wind_speed = source["wind_speed"][start:stop]
            domain_values = classify_domains(mode, wind_speed)
            footprint = meta["footprint_index"][start:stop]
            granule_index = meta["granule_index"][start:stop]
            row_metadata_valid = np.zeros(stop - start, dtype=bool)
            indexed_rows = granule_index >= 0
            row_metadata_valid[indexed_rows] = valid_granules[granule_index[indexed_rows]]
            metadata_valid = (
                (footprint >= 0)
                & (footprint < 8)
                & row_metadata_valid
            )
            for domain_index, domain in enumerate(DOMAINS):
                local = np.flatnonzero(metadata_valid & (domain_values == domain))
                pools[domain]["eligible"] += len(local)
                if not len(local):
                    continue
                rows = start + local.astype(np.int64)
                ids = source["sounding_id"][rows].astype(np.uint64, copy=False)
                keys = splitmix64(ids, seed + domain_index * 1_000_003)
                pools[domain]["rows"] = np.concatenate((pools[domain]["rows"], rows))
                pools[domain]["keys"] = np.concatenate((pools[domain]["keys"], keys))
                if len(pools[domain]["rows"]) >= 2 * rows_per_domain:
                    pools[domain]["rows"], pools[domain]["keys"] = trim(
                        pools[domain]["rows"], pools[domain]["keys"], rows_per_domain
                    )

    selected_rows = []
    selected_keys = []
    selected_domains = []
    selected_svd = []
    eligible_counts = []
    effective_rows_per_domain = min(
        rows_per_domain,
        *(int(pools[domain]["eligible"]) for domain in DOMAINS),
    )
    effective_svd_rows = min(svd_rows_per_domain, effective_rows_per_domain)
    for domain in DOMAINS:
        rows, keys = trim(
            pools[domain]["rows"], pools[domain]["keys"], effective_rows_per_domain
        )
        if len(rows) != effective_rows_per_domain:
            raise RuntimeError(
                f"Only {len(rows):,} eligible {domain.decode()} rows in {path}; "
                f"need {effective_rows_per_domain:,}"
            )
        rank = np.argsort(keys, kind="stable")
        svd_fit = np.zeros(effective_rows_per_domain, dtype=bool)
        svd_fit[rank[:effective_svd_rows]] = True
        selected_rows.append(rows)
        selected_keys.append(keys)
        selected_domains.append(np.full(effective_rows_per_domain, domain, dtype="S10"))
        selected_svd.append(svd_fit)
        eligible_counts.append(pools[domain]["eligible"])

    rows = np.concatenate(selected_rows)
    keys = np.concatenate(selected_keys)
    domains = np.concatenate(selected_domains)
    svd_fit = np.concatenate(selected_svd)
    order = np.argsort(rows, kind="stable")
    rows, keys, domains, svd_fit = (
        rows[order], keys[order], domains[order], svd_fit[order]
    )
    with h5py.File(path, "r", locking=False) as source:
        meta = source["spectral_metadata/rows"]
        result = {
            "rows": rows,
            "keys": keys,
            "sounding_id": sequential_indexed(source["sounding_id"], rows).astype(np.int64),
            "granule_index": sequential_indexed(meta["granule_index"], rows).astype(np.int32),
            "footprint_index": sequential_indexed(meta["footprint_index"], rows).astype(np.int8),
            "mode": sequential_indexed(source["mode"], rows).astype("S2"),
            "domain": domains,
            "svd_fit": svd_fit,
            "eligible_by_domain": np.asarray(eligible_counts, dtype=np.int64),
            "requested_rows_per_domain": np.asarray(rows_per_domain, dtype=np.int64),
            "selected_rows_per_domain": np.asarray(effective_rows_per_domain, dtype=np.int64),
        }
        for name in MODEL_SCALARS:
            result[name] = sequential_indexed(source[name], rows)
        return result


def wavelength(coefficients: np.ndarray) -> np.ndarray:
    samples = np.arange(1, NATIVE_CHANNELS + 1, dtype=np.float64)
    return np.polynomial.polynomial.polyval(samples, np.moveaxis(coefficients, -1, 0))


def interpolation_plan(native: np.ndarray, target: np.ndarray):
    right = np.searchsorted(native, target, side="right")
    in_range = (target >= native[0]) & (target <= native[-1])
    right = np.clip(right, 1, len(native) - 1)
    left = right - 1
    weight = (target - native[left]) / (native[right] - native[left])
    return left, right, weight.astype(np.float32), in_range


def pair_counts(granule: np.ndarray, footprint: np.ndarray):
    pair = np.stack((granule.astype(np.int64), footprint.astype(np.int64)), axis=1)
    unique, counts = np.unique(pair, axis=0, return_counts=True)
    return unique, counts


def derive_protocol(
    metadata_paths: dict[int, Path],
    output: Path,
    edge_bad_frequency_threshold: float = 1.0,
) -> dict:
    """Freeze wavelength grids and edge crops from granule metadata only."""
    bad_count = np.zeros((3, NATIVE_CHANNELS), dtype=np.int64)
    wavelength_sum = np.zeros((3, NATIVE_CHANNELS), dtype=np.float64)
    total_granule_footprints = 0
    valid_by_year = {}

    for year, path in metadata_paths.items():
        accepted = 0
        with h5py.File(path, "r", locking=False) as source:
            group = source["spectral_metadata/granules"]
            valid = (
                (group["l1b_metadata_valid"][:] == 1)
                & (group["run_complete"][:] == 1)
                & np.isin(group["mode"][:], (b"GL", b"ND"))
            )
            for granule in np.flatnonzero(valid):
                coefficients = group["dispersion_coef_samp"][granule]
                if not np.all(np.isfinite(coefficients)):
                    continue
                grid = wavelength(coefficients)
                if not np.all(np.isfinite(grid)) or not np.all(np.diff(grid, axis=-1) > 0):
                    continue
                bad = group["bad_sample_list"][granule] != 0
                wavelength_sum += grid.sum(axis=1)
                bad_count += bad.sum(axis=1, dtype=np.int64)
                total_granule_footprints += grid.shape[1]
                accepted += 1
        valid_by_year[str(year)] = accepted

    if total_granule_footprints == 0:
        raise RuntimeError("No valid GL/ND granule metadata available for wavelength protocol")
    bad_frequency = bad_count / total_granule_footprints
    reference = wavelength_sum / total_granule_footprints
    protocol = {}
    saved = {"bad_frequency_native": bad_frequency, "reference_wavelength_native_um": reference}
    summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "protocol_years": list(metadata_paths),
        "valid_gl_nd_granules_by_year": valid_by_year,
        "total_granule_footprints": total_granule_footprints,
        "reference_grid": "equal-weight mean across all valid GL/ND granules and all eight footprints",
        "edge_rule": (
            "remove contiguous native edge channels bad in at least "
            f"{100 * edge_bad_frequency_threshold:.1f}% of valid granule-footprints"
        ),
        "edge_bad_frequency_threshold": edge_bad_frequency_threshold,
        "annual_samples_used_for_grid": False,
        "bands": {},
    }
    for index, band in enumerate(BANDS):
        left = 0
        while (
            left < NATIVE_CHANNELS
            and bad_frequency[index, left] >= edge_bad_frequency_threshold
        ):
            left += 1
        right = NATIVE_CHANNELS
        while (
            right > left
            and bad_frequency[index, right - 1] >= edge_bad_frequency_threshold
        ):
            right -= 1
        target = reference[index, left:right]
        protocol[band] = {"left": left, "right": right, "target": target}
        saved[f"{band}_target_wavelength_um"] = target
        summary["bands"][band] = {
            "label": BAND_LABELS[band],
            "left_removed": left,
            "right_removed": NATIVE_CHANNELS - right,
            "retained_channels": len(target),
            "native_bad_fraction_percent": float(100 * bad_frequency[index].mean()),
            "post_crop_bad_fraction_percent": float(100 * bad_frequency[index, left:right].mean()),
        }
    np.savez_compressed(output / "protocol.npz", **saved)
    (output / "protocol.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return protocol

def group_positions(granule: np.ndarray, footprint: np.ndarray):
    order = np.lexsort((footprint, granule))
    ordered_granule = granule[order]
    ordered_footprint = footprint[order]
    transitions = np.flatnonzero(
        (ordered_granule[1:] != ordered_granule[:-1])
        | (ordered_footprint[1:] != ordered_footprint[:-1])
    ) + 1
    for positions in np.split(order, transitions):
        if len(positions):
            yield int(granule[positions[0]]), int(footprint[positions[0]]), positions


def preview_resampled(path: Path, sample: dict, protocol: dict, limit: int = 500):
    per_domain = max(1, limit // len(DOMAINS))
    chosen = np.concatenate([
        np.flatnonzero(sample["domain"] == domain)[:per_domain]
        for domain in DOMAINS
    ])
    preview = {band: [] for band in BANDS}
    with h5py.File(path, "r", locking=False) as source:
        granules = source["spectral_metadata/granules"]
        for granule, footprint, local in group_positions(
            sample["granule_index"][chosen], sample["footprint_index"][chosen]
        ):
            positions = np.sort(chosen[local])
            coefficients = granules["dispersion_coef_samp"][granule]
            grid = wavelength(coefficients)
            for band_index, band in enumerate(BANDS):
                radiance = indexed(source[RADIANCE[band]], sample["rows"][positions]).astype(np.float32)
                left, right, weight, _ = interpolation_plan(grid[band_index, footprint], protocol[band]["target"])
                values = radiance[:, left] * (1 - weight) + radiance[:, right] * weight
                preview[band].append(values)
    return {band: np.concatenate(parts, axis=0) for band, parts in preview.items()}


def choose_continuum(previews: list[dict], protocol: dict, count: int = 15) -> None:
    for band in BANDS:
        spectra = np.concatenate([item[band] for item in previews], axis=0)
        median = np.nanmedian(spectra, axis=0)
        best_start = max(
            range(len(median) - count + 1),
            key=lambda start: float(np.nanmean(median[start : start + count])),
        )
        protocol[band]["continuum"] = np.arange(best_start, best_start + count)


def resample_year(path: Path, sample: dict, protocol: dict, output: Path) -> None:
    """Resample a random sample using one sequential read per source HDF5 chunk."""
    if output.exists():
        return
    partial = output.with_suffix(".h5.partial")
    if partial.exists():
        partial.unlink()
    n = len(sample["rows"])
    row_order = np.argsort(sample["rows"], kind="stable")
    sorted_rows = sample["rows"][row_order]
    grouped = list(group_positions(sample["granule_index"], sample["footprint_index"]))
    native_samples = np.arange(1, NATIVE_CHANNELS + 1, dtype=np.float64)

    with h5py.File(path, "r", locking=False) as source, h5py.File(partial, "w") as target:
        target.attrs["status"] = "complete"
        target.attrs["method"] = "metadata-only wavelength grid; chunk-sequential reads; edge crop; continuum normalization"
        for name in (
            "sounding_id", "rows", "granule_index", "footprint_index", "mode",
            "domain", "svd_fit", *MODEL_SCALARS,
        ):
            target.create_dataset(name, data=sample[name], compression="gzip", shuffle=True)
        for band in BANDS:
            channels = len(protocol[band]["target"])
            target.create_dataset(
                f"spectrum_{band}", (n, channels), dtype="f4",
                chunks=(min(512, n), channels), compression="lzf"
            )
            target.create_dataset(
                f"good_mask_{band}_packed", (n, (channels + 7) // 8), dtype="u1",
                chunks=(min(2048, n), (channels + 7) // 8), compression="lzf"
            )
            target.create_dataset(f"target_wavelength_{band}_um", data=protocol[band]["target"])
            target.create_dataset(f"continuum_indices_{band}", data=protocol[band]["continuum"])

        granules = source["spectral_metadata/granules"]
        dispersion = granules["dispersion_coef_samp"][:]
        bad_all = granules["bad_sample_list"][:] != 0

        for band_index, band in enumerate(BANDS):
            dataset = source[RADIANCE[band]]
            chunk_rows = dataset.chunks[0] if dataset.chunks else 4096
            chunk_ids = sorted_rows // chunk_rows
            unique_chunks, starts = np.unique(chunk_ids, return_index=True)
            ends = np.r_[starts[1:], len(sorted_rows)]
            radiance_cache = np.empty((n, NATIVE_CHANNELS), dtype=np.float32)
            print(f"{output.name} {band}: reading {len(unique_chunks):,} source chunks sequentially", flush=True)
            for number, (chunk_id, lo, hi) in enumerate(zip(unique_chunks, starts, ends), start=1):
                block_start = int(chunk_id) * chunk_rows
                block_stop = min(block_start + chunk_rows, len(dataset))
                block = dataset[block_start:block_stop]
                local_rows = sorted_rows[lo:hi] - block_start
                radiance_cache[row_order[lo:hi]] = block[local_rows]
                if number % 500 == 0:
                    print(f"{output.name} {band}: read {number:,}/{len(unique_chunks):,} chunks", flush=True)

            channels = len(protocol[band]["target"])
            spectra_out = np.empty((n, channels), dtype=np.float32)
            packed_out = np.empty((n, (channels + 7) // 8), dtype=np.uint8)
            for group_number, (granule, footprint, positions) in enumerate(grouped, start=1):
                positions = np.sort(positions)
                native_grid = np.polynomial.polynomial.polyval(
                    native_samples, dispersion[granule, band_index, footprint]
                )
                left, right, weight, in_range = interpolation_plan(
                    native_grid, protocol[band]["target"]
                )
                radiance = radiance_cache[positions]
                values = radiance[:, left] * (1 - weight) + radiance[:, right] * weight
                continuum = np.mean(values[:, protocol[band]["continuum"]], axis=1, dtype=np.float64)
                valid_continuum = np.isfinite(continuum) & (continuum > 0)
                if not np.all(valid_continuum):
                    invalid_positions = positions[~valid_continuum]
                    details = [
                        {
                            "sample_position": int(position),
                            "source_row": int(sample["rows"][position]),
                            "sounding_id": int(sample["sounding_id"][position]),
                            "continuum": float(continuum[offset]),
                        }
                        for offset, position in zip(
                            np.flatnonzero(~valid_continuum), invalid_positions
                        )
                    ]
                    raise RuntimeError(
                        f"Invalid {band} continuum in granule {granule}, "
                        f"footprint {footprint}: {details}"
                    )
                values /= continuum[:, None].astype(np.float32)
                if not np.all(np.isfinite(values)):
                    raise RuntimeError(f"Non-finite resampled spectrum in granule {granule}")
                bad = bad_all[granule, band_index, footprint]
                good = in_range & ~bad[left] & ~bad[right]
                spectra_out[positions] = values
                packed = np.packbits(good, bitorder="little")
                packed_out[positions] = packed
                if group_number % 2500 == 0:
                    print(f"{output.name} {band}: resampled {group_number:,}/{len(grouped):,} groups", flush=True)

            for start_row in range(0, n, 4096):
                stop_row = min(start_row + 4096, n)
                target[f"spectrum_{band}"][start_row:stop_row] = spectra_out[start_row:stop_row]
                target[f"good_mask_{band}_packed"][start_row:stop_row] = packed_out[start_row:stop_row]
            target.flush()
            del radiance_cache, spectra_out, packed_out
            print(f"{output.name} {band}: complete", flush=True)
    os.replace(partial, output)

def fit_svd(year_files: list[Path], protocol: dict, output: Path, components: int) -> dict:
    model_path = output / "svd_models.joblib"
    if model_path.exists():
        return joblib.load(model_path)
    models = {}
    summary = {}
    total = 0
    for path in year_files:
        with h5py.File(path, "r") as source:
            total += int(np.count_nonzero(source["svd_fit"][:]))
    for band in BANDS:
        channels = len(protocol[band]["target"])
        mmap_path = output / f"fit_{band}.mmap"
        pooled = np.memmap(mmap_path, mode="w+", dtype="f4", shape=(total, channels))
        cursor = 0
        for path in year_files:
            with h5py.File(path, "r") as source:
                dataset = source[f"spectrum_{band}"]
                fit_mask = source["svd_fit"][:].astype(bool)
                for start in range(0, len(dataset), 5000):
                    stop = min(start + 5000, len(dataset))
                    block_mask = fit_mask[start:stop]
                    if not np.any(block_mask):
                        continue
                    block = dataset[start:stop][block_mask]
                    pooled[cursor:cursor + len(block)] = block
                    cursor += len(block)
        if cursor != total:
            raise RuntimeError(f"SVD row-count mismatch: wrote {cursor:,}, expected {total:,}")
        pooled.flush()
        model = TruncatedSVD(n_components=components, n_iter=7, random_state=SEED)
        model.fit(pooled)
        models[band] = model
        summary[band] = {
            "shape": [total, channels],
            "components": components,
            "explained_variance_ratio_sum": float(model.explained_variance_ratio_.sum()),
            "singular_values": model.singular_values_.tolist(),
        }
        del pooled
        mmap_path.unlink()
    joblib.dump(models, model_path)
    (output / "svd_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return models


def masked_project(input_path: Path, output_path: Path, models: dict, block_size: int = 2048) -> None:
    if output_path.exists():
        return
    partial = output_path.with_suffix(".h5.partial")
    if partial.exists():
        partial.unlink()
    with h5py.File(input_path, "r") as source, h5py.File(partial, "w") as target:
        n = len(source["sounding_id"])
        for name in (
            "sounding_id", "rows", "granule_index", "footprint_index", "mode",
            "domain", "svd_fit", *MODEL_SCALARS,
        ):
            target.create_dataset(name, data=source[name][:], compression="gzip", shuffle=True)
        for band, model in models.items():
            target.create_dataset(
                f"pc_{band}", (n, model.n_components), dtype="f4",
                chunks=(min(4096, n), model.n_components)
            )
            target.create_dataset(
                f"valid_channels_{band}", (n,), dtype="i2",
                chunks=(min(65536, n),), compression="gzip"
            )
            components = model.components_.astype(np.float64)
            channels = components.shape[1]
            for start in range(0, n, block_size):
                stop = min(start + block_size, n)
                spectra = source[f"spectrum_{band}"][start:stop].astype(np.float64)
                packed = source[f"good_mask_{band}_packed"][start:stop]
                unique, inverse = np.unique(packed, axis=0, return_inverse=True)
                scores = np.full((len(spectra), model.n_components), np.nan, dtype=np.float32)
                valid_count = np.zeros(len(spectra), dtype=np.int16)
                for group_index, mask_packed in enumerate(unique):
                    rows = np.flatnonzero(inverse == group_index)
                    good = np.unpackbits(mask_packed, bitorder="little")[:channels].astype(bool)
                    valid_count[rows] = int(good.sum())
                    design = components[:, good].T
                    if design.shape[0] <= model.n_components or np.linalg.matrix_rank(design) < model.n_components:
                        continue
                    pseudo = np.linalg.pinv(design)
                    scores[rows] = (spectra[rows][:, good] @ pseudo.T).astype(np.float32)
                if not np.all(np.isfinite(scores)):
                    raise RuntimeError(f"Masked projection failed for {band}, rows {start}:{stop}")
                target[f"pc_{band}"][start:stop] = scores
                target[f"valid_channels_{band}"][start:stop] = valid_count
        target.attrs["status"] = "complete"
        target.attrs["projection"] = "sample-specific masked least-squares projection"
    os.replace(partial, output_path)


def append_rows(target: h5py.File, payload: dict) -> None:
    count = len(payload["sounding_id"])
    if not count:
        return
    for name, values in payload.items():
        values = np.asarray(values)
        if name not in target:
            shape = (0,) + values.shape[1:]
            chunks = (min(8192, max(1, count)),) + values.shape[1:]
            kwargs = dict(shape=shape, maxshape=(None,) + values.shape[1:], dtype=values.dtype, chunks=chunks)
            if values.dtype.kind not in "SUO":
                kwargs.update(compression="lzf", shuffle=True)
            target.create_dataset(name, **kwargs)
        dataset = target[name]
        start = len(dataset)
        dataset.resize(start + count, axis=0)
        dataset[start:] = values


def project_all_eligible(
    annual_path: Path,
    output_path: Path,
    protocol: dict,
    models: dict,
    chunk_size: int = 100_000,
) -> int:
    """Resample and project every eligible GL/ND row without storing full spectra."""
    if output_path.exists():
        with h5py.File(output_path, "r") as source:
            return len(source["sounding_id"])
    partial = output_path.with_suffix(".h5.partial")
    if partial.exists():
        partial.unlink()
    written = 0
    component_map = {
        band: models[band].components_.astype(np.float64) for band in BANDS
    }
    projection_cache = {band: OrderedDict() for band in BANDS}
    with h5py.File(annual_path, "r", locking=False) as source, h5py.File(partial, "w") as target:
        meta = source["spectral_metadata/rows"]
        granules = source["spectral_metadata/granules"]
        total = len(source["sounding_id"])
        for start in range(0, total, chunk_size):
            stop = min(start + chunk_size, total)
            mode = source["mode"][start:stop]
            footprint_all = meta["footprint_index"][start:stop]
            keep = (
                np.isin(mode, (b"GL", b"ND"))
                & (footprint_all >= 0)
                & (footprint_all < 8)
                & (meta["granule_index"][start:stop] >= 0)
            )
            local = np.flatnonzero(keep)
            if not len(local):
                continue
            rows = start + local.astype(np.int64)
            granule_index = meta["granule_index"][rows].astype(np.int32)
            footprint = footprint_all[local].astype(np.int8)
            payload = {
                "sounding_id": source["sounding_id"][rows].astype(np.int64),
                "original_index": rows,
                "mode": np.asarray(mode[local], dtype="S2"),
                "granule_index": granule_index,
                "footprint_index": footprint,
                "xco2": source["xco2"][rows].astype(np.float32),
                "xco2_apriori": source["xco2_apriori"][rows].astype(np.float32),
                "xco2_uncert": source["xco2_uncert"][rows].astype(np.float32),
                "latitude": source["lat"][rows].astype(np.float32),
                "longitude": source["lon"][rows].astype(np.float32),
            }
            for band in BANDS:
                payload[f"pc_{band}"] = np.full(
                    (len(rows), models[band].n_components), np.nan, dtype=np.float32
                )
                payload[f"valid_channels_{band}"] = np.zeros(len(rows), dtype=np.int16)

            for granule, fp, positions in group_positions(granule_index, footprint):
                positions = np.sort(positions)
                coefficients = granules["dispersion_coef_samp"][granule]
                bad = granules["bad_sample_list"][granule] != 0
                grid = wavelength(coefficients)
                for band_index, band in enumerate(BANDS):
                    radiance = source[RADIANCE[band]][rows[positions]].astype(np.float32)
                    left, right, weight, in_range = interpolation_plan(
                        grid[band_index, fp], protocol[band]["target"]
                    )
                    values = radiance[:, left] * (1 - weight) + radiance[:, right] * weight
                    continuum = np.mean(
                        values[:, protocol[band]["continuum"]], axis=1, dtype=np.float64
                    )
                    if not np.all(np.isfinite(continuum) & (continuum > 0)):
                        raise RuntimeError(
                            f"Invalid continuum in rows {start}:{stop}, granule {granule}"
                        )
                    values /= continuum[:, None].astype(np.float32)
                    good = (
                        in_range
                        & ~bad[band_index, fp, left]
                        & ~bad[band_index, fp, right]
                    )
                    cache_key = np.packbits(good, bitorder="little").tobytes()
                    cached = projection_cache[band].get(cache_key)
                    if cached is None:
                        components = component_map[band]
                        design = components[:, good].T
                        if design.shape[0] <= models[band].n_components or np.linalg.matrix_rank(
                            design
                        ) < models[band].n_components:
                            raise RuntimeError(
                                f"Rank-deficient mask for granule {granule}, band {band}"
                            )
                        pseudo = np.linalg.pinv(design)
                        projection_cache[band][cache_key] = (good, pseudo)
                        if len(projection_cache[band]) > 512:
                            projection_cache[band].popitem(last=False)
                    else:
                        good, pseudo = cached
                        projection_cache[band].move_to_end(cache_key)
                    scores = values[:, good].astype(np.float64) @ pseudo.T
                    payload[f"pc_{band}"][positions] = scores.astype(np.float32)
                    payload[f"valid_channels_{band}"][positions] = int(good.sum())
            for band in BANDS:
                if not np.all(np.isfinite(payload[f"pc_{band}"])):
                    raise RuntimeError(f"Non-finite full projection for {band}, rows {start}:{stop}")
            append_rows(target, payload)
            written += len(rows)
            if written % 1_000_000 < len(rows):
                target.flush()
                print(f"{output_path.name}: projected {written:,} rows", flush=True)
        target.attrs["status"] = "complete"
        target.attrs["eligible_rows"] = written
        target.attrs["projection"] = "resampled continuum-normalized spectra; masked least-squares PC scores"
    os.replace(partial, output_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annual-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--years", type=int, nargs="+", default=list(range(2015, 2026)))
    parser.add_argument("--protocol-years", type=int, nargs="+", default=[2015, 2016, 2017, 2018, 2019])
    parser.add_argument("--rows-per-domain", type=int, default=100_000)
    parser.add_argument("--svd-rows-per-domain", type=int, default=50_000)
    parser.add_argument("--components", type=int, default=10)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    annual = {year: args.annual_root / f"OCO2_L1B_L2_{year}.h5" for year in args.years}
    samples = {}
    for year, path in annual.items():
        sample_path = args.output / f"sample_{year}.npz"
        if sample_path.exists():
            loaded = np.load(sample_path)
            samples[year] = {name: loaded[name] for name in loaded.files}
        else:
            samples[year] = sample_year(
                path,
                args.rows_per_domain,
                args.svd_rows_per_domain,
                SEED + year,
            )
            np.savez_compressed(sample_path, **samples[year])
        counts = {
            domain.decode(): int(np.count_nonzero(samples[year]["domain"] == domain))
            for domain in DOMAINS
        }
        svd_counts = {
            domain.decode(): int(np.count_nonzero(
                (samples[year]["domain"] == domain) & samples[year]["svd_fit"]
            ))
            for domain in DOMAINS
        }
        print(
            f"{year}: modelling sample {counts}; nested SVD sample {svd_counts}",
            flush=True,
        )

    protocol_annual = {year: args.annual_root / f"OCO2_L1B_L2_{year}.h5" for year in args.protocol_years}
    protocol = derive_protocol(protocol_annual, args.output)
    preview_years = [year for year in args.years if year in args.protocol_years]
    previews = [preview_resampled(annual[year], samples[year], protocol) for year in preview_years]
    choose_continuum(previews, protocol)
    with np.load(args.output / "protocol.npz") as stored:
        arrays = {name: stored[name] for name in stored.files}
    for band in BANDS:
        arrays[f"{band}_continuum_indices"] = protocol[band]["continuum"]
    np.savez_compressed(args.output / "protocol.npz", **arrays)

    resampled = []
    for year in args.years:
        path = args.output / f"resampled_{year}.h5"
        resample_year(annual[year], samples[year], protocol, path)
        resampled.append(path)
    training_resampled = [
        path for year, path in zip(args.years, resampled) if year in args.protocol_years
    ]
    models = fit_svd(training_resampled, protocol, args.output, args.components)
    for year, path in zip(args.years, resampled):
        masked_project(path, args.output / f"projected_{year}.h5", models)

    complete = {
        "status": "complete",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "years": args.years,
        "protocol_years": args.protocol_years,
        "domains": [domain.decode() for domain in DOMAINS],
        "requested_rows_per_domain_per_year": args.rows_per_domain,
        "sampling_rule": (
            "equal domain counts per year; min(requested, annual availability across domains)"
        ),
        "svd_rows_per_domain_per_training_year": args.svd_rows_per_domain,
        "total_svd_rows": (
            len(args.protocol_years) * len(DOMAINS) * args.svd_rows_per_domain
        ),
        "components_per_band": args.components,
        "quality_filter": "inherited from the pre-screened annual H5 files",
        "domain_classification": (
            "ND by mode; GL_OCEAN by finite non-negative wind_speed; "
            "GL_LAND by wind_speed retrieval fill -999999"
        ),
        "projection_scope": "balanced three-domain modelling sample",
    }
    (args.output / "complete.json").write_text(json.dumps(complete, indent=2), encoding="utf-8")
    print(json.dumps(complete, indent=2), flush=True)


if __name__ == "__main__":
    main()
