"""Screen original ND products and compare frozen global vs refit regional MLPs."""
import argparse
import concurrent.futures
import csv
import fcntl
import hashlib
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import joblib
import numpy as np


ROOT = Path('/path/to/OCO2')
ML = ROOT / 'output/ML'
SOURCE = ML / 'Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test'
GLOBAL = ML / 'Paper_MainMLP_Epoch500_20261024/studies/ND/delta'
ENGINE = ROOT / 'scripts/ML/paper_future_generalization/prior99_global_and_east_asia_20261018'
BANDS = ('O2', 'W', 'S')
RADIANCE = ('radiance_o2', 'radiance_weak_co2', 'radiance_strong_co2')
SEEDS = (17, 29, 43)


def atomic_npz(path, **arrays):
    partial = path.with_suffix('.partial.npz')
    np.savez_compressed(partial, **arrays)
    os.replace(partial, path)


def atomic_json(path, obj):
    tmp = path.with_suffix('.partial.json')
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False))
    os.replace(tmp, path)


def init_worker():
    global basis, protocol
    basis = joblib.load(SOURCE / 'svd/domain_svd_models.joblib')['ND']
    with np.load(SOURCE / 'protocol.npz') as f:
        protocol = [(f[b + '_target_wavelength_um'], f[b + '_continuum_indices'])
                    for b in BANDS]


def vec(f, key):
    return f[key][:].reshape(-1)


def project(f, frame, foot):
    ds = f['InstrumentHeader']
    coeff = ds['dispersion_coef_samp'][:]
    bad = ds['bad_sample_list'][:] != 0
    scores = np.full((len(frame), 90), np.nan, dtype='f4')
    for bi, (band, name) in enumerate(zip(BANDS, RADIANCE)):
        radiance = f['SoundingMeasurements/' + name]
        frames, inverse = np.unique(frame, return_inverse=True)
        values = radiance[frames][inverse, foot].astype('f4')
        grid, ci = protocol[bi]
        for fp in np.unique(foot):
            positions = np.flatnonzero(foot == fp)
            native = np.polynomial.polynomial.polyval(
                np.arange(1, 1017, dtype='f8'), coeff[bi, fp])
            assert np.all(np.diff(native) > 0)
            right = np.clip(np.searchsorted(native, grid, side='right'), 1, 1015)
            left = right - 1
            w = ((grid - native[left]) / (native[right] - native[left])).astype('f4')
            good = ((grid >= native[0]) & (grid <= native[-1])
                    & ~bad[bi, fp, left] & ~bad[bi, fp, right])
            use = ci[good[ci]]
            if len(use) < 3 or good.sum() <= 50:
                continue
            a = values[positions][:, left] * (1 - w) + values[positions][:, right] * w
            cont = a[:, use].mean(axis=1, dtype='f8')
            keep = np.isfinite(a).all(1) & np.isfinite(cont) & (cont > 0)
            design = basis[band].components_[:, good].T.astype('f8')
            if np.linalg.matrix_rank(design) != 30:
                continue
            normalized = (a[keep][:, good] / cont[keep, None]).astype('f4')
            scores[positions[keep], bi*30:(bi+1)*30] = normalized @ np.linalg.pinv(design).T
    return scores


def read_granule(row, scan_missing):
    geo, ret = 'RetrievalGeometry/', 'RetrievalResults/'
    audit = {k: row[k] for k in ('date', 'orbit_segment', 'build', 'status')}
    audit.update(eligible=0, matched=0, usable=0)
    if row['status'] != 'paired_nonempty' and not scan_missing:
        return audit, None
    try:
        with h5py.File(row['l2_path'], 'r') as f:
            lat, lon = (vec(f, geo + 'retrieval_' + k) for k in ('latitude', 'longitude'))
            ids = vec(f, 'RetrievalHeader/sounding_id')
            m = ((lat > 20) & (lat < 45) & (lon > 110) & (lon < 145)
                 & (vec(f, ret + 'outcome_flag') == 1)
                 & (vec(f, geo + 'retrieval_land_fraction') == 100)
                 & np.isin(ids % 10, np.arange(1, 9)))
            audit['eligible'] = int(m.sum())
            if not m.any() or row['status'] != 'paired_nonempty':
                return audit, None
            sid = ids[m]
            y = (vec(f, ret + 'xco2')[m] * 1e6).astype('f4')
            prior = (vec(f, ret + 'xco2_apriori')[m] * 1e6).astype('f4')
            names = [geo + 'retrieval_' + k for k in (
                'solar_zenith', 'zenith', 'azimuth', 'solar_distance', 'solar_relative_velocity')]
            names += [ret + 'surface_pressure_apriori_fph',
                      'PreprocessingResults/surface_pressure_apriori_abp']
            fields = [vec(f, k)[m].astype('f4') for k in names]
    except (OSError, KeyError) as e:
        audit['error'] = 'L2: ' + str(e)
        return audit, None
    try:
        with h5py.File(row['l1b_path'], 'r') as f:
            raw_ids = f['SoundingGeometry/sounding_id'][:]
            common, ai, bi = np.intersect1d(raw_ids.ravel(), sid, return_indices=True)
            audit['matched'] = len(common)
            if not len(common):
                return audit, None
            frame, foot = np.divmod(ai, raw_ids.shape[1])
            scores = project(f, frame, foot)
    except (OSError, KeyError) as e:
        audit['error'] = 'L1B: ' + str(e)
        return audit, None
    for a in fields:
        a[(~np.isfinite(a)) | (a <= -9999)] = np.nan
    sza, vza, aza, distance, velocity, fph, abp = fields
    def secant(a):
        return np.where((a >= 0) & (a < 90), 1 / np.cos(np.deg2rad(a.astype('f8'))), np.nan)
    fph[fph <= 0] = np.nan
    abp[abp <= 0] = np.nan
    x = np.column_stack((scores, secant(sza[bi]), secant(vza[bi]),
                         np.sin(np.deg2rad(aza[bi])), np.cos(np.deg2rad(aza[bi])),
                         distance[bi], velocity[bi], fph[bi], abp[bi], prior[bi])).astype('f4')
    keep = (np.isfinite(scores).all(1) & np.isfinite(y[bi]) & (y[bi] > 0)
            & np.isfinite(prior[bi]) & (prior[bi] > 0))
    audit['usable'] = int(keep.sum())
    return audit, (common[keep], x[keep], y[bi][keep], prior[bi][keep])


def batch(task):
    path, rows = task
    if path.exists():
        return str(path)
    notes, usable = [], []
    for row in rows:
        audit, arrays = read_granule(row, scan_missing=True)
        notes.append(audit)
        if arrays is not None and len(arrays[0]):
            usable.append(arrays)
    values = [np.concatenate(a) for a in zip(*usable)] if usable else [
        np.empty(0, 'i8'), np.empty((0, 99), 'f4'), np.empty(0, 'f4'), np.empty(0, 'f4')]
    atomic_npz(path, ids=values[0], x=values[1], y=values[2], prior=values[3],
               audit=json.dumps(notes))
    return str(path)


def collect(rows, out, workers, smoke):
    tasks = []
    for year in range(2017, 2026):
        annual = [r for r in rows if int(r['year']) == year]
        if smoke:
            annual = [r for r in annual if r['status'] == 'paired_nonempty'][:100]
        for i in range(0, len(annual), 32):
            tasks.append((out / f'{year}_{i:04d}.npz', annual[i:i+32]))
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, initializer=init_worker) as pool:
        for i, _ in enumerate(pool.map(batch, tasks, chunksize=1), 1):
            if i % 10 == 0 or i == len(tasks):
                print(f'Extracted {i}/{len(tasks)} batches', flush=True)
    annual, audit = {}, []
    for path, _ in tasks:
        with np.load(path) as f:
            audit += json.loads(str(f['audit']))
            annual.setdefault(int(path.name[:4]), []).append(tuple(f[k] for k in ('ids', 'x', 'y', 'prior')))
    result, counts = {}, {}
    for year, parts in annual.items():
        ids, x, y, prior = (np.concatenate(a) for a in zip(*parts))
        # Batch order follows manifest order; choose one observation per sounding ID.
        _, keep = np.unique(ids, return_index=True)
        result[year] = (ids[keep], x[keep], y[keep], prior[keep])
        yy = [r for r in audit if r['date'].startswith(str(year))]
        counts[year] = dict(eligible=sum(r['eligible'] for r in yy), matched=sum(r['matched'] for r in yy),
                            usable=len(keep), duplicate_rows=len(ids)-len(keep),
                            read_errors=sum('error' in r for r in yy))
        assert len(keep), f'No usable rows: {year}'
    atomic_json(out / 'cohort_audit.json', dict(years=counts, files=audit))
    print('Cohort counts', counts, flush=True)
    return result


def compare(data, out, device, smoke):
    """Evaluate the frozen global model; no regional fitting."""
    sys.path.insert(0, str(ENGINE))
    import torch
    import search_mlp_secant as engine
    torch.set_num_threads(4)
    splits = {str(y): data[y] for y in range(2020, 2026)}
    atomic_npz(out/'split_ids.npz', **{y: values[0] for y, values in splits.items()})
    with np.load(GLOBAL/'training_only_scaler.npz') as f:
        scaler = tuple(f[k] for k in ('median', 'mean', 'scale'))
    with h5py.File(SOURCE/'projected_2020.h5', 'r') as f:
        validation_ids = f['sounding_id'][:]
    rows = []
    for seed in SEEDS:
        path = GLOBAL/f'selected_models/seed_{seed}/model.pt'
        cp = torch.load(path, map_location='cpu', weights_only=False)
        model = engine.DenseMLP(99, cp['hp']).to(device)
        model.load_state_dict(cp['state_dict'])
        model.eval()
        for year, (ids, features, ref, prior) in splits.items():
            x = torch.from_numpy(engine.standardize(features.copy(), *scaler))
            with torch.inference_mode():
                delta = np.concatenate([model(z.to(device)).cpu().numpy()
                                        for z in x.split(8192)]).reshape(-1)
            pred = delta.astype('f8') * cp['target_std'] + cp['target_mean'] + prior
            assert np.isfinite(pred).all() and pred.shape == ref.shape
            atomic_npz(out/f'global_frozen_{seed}_{year}.npz',
                       ids=ids, ref=ref, prior=prior, pred=pred)
            keep = ~np.isin(ids, validation_ids) if year == '2020' else np.ones(len(ids), dtype=bool)
            error = pred[keep] - ref[keep]
            rows.append(dict(seed=seed, year=year, n=int(keep.sum()),
                             rmse=float(np.sqrt(np.mean(error**2)))))
    with (out/'metrics.csv').open('w') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    atomic_json(out/'complete.json', dict(smoke=smoke, metrics=len(rows)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('--workers', type=int, default=8)
    p.add_argument('--smoke', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    lock = (args.out/'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    contract = dict(manifest=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
                    script=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(), smoke=args.smoke)
    spec = args.out/'contract.json'
    if spec.exists():
        assert json.loads(spec.read_text()) == contract
    else:
        atomic_json(spec, contract)
    if (args.out/'complete.json').exists():
        return
    with args.manifest.open() as f:
        rows = [r for r in csv.DictReader(f) if r['l2_path'] and r['year'] >= '2017']
    data = collect(rows, args.out, args.workers, args.smoke)
    import torch
    compare(data, args.out, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), args.smoke)


if __name__ == '__main__':
    main()
