#!/usr/bin/env python3
"""Perturb only the XCO2 prior in immutable 99-feature residual models."""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

import numpy as np

SOURCE = Path('/path/to/OCO2/output/ML/Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test')
BASE = Path('/path/to/OCO2/output/ML/Paper_ThreeDomain_Global_Prior99_1519Train_2020Val_2021_2025Test_20261018')
OUT = Path('/path/to/OCO2/output/ML/Paper_PriorSensitivity_20261029')
DOMAINS = ('GL_OCEAN', 'GL_LAND', 'ND')
SEEDS = (17, 29, 43)
YEARS = (2020, 2024)
SCENARIOS = ([dict(name=f'offset_{value:+g}', kind='offset', value=value) for value in (-2., -1., -.5, 0., .5, 1., 2.)]
             + [dict(name=f'noise_{value:g}', kind='noise', value=value) for value in (.5, 1., 2.)])
PATHS = ('coupled', 'feature_only', 'reconstruction_only')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)


def load(path):
    with np.load(path, allow_pickle=False) as a:
        return {k: a[k] for k in a.files}


def immutable(path, value):
    value = json.loads(json.dumps(value))
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Immutable record changed: {path}'
    else:
        write_json(path, value)


def model_folder(domain, family):
    return BASE / ('mlp_search/studies' if family == 'MLP' else 'tree_models') / domain / 'delta'


def model_file(domain, family, seed):
    folder = model_folder(domain, family)
    return folder / (f'selected_models/seed_{seed}/model.pt' if family == 'MLP' else f'LightGBM_{seed}/model.txt')


def original_predictions(domain, family, seed, year):
    p = model_file(domain, family, seed).parent / f'predictions_{year}.npz'
    return p


def prepare_cohort(output, domain, year):
    import h5py
    target = output / 'cohorts' / f'{domain}_{year}.npz'
    target.parent.mkdir(exist_ok=True)
    if target.exists():
        return
    with h5py.File(SOURCE / f'projected_{year}.h5', 'r') as h:
        assert h.attrs['status'] == 'complete'
        keep = np.flatnonzero(h['domain'][:] == domain.encode())
        def read(name):
            return h[name][:][keep]
        assert len(keep) == 200000
        assert np.all(read('mode') == (b'ND' if domain == 'ND' else b'GL'))
        spectra = [read(f'pc_{band}').astype('f4') for band in ('O2', 'W', 'S')]
        assert all(a.shape == (200000, 30) for a in spectra)
        names = ['solar_zenith', 'sensor_zenith', 'AZA', 'retrieval_solar_distance',
                 'retrieval_solar_relative_velocity', 'surface_pressure_apriori_fph', 'surface_pressure_apriori_abp']
        fields = {name: read(name).astype('f4').reshape(-1) for name in names}
        for values in fields.values():
            values[(~np.isfinite(values)) | (values <= -9999)] = np.nan
        def secant(values):
            a = values.astype('f8')
            result = np.full(a.shape, np.nan)
            good = np.isfinite(a) & (a >= 0) & (a < 90)
            result[good] = 1 / np.cos(np.deg2rad(a[good]))
            return result.astype('f4')
        extras = np.column_stack([secant(fields['solar_zenith']), secant(fields['sensor_zenith']),
            np.sin(np.deg2rad(fields['AZA'])), np.cos(np.deg2rad(fields['AZA'])),
            fields['retrieval_solar_distance'], fields['retrieval_solar_relative_velocity'],
            fields['surface_pressure_apriori_fph'], fields['surface_pressure_apriori_abp']])
        extras[(~np.isfinite(extras)) | (extras < -9999)] = np.nan
        for column, name in [(6, 'surface_pressure_apriori_fph'), (7, 'surface_pressure_apriori_abp')]:
            extras[fields[name] <= 0, column] = np.nan
        prior = read('xco2_apriori').astype('f4').reshape(-1)
        y = read('xco2').astype('f4').reshape(-1)
        x = np.column_stack([*spectra, extras, prior]).astype('f4')
        ids, indices = read('sounding_id'), read('rows')
    assert x.shape == (200000, 99) and np.isfinite(x[:, :90]).all()
    assert np.isfinite(prior).all() and np.isfinite(y).all() and len(np.unique(ids)) == len(ids)
    assert np.all(ids // 10**12 == year)
    noise_seed = 20261029 + year * 10 + DOMAINS.index(domain)
    z = np.random.default_rng(noise_seed).standard_normal(len(y))
    z = (z - z.mean()) / z.std()
    temporary = target.with_suffix('.partial.npz')
    np.savez_compressed(temporary, raw_x=x, reference_xco2=y, xco2_apriori=prior,
                        sounding_id=ids, original_index=indices, noise_unit=z)
    os.replace(temporary, target)
    write_json(target.with_suffix('.json'), dict(domain=domain, year=year, n=len(y), noise_seed=noise_seed,
        noise_mean=float(z.mean()), noise_sd=float(z.std()), sha256=sha(target),
        source_size=(SOURCE / f'projected_{year}.h5').stat().st_size,
        source_mtime_ns=(SOURCE / f'projected_{year}.h5').stat().st_mtime_ns))


def score(y, pred):
    y, pred = y.astype('f8'), pred.astype('f8')
    error = pred - y
    assert np.isfinite(error).all()
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(error**2))), bias=float(error.mean()),
                r2=float(1 - np.sum(error**2) / np.sum((y - y.mean())**2)))


def error_for(scenario, data):
    if scenario['kind'] == 'offset':
        return np.full(len(data['reference_xco2']), scenario['value'], dtype='f8')
    return data['noise_unit'] * scenario['value']


def paths_for(base, change, error):
    return dict(coupled=base + change + error, feature_only=base + change, reconstruction_only=base + error)


def run_worker(args):
    domain, family, output = args.domain, args.family, args.output
    if family == 'MLP':
        import torch
        from torch import nn
        torch.set_num_threads(4)
        torch.set_num_interop_threads(1)
        class Model(nn.Module):
            def __init__(self, hp):
                super().__init__()
                activation = {'relu': nn.ReLU, 'gelu': nn.GELU, 'silu': nn.SiLU}[hp['activation']]
                widths = list(map(int, hp['architecture'].split('-')))
                layers, width = [], 99
                for i, next_width in enumerate(widths):
                    layers.append(nn.Linear(width, next_width))
                    if hp['normalization'] == 'layernorm':
                        layers.append(nn.LayerNorm(next_width))
                    layers.append(activation())
                    if hp['dropout'] and i < len(widths) - 1:
                        layers.append(nn.Dropout(hp['dropout']))
                    width = next_width
                layers.append(nn.Linear(width, 1))
                self.network = nn.Sequential(*layers)
            def forward(self, x):
                return self.network(x).squeeze(-1)
    else:
        import lightgbm as lgb
    folder = output / domain / family
    folder.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for seed in SEEDS:
        model_path = model_file(domain, family, seed)
        scaler_path = model_folder(domain, family) / 'training_only_scaler.npz'
        scaler = load(scaler_path)
        assert scaler['mean'].shape == (99,)
        same = load(model_folder(domain, 'MLP') / 'training_only_scaler.npz')
        assert all(np.array_equal(scaler[k], same[k]) for k in ['median', 'mean', 'scale'])
        model_hash, scaler_hash = sha(model_path), sha(scaler_path)
        if family == 'MLP':
            checkpoint = torch.load(model_path, map_location='cpu', weights_only=False)
            model = Model(checkpoint['hp']); model.load_state_dict(checkpoint['state_dict']); model.eval()
            @torch.no_grad()
            def predict(x):
                result = np.concatenate([model(torch.as_tensor(x[i:i+16384])).numpy() for i in range(0, len(x), 16384)])
                return result * checkpoint['target_std'] + checkpoint['target_mean']
        else:
            model = lgb.Booster(model_file=str(model_path))
            def predict(x):
                return model.predict(x, num_threads=8)
        run = folder / f'seed_{seed}'
        run.mkdir(exist_ok=True)
        for year in YEARS:
            done = run / f'complete_{year}.json'
            if done.exists():
                old = json.loads(done.read_text())
                assert old['model_sha256'] == model_hash and old['scaler_sha256'] == scaler_hash
                all_rows.extend(old['metrics']); continue
            started = time.perf_counter()
            data = load(output / 'cohorts' / f'{domain}_{year}.npz')
            x = np.where(np.isfinite(data['raw_x']), data['raw_x'], scaler['median']).astype('f4')
            x -= scaler['mean']; x /= scaler['scale']
            assert np.isfinite(x).all()
            native = predict(x)
            baseline = (native + data['xco2_apriori']).astype('f8')
            source = original_predictions(domain, family, seed, year)
            if source.exists():
                archived = load(source)
                for key in ['sounding_id', 'original_index', 'reference_xco2', 'xco2_apriori']:
                    assert np.array_equal(data[key], archived[key])
                assert np.max(np.abs(baseline - archived['predicted_xco2'])) < 1e-4
                identity_check = 'original ordered prediction identities and values'
            else:
                with (model_folder(domain, family) / 'metrics_runs.csv').open() as handle:
                    prior_metrics = [r for r in csv.DictReader(handle) if str(r['year']) == str(year)
                                     and r['family'] == family and int(r['seed']) == seed]
                assert len(prior_metrics) == 1
                assert abs(score(data['reference_xco2'], baseline)['rmse'] - float(prior_metrics[0]['rmse'])) < 1e-4
                identity_check = 'original source cohort and validation metric'
            changes, rows = [], []
            for scenario in SCENARIOS:
                error = error_for(scenario, data)
                if scenario['kind'] == 'offset' and scenario['value'] == 0:
                    change = np.zeros(len(native), dtype='f4')
                else:
                    altered = x.copy()
                    altered[:, -1] = ((data['xco2_apriori'].astype('f8') + error - scaler['mean'][-1]) / scaler['scale'][-1]).astype('f4')
                    assert np.array_equal(altered[:, :-1], x[:, :-1])
                    change = (predict(altered).astype('f8') - native.astype('f8')).astype('f4')
                changes.append(change)
                results = paths_for(baseline, change.astype('f8'), error)
                assert np.allclose(results['coupled'] - results['feature_only'], error, atol=1e-10, rtol=0)
                assert np.allclose(results['reconstruction_only'] - baseline, error, atol=1e-10, rtol=0)
                for path, values in results.items():
                    shift = values - baseline
                    record = dict(domain=domain, family=family, seed=seed, year=year,
                        scenario=scenario['name'], kind=scenario['kind'], magnitude=scenario['value'], path=path,
                        **score(data['reference_xco2'], values), shift_mean=float(shift.mean()), shift_sd=float(shift.std()),
                        prior_error_mean=float(error.mean()), prior_error_sd=float(error.std()),
                        perturbed_prior_rmse=score(data['reference_xco2'], data['xco2_apriori'].astype('f8') + error)['rmse'])
                    rows.append(record)
                print(f'{domain}/{family}/{seed}/{year} {scenario["name"]} coupled RMSE={rows[-3]["rmse"]:.5f}', flush=True)
            np.savez_compressed(run / f'predictions_{year}.npz', baseline_xco2=baseline,
                residual_delta=np.stack(changes), sounding_id=data['sounding_id'], original_index=data['original_index'])
            write_json(done, dict(status='complete', metrics=rows, model_sha256=model_hash, scaler_sha256=scaler_hash,
                source_identity_check=identity_check, seconds=time.perf_counter()-started))
            assert sha(model_path) == model_hash and sha(scaler_path) == scaler_hash
            all_rows.extend(rows)
    write_json(folder / 'complete.json', dict(status='complete', metrics=all_rows))


def main(args):
    if args.self_test:
        a = np.array([400., 402.]); d = np.array([.2, -.3]); e = np.array([1., -1.])
        p = paths_for(a, d, e)
        assert np.allclose(p['coupled'] - p['feature_only'], e)
        assert np.allclose(p['reconstruction_only'] - a, e)
        assert len(SCENARIOS) == 10
        print('self-test passed'); return
    if args.domain:
        run_worker(args); return
    args.output.mkdir(parents=True, exist_ok=True)
    lock = (args.output / 'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (args.output / 'complete.json').exists():
        print('Already complete; no duplicate inference'); return
    assert (args.output / 'PROTOCOL.md').exists()
    immutable(args.output / 'design.json', dict(domains=DOMAINS, seeds=SEEDS, years=YEARS,
        scenarios=SCENARIOS, paths=PATHS, protocol_sha256=sha(args.output / 'PROTOCOL.md'),
        training_performed=False, xco2_prior_feature=98, features=99))
    inputs = {str(p): sha(p) for domain in DOMAINS for family in ['MLP', 'LightGBM']
              for p in [model_folder(domain, family) / 'training_only_scaler.npz',
                        *[model_file(domain, family, seed) for seed in SEEDS]]}
    immutable(args.output / 'input_manifest.json', inputs)
    for domain in DOMAINS:
        for year in YEARS:
            prepare_cohort(args.output, domain, year)
    def worker(job):
        domain, family = job
        env = {**os.environ, 'OMP_NUM_THREADS': '8' if family == 'LightGBM' else '4',
               'OPENBLAS_NUM_THREADS': '4', 'MKL_NUM_THREADS': '4', 'CUDA_VISIBLE_DEVICES': ''}
        with (args.output / f'{domain}_{family}.log').open('a') as log:
            subprocess.run([sys.executable, '-u', __file__, '--domain', domain, '--family', family,
                '--output', str(args.output)], env=env, stdin=subprocess.DEVNULL, stdout=log,
                stderr=subprocess.STDOUT, check=True)
    with ThreadPoolExecutor(max_workers=6) as pool:
        list(pool.map(worker, [(d, f) for d in DOMAINS for f in ['MLP', 'LightGBM']]))
    rows = [r for d in DOMAINS for f in ['MLP', 'LightGBM']
            for r in json.loads((args.output / d / f / 'complete.json').read_text())['metrics']]
    assert len(rows) == 1080 and len(list(args.output.glob('*/*/seed_*/predictions_*.npz'))) == 36
    assert all(sha(Path(path)) == value for path, value in inputs.items())
    with (args.output / 'metrics.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(args.output / 'complete.json', dict(status='complete', models=18, predictions=36, metric_rows=1080,
        source_models_and_scalers_unchanged=True, training_performed=False))
    print('Complete: 18 frozen models, 36 prediction files, 1080 metric rows', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=OUT)
    parser.add_argument('--domain', choices=DOMAINS)
    parser.add_argument('--family', choices=['MLP', 'LightGBM'])
    parser.add_argument('--self-test', action='store_true')
    main(parser.parse_args())
