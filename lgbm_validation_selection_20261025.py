"""Validation-selected LightGBM controls with unchanged 99-input cohorts."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import fcntl
import hashlib
import itertools
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace

ENGINE = Path('/path/to/OCO2/scripts/ML/paper_future_generalization/prior99_global_and_east_asia_20261018')
sys.path.insert(0, str(ENGINE))
import h5py
import lightgbm as lgb
import numpy as np
import search_mlp_secant as engine
import train_domain_svd50 as features
import two_route_training_reference as reference

ML = Path('/path/to/OCO2/output/ML')
SOURCE = ML / 'Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test'
GLOBAL = ML / 'Paper_ThreeDomain_Global_Prior99_1519Train_2020Val_2021_2025Test_20261018'
OUT = ML / 'Paper_LightGBM_ValidationSelection_20261025'
COHORTS = ('GL_OCEAN', 'GL_LAND', 'ND')
SEEDS = (17, 29, 43)
BASE = dict(learning_rate=.03, num_leaves=31, min_data_in_leaf=100,
            feature_fraction=.9, bagging_fraction=.8, lambda_l2=1.)


def utc():
    return datetime.now(timezone.utc).isoformat()


def digest(a):
    a = np.ascontiguousarray(a)
    return hashlib.sha256(str((a.shape, a.dtype.str)).encode() + a.tobytes()).hexdigest()


def immutable(path, value):
    value = json.loads(json.dumps(value))
    if path.exists():
        assert json.loads(path.read_text()) == value, f'Protocol changed: {path}'
    else:
        engine.write_json(path, value)


def candidates():
    space = dict(learning_rate=[.01, .03, .07], num_leaves=[15, 31, 63, 127],
                 min_data_in_leaf=[20, 100, 300], feature_fraction=[.7, .9, 1.],
                 bagging_fraction=[.8, 1.], lambda_l2=[0., 1., 10.])
    grid = [dict(zip(space, values)) for values in itertools.product(*space.values())]
    grid = [hp for hp in grid if hp != BASE]
    order = np.random.default_rng(20261025).permutation(len(grid))[:23]
    return [BASE.copy()] + [grid[i] for i in order], space


def original(cohort):
    return cohort, GLOBAL, GLOBAL / 'tree_models' / cohort / 'delta'


def global_raw(year, domain):
    x, y, prior = features.load_year(SOURCE, year, domain, include_xco2_prior=True)
    with h5py.File(SOURCE / f'projected_{year}.h5', 'r') as f:
        ix = np.flatnonzero(f['domain'][:] == domain.encode())
        ids, rows = f['sounding_id'][:][ix], f['rows'][:][ix]
        assert np.all(f['mode'][:][ix] == (b'ND' if domain == 'ND' else b'GL'))
    return dict(x=x, y=y, prior=prior, sounding_id=ids, original_index=rows)


def prepare(cohort):
    domain, old_root, old = original(cohort)
    folder = OUT / 'studies' / cohort
    folder.mkdir(parents=True, exist_ok=True)
    parts = [global_raw(year, domain) for year in range(2015, 2020)]
    train = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    val = global_raw(2020, domain)
    assert len(train['y']) == 1000000 and len(val['y']) == 200000
    del parts
    names = json.loads((old / 'feature_names.json').read_text())
    assert names == features.FEATURE_NAMES + ['xco2_apriori'] and len(names) == 99
    with np.load(old / 'training_only_scaler.npz', allow_pickle=False) as f:
        scaler = {k: f[k].copy() for k in ('median', 'mean', 'scale')}
    signature = {'cohort': cohort, 'feature_names': names,
                 'scaler': {k: digest(a) for k, a in scaler.items()},
                 'train': {k: digest(a) for k, a in train.items()},
                 'validation': {k: digest(a) for k, a in val.items()},
                 'train_n': len(train['y']), 'validation_n': len(val['y'])}
    immutable(folder / 'cohort_identity.json', signature)
    if not (folder / 'training_only_scaler.npz').exists():
        shutil.copy2(old / 'training_only_scaler.npz', folder / 'training_only_scaler.npz')
    immutable(folder / 'feature_names.json', names)
    for data in (train, val):
        assert data['x'].shape == (len(data['y']), 99)
        assert np.array_equal(data['x'][:, -1], data['prior'])
        engine.standardize(data['x'], **scaler)
    return train, val, scaler


def metrics(y, pred):
    error = np.asarray(pred, dtype='f8') - np.asarray(y, dtype='f8')
    variance = np.sum((y.astype('f8') - np.mean(y, dtype='f8')) ** 2)
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(np.abs(error))),
                bias=float(np.mean(error)), r2=float(1 - np.sum(error ** 2) / variance))


def fit(cohort, index, hp, seed, train, val, datasets):
    folder = OUT / 'studies' / cohort / 'candidates' / f'{index:02d}' / f'seed_{seed}'
    folder.mkdir(parents=True, exist_ok=True)
    done = folder / 'trained.json'
    if done.exists():
        result = json.loads(done.read_text())
        assert result['hp'] == hp and result['seed'] == seed
        assert result['model_sha256'] == hashlib.sha256((folder / 'model.txt').read_bytes()).hexdigest()
        return result
    started = time.perf_counter()
    if index == 0:
        _, _, old = original(cohort)
        saved = old / f'LightGBM_{seed}'
        info = json.loads((saved / 'trained.json').read_text())
        assert info['feature_count'] == 99 and not info['smoke']
        shutil.copy2(saved / 'model.txt', folder / 'model.txt')
        model = lgb.Booster(model_file=str(folder / 'model.txt'))
        training_seconds = info['train_seconds']
        provenance = str(saved)
        rounds = model.current_iteration()
    else:
        params = dict(objective='regression_l2', metric='rmse', verbosity=-1,
                      num_threads=8, seed=seed, bagging_seed=seed, feature_fraction_seed=seed,
                      data_random_seed=17, bagging_freq=1, feature_pre_filter=False, **hp)
        history = {}
        model = lgb.train(params, datasets[0], num_boost_round=3000,
                          valid_sets=[datasets[1]], valid_names=['validation'],
                          callbacks=[lgb.early_stopping(100, verbose=False), lgb.record_evaluation(history)])
        model.save_model(str(folder / 'model.txt'), num_iteration=model.best_iteration)
        rounds = len(history['validation']['rmse'])
        engine.write_json(folder / 'history.json', history)
        training_seconds = time.perf_counter() - started
        provenance = 'new fit on unchanged full historical training cohort'
    pred = model.predict(val['x'], num_threads=8) + val['prior']
    scores = metrics(val['y'], pred)
    info = dict(candidate=index, hp=hp, seed=seed, validation_rmse=scores['rmse'],
                validation_metrics=scores, best_iteration=int(model.best_iteration if model.best_iteration > 0 else model.current_iteration()),
                stopped_iteration=rounds, hit_3000=rounds == 3000,
                train_seconds=training_seconds, model=str(folder / 'model.txt'),
                model_sha256=hashlib.sha256((folder / 'model.txt').read_bytes()).hexdigest(),
                provenance=provenance, n_train=len(train['y']), n_validation=len(val['y']), finished_utc=utc())
    engine.write_json(done, info)
    print(f'{cohort} candidate={index:02d} seed={seed} val={scores["rmse"]:.6f} '
          f'rounds={rounds} seconds={time.perf_counter()-started:.1f}', flush=True)
    return info


def search(cohort):
    folder = OUT / 'studies' / cohort
    folder.mkdir(parents=True, exist_ok=True)
    with (folder / 'worker.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (folder / 'selection_frozen.json').exists():
            return
        train, val, scaler = prepare(cohort)
        options, _ = candidates()
        datasets = [lgb.Dataset(train['x'], label=train['y'] - train['prior'], free_raw_data=False,
                               params={'feature_pre_filter': False, 'data_random_seed': 17})]
        datasets.append(lgb.Dataset(val['x'], label=val['y'] - val['prior'], reference=datasets[0], free_raw_data=False))
        records = [fit(cohort, i, hp, 17, train, val, datasets) for i, hp in enumerate(options)]
        top = [r['candidate'] for r in sorted(records, key=lambda r: (r['validation_rmse'], r['candidate']))[:3]]
        confirmed = []
        for i in sorted(set(top + [0])):
            runs = [fit(cohort, i, options[i], seed, train, val, datasets) for seed in SEEDS]
            confirmed.append(dict(candidate=i, hp=options[i], runs=runs,
                                  mean_validation_rmse=float(np.mean([r['validation_rmse'] for r in runs]))))
        selected = min(confirmed, key=lambda r: (r['mean_validation_rmse'], r['candidate']))
        engine.write_json(folder / 'selection_frozen.json', dict(
            status='complete', smoke=False, cohort=cohort, feature_count=99, k=[30, 30, 30],
            candidates=24, seeds=list(SEEDS), screening_seed=17, confirmation_candidates=confirmed,
            selected=selected, frozen_utc=utc(), future_test_metrics_used_for_selection=False,
            selected_on='2020 validation',
            test_status='Retrospective; future years have been inspected in earlier project analyses'))


def evaluate(cohort):
    assert all((OUT / 'studies' / c / 'selection_frozen.json').exists() for c in COHORTS)
    folder = OUT / 'studies' / cohort
    if (folder / 'complete.json').exists():
        return
    train, val, scaler = prepare(cohort)
    domain, root, old = original(cohort)
    selected = json.loads((folder / 'selection_frozen.json').read_text())['selected']
    splits = [('2015-2019', 'train', train),
              ('2020', 'validation', val)]
    models = {r['seed']: lgb.Booster(model_file=r['model']) for r in selected['runs']}
    rows, audits, timing = [], [], []

    def score(label, role, data):
        for seed, model in models.items():
            native = model.predict(data['x'], num_threads=8)
            pred = native + data['prior']
            result = dict(cohort=cohort, domain=domain, route='delta', family='LightGBM', seed=seed,
                          year=label, role=role, n=len(pred), **metrics(data['y'], pred))
            rows.append(result)
            if role != 'train':
                target = folder / 'selected_models' / f'seed_{seed}'
                target.mkdir(parents=True, exist_ok=True)
                np.savez_compressed(target / f'predictions_{label}.npz',
                                    sounding_id=data['sounding_id'], original_index=data['original_index'],
                                    reference_xco2=data['y'], xco2_apriori=data['prior'],
                                    predicted_target=native, predicted_xco2=pred)
                if role == 'test':
                    original_file = old / f'LightGBM_{seed}' / f'predictions_{label}.npz'
                    with np.load(original_file, allow_pickle=False) as previous:
                        for name, key in [('sounding_id', 'sounding_id'), ('original_index', 'original_index'),
                                          ('reference_xco2', 'y'), ('xco2_apriori', 'prior')]:
                            assert np.array_equal(previous[name], data[key]), (cohort, seed, label, name)
                    audits.append(dict(cohort=cohort, seed=seed, year=label, n=len(pred),
                                       original_id_reference_prior_exact=True))
    for label, role, data in splits:
        score(label, role, data)
    del train, val, splits
    for year in range(2021, 2026):
        data = global_raw(year, domain)
        engine.standardize(data['x'], **scaler)
        score(str(year), 'test', data)
    if cohort in COHORTS:
        # Rebenchmark the selected comparator; old fixed-model timings are not relabeled.
        raw = global_raw(2020, domain)
        for seed, model in models.items():
            model.predict((raw['x'][:1024] - scaler['mean']) / scaler['scale'], num_threads=4)
            durations = []
            for _ in range(5):
                started = time.perf_counter()
                x = raw['x'].copy()
                engine.standardize(x, **scaler)
                pred = model.predict(x, num_threads=4) + raw['prior']
                durations.append(time.perf_counter() - started)
            singles = []
            for i in range(100):
                started = time.perf_counter()
                x = raw['x'][i:i+1].copy()
                engine.standardize(x, **scaler)
                model.predict(x, num_threads=4) + raw['prior'][i:i+1]
                singles.append(time.perf_counter() - started)
            timing.append(dict(cohort=cohort, seed=seed, n=200000, threads=4,
                               batch_seconds_median=float(np.median(durations)),
                               single_milliseconds_median=float(np.median(singles)*1000),
                               includes='standardization + feature prediction + prior addition',
                               excludes='file I/O and spectral preprocessing',
                               timing_environment='sequential benchmark after all training searches finished'))
    engine.write_csv(folder / 'metrics_runs.csv', rows)
    engine.write_json(folder / 'identity_verification.json', audits)
    engine.write_json(folder / 'timing.json', timing)
    engine.write_json(folder / 'complete.json', dict(status='complete', cohort=cohort,
        models=3, metric_rows=len(rows), predictions=len(list(folder.glob('selected_models/*/predictions_*.npz'))),
        finished_utc=utc()))


def verify():
    rows, checks, selections = [], [], []
    for cohort in COHORTS:
        folder = OUT / 'studies' / cohort
        frozen = json.loads((folder / 'selection_frozen.json').read_text())
        assert frozen['feature_count'] == 99 and frozen['k'] == [30, 30, 30]
        assert not frozen['future_test_metrics_used_for_selection']
        baseline = next(c for c in frozen['confirmation_candidates'] if c['candidate'] == 0)
        assert frozen['selected']['mean_validation_rmse'] <= baseline['mean_validation_rmse'] + 1e-12
        study = list(__import__('csv').DictReader((folder / 'metrics_runs.csv').open()))
        assert len(study) == 21
        assert {int(r['seed']) for r in study} == set(SEEDS)
        for r in study:
            assert np.isfinite([float(r[k]) for k in ('rmse', 'mae', 'bias', 'r2')]).all()
        for path in sorted(folder.glob('selected_models/*/predictions_*.npz')):
            seed = int(path.parent.name.split('_')[-1])
            label = path.stem[len('predictions_'):]
            assert path.stat().st_mtime >= (folder / 'selection_frozen.json').stat().st_mtime
            with np.load(path, allow_pickle=False) as saved:
                assert len(np.unique(saved['sounding_id'])) == len(saved['sounding_id'])
                assert np.array_equal(saved['predicted_xco2'], saved['predicted_target'] + saved['xco2_apriori'])
                recalculated = metrics(saved['reference_xco2'], saved['predicted_xco2'])
                row = next(r for r in study if int(r['seed']) == seed and r['year'] == label)
                for k, value in recalculated.items():
                    assert abs(value - float(row[k])) < 1e-10
            checks.append(dict(cohort=cohort, seed=seed, split=label, independent_metrics_match=True))
        selection = frozen['selected']
        selections.append(dict(cohort=cohort, candidate=selection['candidate'], hp=selection['hp'],
                               mean_validation_rmse=selection['mean_validation_rmse'],
                               baseline_mean_validation_rmse=baseline['mean_validation_rmse']))
        rows.extend(study)
    assert len(rows) == 63 and len(checks) == 54
    engine.write_csv(OUT / 'metrics_runs.csv', rows)
    summary = []
    for cohort, label in sorted({(r['cohort'], r['year']) for r in rows}):
        block = [r for r in rows if r['cohort'] == cohort and r['year'] == label]
        assert len(block) == 3 and len({r['n'] for r in block}) == 1
        record = dict(cohort=cohort, domain=block[0]['domain'], route='delta', family='LightGBM',
                      year=label, role=block[0]['role'], n=int(block[0]['n']), runs=3)
        for key in ('rmse', 'mae', 'bias', 'r2'):
            values = [float(r[key]) for r in block]
            record[key+'_mean'], record[key+'_std'] = float(np.mean(values)), float(np.std(values))
        summary.append(record)
    engine.write_csv(OUT / 'metrics_summary.csv', summary)
    engine.write_json(OUT / 'selected_configurations.json', selections)
    engine.write_json(OUT / 'verification.complete.json', dict(status='verified',
        cohort_count=3, selected_models=9, metric_rows=63, prediction_files=54,
        checks=checks, cohort_scaler_identities_recorded=True,
        source_projection='Unchanged original rank-30 masked least-squares coefficients',
        selected_before_new_future_evaluation=True, test_status='Retrospective, not newly blinded'))
    engine.write_json(OUT / 'complete.json', dict(status='complete', verified=True, finished_utc=utc(),
        cohorts=list(COHORTS), models=9, metric_rows=63, prediction_files=54))


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / 'pipeline.lock').open('w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (OUT / 'complete.json').exists():
            return
        options, space = candidates()
        immutable(OUT / 'protocol.json', dict(feature_count=99, k=[30, 30, 30], route='delta',
            cohorts=list(COHORTS), source=str(SOURCE), candidate_count=24, candidates=options,
            candidate_generation='Baseline plus 23 unique grid draws; fixed seed 20261025', search_space=space,
            training='Full original 1m/domain; unchanged scalers',
            global_validation='Full 200000 rows in 2020',
            screen_seed=17, confirmation='Best three candidates plus original baseline; seeds 17,29,43',
            criterion='Mean validation RMSE across three seeds', maximum_boosting_rounds=3000,
            early_stopping_patience=100, training_threads=8,
            all_selections_frozen_before_future_evaluation=True,
            model_comparison='Both families validation-selected; search budgets are not equal or exhaustive',
            unchanged='SVD, projections, cohorts, MLP checkpoints, target, retrospective tests and partial2025',
            script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest()))
        engine.write_json(OUT / 'pipeline.json', dict(pid=os.getpid(), started_utc=utc()))
        def work(cohort):
            log = OUT / f'{cohort}.log'
            with log.open('a') as handle:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--search', cohort],
                               stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, check=True)
        with ThreadPoolExecutor(max_workers=4) as pool:
            list(pool.map(work, COHORTS))
        engine.write_json(OUT / 'all_selections_frozen.json', dict(cohorts=list(COHORTS), frozen_utc=utc()))
        for cohort in COHORTS:
            with (OUT / f'{cohort}.log').open('a') as handle:
                subprocess.run([sys.executable, '-u', str(Path(__file__).resolve()), '--evaluate', cohort],
                               stdin=subprocess.DEVNULL, stdout=handle, stderr=subprocess.STDOUT, check=True)
        verify()


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--search', choices=COHORTS)
    p.add_argument('--evaluate', choices=COHORTS)
    p.add_argument('--verify', action='store_true')
    a = p.parse_args()
    try:
        if a.search:
            search(a.search)
        elif a.evaluate:
            evaluate(a.evaluate)
        elif a.verify:
            verify()
        else:
            run()
    except Exception as exc:
        OUT.mkdir(parents=True, exist_ok=True)
        engine.write_json(OUT / f'error_{a.search or a.evaluate or "pipeline"}.json',
                          dict(error=repr(exc), utc=utc()))
        raise
