"""Extend cap-limited manuscript runs without changing their optimization path."""
import argparse
import csv
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

ML = Path('/workspace_hdd/xianlei/OCO2/output/ML')
MAIN = ML / 'Paper_ThreeDomain_Global_Prior99_1519Train_2020Val_2021_2025Test_20261018/mlp_search'
SOURCE = ML / 'Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test'
BASELINE = ML / 'Paper_ThreeDomain_Edge30_SVD30_Models_SecantGeometry_1519Train_2020Val_2021_2025Test'
OUT = ML / 'Paper_MainMLP_Epoch500_20261024'
ENGINE = Path('/workspace_hdd/xianlei/OCO2/scripts/ML/paper_future_generalization/prior99_global_and_east_asia_20261018')
DOMAINS = ('GL_OCEAN', 'GL_LAND', 'ND')


def save(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.partial')
    tmp.write_text(json.dumps(value, indent=2))
    os.replace(tmp, path)


def write_csv(path, rows):
    with path.open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def capped(run, cap):
    return run['epochs'] == cap


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--self-test', action='store_true')
    args = parser.parse_args()
    if args.self_test:
        assert capped({'epochs': 200}, 200)
        assert not capped({'epochs': 174}, 200)
        assert not capped({'epochs': 200}, 500)
        print('Cap classification tests passed')
        return
    OUT.mkdir(parents=True, exist_ok=True)
    lock = (OUT / 'pipeline.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (OUT / 'complete.json').exists():
        print('Already complete', flush=True)
        return
    sys.path.insert(0, str(ENGINE))
    import search_mlp_secant as engine
    import torch
    import pandas as pd
    torch.set_num_threads(4)
    torch.set_float32_matmul_precision('highest')
    device = torch.device('cuda:0')
    assert torch.cuda.is_available()
    frozen = {d: json.loads((MAIN / f'studies/{d}/delta/selection_frozen.json').read_text())
              for d in DOMAINS}
    protocol = {
        'source': str(SOURCE), 'original_models': str(MAIN), 'max_epochs': 500,
        'min_epochs': 40, 'patience': 30, 'minimum_improvement_ppm': 1e-5,
        'domains': list(DOMAINS), 'seeds': [17, 29, 43], 'route': 'delta',
        'features': 99, 'components_per_band': [30, 30, 30],
        'training_years': list(range(2015, 2020)), 'validation_year': 2020,
        'reuse': 'Previously early-stopped runs unchanged; extend only runs hitting 200.',
        'scheduler': 'Original learning-rate path preserved, including original scheduler horizon.',
        'selection': 'Frozen architecture; checkpoint selection uses 2020 validation only.',
        'tests': '2021-2025 retrospective; partial 2025; not used to select epoch.',
        'script_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'engine_sha256': hashlib.sha256(Path(engine.__file__).read_bytes()).hexdigest(),
        'original_selection_sha256': {d: hashlib.sha256((MAIN / f'studies/{d}/delta/selection_frozen.json').read_bytes()).hexdigest() for d in DOMAINS},
    }
    if (OUT / 'protocol.json').exists():
        assert json.loads((OUT / 'protocol.json').read_text()) == protocol
    save(OUT / 'protocol.json', protocol)
    audit, selections = [], []
    for domain in DOMAINS:
        folder = OUT / f'studies/{domain}/delta'
        folder.mkdir(parents=True, exist_ok=True)
        selected = frozen[domain]['selected']
        hp = selected['hp']
        runs = []
        trainer = None
        data = None
        settings = SimpleNamespace(source=SOURCE, baseline_root=BASELINE, output=OUT,
            domain=domain, route='delta', include_xco2_prior=True, regional_spec=None,
            smoke=False, max_epochs=500, min_epochs=40, patience=30)
        for old in selected['runs']:
            original = Path(old['run'])
            if capped(old, 200):
                # The cap-limited run uses a constant LR; continuation is exact.
                assert hp['scheduler'] == 'constant'
                run = folder / 'candidates' / engine.hp_id(hp) / f"seed_{old['seed']}"
                if not run.exists():
                    shutil.copytree(original, run, ignore=shutil.ignore_patterns('complete.json'))
                state = torch.load(run / 'resume.pt', map_location='cpu', weights_only=False)
                assert state['epoch'] >= 200 and state['hp'] == hp
                if trainer is None:
                    data = engine.prepare(settings, folder)
                    trainer = engine.Trainer(settings, data, folder, device)
                info = trainer.fit(hp, old['seed'])
                mode = 'continued from original epoch-200 optimizer and RNG state'
                history = json.loads((run / 'history.json').read_text())
            else:
                assert old['epochs'] - old['best_epoch'] >= 30
                info = dict(old)
                mode = 'reused original converged early-stop run'
                history = json.loads((original / 'history.json').read_text())
            runs.append(info)
            tail = history[-20:]
            audit.append(dict(domain=domain, seed=old['seed'], old_epochs=old['epochs'],
                old_best_epoch=old['best_epoch'], old_validation_rmse=old['best_validation_rmse'],
                max_epochs=500, stopped_epoch=info['epochs'], best_epoch=info['best_epoch'],
                validation_rmse=info['best_validation_rmse'], hit_maximum=capped(info, 500),
                stop_reason='maximum_epoch_limit' if capped(info, 500) else 'validation_patience',
                last20_validation_first=tail[0]['validation_rmse'],
                last20_validation_last=tail[-1]['validation_rmse'], execution=mode))
            write_csv(OUT / 'epoch_audit.csv', audit)
            print(json.dumps(audit[-1]), flush=True)
        updated = {**frozen[domain], 'selected': {**selected, 'runs': runs},
                   'epoch_budget': 500, 'architecture_refitted': False}
        save(folder / 'selection_frozen.json', updated)
        selections.append(updated)
        if trainer is not None:
            del trainer, data
            torch.cuda.empty_cache()
    save(OUT / 'search_selection.complete.json', {'status': 'complete', 'selected': selections,
        'selection_year': 2020, 'architecture_search_repeated': False})
    for domain in DOMAINS:
        folder = OUT / f'studies/{domain}/delta'
        settings = SimpleNamespace(source=SOURCE, baseline_root=BASELINE, output=OUT,
            domain=domain, route='delta', include_xco2_prior=True, regional_spec=None, smoke=False)
        if not (folder / 'complete.json').exists():
            data = engine.prepare(settings, folder)
            engine.evaluate_selected(settings, folder, data, device)
            del data
            torch.cuda.empty_cache()
    comparisons = []
    for domain in DOMAINS:
        new = pd.read_csv(OUT / f'studies/{domain}/delta/metrics_runs.csv')
        old = pd.read_csv(MAIN / f'studies/{domain}/delta/metrics_runs.csv')
        assert len(new) == len(old) == 21
        new['domain'] = domain
        new['budget'] = 500
        old['domain'] = domain
        old['budget'] = 200
        comparisons.extend([old, new])
    all_metrics = pd.concat(comparisons, ignore_index=True)
    all_metrics.to_csv(OUT / 'comparison_metrics_runs.csv', index=False)
    assert all_metrics.select_dtypes('number').notna().all().all()
    for domain, digest in protocol['original_selection_sha256'].items():
        assert hashlib.sha256((MAIN / f'studies/{domain}/delta/selection_frozen.json').read_bytes()).hexdigest() == digest
    save(OUT / 'complete.json', {'status': 'complete', 'models': 9,
        'continued_models': sum(r['old_epochs'] == 200 for r in audit),
        'hit_500': [r for r in audit if r['hit_maximum']],
        'annual_and_train_metric_rows_per_budget': 63, 'original_selections_unchanged': True})
    print('Epoch-500 control complete', flush=True)


if __name__ == '__main__':
    main()
