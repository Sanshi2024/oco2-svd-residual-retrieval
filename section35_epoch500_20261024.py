"""Extend the two experiments actually reported in v9 section 3.5 to 500 epochs."""
import argparse
import concurrent.futures
import fcntl
import json
import os
import shutil
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import torch

ENGINE = Path('/path/to/OCO2/scripts/ML/paper_future_generalization/prior99_global_and_east_asia_20261018')
sys.path.insert(0, str(ENGINE))
import search_mlp_secant as engine
import train_domain_svd50 as features
import two_route_training_reference as reference

ML = Path('/path/to/OCO2/output/ML')
SOURCE = ML / 'Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test'
MAIN = ML / 'Paper_ThreeDomain_Global_Prior99_1519Train_2020Val_2021_2025Test_20261018'
OUT = ML / 'Paper_Section35_Epoch500_20261024'
PREVIOUS_ND = ML / 'Paper_ND_BandwisePC_20261023'
PREVIOUS_SHARED = ML / 'Paper_ReviewerRevision_20261021/ablation'
DOMAINS = ('GL_OCEAN', 'GL_LAND', 'ND')
DOMAIN = 'ND'
KIND = 'shared'
SEEDS = (17, 29, 43)
CONFIGS = [(30, 30, 30)] + [tuple(k if j == b else 30 for j in range(3))
                           for b in range(3) for k in (0, 10, 20)]


def sample(year, count):
    x, y, prior = features.load_year(SOURCE, year, DOMAIN, include_xco2_prior=True)
    ix = np.sort(np.random.default_rng(20261021 + year).choice(len(y), count, replace=False))
    return x[ix], y[ix], prior[ix], ix


def worker(ks):
    folder = OUT / KIND / DOMAIN / ('k' + '_'.join(map(str, ks)))
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX)
    if (folder / 'complete.json').exists():
        return
    selection = json.loads((MAIN / f'mlp_search/studies/{DOMAIN}/delta/selection_frozen.json').read_text())
    hp = selection['selected']['hp']
    columns = np.array([30*b + j for b in range(3) for j in range(ks[b])] + list(range(90, 99)))
    parts = [sample(year, 20000) for year in range(2015, 2020)]
    x, y, prior = [np.concatenate([part[i] for part in parts]) for i in range(3)]
    vx, vy, vp, vi = sample(2020, 50000)
    np.savez_compressed(folder/'sample_indices.npz', **{str(2015+i):p[3] for i,p in enumerate(parts)}, val=vi)
    x, vx = x[:, columns].copy(), vx[:, columns].copy()
    median = np.nanmedian(x, axis=0).astype('f4')
    median[~np.isfinite(median)] = 0
    filled = np.where(np.isfinite(x), x, median)
    mean = filled.mean(axis=0, dtype='f8').astype('f4')
    scale = filled.std(axis=0, dtype='f8').astype('f4')
    scale[scale < 1e-8] = 1
    engine.standardize(x, median, mean, scale)
    engine.standardize(vx, median, mean, scale)
    data = engine.Prepared(x,y,prior,vx,vy,vp,median,mean,scale)
    np.savez_compressed(folder/'scaler.npz', median=median,mean=mean,scale=scale,columns=columns)
    protocol = dict(ks=list(ks), source=str(SOURCE), seeds=list(SEEDS), train_n=100000, val_n=50000,
                    domain=DOMAIN,kind=KIND,hp=hp, max_epochs=500, min_epochs=40, patience=30,
                    coefficients='prefixes of the original 30-component masked-LS coefficients; no refit',
                    selection='2020 validation only; retrospective future tests; no architecture search')
    engine.write_json(folder/'protocol.json',protocol)
    args = SimpleNamespace(route='delta',max_epochs=500,min_epochs=40,patience=30)
    trainer = engine.Trainer(args,data,folder,torch.device('cuda:0'))
    runs, audit = [], []
    for seed in SEEDS:
        previous_folder = PREVIOUS_SHARED/DOMAIN/f'k{ks[0]}' if KIND=='shared' else PREVIOUS_ND/folder.name
        previous = previous_folder / 'candidates' / engine.hp_id(hp) / f'seed_{seed}'
        original = None
        if (previous/'complete.json').exists():
            original = json.loads((previous/'complete.json').read_text())
            assert original['hp'] == hp
            run = folder/'candidates'/engine.hp_id(hp)/f'seed_{seed}'
            if not run.exists():
                shutil.copytree(previous,run,
                    ignore=shutil.ignore_patterns('complete.json') if original['epochs']==200 else None)
        info = trainer.fit(hp,seed)
        runs.append(info)
        audit.append(dict(domain=DOMAIN,kind=KIND,ks=list(ks),seed=seed,max_epochs=500,
            old_epochs=original['epochs'] if original else None,
            stopped_epoch=info['epochs'],best_epoch=info['best_epoch'],
            best_validation_rmse=info['best_validation_rmse'],hit_maximum=info['epochs']==500,
            reused_early_stop=bool(original and original['epochs']<200)))
        engine.write_json(folder/'epoch_audit.json',audit)
    rows=[]
    for info in runs:
        cp=torch.load(Path(info['run'])/'best.pt',map_location='cuda:0',weights_only=False)
        model=engine.DenseMLP(len(columns),hp).cuda()
        model.load_state_dict(cp['state_dict'])
        for year in range(2020,2026):
            xx,yy,pp=features.load_year(SOURCE,year,DOMAIN,include_xco2_prior=True)
            xx=xx[:,columns].copy()
            engine.standardize(xx,median,mean,scale)
            pred=reference.torch_predict(model,xx,torch.device('cuda:0'),16384)*cp['target_std']+cp['target_mean']+pp
            assert np.isfinite(pred).all()
            rows.append(dict(domain=DOMAIN,kind=KIND,k_O2=ks[0],k_W=ks[1],k_S=ks[2],seed=info['seed'],year=year,n=len(yy),
                             best_epoch=info['best_epoch'],**reference.metrics(yy,pred)))
        del model
    engine.write_csv(folder/'metrics.csv',rows)
    engine.write_json(folder/'complete.json',dict(status='complete',models=3,rows=len(rows)))


def pipeline():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'complete.json').exists():return
    engine.write_json(OUT/'protocol.json',dict(configurations=CONFIGS,training_years=list(range(2015,2020)),
                      validation_year=2020,test_years=list(range(2021,2026)),partial_2025=True,
                      domains=list(DOMAINS),max_epochs=500,min_epochs=40,patience=30,
                      train_n=100000,early_stop_n=50000,evaluation_n=200000,
                      design='v9 section3.5: shared 0/10/20/30 per band in three domains, plus ND-only one-band-at-a-time',
                      learning_rate='Preserve original scheduler trajectory and restored optimizer/RNG states',
                      previous_nd=str(PREVIOUS_ND),test_selection=False))
    def lane(gpu, jobs):
        for kind,domain,ks in jobs:
            name='k'+'_'.join(map(str,ks))
            with (OUT/f'{kind}_{domain}_{name}.log').open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),'--kind',kind,'--domain',domain,'--ks',*map(str,ks)],
                               stdout=log,stderr=subprocess.STDOUT,check=True,
                               env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4','OPENBLAS_NUM_THREADS':'4'})
    tasks=[('shared',domain,(k,k,k)) for k in (0,10,20,30) for domain in DOMAINS]
    tasks += [('bandwise','ND',ks) for ks in CONFIGS]
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as pool:
        jobs=[pool.submit(lane,gpu,tasks[i::3]) for i,gpu in enumerate((0,1,7))]
        for job in jobs:job.result()
    import csv
    rows=[]
    reference_indices={}
    audits=[]
    for kind,domain,ks in tasks:
        folder=OUT/kind/domain/('k'+'_'.join(map(str,ks)))
        with np.load(folder/'sample_indices.npz') as f:
            arrays={k:f[k].copy() for k in f.files}
        if domain not in reference_indices:reference_indices[domain]=arrays
        for k in arrays:np.testing.assert_array_equal(arrays[k],reference_indices[domain][k])
        old_folder=PREVIOUS_SHARED/domain/f'k{ks[0]}' if kind=='shared' else PREVIOUS_ND/folder.name
        if old_folder.exists():
            with np.load(old_folder/'sample_indices.npz') as previous:
                for k in arrays:np.testing.assert_array_equal(arrays[k],previous[k])
            with np.load(old_folder/'scaler.npz') as previous, np.load(folder/'scaler.npz') as current:
                for k in current.files:np.testing.assert_array_equal(current[k],previous[k])
        audits.extend(json.loads((folder/'epoch_audit.json').read_text()))
        with (folder/'metrics.csv').open() as f:part=list(csv.DictReader(f))
        assert len(part)==18
        rows.extend(part)
    engine.write_csv(OUT/'metrics.csv',rows)
    engine.write_json(OUT/'epoch_audit.json',audits)
    import pandas as pd
    df=pd.DataFrame(rows)
    for col in ('year','rmse','seed','k_O2','k_W','k_S'):df[col]=pd.to_numeric(df[col])
    assert np.isfinite(df['rmse']).all()
    df[df.year==2020].groupby(['kind','domain','k_O2','k_W','k_S']).rmse.agg(['mean','std']).reset_index().to_csv(OUT/'validation_summary.csv',index=False)
    engine.write_json(OUT/'complete.json',dict(status='complete',configurations=22,models=66,metric_rows=396,
                      hit_maximum=[r for r in audits if r['hit_maximum']],identical_subsets_verified=True))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--ks',nargs=3,type=int)
    p.add_argument('--domain',choices=DOMAINS,default='ND')
    p.add_argument('--kind',choices=('shared','bandwise'),default='shared')
    args=p.parse_args();torch.set_num_threads(4)
    DOMAIN=args.domain
    KIND=args.kind
    if args.ks:worker(tuple(args.ks))
    else:pipeline()
