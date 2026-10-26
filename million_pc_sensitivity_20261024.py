"""Million-row PC sensitivity with frozen architectures and validation-only ND LR search."""
import argparse
import concurrent.futures
import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import section35_epoch500_20261024 as base

engine, features, reference = base.engine, base.features, base.reference
OUT = base.ML/'Paper_Million_PC_Sensitivity_20261024'
MAIN500 = base.ML/'Paper_MainMLP_Epoch500_20261024'
SCALER_SOURCE = base.ML/'Paper_ThreeDomain_Edge30_SVD30_Models_SecantGeometry_1519Train_2020Val_2021_2025Test'
KS, SEEDS, RATES = (0,10,20,30), (17,29,43), (1e-4,3e-4,1e-3)
GPUS = (0,1,3,5,6,7)


def hp_for(domain):
    path=base.MAIN/f'mlp_search/studies/{domain}/delta/selection_frozen.json'
    return json.loads(path.read_text())['selected']['hp']


def folder(domain,k,lr=None):
    if lr is None or lr == hp_for(domain)['lr']:
        return OUT/'fixed'/domain/f'k{k}'
    assert domain=='ND'
    return OUT/'nd_lr'/f'k{k}'/f'lr{lr:.8g}'


def columns_for(k):
    assert k in KS
    return np.array([b*30+j for b in range(3) for j in range(k)]+list(range(90,99)))


def prepare(domain,k,dest):
    dest.mkdir(parents=True,exist_ok=True)
    args=SimpleNamespace(source=base.SOURCE,baseline_root=SCALER_SOURCE,domain=domain,
        include_xco2_prior=True,route='delta',smoke=False,regional_spec=None)
    raw=engine.prepare(args,dest)
    columns=columns_for(k)
    np.savez_compressed(dest/'scaler.npz',columns=columns,median=raw.median[columns],
                        mean=raw.mean[columns],scale=raw.scale[columns])
    return engine.Prepared(raw.x[:,columns].copy(),raw.y,raw.prior,
        raw.val_x[:,columns].copy(),raw.val_y,raw.val_prior,
        raw.median[columns],raw.mean[columns],raw.scale[columns])


def run_info(domain,k,lr,seed):
    hp={**hp_for(domain),'lr':lr}
    if domain!='GL_LAND' and k==30 and lr==hp_for(domain)['lr']:
        frozen=json.loads((MAIN500/f'studies/{domain}/delta/selection_frozen.json').read_text())
        assert frozen['selected']['hp']==hp
        info=next(r for r in frozen['selected']['runs'] if r['seed']==seed)
        assert info['status']=='complete' and info['epochs']<=500
        return info
    path=folder(domain,k,lr)/'candidates'/engine.hp_id(hp)/f'seed_{seed}/complete.json'
    return json.loads(path.read_text())


def worker(domain,k,lr,seeds,smoke=False):
    dest=OUT/'smoke' if smoke else folder(domain,k,lr)
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if not smoke and domain!='GL_LAND' and k==30 and lr==hp_for(domain)['lr']:
            engine.write_json(dest/'reuse.json',dict(runs=[run_info(domain,k,lr,s) for s in seeds],
                reason='identical million-row samples, features, scaler, recipe and 500-epoch ceiling'))
            return
        data=prepare(domain,k,dest)
        hp={**hp_for(domain),'lr':lr}
        args=SimpleNamespace(route='delta',max_epochs=1 if smoke else 500,
            min_epochs=1 if smoke else 40,patience=1 if smoke else 30)
        engine.write_json(dest/'protocol.json',dict(domain=domain,k=[k]*3,hp=hp,
            train_n=len(data.y),validation_n=len(data.val_y),max_epochs=args.max_epochs,
            min_epochs=args.min_epochs,patience=args.patience,spectral_method='original masked LS',
            original_scaler_reused=True,smoke=smoke))
        trainer=engine.Trainer(args,data,dest,torch.device('cuda:0'))
        for seed in seeds:
            info=trainer.fit(hp,seed)
            assert info['status']=='complete'
        print('WORKER_OK',domain,k,lr,seeds,flush=True)


def dispatch(jobs,stage):
    def lane(gpu,tasks):
        for domain,k,lr,seeds in tasks:
            logname=f'{stage}_{domain}_k{k}_lr{lr:.8g}.log'
            with (OUT/logname).open('a') as log:
                subprocess.run([sys.executable,'-u',str(Path(__file__).resolve()),
                    '--phase','evaluate' if stage=='evaluate' else 'train','--domain',domain,
                    '--k',str(k),'--lr',str(lr),'--seeds',*map(str,seeds)],
                    stdout=log,stderr=subprocess.STDOUT,check=True,
                    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4',
                         'OPENBLAS_NUM_THREADS':'4'})
    with concurrent.futures.ThreadPoolExecutor(max_workers=len(GPUS)) as pool:
        fs=[pool.submit(lane,gpu,jobs[i::len(GPUS)]) for i,gpu in enumerate(GPUS)]
        for f in fs:
            f.result()


def identity(domain,year,y,prior):
    with h5py.File(base.SOURCE/f'projected_{year}.h5') as f:
        ix=np.flatnonzero(f['domain'][:]==domain.encode())
        ids,original=f['sounding_id'][:][ix],f['rows'][:][ix]
        np.testing.assert_array_equal(y,f['xco2'][:][ix].astype('f4').reshape(-1))
        np.testing.assert_array_equal(prior,f['xco2_apriori'][:][ix].astype('f4').reshape(-1))
        assert len(ids)==len(np.unique(ids))==len(y)
        assert np.all(f['mode'][:][ix]==(b'ND' if domain=='ND' else b'GL'))
    digest=hashlib.sha256()
    for a in (ids,original,y,prior):
        digest.update(np.ascontiguousarray(a).tobytes())
    return ids,original,digest.hexdigest()


def evaluate(domain,k):
    assert (OUT/'selection_frozen.json').exists()
    dest=OUT/'evaluation'/domain/f'k{k}'
    dest.mkdir(parents=True,exist_ok=True)
    with (dest/'worker.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX)
        if (dest/'complete.json').exists():
            return
        data=prepare(domain,k,dest)
        frozen=json.loads((OUT/'selection_frozen.json').read_text())
        recipes=[('fixed',hp_for(domain)['lr'])]
        if domain=='ND':
            recipes.append(('lr_selected',frozen['nd_rates'][str(k)]['lr']))
        rows=[]
        for recipe,lr in recipes:
            for seed in SEEDS:
                result=dest/recipe/f'seed_{seed}'
                result.mkdir(parents=True,exist_ok=True)
                if (result/'complete.json').exists():
                    rows.extend(json.loads((result/'complete.json').read_text())['metrics'])
                    continue
                info=run_info(domain,k,lr,seed)
                cp=torch.load(Path(info['run'])/'best.pt',map_location='cuda:0',weights_only=False)
                expected={**hp_for(domain),'lr':lr}
                assert cp['hp']==expected and cp['seed']==seed
                model=engine.DenseMLP(3*k+9,expected).cuda()
                model.load_state_dict(cp['state_dict'])
                own=[]
                for year in (0,2020,2021,2022,2023,2024,2025):
                    if year==0:
                        xx,yy,pp=data.x,data.y,data.prior
                    elif year==2020:
                        xx,yy,pp=data.val_x,data.val_y,data.val_prior
                    else:
                        xx,yy,pp=features.load_year(base.SOURCE,year,domain,include_xco2_prior=True)
                        xx=xx[:,columns_for(k)].copy()
                        engine.standardize(xx,data.median,data.mean,data.scale)
                    pred=reference.torch_predict(model,xx,torch.device('cuda:0'),16384)
                    pred=pred*cp['target_std']+cp['target_mean']+pp
                    metric=reference.metrics(yy,pred)
                    assert np.isfinite(list(metric.values())).all()
                    if year:
                        ids,original,digest=identity(domain,year,yy,pp)
                        np.savez_compressed(result/f'predictions_{year}.npz',
                            sounding_id=ids,original_index=original,reference_xco2=yy,
                            xco2_apriori=pp,predicted_xco2=pred)
                    else:
                        digest='pooled2015-2019'
                    own.append(dict(domain=domain,k=k,recipe=recipe,lr=lr,seed=seed,year=year,
                        role='train' if year==0 else 'validation' if year==2020 else 'test',
                        n=len(yy),identity_sha256=digest,stopped_epoch=info['epochs'],
                        best_epoch=info['best_epoch'],hit_500=info['epochs']==500,
                        train_seconds=info['train_seconds'],**metric))
                assert abs(own[1]['rmse']-info['best_validation_rmse'])<2e-5
                engine.write_json(result/'complete.json',dict(metrics=own))
                rows.extend(own)
                del model
        engine.write_csv(dest/'metrics.csv',rows)
        engine.write_json(dest/'complete.json',dict(status='complete',rows=len(rows)))


def pipeline():
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'pipeline.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'complete.json').exists():
            return
        protocol=dict(domains=list(base.DOMAINS),ks=list(KS),seeds=list(SEEDS),
            train_years=list(range(2015,2020)),train_n_per_domain=1000000,
            validation_year=2020,validation_n_per_domain=200000,test_years=list(range(2021,2026)),
            partial_2025=True,max_epochs=500,min_epochs=40,patience=30,
            fixed_recipes={d:hp_for(d) for d in base.DOMAINS},
            nd_lr_candidates=[hp_for('ND')['lr'],*RATES],
            selection='2020 full validation only; ND rates seed17 screen, top2 confirmed with3 seeds',
            features='frozen main masked-LS30 features, bandwise equal prefixes, nine ancillary inputs',
            reuse='GL_OCEAN/ND fixed30 only; GL_LAND retrained with shared cosine horizon500 for all k',
            cosine_horizon='GL_LAND500 instead of original200; other hp unchanged; do not equate new GL_LAND30 with original checkpoint',
            expected_final_metric_rows=336,expected_final_prediction_files=288,
            tests='retrospective, previously inspected; no blind-test claim',
            source=str(base.SOURCE),script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
        p=OUT/'protocol.json'
        if p.exists():
            assert json.loads(p.read_text())==protocol
        else:
            engine.write_json(p,protocol)
        dispatch([(d,k,hp_for(d)['lr'],list(SEEDS)) for k in KS for d in base.DOMAINS],'fixed')
        dispatch([('ND',k,lr,[17]) for k in KS for lr in RATES],'screen')
        screening=[]
        shortlist={}
        for k in KS:
            rates=[hp_for('ND')['lr'],*RATES]
            rank=sorted(rates,key=lambda lr:(run_info('ND',k,lr,17)['best_validation_rmse'],lr))
            shortlist[str(k)]=rank[:2]
            screening.extend(dict(k=k,lr=lr,validation_rmse=run_info('ND',k,lr,17)['best_validation_rmse']) for lr in rates)
        engine.write_csv(OUT/'screening.csv',screening)
        engine.write_json(OUT/'shortlist.json',shortlist)
        dispatch([('ND',int(k),lr,[29,43]) for k,rates in shortlist.items() for lr in rates
                  if lr!=hp_for('ND')['lr']],'confirm')
        confirmed=[]
        nd_rates={}
        fixed_prefix={}
        for d in base.DOMAINS:
            scores=[dict(k=k,mean=float(np.mean([run_info(d,k,hp_for(d)['lr'],s)['best_validation_rmse']
                                                for s in SEEDS]))) for k in KS]
            fixed_prefix[d]=min(scores,key=lambda r:(r['mean'],r['k']))
        for k in KS:
            scores=[]
            for lr in shortlist[str(k)]:
                val=[run_info('ND',k,lr,s)['best_validation_rmse'] for s in SEEDS]
                scores.append(dict(k=k,lr=lr,mean=float(np.mean(val)),std=float(np.std(val,ddof=1))))
            confirmed.extend(scores)
            nd_rates[str(k)]=min(scores,key=lambda r:(r['mean'],r['lr']))
        engine.write_csv(OUT/'confirmed_validation.csv',confirmed)
        selected=dict(fixed_prefix_selection=fixed_prefix,nd_rates=nd_rates,test_used_for_selection=False)
        frozen=OUT/'selection_frozen.json'
        if frozen.exists():
            assert json.loads(frozen.read_text())==selected
        else:
            engine.write_json(frozen,selected)
        dispatch([(d,k,hp_for(d)['lr'],list(SEEDS)) for d in base.DOMAINS for k in KS],'evaluate')
        import csv
        rows=[]
        for d in base.DOMAINS:
            for k in KS:
                with (OUT/f'evaluation/{d}/k{k}/metrics.csv').open() as f:
                    rows.extend(csv.DictReader(f))
        assert len(rows)==336
        assert len({(r['domain'],r['k'],r['recipe'],r['seed'],r['year']) for r in rows})==336
        identities={}
        for row in rows:
            assert np.isfinite([float(row[n]) for n in ('rmse','bias','r2')]).all()
            key=row['domain'],row['year']
            value=row['n'],row['identity_sha256']
            if key in identities:
                assert identities[key]==value
            identities[key]=value
        assert len(list((OUT/'evaluation').glob('*/k*/*/seed_*/predictions_*.npz')))==288
        engine.write_csv(OUT/'metrics.csv',rows)
        engine.write_json(OUT/'verification.json',dict(metric_rows=336,prediction_files=288,
            identical_yearly_cohorts=True,all_metrics_finite=True,
            note='Independent checkpoint/prediction audit still required before final reporting'))
        engine.write_json(OUT/'complete.json',dict(status='complete',fixed_models=36,
            nd_rate_selected_model_records=12,metric_rows=336,prediction_files=288))


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--phase',choices=('pipeline','train','evaluate','smoke'),default='pipeline')
    p.add_argument('--domain',choices=base.DOMAINS,default='ND')
    p.add_argument('--k',type=int,choices=KS,default=30)
    p.add_argument('--lr',type=float)
    p.add_argument('--seeds',type=int,nargs='+',default=[17])
    args=p.parse_args()
    torch.set_num_threads(4)
    if args.phase=='pipeline':
        pipeline()
    elif args.phase=='evaluate':
        evaluate(args.domain,args.k)
    else:
        worker(args.domain,args.k,args.lr or hp_for(args.domain)['lr'],args.seeds,args.phase=='smoke')
