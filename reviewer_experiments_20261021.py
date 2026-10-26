"""Matched-budget score ablations and independent diagnostics of frozen main models."""
import argparse
import fcntl
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import h5py
import numpy as np
import torch

ENGINE = Path('/workspace_hdd/xianlei/OCO2/scripts/ML/paper_future_generalization/prior99_global_and_east_asia_20261018')
sys.path.insert(0, str(ENGINE))
import search_mlp_secant as engine
import train_domain_svd50 as features
import two_route_training_reference as reference

ROOT = Path('/workspace_hdd/xianlei/OCO2/output/ML')
SOURCE = ROOT / 'Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test'
MAIN = ROOT / 'Paper_ThreeDomain_Global_Prior99_1519Train_2020Val_2021_2025Test_20261018'
OUT = ROOT / 'Paper_ReviewerRevision_20261021'
DOMAINS = ('GL_OCEAN', 'GL_LAND', 'ND')
SEEDS = (17, 29, 43)


def sample(year, domain, count):
    x, y, p = features.load_year(SOURCE, year, domain, include_xco2_prior=True)
    ix = np.sort(np.random.default_rng(20261021 + year).choice(len(y), count, replace=False))
    return x[ix], y[ix], p[ix], ix


def ablation(domain, k):
    folder = OUT / 'ablation' / domain / f'k{k}'
    folder.mkdir(parents=True, exist_ok=True)
    lock = (folder / 'worker.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    if (folder / 'complete.json').exists():
        return
    selected = json.loads((MAIN / 'mlp_search/studies' / domain / 'delta/selection_frozen.json').read_text())
    hp = selected['selected']['hp']
    columns = np.array([b * 30 + j for b in range(3) for j in range(k)] + list(range(90, 99)))
    parts = [sample(y, domain, 20000) for y in range(2015, 2020)]
    x, y, prior = [np.concatenate([part[i] for part in parts]) for i in range(3)]
    val_x, val_y, val_prior, val_ix = sample(2020, domain, 50000)
    np.savez_compressed(folder / 'sample_indices.npz', **{str(2015+i):p[3] for i,p in enumerate(parts)}, val=val_ix)
    x, val_x = x[:, columns].copy(), val_x[:, columns].copy()
    median = np.nanmedian(x, axis=0).astype('f4')
    median[~np.isfinite(median)] = 0
    filled = np.where(np.isfinite(x), x, median)
    mean = filled.mean(axis=0, dtype='f8').astype('f4')
    scale = filled.std(axis=0, dtype='f8').astype('f4')
    scale[scale < 1e-8] = 1
    engine.standardize(x, median, mean, scale)
    engine.standardize(val_x, median, mean, scale)
    data = engine.Prepared(x,y,prior,val_x,val_y,val_prior,median,mean,scale)
    np.savez_compressed(folder/'scaler.npz',median=median,mean=mean,scale=scale,columns=columns)
    protocol = {'domain':domain,'k':k,'feature_count':len(columns),'train_n':len(y),'val_n':len(val_y),
        'train_years':list(range(2015,2020)), 'validation_year':2020,'seeds':list(SEEDS),
        'method':'Prefix feature ablation of frozen 30-component masked-LS scores, NOT refitted lower-rank projection',
        'hp':hp,'max_epochs':200,'min_epochs':40,'patience':30,
        'selection':'Main validation-selected recipe held fixed across ablations; no new architecture search',
        'test_visibility':'Post-hoc analysis of previously examined future years; not an untouched confirmation',
        'source':str(SOURCE),'main':str(MAIN)}
    engine.write_json(folder/'protocol.json',protocol)
    args=SimpleNamespace(route='delta',max_epochs=200,min_epochs=40,patience=30)
    trainer=engine.Trainer(args,data,folder,torch.device('cuda:0'))
    runs=[trainer.fit(hp,seed) for seed in SEEDS]
    del trainer
    torch.cuda.empty_cache()
    rows=[]
    for info in runs:
        checkpoint=torch.load(Path(info['run'])/'best.pt',map_location='cuda:0',weights_only=False)
        model=engine.DenseMLP(len(columns),hp).cuda()
        model.load_state_dict(checkpoint['state_dict'])
        for year in range(2020,2026):
            xx,yy,pp=features.load_year(SOURCE,year,domain,include_xco2_prior=True)
            xx=xx[:,columns].copy()
            engine.standardize(xx,median,mean,scale)
            pred=reference.torch_predict(model,xx,torch.device('cuda:0'),16384)*checkpoint['target_std']+checkpoint['target_mean']+pp
            rows.append({'domain':domain,'k':k,'seed':info['seed'],'year':year,'n':len(yy),
                'best_epoch':info['best_epoch'],'train_seconds':info['train_seconds'],**reference.metrics(yy,pred)})
        del model
    engine.write_csv(folder/'metrics.csv',rows)
    engine.write_json(folder/'complete.json',{'status':'complete','smoke':False,'runs':len(runs),'rows':len(rows)})


def diagnostics():
    folder=OUT/'diagnostics'
    folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('w')
    fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (folder/'complete.json').exists():return
    annual, spatial, latitudes, audits = [],[],[],[]
    for domain in DOMAINS:
        for year in range(2021,2026):
            with h5py.File(SOURCE/f'projected_{year}.h5') as source:
                ix=np.flatnonzero(source['domain'][:]==domain.encode())
                ids=source['sounding_id'][:][ix]
                yy=source['xco2'][:][ix].reshape(-1).astype('f8')
                pp=source['xco2_apriori'][:][ix].reshape(-1).astype('f8')
                lat=source['lat'][:][ix].reshape(-1)
                lon=source['lon'][:][ix].reshape(-1)
                audits.append({'domain':domain,'year':year,'n':len(ix),'projection':str(source.attrs['projection']),
                    'lat_min':float(lat.min()),'lat_max':float(lat.max()),'lon_min':float(lon.min()),'lon_max':float(lon.max())})
            dates=ids.astype('U')
            days, group=np.unique(np.array([s[:8] for s in dates]),return_inverse=True)
            predictions=[]
            for seed in SEEDS:
                path=MAIN/'mlp_search/studies'/domain/'delta/selected_models'/f'seed_{seed}'/f'predictions_{year}.npz'
                with np.load(path) as p:
                    np.testing.assert_array_equal(p['sounding_id'],ids)
                    np.testing.assert_allclose(p['reference_xco2'],yy,rtol=0,atol=0)
                    predictions.append(p['predicted_xco2'].astype('f8'))
            errors=np.array(predictions)-yy
            # Bootstrap dates, keeping all soundings and all seeds paired within each date.
            count=np.bincount(group)
            sums=np.array([np.bincount(group,weights=e) for e in errors])
            squares=np.array([np.bincount(group,weights=e*e) for e in errors])
            prior_squares=np.bincount(group,weights=(pp-yy)**2)
            boot=[]
            rng=np.random.default_rng(20261021+year)
            for _ in range(1000):
                drawn=rng.integers(0,len(days),len(days));n=count[drawn].sum()
                rmse=np.sqrt(squares[:,drawn].sum(axis=1)/n).mean()
                bias=(sums[:,drawn].sum(axis=1)/n).mean()
                gain=1-rmse/np.sqrt(prior_squares[drawn].sum()/n)
                boot.append([rmse,bias,gain])
            ci=np.quantile(boot,[.025,.975],axis=0)
            rmse=np.sqrt((errors**2).mean(axis=1)).mean()
            bias=errors.mean(axis=1).mean()
            prior_rmse=np.sqrt(np.mean((pp-yy)**2))
            annual.append({'domain':domain,'year':year,'n':len(yy),'dates':len(days),'rmse':rmse,'bias':bias,
                'prior_rmse':prior_rmse,'relative_gain':1-rmse/prior_rmse,
                'rmse_ci_low':ci[0,0],'rmse_ci_high':ci[1,0],'bias_ci_low':ci[0,1],'bias_ci_high':ci[1,1],
                'gain_ci_low':ci[0,2],'gain_ci_high':ci[1,2]})
            grid=np.floor((lat+90)/5).astype(int)*72+np.clip(np.floor((lon+180)/5).astype(int),0,71)
            for cell in np.unique(grid):
                keep=grid==cell
                if keep.sum()<30:continue
                spatial.append({'domain':domain,'year':year,'lat':(cell//72)*5-87.5,'lon':(cell%72)*5-177.5,
                    'n':int(keep.sum()),'bias':float(errors[:,keep].mean()),
                    'rmse':float(np.sqrt((errors[:,keep]**2).mean(axis=1)).mean())})
            for low in range(-90,90,10):
                keep=(lat>=low)&(lat<low+10)
                if keep.sum()<30:continue
                latitudes.append({'domain':domain,'year':year,'latitude':low+5,'n':int(keep.sum()),
                    'bias':float(errors[:,keep].mean()),'rmse':float(np.sqrt((errors[:,keep]**2).mean(axis=1)).mean())})
            print('diagnostics',domain,year,len(yy),flush=True)
    for name,rows in [('annual_bootstrap',annual),('spatial_5degree',spatial),('latitude_10degree',latitudes),('source_audit',audits)]:
        engine.write_csv(folder/f'{name}.csv',rows)
    engine.write_json(folder/'complete.json',{'status':'complete','years':5,'domains':3,'seeds':3,
        'resampling':'1000 within-year date-block draws; mean seed metrics, not ensemble predictions',
        'ci_scope':'Date-block sampling conditional on the sampled cohorts; not coverage or reference uncertainty'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--domain',choices=DOMAINS);p.add_argument('--k',type=int,choices=(0,10,20,30));p.add_argument('--diagnostics',action='store_true')
    a=p.parse_args();torch.set_num_threads(4)
    OUT.mkdir(parents=True,exist_ok=True)
    if a.diagnostics:diagnostics()
    else:ablation(a.domain,a.k)
