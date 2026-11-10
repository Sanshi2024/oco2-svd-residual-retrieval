"""Refresh diagnostics for the newly adopted land-glint residual MLP."""
import csv
import json
from pathlib import Path
import sys
import h5py
import numpy as np
import torch

ROOT = Path('/path/to/OCO2/output/ML')
OUT = ROOT / 'Paper_KD_TargetControl_20261105'
SOURCE = ROOT / 'Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test'
ENGINE = Path('/path/to/OCO2/scripts/ML/paper_future_generalization/prior99_global_and_east_asia_20261018')
sys.path.insert(0, str(ENGINE))
import search_mlp_secant as e
import train_domain_svd50 as f
import two_route_training_reference as r
torch.set_num_threads(4)


def write(name, rows):
    with (OUT / name).open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    domain = 'GL_LAND'
    models = [m for m in json.loads((OUT/domain/'selection_frozen.json').read_text())['models']
              if m['family']=='MLP' and m['route']=='delta']
    assert len(models)==3
    annual, latitudes, noise = [], [], []
    for year in range(2021,2026):
        with h5py.File(SOURCE/f'projected_{year}.h5') as h:
            ix=np.flatnonzero(h['domain'][:]==domain.encode())
            ids=h['sounding_id'][:][ix]
            lat=h['lat'][:][ix].reshape(-1)
        preds=[]
        for m in models:
            with np.load(Path(m['run'])/'evaluation'/f'predictions_{year}.npz') as p:
                assert np.array_equal(ids,p['sounding_id'])
                y=p['reference_xco2'].astype('f8'); prior=p['xco2_apriori'].astype('f8')
                preds.append(p['predicted_xco2'])
        errors=np.array(preds)-y
        days,g=np.unique(np.array([str(s)[:8] for s in ids]),return_inverse=True)
        count=np.bincount(g)
        sums=np.array([np.bincount(g,weights=err) for err in errors])
        rng=np.random.default_rng(20261021+year)
        boot=[]
        for _ in range(1000):
            draw=rng.integers(0,len(days),len(days))
            boot.append((sums[:,draw].sum(axis=1)/count[draw].sum()).mean())
        lo,hi=np.quantile(boot,[.025,.975])
        annual.append(dict(domain=domain,year=year,n=len(y),bias=errors.mean(),bias_ci_low=lo,bias_ci_high=hi))
        for low in range(-90,90,10):
            keep=(lat>=low)&(lat<low+10)
            if keep.sum()>=30:
                latitudes.append(dict(domain=domain,year=year,latitude=low+5,n=keep.sum(),bias=errors[:,keep].mean()))
    scaler=np.load(ROOT/'Paper_MainMLP_Epoch500_20261024/studies/GL_LAND/delta/training_only_scaler.npz')
    for year in (2020,2024):
        x,y,prior=f.load_year(SOURCE,year,domain,None,True)
        e.standardize(x,scaler['median'],scaler['mean'],scaler['scale'])
        z=np.random.default_rng(20261029+year*10+1).standard_normal(len(y))
        z=(z-z.mean())/z.std()
        for m in models:
            cp=torch.load(Path(m['run'])/'best.pt',map_location='cpu',weights_only=False)
            model=e.DenseMLP(99,cp['hp']);model.load_state_dict(cp['state_dict']);model.eval()
            for sigma in (0,.5,1,2):
                altered=x.copy()
                if sigma:
                    altered[:,-1]=((prior.astype('f8')+sigma*z-scaler['mean'][-1])/scaler['scale'][-1]).astype('f4')
                native=r.torch_predict(model,altered,torch.device('cpu'),16384)*cp['target_std']+cp['target_mean']
                pred=native.astype('f8')+prior.astype('f8')+sigma*z
                if sigma==0:
                    with np.load(Path(m['run'])/'evaluation'/f'predictions_{year}.npz') as archived:
                        assert np.max(np.abs(pred-archived['predicted_xco2']))<1e-4
                noise.append(dict(domain=domain,family='MLP',seed=m['seed'],year=year,magnitude=sigma,
                    path='coupled',kind='noise' if sigma else 'offset',n=len(y),**r.metrics(y.astype('f8'),pred),
                    perturbed_prior_rmse=float(np.sqrt(np.mean((prior.astype('f8')+sigma*z-y)**2)))))
    write('land_annual_bootstrap.csv',annual)
    write('land_latitude.csv',latitudes)
    write('land_prior_noise.csv',noise)
    print('Land diagnostics complete: 5 annual intervals, latitude profiles, 24 prior-noise evaluations')


if __name__=='__main__':
    main()
