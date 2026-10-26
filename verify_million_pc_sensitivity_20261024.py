"""Independent full-cohort checkpoint and saved-prediction audit."""
import csv
import json
from pathlib import Path
import h5py
import numpy as np
import torch
import million_pc_sensitivity_20261024 as m


def check_metric(saved, current):
    for name in ('rmse','mae','bias','r2'):
        assert abs(float(saved[name])-float(current[name]))<2e-5,(name,saved,current)


def source_identity(domain, year):
    with h5py.File(m.base.SOURCE/f'projected_{year}.h5') as f:
        keep=np.flatnonzero(f['domain'][:]==domain.encode())
        assert np.all(f['mode'][:][keep]==(b'ND' if domain=='ND' else b'GL'))
        result={dest:f[src][:][keep].reshape(-1) for dest,src in (
            ('sounding_id','sounding_id'),('original_index','rows'),
            ('reference_xco2','xco2'),('xco2_apriori','xco2_apriori'))}
    assert len(np.unique(result['sounding_id']))==len(keep)
    for name in ('reference_xco2','xco2_apriori'):
        result[name]=result[name].astype('f4')
    return result


def main():
    torch.set_num_threads(4)
    out=m.OUT
    frozen=json.loads((out/'selection_frozen.json').read_text())
    assert frozen['test_used_for_selection'] is False
    with (out/'metrics.csv').open() as f:
        rows=list(csv.DictReader(f))
    index={(r['domain'],int(r['k']),r['recipe'],int(r['seed']),int(r['year'])):r for r in rows}
    assert len(index)==len(rows)==336
    assert len(list((out/'evaluation').glob('*/k*/*/seed_*/predictions_*.npz')))==288
    audit=[]
    files=0
    for domain in m.base.DOMAINS:
        parts=[m.features.load_year(m.base.SOURCE,y,domain,include_xco2_prior=True) for y in range(2015,2020)]
        x,y,p=[np.concatenate([part[i] for part in parts]) for i in range(3)]
        del parts
        assert len(y)==1000000
        for year in range(2015,2020):
            assert len(source_identity(domain,year)['sounding_id'])==200000
        median=np.nanmedian(x,axis=0).astype('f4')
        median[~np.isfinite(median)]=0
        filled=np.where(np.isfinite(x),x,median)
        mean=filled.mean(axis=0,dtype='f8').astype('f4')
        scale=filled.std(axis=0,dtype='f8').astype('f4')
        scale[scale==0]=1
        del filled
        m.engine.standardize(x,median,mean,scale)
        vx,vy,vp=m.features.load_year(m.base.SOURCE,2020,domain,include_xco2_prior=True)
        assert len(vy)==200000
        m.engine.standardize(vx,median,mean,scale)
        identities={year:source_identity(domain,year) for year in range(2020,2026)}
        oldrows=[]
        if domain!='GL_LAND':
            with (m.MAIN500/f'studies/{domain}/delta/metrics_runs.csv').open() as f:
                oldrows=list(csv.DictReader(f))
        for k in m.KS:
            dest=out/f'evaluation/{domain}/k{k}'
            columns=m.columns_for(k)
            with np.load(dest/'scaler.npz') as sc:
                np.testing.assert_array_equal(columns,sc['columns'])
                for name,expected in [('median',median),('mean',mean),('scale',scale)]:
                    np.testing.assert_allclose(sc[name],expected[columns],rtol=1e-6,atol=1e-7)
            recipes=[('fixed',m.hp_for(domain)['lr'])]
            if domain=='ND':
                recipes.append(('lr_selected',frozen['nd_rates'][str(k)]['lr']))
            xx=x[:,columns].copy()
            vxx=vx[:,columns].copy()
            for recipe,lr in recipes:
                for seed in m.SEEDS:
                    info=m.run_info(domain,k,lr,seed)
                    cp=torch.load(Path(info['run'])/'best.pt',map_location='cuda:0',weights_only=False)
                    assert cp['hp']=={**m.hp_for(domain),'lr':lr} and cp['seed']==seed
                    hist=json.loads((Path(info['run'])/'history.json').read_text())
                    assert len(hist)==info['epochs']<=500
                    assert [h['epoch'] for h in hist]==list(range(1,len(hist)+1))
                    assert abs(min(h['validation_rmse'] for h in hist)-info['best_validation_rmse'])<1e-5
                    for h in hist:
                        expected_lr=lr if domain!='GL_LAND' else 1e-6+(lr-1e-6)*(1+np.cos(np.pi*(h['epoch']-1)/500))/2
                        assert abs(h['lr']-expected_lr)<1e-10
                    model=m.engine.DenseMLP(len(columns),cp['hp']).cuda()
                    model.load_state_dict(cp['state_dict'])
                    for year,inputs,truth,prior in [(0,xx,y,p),(2020,vxx,vy,vp)]:
                        pred=m.reference.torch_predict(model,inputs,torch.device('cuda:0'),16384)
                        pred=pred*cp['target_std']+cp['target_mean']+prior
                        row=index[domain,k,recipe,seed,year]
                        assert int(row['n'])==len(truth)
                        check_metric(row,m.reference.metrics(truth,pred))
                        if year==2020:
                            assert abs(float(row['rmse'])-info['best_validation_rmse'])<2e-5
                            with np.load(dest/recipe/f'seed_{seed}/predictions_2020.npz') as prediction:
                                np.testing.assert_allclose(pred,prediction['predicted_xco2'],rtol=0,atol=1e-4)
                    for year in range(2020,2026):
                        path=dest/recipe/f'seed_{seed}/predictions_{year}.npz'
                        assert path.stat().st_mtime>=(out/'selection_frozen.json').stat().st_mtime
                        with np.load(path) as pred:
                            for name,expected in identities[year].items():
                                np.testing.assert_array_equal(pred[name],expected)
                            row=index[domain,k,recipe,seed,year]
                            assert int(row['n'])==len(pred['predicted_xco2'])
                            check_metric(row,m.reference.metrics(pred['reference_xco2'],pred['predicted_xco2']))
                        if k==30 and recipe=='fixed' and domain!='GL_LAND':
                            old=next(r for r in oldrows if int(r['seed'])==seed and r['year']==str(year))
                            check_metric(old,row)
                        files+=1
                    audit.append(dict(domain=domain,k=k,recipe=recipe,lr=lr,seed=seed,
                        stopped_epoch=info['epochs'],best_epoch=info['best_epoch'],
                        hit_500=info['epochs']==500,
                        train_rmse=float(index[domain,k,recipe,seed,0]['rmse']),
                        validation_rmse=float(index[domain,k,recipe,seed,2020]['rmse'])))
                    del model
            print('VERIFIED',domain,k,flush=True)
        del x,y,p,vx,vy,vp,xx,vxx
    assert len(audit)==48 and files==288
    m.engine.write_json(out/'independent_verification.json',dict(status='passed',
        model_records=48,metric_rows=336,prediction_files=files,training_rows_per_domain=1000000,
        validation_rows_per_domain=200000,training_only_scalers_recomputed=True,
        exact_source_id_reference_alignment=True,train_validation_predictions_recomputed=True,
        histories_and_scheduler_checked=True,original_ocean_nd_fixed30_reproduced=True,
        cap500_records=[r for r in audit if r['hit_500']],audit=audit))
    print('ALL_VERIFIED',len(audit),files,flush=True)


if __name__=='__main__':
    main()
