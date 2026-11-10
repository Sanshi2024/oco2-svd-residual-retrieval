"""Independent source-identity and metric audit. Never trains models."""
from pathlib import Path
import json,csv,sys
import numpy as np
import h5py
import torch
import lightgbm as lgb
BASE=Path('/path/to/OCO2/scripts/ML/paper_future_generalization')
sys.path.insert(0,str(BASE/'lgbm_validation_selection_20261025'))
import lgbm_validation_selection_20261025 as s
R=s.ML/'Paper_TargetSpecificSearch_20261108'
DOM=('GL_OCEAN','GL_LAND','ND');SEEDS=(17,29,43)
def read(p):return list(csv.DictReader(p.open()))
def metrics(y,p):
 y=y.astype('f8');p=p.astype('f8');e=p-y
 return dict(rmse=float(np.sqrt(np.mean(e*e))),mae=float(np.mean(abs(e))),bias=float(e.mean()),r2=float(1-np.sum(e*e)/np.sum((y-y.mean())**2)))
def check(r,m,tol=1e-5):
 for k,v in m.items():assert np.isfinite(v) and abs(float(r[k])-v)<tol,(k,r[k],v)
def folder(f,d,t):return R/f'mlp/studies/{d}/{t}' if f=='MLP' else R/f'trees/{t}/studies/{d}'
def main():
 assert (R/'complete.json').exists()
 barrier=(R/'all_selections_frozen.json').stat().st_mtime
 rows=[];selections=[];stops=[];preds=0;valchecks=0
 for f in ('MLP','LightGBM'):
  for d in DOM:
   for t in ('absolute','delta'):
    p=folder(f,d,t);a=json.loads((p/'selection_frozen.json').read_text())
    assert (p/'selection_frozen.json').stat().st_mtime<=barrier
    assert a['feature_count']==99 and not a.get('test_metrics_used_for_selection',a.get('future_test_metrics_used_for_selection',True))
    selected=a['selected'];runs=selected['runs'];assert {x['seed'] for x in runs}==set(SEEDS)
    if f=='MLP':
     c=json.loads((p/'config.json').read_text());assert (c['trials'],c['max_epochs'],c['min_epochs'],c['patience'])==(128,500,40,30)
     done=json.loads((p/'search.complete.json').read_text());assert done['unique_finished_candidates']==128
     assert (done['training_rows'],done['validation_rows'])==(1000000,200000)
     confirm=json.loads((p/'confirmation.json').read_text());assert len(confirm)==3
     assert selected['candidate_id']==min(confirm,key=lambda x:x['validation_rmse_mean'])['candidate_id']
    else:
     assert a['candidates']==24
     assert selected['candidate']==min(a['confirmation_candidates'],key=lambda x:(x['mean_validation_rmse'],x['candidate']))['candidate']
     identity=json.loads((p/'cohort_identity.json').read_text())
     assert identity==json.loads((s.ML/f'Paper_LightGBM_ValidationSelection_20261025/studies/{d}/cohort_identity.json').read_text())
    records=read(p/'metrics_runs.csv');assert len(records)==21
    for row in records:
     assert np.isfinite([float(row[k]) for k in ('rmse','mae','bias','r2')]).all()
     if row['role']=='train':assert int(row['n'])==1000000
     if row['role']=='validation':assert int(row['n'])==200000
     rows.append(dict(domain=d,family=f,route=t,seed=row['seed'],year=row['year'],role=row['role'],n=row['n'],**{k:row[k] for k in ('rmse','mae','bias','r2')}))
    with np.load(p/'training_only_scaler.npz') as sc,np.load(s.GLOBAL/f'tree_models/{d}/delta/training_only_scaler.npz') as old:
     for k in ('median','mean','scale'):assert np.array_equal(sc[k],old[k]),(d,f,t,k)
    selections.append(dict(domain=d,family=f,route=t,hp=selected['hp'],validation_rmse=selected.get('validation_rmse_mean',selected.get('mean_validation_rmse'))))
    for run in runs:
     if f=='MLP':
      assert 0<run['best_epoch']<=run['epochs']<=500
      stops.append(dict(domain=d,family=f,route=t,seed=run['seed'],best=run['best_epoch'],stop=run['epochs'],cap=500))
     else:stops.append(dict(domain=d,family=f,route=t,seed=run['seed'],best=run['best_iteration'],stop=run['stopped_iteration'],cap=3000))
 for year in range(2020,2026):
  with h5py.File(s.SOURCE/f'projected_{year}.h5') as h:
   domains=h['domain'][:]
   for d in DOM:
    ix=np.flatnonzero(domains==d.encode());ids=h['sounding_id'][:][ix];indices=h['rows'][:][ix]
    y=h['xco2'][:][ix].astype('f4').reshape(-1);prior=h['xco2_apriori'][:][ix].astype('f4').reshape(-1)
    assert np.all(h['mode'][:][ix]==(b'ND' if d=='ND' else b'GL'))
    for f in ('MLP','LightGBM'):
     if year==2020 and f=='MLP':continue
     for t in ('absolute','delta'):
      p=folder(f,d,t)
      for seed in SEEDS:
       path=p/f'selected_models/seed_{seed}/predictions_{year}.npz';assert path.stat().st_mtime>=barrier
       with np.load(path) as z:
        for k,v in [('sounding_id',ids),('original_index',indices),('reference_xco2',y),('xco2_apriori',prior)]:assert np.array_equal(z[k],v),(d,f,t,year,k)
        row=next(x for x in rows if x['domain']==d and x['family']==f and x['route']==t and int(x['seed'])==seed and x['year']==str(year))
        assert int(row['n'])==len(y);check(row,metrics(y,z['predicted_xco2']))
        expected=z['predicted_target']+prior if t=='delta' else z['predicted_target']
        assert np.allclose(expected,z['predicted_xco2'],rtol=0,atol=1e-4)
        preds+=1
  print('verified prediction year',year,flush=True)
 # Independently recompute all full validation predictions, including unsaved MLP validation arrays.
 torch.set_num_threads(4)
 for d in DOM:
  x,y,prior=s.features.load_year(s.SOURCE,2020,d,include_xco2_prior=True)
  for f in ('MLP','LightGBM'):
   for t in ('absolute','delta'):
    p=folder(f,d,t);sel=json.loads((p/'selection_frozen.json').read_text())['selected']
    with np.load(p/'training_only_scaler.npz') as sc:xx=x.copy();s.engine.standardize(xx,**{k:sc[k] for k in ('median','mean','scale')})
    for run in sel['runs']:
     if f=='MLP':
      checkpoint=torch.load(Path(run['run'])/'best.pt',map_location='cpu',weights_only=False)
      model=s.engine.DenseMLP(99,sel['hp']);model.load_state_dict(checkpoint['state_dict'])
      native=s.reference.torch_predict(model,xx,torch.device('cpu'),16384)*checkpoint['target_std']+checkpoint['target_mean']
     else:native=lgb.Booster(model_file=run['model']).predict(xx,num_threads=4)
     pred=native+prior if t=='delta' else native
     row=next(a for a in rows if a['domain']==d and a['family']==f and a['route']==t and a['year']=='2020' and int(a['seed'])==run['seed'])
     check(row,metrics(y,pred),1e-4);valchecks+=1
  print('verified validation',d,flush=True)
 assert len(rows)==252 and preds==198 and valchecks==36
 s.engine.write_csv(R/'verified_metrics.csv',rows)
 s.engine.write_json(R/'verified_selections.json',selections)
 s.engine.write_json(R/'verified_stopping.json',stops)
 s.engine.write_json(R/'independent_verification.json',dict(status='verified',studies=12,models=36,metric_rows=252,
    prediction_files=198,independently_recomputed_validation=36,ordered_source_identities_exact=True,
    scalers_exact=True,all_selected_before_evaluation=True,retrospective=True,partial2025=True))
 print('VERIFIED',flush=True)
if __name__=='__main__':main()
