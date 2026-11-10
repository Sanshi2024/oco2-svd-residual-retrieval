"""Matched target and prior-noise controls for validation-selected tree recipes."""
from pathlib import Path
import sys, json, os, fcntl, subprocess, hashlib
import numpy as np
import lightgbm as lgb

SCRIPT=Path('/path/to/OCO2/scripts/ML/paper_future_generalization/lgbm_validation_selection_20261025')
sys.path.insert(0,str(SCRIPT))
import lgbm_validation_selection_20261025 as s
OLD=s.OUT
OUT=s.ML/'Paper_SelectedLightGBM_Controls_20261106'
DOMAINS=('GL_OCEAN','GL_LAND','ND')
SEEDS=(17,29,43)

def write(p,x):
    p.parent.mkdir(parents=True,exist_ok=True)
    q=p.with_suffix('.partial');q.write_text(json.dumps(x,indent=2));os.replace(q,p)

def worker(d,phase):
    folder=OUT/d;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'worker.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    frozen=json.loads((OLD/f'studies/{d}/selection_frozen.json').read_text())
    selected=frozen['selected'];hp=selected['hp'] if 'hp' in selected else selected['runs'][0]['hp']
    s.OUT=OUT/'cohorts'
    tr,va,scaler=s.prepare(d)
    assert json.loads((s.OUT/f'studies/{d}/cohort_identity.json').read_text())==json.loads((OLD/f'studies/{d}/cohort_identity.json').read_text())
    if phase=='train':
        models=[]
        datasets=(lgb.Dataset(tr['x'],label=tr['y'],free_raw_data=False),lgb.Dataset(va['x'],label=va['y'],free_raw_data=False))
        for seed in SEEDS:
            run=folder/f'absolute_{seed}';run.mkdir(exist_ok=True)
            if (run/'trained.json').exists():
                info=json.loads((run/'trained.json').read_text())
                assert info['hp']==hp
                assert info['sha256']==hashlib.sha256((run/'model.txt').read_bytes()).hexdigest()
            else:
                history={}
                params=dict(objective='regression_l2',metric='rmse',verbosity=-1,num_threads=8,
                    seed=seed,bagging_seed=seed,feature_fraction_seed=seed,data_random_seed=17,
                    bagging_freq=1,feature_pre_filter=False,**hp)
                model=lgb.train(params,datasets[0],num_boost_round=3000,valid_sets=[datasets[1]],valid_names=['validation'],
                    callbacks=[lgb.early_stopping(100,verbose=False),lgb.record_evaluation(history),lgb.log_evaluation(100)])
                model.save_model(str(run/'model.txt'),num_iteration=model.best_iteration)
                info=dict(seed=seed,hp=hp,model=str(run/'model.txt'),best_iteration=model.best_iteration,
                    stopping_iteration=len(history['validation']['rmse']),validation=s.metrics(va['y'],model.predict(va['x'],num_threads=8)),
                    sha256=hashlib.sha256((run/'model.txt').read_bytes()).hexdigest())
                write(run/'history.json',history);write(run/'trained.json',info)
            models.append(info);print(d,seed,'trained',flush=True)
        write(folder/'selection_frozen.json',dict(models=models,selected_hp_source=str(OLD/f'studies/{d}/selection_frozen.json'),
            source_sha256=hashlib.sha256((OLD/f'studies/{d}/selection_frozen.json').read_bytes()).hexdigest(),frozen_utc=s.utc(),
            absolute_target_specific_search=False,future_test_tuning=False))
        return
    assert (OUT/'all_selections_frozen.json').exists()
    models=json.loads((folder/'selection_frozen.json').read_text())['models']
    rows=[];noise=[]
    for year in (2020,2021,2022,2023,2024,2025):
        raw=s.global_raw(year,d);s.engine.standardize(raw['x'],**scaler)
        for info in models:
            seed=info['seed'];dest=folder/f'absolute_{seed}'
            model=lgb.Booster(model_file=info['model'])
            pred=model.predict(raw['x'],num_threads=8)
            assert np.isfinite(pred).all()
            np.savez_compressed(dest/f'predictions_{year}.npz',sounding_id=raw['sounding_id'],original_index=raw['original_index'],
                reference_xco2=raw['y'],xco2_apriori=raw['prior'],predicted_xco2=pred)
            rows.append(dict(domain=d,family='LightGBM',route='absolute',seed=seed,year=year,n=len(pred),**s.metrics(raw['y'],pred)))
        for info in selected['runs']:
            seed=info['seed'];model=lgb.Booster(model_file=info['model'])
            pred=model.predict(raw['x'],num_threads=8)+raw['prior']
            archive=OLD/f'studies/{d}/selected_models/seed_{seed}/predictions_{year}.npz'
            with np.load(archive) as z:
                for key,source in [('sounding_id','sounding_id'),('original_index','original_index'),('reference_xco2','y'),('xco2_apriori','prior')]:
                    assert np.array_equal(z[key],raw[source]),key
                assert np.max(np.abs(z['predicted_xco2']-pred))<1e-5
            rows.append(dict(domain=d,family='LightGBM',route='delta',seed=seed,year=year,n=len(pred),**s.metrics(raw['y'],pred)))
            if year in (2020,2024):
                z=np.random.default_rng(20261029+year*10+DOMAINS.index(d)).standard_normal(len(pred));z=(z-z.mean())/z.std()
                noisy_predictions=[]
                for sigma in (0,.5,1,2):
                    x=raw['x'].copy();prior=raw['prior'].astype('f8')+sigma*z
                    x[:,-1]=((prior-scaler['mean'][-1])/scaler['scale'][-1]).astype('f4')
                    yp=model.predict(x,num_threads=8)+prior
                    noisy_predictions.append(yp)
                    noise.append(dict(domain=d,family='LightGBM',year=year,seed=seed,magnitude=sigma,
                        path='coupled',kind='noise' if sigma else 'offset',n=len(pred),**s.metrics(raw['y'],yp),
                        perturbed_prior_rmse=s.metrics(raw['y'],prior)['rmse']))
                np.savez_compressed(folder/f'noise_{seed}_{year}.npz',sounding_id=raw['sounding_id'],
                    original_index=raw['original_index'],reference_xco2=raw['y'],xco2_apriori=raw['prior'],
                    noise_vector=z,sigmas=np.array([0,.5,1,2]),predicted_xco2=np.stack(noisy_predictions))
        print(d,year,'evaluated',flush=True)
    s.engine.write_csv(folder/'metrics.csv',rows);s.engine.write_csv(folder/'prior_noise.csv',noise)
    assert len(rows)==36 and len(noise)==24
    write(folder/'complete.json',dict(metric_rows=36,noise_rows=24,identity_checks='exact',status='complete'))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    if len(sys.argv)>1:worker(sys.argv[1],sys.argv[2]);return
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    write(OUT/'protocol.json',dict(domains=DOMAINS,seeds=SEEDS,train_n=1000000,val_n=200000,features=99,
        pcs=30,train_years=list(range(2015,2020)),validation_year=2020,test_years=list(range(2021,2026)),
        config='Existing residual-validation-selected configuration shared by target routes; no new configuration search',
        unchanged='MLP, original coefficients, scalers, cohorts and all historical outputs',max_rounds=3000,patience=100,
        noise_sigmas=[0,.5,1,2],noise_years=[2020,2024],retrospective_tests=True,partial_year=2025,
        residual_models='Reused and predictions recomputed and checked against original arrays'))
    for phase in ('train','evaluate'):
        children=[]
        for d in DOMAINS:
            log=(OUT/f'{d}_{phase}.log').open('a')
            p=subprocess.Popen([sys.executable,__file__,d,phase],stdout=log,stderr=subprocess.STDOUT,
                env={**os.environ,'OMP_NUM_THREADS':'8','MKL_NUM_THREADS':'8'})
            children.append((d,p,log))
        failed=[]
        for d,p,log in children:
            if p.wait():failed.append(d)
            log.close()
        if failed:
            write(OUT/'failure.json',dict(phase=phase,domains=failed));raise RuntimeError(failed)
        if phase=='train':write(OUT/'all_selections_frozen.json',dict(utc=s.utc(),domains=DOMAINS))
    write(OUT/'complete.json',dict(status='complete_pending_independent_verification',models=18,new_models=9,metric_rows=108,noise_rows=72))

if __name__=='__main__':main()
