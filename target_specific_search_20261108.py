"""Equal-budget, validation-only architecture/configuration searches by target."""
from pathlib import Path
import os,sys,json,fcntl,subprocess,hashlib,time
from concurrent.futures import ThreadPoolExecutor
BASE=Path('/path/to/OCO2/scripts/ML/paper_future_generalization')
ENGINE=BASE/'prior99_global_and_east_asia_20261018'
sys.path.insert(0,str(ENGINE))
sys.path.insert(0,str(BASE/'lgbm_validation_selection_20261025'))
import lgbm_validation_selection_20261025 as s
import numpy as np
import lightgbm as lgb
OUT=s.ML/'Paper_TargetSpecificSearch_20261108'
HERE=Path(__file__).resolve().parent
DOMAINS=('GL_OCEAN','GL_LAND','ND'); ROUTES=('absolute','delta')
OLD=s.OUT
def write(p,x):s.engine.write_json(p,x)

def tree(d,route,phase):
    s.OUT=OUT/'trees'/route
    folder=s.OUT/'studies'/d;folder.mkdir(parents=True,exist_ok=True)
    lock=(folder/'task.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if phase=='search':
        if (folder/'selection_frozen.json').exists():return
        if route=='delta':
            tr,va,sc=s.prepare(d)
            identity=json.loads((folder/'cohort_identity.json').read_text())
            assert identity==json.loads((OLD/f'studies/{d}/cohort_identity.json').read_text())
            record=json.loads((OLD/f'studies/{d}/selection_frozen.json').read_text())
            record['reuse_source']=str(OLD/f'studies/{d}/selection_frozen.json')
            write(folder/'selection_frozen.json',record);return
        def fit(cohort,index,hp,seed,train,val,datasets):
            f=folder/f'candidates/{index:02d}/seed_{seed}';f.mkdir(parents=True,exist_ok=True)
            if (f/'trained.json').exists():
                r=json.loads((f/'trained.json').read_text());assert r['hp']==hp
                assert r['model_sha256']==hashlib.sha256((f/'model.txt').read_bytes()).hexdigest();return r
            history={};params=dict(objective='regression_l2',metric='rmse',verbosity=-1,num_threads=8,
                seed=seed,bagging_seed=seed,feature_fraction_seed=seed,data_random_seed=17,
                bagging_freq=1,feature_pre_filter=False,**hp)
            a=lgb.Dataset(train['x'],label=train['y'],free_raw_data=False,
                params={'feature_pre_filter':False,'data_random_seed':17})
            b=lgb.Dataset(val['x'],label=val['y'],reference=a,free_raw_data=False)
            model=lgb.train(params,a,num_boost_round=3000,valid_sets=[b],valid_names=['validation'],
                callbacks=[lgb.early_stopping(100,verbose=False),lgb.record_evaluation(history)])
            model.save_model(str(f/'model.txt'),num_iteration=model.best_iteration)
            metrics=s.metrics(val['y'],model.predict(val['x'],num_threads=8))
            r=dict(candidate=index,hp=hp,seed=seed,validation_rmse=metrics['rmse'],validation_metrics=metrics,
                best_iteration=model.best_iteration,stopped_iteration=len(history['validation']['rmse']),
                model=str(f/'model.txt'),model_sha256=hashlib.sha256((f/'model.txt').read_bytes()).hexdigest(),
                target=route,finished_utc=s.utc())
            write(f/'history.json',history);write(f/'trained.json',r)
            print(d,route,index,seed,metrics,flush=True);return r
        s.fit=fit;s.search(d);return
    assert (OUT/'all_selections_frozen.json').exists()
    if (folder/'complete.json').exists():return
    tr,va,sc=s.prepare(d)
    selected=json.loads((folder/'selection_frozen.json').read_text())['selected']
    models={r['seed']:lgb.Booster(model_file=r['model']) for r in selected['runs']}
    rows=[]
    def score(label,role,data):
        for seed,model in models.items():
            native=model.predict(data['x'],num_threads=8)
            pred=native+data['prior'] if route=='delta' else native
            rows.append(dict(domain=d,route=route,family='LightGBM',seed=seed,year=label,role=role,
                n=len(pred),**s.metrics(data['y'],pred)))
            if role!='train':
                f=folder/f'selected_models/seed_{seed}';f.mkdir(parents=True,exist_ok=True)
                np.savez_compressed(f/f'predictions_{label}.npz',sounding_id=data['sounding_id'],
                    original_index=data['original_index'],reference_xco2=data['y'],xco2_apriori=data['prior'],
                    predicted_xco2=pred,predicted_target=native)
    score('2015-2019','train',tr);score('2020','validation',va);del tr,va
    for y in range(2021,2026):
        raw=s.global_raw(y,d);s.engine.standardize(raw['x'],**sc);score(str(y),'test',raw)
    s.engine.write_csv(folder/'metrics_runs.csv',rows)
    write(folder/'complete.json',dict(status='complete',rows=len(rows)))

def mlp_command(d,route,phase,gpu):
    env={**os.environ,'CUDA_VISIBLE_DEVICES':str(gpu),'OMP_NUM_THREADS':'4',
         'OPENBLAS_NUM_THREADS':'4','MKL_NUM_THREADS':'4'}
    cmd=[sys.executable,'-u',str(HERE/'target_mlp_worker.py'),'--source',str(s.SOURCE),
        '--baseline-root',str(s.ML/'Paper_ThreeDomain_Edge30_SVD30_Models_SecantGeometry_1519Train_2020Val_2021_2025Test'),
        '--output',str(OUT/'mlp'),'--architecture-only-config',str(OUT/'fixed_common_recipes.json'),
        '--include-xco2-prior','--domain',d,'--route',route,'--phase',phase,'--device','cuda:0','--max-epochs','500']
    with (OUT/f'{d}_{route}_mlp_{phase}.log').open('a') as log:
        subprocess.run(cmd,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)

def gpu_idle(g):
    lines=subprocess.check_output(['nvidia-smi','--query-gpu=index,memory.used,utilization.gpu',
        '--format=csv,noheader,nounits'],text=True).splitlines()
    return any(int(i)==g and int(m)<1024 and int(u)<15 for i,m,u in (x.split(',') for x in lines))

def main():
    OUT.mkdir(parents=True,exist_ok=True)
    lock=(OUT/'pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    if (OUT/'complete.json').exists():return
    protocol=dict(domains=DOMAINS,routes=ROUTES,train_years=list(range(2015,2020)),train_n=1000000,
        validation_year=2020,validation_n=200000,test_years=list(range(2021,2026)),features=99,pcs=[30]*3,
        mlp_candidates=128,mlp_search='independent architecture-only TPE, identical domain-specific nonarchitecture recipe across targets',
        hidden_depth=[1,4],hidden_width=[32,640,4],mlp_max_epochs=500,mlp_min_epochs=40,mlp_patience=30,
        tree_candidates=24,tree_max_rounds=3000,tree_patience=100,screen_seed=17,
        confirmation_seeds=[17,29,43],confirmation='MLP top3; LightGBM top3 plus reference',
        selection='mean validation RMSE only, independent per domain/family/target',
        test_tuning=False,retrospective_tests=True,partial_year=2025,all12_frozen_before_evaluation=True,
        reuse='Only existing residual LightGBM search with identical budget/candidates/cohorts; all MLP searches new at common500 epoch horizon',
        unchanged='SVD, masked LS30, original cohorts/scalers, manuscript main MLP checkpoints',eligible_gpus=[1,2,7])
    s.immutable(OUT/'protocol.json',protocol);write(OUT/'pipeline.json',dict(pid=os.getpid(),started=s.utc()))
    old=json.loads((s.GLOBAL/'mlp_search/fixed_nonarchitecture_recipes.json').read_text())
    jobs=[]
    for d in DOMAINS:
        hp=next(x['hp'] for x in old['jobs'] if x['domain']==d and x['route']=='delta')
        for route in ROUTES:jobs.append(dict(domain=d,route=route,hp=hp))
    s.immutable(OUT/'fixed_common_recipes.json',dict(jobs=jobs,selection_year=2020,
        note='same previously fixed domain recipe across both targets; independently selected hidden architectures'))
    def tree_work(d):
        for route in ROUTES:
            with (OUT/f'{d}_{route}_tree_search.log').open('a') as log:
                subprocess.run([sys.executable,'-u',__file__,'tree',d,route,'search'],stdout=log,stderr=subprocess.STDOUT,check=True)
    def gpu_work(g):
        group=[j for i,j in enumerate(jobs) if i%3==[1,2,7].index(g)]
        for j in group:
            while not gpu_idle(g):time.sleep(60)
            mlp_command(j['domain'],j['route'],'search',g)
    with ThreadPoolExecutor(max_workers=6) as pool:
        fs=[pool.submit(tree_work,d) for d in DOMAINS]+[pool.submit(gpu_work,g) for g in (1,2,7)]
        for f in fs:f.result()
    selections=[]
    for family in ('mlp','trees'):
        for d in DOMAINS:
            for route in ROUTES:
                folder=OUT/'mlp'/f'studies/{d}/{route}' if family=='mlp' else OUT/'trees'/route/f'studies/{d}'
                record=json.loads((folder/'selection_frozen.json').read_text());selections.append(str(folder))
                assert not record.get('test_metrics_used_for_selection',record.get('future_test_metrics_used_for_selection',True))
    write(OUT/'all_selections_frozen.json',dict(utc=s.utc(),studies=selections))
    write(OUT/'mlp/search_selection.complete.json',dict(status='complete',studies=6,selection_year=2020))
    for d in DOMAINS:
        for route in ROUTES:
            with (OUT/f'{d}_{route}_tree_evaluate.log').open('a') as log:
                subprocess.run([sys.executable,'-u',__file__,'tree',d,route,'evaluate'],stdout=log,stderr=subprocess.STDOUT,check=True)
            while not gpu_idle(1):time.sleep(60)
            mlp_command(d,route,'evaluate',1)
    write(OUT/'complete.json',dict(status='complete_pending_independent_verification',utc=s.utc(),studies=12))

if __name__=='__main__':
    try:
        if len(sys.argv)>1:tree(*sys.argv[2:])
        else:main()
    except Exception as e:
        OUT.mkdir(parents=True,exist_ok=True);write(OUT/f'failure_{os.getpid()}.json',dict(error=repr(e),utc=s.utc()));raise
