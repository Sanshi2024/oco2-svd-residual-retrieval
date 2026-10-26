"""No-training inference for the packaged residual MLP and LightGBM checkpoints."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def standardized(x, scaler_path):
    x=np.array(x,dtype='float32',copy=True)
    if x.ndim!=2 or x.shape[1]!=99:
        raise ValueError('Expected N x 99 unstandardized features in catalog order')
    if not np.all(np.isfinite(x[:,:90])):
        raise ValueError('Spectral scores must be finite')
    if not np.all(np.isfinite(x[:,-1])):
        raise ValueError('The final XCO2-prior column must be finite')
    with np.load(scaler_path,allow_pickle=False) as scaler:
        median,mean,scale=(scaler[k] for k in ['median','mean','scale'])
        if any(v.shape!=(99,) for v in [median,mean,scale]) or not np.all(scale>0):
            raise ValueError('Invalid training scaler')
        bad=~np.isfinite(x)
        x[bad]=np.broadcast_to(median,x.shape)[bad]
        x-=mean
        x/=scale
    return x


def make_mlp(hp):
    import torch
    from torch import nn
    class DenseMLP(nn.Module):
        def __init__(self):
            super().__init__()
            activation={'relu':nn.ReLU,'gelu':nn.GELU,'silu':nn.SiLU}[hp['activation']]
            widths=list(map(int,hp['architecture'].split('-')))
            size=99;layers=[]
            for index,width in enumerate(widths):
                layers.append(nn.Linear(size,width))
                if hp['normalization']=='layernorm':layers.append(nn.LayerNorm(width))
                layers.append(activation())
                if hp['dropout'] and index<len(widths)-1:layers.append(nn.Dropout(hp['dropout']))
                size=width
            layers.append(nn.Linear(size,1))
            self.network=nn.Sequential(*layers)
        def forward(self,x):return self.network(x).squeeze(-1)
    return DenseMLP()


def predict(x, model_path, scaler_path, family, batch_size=16384):
    prior=np.asarray(x,dtype='float32')[:,-1].copy()
    values=standardized(x,scaler_path)
    if family=='MLP':
        import torch
        torch.set_num_threads(4)
        # Load only the trusted, checksummed author-supplied artifacts.
        saved=torch.load(model_path,map_location='cpu',weights_only=False)
        model=make_mlp(saved['hp'])
        model.load_state_dict(saved['state_dict'],strict=True)
        model.eval()
        with torch.inference_mode():
            residual=np.concatenate([model(torch.from_numpy(values[i:i+batch_size])).numpy()
                                     for i in range(0,len(values),batch_size)])
        residual=residual*saved['target_std']+saved['target_mean']
    elif family=='LightGBM':
        import lightgbm as lgb
        model=lgb.Booster(model_file=str(model_path))
        residual=model.predict(values,num_threads=4)
    else:raise ValueError('Unsupported family: '+family)
    if not np.all(np.isfinite(residual)):raise ValueError('Nonfinite prediction')
    return {'predicted_target':residual,'predicted_xco2':residual+prior,'xco2_apriori':prior}


def verify(root):
    manifest=json.loads((root/'SHA256.json').read_text())
    for record in manifest['files']:
        path=root/record['path']
        if hashlib.sha256(path.read_bytes()).hexdigest()!=record['sha256']:
            raise RuntimeError('Checksum mismatch: '+record['path'])
    return len(manifest['files'])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=Path(__file__).resolve().parent)
    parser.add_argument('--verify',action='store_true')
    parser.add_argument('--cohort',choices=['GL_OCEAN','GL_LAND','ND','EAST_ASIA_ND'])
    parser.add_argument('--family',choices=['MLP','LightGBM'])
    parser.add_argument('--seed',type=int,choices=[17,29,43],default=17)
    parser.add_argument('--input',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    if args.verify:
        print(json.dumps({'files_verified':verify(args.root)}));return
    if not all([args.cohort,args.family,args.input,args.output]):
        parser.error('Inference requires --cohort --family --input --output')
    catalog=json.loads((args.root/'model_catalog.json').read_text())
    record=next(r for r in catalog['models'] if (r['cohort'],r['family'],r['seed'])==(args.cohort,args.family,args.seed))
    paths={field:args.root/record[field] for field in ['model','scaler']}
    for field in paths:
        if not paths[field].is_file():
            raise FileNotFoundError('Server-held artifact not copied into this local preparation package: '+record[field])
    if hashlib.sha256(paths['model'].read_bytes()).hexdigest()!=record['model_sha256']:
        raise RuntimeError('Checkpoint digest differs from the verified selected model')
    with np.load(args.input,allow_pickle=False) as data:
        x=data['x']
        if 'feature_names' in data and list(data['feature_names'].astype(str))!=catalog['feature_names']:
            raise ValueError('Feature-name order mismatch')
        result=predict(x,paths['model'],paths['scaler'],args.family)
        for name in ['sounding_id','original_index']:
            if name in data:
                if len(data[name])!=len(x):raise ValueError('Identity length mismatch')
                result[name]=data[name]
    if args.output.exists():raise FileExistsError('Refuse to overwrite '+str(args.output))
    np.savez_compressed(args.output,**result)
    print(json.dumps({'cohort':args.cohort,'family':args.family,'seed':args.seed,'rows':len(x),'output':str(args.output)}))


if __name__=='__main__':main()
