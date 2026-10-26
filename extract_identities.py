"""Extract ordered final identities from existing projections; never reselect rows."""
import argparse
import json
from pathlib import Path
import hashlib
import h5py
import numpy as np


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--projection-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--years',default=','.join(map(str,range(2015,2026))))
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    records=[]
    for year in map(int,args.years.split(',')):
        path=args.projection_root/f'projected_{year}.h5'
        destination=args.output/f'projected_{year}_identity.npz'
        if destination.exists():raise FileExistsError(destination)
        with h5py.File(path,'r') as source:
            names=['sounding_id','rows','domain','mode','granule_index','footprint_index',
                   'resampled_index','xco2','xco2_apriori']
            arrays={k:source[k][:] for k in names}
            n=len(arrays['sounding_id'])
            assert all(len(v)==n for v in arrays.values())
            assert len(np.unique(arrays['sounding_id']))==n
            counts={d:int(np.sum(arrays['domain']==d.encode())) for d in ['GL_OCEAN','GL_LAND','ND']}
            for domain in counts:
                assert np.all(arrays['mode'][arrays['domain']==domain.encode()]==(b'ND' if domain=='ND' else b'GL'))
            np.savez_compressed(destination,**arrays)
        records.append({'year':year,'source':str(path.resolve()),'output':destination.name,'counts':counts,
                        'kind':'extracted final order, NOT historical removal audit',
                        'sha256':hashlib.sha256(destination.read_bytes()).hexdigest()})
    target=args.output/'identity_inventory.json'
    if target.exists():raise FileExistsError(target)
    target.write_text(json.dumps(records,indent=2))
    print(json.dumps({'annual_manifests':len(records),'inventory':str(target)}))

if __name__=='__main__':main()
