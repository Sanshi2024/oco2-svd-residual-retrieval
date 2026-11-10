"""Quarterly retrospective updates from frozen global ND checkpoints."""
import csv
import fcntl
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import torch

ROOT = Path('/path/to/OCO2')
ML = ROOT / 'output/ML'
MATCH = ML / 'Paper_SpT_Screening_Matched_20261109/sampled_500k'
STUDY = ML / 'Paper_MainMLP_Epoch500_20261024/studies/ND/delta'
OUT = ML / 'Paper_GlobalND_SpT_Rolling_20261110'
sys.path.insert(0, str(ROOT / 'scripts/ML/paper_future_generalization/prior99_global_and_east_asia_20261018'))
from search_mlp_secant import DenseMLP, standardize
from rolling_helpers import QUARTERS, metrics, quarter_bounds, sha, window_indices, write_json


def prepare():
    audit = json.loads((MATCH / 'sample_audit.json').read_text())['years']
    data = []
    with h5py.File(ML / 'Paper_ThreeDomain_Edge30_SVD30_1519Train_2020Val_2021_2025Test/projected_2020.h5') as h:
        val_ids = h['sounding_id'][:]
    with np.load(MATCH / 'split_ids.npz') as split:
        for year in range(2020, 2025):
            arrays = []
            for path in audit[str(year)]['batches']:
                with np.load(MATCH / 'batches' / Path(path).name, allow_pickle=False) as a:
                    arrays.append(tuple(a[k] for k in ('ids', 'x', 'y', 'prior')))
            ids, *rest = [np.concatenate(x) for x in zip(*arrays)]
            _, first = np.unique(ids, return_index=True)
            index = {int(ids[i]): i for i in first}
            order = np.array([index[int(i)] for i in split[str(year)]])
            assert np.array_equal(ids[order], split[str(year)])
            if year == 2020:
                keep = ~np.isin(ids[order], val_ids)
                assert int((~keep).sum()) == 5479
                order = order[keep]
            data.append(tuple(x[order] for x in [ids, *rest]))
    return tuple(np.concatenate(x) for x in zip(*data))


def run(seed):
    device = torch.device('cuda:0')
    torch.set_num_threads(4)
    torch.use_deterministic_algorithms(True)
    ids, raw, ref, prior = prepare()
    dates = ids // 10**8
    checkpoint = STUDY / f'selected_models/seed_{seed}/model.pt'
    scaler = STUDY / 'training_only_scaler.npz'
    hashes = dict(base_sha256=sha(checkpoint), scaler_sha256=sha(scaler))
    cp = torch.load(checkpoint, map_location='cpu', weights_only=False)
    with np.load(scaler) as s:
        x = torch.as_tensor(standardize(raw, *(s[k] for k in ('median', 'mean', 'scale'))), device=device)
    y = torch.as_tensor(ref - prior, device=device)
    assert x.shape == (len(ids), 99) and torch.isfinite(x).all()
    rows = []
    for year, q in QUARTERS:
        folder = OUT / f'seed_{seed}' / f'{year}Q{q}'
        folder.mkdir(parents=True, exist_ok=True)
        done = folder / 'complete.json'
        if done.exists():
            saved = json.loads(done.read_text())
            assert all(saved[k] == v for k, v in hashes.items())
            rows.extend(saved['metrics'])
            continue
        tr, va, monthly = window_indices(ids, year, q)
        _, start, end = quarter_bounds(year, q)
        te = np.flatnonzero((dates >= start) & (dates < end))
        assert len(te) > 30
        torch.manual_seed(seed)
        generator = torch.Generator(device=device).manual_seed(seed + 4 * year + q)
        m = DenseMLP(99, cp['hp']).to(device)
        m.load_state_dict(cp['state_dict'])
        assert sum(p.numel() for p in m.parameters()) == 164793
        opt = torch.optim.AdamW(m.parameters(), lr=1e-4, weight_decay=1e-4)

        @torch.no_grad()
        def predict(i):
            m.eval()
            return torch.cat([m(x[j]).flatten().double() for j in np.array_split(i, max(1, (len(i)+8191)//8192))]) * cp['target_std'] + cp['target_mean']

        def val():
            return float(((predict(va) - y[va].double()).square().mean()).sqrt())

        best, epoch_best, waiting = val(), 0, 0
        history = [best]
        state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
        for epoch in range(1, 501):
            m.train()
            for j in torch.randperm(len(tr), generator=generator, device=device).split(1024):
                idx = torch.as_tensor(tr, device=device)[j]
                target = ((y[idx] - cp['target_mean']) / cp['target_std']).float()
                opt.zero_grad(set_to_none=True)
                loss = (m(x[idx]).flatten() - target).square().mean()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(m.parameters(), 5.)
                opt.step()
            score = val()
            assert np.isfinite(score)
            history.append(score)
            if score < best - 1e-5:
                best, epoch_best, waiting = score, epoch, 0
                state = {k: v.detach().cpu().clone() for k, v in m.state_dict().items()}
            else:
                waiting += 1
            if epoch >= 40 and waiting >= 30:
                break
        torch.save(state, folder / 'best.pt')
        meta = dict(**hashes, best_epoch=epoch_best, stopping_epoch=epoch, validation_rmse=best,
                    history=history, monthly=monthly)
        write_json(folder / 'selection_frozen.json', meta)
        np.savez_compressed(folder / 'update_ids.npz', train=ids[tr], validation=ids[va])
        m.load_state_dict(state)
        updated = predict(te).cpu().numpy() + prior[te]
        with np.load(MATCH / f'global_frozen_{seed}_{year}.npz') as a:
            aid = a['ids']
            mask = ((aid // 10**8) >= start) & ((aid // 10**8) < end)
            assert np.array_equal(aid[mask], ids[te])
            assert np.array_equal(a['ref'][mask], ref[te])
            assert np.array_equal(a['prior'][mask], prior[te])
            frozen = a['pred'][mask]
        np.savez_compressed(folder / 'predictions.npz', ids=ids[te], ref=ref[te], prior=prior[te],
                            fixed=frozen, rolling=updated)
        result = [dict(seed=seed, year=year, quarter=q, family=f, **metrics(ref[te], p))
                  for f, p in [('Fixed MLP', frozen), ('Rolling MLP', updated), ('Prior', prior[te])]]
        write_json(done, dict(**meta, metrics=result))
        rows.extend(result)
        print(seed, year, q, epoch_best, epoch, result, flush=True)
    assert sha(checkpoint) == hashes['base_sha256'] and sha(scaler) == hashes['scaler_sha256']
    assert len(rows) == 39
    with (OUT / f'metrics_{seed}.csv').open('w') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == '__main__':
    seed = int(sys.argv[1])
    assert seed in (17, 29, 43)
    OUT.mkdir(parents=True, exist_ok=True)
    with (OUT / f'seed_{seed}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run(seed)
