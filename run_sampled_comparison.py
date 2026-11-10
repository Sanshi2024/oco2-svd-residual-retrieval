"""Bound archive I/O for the regional ND comparison; reuse exact completed batches."""
import argparse
import concurrent.futures as futures
import csv
import fcntl
import hashlib
import json
import math
import random
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np

import run_regional_comparison as base


def schedule(rows, year, fraction):
    """Interleave months, with a deterministic shuffle within each month."""
    buckets = defaultdict(list)
    # Keep the original batch numbering so old feature caches remain reusable.
    for i in range(0, len(rows), 32):
        group = rows[i:i+32]
        paired = [r for r in group if r['status'] == 'paired_nonempty']
        if paired:
            buckets[group[len(group)//2]['date'][5:7]].append(
                (f'{year}_{i:04d}.npz', paired))
    rng = random.Random(1237 + year)
    months = sorted(buckets)
    rng.shuffle(months)
    for values in buckets.values():
        rng.shuffle(values)
    ordered = []
    for i in range(max(map(len, buckets.values()), default=0)):
        ordered.extend(buckets[m][i] for m in months if i < len(buckets[m]))
    return deque(ordered[:math.ceil(fraction * len(ordered))])


def load(path):
    with np.load(path) as f:
        return tuple(f[k] for k in ('ids', 'x', 'y', 'prior')), json.loads(str(f['audit']))


def collect(rows, out, old, workers, budget, fraction):
    years = range(2017, 2026)
    queues = {y: schedule([r for r in rows if int(r['year']) == y], y, fraction)
              for y in years}
    parts, audits, ids = defaultdict(list), defaultdict(list), defaultdict(set)
    sampled = defaultdict(list)
    groups = [tuple(range(2017, 2020)), tuple(range(2020, 2026))]
    with futures.ProcessPoolExecutor(max_workers=workers, initializer=base.init_worker) as pool:
        while True:
            active = []
            for group in groups:
                if sum(len(ids[y]) for y in group) >= budget and all(ids[y] for y in group):
                    continue
                available = [y for y in group if queues[y]]
                # Seek approximately equal annual quotas; redistribute exhausted years.
                active.extend(sorted(available, key=lambda y: (len(ids[y]), y)))
            if not active:
                break
            pending = []
            for y in active[:workers]:
                name, group = queues[y].popleft()
                cached = old / name
                if cached.exists():
                    pending.append((y, name, cached, None))
                else:
                    path = out / 'batches' / name
                    pending.append((y, name, path, pool.submit(base.batch, (path, group))))
            for y, name, path, task in pending:
                if task is not None:
                    task.result()
                arrays, audit = load(path)
                parts[y].append(arrays)
                ids[y].update(arrays[0].tolist())
                audits[y].extend(audit)
                sampled[y].append(str(path))
            print('Sampled batches', sum(map(len, sampled.values())),
                  'usable unique rows', {y: len(ids[y]) for y in years}, flush=True)
    data, summary = {}, {}
    for y in years:
        assert ids[y], f'No usable rows for {y}'
        arrays = tuple(np.concatenate(a) for a in zip(*parts[y]))
        _, keep = np.unique(arrays[0], return_index=True)
        data[y] = tuple(a[keep] for a in arrays)
        summary[y] = {
            'batches': sampled[y], 'eligible_in_read_files': sum(a['eligible'] for a in audits[y]),
            'unique_usable_before_cap': len(keep),
            'read_errors': [a for a in audits[y] if 'error' in a],
        }
    for group in groups:
        counts = {y: len(data[y][0]) for y in group}
        total = sum(counts.values())
        if total > budget:
            # Balanced capped allocation: small years keep their entire available pool.
            remaining = budget
            for j, y in enumerate(sorted(group, key=lambda y: (counts[y], y))):
                n = min(counts[y], remaining // (len(group) - j))
                keep = np.sort(np.random.default_rng(1237+y).choice(counts[y], n, replace=False))
                data[y] = tuple(a[keep] for a in data[y])
                remaining -= n
            assert remaining == 0
    for y in years:
        summary[y]['selected'] = len(data[y][0])
    base.atomic_json(out/'sample_audit.json', dict(years=summary, files=audits))
    print('Frozen cohort sizes', {y: summary[y]['selected'] for y in years}, flush=True)
    return data


def main():
    p = argparse.ArgumentParser()
    p.add_argument('manifest', type=Path)
    p.add_argument('out', type=Path)
    p.add_argument('old', type=Path)
    p.add_argument('--workers', type=int, default=12)
    p.add_argument('--budget', type=int, default=500000)
    p.add_argument('--max-fraction', type=float, default=.8)
    args = p.parse_args()
    assert args.workers > 0 and args.budget > 0 and 0 < args.max_fraction < 1
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out/'batches').mkdir(exist_ok=True)
    lock = (args.out/'pipeline.lock').open('w')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    def digest(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    hashes = dict(manifest=digest(args.manifest), script=digest(Path(base.__file__)), smoke=False)
    assert json.loads((args.old/'contract.json').read_text()) == hashes
    contract = dict(**hashes, wrapper=digest(Path(__file__)), budget=args.budget,
                    max_fraction=args.max_fraction, workers=args.workers)
    spec = args.out/'contract.json'
    if spec.exists():
        assert json.loads(spec.read_text()) == contract
    else:
        base.atomic_json(spec, contract)
    if (args.out/'complete.json').exists():
        return
    with args.manifest.open() as f:
        rows = [r for r in csv.DictReader(f) if r['l2_path'] and r['year'] >= '2017']
    data = collect(rows, args.out, args.old, args.workers, args.budget, args.max_fraction)
    import torch
    base.compare(data, args.out, torch.device('cuda:0' if torch.cuda.is_available() else 'cpu'), False)


if __name__ == '__main__':
    main()
