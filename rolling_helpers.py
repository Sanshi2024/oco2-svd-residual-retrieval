"""Sampling, metrics and checkpoint metadata for rolling ND evaluation."""
import hashlib
import json
import os
import numpy as np

QUARTERS = [(2021 + i // 4, 1 + i % 4) for i in range(13)]

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix('.partial.json')
    temp.write_text(json.dumps(value, indent=2, allow_nan=False))
    os.replace(temp, path)

def stable_order(values, salt):
    return np.argsort([hashlib.sha256(f'{salt}:{int(x)}'.encode()).digest() for x in values], kind='stable')

def quarter_bounds(year, quarter):
    month = 3 * (quarter - 1) + 1
    lower = (year - 3) * 10000 + month * 100 + 1
    start = year * 10000 + month * 100 + 1
    end = (year + 1) * 10000 + 101 if quarter == 4 else year * 10000 + (month + 3) * 100 + 1
    return lower, start, end

def window_indices(ids, year, quarter):
    dates = ids // 10**8
    months = dates // 100
    lower, start, _ = quarter_bounds(year, quarter)
    eligible = np.flatnonzero((dates >= max(lower, 20200101)) & (dates < start))
    train, validation, monthly = [], [], []
    for month in np.unique(months[eligible]):
        indices = eligible[months[eligible] == month]
        selected = indices[stable_order(ids[indices], f'month-{month}')[:1000]]
        unique_dates = np.unique(dates[selected])
        nval = max(1, int(round(len(unique_dates) * .2))) if len(unique_dates) > 1 else 0
        val_dates = unique_dates[stable_order(unique_dates, f'date-{month}')[:nval]]
        is_val = np.isin(dates[selected], val_dates)
        train.extend(selected[~is_val].tolist())
        validation.extend(selected[is_val].tolist())
        monthly.append(dict(month=int(month), available=len(indices), sampled=len(selected),
                            training=int((~is_val).sum()), validation=int(is_val.sum())))
    train, validation = np.sort(train), np.sort(validation)
    assert len(train) > 100 and len(validation) > 30
    assert not np.intersect1d(dates[train], dates[validation]).size
    assert not np.intersect1d(train, validation).size
    assert np.all(dates[np.r_[train, validation]] < start)
    assert np.all(dates[np.r_[train, validation]] >= max(lower, 20200101))
    assert all(row['sampled'] <= 1000 for row in monthly)
    return train, validation, monthly

def metrics(y, prediction):
    y, prediction = y.astype('f8'), prediction.astype('f8')
    assert np.isfinite(y).all() and np.isfinite(prediction).all() and len(y) > 30
    error = prediction - y
    return dict(n=len(y), rmse=float(np.sqrt(np.mean(error**2))), bias=float(np.mean(error)),
                r2=float(1 - np.sum(error**2) / np.sum((y - y.mean())**2)))
