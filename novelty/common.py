import json
import os
import sys
import numpy as np
import torch

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))), os.path.dirname(os.path.abspath(__file__))]
from train import fit, predict, Scaler
from evaluate import metrics
from models import n_params

torch.set_num_threads(2)


def budget(n_fit):
    """Same early-stopping rule for every architecture; depends only on training-set size."""
    return (15, 2) if n_fit > 100_000 else (200, 10)


def log_to(path):
    f = open(path, "a")
    def _log(s):
        print(s, flush=True); f.write(s + "\n"); f.flush()
    return _log


def run(name, out_dir, model, X, Y, fit_idx, val_idx, test_idx, lr, seed, init_state=None, extra=None, save_model=True):
    """Train with early stopping, predict on test_idx, save json/preds/(weights). Skips if already done."""
    os.makedirs(out_dir, exist_ok=True)
    jp = f"{out_dir}/{name}.json"
    if os.path.exists(jp):
        return json.load(open(jp))
    multi = Y.ndim == 2
    lab = (~np.isnan(Y).all(1)) if multi else ~np.isnan(Y)
    fit_idx, val_idx = fit_idx[lab[fit_idx]], val_idx[lab[val_idx]]
    sc = Scaler(Y[fit_idx])
    if init_state is not None:
        model.load_state_dict(init_state)
    max_ep, pat = budget(len(fit_idx))
    log = log_to(f"{out_dir}/train.log")
    log(f"[{name}] n_fit={len(fit_idx)} n_val={len(val_idx)} lr={lr} seed={seed} params={n_params(model)}")
    model, info = fit(model, X[fit_idx], sc.fwd(Y[fit_idx]), X[val_idx], sc.fwd(Y[val_idx]), lr=lr,
                      max_epochs=max_ep, patience=pat, seed=seed, log=log)
    pred = sc.inv(predict(model, X[test_idx]))
    vpred = sc.inv(predict(model, X[val_idx]))
    if multi:
        res = {"test": {str(j): metrics(Y[test_idx, j], pred[:, j]) for j in range(Y.shape[1])},
               "val": {str(j): metrics(Y[val_idx, j], vpred[:, j]) for j in range(Y.shape[1])}}
    else:
        res = {"test": metrics(Y[test_idx], pred), "val": metrics(Y[val_idx], vpred)}
    res.update(name=name, seed=seed, lr=lr, n_fit=int(len(fit_idx)), n_val=int(len(val_idx)),
               params=n_params(model), **info, **(extra or {}))
    np.save(f"{out_dir}/{name}_pred.npy", pred)
    if save_model:
        torch.save(model.state_dict(), f"{out_dir}/{name}.pt")
    json.dump(res, open(jp, "w"), indent=1)
    t = res["test"]
    log(f"[{name}] done: " + (str({k: round(v['r2'], 4) for k, v in t.items()}) if multi else f"test r2 {t['r2']:.4f}"))
    return res
