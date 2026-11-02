import copy
import time
import numpy as np
import torch
import torch.nn as nn


def masked_mse(pred, y):
    m = ~torch.isnan(y)
    return ((pred[m] - y[m]) ** 2).mean()


def fit(model, x_fit, y_fit, x_val, y_val, lr=1e-3, max_epochs=15, patience=2, batch_size=128, seed=0,
        loss_fn=masked_mse, log=print):
    """Adam + early stopping on validation loss; restores the best epoch.
    y may be (N,) or (N, C) with NaN for missing labels (masked)."""
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    X, Y = torch.as_tensor(x_fit), torch.as_tensor(y_fit, dtype=torch.float32)
    best, best_state, best_ep, bad, hist = np.inf, None, 0, 0, []
    t0 = time.time()
    for ep in range(1, max_epochs + 1):
        model.train()
        perm = torch.randperm(len(X), generator=g)
        for i in range(0, len(X), batch_size):
            idx = perm[i:i + batch_size]
            loss = loss_fn(model(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
        vl = float(loss_fn(torch.as_tensor(predict(model, x_val)), torch.as_tensor(y_val, dtype=torch.float32)))
        hist.append(vl)
        if vl < best - 1e-6:
            best, best_state, best_ep, bad = vl, copy.deepcopy(model.state_dict()), ep, 0
        else:
            bad += 1
        if log:
            log(f"    ep {ep:3d} val_loss {vl:.4f} best {best:.4f}@{best_ep} ({time.time() - t0:.0f}s)")
        if bad >= patience:
            break
    model.load_state_dict(best_state)
    return model, {"best_epoch": best_ep, "epochs_run": ep, "val_loss": best, "val_hist": hist,
                   "minutes": round((time.time() - t0) / 60, 2)}


@torch.no_grad()
def predict(model, x, batch_size=4096):
    model.eval()
    return torch.cat([model(torch.as_tensor(x[i:i + batch_size])) for i in range(0, len(x), batch_size)]).numpy()


class Scaler:
    """Per-column standardisation fitted on the fit split (NaN-aware)."""
    def __init__(self, y):
        self.mu, self.sd = np.nanmean(y, 0), np.nanstd(y, 0)

    def fwd(self, y):
        return ((y - self.mu) / self.sd).astype(np.float32)

    def inv(self, z):
        return z * self.sd + self.mu
