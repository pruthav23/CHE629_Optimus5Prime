import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def metrics(y, p):
    m = ~(np.isnan(y) | np.isnan(p))
    y, p = y[m], p[m]
    r = stats.pearsonr(y, p)[0]
    return {"r2": float(r ** 2), "pearson": float(r), "spearman": float(stats.spearmanr(y, p)[0]),
            "rmse": float(np.sqrt(np.mean((y - p) ** 2))), "n": int(len(y))}


def delta_metrics(obs, pred, hi=0.5):
    """Variant-effect metrics on log2(MRL_var / MRL_wt)."""
    out = metrics(obs, pred)
    out["direction_acc"] = float(np.mean(np.sign(obs) == np.sign(pred)))
    h = np.abs(obs) > hi
    out["n_high"] = int(h.sum())
    if h.sum() >= 3:
        out["r2_high"] = float(stats.pearsonr(obs[h], pred[h])[0] ** 2)
        out["pearson_high"] = float(stats.pearsonr(obs[h], pred[h])[0])
        out["direction_acc_high"] = float(np.mean(np.sign(obs[h]) == np.sign(pred[h])))
    if 0 < h.sum() < len(h):
        out["auroc_high"] = float(roc_auc_score(h, np.abs(pred)))
    return out
