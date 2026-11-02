"""Benchmark models + shared evaluation for the Optimus 5-Prime comparison.

Benchmarks
  1. k-mer linear : position-specific k-mer ridge regression (the paper's own baseline)
  2. LightGBM     : gradient-boosted trees on hand-engineered UTR features
                    (uAUG counts by type, Kozak context, stop codons, GC, MFE, 1-3-mer counts)
  3. BiLSTM       : bidirectional LSTM on one-hot sequence (a different deep-learning family)
Model under test
  Optimus 5-Prime CNN (optimus.py)
"""
import itertools
from collections import Counter
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from scipy import stats
from sklearn.metrics import (accuracy_score, confusion_matrix, mean_absolute_error,
                             mean_squared_error, precision_recall_fscore_support)

STOPS = ("TAA", "TAG", "TGA")


# ----------------------------------------------------------------------------- metrics
def tertile_edges(y_train):
    """Class boundaries (low / medium / high MRL) from the TRAINING labels only."""
    return np.quantile(y_train, [1 / 3, 2 / 3])


def to_class(y, edges):
    return np.digitize(y, edges)          # 0 = low, 1 = medium, 2 = high


def evaluate(y_obs, y_pred, edges):
    c_obs, c_pred = to_class(y_obs, edges), to_class(y_pred, edges)
    p, r, f, _ = precision_recall_fscore_support(c_obs, c_pred, average="macro", zero_division=0)
    return {
        "r2": float(stats.pearsonr(y_obs, y_pred)[0] ** 2),
        "spearman": float(stats.spearmanr(y_obs, y_pred)[0]),
        "rmse": float(np.sqrt(mean_squared_error(y_obs, y_pred))),
        "mae": float(mean_absolute_error(y_obs, y_pred)),
        "accuracy": float(accuracy_score(c_obs, c_pred)),
        "precision_macro": float(p), "recall_macro": float(r), "f1_macro": float(f),
        "confusion_matrix": confusion_matrix(c_obs, c_pred, labels=[0, 1, 2]).tolist(),
    }


# ----------------------------------------------------------------------------- BiLSTM
class BiLSTM(nn.Module):
    """One-hot (N, L, 4) -> BiLSTM(64) -> [mean-pool ; max-pool] -> Dense(40, ReLU) -> Dropout 0.2 -> 1."""
    def __init__(self, hidden=64, dense=40, dropout=0.2):
        super().__init__()
        self.lstm = nn.LSTM(4, hidden, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(4 * hidden, dense), nn.ReLU(), nn.Dropout(dropout), nn.Linear(dense, 1))

    def forward(self, x):
        h, _ = self.lstm(x)
        return self.head(torch.cat([h.mean(1), h.amax(1)], dim=1)).squeeze(-1)


# ----------------------------------------------------------------------------- k-mer linear
def kmer_features(seqs, k, L=50):
    idx = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=k))}
    V, P = len(idx), L - k + 1
    rows, cols = [], []
    for r, s in enumerate(seqs):
        for p in range(P):
            j = idx.get(s[p:p + k])
            if j is not None:
                rows.append(r); cols.append(p * V + j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(seqs), P * V))


# ----------------------------------------------------------------------------- hand-made features
_KMERS = ["".join(p) for k in (1, 2, 3) for p in itertools.product("ACGT", repeat=k)]


def utr_features(seq, mfe):
    """Interpretable features a biologist would compute by hand (no position-specific one-hot)."""
    n = len(seq)
    f = {"mfe": mfe, "gc": (seq.count("G") + seq.count("C")) / n}
    atgs = [p for p in range(n - 2) if seq[p:p + 3] == "ATG"]
    cnt = {"inframe": 0, "uorf": 0, "oorf": 0}
    best_kozak = {"inframe": 0.0, "uorf": 0.0, "oorf": 0.0}
    for p in atgs:
        has_stop = any(seq[q:q + 3] in STOPS for q in range(p + 3, n - 2, 3))
        kind = "uorf" if has_stop else ("inframe" if (n - p) % 3 == 0 else "oorf")
        cnt[kind] += 1
        m3 = seq[p - 3] if p >= 3 else "A"
        p4 = seq[p + 3] if p + 3 < n else "G"
        best_kozak[kind] = max(best_kozak[kind], (m3 in "AG") + (p4 == "G"))
    for k in cnt:
        f[f"n_uaug_{k}"] = cnt[k]; f[f"kozak_{k}"] = best_kozak[k]
    f["n_uaug"] = len(atgs)
    f["first_uaug_dist"] = (n - atgs[0]) if atgs else 0
    f["last_uaug_dist"] = (n - atgs[-1]) if atgs else 0
    for fr in range(3):
        f[f"n_stop_frame{fr}"] = sum(seq[q:q + 3] in STOPS for q in range(fr, n - 2, 3))
    c = Counter(seq[i:i + k] for k in (1, 2, 3) for i in range(n - k + 1))
    for km in _KMERS:
        f[f"cnt_{km}"] = c.get(km, 0)
    return f


def feature_matrix(seqs, mfes):
    import pandas as pd
    return pd.DataFrame([utr_features(s, m) for s, m in zip(seqs, mfes)])
