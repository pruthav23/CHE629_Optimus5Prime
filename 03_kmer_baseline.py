"""Paper's baseline: position-specific k-mer (k=1..6) linear regression with L2 regularisation,
alpha picked by 5-fold CV on the training set. Same train/test split as the CNN."""
import numpy as np, pandas as pd, json, itertools
import scipy.sparse as sp
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from optimus import r2

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")

def featurise(seqs, k, L=50):
    idx = {"".join(p): i for i, p in enumerate(itertools.product("ACGT", repeat=k))}
    V, P = len(idx), L - k + 1
    rows, cols = [], []
    for r, s in enumerate(seqs):
        for p in range(P):
            j = idx.get(s[p:p + k])
            if j is not None:
                rows.append(r); cols.append(p * V + j)
    return sp.csr_matrix((np.ones(len(rows), np.float32), (rows, cols)), shape=(len(seqs), P * V))

out = {}
for k in range(1, 7):
    Xtr, Xte = featurise(tr.utr, k), featurise(te.utr, k)
    ytr = tr.rl.values
    best = None
    for a in [0.1, 1, 10, 100, 1000]:
        cv = []
        for f_tr, f_va in KFold(5, shuffle=True, random_state=0).split(Xtr):
            m = Ridge(alpha=a, solver="sparse_cg" if k > 3 else "auto").fit(Xtr[f_tr], ytr[f_tr])
            cv.append(r2(ytr[f_va], m.predict(Xtr[f_va])))
        if best is None or np.mean(cv) > best[1]:
            best = (a, np.mean(cv))
    m = Ridge(alpha=best[0], solver="sparse_cg" if k > 3 else "auto").fit(Xtr, ytr)
    pred = m.predict(Xte)
    out[k] = {"alpha": best[0], "cv_r2": float(best[1]), "test_r2": float(r2(te.rl.values, pred)), "n_features": Xtr.shape[1]}
    print(k, out[k], flush=True)
    if k == max(out, key=lambda q: out[q]["test_r2"]):
        np.save("results/kmer_best_pred.npy", pred)
json.dump(out, open("results/kmer_baseline.json", "w"), indent=1)
