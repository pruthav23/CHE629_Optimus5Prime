"""Retrained Optimus 5-Prime (paper Fig 3d / Supp Fig 13): random-library train set + designed-library
sub-libraries (GA-evolved, poly-N, controls...). Human UTRs and SNVs are EXCLUDED so Fig 4a stays a held-out test."""
import numpy as np, pandas as pd, torch, json
from optimus import OptimusCNN, train, predict, r2, one_hot
D = "/root/.claude/uploads/40b7f479-ba78-5b37-85f8-c5d028e32b05/8e89dc85-GSM3130443_designed_library.csv.gz"
d = np.load("data/egfp_unmod_1.npz")
des = pd.read_csv(D, low_memory=False)
des = des[~des.library.isin(["human_utrs", "snv", "mcherry"]) & des.rl.notna() & (des.utr.str.len() >= 50)]
des = des[des.total >= 200]                          # crude read-depth filter (~similar to random-library cutoff)
print("designed seqs added:", len(des))
xtr = np.concatenate([d["xtr"], one_hot(des.utr)]); ytr = np.concatenate([d["ytr"], des.rl.values])
mu, sd = d["ytr"].mean(), d["ytr"].std()             # keep same scaler as main model
m = train(OptimusCNN(), xtr, (ytr - mu) / sd, epochs=3, seed=1337)
torch.save(m.state_dict(), "results/cnn_retrained.pt")
res = {"random_test_r2": float(r2(d["yte"], predict(m, d["xte"]) * sd + mu)), "n_designed_added": int(len(des))}
print(res); json.dump(res, open("results/retrained.json", "w"))
