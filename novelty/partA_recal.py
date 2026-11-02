"""Part A calibration check: fit y = a + b * pred on each model's VALIDATION predictions, apply to test, report RMSE
before and after (r² is unchanged by a linear map)."""
import json
import numpy as np
import torch
import common
from data import build_chem_table, train_val, CHEMS
from models import CNN, MultiChem
from optimus import one_hot
from train import predict, Scaler

OUT = "results/novelty/partA"
t = build_chem_table()
Y = t[[f"rl_{c}" for c in CHEMS]].values
train_idx = np.where(t.split == "train")[0]
test_idx = np.load(f"{OUT}/test_idx.npy")
best = json.load(open(f"{OUT}/variant_choice.json"))["chosen"]
res = {}
for s in (0, 1, 2):
    f, v = train_val(train_idx, s, 0.10)
    Xv, Xt = one_hot(t.utr.values[v]), one_hot(t.utr.values[test_idx])
    for j, c in enumerate(CHEMS):
        for kind in ("single", "multi"):
            if kind == "single":
                m = CNN(dropout=0.2); m.load_state_dict(torch.load(f"{OUT}/single_{c}_s{s}.pt"))
                lab = ~np.isnan(Y[f, j]); sc = Scaler(Y[f[lab], j])
                pv, pt = sc.inv(predict(m, Xv)), sc.inv(predict(m, Xt))
            else:
                m = MultiChem(CNN(dropout=0.2), mode=best); m.load_state_dict(torch.load(f"{OUT}/multi_{best}_s{s}.pt"))
                sc = Scaler(Y[f]); pv, pt = sc.inv(predict(m, Xv))[:, j], sc.inv(predict(m, Xt))[:, j]
            ok = ~np.isnan(Y[v, j])
            b, a = np.polyfit(pv[ok], Y[v, j][ok], 1)
            yt = Y[test_idx, j]
            res.setdefault(f"{kind}_{c}", []).append({"rmse_raw": float(np.sqrt(np.mean((pt - yt) ** 2))),
                                                      "rmse_recal": float(np.sqrt(np.mean((a + b * pt - yt) ** 2))),
                                                      "slope": float(b), "intercept": float(a)})
summary = {k: {m: (float(np.mean([x[m] for x in v])), float(np.std([x[m] for x in v]))) for m in v[0]} for k, v in res.items()}
json.dump({"summary": summary, "per_seed": res}, open(f"{OUT}/recalibration.json", "w"), indent=1)
for k, v in summary.items():
    print(f"{k:14s} RMSE raw {v['rmse_raw'][0]:.3f} ± {v['rmse_raw'][1]:.3f} -> val-recalibrated {v['rmse_recal'][0]:.3f} ± {v['rmse_recal'][1]:.3f} (slope {v['slope'][0]:.2f})")
