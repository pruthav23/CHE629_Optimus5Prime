"""Fig 4a: predict MRL of human 5'UTRs + ClinVar SNVs (designed library), top 25k by read depth.
Paper's number (0.81) uses a *retrained* model; we report the original-model numbers too."""
import numpy as np, pandas as pd, json, sys, torch
from optimus import OptimusCNN, load_keras_weights, predict, r2, one_hot
D = "/root/.claude/uploads/40b7f479-ba78-5b37-85f8-c5d028e32b05/8e89dc85-GSM3130443_designed_library.csv.gz"
df = pd.read_csv(D, low_memory=False)
hu = df[df.library.isin(["human_utrs", "snv"])].sort_values("total", ascending=False).iloc[:25000].reset_index(drop=True)
hu.to_csv("data/human_top25k.csv", index=False)
x = one_hot(hu.utr)
tr = np.load("data/egfp_unmod_1.npz")["ytr"]; mu, sd = tr.mean(), tr.std()
out = {}
m = load_keras_weights(OptimusCNN(), "/home/claude/ext/human_5utr_modeling/modeling/saved_models/main_MRL_model.hdf5")
out["authors_main_model"] = float(r2(hu.rl, predict(m, x)))
for path in sys.argv[1:]:
    m = OptimusCNN(); m.load_state_dict(torch.load(path)); p = predict(m, x) * sd + mu
    out[path] = float(r2(hu.rl, p)); np.save(path.replace(".pt", "_human_pred.npy"), p)
print(out); json.dump(out, open("results/human_utrs.json", "w"), indent=1)
