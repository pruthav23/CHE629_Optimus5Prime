"""Did the networks learn the rules we planted? In-silico ATG insertion on the synthetic data.

Take 400 test UTRs with no uAUG, write 'ATG' at every position 0..47, and compare
  true change in MRL (simulator, MFE recomputed)   vs   predicted change (model).
Also report each model's mean prediction for single-uAUG test UTRs per planted class.
"""
import json, numpy as np, pandas as pd, torch
from synth import true_mrl, mfe_batch, uaug_list, classify_uaug
from optimus import OptimusCNN, predict, one_hot
from bench import BiLSTM

tr, te = pd.read_csv("data/synth/train.csv"), pd.read_csv("data/synth/test.csv")
mu, sd = tr.rl.mean(), tr.rl.std()
models = {}
for name, cls in (("cnn", OptimusCNN), ("bilstm", BiLSTM)):
    m = cls(); m.load_state_dict(torch.load(f"results/bench/synth_{name}_s1337.pt")); models[name] = m

base = te[~te.utr.str.contains("ATG")].sample(400, random_state=0).reset_index(drop=True)
rows, mut = [], []
for i, s in enumerate(base.utr):
    for p in range(48):
        m = s[:p] + "ATG" + s[p + 3:]
        if m.count("ATG") == 1:              # exactly one uAUG after insertion
            rows.append((i, p, m)); mut.append(m)
mfe_mut = mfe_batch(mut)
df = pd.DataFrame(rows, columns=["i", "pos", "utr"])
df["dist"] = 50 - df.pos
df["kind"] = [classify_uaug(s, uaug_list(s)[0]) for s in df.utr]
df["true_delta"] = [true_mrl(s, f) for s, f in zip(df.utr, mfe_mut)] - \
                   np.array([true_mrl(base.utr[i], base.mfe[i]) for i in df.i])
x_base, x_mut = one_hot(base.utr), one_hot(df.utr)
out = {}
for name, m in models.items():
    pb = predict(m, x_base) * sd + mu
    df[f"{name}_delta"] = predict(m, x_mut) * sd + mu - pb[df.i]
    out[name] = {"r_true_vs_pred_delta": float(np.corrcoef(df.true_delta, df[f"{name}_delta"])[0, 1]),
                 "mean_delta_by_kind": df.groupby("kind")[f"{name}_delta"].mean().round(3).to_dict()}
out["true"] = {"mean_delta_by_kind": df.groupby("kind").true_delta.mean().round(3).to_dict(),
               "n_by_kind": df.kind.value_counts().to_dict()}
df.to_csv("results/bench/synth_ism.csv", index=False)
json.dump(out, open("results/bench/synth_ism.json", "w"), indent=1)
print(json.dumps(out, indent=1))
