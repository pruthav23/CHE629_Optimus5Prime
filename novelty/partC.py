"""Part C: paired variant-effect prediction on human 5'UTR ClinVar SNVs (unmodified U data only).

Methods (target: log2(MRL_var / MRL_wt)); all start from the Part A U CNN (random library, seeds 0-2):
  subtract     : paper approach, predict each sequence with the random-library CNN and subtract (no human data)
  subtract_ft  : same, after fine-tuning the CNN on absolute MRL of human sequences in the training folds
                 (controls for in-domain data, so the paired objective is tested separately from data access)
  siamese      : shared encoder on (WT, variant), head on [h_var - h_wt, h_wt * h_var], trained on the delta
  siamese_aux  : same + auxiliary loss on both absolute MRLs (weight 0.5)
5-fold CV grouped by connected WT/variant component; training pairs = ClinVar + synthesis-error pairs of the
training folds; evaluation = the authors' 1,584 ClinVar pairs, pooled out-of-fold. 3 seeds per method.
"""
import json, os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from common import log_to
from data import build_pairs, build_chem_table, train_val, DESIGNED
from evaluate import delta_metrics
from models import CNN, Siamese
from optimus import one_hot
from train import fit, predict, Scaler
from partC_common import pretrained, mrl

OUT = "results/novelty/partC"
os.makedirs(OUT, exist_ok=True)
SEEDS, FOLDS, LR, AUX_W = [0, 1, 2], 5, 3e-4, 0.5
log = log_to(f"{OUT}/train.log")
P = build_pairs()
XW, XV = one_hot(P.wt_utr), one_hot(P.var_utr)
EV = P["eval"].values


class PairWrap(nn.Module):
    def __init__(self, s):
        super().__init__(); self.s = s

    def forward(self, x):
        d, w, v = self.s(x[:, 0], x[:, 1])
        return torch.stack([d, w, v], 1)


def pair_loss(aux):
    def f(pred, y):
        return ((pred[:, 0] - y[:, 0]) ** 2).mean() + aux * (((pred[:, 1:] - y[:, 1:]) ** 2).mean())
    return f


def group_val(idx, seed):
    g = P.group.values[idx]
    ug = np.unique(g)
    rng = np.random.default_rng(30_000 + seed)
    vg = set(rng.choice(ug, max(1, len(ug) // 10), replace=False))
    m = np.array([x in vg for x in g])
    return idx[~m], idx[m]


def human_abs_table():
    d = pd.read_csv(DESIGNED, low_memory=False)
    d = d[d.library.isin(["human_utrs", "snv"]) & (d.total >= 620) & (d.rl > 0)]
    return d.sort_values("total", ascending=False).drop_duplicates("utr")[["utr", "rl"]].reset_index(drop=True)


def run_fold_methods(seed):
    path = f"{OUT}/oof_seed{seed}.csv"
    if os.path.exists(path):
        return pd.read_csv(path)
    base, sc = pretrained(seed)
    oof = pd.DataFrame(index=P.index)
    oof["subtract"] = np.log2(mrl(base, sc, XV) / mrl(base, sc, XW))
    H = human_abs_table()
    scales = {}
    XH = one_hot(H.utr)
    for k in range(FOLDS):
        te = np.where(P.fold.values == k)[0]
        tr = np.where(P.fold.values != k)[0]
        test_seqs = set(P.wt_utr.values[te]) | set(P.var_utr.values[te])
        # subtract_ft: fine-tune on absolute MRL of human sequences outside the test fold
        hi = np.array([u not in test_seqs for u in H.utr])
        hidx = np.where(hi)[0]
        hf, hv = train_val(hidx, seed + 100 * k, 0.10)
        m, _ = pretrained(seed)
        hsc = Scaler(H.rl.values[hf])
        log(f"[seed {seed} fold {k}] subtract_ft n_fit={len(hf)}")
        m, info = fit(m, XH[hf], hsc.fwd(H.rl.values[hf]), XH[hv], hsc.fwd(H.rl.values[hv]), lr=LR, max_epochs=200,
                      patience=10, seed=seed, log=None)
        pw, pv = np.clip(hsc.inv(predict(m, XW[te])), 0.1, None), np.clip(hsc.inv(predict(m, XV[te])), 0.1, None)
        oof.loc[te, "subtract_ft"] = np.log2(pv / pw)
        # siamese variants
        f, v = group_val(tr, seed + 100 * k)
        dsd = P.delta.values[f].std()
        scales[k] = float(dsd)
        asc = Scaler(np.concatenate([P.wt_rl.values[f], P.var_rl.values[f]]))
        Y = np.stack([P.delta.values / dsd, asc.fwd(P.wt_rl.values), asc.fwd(P.var_rl.values)], 1).astype(np.float32)
        XP = np.stack([XW, XV], 1)
        for name, aux in (("siamese", 0.0), ("siamese_aux", AUX_W)):
            trunk, _ = pretrained(seed)
            net = PairWrap(Siamese(trunk))
            net, info = fit(net, XP[f], Y[f], XP[v], Y[v], lr=LR, max_epochs=200, patience=10, seed=seed,
                            loss_fn=pair_loss(aux), log=None)
            log(f"[seed {seed} fold {k}] {name} n_fit={len(f)} best_epoch={info['best_epoch']} val_loss={info['val_loss']:.4f}")
            oof.loc[te, name] = predict(net, XP[te])[:, 0] * dsd
            oof.loc[te, f"{name}_val_loss"] = info["val_loss"]
            torch.save(net.s.state_dict(), f"{OUT}/{name}_seed{seed}_fold{k}.pt")
    json.dump(scales, open(f"{OUT}/delta_scale_seed{seed}.json", "w"))
    oof.to_csv(path, index=False)
    return oof


METHODS = ["subtract", "subtract_ft", "siamese", "siamese_aux"]
per_seed = {}
for s in SEEDS:
    oof = run_fold_methods(s)
    per_seed[s] = {m: delta_metrics(P.delta.values[EV], oof[m].values[EV]) for m in METHODS}
    per_seed[s].update({f"{m}_synth": delta_metrics(P.delta.values[P.source == "synth"], oof[m].values[P.source == "synth"])
                        for m in METHODS})
    log(f"seed {s}: " + str({m: round(per_seed[s][m]["r2"], 3) for m in METHODS}))
keys = ["r2", "pearson", "spearman", "rmse", "direction_acc", "r2_high", "pearson_high", "direction_acc_high", "auroc_high"]
summary = {}
for m in METHODS + [f"{m}_synth" for m in METHODS]:
    summary[m] = {k: (float(np.mean([per_seed[s][m][k] for s in SEEDS])), float(np.std([per_seed[s][m][k] for s in SEEDS])))
                  for k in keys if k in per_seed[SEEDS[0]][m]}
    summary[m]["n"], summary[m]["n_high"] = per_seed[SEEDS[0]][m]["n"], per_seed[SEEDS[0]][m]["n_high"]
summary["authors_retrained_subtract"] = delta_metrics(P.delta.values[EV], P.authors_log_pred_diff.values[EV])
json.dump({"summary": summary, "per_seed": per_seed}, open(f"{OUT}/metrics.json", "w"), indent=1)
print(json.dumps({m: {k: v for k, v in summary[m].items() if k in ("r2", "direction_acc", "direction_acc_high", "auroc_high")}
                  for m in METHODS + ["authors_retrained_subtract"]}, indent=1))
print("PART C CV DONE", flush=True)
