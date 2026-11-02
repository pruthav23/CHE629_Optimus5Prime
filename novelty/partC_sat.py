"""Part C: in-silico saturation mutagenesis of the CPOX, TMEM127 and RPL5 5'UTRs (paper Fig. 4c).
Every single-nucleotide substitution (150) is scored; the ClinVar variant is ranked by predicted log2 change
(most negative = rank 1, since all three highlighted variants lower MRL). Siamese models are the CV-fold models
in which that gene's pair was held out; scores are averaged over 3 seeds."""
import common
import json
import numpy as np
import torch
from data import build_pairs
from models import CNN, Siamese
from optimus import one_hot
from partC_common import pretrained, mrl

OUT = "results/novelty/partC"
P = build_pairs()
TARGETS = {"CPOX": "rs867711777", "TMEM127": "rs121908813", "RPL5": "rs376208311"}


def mutants(wt):
    seqs, where = [], []
    for p in range(50):
        for b in "ACGT":
            if b != wt[p]:
                seqs.append(wt[:p] + b + wt[p + 1:]); where.append((p, b))
    return seqs, where


res = {}
for gene, rs in TARGETS.items():
    rows = P[(P.rs == rs) & P["eval"]]
    wt = rows.wt_utr.iloc[0]
    fold = int(rows.fold.iloc[0])
    seqs, where = mutants(wt)
    xw, xm = one_hot([wt] * len(seqs)), one_hot(seqs)
    preds = {"subtract": [], "siamese": [], "siamese_aux": []}
    for s in (0, 1, 2):
        base, sc = pretrained(s)
        preds["subtract"].append(np.log2(mrl(base, sc, xm) / mrl(base, sc, xw)))
        dsd = json.load(open(f"{OUT}/delta_scale_seed{s}.json"))[str(fold)]
        for name in ("siamese", "siamese_aux"):
            net = Siamese(CNN(dropout=0.2)); net.load_state_dict(torch.load(f"{OUT}/{name}_seed{s}_fold{fold}.pt")); net.eval()
            with torch.no_grad():
                preds[name].append(net(torch.as_tensor(xw), torch.as_tensor(xm))[0].numpy() * dsd)
    preds = {k: np.mean(v, 0) for k, v in preds.items()}
    out = {"wt": wt, "fold": fold, "variants": []}
    for _, r in rows.iterrows():
        diff = [i for i in range(50) if r.var_utr[i] != wt[i]]
        entry = {"observed_log2": float(r.delta), "positions": diff}
        if len(diff) == 1:
            k = seqs.index(r.var_utr)
            p, b = where[k]
            entry["change"] = f"{wt[p]}{p - 50}{b} (position relative to CDS start)"
            for m, v in preds.items():
                order = np.argsort(v)
                entry[m] = {"pred_log2": float(v[k]), "rank_most_negative": int(np.where(order == k)[0][0]) + 1,
                            "rank_abs": int(np.where(np.argsort(-np.abs(v)) == k)[0][0]) + 1, "of": len(v)}
        out["variants"].append(entry)
    out["top5_subtract"] = [f"{wt[where[i][0]]}{where[i][0] - 50}{where[i][1]}" for i in np.argsort(preds["subtract"])[:5]]
    out["top5_siamese"] = [f"{wt[where[i][0]]}{where[i][0] - 50}{where[i][1]}" for i in np.argsort(preds["siamese"])[:5]]
    np.savez(f"{OUT}/saturation_{gene}.npz", where=np.array([f"{p}{b}" for p, b in where]), **preds)
    res[gene] = out
json.dump(res, open(f"{OUT}/saturation_mutagenesis.json", "w"), indent=1)
print(json.dumps(res, indent=1))
