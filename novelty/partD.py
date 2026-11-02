"""Part D: attribution and motif discovery.

usage: python3 novelty/partD.py {chem|arch}
  chem : single-chemistry CNNs from Part A (U, psi, m1psi; seeds 0-2): IG + ISM on 2,000 shared test UTRs,
         TF-MoDISco on the U model, k-mer enrichment per chemistry, known-element annotation,
         structure-related sensitivity per chemistry.
  arch : CNN vs transformer (vs hybrid if trained), Part B full-data seed-0 models: attribution agreement.
"""
import common
import json, os, sys
import numpy as np
import pandas as pd
import torch
from scipy import stats
from attribution import integrated_gradients, ism, ism_contrib, top_kmers, annotate, seq_of
from data import build_chem_table, CHEMS
from models import CNN, build
from optimus import one_hot

OUT = "results/novelty/partD"
os.makedirs(OUT, exist_ok=True)
t = build_chem_table()
Y = t[[f"rl_{c}" for c in CHEMS]].values
test = np.where((t.split == "test") & ~np.isnan(Y).any(1))[0]
rng = np.random.default_rng(0)
S = np.sort(rng.choice(test, 2000, replace=False))
np.save(f"{OUT}/attribution_idx.npy", S)
X = one_hot(t.utr.values[S])
SEQS = t.utr.values[S]
NO_AUG = np.array(["ATG" not in s for s in SEQS])


def load_single(c, s):
    m = CNN(dropout=0.2); m.load_state_dict(torch.load(f"results/novelty/partA/single_{c}_s{s}.pt")); return m.eval()


def attributions(model, tag):
    p = f"{OUT}/attr_{tag}.npz"
    if os.path.exists(p):
        z = np.load(p); return z["ig"], z["hyp"], z["ism"]
    ig, hyp = integrated_gradients(model, X)
    d = np.concatenate([ism(model, X[i:i + 250]) for i in range(0, len(X), 250)])
    np.savez_compressed(p, ig=ig, hyp=hyp, ism=d)
    return ig, hyp, d


def gc_sensitivity(d):
    """Mean predicted change (model units, standardised MRL) for A/T->G/C substitutions in uAUG-free UTRs,
    excluding substitutions that create an ATG."""
    vals = []
    for n in np.where(NO_AUG)[0]:
        s = SEQS[n]
        for p in range(50):
            if s[p] in "AT":
                for b in (1, 2):
                    m = s[:p] + "ACGT"[b] + s[p + 1:]
                    if "ATG" not in m:
                        vals.append(d[n, p, b])
    return float(np.mean(vals))


def uaug_creation(d):
    """Mean predicted change when a single substitution creates an ATG (in uAUG-free UTRs)."""
    vals = []
    for n in np.where(NO_AUG)[0]:
        s = SEQS[n]
        for p in range(50):
            for b in range(4):
                if "ACGT"[b] != s[p] and "ATG" in s[:p] + "ACGT"[b] + s[p + 1:]:
                    vals.append(d[n, p, b])
    return float(np.mean(vals))


def run_modisco(hyp, tag):
    from modiscolite import tfmodisco
    pos, neg = tfmodisco.TFMoDISco(one_hot=X.astype(np.float32), hypothetical_contribs=hyp.astype(np.float32),
                                   sliding_window_size=6, flank_size=2, trim_to_window_size=10, initial_flank_to_add=2,
                                   final_flank_to_add=0, min_ic_windowsize=4, max_seqlets_per_metacluster=20000,
                                   target_seqlet_fdr=0.2, final_min_cluster_size=20, verbose=False)
    out = []
    for sign, pats in (("activating", pos), ("repressive", neg)):
        for i, pat in enumerate(pats or []):
            ppm = pat.sequence
            cons = "".join("ACGT"[j] for j in ppm.argmax(1))
            ic = (ppm * np.log2(np.clip(ppm, 1e-3, 1) / 0.25)).sum(1)
            core = "".join(cons[k] for k in range(len(cons)) if ic[k] > 0.5)
            out.append({"model": tag, "sign": sign, "pattern": i, "n_seqlets": len(pat.seqlets),
                        "consensus": cons, "high_ic_core": core, "known": annotate(core if core else cons),
                        "ppm": ppm.round(3).tolist()})
    return out


def stage_chem():
    res = {"n_seqs": int(len(S)), "n_uaug_free": int(NO_AUG.sum())}
    mfe = t.mfe.values[S]
    obs = {c: float(stats.pearsonr(mfe[~np.isnan(t[f"rl_{c}"].values[S])], t[f"rl_{c}"].values[S][~np.isnan(t[f"rl_{c}"].values[S])])[0])
           for c in CHEMS}
    allt = t.iloc[test]
    res["observed_r_MRL_vs_MFE_full_test"] = {c: float(stats.pearsonr(allt.mfe, allt[f"rl_{c}"])[0]) for c in CHEMS}
    res["observed_r_MRL_vs_MFE_attr_subset"] = obs
    per = {c: {"gc_sens": [], "uaug_create": [], "pred_r_vs_mfe_uaug_free": []} for c in CHEMS}
    contribs = {}
    for c in CHEMS:
        for s in (0, 1, 2):
            m = load_single(c, s)
            ig, hyp, d = attributions(m, f"{c}_s{s}")
            per[c]["gc_sens"].append(gc_sensitivity(d))
            per[c]["uaug_create"].append(uaug_creation(d))
            with torch.no_grad():
                pr = m(torch.as_tensor(X)).numpy()
            per[c]["pred_r_vs_mfe_uaug_free"].append(float(stats.pearsonr(mfe[NO_AUG], pr[NO_AUG])[0]))
            if s == 0:
                contribs[c] = (ig, hyp, d)
    res["per_chem"] = {c: {k: (float(np.mean(v)), float(np.std(v))) for k, v in per[c].items()} for c in CHEMS}
    res["per_chem"]["note"] = ("model units = standardised MRL of that chemistry; gc_sens = mean change for A/T->G/C "
                               "substitutions in uAUG-free UTRs (no ATG created); uaug_create = mean change when a substitution "
                               "creates an ATG; ratio gc_sens/uaug_create compares structure vs uAUG reliance")
    for c in CHEMS:
        g, u = res["per_chem"][c]["gc_sens"][0], res["per_chem"][c]["uaug_create"][0]
        res["per_chem"][c]["gc_over_uaug"] = float(g / u)
    pp = {c: ism_contrib(contribs[c][2], X).sum(-1) for c in CHEMS}
    res["attr_agreement_ism_per_seq_r"] = {f"{a}_vs_{b}": float(np.nanmean([stats.pearsonr(pp[a][i], pp[b][i])[0] for i in range(len(S))]))
                                           for a, b in (("U", "psi"), ("U", "m1psi"), ("psi", "m1psi"))}
    kms = []
    for c in CHEMS:
        for sign, lab in ((-1, "repressive"), (1, "activating")):
            df = top_kmers(ism_contrib(contribs[c][2], X), X, k=6, top_frac=0.05, sign=sign).head(15)
            df["chem"], df["sign"] = c, lab
            df["known"] = [";".join(annotate(k)) for k in df.kmer]
            kms.append(df)
    pd.concat(kms).to_csv(f"{OUT}/kmer_enrichment_by_chem.csv", index=False)
    motifs = []
    for c in CHEMS:
        try:
            motifs += run_modisco(contribs[c][1], f"{c}_s0")
        except Exception as e:
            motifs.append({"model": f"{c}_s0", "error": repr(e)})
    json.dump(motifs, open(f"{OUT}/modisco_motifs.json", "w"), indent=1)
    res["modisco_summary"] = [{k: v for k, v in mm.items() if k != "ppm"} for mm in motifs]
    json.dump(res, open(f"{OUT}/chem_attribution.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "modisco_summary"}, indent=1))
    for mm in res["modisco_summary"]:
        print(mm)


def stage_arch():
    B = "results/novelty/partB/full"
    models = {}
    for arch in ("cnn", "transformer", "hybrid"):
        cf = f"{B}/{arch}_choice.json"
        if not os.path.exists(cf):
            continue
        best = json.load(open(cf))["chosen"]
        pt = f"{B}/{arch}_full_{best}_s0.pt"
        if os.path.exists(pt):
            do = float(best.split("_do")[1])
            m = build(arch, do); m.load_state_dict(torch.load(pt)); models[arch] = m.eval()
    res, pp, igp = {}, {}, {}
    for a, m in models.items():
        ig, hyp, d = attributions(m, f"arch_{a}")
        pp[a], igp[a] = ism_contrib(d, X).sum(-1), ig.sum(-1)
        res[f"{a}_uaug_create"], res[f"{a}_gc_sens"] = uaug_creation(d), gc_sensitivity(d)
    names = list(models)
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a, b = names[i], names[j]
            res[f"{a}_vs_{b}"] = {
                "ism_per_seq_r": float(np.nanmean([stats.pearsonr(pp[a][n], pp[b][n])[0] for n in range(len(S))])),
                "ig_per_seq_r": float(np.nanmean([stats.pearsonr(igp[a][n], igp[b][n])[0] for n in range(len(S))])),
                "top5_position_overlap": float(np.mean([len(set(np.argsort(-np.abs(pp[a][n]))[:5]) & set(np.argsort(-np.abs(pp[b][n]))[:5])) / 5
                                                        for n in range(len(S))]))}
    if "cnn" in models:
        res["within_cnn_seed_agreement_note"] = "see chem stage: CNN seeds share architecture; compare to cross-arch r"
        m1 = load_single("U", 1)
        _, _, d1 = attributions(m1, "U_s1")
        p1 = ism_contrib(d1, X).sum(-1)
        res["cnn_seed0_vs_seed1_ism_per_seq_r"] = float(np.nanmean([stats.pearsonr(pp["cnn"][n], p1[n])[0] for n in range(len(S))]))
    kms = []
    for a in models:
        for sign, lab in ((-1, "repressive"), (1, "activating")):
            _, _, d = attributions(models[a], f"arch_{a}")
            df = top_kmers(ism_contrib(d, X), X, k=6, top_frac=0.05, sign=sign).head(15)
            df["arch"], df["sign"] = a, lab
            df["known"] = [";".join(annotate(k)) for k in df.kmer]
            kms.append(df)
    pd.concat(kms).to_csv(f"{OUT}/kmer_enrichment_by_arch.csv", index=False)
    json.dump(res, open(f"{OUT}/arch_attribution.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    {"chem": stage_chem, "arch": stage_arch}[sys.argv[1]]()
    print(f"PART D {sys.argv[1]} DONE", flush=True)
