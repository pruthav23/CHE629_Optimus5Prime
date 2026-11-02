import os
import numpy as np
import pandas as pd

UP = "/mnt/user-data/uploads"
UP2 = "/root/.claude/uploads/40b7f479-ba78-5b37-85f8-c5d028e32b05"
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHEMS = ["U", "psi", "m1psi"]
REP1 = {"U": f"{UP}/GSM3130435_egfp_unmod_1_csv.gz", "psi": f"{UP}/GSM3130437_egfp_pseudo_1_csv.gz",
        "m1psi": f"{UP}/GSM3130439_egfp_m1pseudo_1_csv.gz"}
REP2 = {"U": f"{UP}/GSM3130436_egfp_unmod_2_csv.gz", "psi": f"{UP}/GSM3130438_egfp_pseudo_2_csv.gz",
        "m1psi": f"{UP}/GSM3130440_egfp_m1pseudo_2_csv.gz"}
DESIGNED = f"{UP2}/8e89dc85-GSM3130443_designed_library.csv.gz"
CHEM_TABLE = f"{ROOT}/data/chem_table.csv"


def _chem_labels(path):
    d = pd.read_csv(path, usecols=lambda c: c in ("utr", "rl", "total", "total_reads"))
    dep = "total_reads" if "total_reads" in d else "total"
    d["utr"] = d.utr.str[:50]
    d = d.dropna(subset=["rl"]).sort_values(dep, ascending=False).drop_duplicates("utr")
    return d.set_index("utr").rl


def build_chem_table():
    """U split from the reproduction (data/train.csv, data/test.csv) with rep1 MRL for every chemistry
    and rep2 MRL (test rows only, for the replicate ceiling). Missing labels are NaN."""
    if os.path.exists(CHEM_TABLE):
        return pd.read_csv(CHEM_TABLE)
    tr = pd.read_csv(f"{ROOT}/data/train.csv").assign(split="train")
    te = pd.read_csv(f"{ROOT}/data/test.csv").assign(split="test")
    t = pd.concat([tr, te], ignore_index=True)[["utr", "split", "rl", "mfe"]].rename(columns={"rl": "rl_U"})
    for c in ["psi", "m1psi"]:
        t[f"rl_{c}"] = t.utr.map(_chem_labels(REP1[c]))
    for c in CHEMS:
        t[f"rep2_{c}"] = t.utr.map(_chem_labels(REP2[c]))
    t.to_csv(CHEM_TABLE, index=False)
    return t


def u_split():
    tr, te = pd.read_csv(f"{ROOT}/data/train.csv"), pd.read_csv(f"{ROOT}/data/test.csv")
    return tr, te


def nested_subsets(n_train, sizes, seed, out_dir):
    """Nested random subsets of training indices (smaller contained in larger); saved to disk."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(n_train)
    subs = {s: np.sort(perm[:s]) for s in sizes}
    os.makedirs(out_dir, exist_ok=True)
    np.savez(f"{out_dir}/subsets_seed{seed}.npz", **{str(k): v for k, v in subs.items()})
    return subs


def train_val(idx, seed, val_frac):
    """Split a training subset into fit / validation indices."""
    rng = np.random.default_rng(10_000 + seed)
    p = rng.permutation(idx)
    n_val = max(1, int(round(len(idx) * val_frac)))
    return np.sort(p[n_val:]), np.sort(p[:n_val])


def random_split(n, n_test, seed):
    rng = np.random.default_rng(20_000 + seed)
    p = rng.permutation(n)
    return np.sort(p[n_test:]), np.sort(p[:n_test])


AUTHORS_SNV = f"{ROOT}/../ext/human_5utr_modeling/human_5utrs/data/snv_phenotype_log_diff.csv"
PAIRS = f"{ROOT}/data/variant_pairs.csv"


def _ham(a, b):
    return sum(x != y for x, y in zip(a, b)) if len(a) == len(b) else 99


def build_pairs(min_reads=620, n_folds=5, seed=0):
    """(WT, variant) pairs from the designed library (unmodified U only).
    clinvar: designed SNV rows paired with the 'normal' row of the same genomic window (as the authors did);
             eval=True reproduces the authors' filter (variant reads >= 620, non-zero observed change).
    synth  : single-nucleotide synthesis errors paired with their designed parent; both >= min_reads; training only.
    fold   : CV fold by connected component of sequences linked through any pair (no WT/variant leakage)."""
    if os.path.exists(PAIRS):
        return pd.read_csv(PAIRS)
    d = pd.read_csv(DESIGNED, low_memory=False)
    d = d[d.library.isin(["human_utrs", "snv"])]
    snv = d[(d.library == "snv") & d.designed]
    nor = snv[snv.info4 == "normal"].set_index("info2")
    var = snv[(snv.info4 == "variant") & snv.info2.isin(nor.index)]
    a = pd.read_csv(AUTHORS_SNV, low_memory=False)
    a = a[a.info4 == "variant"].drop_duplicates("utr").set_index("utr")
    cv = pd.DataFrame({"wt_utr": nor.loc[var.info2, "utr"].values, "var_utr": var.utr.values,
                       "wt_rl": nor.loc[var.info2, "rl"].values, "var_rl": var.rl.values,
                       "wt_total": nor.loc[var.info2, "total"].values, "var_total": var.total.values,
                       "rs": var.info1.values, "source": "clinvar"})
    cv["gene"] = cv.var_utr.map(a.gene_symbol)
    cv["clin_sig"] = cv.var_utr.map(a.clin_sig)
    ev = a[(a.obs_diff != 0) & (a.total >= min_reads)].index
    cv["eval"] = cv.var_utr.isin(ev)
    cv["authors_log_pred_diff"] = cv.var_utr.map(a.log_pred_diff)
    des = d[d.designed].sort_values("total", ascending=False).drop_duplicates("utr").set_index("utr")
    err = d[~d.designed & d.mother.isin(des.index)].copy()
    err = err[[_ham(u, m) == 1 for u, m in zip(err.utr, err.mother)]]
    err = err[~err.utr.isin(des.index)].sort_values("total", ascending=False).drop_duplicates("utr")
    sy = pd.DataFrame({"wt_utr": err.mother.values, "var_utr": err.utr.values,
                       "wt_rl": des.loc[err.mother, "rl"].values, "var_rl": err.rl.values,
                       "wt_total": des.loc[err.mother, "total"].values, "var_total": err.total.values,
                       "rs": None, "source": "synth", "gene": None, "clin_sig": None, "eval": False,
                       "authors_log_pred_diff": np.nan})
    sy = sy[(sy.wt_total >= min_reads) & (sy.var_total >= min_reads)]
    p = pd.concat([cv, sy], ignore_index=True)
    p = p[(p.wt_rl > 0) & (p.var_rl > 0)].reset_index(drop=True)
    p["delta"] = np.log2(p.var_rl / p.wt_rl)
    parent = {}
    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    for w, v in zip(p.wt_utr, p.var_utr):
        parent[find(w)] = find(v)
    p["group"] = [find(w) for w in p.wt_utr]
    groups = p.group.unique()
    rng = np.random.default_rng(seed)
    fold_of = {g: i % n_folds for i, g in enumerate(rng.permutation(groups))}
    p["fold"] = p.group.map(fold_of)
    p["group"] = p.group.map({g: i for i, g in enumerate(groups)})
    p.to_csv(PAIRS, index=False)
    return p
