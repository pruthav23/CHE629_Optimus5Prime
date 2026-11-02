"""Collect saved results into tables (markdown/CSV) and figures. usage: python3 novelty/report.py {A|B|C|D|all}"""
import glob, json, os, pickle, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R, F = "results/novelty", "figures/novelty"
os.makedirs(F, exist_ok=True)
INK, MUTED, GRID = "#1f1f1e", "#73726c", "#e4e3dd"
COL = {"cnn": "#2a78d6", "transformer": "#eb6834", "hybrid": "#1baf7a",
       "scratch": "#2a78d6", "finetune": "#eb6834", "multi": "#1baf7a"}
CHEMS, CLAB = ["U", "psi", "m1psi"], {"U": "U", "psi": "Ψ", "m1psi": "m1Ψ"}
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11.5,
                     "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})


def ms(v, d=3):
    return f"{np.mean(v):.{d}f} ± {np.std(v):.{d}f}" if len(v) > 1 else f"{v[0]:.{d}f}"


def r2(a, b):
    m = ~(np.isnan(a) | np.isnan(b))
    return np.corrcoef(a[m], b[m])[0, 1] ** 2


# ------------------------------------------------------------------------------------------------ Part A
def part_a():
    sys.path[:0] = [".", "novelty"]
    from data import build_chem_table
    t = build_chem_table()
    idx = np.load(f"{R}/partA/test_idx.npy")
    Y = {c: t[f"rl_{c}"].values[idx] for c in CHEMS}
    choice = json.load(open(f"{R}/partA/variant_choice.json"))
    best = choice["chosen"]
    rows, mats = [], {}
    for c in CHEMS:
        runs = sorted(glob.glob(f"{R}/partA/single_{c}_s?_pred.npy"))
        mats[f"single {CLAB[c]}"] = {c2: [r2(Y[c2], np.load(p)) for p in runs] for c2 in CHEMS}
    runs = sorted(glob.glob(f"{R}/partA/multi_{best}_s?_pred.npy"))
    for j, c in enumerate(CHEMS):
        mats[f"multi head {CLAB[c]}"] = {c2: [r2(Y[c2], np.load(p)[:, j]) for p in runs] for c2 in CHEMS}
    ceil = {c: np.sqrt(r2(t[f"rl_{c}"].values[idx], t[f"rep2_{c}"].values[idx])) for c in CHEMS}
    with open(f"{R}/partA/summary.md", "w") as f:
        f.write(f"Conditioning variant chosen on validation (seed 0, mean val r²): {choice['val_mean_r2']} → **{best}**\n\n")
        f.write("Cross-chemistry r² (rows: model trained on / head; columns: test labels; n = %d shared test UTRs; mean ± sd over seeds)\n\n" % len(idx))
        f.write("| Model | " + " | ".join(f"test {CLAB[c]}" for c in CHEMS) + " |\n|---|---|---|---|\n")
        for k, v in mats.items():
            f.write(f"| {k} | " + " | ".join(ms(v[c]) for c in CHEMS) + " |\n")
        f.write("| noise ceiling for r² = Pearson r(rep1, rep2) | " + " | ".join(f"{ceil[c]:.3f}" for c in CHEMS) + " |\n\n")
        f.write("Own-chemistry metrics (mean ± sd over 3 seeds)\n\n| Chemistry | Model | r² | Pearson | Spearman | RMSE |\n|---|---|---|---|---|---|\n")
        for j, c in enumerate(CHEMS):
            for lab, pat, col in (("single", f"single_{c}_s?.json", None), (f"multi ({best})", f"multi_{best}_s?.json", str(j))):
                js = [json.load(open(p)) for p in sorted(glob.glob(f"{R}/partA/{pat}"))]
                get = (lambda x, k: x["test"][k]) if col is None else (lambda x, k: x["test"][col][k])
                f.write(f"| {CLAB[c]} | {lab} | " + " | ".join(ms([get(x, k) for x in js]) for k in ("r2", "pearson", "spearman", "rmse")) + " |\n")
    try:
        d = pickle.load(open("../ext/human_5utr_modeling/modeling/cds_modification_comparisons/saved_data/model_performance.dict", "rb"), encoding="latin1")
        paper = {"single U (paper)": d["e1_model"], "single Ψ (paper)": d["ep1_model"], "single m1Ψ (paper)": d["emp1_model"]}
        with open(f"{R}/partA/summary.md", "a") as f:
            f.write("\nPaper Fig. 2h reference (authors' saved performance dict; their test set, rep-1 columns)\n\n| Model | test U | test Ψ | test m1Ψ |\n|---|---|---|---|\n")
            for k, v in paper.items():
                f.write(f"| {k} | {v[0]:.3f} | {v[2]:.3f} | {v[4]:.3f} |\n")
    except Exception as e:
        print("paper dict not loaded:", e)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for ax, keys, title in ((axes[0], [k for k in mats if k.startswith("single")], "Separate single-chemistry CNNs"),
                            (axes[1], [k for k in mats if k.startswith("multi")], f"One multi-chemistry model ({best})")):
        M = np.array([[np.mean(mats[k][c]) for c in CHEMS] for k in keys])
        ax.imshow(M, cmap="Blues", vmin=0.5, vmax=1)
        for i in range(3):
            for j in range(3):
                ax.text(j, i, f"{M[i, j]:.3f}", ha="center", va="center", color="white" if M[i, j] > 0.82 else INK, size=10)
        ax.set_xticks(range(3), [f"test {CLAB[c]}" for c in CHEMS]); ax.set_yticks(range(3), keys)
        ax.set_title(title, loc="left")
        for s in ax.spines.values(): s.set_visible(False)
    fig.suptitle(f"Cross-chemistry r² on {len(idx):,} shared test UTRs (mean of 3 seeds); replicate ceilings "
                 + ", ".join(f"{CLAB[c]} {ceil[c]:.2f}" for c in CHEMS), x=0.02, ha="left", size=10, color=MUTED)
    fig.tight_layout(); fig.savefig(f"{F}/A_cross_chemistry.png"); plt.close(fig)
    print(open(f"{R}/partA/summary.md").read())


# ------------------------------------------------------------------------------------------------ Part B
def collect_b():
    rows = []
    for size in ("1k", "5k", "full"):
        for arch in ("cnn", "transformer", "hybrid"):
            cf = f"{R}/partB/{size}/{arch}_choice.json"
            if not os.path.exists(cf):
                continue
            best = json.load(open(cf))["chosen"]
            for p in sorted(glob.glob(f"{R}/partB/{size}/{arch}_{size}_{best}_s?.json")):
                j = json.load(open(p))
                rows.append({"size": size, "n_train_total": {"1k": 1000, "5k": 5000, "full": 260000}[size], "arch": arch,
                             "config": best, "seed": j["seed"], "params": j["params"], "best_epoch": j["best_epoch"],
                             **{k: j["test"][k] for k in ("r2", "pearson", "spearman", "rmse")}})
    return pd.DataFrame(rows)


def part_b():
    df = collect_b()
    df.to_csv(f"{R}/partB/all_runs.csv", index=False)
    g = df.groupby(["size", "arch"], sort=False)
    with open(f"{R}/partB/summary.md", "w") as f:
        f.write("U data, fixed 20k high-read test set. Mean ± sd over seeds (5 at 1k/5k, 3 at full).\n\n"
                "| Size | Architecture | Params | Config (val-chosen) | Seeds | r² | Pearson | Spearman | RMSE | Best epoch |\n|---|---|---|---|---|---|---|---|---|---|\n")
        for (size, arch), d in g:
            f.write(f"| {size} | {arch} | {d.params.iloc[0]:,} | {d.config.iloc[0]} | {len(d)} | {ms(d.r2.values)} | {ms(d.pearson.values)} | "
                    f"{ms(d.spearman.values)} | {ms(d.rmse.values)} | {ms(d.best_epoch.values, 1)} |\n")
        rs = sorted(glob.glob(f"{R}/partB/randsplit/*_s0.json"))
        if rs:
            f.write("\nRandom-split sanity check (full data, random 20k test, seed 0 only)\n\n| Architecture | r² random split | r² high-read split (same seed) |\n|---|---|---|\n")
            for p in rs:
                j = json.load(open(p)); arch = j["name"].split("_")[0]
                hr = df[(df.arch == arch) & (df["size"] == "full") & (df.seed == 0)].r2
                f.write(f"| {arch} | {j['test']['r2']:.3f} | {hr.iloc[0]:.3f} |\n" if len(hr) else f"| {arch} | {j['test']['r2']:.3f} | – |\n")
    sens = []
    for size in ("1k", "5k"):
        for arch in ("cnn", "transformer", "hybrid"):
            ps = sorted(glob.glob(f"{R}/partB/sens_patience30/{size}/{arch}_{size}_*_s?.json"))
            if ps:
                sens.append((size, arch, [json.load(open(p))["test"]["r2"] for p in ps]))
    if sens:
        with open(f"{R}/partB/summary.md", "a") as f:
            f.write("\nEarly-stopping sensitivity (same configs and subsets; patience 30 instead of 10, max 300 epochs; all architectures)\n\n"
                    "| Size | Architecture | r² patience 10 (main) | r² patience 30 | per-seed r² (patience 30) |\n|---|---|---|---|---|\n")
            for size, arch, v in sens:
                main = df[(df["size"] == size) & (df.arch == arch)].r2.values
                f.write(f"| {size} | {arch} | {ms(main)} | {ms(v)} | {', '.join(f'{x:.2f}' for x in v)} |\n")
    fig, ax = plt.subplots(figsize=(6.4, 4))
    for arch in ("cnn", "transformer", "hybrid"):
        d = df[df.arch == arch]
        if d.empty: continue
        s = d.groupby("n_train_total").r2.agg(["mean", "std"]).reset_index()
        ax.errorbar(s.n_train_total * (1 + 0.04 * ["cnn", "transformer", "hybrid"].index(arch)), s["mean"], yerr=s["std"].fillna(0),
                    color=COL[arch], marker="o", ms=6, lw=2, capsize=3, label={"cnn": "CNN (Optimus 5-Prime)", "transformer": "Transformer",
                                                                                 "hybrid": "CNN + attention hybrid"}[arch])
    ax.set_xscale("log"); ax.set_xticks([1000, 5000, 260000], ["1k", "5k", "260k"])
    ax.set(xlabel="Training sequences (log scale)", ylabel="Test r² (20k high-read test set)", ylim=(0, 1))
    ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True)
    ax.set_title("Learning curves (mean ± sd over seeds)", loc="left"); ax.legend(frameon=False, loc="lower right")
    fig.savefig(f"{F}/B_learning_curves.png"); plt.close(fig)
    chem_regimes()
    print(open(f"{R}/partB/summary.md").read())


def chem_regimes():
    rows = []
    for size in ("1k", "5k"):
        ch = f"{R}/partB/chem_{size}/choices.json"
        if not os.path.exists(ch): continue
        choices = json.load(open(ch))
        for name, lr in choices.items():
            for p in sorted(glob.glob(f"{R}/partB/chem_{size}/{name}_{size}_{lr}_s?.json")):
                j = json.load(open(p))
                if name == "multi":
                    for k, c in enumerate(CHEMS):
                        rows.append({"size": size, "method": "multi", "chem": c, "seed": j["seed"], "r2": j["test"][str(k)]["r2"]})
                else:
                    m, c = name.split("_")
                    rows.append({"size": size, "method": m, "chem": c, "seed": j["seed"], "r2": j["test"]["r2"]})
    best = json.load(open(f"{R}/partA/variant_choice.json"))["chosen"] if os.path.exists(f"{R}/partA/variant_choice.json") else "heads"
    for k, c in enumerate(CHEMS):
        for p in sorted(glob.glob(f"{R}/partA/single_{c}_s?.json")):
            rows.append({"size": "full", "method": "scratch", "chem": c, "seed": json.load(open(p)).get("seed"), "r2": json.load(open(p))["test"]["r2"]})
        for p in sorted(glob.glob(f"{R}/partA/multi_{best}_s?.json")):
            j = json.load(open(p)); rows.append({"size": "full", "method": "multi", "chem": c, "seed": j.get("seed"), "r2": j["test"][str(k)]["r2"]})
    df = pd.DataFrame(rows)
    if df.empty: return
    df.to_csv(f"{R}/partB/chem_regimes_runs.csv", index=False)
    with open(f"{R}/partB/summary.md", "a") as f:
        f.write("\nChemistry regimes (CNN trunk; test = 19,968 shared UTRs; mean ± sd; 5 seeds at 1k/5k, 3 at full). "
                "scratch = single-chemistry model; finetune = U-pretrained (full U data) then trained on the target chemistry; "
                "multi = one model trained on all three labels of the same sequences.\n\n| Size | Method | U | Ψ | m1Ψ |\n|---|---|---|---|---|\n")
        for (size, m), d in df.groupby(["size", "method"], sort=False):
            f.write(f"| {size} | {m} | " + " | ".join(ms(d[d.chem == c].r2.values) if (d.chem == c).any() else "–" for c in CHEMS) + " |\n")
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.6), sharey=True)
    xs = {"1k": 1000, "5k": 5000, "full": 260000}
    for ax, c in zip(axes, CHEMS):
        for m, lab in (("scratch", "single, from scratch"), ("finetune", "U-pretrained → fine-tune"), ("multi", "multi-chemistry")):
            d = df[(df.chem == c) & (df.method == m)]
            if d.empty: continue
            s = d.groupby("size").r2.agg(["mean", "std"]).reindex([k for k in xs if k in d["size"].values])
            ax.errorbar([xs[k] for k in s.index], s["mean"], yerr=s["std"].fillna(0), color=COL[m], marker="o", lw=2, capsize=3, label=lab)
        ax.set_xscale("log"); ax.set_xticks(list(xs.values()), ["1k", "5k", "260k"]); ax.set_title(f"Test on {CLAB[c]}", loc="left")
        ax.yaxis.grid(True, color=GRID); ax.set_axisbelow(True); ax.set_xlabel("Training sequences")
    axes[0].set_ylabel("Test r²"); axes[1].legend(frameon=False, loc="lower right", fontsize=9)
    fig.savefig(f"{F}/B_chemistry_regimes.png"); plt.close(fig)


# ------------------------------------------------------------------------------------------------ Part C
def part_c():
    m = json.load(open(f"{R}/partC/metrics.json"))["summary"]
    keys = [("r2", "r²"), ("pearson", "Pearson"), ("direction_acc", "Direction acc."), ("r2_high", "r² high-effect"),
            ("direction_acc_high", "Direction acc. high-effect"), ("auroc_high", "AUROC large effect")]
    lab = {"subtract": "Subtract (paper approach, random-library CNN)", "subtract_ft": "Subtract after fine-tuning on human MRLs",
           "siamese": "Siamese, trained on Δ", "siamese_aux": "Siamese, Δ + auxiliary absolute loss"}
    with open(f"{R}/partC/summary.md", "w") as f:
        a = m["authors_retrained_subtract"]
        f.write(f"ClinVar pairs n = {a['n']} (high-effect |log2 Δ| > 0.5: n = {a['n_high']}); pooled out-of-fold, 5-fold CV grouped by WT component; mean ± sd over 3 seeds.\n\n")
        f.write("| Method | " + " | ".join(k[1] for k in keys) + " |\n|---|" + "---|" * len(keys) + "\n")
        f.write("| Paper reference (stated): retrained model, subtraction | 0.555 | – | 64% | 76% (r²) | – | – |\n")
        f.write("| Authors' released predictions, recomputed on this set | " + " | ".join(f"{a[k]:.3f}" for k, _ in keys) + " |\n")
        for meth in ("subtract", "subtract_ft", "siamese", "siamese_aux"):
            f.write(f"| {lab[meth]} | " + " | ".join(f"{m[meth][k][0]:.3f} ± {m[meth][k][1]:.3f}" for k, _ in keys) + " |\n")
        f.write("\nSecondary: held-out synthesis-error pairs (noisier labels)\n\n| Method | r² | Direction acc. |\n|---|---|---|\n")
        for meth in ("subtract", "subtract_ft", "siamese", "siamese_aux"):
            s = m[f"{meth}_synth"]
            f.write(f"| {lab[meth]} | {s['r2'][0]:.3f} ± {s['r2'][1]:.3f} | {s['direction_acc'][0]:.3f} ± {s['direction_acc'][1]:.3f} |\n")
        sat = f"{R}/partC/saturation_mutagenesis.json"
        if os.path.exists(sat):
            s = json.load(open(sat))
            f.write("\nSaturation mutagenesis (150 substitutions per UTR; rank 1 = most MRL-decreasing predicted)\n\n"
                    "| Gene | ClinVar change | Observed log2 Δ | Subtract: pred / rank | Siamese: pred / rank | Siamese+aux: pred / rank |\n|---|---|---|---|---|---|\n")
            for gene, o in s.items():
                for v in o["variants"]:
                    if "change" not in v: continue
                    f.write(f"| {gene} | {v['change']} | {v['observed_log2']:.2f} | " + " | ".join(
                        f"{v[k]['pred_log2']:.2f} / {v[k]['rank_most_negative']}" for k in ("subtract", "siamese", "siamese_aux")) + " |\n")
    print(open(f"{R}/partC/summary.md").read())


def part_d():
    for p in (f"{R}/partD/chem_attribution.json", f"{R}/partD/arch_attribution.json"):
        if os.path.exists(p):
            print(p); print(json.dumps(json.load(open(p)), indent=1)[:4000])


if __name__ == "__main__":
    what = sys.argv[1]
    for k, fn in (("A", part_a), ("B", part_b), ("C", part_c), ("D", part_d)):
        if what in (k, "all"):
            fn()
