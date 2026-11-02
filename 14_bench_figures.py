import json, glob, os, numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from synth import NOISE_SD

INK, MUTED, GRID = "#1f1f1e", "#73726c", "#e4e3dd"
MODELS = [("cnn", "Optimus 5-Prime CNN", "#2a78d6"), ("bilstm", "BiLSTM", "#eb6834"),
          ("lgbm", "LightGBM (hand features)", "#1baf7a"), ("kmer", "k-mer linear", "#eda100")]
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 11.5, "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})
os.makedirs("figures/bench", exist_ok=True)


def load(ds, m):
    runs = [json.load(open(f)) for f in sorted(glob.glob(f"results/bench/{ds}_{m}*.json")) if not f.endswith("ism.json")]
    return runs


te_s = pd.read_csv("data/synth/test.csv")
ceiling = np.var(te_s.rl_true) / (np.var(te_s.rl_true) + NOISE_SD ** 2)

# ---------------------------------------------------------------- summary table
rows = []
for ds in ("synth", "real"):
    for m, label, _ in MODELS:
        runs = load(ds, m)
        if not runs: continue
        agg = {k: np.mean([r[k] for r in runs]) for k in ("r2", "spearman", "rmse", "mae", "accuracy",
                                                           "precision_macro", "recall_macro", "f1_macro")}
        agg["r2_sd"] = np.std([r["r2"] for r in runs]) if len(runs) > 1 else np.nan
        agg["r2_vs_true_mrl"] = np.mean([r.get("r2_vs_true_mrl", np.nan) for r in runs])
        rows.append({"dataset": ds, "model": label, "n_runs": len(runs), **agg})
summ = pd.DataFrame(rows)
summ.to_csv("results/bench/summary.csv", index=False)
with open("results/bench/summary.md", "w") as f:
    for ds, title in (("synth", f"Synthetic (noise ceiling r² = {ceiling:.3f})"), ("real", "Real eGFP (top-20k test)")):
        f.write(f"### {title}\n\n| Model | r² | Spearman | RMSE | MAE | Accuracy | Precision | Recall | F1 |\n|---|---|---|---|---|---|---|---|---|\n")
        for _, r in summ[summ.dataset == ds].iterrows():
            r2s = f"{r.r2:.3f}" + (f" ± {r.r2_sd:.3f}" if r.n_runs > 1 else "")
            f.write(f"| {r.model} | {r2s} | {r.spearman:.3f} | {r.rmse:.3f} | {r.mae:.3f} | {r.accuracy:.3f} | "
                    f"{r.precision_macro:.3f} | {r.recall_macro:.3f} | {r.f1_macro:.3f} |\n")
        f.write("\n")
print(open("results/bench/summary.md").read())

# ---------------------------------------------------------------- r2 small multiples
fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharey=True)
for ax, ds, title in zip(axes, ("synth", "real"), ("Synthetic data (known rules)", "Real eGFP polysome data")):
    present = [(m, l, c) for m, l, c in MODELS if load(ds, m)]
    y = np.arange(len(MODELS))[::-1]
    for yi, (m, l, c) in zip(y, MODELS):
        runs = load(ds, m)
        if not runs: continue
        v = [r["r2"] for r in runs]
        ax.barh(yi, np.mean(v), color=c, height=0.6)
        if len(v) > 1:
            ax.errorbar(np.mean(v), yi, xerr=np.std(v), color=INK, capsize=3, lw=1)
        ax.text(np.mean(v) + 0.015 + (np.std(v) if len(v) > 1 else 0), yi, f"{np.mean(v):.3f}", va="center", color=INK, size=9.5,
                bbox=dict(fc="white", ec="none", pad=1.5), zorder=5)
    if ds == "synth":
        ax.axvline(ceiling, color=MUTED, ls="--", lw=1)
        ax.text(ceiling - 0.01, -0.75, f"noise ceiling {ceiling:.3f}", color=MUTED, size=8.5, ha="right")
    ax.set_yticks(y, [l for _, l, _ in MODELS]); ax.set_xlim(0, 1.08); ax.set_ylim(-1, len(MODELS) - 0.4)
    ax.set_xlabel("Test r² (n = 20,000)"); ax.set_title(title, loc="left")
    ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
fig.savefig("figures/bench/r2_synth_vs_real.png"); plt.close(fig)

# ---------------------------------------------------------------- confusion matrices
fig, axes = plt.subplots(2, 4, figsize=(12, 6.2))
labels = ["low", "med", "high"]
for i, ds in enumerate(("synth", "real")):
    for j, (m, l, _) in enumerate(MODELS):
        ax = axes[i, j]; runs = load(ds, m)
        if not runs: ax.axis("off"); continue
        r0 = next((r for r in runs if r["seed"] == 1337), runs[0])
        cm = np.array(r0["confusion_matrix"], float)
        cmn = cm / cm.sum(1, keepdims=True)
        ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
        for a in range(3):
            for b in range(3):
                ax.text(b, a, f"{cmn[a, b]:.2f}\n({int(cm[a, b]):,})", ha="center", va="center", size=8,
                        color="white" if cmn[a, b] > 0.6 else INK)
        ax.set_xticks(range(3), labels); ax.set_yticks(range(3), labels)
        ax.set_title(f"{l}\nF1 {r0['f1_macro']:.3f}", size=9.5)
        if j == 0: ax.set_ylabel(("Synthetic" if ds == "synth" else "Real eGFP") + "\nobserved class")
        if i == 1: ax.set_xlabel("predicted class")
        for s in ax.spines.values(): s.set_visible(False)
fig.suptitle("Confusion matrices (seed 1337): MRL binned into training-set tertiles, row-normalised", x=0.02, ha="left",
             fontweight="bold", size=11.5)
fig.tight_layout(); fig.savefig("figures/bench/confusion_matrices.png"); plt.close(fig)

# ---------------------------------------------------------------- ATG insertion (rule recovery)
if os.path.exists("results/bench/synth_ism.csv"):
    d = pd.read_csv("results/bench/synth_ism.csv")
    g = d.groupby("dist")[["true_delta", "cnn_delta", "bilstm_delta"]].mean()
    ism = json.load(open("results/bench/synth_ism.json"))
    fig, ax = plt.subplots(figsize=(8, 3.8))
    ax.axhline(0, color=GRID, lw=1)
    ax.plot(g.index, g.true_delta, color=INK, lw=2, marker="o", ms=4, label="ground truth (simulator)")
    ax.plot(g.index, g.cnn_delta, color="#2a78d6", lw=2, label=f"CNN  (r = {ism['cnn']['r_true_vs_pred_delta']:.2f})")
    ax.plot(g.index, g.bilstm_delta, color="#eb6834", lw=2, label=f"BiLSTM  (r = {ism['bilstm']['r_true_vs_pred_delta']:.2f})")
    ax.invert_xaxis()
    ax.set(xlabel="Inserted ATG position (nt upstream of CDS start)", ylabel="Mean change in MRL")
    ax.set_title("Synthetic: effect of inserting one uAUG, true vs learned", loc="left")
    ax.legend(frameon=False, loc="lower left", ncols=3, bbox_to_anchor=(0, -0.42))
    fig.savefig("figures/bench/synth_atg_insertion.png"); plt.close(fig)
print("figures done")
