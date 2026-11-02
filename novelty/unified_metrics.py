"""Unified benchmark + novelty table on the real eGFP 20k test set, with regression AND
classification metrics (MRL binned into training-set tertiles: low / medium / high).

Produces results/unified/metrics.json, summary.md, and figures/unified/confusion_matrices.png.
"""
import glob, json, os, sys
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path[:0] = [os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                os.path.dirname(os.path.abspath(__file__))]
from bench import evaluate, tertile_edges
from data import build_chem_table, CHEMS

OUT, FIG = "results/unified", "figures/unified"
os.makedirs(OUT, exist_ok=True); os.makedirs(FIG, exist_ok=True)
INK, MUTED = "#1f1f1e", "#73726c"
plt.rcParams.update({"font.size": 10, "savefig.dpi": 200, "savefig.bbox": "tight",
                     "axes.titlesize": 10.5, "axes.titleweight": "bold"})

tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
y, edges = te.rl.values, tertile_edges(tr.rl.values)
t = build_chem_table()
sub = np.load("results/novelty/partA/test_idx.npy")          # 19,968 rows with labels in all chemistries
pos = {i: k for k, i in enumerate(np.where(t.split.values == "test")[0])}
sub_rows = np.array([pos[i] for i in sub])

ROLE = {"paper": "Paper model (reproduction)", "bench": "Benchmark", "ours": "Ours (novelty)"}
MODELS = [
    ("Optimus 5-Prime CNN", "paper", sorted(glob.glob("results/cnn_pred_seed*.npy")), None),
    ("k-mer linear (position-specific)", "bench", ["results/bench/real_kmer_pred.npy"], None),
    ("LightGBM (hand-crafted features)", "bench", ["results/bench/real_lgbm_pred.npy"], None),
    ("BiLSTM", "bench", sorted(glob.glob("results/bench/real_bilstm_s*_pred.npy")), None),
    ("Transformer (ours)", "ours", sorted(glob.glob("results/novelty/partB/full/transformer_full_lr0.0003_do0.2_s?_pred.npy")), None),
    ("CNN + attention hybrid (ours)", "ours", sorted(glob.glob("results/novelty/partB/full/hybrid_full_lr0.0003_do0.2_s?_pred.npy")), None),
    ("Multi-chemistry CNN, U head (ours)", "ours", sorted(glob.glob("results/novelty/partA/multi_heads_s?_pred.npy")), 0),
]

RECAL = json.load(open("results/novelty/partA/recalibration.json"))["per_seed"]["multi_U"] \
    if os.path.exists("results/novelty/partA/recalibration.json") else None
if RECAL:
    MODELS.append(("Multi-chemistry CNN, U head, recalibrated (ours)", "ours",
                   sorted(glob.glob("results/novelty/partA/multi_heads_s?_pred.npy")), 0))

rows, per_model = [], {}
for name, role, paths, col in MODELS:
    recal = RECAL if "recalibrated" in name else None
    if not paths:
        print("MISSING:", name); continue
    runs = []
    for p in paths:
        pr = np.load(p)
        if col is not None:                       # multi-chemistry: take one head, rows are the 19,968 subset
            v = pr[:, col]
            if recal:
                c = recal[len(runs)]
                v = c["intercept"] + c["slope"] * v
            runs.append((evaluate(y[sub_rows], v, edges), len(sub_rows)))
        else:
            runs.append((evaluate(y, pr, edges), len(y)))
    agg = {k: [r[0][k] for r in runs] for k in runs[0][0] if k != "confusion_matrix"}
    per_model[name] = {"role": role, "n_seeds": len(runs), "n_test": runs[0][1],
                       "confusion_matrix_seed0": runs[0][0]["confusion_matrix"],
                       **{k: [float(np.mean(v)), float(np.std(v))] for k, v in agg.items()}}
    rows.append(name)

json.dump(per_model, open(f"{OUT}/metrics.json", "w"), indent=1)

def cell(name, k, d=3):
    mu, sd = per_model[name][k]
    return f"{mu:.{d}f} ± {sd:.{d}f}" if per_model[name]["n_seeds"] > 1 else f"{mu:.{d}f}"

with open(f"{OUT}/summary.md", "w") as f:
    f.write("## Unified comparison, real eGFP held-out test set\n\n"
            "Regression metrics on MRL; classification metrics after binning MRL into tertiles of the "
            "**training** MRL distribution (low / medium / high), precision, recall and F1 macro-averaged. "
            "Mean ± sd over seeds where more than one seed was run.\n\n")
    f.write("| Model | Role | Seeds | n | r² | Spearman | RMSE | MAE | Accuracy | Precision | Recall | F1 |\n")
    f.write("|---|---|---|---|---|---|---|---|---|---|---|---|\n")
    for n in rows:
        m = per_model[n]
        f.write(f"| {n} | {ROLE[m['role']]} | {m['n_seeds']} | {m['n_test']:,} | " +
                " | ".join(cell(n, k) for k in ("r2", "spearman", "rmse", "mae", "accuracy",
                                                "precision_macro", "recall_macro", "f1_macro")) + " |\n")
    f.write("\nClass boundaries (training tertiles of MRL): "
            f"low < {edges[0]:.2f} ≤ medium < {edges[1]:.2f} ≤ high.\n")

n = len(rows)
fig, axes = plt.subplots(1, n, figsize=(2.6 * n, 3.0))
labels = ["low", "med", "high"]
for ax, name in zip(np.atleast_1d(axes), rows):
    cm = np.array(per_model[name]["confusion_matrix_seed0"], float)
    cmn = cm / cm.sum(1, keepdims=True)
    ax.imshow(cmn, cmap="Blues", vmin=0, vmax=1)
    for a in range(3):
        for b in range(3):
            ax.text(b, a, f"{cmn[a, b]:.2f}", ha="center", va="center", size=8.5,
                    color="white" if cmn[a, b] > 0.6 else INK)
    ax.set_xticks(range(3), labels, fontsize=8); ax.set_yticks(range(3), labels, fontsize=8)
    short = name.replace(", recalibrated (ours)", "\nrecalibrated").replace(" (ours)", "").replace(" (position-specific)", "").replace(" (hand-crafted features)", "")
    ax.set_title(f"{short}\nF1 {per_model[name]['f1_macro'][0]:.3f}", fontsize=8.5)
    ax.set_xlabel("predicted", fontsize=8)
    for s in ax.spines.values(): s.set_visible(False)
np.atleast_1d(axes)[0].set_ylabel("observed", fontsize=8)
fig.suptitle("Confusion matrices on the real eGFP test set (seed 0, row-normalised)", x=0.02, ha="left",
             fontweight="bold", size=10.5)
fig.tight_layout(); fig.savefig(f"{FIG}/confusion_matrices.png"); plt.close(fig)
print(open(f"{OUT}/summary.md").read())
