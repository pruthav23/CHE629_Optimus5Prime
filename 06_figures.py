import numpy as np, pandas as pd, json, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
from optimus import r2
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#73726c"
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.titlesize": 12, "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})
te, tr = pd.read_csv("data/test.csv"), pd.read_csv("data/train.csv")
te["pred"] = np.load("results/cnn_pred_seed1337.npy")

# Fig 2b -------------------------------------------------------------
f, ax = plt.subplots(figsize=(5.2, 5))
has = te.utr.str.contains("ATG")
for m, c, lab in [(has, BLUE, "uAUG"), (~has, ORANGE, "no uAUG")]:
    ax.scatter(te.rl[m], te.pred[m], s=3, alpha=0.25, color=c, lw=0, label=f"{lab} (n={m.sum():,})")
ax.plot([1, 9], [1, 9], color=MUTED, lw=1, ls="--")
ax.set(xlim=(1, 9), ylim=(1, 9), xlabel="Observed MRL", ylabel="Predicted MRL",
       title="Optimus 5-Prime, eGFP held-out test (n=20,000)")
ax.text(0.97, 0.04, f"r² = {r2(te.rl, te.pred):.3f}  (paper: 0.93)", transform=ax.transAxes, ha="right", color=INK)
leg = ax.legend(loc="upper left", frameon=False, markerscale=5)
for h in leg.legend_handles: h.set_alpha(1)
f.savefig("figures/fig2b_test_scatter.png"); plt.close(f)

# Fig 1c: single-uAUG position / frame effect --------------------------
al = pd.concat([tr, te])
one = al[al.utr.str.count("ATG") == 1].copy()
one["pos"] = one.utr.str.find("ATG")
one["dist"] = 50 - one.pos                       # nt upstream of the CDS start codon
g = one.groupby("dist").rl.agg(["mean", "count"]).reset_index()
g = g[g["count"] >= 50]
inf = g.dist % 3 == 0
f, ax = plt.subplots(figsize=(7.5, 3.8))
ax.axhline(al[~al.utr.str.contains("ATG")].rl.mean(), color=MUTED, lw=1, ls="--")
ax.text(g.dist.min() + 1, al[~al.utr.str.contains("ATG")].rl.mean() - 0.18, "mean MRL, no uAUG", ha="left", color=MUTED, size=9)
ax.scatter(g.dist[inf], g["mean"][inf], color=BLUE, s=36, label="in frame with CDS", zorder=3, edgecolor="white", lw=1)
ax.scatter(g.dist[~inf], g["mean"][~inf], color=ORANGE, s=36, label="out of frame", zorder=3, edgecolor="white", lw=1)
ax.invert_xaxis()
ax.set(xlabel="uAUG position (nt upstream of CDS start)", ylabel="Mean observed MRL",
       title="Single-uAUG UTRs: out-of-frame uAUGs depress ribosome load")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(0, 0.92))
f.savefig("figures/fig1c_uaug_frame.png"); plt.close(f)

# Model comparison ------------------------------------------------------
km = json.load(open("results/kmer_baseline.json"))
seeds = [json.load(open(f"results/cnn_seed{s}.json"))["test_r2"] for s in (1337, 1, 2)
         if __import__("os").path.exists(f"results/cnn_seed{s}.json")]
rows = [(f"k-mer linear, k={k}" + ("  [paper best: 0.66]" if k == "6" else ""), v["test_r2"], BLUE) for k, v in km.items()]
rows += [(f"CNN, ours (mean of {len(seeds)} seed{'s' if len(seeds)>1 else ''})  [paper: 0.93]", np.mean(seeds), ORANGE),
         ("CNN (authors' saved weights)", 0.9338, ORANGE)]
f, ax = plt.subplots(figsize=(7, 3.8))
y = np.arange(len(rows))[::-1]
ax.barh(y, [r[1] for r in rows], color=[r[2] for r in rows], height=0.62)
for yi, r in zip(y, rows): ax.text(r[1] + 0.01, yi, f"{r[1]:.3f}", va="center", color=INK, size=10)
ax.set_yticks(y, [r[0] for r in rows]); ax.set_xlim(0, 1.08); ax.set_xlabel("Test r² (n=20,000)")
ax.set_title("CNN vs position-specific k-mer baseline", loc="left", pad=10)
f.savefig("figures/model_comparison.png"); plt.close(f)

# Fig 4a: human UTRs -----------------------------------------------------
import os
hp = "results/cnn_seed1337_human_pred.npy"
hu = pd.read_csv("data/human_top25k.csv", low_memory=False); hu["pred"] = np.load(hp)
snv = hu.info4 == "variant"
f, ax = plt.subplots(figsize=(5.2, 5))
ax.scatter(hu.rl[~snv], hu.pred[~snv], s=3, alpha=0.3, color=BLUE, lw=0, label=f"common ({(~snv).sum():,})")
ax.scatter(hu.rl[snv], hu.pred[snv], s=3, alpha=0.5, color=ORANGE, lw=0, label=f"SNV ({snv.sum():,})")
ax.set(xlim=(1, 9), ylim=(1, 9), xlabel="Observed MRL", ylabel="Predicted MRL",
       title=f"Human 5′UTRs ({'retrained' if 'retrained' in hp else 'original'} model)")
ax.plot([1, 9], [1, 9], color=MUTED, lw=1, ls="--")
ax.text(0.97, 0.04, f"r² = {r2(hu.rl, hu.pred):.3f}  (paper: 0.81)", transform=ax.transAxes, ha="right", color=INK)
leg = ax.legend(loc="upper left", frameon=False, markerscale=5)
for h in leg.legend_handles: h.set_alpha(1)
f.savefig("figures/fig4a_human_utrs.png"); plt.close(f)
print("ok")
