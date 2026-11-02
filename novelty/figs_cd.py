import json, sys
import numpy as np, pandas as pd, matplotlib
matplotlib.use("Agg"); import matplotlib.pyplot as plt
sys.path[:0] = [".", "novelty"]
from data import build_pairs, build_chem_table
from attribution import ism_contrib
from optimus import one_hot
INK, MUTED, GRID = "#1f1f1e", "#73726c", "#e4e3dd"
plt.rcParams.update({"font.size": 10.5, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": MUTED,
                     "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 11.5,
                     "axes.titleweight": "bold", "savefig.dpi": 200, "savefig.bbox": "tight"})
P = build_pairs(); ev = P["eval"].values; obs = P.delta.values[ev]
oof = [pd.read_csv(f"results/novelty/partC/oof_seed{s}.csv") for s in range(3)]
fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), sharey=True)
for ax, m, title in ((axes[0], "subtract", "Subtraction (paper approach)"), (axes[1], "siamese_aux", "Siamese, Δ + auxiliary loss")):
    pred = np.mean([o[m].values[ev] for o in oof], 0)
    hi = np.abs(obs) > 0.5
    ax.axhline(0, color=GRID, lw=1); ax.axvline(0, color=GRID, lw=1)
    ax.scatter(obs[~hi], pred[~hi], s=8, color="#2a78d6", alpha=0.35, lw=0, label=f"|log2 Δ| ≤ 0.5 (n={(~hi).sum():,})")
    ax.scatter(obs[hi], pred[hi], s=26, color="#eb6834", edgecolor="white", lw=0.8, label=f"high-effect (n={hi.sum()})")
    r2 = np.corrcoef(obs, pred)[0, 1] ** 2; d = np.mean(np.sign(obs) == np.sign(pred))
    ax.text(0.97, 0.04, f"r² {r2:.3f} · direction {d:.1%}", transform=ax.transAxes, ha="right", color=INK)
    ax.set(xlim=(-2, 1.5), ylim=(-2, 1.5), xlabel="Observed log2(MRL variant / WT)"); ax.set_title(title, loc="left")
axes[0].set_ylabel("Predicted log2 change (3-seed ensemble)"); axes[0].legend(frameon=False, loc="upper left", fontsize=9)
fig.savefig("figures/novelty/C_variant_effects.png"); plt.close(fig)

motifs = json.load(open("results/novelty/partD/modisco_motifs.json"))
ppm = np.array([m for m in motifs if m.get("model") == "U_s0"][0]["ppm"])
t = build_chem_table(); S = np.load("results/novelty/partD/attribution_idx.npy"); seqs = t.utr.values[S]
X = one_hot(seqs); free = np.array(["ATG" not in s for s in seqs])
C = ism_contrib(np.load("results/novelty/partD/attr_U_s0.npz")["ism"], X).sum(-1)[free]; base = X[free].argmax(-1)
fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [1, 2]})
ax = axes[0]
ax.imshow(ppm.T, cmap="Blues", vmin=0, vmax=1, aspect="auto")
for i in range(ppm.shape[0]):
    for b in range(4):
        if ppm[i, b] > 0.4:
            ax.text(i, b, "ACGT"[b], ha="center", va="center", color="white" if ppm[i, b] > 0.7 else INK, fontweight="bold")
ax.set_yticks(range(4), list("ACGT")); ax.set_xticks(range(ppm.shape[0]), [str(i + 1) for i in range(ppm.shape[0])], fontsize=8)
ax.set_title("TF-MoDISco motif, U model (repressive)", loc="left", fontsize=10.5); ax.set_xlabel("motif position")
for sp in ax.spines.values(): sp.set_visible(False)
ax = axes[1]
cols = {"A": "#1baf7a", "C": "#2a78d6", "G": "#eda100", "T": "#eb6834"}
for b, nm in enumerate("ACGT"):
    prof = [C[:, p][base[:, p] == b].mean() for p in range(50)]
    ax.plot(np.arange(-50, 0), prof, color=cols[nm], lw=1.8, label=nm)
ax.axhline(0, color=GRID, lw=1)
ax.set(xlabel="Position relative to CDS start codon", ylabel="Mean ISM contribution")
ax.set_title("Per-base contribution, uAUG-free UTRs (U model)", loc="left", fontsize=10.5)
ax.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=9)
fig.savefig("figures/novelty/D_motif_and_profile.png"); plt.close(fig)
print("ok")
