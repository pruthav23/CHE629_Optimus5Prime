"""Part A: one multi-chemistry model (shared CNN trunk + chemistry conditioning) vs three single-chemistry CNNs.
Full training data (U split, 260k; 10% validation), paper CNN config (lr 1e-3, dense dropout 0.2), early stopping.
Test = the 19,968 U-test sequences that have labels in all three chemistries."""
import json
import numpy as np
from common import run
from data import build_chem_table, train_val, CHEMS
from models import CNN, MultiChem
from optimus import one_hot

OUT = "results/novelty/partA"
SEEDS = [0, 1, 2]
t = build_chem_table()
X = one_hot(t.utr)
Y = t[[f"rl_{c}" for c in CHEMS]].values.astype(np.float32)
train_idx = np.where(t.split == "train")[0]
test_idx = np.where((t.split == "test") & ~np.isnan(Y).any(1))[0]

def splits(seed):
    return train_val(train_idx, seed, 0.10)

# 1) conditioning variant chosen on validation (seed 0)
val_score = {}
for mode in ["heads", "embed"]:
    f, v = splits(0)
    r = run(f"multi_{mode}_s0", OUT, MultiChem(CNN(dropout=0.2), mode=mode), X, Y, f, v, test_idx, 1e-3, 0)
    val_score[mode] = float(np.mean([r["val"][str(j)]["r2"] for j in range(3)]))
best = max(val_score, key=val_score.get)
json.dump({"val_mean_r2": val_score, "chosen": best}, open(f"{OUT}/variant_choice.json", "w"), indent=1)
print("variant choice:", val_score, "->", best, flush=True)

# 2) single-chemistry CNNs and the chosen multi model, 3 seeds
for seed in SEEDS:
    f, v = splits(seed)
    for j, c in enumerate(CHEMS):
        run(f"single_{c}_s{seed}", OUT, CNN(dropout=0.2), X, Y[:, j], f, v, test_idx, 1e-3, seed)
    run(f"multi_{best}_s{seed}", OUT, MultiChem(CNN(dropout=0.2), mode=best), X, Y, f, v, test_idx, 1e-3, seed)
np.save(f"{OUT}/test_idx.npy", test_idx)
print("PART A TRAINING DONE", flush=True)
