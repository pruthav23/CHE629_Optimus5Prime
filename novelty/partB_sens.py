"""Early-stopping sensitivity (Part B): same chosen configs and subsets as the main 1k/5k runs, patience 30
(max 300 epochs) instead of 10, identical for all architectures. Reported separately from the main protocol."""
import json
import numpy as np
import common
from common import run
from partB import X, YU, test_idx, subset, ARCHS, SIZES, SMALL_GRID, SMALL_SEEDS, tag, OUT
from data import train_val
from models import build

common.budget = lambda n_fit: (300, 30)
for size in SIZES:
    for arch in ARCHS:
        best = json.load(open(f"{OUT}/{size}/{arch}_choice.json"))["chosen"]
        lr, do = [c for c in SMALL_GRID if tag(*c) == best][0]
        for s in SMALL_SEEDS:
            f, v = train_val(subset(size, s), s, 0.20)
            run(f"{arch}_{size}_{best}_s{s}", f"{OUT}/sens_patience30/{size}", build(arch, do), X, YU, f, v, test_idx, lr, s,
                save_model=False)
print("PART B SENS DONE", flush=True)
