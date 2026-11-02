"""Part B: CNN vs transformer vs hybrid across training-set sizes (U data), plus multi- vs single-chemistry
regimes and U->Psi/m1Psi transfer.

usage: python3 novelty/partB.py {small|chem|full|randsplit}
  small      : 1k and 5k, all architectures (tune 4 configs on seed 0, then seeds 1-4 with the chosen config)
  chem       : 1k and 5k chemistry regimes with the CNN: scratch single, U-pretrained fine-tune, multi (lr tuned on seed 0)
  full       : full 260k, all architectures (tune lr on seed 0, then seeds 1-2)
  randsplit  : full data, random 20k test split, chosen config, seed 0 (single-seed sanity check)
"""
import json, os, sys
import numpy as np
import torch
from common import run
from data import build_chem_table, nested_subsets, train_val, random_split, CHEMS
from models import build, CNN, MultiChem
from optimus import one_hot

OUT = "results/novelty/partB"
SIZES = {"1k": 1000, "5k": 5000}
ARCHS = ["cnn", "transformer", "hybrid"]
SMALL_GRID = [(lr, do) for lr in (1e-3, 3e-4) for do in (0.1, 0.3)]
FULL_GRID = [(1e-3, 0.2), (3e-4, 0.2)]
SMALL_SEEDS, FULL_SEEDS = [0, 1, 2, 3, 4], [0, 1, 2]

t = build_chem_table()
X = one_hot(t.utr)
YU = t.rl_U.values.astype(np.float32)
YC = t[[f"rl_{c}" for c in CHEMS]].values.astype(np.float32)
train_idx = np.where(t.split == "train")[0]
test_idx = np.where(t.split == "test")[0]
chem_test_idx = np.where((t.split == "test") & ~np.isnan(YC).any(1))[0]
os.makedirs(OUT, exist_ok=True)


def subset(size_key, seed):
    subs = nested_subsets(len(train_idx), list(SIZES.values()), seed, f"{OUT}/subsets")
    return train_idx[subs[SIZES[size_key]]]


def tag(lr, do):
    return f"lr{lr:g}_do{do:g}"


def choose(results):
    def score(k):
        v = results[k]["val"]["r2"] if "r2" in results[k]["val"] else np.mean([x["r2"] for x in results[k]["val"].values()])
        return -np.inf if np.isnan(v) else v
    return max(results, key=score)


def stage_small():
    for size in SIZES:
        for arch in ARCHS:
            f, v = train_val(subset(size, 0), 0, 0.20)
            res = {tag(*c): run(f"{arch}_{size}_{tag(*c)}_s0", f"{OUT}/{size}", build(arch, c[1]), X, YU, f, v,
                                test_idx, c[0], 0, save_model=False) for c in SMALL_GRID}
            best = choose(res)
            json.dump({"chosen": best, "val_r2": {k: r["val"]["r2"] for k, r in res.items()}},
                      open(f"{OUT}/{size}/{arch}_choice.json", "w"), indent=1)
            lr, do = [c for c in SMALL_GRID if tag(*c) == best][0]
            for s in SMALL_SEEDS[1:]:
                f, v = train_val(subset(size, s), s, 0.20)
                run(f"{arch}_{size}_{best}_s{s}", f"{OUT}/{size}", build(arch, do), X, YU, f, v, test_idx, lr, s,
                    save_model=False)


def cnn_dropout(size):
    return json.load(open(f"{OUT}/{size}/cnn_choice.json"))["chosen"].split("_do")[1]


def stage_chem():
    """Per size: scratch single-chemistry CNNs (U, psi, m1psi); U-pretrained CNN fine-tuned on psi / m1psi;
    one multi-chemistry model (Part A's chosen variant) on all three labels of the same sequences.
    Equal budget per method: lr in {1e-3, 3e-4} tuned on seed-0 validation; dropout = the CNN's choice at that size."""
    variant = json.load(open("results/novelty/partA/variant_choice.json"))["chosen"]
    for size in SIZES:
        do = float(cnn_dropout(size))
        out = f"{OUT}/chem_{size}"

        def jobs(s):
            pre = torch.load(f"results/novelty/partA/single_U_s{s % 3}.pt")
            out_ = [(f"scratch_{c}", lambda: CNN(dropout=do), YC[:, j], None) for j, c in enumerate(CHEMS)]
            out_ += [(f"finetune_{c}", lambda: CNN(dropout=do), YC[:, j], pre) for j, c in enumerate(CHEMS) if c != "U"]
            out_ += [("multi", lambda: MultiChem(CNN(dropout=do), mode=variant), YC, None)]
            return out_

        chosen = {}
        f, v = train_val(subset(size, 0), 0, 0.20)
        for name, make, y, init in jobs(0):
            res = {f"lr{lr:g}": run(f"{name}_{size}_lr{lr:g}_s0", out, make(), X, y, f, v, chem_test_idx, lr, 0,
                                    init_state=init, save_model=False) for lr in (1e-3, 3e-4)}
            chosen[name] = choose(res)
        json.dump(chosen, open(f"{out}/choices.json", "w"), indent=1)
        for s in SMALL_SEEDS[1:]:
            f, v = train_val(subset(size, s), s, 0.20)
            for name, make, y, init in jobs(s):
                run(f"{name}_{size}_{chosen[name]}_s{s}", out, make(), X, y, f, v, chem_test_idx,
                    float(chosen[name][2:]), s, init_state=init, save_model=False)


def stage_full():
    out = f"{OUT}/full"
    for arch in ARCHS:
        res = {}
        for lr, do in FULL_GRID:
            f, v = train_val(train_idx, 0, 0.10)
            name = f"{arch}_full_{tag(lr, do)}_s0"
            init = None
            src = f"results/novelty/partA/single_U_s0.pt"
            if arch == "cnn" and (lr, do) == (1e-3, 0.2) and os.path.exists(src):
                res[tag(lr, do)] = reuse_partA(name, out, 0, f, v)
                continue
            res[tag(lr, do)] = run(name, out, build(arch, do), X, YU, f, v, test_idx, lr, 0)
        best = choose(res)
        json.dump({"chosen": best, "val_r2": {k: r["val"]["r2"] for k, r in res.items()}},
                  open(f"{out}/{arch}_choice.json", "w"), indent=1)
        lr, do = [c for c in FULL_GRID if tag(*c) == best][0]
        for s in FULL_SEEDS[1:]:
            f, v = train_val(train_idx, s, 0.10)
            name = f"{arch}_full_{best}_s{s}"
            if arch == "cnn" and (lr, do) == (1e-3, 0.2) and os.path.exists(f"results/novelty/partA/single_U_s{s}.pt"):
                reuse_partA(name, out, s, f, v)
            else:
                run(name, out, build(arch, do), X, YU, f, v, test_idx, lr, s)


def reuse_partA(name, out, seed, f, v):
    """Identical config/split/seed to Part A's single-U CNN: reuse its weights, evaluate on the full 20k U test."""
    from train import predict, Scaler
    from evaluate import metrics
    jp = f"{out}/{name}.json"
    if os.path.exists(jp):
        return json.load(open(jp))
    a = json.load(open(f"results/novelty/partA/single_U_s{seed}.json"))
    m = CNN(dropout=0.2); m.load_state_dict(torch.load(f"results/novelty/partA/single_U_s{seed}.pt"))
    sc = Scaler(YU[f])
    pred, vpred = sc.inv(predict(m, X[test_idx])), sc.inv(predict(m, X[v]))
    res = {k: a[k] for k in ("best_epoch", "epochs_run", "val_loss", "val_hist", "minutes", "params", "n_fit", "n_val")}
    res.update(test=metrics(YU[test_idx], pred), val=metrics(YU[v], vpred), name=name, seed=seed, lr=1e-3,
               reused_from=f"partA/single_U_s{seed}")
    os.makedirs(out, exist_ok=True)
    np.save(f"{out}/{name}_pred.npy", pred); torch.save(m.state_dict(), f"{out}/{name}.pt")
    json.dump(res, open(jp, "w"), indent=1)
    return res


def stage_randsplit():
    out = f"{OUT}/randsplit"
    pool = np.concatenate([train_idx, test_idx])
    tr, te = random_split(len(pool), 20_000, 0)
    tr, te = pool[tr], pool[te]
    np.savez(f"{OUT}/randsplit_idx.npz", train=tr, test=te)
    for arch in ARCHS:
        best = json.load(open(f"{OUT}/full/{arch}_choice.json"))["chosen"]
        lr, do = [c for c in FULL_GRID if tag(*c) == best][0]
        f, v = train_val(tr, 0, 0.10)
        run(f"{arch}_randsplit_{best}_s0", out, build(arch, do), X, YU, f, v, te, lr, 0, save_model=False)


if __name__ == "__main__":
    {"small": stage_small, "chem": stage_chem, "full": stage_full, "randsplit": stage_randsplit}[sys.argv[1]]()
    print(f"PART B {sys.argv[1]} DONE", flush=True)
