"""Live demo for the CHE629 presentation. Runs on CPU in about a minute using the shipped weights.

usage:
  python3 demo.py                 # all sections
  python3 demo.py reproduce       # paper CNN vs our hybrid on the 20k held-out test set
  python3 demo.py benchmarks      # the unified benchmark table (regression + classification metrics)
  python3 demo.py variant         # saturation mutagenesis: do we rank the ClinVar variants first?
  python3 demo.py train           # 60-second live training run on a 5k subset, to show training works
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch

sys.path[:0] = [os.path.dirname(os.path.abspath(__file__)), os.path.join(os.path.dirname(os.path.abspath(__file__)), "novelty")]
torch.set_num_threads(2)
from optimus import one_hot
from bench import evaluate, tertile_edges
from models import CNN, build
from train import predict, Scaler, fit

W = "weights"
BAR = "=" * 78


def head(s):
    print(f"\n{BAR}\n{s}\n{BAR}", flush=True)


def load(arch, path, dropout=0.2):
    m = build(arch, dropout)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.eval()


def reproduce():
    head("1. REPRODUCTION + BEST MODEL  (held-out 20,000 UTRs, never used for training or tuning)")
    te = pd.read_csv("data/test.csv")
    x, y = one_hot(te.utr), te.rl.values
    edges = np.array(json.load(open(f"{W}/class_edges.json"))["edges"])
    sc = json.load(open(f"{W}/scaler_s0.json")); mu, sd = sc["mu"], sc["sd"]
    for label, arch, path, paper in (("Optimus 5-Prime CNN (paper)", "cnn", f"{W}/cnn_full_s0.pt", "0.93"),
                                     ("CNN + attention hybrid (ours)", "hybrid", f"{W}/hybrid_full_s0.pt", "--")):
        t0 = time.time()
        m = load(arch, path)
        p = predict(m, x) * sd + mu
        r = evaluate(y, p, edges)
        print(f"{label:32s} r2={r['r2']:.3f} (paper {paper})  F1={r['f1_macro']:.3f}  "
              f"RMSE={r['rmse']:.3f}  [{time.time() - t0:.1f}s]", flush=True)


def benchmarks():
    head("2. BENCHMARK COMPARISON  (precomputed; regenerate with novelty/unified_metrics.py)")
    print(open("results/unified/summary.md").read())


def variant():
    head("3. VARIANT EFFECT  (saturation mutagenesis of 3 disease 5'UTRs; rank 1 = most MRL-lowering)")
    sat = json.load(open("results/novelty/partC/saturation_mutagenesis.json"))
    sc = json.load(open(f"{W}/scaler_s0.json")); mu, sd = sc["mu"], sc["sd"]
    m = load("cnn", f"{W}/cnn_full_s0.pt")
    for gene, o in sat.items():
        wt = o["wt"]
        muts, where = [], []
        for pos in range(50):
            for b in "ACGT":
                if b != wt[pos]:
                    muts.append(wt[:pos] + b + wt[pos + 1:]); where.append(f"{wt[pos]}{pos - 50}{b}")
        pw = np.clip(predict(m, one_hot([wt])) * sd + mu, 0.1, None)
        pm = np.clip(predict(m, one_hot(muts)) * sd + mu, 0.1, None)
        delta = np.log2(pm / pw)
        order = np.argsort(delta)
        for v in o["variants"]:
            if "change" not in v:
                continue
            key = v["change"].split(" ")[0]
            k = where.index(key)
            rank = int(np.where(order == k)[0][0]) + 1
            flag = "<-- ClinVar variant ranked FIRST" if rank == 1 else ""
            print(f"{gene:9s} {key:8s} observed log2 {v['observed_log2']:+.2f}  predicted {delta[k]:+.2f}  "
                  f"rank {rank}/{len(delta)} {flag}", flush=True)


def train_live(n=5000, seconds=60):
    head(f"4. LIVE TRAINING  ({n:,} sequences, ~{seconds}s, to show the pipeline runs end to end)")
    if not os.path.exists("data/train.csv"):
        print("data/train.csv not present (it is too large to ship in the repository).")
        print("Regenerate it with:  python3 00_prep.py <GSM3130435_egfp_unmod_1.csv.gz from GEO GSE114002>")
        print("Sections 1-3 above run without it.")
        return
    tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(tr))[:n]
    fit_i, val_i = idx[: int(0.8 * n)], idx[int(0.8 * n):]
    x, y = one_hot(tr.utr.values[idx]), tr.rl.values[idx]
    sc = Scaler(y[: len(fit_i)])
    xt, yt = one_hot(te.utr), te.rl.values
    t0 = time.time()
    m, info = fit(CNN(dropout=0.1), x[: len(fit_i)], sc.fwd(y[: len(fit_i)]), x[len(fit_i):], sc.fwd(y[len(fit_i):]),
                  lr=1e-3, max_epochs=6, patience=6, seed=0, log=None)
    p = sc.inv(predict(m, xt))
    r = evaluate(yt, p, np.array(json.load(open(f"{W}/class_edges.json"))["edges"]))
    print(f"trained in {time.time() - t0:.0f}s, best epoch {info['best_epoch']} -> "
          f"test r2={r['r2']:.3f} F1={r['f1_macro']:.3f}", flush=True)
    print("(the full 260k run reaches r2 0.92; see results/novelty/partB/summary.md)")


if __name__ == "__main__":
    what = sys.argv[1] if len(sys.argv) > 1 else "all"
    if not os.path.exists(f"{W}/cnn_full_s0.pt"):
        sys.exit(f"missing {W}/cnn_full_s0.pt -- run scripts/export_demo_weights.py first")
    for k, fn in (("reproduce", reproduce), ("benchmarks", benchmarks), ("variant", variant), ("train", train_live)):
        if what in (k, "all"):
            fn()
