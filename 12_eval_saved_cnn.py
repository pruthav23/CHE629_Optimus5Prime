"""Score the 3 real-data CNN runs from the reproduction step with the shared benchmark metrics."""
import json, numpy as np, pandas as pd
from bench import evaluate, tertile_edges
tr, te = pd.read_csv("data/train.csv"), pd.read_csv("data/test.csv")
edges = tertile_edges(tr.rl.values)
for s in (1337, 1, 2):
    res = {"dataset": "real", "model": "cnn", "seed": s, **evaluate(te.rl.values, np.load(f"results/cnn_pred_seed{s}.npy"), edges),
           "minutes": json.load(open(f"results/cnn_seed{s}.json"))["minutes"]}
    np.save(f"results/bench/real_cnn_s{s}_pred.npy", np.load(f"results/cnn_pred_seed{s}.npy"))
    json.dump(res, open(f"results/bench/real_cnn_s{s}.json", "w"), indent=1)
    print(s, {k: round(v, 4) for k, v in res.items() if isinstance(v, float)})
