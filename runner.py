"""Train one model on one dataset and evaluate it on that dataset's 20k test set.

usage: python3 runner.py {synth|real} {cnn|bilstm|lgbm|kmer} [seed]

Same protocol for every model:
  * identical train/test split (real: paper's top-20k-by-reads test set; synth: random 20k)
  * last 10% of the shuffled training set = validation, used ONLY by the classical models
    for hyperparameter choice / early stopping; neural nets use the paper's fixed 3-epoch recipe
  * MRL standardised with training-set mean/sd for the neural nets
  * low/medium/high classes for the confusion matrix = tertiles of the training MRL
"""
import json, sys, time, os
import numpy as np, pandas as pd, torch
from optimus import OptimusCNN, train, predict, one_hot
from bench import BiLSTM, evaluate, tertile_edges, kmer_features, feature_matrix

def run_model(dataset, model_name, seed=1337):
    root = "data/synth" if dataset == "synth" else "data"
    out = f"results/bench/{dataset}_{model_name}" + (f"_s{seed}" if model_name in ("cnn", "bilstm") else "")
    os.makedirs("results/bench", exist_ok=True)

    tr, te = pd.read_csv(f"{root}/train.csv"), pd.read_csv(f"{root}/test.csv")
    ytr, yte = tr.rl.values.astype(np.float32), te.rl.values.astype(np.float32)
    edges = tertile_edges(ytr)
    rng = np.random.default_rng(0)
    perm = rng.permutation(len(tr)); n_val = len(tr) // 10
    i_fit, i_val = perm[n_val:], perm[:n_val]
    t0 = time.time(); extra = {}

    if model_name in ("cnn", "bilstm"):
        xtr, xte = one_hot(tr.utr), one_hot(te.utr)
        mu, sd = ytr.mean(), ytr.std()
        net = OptimusCNN() if model_name == "cnn" else BiLSTM()
        net = train(net, xtr, (ytr - mu) / sd, epochs=3, seed=seed, log_every=0)
        pred = predict(net, xte) * sd + mu
        torch.save(net.state_dict(), out + ".pt")
        extra["n_params"] = int(sum(p.numel() for p in net.parameters()))

    elif model_name == "kmer":
        from sklearn.linear_model import Ridge
        best = None
        for k in (3, 4, 5, 6):
            X = kmer_features(tr.utr, k)
            for a in (1, 10, 100, 1000):
                m = Ridge(alpha=a, solver="sparse_cg").fit(X[i_fit], ytr[i_fit])
                v = np.corrcoef(ytr[i_val], m.predict(X[i_val]))[0, 1] ** 2
                print(f"k={k} alpha={a} val r2={v:.4f}", flush=True)
                if best is None or v > best[2]:
                    best = (k, a, v)
        k, a, _ = best
        m = Ridge(alpha=a, solver="sparse_cg").fit(kmer_features(tr.utr, k), ytr)
        pred = m.predict(kmer_features(te.utr, k))
        extra.update(k=k, alpha=a)

    elif model_name == "lgbm":
        import lightgbm as lgb
        Ftr, Fte = feature_matrix(tr.utr, tr.mfe), feature_matrix(te.utr, te.mfe)
        m = lgb.LGBMRegressor(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=50,
                              subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=seed, verbose=-1)
        m.fit(Ftr.iloc[i_fit], ytr[i_fit], eval_set=[(Ftr.iloc[i_val], ytr[i_val])],
              callbacks=[lgb.early_stopping(100, verbose=False)])
        pred = m.predict(Fte)
        imp = pd.Series(m.booster_.feature_importance("gain"), index=Ftr.columns).sort_values(ascending=False)
        extra.update(best_iter=int(m.best_iteration_), n_features=Ftr.shape[1],
                     top_features=(imp.head(10) / imp.sum()).round(4).to_dict())

    res = {"dataset": dataset, "model": model_name, "seed": seed, **evaluate(yte, pred, edges),
           "minutes": round((time.time() - t0) / 60, 1), **extra}
    if dataset == "synth":   # how close to the noise ceiling, and how well the hidden true function is recovered
        res["r2_vs_true_mrl"] = float(np.corrcoef(te.rl_true, pred)[0, 1] ** 2)
    np.save(out + "_pred.npy", pred)
    json.dump(res, open(out + ".json", "w"), indent=1)
    print({k: v for k, v in res.items() if k != "confusion_matrix"})
    return res


if __name__ == "__main__":
    run_model(sys.argv[1], sys.argv[2], int(sys.argv[3]) if len(sys.argv) > 3 else 1337)
