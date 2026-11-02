import numpy as np, pandas as pd, time
from synth import make_dataset, NOISE_SD, mfe_batch
t = time.time()
seqs, mfe, yt, yo = make_dataset(280_000, seed=0)
df = pd.DataFrame({"utr": seqs, "mfe": mfe, "rl_true": yt, "rl": yo})
te, tr = df.iloc[:20_000], df.iloc[20_000:]          # sequences are i.i.d. random, so this is a random split
tr.to_csv("data/synth/train.csv", index=False); te.to_csv("data/synth/test.csv", index=False)
ceil = np.var(te.rl_true) / (np.var(te.rl_true) + NOISE_SD**2)
print(f"synthetic done in {time.time()-t:.0f}s; MRL obs mean {df.rl.mean():.2f} sd {df.rl.std():.2f}; "
      f"r2(true,obs) on test = {np.corrcoef(te.rl_true, te.rl)[0,1]**2:.3f}; theoretical ceiling {ceil:.3f}")
# MFE for the real eGFP data (needed by the LightGBM benchmark)
for split in ("train", "test"):
    d = pd.read_csv(f"data/{split}.csv")
    d["mfe"] = mfe_batch(list(d.utr))
    d.to_csv(f"data/{split}.csv", index=False)
print(f"all done {time.time()-t:.0f}s")
