import numpy as np, sys
from optimus import load_egfp, one_hot
tr, te = load_egfp(sys.argv[1])
print(len(tr), len(te), te.total_reads.min(), tr.total_reads.min())
np.savez_compressed("data/egfp_unmod_1.npz", xtr=one_hot(tr.utr), ytr=tr.rl.values, xte=one_hot(te.utr), yte=te.rl.values)
tr.to_csv("data/train.csv", index=False); te.to_csv("data/test.csv", index=False)
