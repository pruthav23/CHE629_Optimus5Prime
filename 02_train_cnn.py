"""Train Optimus 5-Prime from scratch on eGFP unmod rep1 (260k train / 20k test)."""
import numpy as np, time, torch, json, sys
from optimus import OptimusCNN, train, predict, r2
seed = int(sys.argv[1]) if len(sys.argv) > 1 else 1337
d = np.load("data/egfp_unmod_1.npz")
mu, sd = d["ytr"].mean(), d["ytr"].std()
t = time.time()
m = train(OptimusCNN(), d["xtr"], (d["ytr"] - mu) / sd, epochs=3, seed=seed)
pred = predict(m, d["xte"]) * sd + mu
res = {"seed": seed, "test_r2": float(r2(d["yte"], pred)), "minutes": (time.time() - t) / 60}
print(res)
torch.save(m.state_dict(), f"results/cnn_seed{seed}.pt")
np.save(f"results/cnn_pred_seed{seed}.npy", pred)
json.dump(res, open(f"results/cnn_seed{seed}.json", "w"))
