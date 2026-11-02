import numpy as np
from optimus import OptimusCNN, load_keras_weights, predict, r2
d = np.load("data/egfp_unmod_1.npz")
m = load_keras_weights(OptimusCNN(), "/home/claude/ext/human_5utr_modeling/modeling/saved_models/main_MRL_model.hdf5")
p = predict(m, d["xte"])
print("authors' saved model, test r2 =", round(r2(d["yte"], p), 4))
