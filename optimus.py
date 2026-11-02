"""Optimus 5-Prime reproduction (Sample et al., Nat Biotech 2019) in PyTorch.

Mirrors pjsample/human_5utr_modeling/modeling/training_MRL_CNN.ipynb.
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats

NUC = {"A": 0, "C": 1, "G": 2, "T": 3}


def load_egfp(path, n_keep=280_000, n_test=20_000):
    """Sort by read depth, keep the top n_keep; the top n_test (highest coverage) are the test set."""
    df = pd.read_csv(path, usecols=["utr", "total_reads", "rl"])
    df = df.sort_values("total_reads", ascending=False).reset_index(drop=True).iloc[:n_keep]
    return df.iloc[n_test:].reset_index(drop=True), df.iloc[:n_test].reset_index(drop=True)


def one_hot(seqs, seq_len=50):
    """(N, seq_len, 4) float32. Unknown bases / padding -> all zeros."""
    x = np.zeros((len(seqs), seq_len, 4), dtype=np.float32)
    for i, s in enumerate(seqs):
        for j, c in enumerate(s[:seq_len].upper()):
            k = NUC.get(c)
            if k is not None:
                x[i, j, k] = 1.0
    return x


def r2(obs, pred):
    """Squared Pearson r, as in the paper (scipy linregress r_value**2)."""
    return stats.linregress(obs, pred).rvalue ** 2


class SamePad(nn.Module):
    """Keras 'same' padding for even kernels: pad (k-1)//2 left, the rest right."""
    def __init__(self, k):
        super().__init__()
        self.l, self.r = (k - 1) // 2, k - 1 - (k - 1) // 2

    def forward(self, x):
        return nn.functional.pad(x, (self.l, self.r))


class OptimusCNN(nn.Module):
    """3x Conv1D(120, k=8, same, ReLU) -> Flatten -> Dense(40, ReLU) -> Dropout(0.2) -> Dense(1)."""
    def __init__(self, seq_len=50, n_filters=120, k=8, n_layers=3, hidden=40,
                 dropout_conv=0.0, dropout_dense=0.2):
        super().__init__()
        layers, c_in = [], 4
        for i in range(n_layers):
            layers += [SamePad(k), nn.Conv1d(c_in, n_filters, k), nn.ReLU()]
            if i >= 1:
                layers.append(nn.Dropout(dropout_conv))
            c_in = n_filters
        self.conv = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Linear(seq_len * n_filters, hidden), nn.ReLU(),
                                  nn.Dropout(dropout_dense), nn.Linear(hidden, 1))

    def forward(self, x):                    # x: (N, L, 4), Keras layout
        h = self.conv(x.transpose(1, 2))     # (N, C, L)
        h = h.transpose(1, 2).flatten(1)     # flatten time-major, same order as Keras
        return self.head(h).squeeze(-1)


def train(model, x, y, epochs=3, batch_size=128, lr=1e-3, seed=1337, log_every=500):
    torch.manual_seed(seed)
    opt = torch.optim.Adam(model.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8)
    X, Y = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))
    n = len(X)
    for ep in range(epochs):
        model.train()
        perm = torch.randperm(n)
        tot = 0.0
        for b, i in enumerate(range(0, n, batch_size)):
            idx = perm[i:i + batch_size]
            loss = nn.functional.mse_loss(model(X[idx]), Y[idx])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += loss.item() * len(idx)
            if log_every and b % log_every == 0:
                print(f"  epoch {ep+1} batch {b}/{n//batch_size} loss {loss.item():.4f}", flush=True)
        print(f"epoch {ep+1}: train MSE {tot/n:.4f}", flush=True)
    return model


@torch.no_grad()
def predict(model, x, batch_size=4096):
    model.eval()
    return np.concatenate([model(torch.from_numpy(x[i:i + batch_size])).numpy()
                           for i in range(0, len(x), batch_size)])


def load_keras_weights(model, h5_path):
    """Load the authors' trained Keras weights (e.g. main_MRL_model.hdf5) into OptimusCNN."""
    import h5py
    convs = [m for m in model.conv if isinstance(m, nn.Conv1d)]
    denses = [m for m in model.head if isinstance(m, nn.Linear)]
    with h5py.File(h5_path, "r") as f:
        g = f["model_weights"]
        names = sorted([k for k in g.keys() if k.startswith("conv1d")], key=lambda s: int(s.split("_")[-1]))
        dnames = sorted([k for k in g.keys() if k.startswith("dense")], key=lambda s: int(s.split("_")[-1]))

        def get(name):
            grp = g[name]
            while "kernel:0" not in grp:
                grp = grp[list(grp.keys())[0]]
            return np.array(grp["kernel:0"]), np.array(grp["bias:0"])

        for m, nm in zip(convs, names):
            w, b = get(nm)                                   # keras (k, in, out)
            m.weight.data = torch.from_numpy(w.transpose(2, 1, 0).copy())
            m.bias.data = torch.from_numpy(b)
        for m, nm in zip(denses, dnames):
            w, b = get(nm)                                   # keras (in, out)
            m.weight.data = torch.from_numpy(w.T.copy())
            m.bias.data = torch.from_numpy(b)
    return model
