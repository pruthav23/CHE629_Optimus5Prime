import math
import torch
import torch.nn as nn
from optimus import SamePad


class CNN(nn.Module):
    """Optimus 5-Prime CNN. encode() returns the 40-d representation."""
    def __init__(self, seq_len=50, n_filters=120, k=8, hidden=40, dropout=0.2):
        super().__init__()
        layers, c = [], 4
        for _ in range(3):
            layers += [SamePad(k), nn.Conv1d(c, n_filters, k), nn.ReLU()]
            c = n_filters
        self.conv = nn.Sequential(*layers)
        self.fc = nn.Sequential(nn.Linear(seq_len * n_filters, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 1)
        self.rep_dim = hidden

    def encode(self, x):
        h = self.conv(x.transpose(1, 2)).transpose(1, 2).flatten(1)
        return self.fc(h)

    def forward(self, x):
        return self.out(self.encode(x)).squeeze(-1)


class Transformer(nn.Module):
    """Pre-LN transformer encoder over the 50 one-hot positions, learned positional embedding, mean pooling.
    Dropout is applied only in the dense head, the same place as in the CNN."""
    def __init__(self, seq_len=50, d=96, layers=4, heads=4, ff=384, hidden=40, dropout=0.1, conv_stem=False):
        super().__init__()
        if conv_stem:
            self.stem = nn.Sequential(SamePad(8), nn.Conv1d(4, d, 8), nn.ReLU(), SamePad(8), nn.Conv1d(d, d, 8), nn.ReLU())
        else:
            self.stem = None
            self.embed = nn.Linear(4, d)
        self.pos = nn.Parameter(torch.randn(1, seq_len, d) * 0.02)
        block = nn.TransformerEncoderLayer(d, heads, ff, 0.0, batch_first=True, norm_first=True, activation="gelu")
        self.enc = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.fc = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(), nn.Dropout(dropout))
        self.out = nn.Linear(hidden, 1)
        self.rep_dim = hidden

    def tokens(self, x):
        h = self.stem(x.transpose(1, 2)).transpose(1, 2) if self.stem is not None else self.embed(x)
        return self.norm(self.enc(h + self.pos))

    def encode(self, x):
        return self.fc(self.tokens(x).mean(1))

    def forward(self, x):
        return self.out(self.encode(x)).squeeze(-1)


def Hybrid(dropout=0.1, **kw):
    """CNN+attention: two conv layers (96 filters, k=8) feeding a 3-layer transformer."""
    return Transformer(d=96, layers=3, heads=4, ff=384, dropout=dropout, conv_stem=True, **kw)


ARCHS = {"cnn": CNN, "transformer": Transformer, "hybrid": Hybrid}


def build(arch, dropout):
    return ARCHS[arch](dropout=dropout)


def n_params(m):
    return sum(p.numel() for p in m.parameters())


class MultiChem(nn.Module):
    """Shared trunk + chemistry conditioning. mode='heads': one MLP head per chemistry;
    mode='embed': chemistry embedding concatenated to the shared representation, one shared MLP head."""
    def __init__(self, trunk, n_chem=3, mode="heads", emb_dim=8, head_hidden=32):
        super().__init__()
        self.trunk, self.mode, self.n_chem = trunk, mode, n_chem
        r = trunk.rep_dim
        if mode == "heads":
            self.heads = nn.ModuleList([nn.Sequential(nn.Linear(r, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))
                                        for _ in range(n_chem)])
        else:
            self.emb = nn.Embedding(n_chem, emb_dim)
            self.head = nn.Sequential(nn.Linear(r + emb_dim, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))

    def forward(self, x):
        """Returns (N, n_chem): the prediction for every chemistry."""
        h = self.trunk.encode(x)
        if self.mode == "heads":
            return torch.cat([hd(h) for hd in self.heads], 1)
        e = self.emb.weight
        hh = torch.cat([h.unsqueeze(1).expand(-1, self.n_chem, -1), e.unsqueeze(0).expand(h.size(0), -1, -1)], -1)
        return self.head(hh).squeeze(-1)


class Siamese(nn.Module):
    """Shared encoder over (WT, variant); head on [h_var - h_wt, h_wt * h_var] -> delta.
    Also returns per-sequence absolute predictions for the auxiliary loss."""
    def __init__(self, trunk, head_hidden=32):
        super().__init__()
        self.trunk = trunk
        r = trunk.rep_dim
        self.delta_head = nn.Sequential(nn.Linear(2 * r, head_hidden), nn.ReLU(), nn.Linear(head_hidden, 1))

    def forward(self, x_wt, x_var):
        hw, hv = self.trunk.encode(x_wt), self.trunk.encode(x_var)
        delta = self.delta_head(torch.cat([hv - hw, hw * hv], 1)).squeeze(-1)
        return delta, self.trunk.out(hw).squeeze(-1), self.trunk.out(hv).squeeze(-1)
