import itertools
import numpy as np
import torch


def _f(model, head=None):
    def fn(x):
        out = model(x)
        return out[:, head] if head is not None else out
    return fn


def integrated_gradients(model, x, head=None, steps=32, batch=256):
    """IG from a uniform (0.25) baseline. Returns (contrib, hyp): contrib = IG on the observed base (N, L, 4);
    hyp = mean-centred average path gradients (hypothetical contributions, for TF-MoDISco)."""
    model.eval()
    fn = _f(model, head)
    x = torch.as_tensor(x, dtype=torch.float32)
    base = torch.full_like(x[:1], 0.25)
    contrib, hyp = [], []
    alphas = (torch.arange(steps, dtype=torch.float32) + 0.5) / steps
    for i in range(0, len(x), batch):
        xb = x[i:i + batch]
        g = torch.zeros_like(xb)
        for a in alphas:
            xi = (base + a * (xb - base)).requires_grad_(True)
            fn(xi).sum().backward()
            g += xi.grad
        g /= steps
        contrib.append(((xb - base) * g).detach())
        hyp.append((g - g.mean(-1, keepdim=True)).detach())
    return torch.cat(contrib).numpy(), torch.cat(hyp).numpy()


@torch.no_grad()
def ism(model, x, head=None, batch=4096):
    """Single-nucleotide scanning. Returns (N, L, 4): prediction(mutant) - prediction(reference); 0 at the reference base."""
    model.eval()
    fn = _f(model, head)
    x = np.asarray(x, dtype=np.float32)
    N, L, _ = x.shape
    ref = fn(torch.as_tensor(x)).numpy()
    muts = np.repeat(x[:, None, None], 4, 2).repeat(L, 1).reshape(N, L, 4, L, 4)
    for p in range(L):
        for b in range(4):
            muts[:, p, b, p] = 0
            muts[:, p, b, p, b] = 1
    flat = muts.reshape(-1, L, 4)
    pred = np.concatenate([fn(torch.as_tensor(flat[i:i + batch])).numpy() for i in range(0, len(flat), batch)])
    return pred.reshape(N, L, 4) - ref[:, None, None]


def ism_contrib(delta, x):
    """Contribution of the observed base = ref - mean over the 3 substitutions (positive = observed base raises MRL)."""
    return (-delta.sum(-1) / 3)[..., None] * x


BASES = "ACGT"


def seq_of(x):
    return "".join(BASES[i] for i in x.argmax(-1))


def top_kmers(contrib, x, k=6, top_frac=0.1, sign=-1):
    """k-mer enrichment among high-|attribution| windows. sign=-1: windows with the most negative summed
    contribution (repressive); sign=+1: most positive (activating). Returns DataFrame of k-mer, count, background, log2 enrichment."""
    import pandas as pd
    per_pos = contrib.sum(-1)
    N, L = per_pos.shape
    win = np.stack([per_pos[:, i:i + k].sum(1) for i in range(L - k + 1)], 1)
    score = sign * win
    thr = np.quantile(score, 1 - top_frac)
    seqs = [seq_of(s) for s in x]
    fg, bg = {}, {}
    for n, s in enumerate(seqs):
        for i in range(L - k + 1):
            km = s[i:i + k]
            bg[km] = bg.get(km, 0) + 1
            if score[n, i] >= thr:
                fg[km] = fg.get(km, 0) + 1
    tot_fg, tot_bg = sum(fg.values()), sum(bg.values())
    df = pd.DataFrame([(km, c, bg[km]) for km, c in fg.items()], columns=["kmer", "fg", "bg"])
    df["log2_enrich"] = np.log2(((df.fg + 1) / tot_fg) / ((df.bg + 1) / tot_bg))
    return df[df.fg >= 20].sort_values("log2_enrich", ascending=False).reset_index(drop=True)


KNOWN = {
    "uAUG (ATG)": lambda s: "ATG" in s,
    "Kozak-strong uAUG ([AG]NNATGG)": lambda s: any(s[i:i + 3] == "ATG" and i >= 3 and s[i - 3] in "AG" and i + 3 < len(s) and s[i + 3] == "G" for i in range(len(s))),
    "stop codon (TAA/TAG/TGA)": lambda s: any(c in s for c in ("TAA", "TAG", "TGA")),
    "near-cognate start (CTG/GTG)": lambda s: "CTG" in s or "GTG" in s,
    "GC-rich (>=80% G/C)": lambda s: (s.count("G") + s.count("C")) / len(s) >= 0.8,
    "AU-rich (>=80% A/T)": lambda s: (s.count("A") + s.count("T")) / len(s) >= 0.8,
    "ARE core (ATTTA)": lambda s: "ATTTA" in s,
    "Pumilio (TGTA.ATA)": lambda s: any(s[i:i + 4] == "TGTA" and s[i + 5:i + 8] == "ATA" for i in range(len(s) - 7)),
    "poly-U (TTTTT)": lambda s: "TTTTT" in s,
    "G-run (GGGG)": lambda s: "GGGG" in s,
}


def annotate(consensus):
    hits = [name for name, f in KNOWN.items() if f(consensus)]
    return hits if hits else ["UNMATCHED"]
