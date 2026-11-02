"""Synthetic 5'UTR -> MRL simulator with known, biologically motivated rules.

Each 50-nt UTR sits directly upstream of the CDS start codon (position 50 == CDS AUG).
Ground-truth MRL follows a leaky-scanning model:

    MRL_true = 1 + 7 * structure_factor * prod_i (1 - r_i)

over every upstream AUG i, where the repression r_i depends on
  * whether an in-frame stop codon follows, and the frame relative to the CDS:
      - stop codon before the CDS (uORF, reinitiation possible) ......... base 0.40
      - no stop, in frame with CDS (N-terminal extension) ............... base 0.12
      - no stop, out of frame (overlapping oORF) ......................... base 0.75
  * Kozak context: strength = 0.35 + 0.35*[purine at -3] + 0.30*[G at +4]
and the 5' structure factor = exp(0.025 * MFE) (MFE in kcal/mol, <= 0; ViennaRNA).
Observed MRL = MRL_true + N(0, NOISE_SD).

Because MRL_true is known, the best achievable r2 on the noisy labels is
    var(MRL_true) / (var(MRL_true) + NOISE_SD**2).
"""
import numpy as np
from multiprocessing import Pool

STOPS = {"TAA", "TAG", "TGA"}
BASE_REPRESSION = {"inframe": 0.12, "uorf": 0.40, "oorf": 0.75}
NOISE_SD = 0.45
L = 50


def classify_uaug(seq, p):
    """Class of the ATG starting at p in a UTR of length len(seq) that abuts the CDS."""
    for q in range(p + 3, len(seq) - 2, 3):     # any in-frame stop before the CDS -> self-contained uORF
        if seq[q:q + 3] in STOPS:
            return "uorf"
    if (len(seq) - p) % 3 == 0:                  # reads straight into the CDS in frame
        return "inframe"
    return "oorf"                                # out of frame, overlaps the CDS start


def kozak(seq, p):
    m3 = seq[p - 3] if p >= 3 else "A"          # upstream of UTR is the (purine-rich) vector cap region
    p4 = seq[p + 3] if p + 3 < len(seq) else "G"  # if ATG ends the UTR, +4 is the CDS's first base (G of eGFP GTG)
    return 0.35 + 0.35 * (m3 in "AG") + 0.30 * (p4 == "G")


def uaug_list(seq):
    return [p for p in range(len(seq) - 2) if seq[p:p + 3] == "ATG"]


def true_mrl(seq, mfe):
    keep = 1.0
    for p in uaug_list(seq):
        keep *= 1 - BASE_REPRESSION[classify_uaug(seq, p)] * kozak(seq, p)
    return 1 + 7 * np.exp(0.025 * mfe) * keep


def _mfe(s):
    import RNA
    return RNA.fold(s)[1]


def mfe_batch(seqs, procs=2):
    with Pool(procs) as pool:
        return np.array(pool.map(_mfe, seqs, chunksize=2000), dtype=np.float32)


def make_dataset(n, seed=0, procs=2):
    rng = np.random.default_rng(seed)
    seqs = ["".join(s) for s in rng.choice(list("ACGT"), size=(n, L))]
    mfe = mfe_batch(seqs, procs)
    y_true = np.array([true_mrl(s, m) for s, m in zip(seqs, mfe)], dtype=np.float32)
    y_obs = y_true + rng.normal(0, NOISE_SD, n).astype(np.float32)
    return seqs, mfe, y_true, y_obs
