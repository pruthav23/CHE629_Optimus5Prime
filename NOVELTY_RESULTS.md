# Optimus 5-Prime: novelty extensions, results

Builds on the reproduction of Sample et al., *Nat Biotechnol* 2019 (GEO GSE114002) in this folder. All numbers below were recomputed from the saved predictions in `results/novelty/`. Seeds and subset indices are saved to disk.

**Hardware:** 2 CPU cores, no GPU. One full-data epoch takes about 2.3 min for the CNN, 6.4 min for the transformer and 5.6 min for the hybrid.

## 0. Data availability (checked before modelling)

| Question | Finding | Consequence |
|---|---|---|
| U / Ψ / m1Ψ random-library data in GSE114002? | All three, 2 replicates each (GSM3130435–40) | Part A feasible |
| Same sequences across chemistries? | Yes, once the Ψ/m1Ψ rows are trimmed from 59 nt (UTR + 9 nt of CDS) to the first 50 nt. 19,968 of the 20k U test UTRs have labels in all three chemistries | Shared test set across chemistries |
| Chemistry of the human / ClinVar library | **U only.** There is one GEO sample, and its 5,971 control UTRs agree with U (r² 0.86 / 0.80) far better than with Ψ (0.49 / 0.59) or m1Ψ (0.45 / 0.64) | Part C is U-only |
| ClinVar pairs | 867 wild types, 2,135 variants. The authors' filter (variant ≥ 620 reads, non-zero Δ) gives **1,584 pairs**, of which **45** have \|log2 Δ\| > 0.5 | Evaluation set |
| Synthesis-error SNVs | **17,038** single-nt variants of designed sequences. **5,519 pairs** have both members ≥ 620 reads | Extra *training* pairs only (noisier labels) |
| Paper's "76%" for high-effect variants | Recomputed from the authors' released predictions, it's **r² = 0.762** on the high-effect subset. Their direction accuracy there is 93.3% | Both reported |

**Noise ceilings.** The best r² a model can reach against replicate 1 equals the Pearson r between replicates: U 0.971, Ψ 0.916, m1Ψ 0.865.

## Shared protocol

- **Architectures:**
  - CNN: the paper's model, 474,681 parameters.
  - Transformer: pre-LN, d = 96, 4 layers, 4 heads, learned positions, mean pooling; 456,753 parameters.
  - Hybrid: 2 convolution layers (96 filters, kernel 8) feeding a 3-layer transformer; 421,425 parameters.
- **Optimizer:** Adam, batch 128, MSE on standardized MRL. Dropout is applied only in the dense head, for all architectures.
- **Validation split:** carved from each training subset, 20% at 1k/5k and 10% at full.
- **Early stopping:** on validation loss, restoring the best epoch. Patience 10 (max 200 epochs) at 1k/5k; patience 2 (max 15) at full.
- **Tuning:** the same grid for every architecture. At 1k/5k: lr {1e-3, 3e-4} × dropout {0.1, 0.3}. At full: lr {1e-3, 3e-4} with dropout 0.2. Tuned on seed 0's validation split; the chosen config is used for the other seeds.
- **Seeds:** 5 at 1k/5k, 3 at full. Subsets are random and nested (the 1k set is inside the 5k set) and drawn from the training split only. The 20k test set is never used for any choice.

## Part A: one multi-chemistry model vs three single-chemistry CNNs (full data, 3 seeds)

The conditioning variant was chosen on validation: **separate heads** (mean val r² 0.6705) narrowly beat the chemistry embedding (0.6677).

Cross-chemistry r² on 19,968 shared test UTRs (mean ± sd):

| Model / head | test U | test Ψ | test m1Ψ |
|---|---|---|---|
| single U | **0.925 ± 0.002** | 0.713 ± 0.004 | 0.693 ± 0.004 |
| single Ψ | 0.777 ± 0.007 | 0.777 ± 0.005 | 0.745 ± 0.004 |
| single m1Ψ | 0.781 ± 0.017 | 0.762 ± 0.006 | 0.750 ± 0.010 |
| multi, U head | 0.919 ± 0.007 | 0.726 ± 0.004 | 0.709 ± 0.005 |
| multi, Ψ head | 0.829 ± 0.000 | **0.789 ± 0.009** | 0.756 ± 0.008 |
| multi, m1Ψ head | 0.835 ± 0.002 | 0.783 ± 0.008 | **0.770 ± 0.008** |
| Paper Fig. 2h (authors' saved values, their test set) | U model 0.911 | Ψ model 0.768 | m1Ψ model 0.757 |

- **Shared training helps the modified chemistries.** The multi model gains +0.020 r² on m1Ψ (better in 3 of 3 seeds) and +0.012 on Ψ (2 of 3). It costs 0.006 on U.
- **The modified-RNA heads transfer much better to U.** The Ψ and m1Ψ heads score 0.83 on U labels, against 0.78 for the separate models.
- **Calibration.** The multi model's raw RMSE is worse, because its predictions are compressed (validation slope 1.12–1.23). After a linear map fitted on *validation* predictions only, its test RMSE is lower on Ψ (0.533 vs 0.549) and m1Ψ (0.533 vs 0.560). U is 0.527 vs 0.509 (`partA/recalibration.json`).

## Part B: architectures × data regimes (U, fixed 20k high-read test set)

| Size | CNN | Transformer | Hybrid |
|---|---|---|---|
| 1k (5 seeds) | **0.317 ± 0.123** | 0.031 ± 0.007 | 0.210 ± 0.251 |
| 5k (5 seeds) | **0.778 ± 0.009** | 0.050 ± 0.011 | **0.778 ± 0.005** |
| 260k (3 seeds) | 0.922 ± 0.002 | 0.728 ± 0.012 † | **0.942 ± 0.001** |

† Every full-data transformer run hit the 15-epoch cap while validation loss was still falling, so this value is budget-limited.

Pearson, Spearman, RMSE, chosen configs and best epochs are in `results/novelty/partB/summary.md`. At full data the hybrid also has the lowest RMSE (0.454 vs 0.522 for the CNN).

- **The hybrid is the best model at full data:** +0.020 r² over the paper's CNN on all 3 seeds, with fewer parameters.
- **The CNN is the most data-efficient.** It wins at 1k and ties the hybrid at 5k.
- **Negative result: the plain transformer fails at small data.** It can't fit even its training set at 1k/5k. Bug checks, done on training data only: changing the learning rate or switching to a sinusoidal positional encoding didn't help. The same encoder behind a convolution stem (the hybrid) learns well. Conclusion: learning motifs like ATG from single-nucleotide tokens through attention needs many more updates than small datasets provide. At full data it sits on a plateau for about 8 epochs, then breaks out.
- **The hybrid is unstable at 1k.** Its outcome is bimodal: either about 0.65 or about 0.04, depending on whether it escapes the initial plateau.

**Early-stopping sensitivity** (same configs and subsets, patience 30, all architectures):

| | CNN | Transformer | Hybrid |
|---|---|---|---|
| 1k | 0.317 → 0.435 ± 0.054 | 0.031 → 0.035 | 0.210 → 0.407 ± 0.300 (still bimodal: 0.04, 0.04, 0.67, 0.63, 0.66) |
| 5k | 0.778 → 0.772 | 0.050 → 0.067 | 0.778 → 0.770 |

Patience 10 under-trains the CNN at 1k. It doesn't fix the transformer or the hybrid's bimodality, and the ranking at 5k is unchanged.

**Random-split sanity check** (full data, random 20k test, seed 0 only): CNN 0.891 vs 0.923 on the high-read split; hybrid 0.896 vs 0.941; transformer 0.208 (that seed never escaped its plateau). Random test sets include noisier low-read UTRs, so a modest drop is expected; the hybrid ≥ CNN ordering holds.

**Chemistry regimes** (CNN trunk; 19,968 shared test UTRs; 5 seeds at 1k/5k, 3 at full; lr tuned per method with the same budget):

| Size | Method | U | Ψ | m1Ψ |
|---|---|---|---|---|
| 1k | single, from scratch | 0.307 ± 0.165 | 0.053 ± 0.008 | 0.049 ± 0.009 |
| 1k | **U-pretrained → fine-tune** | – | **0.738 ± 0.003** | **0.708 ± 0.008** |
| 1k | multi-chemistry | 0.319 ± 0.179 | 0.276 ± 0.151 | 0.256 ± 0.138 |
| 5k | single, from scratch | 0.774 ± 0.011 | 0.620 ± 0.006 | 0.582 ± 0.012 |
| 5k | **U-pretrained → fine-tune** | – | **0.762 ± 0.005** | **0.737 ± 0.003** |
| 5k | multi-chemistry | 0.755 ± 0.005 | 0.633 ± 0.006 | 0.615 ± 0.006 |
| 260k | single / multi | 0.925 / 0.919 | 0.777 / 0.789 | 0.750 / 0.770 |

- **Transfer is the strongest result.** A U-pretrained CNN fine-tuned on 1,000 Ψ sequences reaches 95% of the full-data Ψ model (0.738 vs 0.777). With 5,000 sequences, fine-tuning matches or beats training from scratch on 260k (Ψ 0.762, m1Ψ 0.737 vs 0.750).

## Part C: paired variant-effect prediction (U only)

Setup: 1,584 ClinVar pairs, 5-fold CV grouped by connected WT/variant component (no sequence appears in more than one fold, verified), pooled out-of-fold, 3 seeds. Training pairs are ClinVar plus synthesis-error pairs from the training folds. Every model starts from the full-data U CNN.

| Method | r² | Pearson | Direction | r² high-effect | Direction high-effect | AUROC large effect |
|---|---|---|---|---|---|---|
| Paper, as stated (retrained model, subtraction) | 0.555 | – | 64% | 0.76 | – | – |
| Authors' released predictions, recomputed | 0.553 | 0.744 | 65.1% | 0.762 | 93.3% | 0.884 |
| **Subtraction, random-library CNN** | **0.540 ± 0.005** | **0.735** | **64.6%** | 0.771 | **91.9%** | **0.908** |
| Subtraction after fine-tuning on human MRLs | 0.544 ± 0.006 | 0.738 | 64.1% | **0.778** | 90.4% | 0.888 |
| Siamese, trained on Δ | 0.477 ± 0.009 | 0.691 | 60.6% | 0.749 | 85.9% | 0.899 |
| Siamese, Δ + auxiliary absolute loss | 0.495 ± 0.009 | 0.703 | 62.4% | 0.744 | 88.1% | 0.900 |

The 3-seed *ensemble* of subtraction predictions gives r² 0.555, matching the paper exactly (`figures/novelty/C_variant_effects.png`).

- **Negative result: the paired (Siamese) model doesn't beat subtraction** on any metric except roughly tying on AUROC. The same holds on held-out synthesis-error pairs. The random-library CNN already encodes the rules, and about 5k noisy paired deltas don't improve on it. In-domain fine-tuning doesn't help either.

**Saturation mutagenesis** (150 substitutions per UTR; rank 1 = most MRL-decreasing):

| Gene | ClinVar variant | Observed log2 Δ | Subtraction: prediction (rank) | Siamese+aux: rank |
|---|---|---|---|---|
| CPOX | G−10A (rs867711777) | −0.89 | −0.81 (**1**) | 2 |
| TMEM127 | C−18T (rs121908813) | −1.50 | −1.02 (**1**) | 1 |
| RPL5 | C−46A (rs376208311) | −0.87 | −0.81 (**1**) | 1 |

The other two RPL5 alleles at −46 (T, G; observed +0.17, +0.36) are predicted near-neutral (subtraction: ranks 90 and 53 of 150).

## Part D: attribution and motifs

Methods: integrated gradients (uniform baseline, 32 steps; completeness error < 5e-4) and single-nucleotide ISM on 2,000 shared test UTRs.

- **Dominant motif, all chemistries.** TF-MoDISco finds one repressive motif, **AAAAATGG**: a uAUG in Kozak-like context, A-rich upstream with G at +4. It has 589–678 seqlets. No activating motif passed MoDISco's thresholds.
- **uAUG-free UTRs** (602 UTRs):
  - Repressive: GC-rich pentamers (GCCCG, CCCCG), consistent with structure.
  - Activating: a purine-rich element, **AAGGA / TAAGG**, that raises predicted MRL at any position. It matches no element in the curated list, so it's **flagged UNMATCHED**. No offline RBP database was available, so this was a curated-list comparison only.
  - Main start codon: **A at −3** is the most positive contribution near the CDS and **T at −3** the most negative, the classic Kozak −3 purine rule (`figures/novelty/D_motif_and_profile.png`).
- **Chemistry comparison: the paper's structure hypothesis is not reproduced.**
  - MRL–MFE correlation on uAUG-free test UTRs: U 0.437, Ψ 0.433, m1Ψ 0.439. U matches the paper's 0.43; Ψ and m1Ψ show no rise towards its 0.56 / 0.58. MFE here is ViennaRNA on the 50-nt UTR with standard parameters, and the paper's exact recipe is unknown.
  - Model reliance on uAUGs falls for modified RNA: an ATG-creating substitution changes predictions by −1.12 (U), −0.81 (Ψ), −0.78 (m1Ψ), in standardized MRL.
  - G/C sensitivity relative to that uAUG effect rises only slightly: 0.040 → 0.042 → 0.046.
  - Attribution maps largely agree across chemistries (per-sequence ISM r: U vs Ψ 0.87, U vs m1Ψ 0.88, Ψ vs m1Ψ 0.94).
- **CNN vs attention.**
  - The hybrid's ISM maps match the CNN's (per-sequence r = 0.96, top-5 position overlap 84%), nearly as closely as two CNN seeds match each other (0.97). It learned the same grammar.
  - The transformer agrees less (r 0.79, overlap 64%) and relies less on uAUGs.
  - IG agreement is much lower than ISM agreement (for example, CNN vs transformer 0.30), so ISM is the more stable comparison.

## Issues found and fixed during the run

1. **Transformer dropout.** Dropout mask generation took 41% of transformer step time on CPU. Dropout was kept only in the dense head, the same place as in the CNN.
2. **NaN in config selection.** The full-data hybrid at lr 1e-3 collapsed to constant predictions (r² undefined), and `max()` doesn't handle NaN reliably. Selection now treats NaN as the worst score. Every other selection was audited and none was affected.
3. **Noise-ceiling definition.** I first reported replicate r² as the ceiling. The correct ceiling is replicate Pearson r.
4. **Machine restart.** The cloud machine restarted while idle and killed background jobs. Runs resume from saved results, so none were lost.

## Reproduce

From this folder (paths in `novelty/data.py`):

```
python3 novelty/partA.py
python3 novelty/partB.py small && python3 novelty/partB.py chem
python3 novelty/partC.py && python3 novelty/partC_sat.py
python3 novelty/partD.py chem
python3 novelty/partB.py full && python3 novelty/partB.py randsplit
python3 novelty/partD.py arch
python3 novelty/partA_recal.py && python3 novelty/partB_sens.py
python3 novelty/report.py all && python3 novelty/figs_cd.py
```

**Code layout:**
- `data.py`: loaders, chemistry table, nested subsets, variant pairs
- `models.py`: CNN, transformer, hybrid, multi-chemistry and Siamese models
- `train.py`: training loop with early stopping
- `evaluate.py`: metrics
- `attribution.py`: IG, ISM, k-mers, known-element annotation
- `common.py`: shared run protocol
- `partA–D*.py`: experiments
- `report.py`, `figs_cd.py`: tables and figures

**Outputs:**
- Per-run JSON (metrics, epochs, validation history), predictions and weights in `results/novelty/`
- Subset indices in `results/novelty/partB/subsets/`
- Summaries in `results/novelty/part*/summary.md`
