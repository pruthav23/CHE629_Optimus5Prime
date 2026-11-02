# CHE629 submission checklist

Audited against the project guidelines and the grading scheme.

## Guidelines compliance

| Requirement | Status | Evidence |
|---|---|---|
| Report in ICML format | Done | `paper/main.tex` uses `\usepackage[accepted]{icml2026}` and the ICML title/author macros |
| Page limit 4–10 (strict) | 9 pages | Compiled locally with a stand-in style file. **Re-check in Overleaf** with the official `icml2026.sty` |
| Working code | Done | Full pipeline runs; `demo.py` runs in ~90 s |
| GitHub repository link in the paper | Done | https://github.com/pruthav23/CHE629_Optimus5Prime |
| Code runnable live during the presentation | Done | `demo.py` has four sections, each under ~40 s, using shipped weights |
| Author names | Done | Tushar Srikrishna Sridhar (231103), Anika Gupta (230137), Prutha Vinay (230807) |

## Grading scheme coverage

### Paper understanding — 15
Report §2 covers the biological problem (ribosome scanning, uAUGs, uORFs, Kozak context,
secondary structure), the dataset (MPRA + polysome profiling, 280k random 50-mers, MRL
definition, all six sub-libraries) and the deep-learning formulation (one-hot 50×4 → 3 conv
layers → dense → scalar, MSE on standardised MRL, r² as squared Pearson correlation).

### Reproduction — 25
- **Paper dataset:** r² = 0.919 ± 0.006 over 3 seeds vs 0.93 published. Pipeline independently
  validated by loading the authors' own Keras weights into our PyTorch model (r² = 0.934).
- **Synthetic dataset:** 280k sequences from a leaky-scanning simulator with known rules and a
  computable noise ceiling (0.918); the CNN reaches 0.856 and recovers the planted frame rules
  quantitatively (in-silico ATG insertion, r = 0.95 against ground truth).
- **Preprocessing:** follows the original exactly (top 280k by read depth, top 20k held out).
  The Ψ/m1Ψ 59-nt trimming issue is documented and handled.
- **Train/validation/test:** the published recipe has no validation set (fixed 3 epochs). Every
  experiment beyond the literal reproduction carves a validation split from the training pool
  (10% full data, 20% small data) and uses it for all model selection and early stopping. Under
  that protocol the same architecture gives r² = 0.922 ± 0.002. The test set is never used for
  any choice.
- **Controls:** replicate noise ceilings (U 0.971, Ψ 0.916, m1Ψ 0.865) and a random-split
  control in addition to the paper's read-depth split.

### Benchmark comparison — 20
Three benchmarks, not two: position-specific *k*-mer ridge regression (the paper's own
baseline), LightGBM on 98 hand-crafted biological features, and a BiLSTM. All trained under one
shared protocol on both the synthetic and the real dataset.
Metrics: r², Spearman, RMSE, MAE, plus accuracy and macro precision/recall/F1 after binning MRL
into training-set tertiles, with confusion matrices for every model
(`results/unified/summary.md`, `figures/unified/confusion_matrices.png`, report §5 and Table 1).
Interpretation covers four points, including why LightGBM's synthetic score is near-oracle and
why classification metrics expose a calibration failure that r² hides.

### Novelty / extension — 40
Four extensions (report §6, full detail in `NOVELTY_RESULTS.md`):
1. **Chemistry-conditioned multi-task model** — shared trunk + per-chemistry heads; beats three
   separate models on Ψ (+0.012) and m1Ψ (+0.020 r²) and transfers far better across chemistries.
2. **Architecture × data regime study** — CNN vs transformer vs CNN+attention hybrid at 1k / 5k /
   260k, matched parameter counts and tuning budgets, 5 and 3 seeds. The hybrid reaches
   r² = 0.942 ± 0.001, above the published CNN on every seed.
3. **Cross-chemistry transfer** — fine-tuning on 1,000 Ψ sequences reaches 95% of full-data
   performance.
4. **Attribution and motif discovery** — integrated gradients + ISM + TF-MoDISco; recovers the
   Kozak-context uAUG motif de novo and shows the hybrid learns the same grammar as the CNN.

Plus two negative results reported in full: the Siamese paired variant-effect model does not
beat subtraction, and the paper's secondary-structure/chemistry claim does not reproduce.

## Before you submit

1. Push this code to https://github.com/pruthav23/CHE629_Optimus5Prime and make the repository **public**
   (an evaluator must be able to open the link in the paper).
2. Fill in the "Author Contributions" section with who did what.
3. Upload the paper to Overleaf with the **official** `icml2026.sty` and confirm it still fits
   within 10 pages.
4. Rehearse `python3 demo.py` once on the machine you will present from.

## Known limitations stated in the report

All experiments ran on 2 CPU cores without a GPU, which forced a 15-epoch cap at full data. The
transformer was still improving when stopped, so its ceiling is unknown, and the hybrid's margin
over the CNN is established only within that shared budget. These are stated in the report
wherever they limit a conclusion.
