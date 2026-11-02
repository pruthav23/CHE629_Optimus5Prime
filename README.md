# Beyond Optimus 5-Prime — CHE629 Course Project

Tushar Srikrishna Sridhar (231103) · Anika Gupta (230137) · Prutha Vinay (230807)
Indian Institute of Technology Kanpur

Reproduction of **Sample et al. (2019)**, *"Human 5′ UTR design and variant effect prediction
from a massively parallel translation assay"* (Nature Biotechnology 37:803–809, GEO GSE114002),
plus four extensions.

**Report:** [`paper/main.pdf`](paper/main.pdf) (ICML format, 9 pages) · **Source:** `paper/main.tex`

---

## Quick start — live demo (~90 seconds, CPU, no training required)

```bash
pip install -r requirements.txt
python3 demo.py                 # all four sections
```

| Section | Command | What it shows |
|---|---|---|
| 1. Reproduction | `python3 demo.py reproduce` | Paper CNN (r² 0.923) and our hybrid (r² 0.941) on the 20k held-out test set |
| 2. Benchmarks | `python3 demo.py benchmarks` | Unified table: r², Spearman, RMSE, accuracy, precision, recall, F1 |
| 3. Variant effect | `python3 demo.py variant` | Saturation mutagenesis: the ClinVar variants of CPOX, TMEM127 and RPL5 rank #1 of 150 |
| 4. Live training | `python3 demo.py train` | Trains a model from scratch on 5,000 sequences in ~40 s |

Sections 1–3 need only `data/test.csv` and the shipped weights in `weights/`.
Section 4 additionally needs `data/train.csv` (regenerate with `python3 00_prep.py <GEO file>`).

---

## Where each graded component lives

| Component | Marks | Where |
|---|---|---|
| **Paper understanding** | 15 | `paper/main.tex` §2 (biological problem, dataset, DL formulation) |
| **Reproduction** (synthetic + paper dataset, train/val/test, preprocessing) | 25 | `synth.py`, `10_make_synthetic.py`, `00_prep.py`–`06_figures.py`, `novelty/data.py`; report §3–4 |
| **Benchmark comparison** (3 benchmarks, accuracy/precision/recall/F1/confusion) | 20 | `bench.py`, `runner.py`, `novelty/unified_metrics.py`; `results/unified/summary.md`; report §5 |
| **Novelty / extension** | 40 | `novelty/partA–D*.py`; `NOVELTY_RESULTS.md`; report §6 |

---

## Headline results

Real eGFP held-out test set (20,000 UTRs, never used for training or model selection):

| Model | Role | r² | Accuracy | F1 (macro) |
|---|---|---|---|---|
| Optimus 5-Prime CNN | paper reproduction | 0.919 ± 0.006 | 0.869 | 0.819 |
| *k*-mer linear | benchmark 1 | 0.640 | 0.696 | 0.636 |
| LightGBM (hand features) | benchmark 2 | 0.893 | 0.848 | 0.791 |
| BiLSTM | benchmark 3 | 0.866 ± 0.026 | 0.819 | 0.763 |
| **CNN + attention hybrid** | **ours** | **0.942 ± 0.001** | **0.886** | **0.841** |

Published value: r² = 0.93. Loading the authors' own Keras weights into our PyTorch
implementation gives r² = 0.934, which validates the pipeline.

Extensions (full detail in `NOVELTY_RESULTS.md`):

- **Chemistry-conditioned model** — one shared model beats three separate ones on Ψ (+0.012) and m1Ψ (+0.020 r²).
- **Architecture × data regime** — hybrid 0.942 vs CNN 0.922 at 260k; the pure transformer fails below 260k.
- **Transfer** — fine-tuning on 1,000 Ψ sequences reaches r² 0.738, i.e. 95% of a model trained on 260k.
- **Interpretability** — TF-MoDISco recovers the Kozak-context uAUG motif *de novo*; the hybrid learns the same grammar as the CNN (ISM agreement r = 0.96).
- **Two negative results, reported in full** — a Siamese paired model does not beat subtraction for variant effects; the paper's secondary-structure/chemistry claim does not reproduce.

Detailed reproduction notes (including deviations from the original code) are in
[`docs/REPRODUCTION_NOTES.md`](docs/REPRODUCTION_NOTES.md).

---

## Data

Download from GEO **GSE114002** (not redistributed here):

| File | Used for |
|---|---|
| `GSM3130435_egfp_unmod_1.csv.gz` | main training/test data (unmodified, U) |
| `GSM3130436_egfp_unmod_2.csv.gz` | replicate, for the noise ceiling |
| `GSM3130437/38_egfp_pseudo_1/2.csv.gz` | pseudouridine (Ψ) |
| `GSM3130439/40_egfp_m1pseudo_1/2.csv.gz` | 1-methylpseudouridine (m1Ψ) |
| `GSM3130443_designed_library.csv.gz` | human 5′UTRs + ClinVar variants |

Paths are set at the top of `novelty/data.py`. **Note:** the Ψ/m1Ψ files store 59 nt per row
(50 nt UTR + 9 nt CDS) and must be trimmed to the first 50 nt, otherwise they do not match the
unmodified library at all. `novelty/data.py` does this.

## Full pipeline

```bash
# reproduction + benchmarks
python3 00_prep.py <path to GSM3130435_egfp_unmod_1.csv.gz>
python3 01_check_author_weights.py      # authors' Keras weights -> our model, r2 0.934
python3 10_make_synthetic.py            # synthetic dataset with known rules
python3 runner.py synth cnn 1337        # {synth|real} x {cnn|bilstm|lgbm|kmer} x seed
python3 runner.py real  cnn 1337
python3 14_bench_figures.py

# extensions (order matters: later parts load Part A weights)
python3 novelty/partA.py
python3 novelty/partB.py small && python3 novelty/partB.py chem
python3 novelty/partC.py && python3 novelty/partC_sat.py
python3 novelty/partD.py chem
python3 novelty/partB.py full && python3 novelty/partB.py randsplit
python3 novelty/partD.py arch
python3 novelty/partA_recal.py && python3 novelty/partB_sens.py

# tables and figures
python3 novelty/report.py all && python3 novelty/figs_cd.py && python3 novelty/unified_metrics.py
```

Every stage skips work that is already on disk, so it is safe to re-run.

## Layout

```
optimus.py        paper CNN, data loading, one-hot, Keras-weight loader
synth.py          synthetic 5'UTR simulator (known rules, computable noise ceiling)
bench.py          benchmark models + shared metrics (incl. classification)
runner.py         one model on one dataset under the shared protocol
novelty/
  data.py         chemistry table, nested subsets, variant pairs, splits
  models.py       CNN, transformer, hybrid, multi-chemistry, Siamese
  train.py        training loop with early stopping
  evaluate.py     regression + variant-effect metrics
  attribution.py  integrated gradients, ISM, k-mer enrichment, motif annotation
  common.py       shared run protocol (saves metrics, predictions, weights)
  partA-D*.py     the four extensions
  report.py, figs_cd.py, unified_metrics.py   tables and figures
demo.py           live presentation demo
scripts/          weight export for the demo
paper/            ICML report (main.tex, refs.bib, figs/)
results/          per-run metrics, predictions, summaries (one JSON per run)
figures/          every figure in the report
weights/          the few trained weights the demo needs
NOVELTY_RESULTS.md  full extension write-up with all tables
```

## Reproducibility notes

- Seeds are fixed and logged in every result JSON; training subset indices are saved under
  `results/novelty/partB/subsets/`.
- The **20,000-UTR test set is never used** for hyperparameter choice, early stopping or
  calibration. All model selection uses a validation split carved from the training pool
  (10% at full data, 20% in small-data regimes).
- Each result JSON records the config, seed, parameter count, best epoch and the full
  validation curve.
- Environment: Python 3, PyTorch 2.x (CPU), ViennaRNA 2.x, LightGBM 4.x, modisco-lite.
  All experiments ran on 2 CPU cores without a GPU; this imposed a 15-epoch cap at full data,
  which is stated in the report wherever it limits a conclusion.

## License

Code released under the MIT License (`LICENSE`). Data belongs to Sample et al. and GEO.
