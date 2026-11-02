# Optimus 5-Prime: synthetic test, reproduction and benchmarks (Sample et al., Nat Biotech 2019)

PyTorch re-implementation of the MRL CNN from `pjsample/human_5utr_modeling`, trained on GEO GSE114002.
The CNN is tested **first on synthetic data with known rules, then on the real eGFP data**, against **three benchmarks**:
a BiLSTM, LightGBM on hand-made UTR features, and the paper's position-specific k-mer linear model.

## Benchmark results (test set n = 20,000; neural nets mean ± sd over 3 seeds)

### Synthetic (noise ceiling r² = 0.918)

| Model | r² | Spearman | RMSE | MAE | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Optimus 5-Prime CNN | 0.856 ± 0.006 | 0.889 | 0.603 | 0.478 | 0.753 | 0.754 | 0.753 | 0.751 |
| BiLSTM | 0.748 ± 0.063 | 0.847 | 0.793 | 0.621 | 0.711 | 0.712 | 0.711 | 0.711 |
| LightGBM (hand features) | 0.915 | 0.936 | 0.456 | 0.364 | 0.815 | 0.815 | 0.814 | 0.815 |
| k-mer linear | 0.738 | 0.848 | 0.801 | 0.629 | 0.700 | 0.714 | 0.700 | 0.703 |

### Real eGFP (top-20k test)

| Model | r² | Spearman | RMSE | MAE | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|---|---|---|---|
| Optimus 5-Prime CNN | 0.919 ± 0.006 | 0.942 | 0.547 | 0.397 | 0.869 | 0.822 | 0.822 | 0.819 |
| BiLSTM | 0.866 ± 0.026 | 0.907 | 0.746 | 0.552 | 0.819 | 0.773 | 0.769 | 0.763 |
| LightGBM (hand features) | 0.893 | 0.922 | 0.620 | 0.449 | 0.848 | 0.795 | 0.791 | 0.791 |
| k-mer linear | 0.640 | 0.832 | 1.155 | 0.889 | 0.696 | 0.723 | 0.659 | 0.636 |

Classification metrics: MRL is binned into low/medium/high using the training-set tertiles. Precision, recall and F1 are macro-averaged.

### Synthetic stage: what the simulator plants (`synth.py`)
MRL_true = 1 + 7 · exp(0.025·MFE) · Π(1 − r_i) over upstream AUGs. Repression r_i = base × Kozak strength, with base 0.40 for a uORF (in-frame stop before the CDS), 0.75 for an overlapping ORF, and 0.12 for an in-frame extension. Observed = true + N(0, 0.45²), giving a noise ceiling of r² = 0.918.

### Key findings
- **The CNN learns the planted uAUG rules almost exactly.** Inserting one ATG into uAUG-free test UTRs changes MRL by −0.43 / −1.38 / −2.39 for in-frame / uORF / overlapping ORF under the CNN, against −0.40 / −1.36 / −2.52 in the simulator (r = 0.95 across all insertions; `figures/bench/synth_atg_insertion.png`).
- **The CNN's remaining synthetic error is RNA structure.** Against the true MRL it reaches r² = 0.92 on UTRs with a uAUG but only 0.54 on uAUG-free UTRs, where folding energy is the only thing that varies. Residuals correlate with MFE (r = −0.53). Adding structure information is a well-supported direction for improving the model.
- **The BiLSTM is unstable within the paper's 3-epoch budget.** One of 3 seeds learned the frame rule (synthetic r² 0.84, insertion r = 0.90). The other two only learned "ATG is bad" (r² ≈ 0.70, insertion r ≈ 0.6). The CNN learned the rule on every seed.
- **LightGBM is a strong baseline, but it starts with an advantage.** Its features (uAUG type, Kozak, MFE) encode the domain knowledge by hand. On synthetic data these features are essentially the answer key, so its 0.915 there is near-oracle. On real data it reaches 0.893, still below the CNN, which learns everything from raw sequence.
- **k-mer linear** is weakest on both datasets (real 0.640 here with k picked on a validation split; 0.655 at k=6 with 5-fold CV earlier; paper 0.66).

### Protocol (identical for every model)
- Real split = the paper's: top 280k UTRs by reads, top 20k = test. Synthetic split: 260k train / 20k test, random.
- Neural nets: the paper's fixed recipe (Adam 1e-3, batch 128, MSE, 3 epochs), with MRL standardised using training-set statistics.
- Classical models: the last 10% of the shuffled training set is a validation split, used only for choosing k and alpha (k-mer) and for early stopping (LightGBM).

## Paper reproduction (earlier step)


| Experiment | Ours | Paper |
|---|---|---|
| CNN, eGFP unmod rep1, held-out top-20k test | **0.919 ± 0.006** (seeds: 0.924, 0.921, 0.910) | 0.93 (Fig 2b) |
| Authors' `main_MRL_model.hdf5` loaded into our PyTorch model, same test set | 0.934 | 0.93 |
| Position-specific k-mer ridge regression (k = 1..6) | 0.128 / 0.353 / 0.628 / 0.640 / 0.652 / **0.655** | best 0.66 (Supp Fig 6-7) |
| Human 5'UTRs + ClinVar SNVs, top 25k by reads | **0.784** (authors' original weights: 0.789) | 0.81 (Fig 4a, retrained model) |
| Retrained CNN (random + designed sub-libraries), human UTRs | 0.771 (random test: 0.915) | — |

Qualitative: Fig 1c's frame effect reproduces. Out-of-frame uAUGs depress MRL to ~4.5–5.5, while in-frame uAUGs are milder and approach no-uAUG levels near the CDS.

## Notes / deviations
- **Split:** identical to the original notebook. Sort by `total_reads`, keep the top 280k; the first 20k are the test set, the remaining 260k the training set.
- **Scaling:** the original `test_data()` fits the inverse StandardScaler on the *test* MRLs; we use train-set mean/sd. r² (squared Pearson) is invariant to this.
- **Keras 'same' padding** for even kernels (k=8) pads 3 left / 4 right; `SamePad` replicates this so the authors' weights load exactly (verified: 0.934).
- **Seed variance:** with 3 epochs, no LR schedule and no early stopping, r² varies ~±0.01 across seeds. The paper reports one run.
- **Human UTR gap (0.78 vs 0.81):** the paper uses a model retrained on extra designed-library data. Neither those weights (`retrained_main_MRL_model.hdf5`) nor the exact recipe are in the repo. Our guess at the recipe didn't close the gap. The authors' own *original* model scores 0.789 here, so the pipeline is right.
- Predictions on human UTRs are systematically higher than observed (an offset between the two experiments). r² ignores this; report MAE/RMSE too if you use these numbers.
- k-mer baseline: ridge, alpha in {0.1…1000} chosen by 5-fold CV on the training set (the paper says "regularisation + 5-fold CV" without more detail).

## Files
- `optimus5prime_reproduction.ipynb`: Colab notebook. Part A synthetic → Part B real → Part C comparison → Part D human UTRs (set `DATA_DIR`)
- `optimus.py` (CNN, data, training), `synth.py` (simulator), `bench.py` (benchmarks + metrics), `runner.py` (shared protocol: `python3 runner.py {synth|real} {cnn|bilstm|lgbm|kmer} [seed]`)
- `10_make_synthetic.py`, `12_eval_saved_cnn.py`, `13_synth_rule_recovery.py`, `14_bench_figures.py`: benchmark-stage scripts
- `00_prep.py` … `06_figures.py`: the earlier reproduction-only scripts
- `results/bench/`: per-run metrics JSON (incl. confusion matrices), predictions, weights, `summary.md` / `summary.csv`
- `figures/bench/`: r² synthetic vs real, confusion matrices, ATG-insertion rule recovery. `figures/`: paper-figure reproductions.

Environment: Python 3, torch 2.x, ViennaRNA 2.x (MFE), LightGBM 4.x. CPU only; ~10 min per CNN and ~5–10 min per BiLSTM on 2 cores.
