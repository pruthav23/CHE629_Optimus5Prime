"""Build the Colab notebook: synthetic stage -> real stage -> benchmark comparison."""
import nbformat as nbf
nb = nbf.v4.new_notebook(); C = []
md = lambda s: C.append(nbf.v4.new_markdown_cell(s))
code = lambda s: C.append(nbf.v4.new_code_cell(s.strip()))
summary = open("results/bench/summary.md").read()

md(f"""# Optimus 5-Prime: synthetic-data test, real-data reproduction, and benchmark comparison
Sample et al., *Nat Biotech* 2019. PyTorch re-implementation of `pjsample/human_5utr_modeling`.

**Protocol.** The model under test (the Optimus 5-Prime CNN) and three benchmarks:
**BiLSTM**, **LightGBM on hand-made features**, and the paper's **position-specific k-mer linear model**.
Each is trained and tested first on **synthetic data with known rules (Part A)** and then on the **real eGFP polysome data (Part B)**.
Every model uses the same split. MRL is also binned into low/medium/high (training-set tertiles) for accuracy, precision, recall, F1 and confusion matrices.

**Results obtained (CPU; neural nets mean ± sd over 3 seeds):**

{summary}
Paper reference values: CNN 0.93, k-mer best 0.66 (real data).

Runtime: a Colab GPU is recommended (Runtime > Change runtime type > GPU). On CPU, budget about 2 h for everything.
Put the GEO files in `DATA_DIR`: `GSM3130435_egfp_unmod_1.csv.gz` (required) and `GSM3130443_designed_library.csv.gz` (Part D only).""")

code("""!pip -q install h5py ViennaRNA lightgbm
!git clone -q --depth 1 https://github.com/pjsample/human_5utr_modeling.git
import os
DATA_DIR = "/content/drive/MyDrive/optimus_data"   # <- change me
# from google.colab import drive; drive.mount('/content/drive')
EGFP = f"{DATA_DIR}/GSM3130435_egfp_unmod_1.csv.gz"
DESIGNED = f"{DATA_DIR}/GSM3130443_designed_library.csv.gz"
AUTHOR_H5 = "human_5utr_modeling/modeling/saved_models/main_MRL_model.hdf5"
for d in ("data/synth", "results/bench", "figures/bench"): os.makedirs(d, exist_ok=True)""")

md("## 0. Code modules\nWritten to files so the multiprocessing MFE step and the scripts can import them.\n\n"
   "**`optimus.py`**: the Optimus 5-Prime CNN (3x Conv1D(120, k=8) > Dense(40) > Dropout 0.2 > 1; Adam 1e-3, batch 128, MSE, 3 epochs), data loading, one-hot encoding, and the Keras-weight loader.")
src = open("optimus.py").read()
src = (src.replace("import torch\n", "import torch\nDEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'\n", 1)
       .replace("    X, Y = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))",
                "    model.to(DEVICE)\n    X, Y = torch.from_numpy(x), torch.from_numpy(y.astype(np.float32))")
       .replace("loss = nn.functional.mse_loss(model(X[idx]), Y[idx])",
                "loss = nn.functional.mse_loss(model(X[idx].to(DEVICE)), Y[idx].to(DEVICE))")
       .replace("    model.eval()\n    return np.concatenate([model(torch.from_numpy(x[i:i + batch_size])).numpy()",
                "    model.eval(); dev = next(model.parameters()).device\n    return np.concatenate([model(torch.from_numpy(x[i:i + batch_size]).to(dev)).cpu().numpy()"))
assert "DEVICE)" in src and ".cpu().numpy()" in src
code("%%writefile optimus.py\n" + src)
md("**`synth.py`**: the synthetic-data simulator. Its docstring lists the planted rules.")
code("%%writefile synth.py\n" + open("synth.py").read())
md("**`bench.py`**: the benchmark models (BiLSTM, k-mer features, hand-made features for LightGBM) and the shared metrics.")
code("%%writefile bench.py\n" + open("bench.py").read())
md("**`runner.py`**: trains one model on one dataset under the shared protocol and saves its metrics and predictions.")
code("%%writefile runner.py\n" + open("runner.py").read().replace("torch.load(", "torch.load(map_location='cpu', f="))
code("from runner import run_model\nfrom optimus import *\nimport json, numpy as np, pandas as pd")

md("""# Part A: synthetic data (known ground truth)
280,000 random 50-nt UTRs. The true MRL comes from planted rules: uORF, overlapping-ORF and in-frame uAUG penalties scaled by Kozak strength, plus a folding-energy penalty. Gaussian noise (sd 0.45) is added on top.
Because the true MRL is known, the **noise ceiling** (the best r² any model can reach) can be computed, and we can check *which rules* each model learned.""")
code("""from synth import make_dataset, NOISE_SD
seqs, mfe, yt, yo = make_dataset(280_000, seed=0)
df = pd.DataFrame({"utr": seqs, "mfe": mfe, "rl_true": yt, "rl": yo})
df.iloc[20_000:].to_csv("data/synth/train.csv", index=False); df.iloc[:20_000].to_csv("data/synth/test.csv", index=False)
te_s = df.iloc[:20_000]
print("MRL mean %.2f sd %.2f | noise ceiling r2 = %.3f" % (df.rl.mean(), df.rl.std(),
      np.var(te_s.rl_true) / (np.var(te_s.rl_true) + NOISE_SD**2)))""")
md("### A1. Train and test all four models on synthetic data")
code("""SEEDS = [1337, 1, 2]
for m in ["cnn", "bilstm"]:
    for s in SEEDS:
        run_model("synth", m, s)
run_model("synth", "lgbm"); run_model("synth", "kmer");""")
md("### A2. Did the networks learn the planted rules?\nAn ATG is inserted at every position of 400 uAUG-free test UTRs, and each model's predicted change in MRL is compared with the true change from the simulator.")
code("%%writefile 13_synth_rule_recovery.py\n" + open("13_synth_rule_recovery.py").read().replace("torch.load(", "torch.load(map_location='cpu', f="))
code("!python 13_synth_rule_recovery.py")
code("""te_s = pd.read_csv("data/synth/test.csv"); p = np.load("results/bench/synth_cnn_s1337_pred.npy")
no = ~te_s.utr.str.contains("ATG")
print("CNN r2 vs TRUE MRL | with uAUG: %.3f   no uAUG (only folding energy varies): %.3f" % (
      np.corrcoef(p[~no], te_s.rl_true[~no])[0,1]**2, np.corrcoef(p[no], te_s.rl_true[no])[0,1]**2))""")

md("""# Part B: real eGFP polysome data (GSE114002, unmodified rep 1)
Same split as the paper: the top 280k UTRs by read depth, of which the top 20k form the test set. MFE is computed for the LightGBM features.""")
code("""tr, te = load_egfp(EGFP)
from synth import mfe_batch
tr["mfe"], te["mfe"] = mfe_batch(list(tr.utr)), mfe_batch(list(te.utr))
tr.to_csv("data/train.csv", index=False); te.to_csv("data/test.csv", index=False)
print(len(tr), len(te))""")
md("### B1. Pipeline check: the authors' trained weights should give r² ≈ 0.93")
code("""m = load_keras_weights(OptimusCNN(), AUTHOR_H5)
print("authors' model test r2:", r2(te.rl.values, predict(m, one_hot(te.utr))))""")
md("### B2. Train and test all four models on real data")
code("""for m in ["cnn", "bilstm"]:
    for s in SEEDS:
        run_model("real", m, s)
run_model("real", "lgbm"); run_model("real", "kmer");""")

md("# Part C: comparison (tables, r² chart, confusion matrices, rule-recovery plot)")
code("%%writefile 14_bench_figures.py\n" + open("14_bench_figures.py").read())
code("!python 14_bench_figures.py")
code("""from IPython.display import Image, display, Markdown
display(Markdown(open("results/bench/summary.md").read()))
for p in ["r2_synth_vs_real", "confusion_matrices", "synth_atg_insertion"]:
    display(Image(f"figures/bench/{p}.png", width=900))""")

md("# Part D (optional): human 5'UTRs + ClinVar SNVs\nPaper Fig 4a reports 0.81 using a *retrained* model whose weights and exact recipe are not in the repo. The original model gives about 0.78–0.79.")
code("""df = pd.read_csv(DESIGNED, low_memory=False)
hu = df[df.library.isin(["human_utrs", "snv"])].sort_values("total", ascending=False).iloc[:25000]
x = one_hot(hu.utr); ytr = pd.read_csv("data/train.csv").rl.values; mu, sd = ytr.mean(), ytr.std()
print("authors' original weights:", r2(hu.rl, predict(load_keras_weights(OptimusCNN(), AUTHOR_H5), x)))
for s in SEEDS:
    m = OptimusCNN(); m.load_state_dict(torch.load(f"results/bench/real_cnn_s{s}.pt", map_location="cpu"))
    print("our CNN seed", s, r2(hu.rl, predict(m, x) * sd + mu))""")
nb.cells = C
nbf.write(nb, "optimus5prime_reproduction.ipynb")
print("written", len(C), "cells")
