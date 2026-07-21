# AI4WINE: A Generative Machine Learning Benchmark for Wine Quality Classification

Benchmark of three open-weights instruction-tuned local LLMs (12–14 B parameters) on a
3-class ordinal wine quality classification task over expert review text, across four
prompting strategies, with protocol-matched supervised baselines, bootstrap confidence
intervals, and exact McNemar significance tests.

**Models compared** (all served via Ollama at Q4\_K\_M quantisation):
- **Qwen-2.5-14B-Instruct** (Alibaba)
- **Mistral-Nemo-12B-Instruct** (Mistral AI)
- **Gemma-3-12B-Instruct** (Google)

**Prompting strategies:**
- Zero-shot
- Few-shot with K=4 random exemplars per class (12 demonstrations total)
- Retrieval-Augmented few-shot, **class-balanced** (K=6, top-2 per class via BGE cosine similarity)
- Retrieval-Augmented few-shot, **global top-K** (K=6, nearest neighbours regardless of class)
- Self-consistency (majority vote over N=3 sampled completions, temp=0.6) applied to the
  highest-accuracy single-shot cell

**Supervised baselines** (evaluated under the identical input serialisation and test protocol):
majority-class, TF-IDF + Logistic Regression (default and class-balanced).

**Dataset:** [`james-burton/wine_reviews_ordinal`](https://huggingface.co/datasets/james-burton/wine_reviews_ordinal)
— 105 154 expert wine reviews from WineMag.com (Burton 2023, originally compiled by
Thoutt 2017), with predefined train / val / test splits (71 504 / 12 619 / 21 031). The
integer `variety` code (0–29) is decoded to grape-variety names via the sibling
`james-burton/wine_reviews` release (verified by joining on review text).

**Task:** 3-class quality classification (Low / Medium / High) via WineEnthusiast
editorial tiers collapsed onto fixed thresholds:

| Class | Range | WineEnthusiast tier(s) | Train prior |
|---|---|---|---|
| 0 - Low    | 80–82 | acceptable                  | 2.3%  |
| 1 - Medium | 83–90 | good + very good            | 70.3% |
| 2 - High   | 91+   | excellent + superb + classic | 27.4% |

**Hardware:** Intel Core 7 240H (16 cores) + NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM),
CUDA 12.8.

---

## Headline Result

The best single configuration is **Gemma-3-12B with global retrieval-augmented prompting**,
attaining **0.868 accuracy** (95% CI [0.838, 0.896]) on the stratified 500-review test
subsample, together with the **highest macro-F₁ (0.744)** of every system evaluated —
LLM or supervised (κ = 0.678, MCC = 0.683, weighted F₁ = 0.865). Mistral-Nemo-12B with
class-balanced retrieval follows closely at 0.862.

These results are obtained with **no fine-tuning** — only prompt engineering on frozen
Q4-quantised models. Retrieval augmentation lifts Mistral-Nemo-12B by 8.0 pp over
zero-shot (exact McNemar p < 0.001), and the strongest prompted LLMs reach **statistical
parity** (p = 0.435) with a TF-IDF + logistic-regression baseline trained on the full
71.5k-review corpus. Comparisons with fine-tuned encoder results reported in prior work
use different corpora and protocols and are treated as context, not head-to-head.

---

## Setup

### 1. Create Python Environment (Windows 11)

Open **PowerShell** (NOT Git Bash):

```powershell
conda create -n wine-quality python=3.11 -y
conda activate wine-quality
```

### 2. Install PyTorch with CUDA 12.8

```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA works:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### 3. Install ML, NLP & Notebook Packages

```powershell
pip install datasets ollama sentence-transformers ^
            pandas numpy matplotlib seaborn scipy scikit-learn tqdm ^
            ipykernel jupyter
```

### 4. Register Jupyter Kernel for VSCode

```powershell
python -m ipykernel install --user --name wine-quality --display-name "Python (wine-quality)"
```

### 5. Install Ollama and Pull the Three LLMs

1. **Download & install Ollama:** https://ollama.com/download (Windows installer).

2. **Start the Ollama server in a dedicated PowerShell window** (leave open during runs):
   ```powershell
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" serve
   ```

3. **Pull the three models** (one-time, ~22 GB total on disc):
   ```powershell
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull qwen2.5:14b
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull mistral-nemo:12b
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull gemma3:12b
   ```

   Verify all three are present:
   ```powershell
   curl http://localhost:11434/api/tags
   ```
   Should return JSON listing the three models.

---

## Running the Notebook

1. Open `wine_llm_comparison.ipynb` in VSCode and select the
   **Python (wine-quality)** kernel.
2. Confirm the Ollama server is running (Setup §5.2). The Sanity Check cell (§7b) fails
   fast if any model is unreachable.
3. Run cells top-to-bottom. Encoding the full 71.5k-review retrieval pool (§5) takes
   ~10 minutes on the GPU; the main sweep (§8) is the long stage (~2–3 h total). Every
   completed (model, strategy) run is checkpointed to `llm_runs_checkpoint.pkl`, so an
   interrupted run resumes where it left off on re-execution.

---

## Configuration Knobs (configuration cell)

| Constant | Value | Notes |
|---|---|---|
| `N_TEST_EVAL`       | 500    | class-proportional test subset size (realised 501: 11 / 352 / 138) |
| `N_RETRIEVAL_POOL`  | `None` | `None` uses the full training split (71 504) as the retrieval pool |
| `FEWSHOT_K_RANDOM`  | 4      | random exemplars per class for the few-shot strategy |
| `RETRIEVAL_K`       | 6      | retrieved exemplars total (2 per class for balanced, top-6 for global) |
| `NUM_CTX`           | 4096   | serving context window; prevents silent truncation of long prompts |
| `SC_NUM_SAMPLES`    | 3      | self-consistency votes |
| `SC_TEMPERATURE`    | 0.6    | sampling temperature for self-consistency |
| `N_BOOTSTRAP`       | 10000  | resamples for bootstrap confidence intervals |
| `N_LOW_DIAG`        | 100    | extra minority-class (Low) diagnostic subset size |
| `RANDOM_SEED`       | 42     | reproducibility seed for sampling and shuffling |

The pool size was selected on the **validation split** (`llm_pool_validation.csv`): the
full training pool beat a 3 000-review stratified subsample by +5.3 pp accuracy, and
K = 12 exemplars lost to K = 6, so K = 6 is retained.

The sentence encoder is **BAAI/bge-large-en-v1.5** (1024-dim, frozen), driving both the
class-balanced and global kNN retrieval.

---

## Troubleshooting

### `ollama: command not found` (Git Bash / MINGW64)
Git Bash doesn't inherit Windows `PATH`. Use **PowerShell**:
```powershell
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull mistral-nemo:12b
```

### Wrong kernel selected (`base (Python 3.13.x)`)
Reinstall the kernel and reload VSCode:
```powershell
conda activate wine-quality
python -m ipykernel install --user --name wine-quality --display-name "Python (wine-quality)" --force
```
Then `Ctrl+Shift+P` → "Developer: Reload Window".

### Sanity check returns empty output for a model
Symptom: `[OK] Qwen-2.5-14B latency=1.4s pred=None raw=''` (empty string).
Cause: model warm-up race condition on first call after a cold load into VRAM.
Workaround: re-run the sanity check cell once; the empty return is intermittent and
parses cleanly during the actual sweep.

### Prompts appear truncated / few-shot underperforms
Ollama's default context window (2048 tokens) silently truncates the *start* of longer
prompts, cutting into the system prompt. `NUM_CTX = 4096` in the configuration cell
covers every prompt in this notebook (max ~2.3k tokens); do not lower it.

### Model swap is slow during the sweep
Ollama unloads a model from VRAM when a different one is requested, which can take
10–30 s on first reload. The sweep loops outermost over models so each model is loaded
only once for all its strategies.

---

## Project Structure

```
Wine-Quality/
├── README.md                       ← You are here
├── wine_llm_comparison.ipynb       ← The notebook
├── worked_example.md               ← Full end-to-end prompt example (input → prompt → output)
├── llm_results.csv                 ← Main metrics table (all configurations)
├── llm_stats_ci.csv                ← Bootstrap 95% confidence intervals
├── llm_stats_mcnemar.csv           ← Paired McNemar significance tests
├── llm_per_class.csv               ← Per-class precision / recall / F1
├── llm_low_diagnostic.csv          ← Minority-class (Low) recall diagnostic
├── llm_pool_validation.csv         ← Retrieval-pool size selection (validation split)
├── llm_runs.pkl                    ← Raw run metadata + first 50 outputs per run
├── llm_accuracy_comparison.png     ← Accuracy bar chart
├── llm_accuracy_heatmap.png        ← Model × strategy accuracy heat-map
└── llm_confusion_matrices.png      ← Grid of confusion matrices
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` 2.x (cu128)         | GPU backend for the BGE encoder |
| `sentence-transformers`     | Frozen BGE sentence embeddings (kNN retrieval only) |
| `ollama`                    | Local LLM inference (Qwen 2.5, Mistral-Nemo, Gemma 3) |
| `datasets`                  | HuggingFace dataset loading |
| `scikit-learn`              | Metrics + TF-IDF / logistic-regression baselines |
| `scipy`                     | Bootstrap CIs, McNemar tests, Spearman correlation |
| `pandas`, `numpy`           | Tabular data handling |
| `matplotlib`, `seaborn`     | Plots |

---

## Contributions

1. **Controlled balanced-vs-global retrieval ablation.** Class-balanced (top-2 per class)
   and global (top-6) exemplar selection share an identical prompt, isolating the specific
   effect of class balancing. Its sign is model-dependent: global retrieval is
   significantly better for Gemma-3 and Qwen-2.5, while balancing helps Mistral-Nemo.
2. **Corpus-informed system prompt.** Statistical priors (class base rates, a price ladder,
   a review-length ladder, grape-variety and origin priors, discriminative descriptors)
   are measured **exclusively on the training split** and stated in the prompt as guidance.
3. **Protocol-matched supervised baselines.** Majority-class and TF-IDF + logistic
   regression are evaluated on the exact same serialised input and test subset as the
   LLMs, giving a fair reference point; the strongest prompted LLMs reach statistical
   parity with a model trained on the full corpus.
4. **Statistical validation.** Every headline comparison is reported with a bootstrap 95%
   confidence interval and an exact paired McNemar test, rather than raw point estimates.
5. **Quality-tier binning** from the WineEnthusiast editorial rubric, mapping the six
   published tiers onto a three-class partition that respects the magazine's own
   thresholds rather than a fitted percentile scheme.
6. **Single-digit output protocol** with simplified regex parsing in place of the
   structured-JSON convention, eliminating format-drift parse failures across
   heterogeneous LLM backbones (zero parse failures across all deterministic responses).
7. **Honest self-consistency finding.** Applied to the strongest cell, self-consistency
   *decreased* accuracy (0.868 → 0.852): without chain-of-thought to marginalise over,
   stochastic resampling adds decoding noise rather than exploring alternative reasoning.
8. **Minority-class diagnostic.** A dedicated 100-review Low-class evaluation with Wilson
   confidence intervals surfaces the remaining gap between the LLMs and the balanced
   supervised baseline on the rare class.
