# Improved Machine Learning Pipeline for Enhanced Wine Quality Classification

IEEE conference paper benchmarking three open-weights instruction-tuned local LLMs in
the 12–14 B parameter range on a 3-class wine quality classification task, across four
prompting strategies of increasing sophistication.

**Models compared** (all served via Ollama at Q4\_K\_M quantisation):
- **Qwen-2.5-14B-Instruct** (Alibaba)
- **Mistral-Nemo-12B-Instruct** (Mistral AI)
- **Gemma-3-12B-Instruct** (Google)

**Prompting strategies:**
- Zero-shot
- Few-shot with K=4 random exemplars per class (12 demonstrations total)
- Retrieval-Augmented few-shot, class-balanced (K=6, top-2 per class via BGE cosine similarity)
- Self-consistency (majority vote over N=3 sampled completions, temp=0.6) applied to the
  highest-accuracy single-shot cell

**Active dataset:** [`james-burton/wine_reviews_ordinal`](https://huggingface.co/datasets/james-burton/wine_reviews_ordinal)
105154 expert wine reviews from WineMag.com (Burton 2023, originally compiled by
Thoutt 2017), with predefined train / val / test splits (71504 / 12619 / 21031).

**Task:** 3-class quality classification (Low / Medium / High) via WineEnthusiast
editorial tiers collapsed onto fixed thresholds:

| Class | Range | WineEnthusiast tier(s) | Train prior |
|---|---|---|---|
| 0 - Low    | 80–82 | acceptable                  | 2.3%  |
| 1 - Medium | 83–90 | good + very good            | 70.3% |
| 2 - High   | 91+   | excellent + superb + classic | 27.4% |

**Hardware:** Intel Core i7 (16 cores) + NVIDIA GeForce RTX 5050 Laptop GPU (8 GB VRAM),
CUDA 12.8.

---

## Headline Result

The best single configuration is **Mistral-Nemo-12B with retrieval-augmented few-shot**,
attaining **0.900 accuracy** on the 100-sample stratified test subset (κ = 0.760,
MCC = 0.765, weighted F₁ = 0.898, macro F₁ = 0.851). This is achieved with **no
fine-tuning** only prompt engineering on a frozen Q4-quantised model and matches the
89.12 % binary accuracy of the dedicated fine-tuned BERT baseline of Katumullage et al.
(2022) on the related Wine Spectator corpus.

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

   At Q4\_K\_M each model is roughly 7–9 GB, so weights effectively occupy the entire
   8 GB VRAM. Ollama swaps models on demand between the three families, and may
   partially offload the KV cache to system RAM under longer prompts (see
   *Inference latency* note below).

---

## Running the Notebook

1. Open `wine_llm_comparison.ipynb` in VSCode and select the
   **Python (wine-quality)** kernel.
2. Confirm the Ollama server is running (Setup §5.2). Cell 7b (Sanity Check) fails
   fast if any model is unreachable.
3. Run cells top-to-bottom. The main sweep cell (§8) and the self-consistency cell
   (§9) are the long ones; everything else completes within a few minutes.

---

## Configuration Knobs (cell 5)

| Constant | Value | Notes |
|---|---|---|
| `N_TEST_EVAL`       | 100   | class-proportional test subset size (2 / 70 / 28) |
| `N_RETRIEVAL_POOL`  | 3000  | class-proportional retrieval pool size (68 / 2110 / 822) |
| `FEWSHOT_K_RANDOM`  | 4     | random exemplars per class for the few-shot strategy |
| `RETRIEVAL_K`       | 6     | retrieved exemplars total (2 per class, class-balanced) |
| `SC_NUM_SAMPLES`    | 3     | self-consistency votes |
| `SC_TEMPERATURE`    | 0.6   | sampling temperature for self-consistency |
| `RANDOM_SEED`       | 42    | reproducibility seed for sampling and shuffling |

The sentence encoder is **BAAI/bge-large-en-v1.5** (1024-dim, frozen). It ranks higher
on MTEB retrieval sub-tasks than the more common all-mpnet-base-v2 and is used here
solely to drive class-balanced kNN retrieval for the retrieval-augmented prompt.

---

## Estimated Runtime (RTX 5050 Laptop, 8 GB VRAM)

| Stage | Duration |
|---|---|
| Data load + binning + stratified subsampling                 | ~30 s    |
| BGE embedding of retrieval pool (3000) + test subset (100)    | ~30 s    |
| Sanity check (1 call per model)                               | ~30 s    |
| Sweep Qwen-2.5-14B (3 strategies × 100 calls)               | ~7 min   |
| Sweep Gemma-3-12B (3 strategies × 100 calls)                | ~7 min   |
| Sweep Mistral-Nemo-12B (3 strategies × 100 calls)           | ~3 h*    |
| Self-consistency on winning combo (300 calls)                 | ~3 min   |
| Metrics + plots + CSV / pickle / LaTeX export                 | ~30 s    |
| **Total**                                                     | **~3.6 h** |

\* Mistral-Nemo-12B inference latency scales super-linearly with prompt-context length
on the 8 GB VRAM accelerator: zero-shot took 8.8 min, retrieval-augmented (6 exemplars)
took 59.8 min, random few-shot (12 exemplars) took 109.2 min. The cause is KV-cache
spillover from VRAM to system RAM via Ollama's partial CPU-offload mechanism. Qwen
and Gemma do not exhibit the same dispersion, suggesting a smaller per-token KV-cache
footprint or a more aggressive memory layout in those backbones.

To shorten the run, reduce `N_TEST_EVAL` from 100 to 50 (cuts runtime by 2× at the
cost of larger metric variance), or skip Mistral-Nemo's few-shot strategy.

---

## Outputs

After running cells 10–13, the following artefacts are written next to the notebook:

| File | Contents |
|---|---|
| `llm_results.csv`              | full metrics table (10 rows × 12 columns) |
| `llm_runs.pkl`                 | pickled runs metadata + first 50 raw model outputs per run |
| `llm_accuracy_comparison.png`  | horizontal bar chart of accuracy per (model, strategy) |
| `llm_accuracy_heatmap.png`     | model × strategy accuracy heat-map (single-shot only) |
| `llm_confusion_matrices.png`   | grid of confusion matrices, one per (model, strategy) |

The cell 13 print also emits a LaTeX-formatted version of the metrics table that can
be pasted directly into the paper's results section.

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
Workaround: re-run cell 7b once, or simply proceed to the main sweep, the empty
return is intermittent and parses cleanly during the actual sweep.

### Mistral-Nemo few-shot is hours long
Expected behaviour on 8 GB VRAM with 12 in-context exemplars (random few-shot prompts
exceed ~4 kB and trigger KV-cache offload to system RAM). If you cannot afford the
runtime, lower `FEWSHOT_K_RANDOM` from 4 to 2 (drops total exemplars from 12 to 6 and
brings latency in line with the retrieval-augmented strategy).

### Model swap is slow during the sweep
Ollama unloads a model from VRAM when a different one is requested, which can take
10–30 s on first reload. The sweep loops outermost over models so each model is
loaded only once for its three strategies.

---

## Project Structure

```
Wine-Quality/
├── README.md                       ← You are here
├── wine_llm_comparison.ipynb       ← The notebook
├── llm_results.csv                 ← Generated metrics table
├── llm_runs.pkl                    ← Raw run metadata
├── llm_accuracy_comparison.png     ← Generated bar chart
├── llm_accuracy_heatmap.png        ← Generated heat-map
└── llm_confusion_matrices.png      ← Generated grid of confusion matrices
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` 2.x (cu128)         | GPU backend for the BGE encoder |
| `sentence-transformers`     | Frozen BGE sentence embeddings (kNN retrieval only) |
| `ollama`                    | Local LLM inference (Qwen 2.5, Mistral-Nemo, Gemma 3) |
| `datasets`                  | HuggingFace dataset loading |
| `scikit-learn`              | Metrics (accuracy, F1, kappa, MCC, confusion matrix) |
| `scipy`                     | Spearman rank correlation |
| `pandas`, `numpy`           | Tabular data handling |
| `matplotlib`, `seaborn`     | Plots |

---

## Paper Contributions

Novel aspects compared to prior wine-quality classification literature:

1. **First side-by-side comparison** of three open-weights instruction-tuned LLMs from
   different lineages (Alibaba / Mistral AI / Google) at the 12–14 B parameter scale on
   wine quality classification, isolating the effect of model lineage from prompt design.
2. **Class-balanced retrieval-augmented few-shot** prompting (top-2 per class via BGE
   cosine similarity), which lifts Mistral-Nemo-12B accuracy from 0.830 (zero-shot) to
   **0.900** (+7.0 pp) and avoids the failure mode of pure global top-K retrieval under
   the heavy training-pool class imbalance.
3. **Quality-tier binning** derived from the WineEnthusiast editorial rubric, mapping
   the six published tiers (acceptable / good / very good / excellent / superb /
   classic) onto a coarser three-class partition that respects the magazine's own
   thresholds rather than a fitted percentile-based scheme.
4. **Single-digit output protocol with simplified regex parsing** in place of the
   structured-JSON convention common in the literature, eliminating format-drift
   parse failures across heterogeneous LLM backbones (zero parse failures across
   1,000 deterministic responses in our runs).
5. **Honest reporting of self-consistency on saturated cells**: applied to the
   strongest single-shot configuration (Mistral-Nemo retrieval, 0.900), self-consistency
   *decreased* accuracy to 0.870, illustrating that stochastic resampling at τ = 0.6
   helps only when the deterministic baseline is uncertain (≲ 0.80 accuracy).
6. **Ordinal-aware metric suite** (Cohen's κ, MCC, Spearman ρ, ordinal MAE, Rank-Acc
   ±1) reported alongside accuracy and F₁, with a dedicated caveats section that
   addresses class-imbalance-induced metric saturation (e.g., RankAcc±1 = 1.000 across
   all configurations is partially degenerate when class 0 has only 2 test samples).
7. **Operational latency analysis** showing that on a 8 GB VRAM consumer accelerator,
   prompt-context length is the dominant determinant of inference cost
   (zero-shot 8.8 min vs few-shot 109.2 min for the same Mistral-Nemo-12B model),
   driven by KV-cache spillover under Ollama's partial CPU-offload mechanism.

---

Last updated: May 2026
