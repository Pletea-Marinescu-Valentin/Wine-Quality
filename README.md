# Improved Machine Learning Pipeline for Enhanced Wine Quality Classification

IEEE conference paper benchmarking three open-weights instruction-tuned local LLMs on a
3-class wine quality classification task, across four prompting strategies of increasing
sophistication.

**Models compared** (all via Ollama, Q4\_K\_M quantisation, 7–8 B parameters each):
- Llama 3.1 8B Instruct (Meta)
- Qwen 2.5 7B Instruct (Alibaba)
- Mistral 7B Instruct v0.3 (Mistral AI)

**Prompting strategies:**
- Zero-shot Chain-of-Thought
- Few-shot CoT with K=3 random exemplars per class
- Retrieval-Augmented Few-shot CoT (kNN K=5 via SBERT cosine similarity)
- Self-consistency (majority vote of N=3 sampled CoT runs), applied to the winning
  (model, strategy) combination

**Active dataset:** [`james-burton/wine_reviews_ordinal`](https://huggingface.co/datasets/james-burton/wine_reviews_ordinal)
105154 expert wine reviews from WineMag.com (Burton 2023, originally compiled by
Thoutt 2017), reformulated for ordinal classification with predefined train / val / test
splits.

**Task:** 3-class quality classification (Low / Medium / High) via tertile binning of
WineEnthusiast points (80–100).

**Hardware:** RTX 5050 Laptop (8 GB VRAM), CUDA 12.8. All three LLMs fit comfortably at
Q4\_K\_M; Ollama swaps models on demand within a single run.

---

## Setup

### 1. Create Python Environment (Windows 11)

Open **PowerShell** (NOT Git Bash):

```powershell
conda create -n wine-quality python=3.11 -y
conda activate wine-quality
```

### 2. Install PyTorch with CUDA 12.8 (Blackwell GPU support)

```powershell
conda activate wine-quality
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Verify CUDA works:
```powershell
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0))"
```

### 3. Install ML, NLP & Notebook Packages

```powershell
conda activate wine-quality
pip install datasets ollama sentence-transformers ^
            pandas numpy matplotlib seaborn scipy scikit-learn tqdm ^
            ipykernel jupyter
```

### 4. Register Jupyter Kernel for VSCode

```powershell
conda activate wine-quality
python -m ipykernel install --user --name wine-quality --display-name "Python (wine-quality)"
```

Verify the kernel is registered:
```powershell
jupyter kernelspec list
```

### 5. Install Ollama and Pull the Three LLMs

1. **Download & install Ollama:** https://ollama.com/download (Windows installer).

2. **Start the Ollama server in a dedicated PowerShell window** (leave it open for the
   duration of the run):
   ```powershell
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" serve
   ```

3. **Pull the three models** (one-time, ~13 GB on disc):
   ```powershell
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull llama3.1:8b
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull qwen2.5:7b
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull mistral:7b
   ```

   Verify all three are present:
   ```powershell
   curl http://localhost:11434/api/tags
   ```
   Should return JSON listing the three models.

---

## Running the Notebook

### 1. Open in VSCode

Navigate to `c:\Users\plete\Desktop\Wine-Quality\` and open
`wine_llm_comparison.ipynb`.

### 2. Select Kernel

Click the kernel dropdown (top-right) → select **`Python (wine-quality)`**.

### 3. Before Running

Confirm the Ollama server is running (Setup §5.2). The Sanity Check cell (§7b) will
fail-fast if any model is missing or the server is unreachable.

### 4. Execute Cells in Order

Run top-to-bottom the main sweep cell (§8) and the self-consistency cell (§9) are the
long ones; everything else completes within minutes.

---

## Troubleshooting

### `ollama: command not found` (Git Bash / MINGW64)
Git Bash doesn't inherit Windows `PATH`. Use **PowerShell** instead:
```powershell
& "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull llama3.1:8b
```

### Wrong kernel selected (`base (Python 3.13.x)`)
Reinstall the kernel and reload VSCode:
```powershell
conda activate wine-quality
python -m ipykernel install --user --name wine-quality --display-name "Python (wine-quality)" --force
```
Then `Ctrl+Shift+P` → "Developer: Reload Window".

### Model swap is slow during the sweep
Ollama unloads a model from VRAM when a different one is requested, which can take
10–30 s on first reload. This is expected on 8 GB VRAM. The sweep loops outermost over
models so each model is loaded only once for its three strategies.

### LLM cell hangs on the first call of each model
First inference loads the 4–5 GB model file into RAM/VRAM (~10–20 s). Subsequent calls
of the same model are fast (~2–5 s each). Verify Ollama is reachable:
```powershell
curl http://localhost:11434/api/tags
```
Should return JSON listing all three models.

### High parse-failure count in a particular run
The notebook's three-stage regex parser tolerates JSON, "label: N" prose, and lone
digits. If a particular model still produces unparseable output frequently, inspect the
first 50 raw outputs in the saved `llm_runs.pkl` to refine `format_exemplar()` or the
system prompt.

---

## Project Structure

```
Wine-Quality/
├── README.md                       ← You are here
├── paper.tex                       ← IEEE conference paper draft
├── wine_llm_comparison.ipynb       ← The notebook
├── llm_results.csv                 ← Generated metrics table
├── llm_runs.pkl                    ← Raw run metadata (first 50 model outputs each)
├── llm_accuracy_comparison.png     ← Generated bar chart
├── llm_accuracy_heatmap.png        ← Generated heat-map
└── llm_confusion_matrices.png      ← Generated grid of confusion matrices
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` 2.x (cu128) | GPU backend for SBERT |
| `sentence-transformers` | Frozen MPNet sentence embeddings (used only for kNN retrieval) |
| `ollama` | Local LLM inference (Llama 3.1, Qwen 2.5, Mistral) |
| `datasets` | HuggingFace dataset loading |
| `scikit-learn` | Metrics (accuracy, F1, kappa, MCC, confusion matrix) |
| `scipy` | Spearman rank correlation |
| `pandas`, `numpy` | Tabular data handling |
| `matplotlib`, `seaborn` | Plots |

---

## Paper Contributions

Novel aspects compared to prior wine quality classification literature:

1. **First side-by-side comparison** of three open-weights instruction-tuned LLMs at
   the same parameter scale (7–8 B) on wine quality classification, isolating the
   effect of model lineage from prompt design.
2. **Retrieval-augmented few-shot CoT** with semantic-similarity exemplar selection,
   benchmarked against both zero-shot and random few-shot baselines under a unified
   prompt template.
3. **Self-consistency voting** as a final boost on the winning model-strategy
   combination, marginalising single-shot stochasticity.
4. **Robust three-stage output parser** (strict-JSON regex → keyword regex → digit
   fallback) that achieves near-zero parse-failure rate where naive single-character
   parsing collapses LLM accuracy to chance level.
5. **Ordinal-aware metric suite** (Cohen's κ, MCC, Spearman ρ, ordinal MAE, Rank-Acc
   ±1) absent from prior wine-quality LLM literature.
