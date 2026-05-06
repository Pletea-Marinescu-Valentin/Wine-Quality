# Wine Quality Classification — Multi-Method Benchmark

IEEE International Conference paper benchmarking classical ML, gradient-boosted ensembles, deep tabular models (TabPFN), text transformers (DistilBERT, RoBERTa, DeBERTa-v3), local LLM zero-shot inference (Llama 3.1 8B), and multimodal late fusion on a large wine review corpus.

**Active dataset:** [`james-burton/wine_reviews_ordinal`](https://huggingface.co/datasets/james-burton/wine_reviews_ordinal) — 105,154 expert wine reviews from WineMag.com (Burton 2023, originally compiled by Thoutt 2017), reformulated for ordinal classification.

**Task:** 3-class quality classification (Low / Medium / High) via tertile binning of WineEnthusiast points (80–100).

---

## Notebooks

| File | Purpose |
|---|---|---|
| `wine_quality_classification.ipynb` | High-accuracy pipeline. Optuna 200-trial LightGBM, 100-trial CatBoost, DeBERTa-v3-base, sentence embeddings, target encoding, stacking, late fusion. |

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
pip install datasets transformers accelerate xgboost lightgbm catboost ^
            scikit-learn pandas numpy matplotlib seaborn optuna tqdm ^
            sentence-transformers sentencepiece protobuf ^
            ollama ipykernel jupyter
pip install tabpfn   # optional - the TabPFN cell falls back gracefully if missing
```

### 4. Register Jupyter Kernel for VSCode

```powershell
conda activate wine-quality
python -m ipykernel install --user --name wine-quality --display-name "Python (wine-quality)"
```

Verify kernel is registered:
```powershell
jupyter kernelspec list
```

### 5. Install & Configure Ollama (Local LLM)

1. **Download & install:** https://ollama.com/download (Windows installer)

2. **Pull Llama 3.1 8B** (one-time, ~4.7 GB):
   ```powershell
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" pull llama3.1:8b
   ```

3. **Start Ollama server** (runs in background):
   ```powershell
   & "C:\Users\$env:USERNAME\AppData\Local\Programs\Ollama\ollama.exe" serve
   ```

   Leave this PowerShell window open while the notebook runs.

---

## Running the Active Notebook

### 1. Open in VSCode

Navigate to `c:\Users\plete\Desktop\Wine-Quality\` and open `wine_quality_classification.ipynb`.

### 2. Select Kernel

Click the kernel dropdown (top-right) → select **`Python (wine-quality)`**.

### 3. Before Running

Make sure Ollama is running (see Setup §5). The notebook will skip the LLM cell gracefully if Ollama is not reachable.

### 4. Execute Cells in Order

The notebook is organized into 23 sections. Run top-to-bottom — each cell depends on variables defined in earlier cells.

---

## Estimated Runtime (RTX 5050)

| Stage | Duration |
|---|---|
| Data loading + feature engineering | ~5 min |
| SBERT encoding (105K reviews) | 15–20 min |
| Classical models (RF, ET, XGB, LGB, CatBoost, LR) | ~10 min |
| LightGBM Optuna (200 trials × 5-fold CV) | 60–90 min |
| CatBoost Optuna (100 trials × 5-fold CV) | 30–45 min |
| AttentionMLP (200 epochs, early stopping) | ~5 min |
| TabPFN (8K subsample) | 2–3 min |
| DistilBERT (10 epochs, max_len 256) | 30–45 min |
| RoBERTa-base (10 epochs) | 60–90 min |
| DeBERTa-v3-base (10 epochs) | 90–120 min |
| LLM CoT + few-shot (1000 samples) | 15–25 min |
| Stacking (5-fold) + late fusion + viz | ~15 min |
| **Total** | **~6–8 hours** |

To speed things up, reduce `OPTUNA_TRIALS_LGB`, `OPTUNA_TRIALS_CAT`, `NUM_EPOCHS_BASE`, or skip transformers individually.

---

## Troubleshooting

### `[transformers] LOAD REPORT` warnings
Informational only — emitted when fine-tuning a pretrained encoder for a downstream classification task. The classification head (`pre_classifier`, `classifier`) is newly initialized; the language modeling head is dropped. The notebook silences these with `transformers.logging.set_verbosity_error()`.

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

### LLM cell hangs on first call
First inference loads the 4.7 GB model into RAM (~10–20 s). Subsequent calls are fast (~2–5 s each). Verify Ollama is reachable:
```powershell
curl http://localhost:11434/api/tags
```
Should return JSON listing `llama3.1:8b`.

---

## Project Structure

```
Wine-Quality/
├── README.md                           ← You are here
└── wine_quality_classification.ipynb   ← high-accuracy notebook
```

---

## Key Dependencies

| Package | Purpose |
|---|---|
| `torch` 2.x (cu128) | Deep learning backbone |
| `transformers` | Text encoders (DistilBERT, RoBERTa, DeBERTa-v3) |
| `sentence-transformers` | Frozen MPNet sentence embeddings |
| `optuna` | Bayesian hyperparameter search (TPE sampler) |
| `lightgbm`, `xgboost`, `catboost` | Gradient boosting |
| `tabpfn` | Pretrained tabular foundation model (optional) |
| `datasets` | HuggingFace dataset loading |
| `ollama` | Local LLM inference (Llama 3.1 8B) |
| `scikit-learn` | Classical ML, stacking, metrics |

---

## Paper Contributions

Novel aspects compared to prior wine quality classification literature:

1. **Tabular foundation model on wine quality** — first application of TabPFN.
2. **Sentence embeddings as tabular features** — frozen MPNet vectors fused into the gradient-boosted feature space (3,781-dim total).
3. **Out-of-fold smoothed target encoding** for high-cardinality categoricals (country, region, variety) under leak-safe cross-validation.
4. **Layer-wise learning-rate decay + cosine warmup + fp16** chained with Bayesian optimization for fair transformer / GBDT comparison.
5. **Few-shot LLM zero-shot baseline** with chain-of-thought prompting (Llama 3.1 8B local).
6. **Calibration & ordinal metrics suite** (ECE, Brier, MCC, Spearman ρ, ordinal MAE, Rank Acc ±1) absent from prior wine quality work.
7. **Multimodal late fusion** of stacking ensemble + best text transformer under three combination strategies (avg, weighted, max-confidence).
