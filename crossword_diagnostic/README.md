# Crossword Structural Diagnostics for Quantized LLMs

[Interactive companion](https://iig-letters-act.web.app/coding-llm)

Applies crossword decomposition (two-way ANOVA without interaction) to the weight matrices of quantized LLMs.

1. Do trained weights develop additive row-column structure?
2. Does quantization error concentrate in row/column effects or the residual?
3. Can crossword structure predict which layers are quantization-sensitive?

## How It Works

```
GGUF / Safetensors Model
          |
          v
┌─────────────────────┐
│  Extract Weights    │  gguf-py (BF16/F16) or safetensors (HF format)
│  Per Layer          │
└─────────┬───────────┘
          |
          v
┌─────────────────────┐
│  Crossword          │  W = mu + r 1^T + 1 c^T + R
│  Decomposition      │
└─────────┬───────────┘
          |
          v
┌─────────────────────┐
│  Diagnostics        │  rho^2 (variance explained)
│  Per Layer          │  Compression gain (bits/param)
│  Per Quant Level    │  Quantization error structure
└─────────┬───────────┘
          |
          v
   CSV + Figures + LaTeX Tables
```

## Setup

```bash
pip install numpy matplotlib

# For GGUF files (BF16/F16):
pip install gguf

# For HuggingFace safetensors (alternative):
pip install safetensors
```

## Getting the Model Weights

### Option A: Download pre-built BF16 GGUF (recommended)

```bash
pip install -U "huggingface_hub[cli]"

# BF16 full precision, 16.4 GB
huggingface-cli download bartowski/Qwen_Qwen3-8B-GGUF \
  --include "Qwen3-8B-bf16.gguf" --local-dir ./models
```

### Option B: Use HuggingFace safetensors directly

```bash
# Downloads the original model (~16 GB)
huggingface-cli download Qwen/Qwen3-8B --local-dir ./models/qwen3-8b
```

### A note on quantized GGUF files

Q4_K_M and Q2_K GGUF files store weights in block-quantized format
(not plain floats). The gguf Python package cannot dequantize these
directly. For the quantization error analysis, the tool compares the
BF16 weights against simulated quantization at equivalent bit levels.

## Usage

```bash
# Analyze the BF16 model (trained weights)
python crossword_gguf.py --model ./models/Qwen3-8B-bf16.gguf

# Or from safetensors directory
python crossword_gguf.py --model ./models/qwen3-8b

# Quick test on first 4 layers
python crossword_gguf.py --model ./models/Qwen3-8B-bf16.gguf --max-layers 4

# Full analysis with output
python crossword_gguf.py \
  --model ./models/Qwen3-8B-bf16.gguf \
  --output results/
```

## Key Metrics

- **rho^2 (variance explained)**: Fraction of matrix variance captured by row + column effects. High rho^2 means crossword encoding is effective.
- **Compression gain**: 1/2 log2(Var(W)/Var(R)) bits per parameter.
- **SVD-r1 comparison**: Rank-1 SVD at equal parameter budget. SVD is always at least as good as crossword encoding (Proposition 2).

## Background

The crossword encoding framework shows that:

- Word co-occurrence matrices have 14 to 29% crossword structure
- Initialized neural network weights have less than 0.3%
- At equal parameter count, rank-1 SVD captures roughly 2x more variance than crossword encoding

This tool extends that analysis to trained weights, testing whether training induces additive structure and whether crossword decomposition can serve as a diagnostic for quantization quality.

## Results on Qwen-3-8B

Qwen-3-8B benchmarks at F16/Q4_K_M/Q2_K show Q4_K_M is the sweet spot (coherent output, ~5 GB) while Q2_K degrades badly (repetition, lost reasoning, ~3 GB). This tool characterizes the structural properties of each layer's weight matrix to explain why.

Trained Qwen-3-8B weights average 0.28% crossword structure, essentially identical to the 0.27% baseline for initialized weights. Training does not induce additive row-column structure. MLP gate projections are a notable outlier at 1.43%. The SVD-to-crossword ratio serves as a diagnostic for whether structure is additive or multiplicative.

## Repo Structure

```
crossword_diagnostic/
├── README.md
├── crossword_gguf.py          # Main analysis script
├── crossword_decomposition.py # Core math (numpy only)
├── run_analysis.py            # Direct analysis on safetensors
└── results/                   # Output directory (generated)
```
