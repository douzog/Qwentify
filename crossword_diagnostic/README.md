# Crossword Structural Diagnostics for LLM Quantization

[Interactive companion](https://iig-letters-act.web.app/coding-llm)

Applies crossword decomposition (two-way ANOVA without interaction) to weight
matrices from open-weight language models. A cheap, data-free diagnostic for
mixed-precision quantization.

## What This Does

Given a weight matrix W, the crossword decomposition splits it into:

```
W = μ·11ᵀ + r·1ᵀ + 1·cᵀ + R
```

where μ is the grand mean, r is row effects, c is column effects, and R is
the residual. The diagnostic metric ρ² = 1 − Var(R)/Var(W) measures how much
of the matrix is row-column separable.

Layers with higher ρ² tolerate aggressive quantization. Layers with near-zero
ρ² need higher precision. No calibration data, no forward passes, runs in
seconds per layer.

## Models Tested

Seven open-weight models spanning four organizations and three orders of
magnitude in parameter count:

| Model          | Org          | Params | Architecture   |
| -------------- | ------------ | ------ | -------------- |
| Qwen3-0.6B     | Alibaba      | 0.6B   | Dense, GQA     |
| StableLM2-1.6B | Stability AI | 1.6B   | Dense, MHA     |
| Qwen3-4B       | Alibaba      | 4B     | Dense, GQA     |
| Mistral-7B     | Mistral      | 7B     | Dense, GQA+SWA |
| Falcon3-7B     | TII          | 7B     | Dense, GQA     |
| Qwen3-8B       | Alibaba      | 8.2B   | Dense, GQA     |
| Phi-4-14B      | Microsoft    | 14B    | Dense, GQA     |

## Key Findings

**1. Trained weights lack additive structure.** Across all seven models,
overall ρ² averages below 0.4%. Training does not induce row-column
separability regardless of architecture or scale.

**2. MLP gate projections are the outlier.** Gate ρ² ranges from 0.2% to
1.5% across models, with 87–98% of additive structure concentrated in column
effects. This is a property of the SwiGLU/GeGLU gating mechanism, not
architecture-specific.

**3. Basis-invariant.** Hadamard rotation changes ρ² by less than 1%. The
additive structure is a genuine property, not a coordinate artifact.
Rotation-based quantization methods (QuIP, QuaRot, SpinQuant) do not interact
with crossword structure.

**4. Activation weighting amplifies gate structure.** Scaling columns by L2
norm (proxy for activation magnitude) amplifies gate ρ² by up to +133%. The
salient channels carry the additive column effect.

**5. Embedding inverse scaling.** Embedding ρ² scales inversely with model
size: 10.6% at 0.6B, 8.0% at 4B, 1.7% at 8B, 1.3% at 14B.

**6. Clean dominance taxonomy.** Embedding and gate are column-dominated;
attention K/V are column-leaning; attention Q/O and MLP up/down are balanced
at the noise floor.

### Cross-Model Results

| Model          | Params | Overall ρ² | Gate ρ² | Gate Col% | Embed ρ² |
| -------------- | ------ | ---------- | ------- | --------- | -------- |
| Qwen3-0.6B     | 0.6B   | 0.364%     | 0.872%  | 88.5%     | 10.62%   |
| StableLM2-1.6B | 1.6B   | 0.182%     | 0.594%  | 88.0%     | 0.53%    |
| Qwen3-4B       | 4B     | 0.355%     | 1.460%  | 97.9%     | 8.04%    |
| Mistral-7B     | 7B     | 0.182%     | 0.614%  | 96.6%     | 2.45%    |
| Falcon3-7B     | 7B     | 0.202%     | 0.751%  | 94.8%     | 0.65%    |
| Qwen3-8B       | 8.2B   | 0.285%     | 1.431%  | 96.4%     | 1.67%    |
| Phi-4-14B      | 14B    | 0.133%     | 0.192%  | 87.3%     | 1.27%    |

## Setup

```bash
pip install numpy matplotlib scipy

# For HuggingFace safetensors (recommended):
pip install safetensors torch

# For GGUF files (BF16/F16):
pip install gguf
```

## Getting Model Weights

```bash
pip install -U "huggingface_hub[cli]"

# Download any of the supported models:
huggingface-cli download Qwen/Qwen3-8B
huggingface-cli download Qwen/Qwen3-4B
huggingface-cli download Qwen/Qwen3-0.6B
huggingface-cli download mistralai/Mistral-7B-v0.1
huggingface-cli download tiiuae/Falcon3-7B-Base
huggingface-cli download microsoft/phi-4
huggingface-cli download stabilityai/stablelm-2-1_6b
```

## Usage

### Multi-model analysis (run_analysis.py)

```bash
# List available models (checks HF cache)
python run_analysis.py --list-models

# Analyze all available models
python run_analysis.py

# Analyze a specific model
python run_analysis.py --model qwen3-8b

# Fast mode (skip Hadamard/activation experiments)
python run_analysis.py --no-experiments

# Custom safetensors path
python run_analysis.py --path /path/to/model-00001-of-00005.safetensors
```

### GGUF analysis (crossword_gguf.py)

```bash
# Analyze a BF16 GGUF model
python crossword_gguf.py --model ./models/Qwen3-8B-bf16.gguf

# Compare F16 vs quantized
python crossword_gguf.py \
  --model ./models/Qwen3-8B-bf16.gguf \
  --compare ./models/Qwen3-8B-Q4_K_M.gguf

# Quick test on first 4 layers
python crossword_gguf.py --model ./models/Qwen3-8B-bf16.gguf --max-layers 4
```

## Experiments

The analysis runs four experiments from the paper:

1. **Baseline decomposition** — ρ², row/column variance, SVD rank-1 comparison
2. **Hadamard rotation** — Tests whether additive structure is basis-invariant
3. **Activation weighting** — Scales columns by L2 norm to test salient channel concentration
4. **Row-vs-column dominance** — Classifies each layer type as column-dominated, row-dominated, or balanced

## Key Metrics

- **ρ² (rho-squared)**: Fraction of matrix variance captured by row + column effects. High ρ² → crossword encoding is effective → layer tolerates aggressive quantization.
- **Row dominance**: Var(row) / (Var(row) + Var(col)). Near 0 = column-dominated, near 1 = row-dominated.
- **SVD/CW ratio**: SVD rank-1 vs crossword at equal parameter budget. Near 1 = genuinely additive. Much greater than 1 = multiplicative structure.
- **Compression gain**: ½ log₂(Var(W)/Var(R)) bits per parameter.

## Repo Structure

```
crossword_diagnostic/
├── README.md
├── crossword_decomposition.py  # Core math (numpy only)
│                                 Crossword decompose, SVD comparison,
│                                 Hadamard rotation, activation weighting,
│                                 row dominance
├── crossword_gguf.py           # GGUF/safetensors analysis pipeline
│                                 Single model + quantization comparison
├── run_analysis.py             # Multi-model analysis script
│                                 7-model registry, all experiments,
│                                 cross-model comparison tables + figures
└── results/                    # Output directory (generated)
    ├── crossword_*.csv         # Per-model results
    ├── crossword_all_models.csv # Combined cross-model results
    ├── fig1_cw_by_layer_type.png
    ├── fig2_cw_vs_svd.png
    ├── fig3_depth_profile.png
    ├── fig4_comparison.png
    └── fig5_dominance.png
```
