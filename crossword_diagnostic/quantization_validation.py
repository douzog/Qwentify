#!/usr/bin/env python3
"""
Quantization Validation: Does rho^2 predict quantization sensitivity?
=====================================================================

Phase 1: Reconstruction error correlation
------------------------------------------
For each weight matrix in shard 1, simulate quantization at multiple
bit levels (2, 3, 4, 6, 8 bit) and measure reconstruction error.
Then test whether rho^2 (from crossword decomposition) correlates
with quantization robustness.

The hypothesis: layers with higher rho^2 have more structure that
survives coarse quantization, so they should show lower relative
reconstruction error at low bit widths.

Phase 2: Perplexity validation (per-layer)
-------------------------------------------
For a full model, swap each layer's weights with quantized versions
one at a time, run a forward pass on WikiText-2, and measure
perplexity degradation. Correlate with rho^2.

Usage:
    # Phase 1: reconstruction error (fast, shard 1 only)
    python quantization_validation.py --phase 1

    # Phase 1 on a specific model
    python quantization_validation.py --phase 1 --model qwen3-0.6b

    # Phase 2: perplexity (needs full model, slower)
    python quantization_validation.py --phase 2 --model qwen3-0.6b
"""

import sys
import os
import argparse
import csv
import time
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from crossword_decomposition import crossword_decompose, row_dominance
from run_analysis import (
    MODEL_REGISTRY, OUTPUT_DIR, find_shard_path, load_shard,
    classify_tensor, extract_layer_index, subsample_if_needed,
)


# ── Quantization Simulation ─────────────────────────────────────────────────

def uniform_quantize(W: np.ndarray, bits: int) -> np.ndarray:
    """
    Simulate uniform round-to-nearest quantization.

    Maps W to `2^bits` levels between min(W) and max(W), then
    dequantizes back to float. This is the simplest quantization
    scheme and a reasonable baseline.
    """
    if bits >= 16:
        return W.copy()

    w_min = W.min()
    w_max = W.max()

    if w_max == w_min:
        return W.copy()

    n_levels = 2 ** bits
    scale = (w_max - w_min) / (n_levels - 1)

    # Quantize: map to integers, then dequantize back
    W_int = np.round((W - w_min) / scale).astype(np.int32)
    W_int = np.clip(W_int, 0, n_levels - 1)
    W_deq = W_int.astype(np.float32) * scale + w_min

    return W_deq


def block_quantize(W: np.ndarray, bits: int,
                   block_size: int = 256) -> np.ndarray:
    """
    Simulate block quantization (closer to K-quant behavior).

    Quantizes each contiguous block of `block_size` weights
    independently with its own scale and zero point. This is
    how llama.cpp's K-quants work: 256-weight blocks with
    per-block scaling.
    """
    if bits >= 16:
        return W.copy()

    flat = W.flatten()
    n = len(flat)
    out = np.empty_like(flat)
    n_levels = 2 ** bits

    for start in range(0, n, block_size):
        end = min(start + block_size, n)
        block = flat[start:end]

        b_min = block.min()
        b_max = block.max()

        if b_max == b_min:
            out[start:end] = block
            continue

        scale = (b_max - b_min) / (n_levels - 1)
        quantized = np.round((block - b_min) / scale).astype(np.int32)
        quantized = np.clip(quantized, 0, n_levels - 1)
        out[start:end] = quantized.astype(np.float32) * scale + b_min

    return out.reshape(W.shape)


def measure_quantization_error(W: np.ndarray, bits: int,
                               method: str = "block") -> dict:
    """
    Quantize W at the given bit level and measure reconstruction error.

    Returns dict with error metrics.
    """
    if method == "block":
        W_q = block_quantize(W, bits)
    else:
        W_q = uniform_quantize(W, bits)

    error = W - W_q

    frob_orig = np.linalg.norm(W, 'fro')
    frob_error = np.linalg.norm(error, 'fro')

    relative_error = frob_error / frob_orig if frob_orig > 0 else 0
    mse = np.mean(error ** 2)
    snr = 10 * np.log10(np.var(W) / np.var(error)) if np.var(error) > 0 else float('inf')

    return {
        "bits": bits,
        "method": method,
        "relative_error": relative_error,
        "mse": mse,
        "snr_db": snr,
        "max_abs_error": float(np.max(np.abs(error))),
        "mean_abs_error": float(np.mean(np.abs(error))),
    }


# ── Phase 1: Reconstruction Error Correlation ───────────────────────────────

def run_phase1(model_key: str, shard_path: str,
               output_dir: str) -> List[dict]:
    """
    For each weight matrix: compute rho^2, then quantize at multiple
    bit levels and measure reconstruction error. Output a table that
    lets us test the correlation.
    """
    print(f"\n{'=' * 70}")
    print(f"  PHASE 1: RECONSTRUCTION ERROR vs rho^2")
    print(f"  Model: {model_key}")
    print(f"{'=' * 70}\n")

    tensors = load_shard(shard_path)
    print(f"  Loaded {len(tensors)} weight matrices\n")

    bit_levels = [2, 3, 4, 6, 8]
    rng = np.random.default_rng(42)
    results = []

    for name, W_orig in sorted(tensors.items()):
        layer_type = classify_tensor(name)
        layer_idx = extract_layer_index(name)

        W, was_sampled = subsample_if_needed(W_orig, name, rng)
        m, n = W.shape

        # Crossword decomposition
        decomp = crossword_decompose(W, layer_name=name, layer_type=layer_type)
        rho2 = decomp.var_explained
        dom = row_dominance(decomp)

        # Quantize at each bit level
        for bits in bit_levels:
            err = measure_quantization_error(W, bits, method="block")

            row = {
                "model": model_key,
                "tensor_name": name,
                "layer_type": layer_type,
                "layer_index": layer_idx,
                "shape": f"{m}x{n}",
                "rho2_pct": rho2 * 100,
                "row_dominance": dom * 100,
                "bits": bits,
                "relative_error": err["relative_error"],
                "mse": err["mse"],
                "snr_db": err["snr_db"],
            }
            results.append(row)

        # Progress
        errs_at_4bit = [r for r in results
                        if r["tensor_name"] == name and r["bits"] == 4]
        if errs_at_4bit:
            e4 = errs_at_4bit[0]["relative_error"]
            print(f"  {name:<50} rho2={rho2*100:>6.3f}%  "
                  f"err@4bit={e4:.4f}")

    return results


def compute_correlations(results: List[dict]) -> dict:
    """
    Compute Spearman rank correlation between rho^2 and quantization
    error at each bit level.
    """
    from scipy.stats import spearmanr

    bit_levels = sorted(set(r["bits"] for r in results))
    correlations = {}

    for bits in bit_levels:
        rows = [r for r in results if r["bits"] == bits]

        rho2_vals = [r["rho2_pct"] for r in rows]
        error_vals = [r["relative_error"] for r in rows]

        if len(set(rho2_vals)) < 3:
            continue

        # Negative correlation expected: higher rho2 -> lower error
        corr, pval = spearmanr(rho2_vals, error_vals)
        correlations[bits] = {"spearman_r": corr, "p_value": pval, "n": len(rows)}

    return correlations


def print_phase1_summary(results: List[dict], correlations: dict):
    """Print Phase 1 results."""
    print(f"\n{'=' * 70}")
    print(f"  CORRELATION: rho^2 vs QUANTIZATION ERROR")
    print(f"{'=' * 70}\n")

    print(f"  {'Bits':>4}  {'Spearman r':>11}  {'p-value':>10}  {'n':>4}  {'Interpretation'}")
    print(f"  {'-'*55}")

    for bits in sorted(correlations.keys()):
        c = correlations[bits]
        r = c["spearman_r"]
        p = c["p_value"]
        n = c["n"]

        if p < 0.01:
            sig = "**"
        elif p < 0.05:
            sig = "*"
        else:
            sig = "ns"

        if r < -0.3 and p < 0.05:
            interp = "higher rho2 -> lower error"
        elif r > 0.3 and p < 0.05:
            interp = "higher rho2 -> HIGHER error (!)"
        else:
            interp = "no significant relationship"

        print(f"  {bits:>4}  {r:>+11.4f}  {p:>10.2e}  {n:>4}  {sig} {interp}")

    # Per-layer-type breakdown at 4-bit
    print(f"\n{'=' * 70}")
    print(f"  PER-LAYER-TYPE: 4-BIT QUANTIZATION ERROR")
    print(f"{'=' * 70}\n")

    rows_4bit = [r for r in results if r["bits"] == 4]
    by_type = defaultdict(list)
    for r in rows_4bit:
        by_type[r["layer_type"]].append(r)

    print(f"  {'Type':<16} {'Count':>5} {'Avg rho2':>9} "
          f"{'Avg RelErr':>11} {'Avg SNR':>9}")
    print(f"  {'-'*55}")

    for ltype in sorted(by_type.keys()):
        rows = by_type[ltype]
        n = len(rows)
        avg_rho2 = np.mean([r["rho2_pct"] for r in rows])
        avg_err = np.mean([r["relative_error"] for r in rows])
        avg_snr = np.mean([r["snr_db"] for r in rows])
        print(f"  {ltype:<16} {n:>5} {avg_rho2:>8.3f}% "
              f"{avg_err:>11.6f} {avg_snr:>8.1f}dB")


# ── Phase 2: Perplexity Validation ──────────────────────────────────────────

def run_phase2(model_name: str, output_dir: str,
               max_layers: Optional[int] = None,
               bits_to_test: Optional[List[int]] = None) -> List[dict]:
    """
    Per-layer quantization perplexity experiment.

    For each layer:
    1. Load the full model
    2. Replace that layer's weights with quantized version
    3. Run forward pass on WikiText-2 test set
    4. Measure perplexity
    5. Restore original weights

    This gives per-layer quantization sensitivity measured by
    actual perplexity degradation.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from datasets import load_dataset
    except ImportError as e:
        print(f"  Phase 2 requires: pip install transformers datasets torch")
        print(f"  Missing: {e}")
        sys.exit(1)

    if bits_to_test is None:
        bits_to_test = [2, 4]

    # Map model keys to HF model IDs
    hf_model_map = {
        "qwen3-0.6b": "Qwen/Qwen3-0.6B",
        "gemma3-1b": "google/gemma-3-1b-pt",
        "qwen3-4b": "Qwen/Qwen3-4B",
        "gemma3-4b": "google/gemma-3-4b-pt",
        "qwen3-8b": "Qwen/Qwen3-8B",
        "stablelm2-1.6b": "stabilityai/stablelm-2-1_6b",
        "mistral-7b": "mistralai/Mistral-7B-v0.3",
        "falcon3-7b": "tiiuae/Falcon3-7B-Base",
        "phi-4-14b": "microsoft/phi-4",
    }

    hf_id = hf_model_map.get(model_name)
    if not hf_id:
        print(f"  Unknown model: {model_name}")
        print(f"  Available: {', '.join(hf_model_map.keys())}")
        sys.exit(1)

    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: PER-LAYER PERPLEXITY VALIDATION")
    print(f"  Model: {hf_id}")
    print(f"{'=' * 70}\n")

    # Load model
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"  Device: {device}")
    print(f"  Loading model...")

    model = AutoModelForCausalLM.from_pretrained(
        hf_id, torch_dtype=torch.float16, device_map=device,
    )
    tokenizer = AutoTokenizer.from_pretrained(hf_id)
    model.eval()

    # Load evaluation data
    print(f"  Loading WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(dataset["text"])

    # Tokenize with a reasonable context window
    max_length = 2048
    encodings = tokenizer(text, return_tensors="pt", truncation=True,
                          max_length=max_length)
    input_ids = encodings.input_ids.to(device)

    # Baseline perplexity
    print(f"  Computing baseline perplexity...")
    baseline_ppl = compute_perplexity(model, input_ids)
    print(f"  Baseline perplexity: {baseline_ppl:.2f}")

    # Find quantizable layers
    results = []
    layers_tested = 0

    for name, param in model.named_parameters():
        if param.ndim != 2:
            continue

        layer_type = classify_tensor(name)
        layer_idx = extract_layer_index(name)

        if max_layers is not None and layer_idx >= 0 and layer_idx >= max_layers:
            continue

        # Skip tiny matrices
        if param.shape[0] < 64 or param.shape[1] < 64:
            continue

        # Compute rho^2 on the original weights
        W_np = param.detach().cpu().float().numpy()
        decomp = crossword_decompose(W_np, layer_name=name,
                                     layer_type=layer_type)
        rho2 = decomp.var_explained
        dom = row_dominance(decomp)

        for bits in bits_to_test:
            # Quantize
            W_q = block_quantize(W_np, bits)
            W_q_tensor = torch.from_numpy(W_q).to(param.dtype).to(param.device)

            # Swap weights
            original_data = param.data.clone()
            param.data = W_q_tensor

            # Measure perplexity
            ppl = compute_perplexity(model, input_ids)

            # Restore
            param.data = original_data

            ppl_increase = ppl - baseline_ppl
            ppl_ratio = ppl / baseline_ppl

            row = {
                "model": model_name,
                "tensor_name": name,
                "layer_type": layer_type,
                "layer_index": layer_idx,
                "rho2_pct": rho2 * 100,
                "row_dominance": dom * 100,
                "bits": bits,
                "baseline_ppl": baseline_ppl,
                "quantized_ppl": ppl,
                "ppl_increase": ppl_increase,
                "ppl_ratio": ppl_ratio,
            }
            results.append(row)

            print(f"  {name:<45} rho2={rho2*100:>6.3f}%  "
                  f"{bits}bit: ppl={ppl:.2f} (+{ppl_increase:.2f})")

        layers_tested += 1

    print(f"\n  Tested {layers_tested} layers")
    return results


def compute_perplexity(model, input_ids) -> float:
    """Compute perplexity on the given input_ids."""
    import torch

    with torch.no_grad():
        outputs = model(input_ids, labels=input_ids)
        loss = outputs.loss.item()

    return np.exp(loss)


# ── Figures ──────────────────────────────────────────────────────────────────


def generate_validation_figures(results: List[dict], correlations: dict,
                                output_dir: str, phase: int = 1):
    """Generate validation figures."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    type_color_map = {
        "attn_q": "#e41a1c", "attn_k": "#ff7f00",
        "attn_v": "#984ea3", "attn_o": "#a65628",
        "mlp_gate": "#377eb8", "mlp_up": "#4daf4a",
        "mlp_down": "#f781bf", "embedding": "#000000",
        "attention_q": "#e41a1c", "attention_k": "#ff7f00",
        "attention_v": "#984ea3", "attention_o": "#a65628",
    }

    y_key = "relative_error" if phase == 1 else "ppl_ratio"
    y_label = "Relative Reconstruction Error" if phase == 1 else "Perplexity Ratio (quantized / baseline)"

    # ── Scatter: rho^2 vs error at each bit level ──
    bit_levels = sorted(set(r["bits"] for r in results))
    n_bits = len(bit_levels)

    fig, axes = plt.subplots(1, n_bits, figsize=(5 * n_bits, 5),
                             squeeze=False, sharey=False)

    for i, bits in enumerate(bit_levels):
        ax = axes[0][i]
        rows = [r for r in results if r["bits"] == bits]

        for r in rows:
            c = type_color_map.get(r["layer_type"], "#999999")
            ax.scatter(r["rho2_pct"], r[y_key], c=c, alpha=0.6,
                       s=30, edgecolors="black", linewidth=0.3)

        ax.set_xlabel("ρ² (%)")
        if i == 0:
            ax.set_ylabel(y_label)
        ax.set_title(f"{bits}-bit quantization")
        ax.grid(alpha=0.3)

        # Add correlation annotation
        if bits in correlations:
            c = correlations[bits]
            ax.text(0.95, 0.95,
                    f"r={c['spearman_r']:.3f}\np={c['p_value']:.2e}",
                    transform=ax.transAxes, ha="right", va="top",
                    fontsize=8, family="monospace",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              alpha=0.8))

    # Legend
    from matplotlib.lines import Line2D
    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=8, label=t)
               for t, c in type_color_map.items()
               if t in set(r["layer_type"] for r in results)]
    if handles:
        axes[0][-1].legend(handles=handles, fontsize=6, loc="upper left")

    fig.suptitle(f"Phase {phase}: ρ² vs Quantization {'Error' if phase == 1 else 'Perplexity'}",
                 fontsize=12)
    fig.tight_layout()
    fname = f"fig_phase{phase}_rho2_vs_error.png"
    fig.savefig(os.path.join(output_dir, fname), dpi=150)
    plt.close(fig)
    print(f"  Saved {fname}")

    # ── Bar chart: per-type error at 4-bit ──
    rows_4bit = [r for r in results if r["bits"] == 4]
    if rows_4bit:
        by_type = defaultdict(list)
        for r in rows_4bit:
            by_type[r["layer_type"]].append(r)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        types = sorted(by_type.keys())
        x = range(len(types))

        # Left: rho^2 by type
        rho2_means = [np.mean([r["rho2_pct"] for r in by_type[t]]) for t in types]
        ax1.bar(x, rho2_means, color="#407bff", alpha=0.85,
                edgecolor="black", linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(types, rotation=45, ha="right", fontsize=8)
        ax1.set_ylabel("ρ² (%)")
        ax1.set_title("Crossword Structure by Layer Type")
        ax1.grid(axis="y", alpha=0.3)

        # Right: error by type
        err_means = [np.mean([r[y_key] for r in by_type[t]]) for t in types]
        ax2.bar(x, err_means, color="#d62728", alpha=0.85,
                edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(types, rotation=45, ha="right", fontsize=8)
        ax2.set_ylabel(y_label)
        ax2.set_title("4-bit Quantization Error by Layer Type")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("Do layers with more structure quantize better?", fontsize=12)
        fig.tight_layout()
        fname = f"fig_phase{phase}_by_type.png"
        fig.savefig(os.path.join(output_dir, fname), dpi=150)
        plt.close(fig)
        print(f"  Saved {fname}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Validate rho^2 as a quantization sensitivity predictor.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--phase", type=int, default=1, choices=[1, 2],
        help="Phase 1: reconstruction error. Phase 2: perplexity.",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3-0.6b",
        help="Model key (default: qwen3-0.6b, smallest model).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory.",
    )
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Only test the first N layers (Phase 2, for speed).",
    )
    args = parser.parse_args()

    output_dir = args.output or os.path.join(OUTPUT_DIR, "validation")
    os.makedirs(output_dir, exist_ok=True)

    if args.phase == 1:
        # Phase 1: reconstruction error
        shard_path = find_shard_path(args.model)
        if not shard_path:
            print(f"Model {args.model} not found in HF cache.")
            print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)

        results = run_phase1(args.model, shard_path, output_dir)

        # Compute correlations
        try:
            correlations = compute_correlations(results)
        except ImportError:
            print("  scipy not available for correlation test.")
            print("  Install: pip install scipy")
            correlations = {}

        if correlations:
            print_phase1_summary(results, correlations)

        # Write CSV
        csv_path = os.path.join(output_dir,
                                f"phase1_{args.model.replace('-', '_')}.csv")
        if results:
            keys = results[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  CSV: {csv_path}")

        # Figures
        print("\nGenerating figures...")
        generate_validation_figures(results, correlations, output_dir, phase=1)

    elif args.phase == 2:
        # Phase 2: perplexity
        results = run_phase2(args.model, output_dir,
                             max_layers=args.max_layers)

        # Compute correlations
        try:
            correlations = compute_correlations(results)
        except ImportError:
            correlations = {}

        if correlations:
            print(f"\n{'=' * 70}")
            print(f"  PERPLEXITY CORRELATION")
            print(f"{'=' * 70}\n")
            for bits, c in sorted(correlations.items()):
                print(f"  {bits}-bit: Spearman r={c['spearman_r']:.4f}, "
                      f"p={c['p_value']:.2e}")

        # Write CSV
        csv_path = os.path.join(output_dir,
                                f"phase2_{args.model.replace('-', '_')}.csv")
        if results:
            keys = results[0].keys()
            with open(csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=keys)
                writer.writeheader()
                writer.writerows(results)
            print(f"\n  CSV: {csv_path}")

        # Figures
        print("\nGenerating figures...")
        generate_validation_figures(results, correlations, output_dir, phase=2)

    print(f"\n  Output: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
