#!/usr/bin/env python3
"""
Structural Diagnostics for Quantized LLMs
==========================================

Applies crossword decomposition to GGUF model weight matrices,
comparing structural properties across quantization levels.

Usage:
    # Single model analysis
    python crossword_gguf.py --model path/to/model.gguf

    # Compare F16 vs quantized
    python crossword_gguf.py \\
        --model path/to/f16.gguf \\
        --compare path/to/q4km.gguf path/to/q2k.gguf

    # With output directory
    python crossword_gguf.py \\
        --model path/to/f16.gguf \\
        --compare path/to/q4km.gguf path/to/q2k.gguf \\
        --output results/

Dependencies: numpy, matplotlib, gguf
    pip install numpy matplotlib gguf
"""

import argparse
import csv
import os
import sys
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

from crossword_decomposition import (
    CrosswordResult,
    compare_quantization_error,
    crossword_decompose,
    crossword_after_hadamard,
    crossword_activation_weighted,
    row_dominance,
    svd_variance_explained,
)


# ── GGUF Weight Extraction ───────────────────────────────────────────────────

def classify_tensor(name: str) -> str:
    """Classify a GGUF tensor name into a layer type."""
    n = name.lower()
    if "attn" in n or "attention" in n:
        if "q_proj" in n or ".q." in n:
            return "attention_q"
        elif "k_proj" in n or ".k." in n:
            return "attention_k"
        elif "v_proj" in n or ".v." in n:
            return "attention_v"
        elif "o_proj" in n or "out" in n:
            return "attention_o"
        return "attention_other"
    elif "mlp" in n or "ffn" in n or "feed_forward" in n:
        if "gate" in n:
            return "mlp_gate"
        elif "up" in n:
            return "mlp_up"
        elif "down" in n:
            return "mlp_down"
        return "mlp_other"
    elif "embed" in n or "token" in n:
        return "embedding"
    elif "norm" in n or "ln" in n:
        return "norm"
    elif "lm_head" in n or "output" in n:
        return "lm_head"
    return "other"


def extract_layer_index(name: str) -> Optional[int]:
    """Extract the layer number from a tensor name like 'blk.5.attn_q'."""
    import re
    match = re.search(r'(?:blk|layer|layers?)[._](\d+)', name.lower())
    if match:
        return int(match.group(1))
    return None


def _try_gguf_reader(model_path: str, max_layers: Optional[int],
                     skip_1d: bool) -> Optional[Dict[str, np.ndarray]]:
    """
    Attempt to load tensors via the gguf package's GGUFReader.

    Works well for F16/BF16/F32 GGUF files where tensor data is stored
    as plain floats. For quantized files (Q4_K_M, Q2_K, etc.), the data
    is in block-quantized format and needs dequantization — this loader
    will skip those tensors and return None if nothing is loadable.
    """
    try:
        from gguf import GGUFReader
    except ImportError:
        return None

    reader = GGUFReader(model_path)
    tensors = {}
    skipped = 0

    # GGUF type IDs for float formats we can read directly
    # F32=0, F16=1, BF16=30 (varies by gguf version)
    FLOAT_TYPES = {"F32", "F16", "BF16"}

    for tensor in reader.tensors:
        name = tensor.name

        if max_layers is not None:
            idx = extract_layer_index(name)
            if idx is not None and idx >= max_layers:
                skipped += 1
                continue

        # Check if this tensor is in a float format we can read
        ttype = str(tensor.tensor_type).split(".")[-1]
        if ttype not in FLOAT_TYPES:
            skipped += 1
            continue

        shape = tuple(reversed(tensor.shape))
        if len(shape) != 2:
            skipped += 1
            continue

        if skip_1d and (shape[0] == 1 or shape[1] == 1):
            skipped += 1
            continue

        try:
            data = tensor.data.copy().astype(np.float32).reshape(shape)
            tensors[name] = data
        except (ValueError, RuntimeError):
            skipped += 1

    if tensors:
        print(f"  [GGUFReader] Loaded {len(tensors)} 2D tensors, "
              f"skipped {skipped}")
    return tensors if tensors else None


def _try_llama_cpp(model_path: str, max_layers: Optional[int],
                   skip_1d: bool) -> Optional[Dict[str, np.ndarray]]:
    """
    Attempt to load and dequantize tensors via llama-cpp-python.

    This handles quantized GGUF files (Q4_K_M, Q2_K, etc.) by using
    llama.cpp's native dequantization. Requires llama-cpp-python.
    """
    try:
        from llama_cpp import Llama
        import ctypes
    except ImportError:
        return None

    # llama-cpp-python doesn't expose raw tensor access easily.
    # For quantized models, the recommended path is to use
    # llama.cpp's convert tool to dump weights, or use the
    # safetensors/HF format instead.
    #
    # We return None here and fall back to the HF loader.
    return None


def _try_safetensors(model_dir: str, max_layers: Optional[int],
                     skip_1d: bool) -> Optional[Dict[str, np.ndarray]]:
    """
    Load tensors from HuggingFace safetensors format (the original
    unquantized model). This is the most reliable path for getting
    full-precision weights.

    Expects a directory containing model.safetensors or
    model-00001-of-*.safetensors files.
    """
    try:
        from safetensors import safe_open
    except ImportError:
        return None

    import glob

    # Find safetensors files
    patterns = [
        os.path.join(model_dir, "model.safetensors"),
        os.path.join(model_dir, "model-*.safetensors"),
    ]

    files = []
    for pat in patterns:
        files.extend(glob.glob(pat))

    if not files:
        return None

    tensors = {}
    skipped = 0

    for fpath in sorted(files):
        with safe_open(fpath, framework="numpy") as f:
            for name in f.keys():
                if max_layers is not None:
                    idx = extract_layer_index(name)
                    if idx is not None and idx >= max_layers:
                        skipped += 1
                        continue

                data = f.get_tensor(name)
                if data.ndim != 2:
                    skipped += 1
                    continue

                if skip_1d and (data.shape[0] == 1 or data.shape[1] == 1):
                    skipped += 1
                    continue

                tensors[name] = data.astype(np.float32)

    if tensors:
        print(f"  [safetensors] Loaded {len(tensors)} 2D tensors, "
              f"skipped {skipped}")
    return tensors if tensors else None


def load_gguf_tensors(model_path: str,
                      max_layers: Optional[int] = None,
                      skip_1d: bool = True) -> Dict[str, np.ndarray]:
    """
    Load weight tensors from a GGUF file or HuggingFace model directory.

    Tries multiple backends in order:
    1. gguf GGUFReader (works for F16/BF16/F32 GGUF files)
    2. safetensors (if model_path is a directory with .safetensors files)

    For quantized GGUF files (Q4_K_M, Q2_K), the tensors are stored in
    block-quantized format. Pure-Python dequantization is complex and slow.
    Recommended workflow:
    - Use the BF16/F16 GGUF for the primary (original) model analysis
    - For quantized comparison, either:
      (a) Use the HuggingFace safetensors originals + simulate quantization
      (b) Use llama.cpp's `llama-export` to dump dequantized weights

    Parameters
    ----------
    model_path : str
        Path to a .gguf file or a directory with safetensors files.
    max_layers : int, optional
        Only load tensors from the first N transformer layers.
    skip_1d : bool
        Skip 1D tensors (biases, norms).

    Returns
    -------
    dict mapping tensor name -> numpy array (float32).
    """
    print(f"Loading: {model_path}")

    # Try GGUF reader first (for F16/BF16 GGUF files)
    if model_path.endswith(".gguf"):
        result = _try_gguf_reader(model_path, max_layers, skip_1d)
        if result:
            return result
        print("  GGUF file has quantized tensors, cannot read directly.")
        print("  Use a BF16/F16 GGUF or a safetensors directory instead.")
        print("  See README.md for the recommended workflow.")
        sys.exit(1)

    # Try safetensors (for HF model directories)
    if os.path.isdir(model_path):
        result = _try_safetensors(model_path, max_layers, skip_1d)
        if result:
            return result

    print(f"  Could not load tensors from: {model_path}")
    print(f"  Supported formats:")
    print(f"    - F16/BF16 GGUF file (pip install gguf)")
    print(f"    - Directory with .safetensors files (pip install safetensors)")
    sys.exit(1)


# ── Analysis Pipeline ────────────────────────────────────────────────────────

def analyze_single_model(tensors: Dict[str, np.ndarray],
                         model_label: str = "",
                         run_experiments: bool = True) -> List[dict]:
    """
    Run crossword decomposition on all weight matrices in a model.

    Includes experiments from the paper:
    - Hadamard rotation (basis-invariance test)
    - Activation weighting (salient channel amplification)
    - Row-vs-column dominance

    Returns a list of per-layer result dicts.
    """
    results = []

    for name, W in sorted(tensors.items()):
        layer_type = classify_tensor(name)
        layer_idx = extract_layer_index(name)

        # Run crossword decomposition
        decomp = crossword_decompose(
            W, layer_name=name, layer_type=layer_type
        )

        # Run rank-1 SVD for comparison (same parameter budget)
        svd_ve, svd_params = svd_variance_explained(W, rank=1)

        # Row-vs-column dominance
        dom = row_dominance(decomp)

        row = {
            "model": model_label,
            "tensor_name": name,
            "layer_type": layer_type,
            "layer_index": layer_idx if layer_idx is not None else -1,
            "shape": f"{decomp.m}x{decomp.n}",
            "n_params": decomp.m * decomp.n,
            "var_total": decomp.var_total,
            "rho2_pct": decomp.var_explained * 100,
            "var_row_pct": (decomp.var_row / decomp.var_total * 100
                           if decomp.var_total > 0 else 0),
            "var_col_pct": (decomp.var_col / decomp.var_total * 100
                           if decomp.var_total > 0 else 0),
            "row_dominance": dom * 100,
            "svd_r1_pct": svd_ve * 100,
            "compression_gain_bpp": decomp.compression_gain_bpp,
            "overhead_bpp": decomp.overhead_bpp,
            "svd_to_cw_ratio": (svd_ve / decomp.var_explained
                                if decomp.var_explained > 0 else float('inf')),
        }

        # Experiment 1: Hadamard rotation (basis-invariance)
        if run_experiments and layer_type in (
            "embedding", "mlp_gate", "mlp_down", "attention_q", "attention_k"
        ):
            try:
                had_decomp = crossword_after_hadamard(W)
                row["rho2_hadamard_pct"] = had_decomp.var_explained * 100
                row["hadamard_ratio"] = (
                    had_decomp.var_explained / decomp.var_explained
                    if decomp.var_explained > 0 else float('inf')
                )
            except Exception:
                row["rho2_hadamard_pct"] = None
                row["hadamard_ratio"] = None
        else:
            row["rho2_hadamard_pct"] = None
            row["hadamard_ratio"] = None

        # Experiment 2: Activation weighting
        if run_experiments and layer_type in (
            "mlp_gate", "mlp_down", "mlp_up", "attention_q", "attention_v"
        ):
            try:
                act_decomp = crossword_activation_weighted(W)
                row["rho2_actweight_pct"] = act_decomp.var_explained * 100
                raw = decomp.var_explained * 100
                weighted = act_decomp.var_explained * 100
                row["actweight_change_pct"] = (
                    (weighted - raw) / raw * 100 if raw > 0 else 0
                )
            except Exception:
                row["rho2_actweight_pct"] = None
                row["actweight_change_pct"] = None
        else:
            row["rho2_actweight_pct"] = None
            row["actweight_change_pct"] = None

        results.append(row)

    return results


def analyze_quantization_comparison(
    tensors_original: Dict[str, np.ndarray],
    quantized_models: Dict[str, Dict[str, np.ndarray]],
) -> List[dict]:
    """
    Compare crossword structure of quantization error across quant levels.

    For each shared tensor, computes:
    - Crossword decomposition of the error matrix (W_f16 - W_quant)
    - Whether error is additive (high rho^2) or distributed (low rho^2)
    """
    results = []

    for name, W_orig in sorted(tensors_original.items()):
        for quant_label, quant_tensors in quantized_models.items():
            if name not in quant_tensors:
                continue

            W_quant = quant_tensors[name]

            # Shape mismatch can happen with some quant methods
            if W_orig.shape != W_quant.shape:
                continue

            comparison = compare_quantization_error(
                W_orig, W_quant,
                layer_name=name,
                quant_label=quant_label,
            )

            layer_type = classify_tensor(name)
            layer_idx = extract_layer_index(name)

            results.append({
                "tensor_name": name,
                "quant_method": quant_label,
                "layer_type": layer_type,
                "layer_index": layer_idx if layer_idx is not None else -1,
                "shape": f"{W_orig.shape[0]}x{W_orig.shape[1]}",
                "relative_error": comparison["relative_error"],
                "error_frobenius": comparison["error_frobenius"],
                "error_mean_abs": comparison["error_mean"],
                "error_max_abs": comparison["error_max"],
                "error_var_explained": comparison["error_var_explained"] * 100,
                "error_gain_bpp": comparison["error_compression_gain"],
                "original_var_explained": comparison["original_var_explained"] * 100,
            })

    return results


# ── Output: CSV ──────────────────────────────────────────────────────────────

def write_csv_results(results: List[dict], filepath: str):
    """Write a list of result dicts to CSV."""
    if not results:
        print(f"  No results to write to {filepath}")
        return

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    keys = results[0].keys()

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Wrote {len(results)} rows to {filepath}")


# ── Output: Summary Tables ───────────────────────────────────────────────────

def print_summary(results: List[dict], title: str = ""):
    """Print a formatted summary table to stdout."""
    if not results:
        return

    print()
    print("=" * 78)
    print(f"  {title}")
    print("=" * 78)

    # Group by layer type
    by_type = defaultdict(list)
    for r in results:
        by_type[r["layer_type"]].append(r)

    print(f"{'Layer Type':<20} {'Count':>5} {'Avg rho^2':>10} "
          f"{'Avg SVD-r1':>10} {'Avg Gain':>10} {'Row%':>6}")
    print("-" * 66)

    for ltype in sorted(by_type.keys()):
        rows = by_type[ltype]
        n = len(rows)
        avg_cw = np.mean([r["rho2_pct"] for r in rows])
        avg_svd = np.mean([r["svd_r1_pct"] for r in rows])
        avg_gain = np.mean([r["compression_gain_bpp"] for r in rows])
        avg_dom = np.mean([r["row_dominance"] for r in rows])
        print(f"{ltype:<20} {n:>5} {avg_cw:>9.3f}% "
              f"{avg_svd:>9.3f}% {avg_gain:>9.4f} {avg_dom:>5.1f}%")

    # Overall
    all_cw = [r["rho2_pct"] for r in results]
    all_svd = [r["svd_r1_pct"] for r in results]
    print("-" * 66)
    print(f"{'OVERALL':<20} {len(results):>5} "
          f"{np.mean(all_cw):>9.3f}% {np.mean(all_svd):>9.3f}%")
    print()


def print_error_summary(results: List[dict]):
    """Print summary of quantization error structure."""
    if not results:
        return

    print()
    print("=" * 78)
    print("  QUANTIZATION ERROR STRUCTURE")
    print("=" * 78)

    # Group by quant method
    by_quant = defaultdict(list)
    for r in results:
        by_quant[r["quant_method"]].append(r)

    for qmethod in sorted(by_quant.keys()):
        rows = by_quant[qmethod]
        print(f"\n  {qmethod}:")
        print(f"  {'Layer Type':<20} {'Rel Error':>10} "
              f"{'Error rho^2':>12} {'Orig rho^2':>12}")
        print(f"  {'-'*58}")

        by_type = defaultdict(list)
        for r in rows:
            by_type[r["layer_type"]].append(r)

        for ltype in sorted(by_type.keys()):
            trows = by_type[ltype]
            avg_re = np.mean([r["relative_error"] for r in trows])
            avg_err_ve = np.mean([r["error_var_explained"] for r in trows])
            avg_orig_ve = np.mean([r["original_var_explained"] for r in trows])
            print(f"  {ltype:<20} {avg_re:>9.4f} "
                  f"{avg_err_ve:>11.3f}% {avg_orig_ve:>11.3f}%")

    print()


# ── Output: Figures ──────────────────────────────────────────────────────────

def plot_results(results: List[dict], output_dir: str,
                 error_results: Optional[List[dict]] = None):
    """Generate diagnostic figures."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use("Agg")
    except ImportError:
        print("  matplotlib not available, skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    # ── Figure 1: Variance explained by layer type ──
    fig, ax = plt.subplots(figsize=(10, 6))

    by_type = defaultdict(list)
    for r in results:
        by_type[r["layer_type"]].append(r["rho2_pct"])

    types = sorted(by_type.keys())
    positions = range(len(types))
    means = [np.mean(by_type[t]) for t in types]
    stds = [np.std(by_type[t]) for t in types]

    ax.bar(positions, means, yerr=stds, capsize=4,
           color="#407bff", alpha=0.8, edgecolor="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Variance Explained by Crossword Structure (%)")
    ax.set_title("Crossword Structure (ρ²) by Layer Type")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_cw_by_layer_type.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig1_cw_by_layer_type.png")

    # ── Figure 2: Crossword vs SVD-r1 scatter ──
    fig, ax = plt.subplots(figsize=(7, 7))

    cw_vals = [r["rho2_pct"] for r in results]
    svd_vals = [r["svd_r1_pct"] for r in results]

    ax.scatter(cw_vals, svd_vals, alpha=0.5, s=20, c="#407bff")
    lim = max(max(cw_vals), max(svd_vals)) * 1.1
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Crossword Var. Explained (%)")
    ax.set_ylabel("SVD Rank-1 Var. Explained (%)")
    ax.set_title("Crossword vs SVD (Equal Parameter Budget)")
    ax.legend()
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_cw_vs_svd.png"),
                dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved fig2_cw_vs_svd.png")

    # ── Figure 3: rho^2 across layers (depth profile) ──
    layers_with_idx = [(r["layer_index"], r["rho2_pct"],
                        r["layer_type"])
                       for r in results if r["layer_index"] >= 0]

    if layers_with_idx:
        fig, ax = plt.subplots(figsize=(12, 5))

        # Color by type
        type_colors = {
            "attention_q": "#e41a1c", "attention_k": "#ff7f00",
            "attention_v": "#984ea3", "attention_o": "#a65628",
            "mlp_gate": "#377eb8", "mlp_up": "#4daf4a",
            "mlp_down": "#f781bf",
        }

        for idx, ve, lt in layers_with_idx:
            color = type_colors.get(lt, "#999999")
            ax.scatter(idx, ve, c=color, s=30, alpha=0.7)

        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Crossword Var. Explained (%)")
        ax.set_title("Crossword Structure Across Model Depth")
        ax.grid(alpha=0.3)

        # Legend
        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=8, label=t)
                   for t, c in type_colors.items()]
        ax.legend(handles=handles, fontsize=8, ncol=2)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig3_depth_profile.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig3_depth_profile.png")

    # ── Figure 4: Quantization error structure (if comparison data) ──
    if error_results:
        by_quant = defaultdict(list)
        for r in error_results:
            by_quant[r["quant_method"]].append(r)

        fig, axes = plt.subplots(1, len(by_quant), figsize=(7 * len(by_quant), 6),
                                 squeeze=False)

        for i, (qm, rows) in enumerate(sorted(by_quant.items())):
            ax = axes[0][i]

            by_type = defaultdict(list)
            for r in rows:
                by_type[r["layer_type"]].append(r["error_var_explained"])

            types = sorted(by_type.keys())
            positions = range(len(types))
            means = [np.mean(by_type[t]) for t in types]

            ax.bar(positions, means, color="#d62728", alpha=0.8,
                   edgecolor="black", linewidth=0.5)
            ax.set_xticks(positions)
            ax.set_xticklabels(types, rotation=45, ha="right", fontsize=8)
            ax.set_ylabel("Error Var. Explained by CW (%)")
            ax.set_title(f"Quant Error Structure: {qm}")
            ax.grid(axis="y", alpha=0.3)

        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig4_error_structure.png"),
                    dpi=150, bbox_inches="tight")
        plt.close(fig)
        print(f"  Saved fig4_error_structure.png")


# ── LaTeX Table Output ───────────────────────────────────────────────────────

def write_latex_tables(results: List[dict], output_dir: str,
                       error_results: Optional[List[dict]] = None):
    """Generate LaTeX table fragments."""
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, "tables.tex")

    with open(filepath, "w") as f:
        # Table 1: Per-layer-type summary
        f.write("% Auto-generated by crossword_gguf.py\n\n")
        f.write("\\begin{table}[h]\n\\centering\n")
        f.write("\\caption{Crossword structure in trained Qwen-3-8B weights.}\n")
        f.write("\\label{tab:qwen_crossword}\n")
        f.write("\\begin{tabular}{lcccc}\n\\toprule\n")
        f.write("Layer Type & Count & Avg $\\rho^2$ (\\%) "
                "& Avg SVD-r1 (\\%) & Avg Gain (bpp) \\\\\n")
        f.write("\\midrule\n")

        by_type = defaultdict(list)
        for r in results:
            by_type[r["layer_type"]].append(r)

        for ltype in sorted(by_type.keys()):
            rows = by_type[ltype]
            n = len(rows)
            avg_cw = np.mean([r["rho2_pct"] for r in rows])
            avg_svd = np.mean([r["svd_r1_pct"] for r in rows])
            avg_gain = np.mean([r["compression_gain_bpp"] for r in rows])
            f.write(f"{ltype} & {n} & {avg_cw:.3f} "
                    f"& {avg_svd:.3f} & {avg_gain:.4f} \\\\\n")

        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n\n")

        # Table 2: Quantization error structure (if available)
        if error_results:
            f.write("\\begin{table}[h]\n\\centering\n")
            f.write("\\caption{Crossword structure in quantization error "
                    "matrices ($W_{\\text{F16}} - W_{\\text{quant}}$).}\n")
            f.write("\\label{tab:quant_error_structure}\n")
            f.write("\\begin{tabular}{llccc}\n\\toprule\n")
            f.write("Quant & Layer Type & Rel.~Error "
                    "& Error $\\rho^2$ (\\%) & Orig $\\rho^2$ (\\%) \\\\\n")
            f.write("\\midrule\n")

            by_quant = defaultdict(lambda: defaultdict(list))
            for r in error_results:
                by_quant[r["quant_method"]][r["layer_type"]].append(r)

            for qm in sorted(by_quant.keys()):
                first = True
                for lt in sorted(by_quant[qm].keys()):
                    rows = by_quant[qm][lt]
                    avg_re = np.mean([r["relative_error"] for r in rows])
                    avg_eve = np.mean([r["error_var_explained"] for r in rows])
                    avg_ove = np.mean([r["original_var_explained"]
                                       for r in rows])
                    label = qm if first else ""
                    first = False
                    f.write(f"{label} & {lt} & {avg_re:.4f} "
                            f"& {avg_eve:.3f} & {avg_ove:.3f} \\\\\n")
                f.write("\\midrule\n")

            f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")

    print(f"  Wrote LaTeX tables to {filepath}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Crossword structural diagnostics for quantized LLMs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to the primary GGUF model (typically F16).",
    )
    parser.add_argument(
        "--compare", nargs="*", default=[],
        help="Paths to quantized GGUF models to compare against --model.",
    )
    parser.add_argument(
        "--output", default="results",
        help="Output directory for CSV, figures, and LaTeX (default: results/).",
    )
    parser.add_argument(
        "--max-layers", type=int, default=None,
        help="Only analyze the first N transformer layers (for speed).",
    )
    args = parser.parse_args()

    output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    # ── Load primary model ──
    print("\n" + "=" * 60)
    print("  CROSSWORD STRUCTURAL DIAGNOSTIC")
    print("=" * 60)

    primary_tensors = load_gguf_tensors(
        args.model, max_layers=args.max_layers
    )
    model_label = os.path.basename(args.model).replace(".gguf", "")

    # ── Analyze primary model ──
    print(f"\nAnalyzing {model_label}...")
    primary_results = analyze_single_model(primary_tensors, model_label,
                                           run_experiments=True)
    print_summary(primary_results, f"CROSSWORD STRUCTURE: {model_label}")

    write_csv_results(
        primary_results,
        os.path.join(output_dir, f"crossword_{model_label}.csv"),
    )

    # ── Compare with quantized models ──
    error_results = []

    if args.compare:
        quantized_models = {}
        for qpath in args.compare:
            qlabel = os.path.basename(qpath).replace(".gguf", "")
            print(f"\nLoading comparison model: {qlabel}")
            qtensors = load_gguf_tensors(qpath, max_layers=args.max_layers)
            quantized_models[qlabel] = qtensors

            # Also analyze the quantized model on its own
            qresults = analyze_single_model(qtensors, qlabel,
                                            run_experiments=False)
            print_summary(qresults, f"CROSSWORD STRUCTURE: {qlabel}")
            write_csv_results(
                qresults,
                os.path.join(output_dir, f"crossword_{qlabel}.csv"),
            )

        # Compare quantization error structure
        print("\nAnalyzing quantization error structure...")
        error_results = analyze_quantization_comparison(
            primary_tensors, quantized_models
        )
        print_error_summary(error_results)

        if error_results:
            write_csv_results(
                error_results,
                os.path.join(output_dir, "quantization_error_structure.csv"),
            )

    # ── Generate figures ──
    print("\nGenerating figures...")
    plot_results(primary_results, output_dir, error_results or None)

    # ── Generate LaTeX tables ──
    print("\nGenerating LaTeX tables...")
    write_latex_tables(primary_results, output_dir, error_results or None)

    # ── Final summary ──
    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"\nOutput directory: {output_dir}/")
    print(f"  CSV files:    crossword_*.csv")
    if error_results:
        print(f"  Error CSV:    quantization_error_structure.csv")
    print(f"  Figures:      fig*.png")
    print(f"  LaTeX:        tables.tex")
    print()


if __name__ == "__main__":
    main()
