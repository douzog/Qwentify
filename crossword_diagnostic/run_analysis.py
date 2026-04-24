#!/usr/bin/env python3
"""
Multi-Model Crossword Decomposition Analysis
=============================================

Runs crossword decomposition on weight matrices from multiple open-weight
language models, reproducing the experiments from:

    "Crossword Decomposition as a Structural Diagnostic for LLM Quantization"

Supported models (shard 1 / first safetensors file):
    - Qwen3-0.6B, Qwen3-4B, Qwen3-8B  (Alibaba)
    - Mistral-7B                        (Mistral AI)
    - Falcon3-7B                        (TII)
    - Phi-4-14B                         (Microsoft)
    - StableLM2-1.6B                    (Stability AI)

Experiments:
    1. Baseline crossword decomposition + SVD comparison
    2. Hadamard rotation (basis-invariance test)
    3. Activation weighting (salient channel amplification)
    4. MLP deep dive (gate vs up vs down)
    5. Row-vs-column dominance taxonomy

Usage:
    # Analyze all available models
    python run_analysis.py

    # Analyze a specific model
    python run_analysis.py --model qwen3-8b

    # Analyze a custom safetensors path
    python run_analysis.py --path /path/to/model-00001-of-*.safetensors
"""

import sys
import os
import time
import re
import csv
import argparse
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from crossword_decomposition import (
    crossword_decompose,
    svd_variance_explained,
    crossword_after_hadamard,
    crossword_activation_weighted,
    row_dominance,
)

# ── Config ───────────────────────────────────────────────────────────────────

HF_CACHE = os.path.join(os.path.expanduser("~"), ".cache/huggingface/hub")

# Model registry: name -> (HF repo pattern, shard pattern, param count)
MODEL_REGISTRY = {
    "qwen3-0.6b": {
        "repo": "models--Qwen--Qwen3-0.6B",
        "shard": "model-00001-of-*.safetensors",
        "params": "0.6B",
        "org": "Alibaba",
        "arch": "Dense, GQA",
    },
    "gemma3-1b": {
        "repo": "models--google--gemma-3-1b-pt",
        "alt_repo": "models--unsloth--gemma-3-1b-pt",
        "shard": "model-00001-of-*.safetensors",
        "params": "1B",
        "org": "Google",
        "arch": "Dense, GQA, GeGLU",
    },
    "stablelm2-1.6b": {
        "repo": "models--stabilityai--stablelm-2-1_6b",
        "shard": "model-00001-of-*.safetensors",
        "params": "1.6B",
        "org": "Stability AI",
        "arch": "Dense, MHA",
    },
    "qwen3-4b": {
        "repo": "models--Qwen--Qwen3-4B",
        "shard": "model-00001-of-*.safetensors",
        "params": "4B",
        "org": "Alibaba",
        "arch": "Dense, GQA",
    },
    "gemma3-4b": {
        "repo": "models--google--gemma-3-4b-pt",
        "shard": "model-00001-of-*.safetensors",
        "params": "4B",
        "org": "Google",
        "arch": "Dense, GQA, GeGLU",
    },
    "mistral-7b": {
        "repo": "models--mistralai--Mistral-7B-v0.1",
        "shard": "model-00001-of-*.safetensors",
        "params": "7B",
        "org": "Mistral",
        "arch": "Dense, GQA+SWA",
    },
    "falcon3-7b": {
        "repo": "models--tiiuae--Falcon3-7B-Base",
        "shard": "model-00001-of-*.safetensors",
        "params": "7B",
        "org": "TII",
        "arch": "Dense, GQA",
    },
    "qwen3-8b": {
        "repo": "models--Qwen--Qwen3-8B",
        "shard": "model-00001-of-*.safetensors",
        "params": "8.2B",
        "org": "Alibaba",
        "arch": "Dense, GQA",
    },
    "gemma3-12b": {
        "repo": "models--google--gemma-3-12b-pt",
        "shard": "model-00001-of-*.safetensors",
        "params": "12B",
        "org": "Google",
        "arch": "Dense, GQA, GeGLU",
    },
    "phi-4-14b": {
        "repo": "models--microsoft--phi-4",
        "shard": "model-00001-of-*.safetensors",
        "params": "14B",
        "org": "Microsoft",
        "arch": "Dense, GQA",
    },
    "gemma3-27b": {
        "repo": "models--google--gemma-3-27b-pt",
        "shard": "model-00001-of-*.safetensors",
        "params": "27B",
        "org": "Google",
        "arch": "Dense, GQA, GeGLU",
    },
}

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")

# Maximum matrix size before subsampling (paper: 20M entries)
MAX_ENTRIES = 20_000_000
SUBSAMPLE_DIM = 4096


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_tensor(name: str) -> str:
    """Classify a tensor name into a layer type."""
    n = name.lower()
    if "self_attn" in n or "attention" in n or "attn" in n:
        if "q_proj" in n or ".q." in n:
            return "attn_q"
        elif "k_proj" in n or ".k." in n:
            return "attn_k"
        elif "v_proj" in n or ".v." in n:
            return "attn_v"
        elif "o_proj" in n or "out" in n or ".o." in n:
            return "attn_o"
        elif "q_norm" in n or "k_norm" in n:
            return "attn_norm"
        return "attn_other"
    elif "mlp" in n or "ffn" in n or "feed_forward" in n:
        if "gate" in n:
            return "mlp_gate"
        elif "up" in n:
            return "mlp_up"
        elif "down" in n:
            return "mlp_down"
        return "mlp_other"
    elif "embed" in n:
        return "embedding"
    elif "norm" in n:
        return "norm"
    return "other"


def extract_layer_index(name: str) -> int:
    """Extract the layer number from a tensor name."""
    match = re.search(r'(?:layers?|blk)[._](\d+)', name.lower())
    return int(match.group(1)) if match else -1


def find_shard_path(model_key: str) -> Optional[str]:
    """Locate the first safetensors shard for a registered model."""
    import glob

    if model_key not in MODEL_REGISTRY:
        return None

    info = MODEL_REGISTRY[model_key]

    # Try primary repo, then alt_repo if defined
    repos_to_try = [info["repo"]]
    if "alt_repo" in info:
        repos_to_try.append(info["alt_repo"])

    for repo_name in repos_to_try:
        repo_dir = os.path.join(HF_CACHE, repo_name)

        if not os.path.isdir(repo_dir):
            continue

        # Find the snapshot directory
        snapshots_dir = os.path.join(repo_dir, "snapshots")
        if not os.path.isdir(snapshots_dir):
            continue

        # Use the first (or only) snapshot
        snapshots = sorted(os.listdir(snapshots_dir))
        if not snapshots:
            continue

        snap_dir = os.path.join(snapshots_dir, snapshots[-1])

        # Find the shard file
        pattern = os.path.join(snap_dir, info["shard"])
        matches = sorted(glob.glob(pattern))

        # Also try model.safetensors (single-file models)
        if not matches:
            single = os.path.join(snap_dir, "model.safetensors")
            if os.path.exists(single):
                matches = [single]

        if matches:
            return matches[0]

    return None


def load_shard(path: str) -> Dict[str, np.ndarray]:
    """Load 2D weight tensors from a safetensors file."""
    try:
        from safetensors import safe_open
    except ImportError:
        print("  safetensors not installed. Run: pip install safetensors")
        sys.exit(1)

    try:
        import torch
        f = safe_open(path, framework="pt")
    except Exception:
        f = safe_open(path, framework="numpy")

    tensors = {}
    for name in sorted(f.keys()):
        t = f.get_tensor(name)

        # Convert to numpy float32
        if hasattr(t, 'numpy'):
            # PyTorch tensor
            W = t.float().numpy()
        else:
            W = np.asarray(t, dtype=np.float32)

        # Skip non-2D
        if W.ndim != 2:
            continue

        m, n = W.shape
        # Skip tiny matrices (norms stored as 2D with dim=1)
        if m < 4 or n < 4:
            continue

        tensors[name] = W

    return tensors


def subsample_if_needed(W: np.ndarray, name: str,
                        rng: np.random.Generator) -> Tuple[np.ndarray, bool]:
    """Subsample large matrices to SUBSAMPLE_DIM x SUBSAMPLE_DIM."""
    m, n = W.shape
    if m * n <= MAX_ENTRIES:
        return W, False

    sample_rows = min(SUBSAMPLE_DIM, m)
    sample_cols = min(SUBSAMPLE_DIM, n)
    row_idx = rng.choice(m, size=sample_rows, replace=False)
    col_idx = rng.choice(n, size=sample_cols, replace=False)
    return W[np.ix_(row_idx, col_idx)], True


# ── Analysis Pipeline ────────────────────────────────────────────────────────

def analyze_model(model_key: str, shard_path: str,
                  run_experiments: bool = True) -> List[dict]:
    """
    Run full crossword analysis on a single model's shard.

    Returns a list of per-tensor result dicts.
    """
    info = MODEL_REGISTRY.get(model_key, {
        "params": "?", "org": "?", "arch": "?"
    })

    print(f"\n{'=' * 70}")
    print(f"  {model_key.upper()} ({info['params']} params, {info['org']})")
    print(f"  {os.path.basename(shard_path)}")
    print(f"{'=' * 70}\n")

    tensors = load_shard(shard_path)
    print(f"  Loaded {len(tensors)} 2D weight matrices\n")

    rng = np.random.default_rng(42)
    results = []
    total_start = time.time()

    for name, W_orig in tensors.items():
        layer_type = classify_tensor(name)
        layer_idx = extract_layer_index(name)

        # Subsample large matrices
        W, was_sampled = subsample_if_needed(W_orig, name, rng)
        m, n = W.shape

        if was_sampled:
            print(f"  {name:<55} {W_orig.shape[0]:>5}x{W_orig.shape[1]:<5} "
                  f"SAMPLED to {m}x{n}")

        start = time.time()

        # ── Baseline decomposition ──
        decomp = crossword_decompose(W, layer_name=name, layer_type=layer_type)
        svd_ve, svd_params = svd_variance_explained(W, rank=1)
        dom = row_dominance(decomp)

        row = {
            "model": model_key,
            "tensor_name": name,
            "layer_type": layer_type,
            "layer_index": layer_idx,
            "shape": f"{m}x{n}",
            "n_params": m * n,
            "var_total": decomp.var_total,
            "rho2_pct": decomp.var_explained * 100,
            "var_row_pct": (decomp.var_row / decomp.var_total * 100
                           if decomp.var_total > 0 else 0),
            "var_col_pct": (decomp.var_col / decomp.var_total * 100
                           if decomp.var_total > 0 else 0),
            "row_dominance": dom * 100,
            "svd_r1_pct": svd_ve * 100,
            "compression_gain_bpp": decomp.compression_gain_bpp,
            "svd_to_cw_ratio": (svd_ve / decomp.var_explained
                                if decomp.var_explained > 0
                                else float('inf')),
        }

        # ── Experiment 1: Hadamard rotation ──
        if run_experiments and layer_type in (
            "embedding", "mlp_gate", "mlp_down", "attn_q", "attn_k"
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

        # ── Experiment 2: Activation weighting ──
        if run_experiments and layer_type in (
            "mlp_gate", "mlp_down", "mlp_up", "attn_q", "attn_v"
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

        elapsed = time.time() - start
        row["time_sec"] = round(elapsed, 2)
        results.append(row)

        # Progress
        rho2_str = f"{decomp.var_explained * 100:>7.3f}%"
        svd_str = f"{svd_ve * 100:>7.3f}%"
        dom_str = f"row={dom * 100:>4.1f}%"
        print(f"  {name:<55} {m:>5}x{n:<5} "
              f"rho2={rho2_str}  SVD1={svd_str}  {dom_str}  "
              f"({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\n  Analyzed {len(results)} matrices in {total_elapsed:.1f}s")

    return results


# ── Summary Tables ───────────────────────────────────────────────────────────

def print_by_type_summary(results: List[dict], model_key: str):
    """Print per-layer-type summary (paper Table 2)."""
    print(f"\n{'=' * 78}")
    print(f"  BY LAYER TYPE: {model_key}")
    print(f"{'=' * 78}")

    by_type = defaultdict(list)
    for r in results:
        by_type[r["layer_type"]].append(r)

    print(f"{'Type':<16} {'Count':>5} {'Avg rho2':>10} "
          f"{'Avg SVD-r1':>10} {'SVD/CW':>8} {'Row%':>6}")
    print("-" * 60)

    for ltype in sorted(by_type.keys()):
        rows = by_type[ltype]
        n = len(rows)
        avg_cw = np.mean([r["rho2_pct"] for r in rows])
        avg_svd = np.mean([r["svd_r1_pct"] for r in rows])
        avg_ratio = np.mean([r["svd_to_cw_ratio"] for r in rows
                             if r["svd_to_cw_ratio"] < float('inf')])
        avg_dom = np.mean([r["row_dominance"] for r in rows])
        print(f"{ltype:<16} {n:>5} {avg_cw:>9.3f}% "
              f"{avg_svd:>9.3f}% {avg_ratio:>7.1f}x {avg_dom:>5.1f}%")

    all_cw = [r["rho2_pct"] for r in results]
    print("-" * 60)
    print(f"{'OVERALL':<16} {len(results):>5} {np.mean(all_cw):>9.3f}%")


def print_cross_model_summary(all_results: Dict[str, List[dict]]):
    """Print cross-model comparison (paper Table 1)."""
    print(f"\n{'=' * 90}")
    print(f"  CROSS-MODEL COMPARISON")
    print(f"{'=' * 90}")

    print(f"{'Model':<20} {'Params':>6} {'Overall rho2':>13} "
          f"{'Gate rho2':>10} {'Gate Col%':>10} {'Embed rho2':>11}")
    print("-" * 75)

    for model_key in MODEL_REGISTRY:
        if model_key not in all_results:
            continue

        results = all_results[model_key]
        info = MODEL_REGISTRY[model_key]

        overall = np.mean([r["rho2_pct"] for r in results])

        gate_rows = [r for r in results if r["layer_type"] == "mlp_gate"]
        gate_rho2 = np.mean([r["rho2_pct"] for r in gate_rows]) if gate_rows else 0
        gate_col = np.mean([100 - r["row_dominance"] for r in gate_rows]) if gate_rows else 0

        embed_rows = [r for r in results if r["layer_type"] == "embedding"]
        embed_rho2 = np.mean([r["rho2_pct"] for r in embed_rows]) if embed_rows else 0

        print(f"{model_key:<20} {info['params']:>6} {overall:>12.3f}% "
              f"{gate_rho2:>9.3f}% {gate_col:>9.1f}% {embed_rho2:>10.2f}%")


def print_hadamard_summary(results: List[dict], model_key: str):
    """Print Hadamard invariance results (paper Table 4)."""
    had_rows = [r for r in results
                if r.get("rho2_hadamard_pct") is not None]
    if not had_rows:
        return

    print(f"\n{'=' * 70}")
    print(f"  HADAMARD INVARIANCE: {model_key}")
    print(f"{'=' * 70}")

    print(f"{'Matrix':<50} {'Before':>8} {'After':>8} {'Ratio':>7}")
    print("-" * 75)

    for r in had_rows[:10]:  # Show first 10
        before = r["rho2_pct"]
        after = r["rho2_hadamard_pct"]
        ratio = r["hadamard_ratio"] if r["hadamard_ratio"] != float('inf') else 0
        print(f"{r['tensor_name']:<50} {before:>7.3f}% {after:>7.3f}% {ratio:>6.2f}")


def print_activation_summary(results: List[dict], model_key: str):
    """Print activation weighting results (paper Table 5)."""
    act_rows = [r for r in results
                if r.get("rho2_actweight_pct") is not None]
    if not act_rows:
        return

    print(f"\n{'=' * 70}")
    print(f"  ACTIVATION WEIGHTING: {model_key}")
    print(f"{'=' * 70}")

    print(f"{'Matrix':<50} {'Raw':>8} {'Weighted':>9} {'Change':>8}")
    print("-" * 78)

    for r in act_rows[:10]:
        raw = r["rho2_pct"]
        weighted = r["rho2_actweight_pct"]
        change = r["actweight_change_pct"]
        print(f"{r['tensor_name']:<50} {raw:>7.3f}% {weighted:>8.3f}% "
              f"{change:>+7.1f}%")


def print_dominance_summary(results: List[dict], model_key: str):
    """Print row-vs-column dominance (paper Table 6)."""
    print(f"\n{'=' * 70}")
    print(f"  ROW vs COLUMN DOMINANCE: {model_key}")
    print(f"{'=' * 70}")

    by_type = defaultdict(list)
    for r in results:
        by_type[r["layer_type"]].append(r)

    print(f"{'Type':<16} {'Count':>5} {'rho2':>8} {'Row%':>7} {'Dominant':>12}")
    print("-" * 52)

    for ltype in sorted(by_type.keys()):
        rows = by_type[ltype]
        n = len(rows)
        avg_rho2 = np.mean([r["rho2_pct"] for r in rows])
        avg_dom = np.mean([r["row_dominance"] for r in rows])

        if avg_dom < 30:
            label = "column"
        elif avg_dom > 70:
            label = "row"
        else:
            label = "balanced"

        print(f"{ltype:<16} {n:>5} {avg_rho2:>7.3f}% {avg_dom:>6.1f}% {label:>12}")


# ── CSV Output ───────────────────────────────────────────────────────────────

def write_results_csv(results: List[dict], filepath: str):
    """Write results to CSV."""
    if not results:
        return

    os.makedirs(os.path.dirname(filepath) or ".", exist_ok=True)
    keys = results[0].keys()

    with open(filepath, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)

    print(f"  Wrote {len(results)} rows to {filepath}")


# ── Figure Generation ────────────────────────────────────────────────────────


def generate_figures(all_results: Dict[str, List[dict]], output_dir: str):
    """Generate diagnostic figures for all analyzed models."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except ImportError:
        print("\n  matplotlib not available, skipping figures")
        return

    os.makedirs(output_dir, exist_ok=True)

    type_color_map = {
        "attn_q": "#e41a1c", "attn_k": "#ff7f00",
        "attn_v": "#984ea3", "attn_o": "#a65628",
        "mlp_gate": "#377eb8", "mlp_up": "#4daf4a",
        "mlp_down": "#f781bf", "embedding": "#000000",
    }

    # Flatten all results
    flat = []
    for model_key, results in all_results.items():
        flat.extend(results)

    # ── Fig 1: rho^2 by layer type (all models) ──
    by_type = defaultdict(list)
    for r in flat:
        by_type[r["layer_type"]].append(r["rho2_pct"])

    types = sorted(by_type.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(types))
    means = [np.mean(by_type[t]) for t in types]
    stds = [np.std(by_type[t]) for t in types]

    ax.bar(positions, means, yerr=stds, capsize=4,
           color="#407bff", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.set_xticks(positions)
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Variance Explained by Crossword Structure (%)")
    ax.set_title("Crossword Structure (ρ²) by Layer Type\n"
                 f"({len(all_results)} models, {len(flat)} matrices)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_cw_by_layer_type.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_cw_by_layer_type.png")

    # ── Fig 2: CW vs SVD scatter ──
    fig, ax = plt.subplots(figsize=(7, 7))
    for r in flat:
        c = type_color_map.get(r["layer_type"], "#999999")
        ax.scatter(r["rho2_pct"], r["svd_r1_pct"], c=c, alpha=0.5,
                   s=25, edgecolors="black", linewidth=0.2)

    cw_vals = [r["rho2_pct"] for r in flat]
    svd_vals = [r["svd_r1_pct"] for r in flat]
    lim = max(max(cw_vals), max(svd_vals)) * 1.15
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Crossword Var. Explained (%)")
    ax.set_ylabel("SVD Rank-1 Var. Explained (%)")
    ax.set_title("Crossword vs SVD-r1 (Equal Parameter Budget)")

    handles = [Line2D([0], [0], marker="o", color="w",
                      markerfacecolor=c, markersize=8, label=t)
               for t, c in type_color_map.items()
               if t in set(r["layer_type"] for r in flat)]
    ax.legend(handles=handles, fontsize=7, ncol=2)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_cw_vs_svd.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_cw_vs_svd.png")

    # ── Fig 3: Depth profile (per model) ──
    fig, axes = plt.subplots(1, len(all_results),
                             figsize=(5 * len(all_results), 5),
                             squeeze=False, sharey=True)

    for i, (model_key, results) in enumerate(sorted(all_results.items())):
        ax = axes[0][i]
        layers = [(r["layer_index"], r["rho2_pct"], r["layer_type"])
                  for r in results if r["layer_index"] >= 0]

        for idx, ve, lt in layers:
            c = type_color_map.get(lt, "#999999")
            ax.scatter(idx, ve, c=c, s=30, alpha=0.7,
                       edgecolors="black", linewidth=0.3)

        ax.set_xlabel("Layer Index")
        if i == 0:
            ax.set_ylabel("ρ² (%)")
        ax.set_title(model_key, fontsize=10)
        ax.grid(alpha=0.3)

    fig.suptitle("Crossword Structure Across Model Depth", fontsize=12)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig3_depth_profile.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig3_depth_profile.png")

    # ── Fig 4: Cross-model comparison (gate rho2 + embed rho2) ──
    models_with_data = [k for k in MODEL_REGISTRY if k in all_results]
    if len(models_with_data) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        model_labels = []
        gate_vals = []
        embed_vals = []
        overall_vals = []

        for mk in models_with_data:
            results = all_results[mk]
            info = MODEL_REGISTRY.get(mk, {})
            label = f"{mk}\n({info.get('params', '?')})"
            model_labels.append(label)

            gate_rows = [r for r in results if r["layer_type"] == "mlp_gate"]
            gate_vals.append(np.mean([r["rho2_pct"] for r in gate_rows])
                             if gate_rows else 0)

            embed_rows = [r for r in results if r["layer_type"] == "embedding"]
            embed_vals.append(np.mean([r["rho2_pct"] for r in embed_rows])
                              if embed_rows else 0)

            overall_vals.append(np.mean([r["rho2_pct"] for r in results]))

        x = range(len(model_labels))

        ax1.bar(x, gate_vals, color="#377eb8", alpha=0.85,
                edgecolor="black", linewidth=0.5)
        ax1.set_xticks(x)
        ax1.set_xticklabels(model_labels, fontsize=8)
        ax1.set_ylabel("Gate ρ² (%)")
        ax1.set_title("MLP Gate Crossword Structure")
        ax1.grid(axis="y", alpha=0.3)

        ax2.bar(x, embed_vals, color="#2ca02c", alpha=0.85,
                edgecolor="black", linewidth=0.5)
        ax2.set_xticks(x)
        ax2.set_xticklabels(model_labels, fontsize=8)
        ax2.set_ylabel("Embedding ρ² (%)")
        ax2.set_title("Embedding Crossword Structure (Inverse Scaling)")
        ax2.grid(axis="y", alpha=0.3)

        fig.suptitle("Cross-Model Comparison", fontsize=12)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig4_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved fig4_comparison.png")
    else:
        # Single model: comparison bar chart
        results = list(all_results.values())[0]
        fig, ax = plt.subplots(figsize=(8, 5))
        categories = ["Initialized\n(Xavier)",
                      f"Trained\n({list(all_results.keys())[0]})",
                      "CHILDES-1\nCo-occurrence"]
        values = [0.27, np.mean([r["rho2_pct"] for r in results]), 24.1]
        bar_colors = ["#d62728", "#407bff", "#2ca02c"]

        ax.bar(categories, values, color=bar_colors, alpha=0.85,
               edgecolor="black", linewidth=0.5)
        ax.set_ylabel("Variance Explained by Crossword Structure (%)")
        ax.set_title("Crossword Structure: Where Do Trained Weights Fall?")
        ax.grid(axis="y", alpha=0.3)
        for i, v in enumerate(values):
            ax.text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=10,
                    fontweight="bold")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig4_comparison.png"), dpi=150)
        plt.close(fig)
        print(f"  Saved fig4_comparison.png")

    # ── Fig 5: Row-vs-column dominance ──
    fig, ax = plt.subplots(figsize=(10, 6))

    by_type_dom = defaultdict(list)
    for r in flat:
        by_type_dom[r["layer_type"]].append(r["row_dominance"])

    types = sorted(by_type_dom.keys())
    positions = range(len(types))
    means = [np.mean(by_type_dom[t]) for t in types]

    colors = ["#377eb8" if m < 30 else "#d62728" if m > 70 else "#999999"
              for m in means]

    ax.bar(positions, means, color=colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.axhline(y=50, color="black", linestyle="--", alpha=0.3)
    ax.set_xticks(positions)
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Row Dominance (%)")
    ax.set_title("Row vs Column Dominance by Layer Type\n"
                 "(< 30% = column-dominated, > 70% = row-dominated)")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig5_dominance.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig5_dominance.png")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Multi-model crossword decomposition analysis.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=("Model key to analyze (e.g., qwen3-8b). "
              "If omitted, analyzes all available models."),
    )
    parser.add_argument(
        "--path", type=str, default=None,
        help="Direct path to a safetensors file (overrides --model).",
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output directory (default: results/).",
    )
    parser.add_argument(
        "--no-experiments", action="store_true",
        help="Skip Hadamard/activation experiments (faster).",
    )
    parser.add_argument(
        "--list-models", action="store_true",
        help="List registered models and their availability.",
    )
    args = parser.parse_args()

    output_dir = args.output or OUTPUT_DIR
    os.makedirs(output_dir, exist_ok=True)

    # ── List models ──
    if args.list_models:
        print(f"\n{'Model':<20} {'Params':>6} {'Org':<15} {'Available':>10}")
        print("-" * 55)
        for key, info in MODEL_REGISTRY.items():
            path = find_shard_path(key)
            avail = "YES" if path else "no"
            print(f"{key:<20} {info['params']:>6} {info['org']:<15} {avail:>10}")
        print()
        return

    # ── Determine which models to analyze ──
    models_to_run = []

    if args.path:
        # Direct path mode
        model_key = os.path.basename(args.path).replace(".safetensors", "")
        models_to_run.append((model_key, args.path))
    elif args.model:
        # Single model mode
        if args.model not in MODEL_REGISTRY:
            print(f"Unknown model: {args.model}")
            print(f"Available: {', '.join(MODEL_REGISTRY.keys())}")
            sys.exit(1)
        path = find_shard_path(args.model)
        if not path:
            print(f"Model {args.model} not found in HF cache.")
            print(f"Download it first:")
            info = MODEL_REGISTRY[args.model]
            repo = info["repo"].replace("models--", "").replace("--", "/")
            print(f"  huggingface-cli download {repo}")
            sys.exit(1)
        models_to_run.append((args.model, path))
    else:
        # All available models
        for key in MODEL_REGISTRY:
            path = find_shard_path(key)
            if path:
                models_to_run.append((key, path))

        if not models_to_run:
            print("No models found in HF cache.")
            print("Download at least one model first. Available models:")
            for key, info in MODEL_REGISTRY.items():
                repo = info["repo"].replace("models--", "").replace("--", "/")
                print(f"  huggingface-cli download {repo}")
            sys.exit(1)

    # ── Run analysis ──
    print("\n" + "=" * 70)
    print("  CROSSWORD DECOMPOSITION: MULTI-MODEL ANALYSIS")
    print(f"  Models: {len(models_to_run)}")
    print("=" * 70)

    all_results = {}
    run_experiments = not args.no_experiments

    for model_key, shard_path in models_to_run:
        results = analyze_model(model_key, shard_path, run_experiments)
        all_results[model_key] = results

        # Per-model summaries
        print_by_type_summary(results, model_key)
        if run_experiments:
            print_hadamard_summary(results, model_key)
            print_activation_summary(results, model_key)
        print_dominance_summary(results, model_key)

        # Per-model CSV
        csv_name = f"crossword_{model_key.replace('-', '_')}.csv"
        write_results_csv(results, os.path.join(output_dir, csv_name))

    # ── Cross-model summary ──
    if len(all_results) > 1:
        print_cross_model_summary(all_results)

        # Combined CSV
        flat = []
        for results in all_results.values():
            flat.extend(results)
        write_results_csv(flat, os.path.join(output_dir,
                                             "crossword_all_models.csv"))

    # ── Figures ──
    print("\nGenerating figures...")
    generate_figures(all_results, output_dir)

    # ── Key findings ──
    flat = []
    for results in all_results.values():
        flat.extend(results)

    all_rho2 = [r["rho2_pct"] for r in flat]

    print(f"\n{'=' * 70}")
    print(f"  KEY FINDINGS")
    print(f"{'=' * 70}")
    print(f"  Models analyzed:     {len(all_results)}")
    print(f"  Total matrices:      {len(flat)}")
    print(f"  Overall avg rho2:    {np.mean(all_rho2):.3f}%")
    print(f"  Max overall rho2:    {max(all_rho2):.3f}%")

    gate_rows = [r for r in flat if r["layer_type"] == "mlp_gate"]
    if gate_rows:
        gate_rho2 = np.mean([r["rho2_pct"] for r in gate_rows])
        gate_col = np.mean([100 - r["row_dominance"] for r in gate_rows])
        print(f"  Gate avg rho2:       {gate_rho2:.3f}%")
        print(f"  Gate col dominance:  {gate_col:.1f}%")

    embed_rows = [r for r in flat if r["layer_type"] == "embedding"]
    if embed_rows:
        for r in embed_rows:
            model = r["model"]
            info = MODEL_REGISTRY.get(model, {})
            print(f"  Embed rho2 ({info.get('params', '?')}): "
                  f"{r['rho2_pct']:.2f}%")

    # Proposition 2 check
    prop2_holds = all(r["svd_r1_pct"] >= r["rho2_pct"] for r in flat)
    print(f"\n  Proposition 2 (SVD-r1 >= CW): "
          f"{'CONFIRMED' if prop2_holds else 'VIOLATED'} "
          f"on all {len(flat)} matrices")

    print(f"\n  Output: {output_dir}/")
    print()


if __name__ == "__main__":
    main()
