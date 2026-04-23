#!/usr/bin/env python3
"""
Run crossword decomposition on Qwen-3-8B shard 1 (layers 0-6 + embedding).

This script loads real trained weights from the safetensors file,
runs crossword decomposition on every 2D weight matrix, compares
against SVD rank-1, and outputs CSV + figures.
"""

import sys
import os
import time
import re
import csv
from collections import defaultdict

import numpy as np
import torch
from safetensors import safe_open

sys.path.insert(0, os.path.dirname(__file__))
from crossword_decomposition import crossword_decompose, svd_variance_explained

# ── Config ───────────────────────────────────────────────────────────────────

SHARD_PATH = os.path.join(
    os.path.expanduser("~"),
    ".cache/huggingface/hub/models--Qwen--Qwen3-8B/"
    "snapshots/b968826d9c46dd6066d109eabc6255188de91218/"
    "model-00001-of-00005.safetensors"
)

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "results")


# ── Helpers ──────────────────────────────────────────────────────────────────

def classify_tensor(name):
    n = name.lower()
    if "self_attn" in n or "attention" in n:
        if "q_proj" in n:
            return "attention_q"
        elif "k_proj" in n:
            return "attention_k"
        elif "v_proj" in n:
            return "attention_v"
        elif "o_proj" in n:
            return "attention_o"
        elif "q_norm" in n or "k_norm" in n:
            return "attention_norm"
        return "attention_other"
    elif "mlp" in n:
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


def extract_layer_index(name):
    match = re.search(r'layers\.(\d+)', name)
    return int(match.group(1)) if match else -1


# ── Main Analysis ────────────────────────────────────────────────────────────

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("  CROSSWORD DECOMPOSITION: Qwen-3-8B Trained Weights")
    print("  Layers 0-6 + Embedding (shard 1)")
    print("=" * 70)
    print()

    if not os.path.exists(SHARD_PATH):
        print(f"Shard not found: {SHARD_PATH}")
        sys.exit(1)

    print(f"Loading: {os.path.basename(SHARD_PATH)}")
    f = safe_open(SHARD_PATH, framework="pt")
    tensor_names = sorted(f.keys())
    print(f"  Total tensors in shard: {len(tensor_names)}")

    results = []
    total_start = time.time()

    for name in tensor_names:
        t = f.get_tensor(name)

        # Skip 1D tensors (norms, biases)
        if t.ndim != 2:
            continue

        # Convert bf16 -> float32 numpy
        W = t.float().numpy()
        m, n = W.shape
        layer_type = classify_tensor(name)
        layer_idx = extract_layer_index(name)

        # Skip very small matrices (norms stored as 2D with dim=1)
        if m < 4 or n < 4:
            continue

        # Skip the full embedding matrix — it's 151936x4096 (622M entries)
        # and SVD alone takes 10+ minutes. We sample it instead.
        # Also sample MLP matrices (4096x12288 = 50M entries).
        if m * n > 20_000_000:
            sample_rows = min(4096, m)
            sample_cols = min(4096, n)
            rng = np.random.default_rng(42)
            row_idx = rng.choice(m, size=sample_rows, replace=False)
            col_idx = rng.choice(n, size=sample_cols, replace=False)
            W_sampled = W[np.ix_(row_idx, col_idx)]
            print(f"  {name:<55} {m:>5}x{n:<5} "
                  f"SAMPLED to {sample_rows}x{sample_cols}")
            W = W_sampled
            m, n = W.shape

        start = time.time()

        # Crossword decomposition
        decomp = crossword_decompose(W, layer_name=name, layer_type=layer_type)

        # SVD rank-1 comparison
        svd_ve, svd_params = svd_variance_explained(W, rank=1)

        elapsed = time.time() - start

        row = {
            "tensor_name": name,
            "layer_type": layer_type,
            "layer_index": layer_idx,
            "shape": f"{m}x{n}",
            "n_params": m * n,
            "var_total": decomp.var_total,
            "var_explained_cw": decomp.var_explained * 100,
            "var_row_pct": (decomp.var_row / decomp.var_total * 100
                           if decomp.var_total > 0 else 0),
            "var_col_pct": (decomp.var_col / decomp.var_total * 100
                           if decomp.var_total > 0 else 0),
            "var_explained_svd1": svd_ve * 100,
            "compression_gain_bpp": decomp.compression_gain_bpp,
            "svd_to_cw_ratio": (svd_ve / decomp.var_explained
                                if decomp.var_explained > 0 else float('inf')),
            "time_sec": round(elapsed, 2),
        }
        results.append(row)

        # Print progress
        print(f"  {name:<55} {m:>5}x{n:<5} "
              f"rho2={decomp.var_explained*100:>7.3f}%  "
              f"SVD1={svd_ve*100:>7.3f}%  "
              f"gain={decomp.compression_gain_bpp:>7.4f} bpp  "
              f"({elapsed:.1f}s)")

    total_elapsed = time.time() - total_start
    print(f"\nAnalyzed {len(results)} weight matrices in {total_elapsed:.1f}s")

    # ── Write CSV ────────────────────────────────────────────────────────
    csv_path = os.path.join(OUTPUT_DIR, "qwen3_8b_crossword_analysis.csv")
    with open(csv_path, "w", newline="") as csvf:
        writer = csv.DictWriter(csvf, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"\nCSV saved: {csv_path}")

    # ── Summary by layer type ────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  SUMMARY BY LAYER TYPE")
    print("=" * 70)

    by_type = defaultdict(list)
    for r in results:
        by_type[r["layer_type"]].append(r)

    print(f"{'Layer Type':<20} {'Count':>5} {'Avg rho^2':>12} "
          f"{'Avg SVD-r1':>12} {'Avg Gain':>12} {'SVD/CW':>8}")
    print("-" * 72)

    for ltype in sorted(by_type.keys()):
        rows = by_type[ltype]
        n = len(rows)
        avg_cw = np.mean([r["var_explained_cw"] for r in rows])
        avg_svd = np.mean([r["var_explained_svd1"] for r in rows])
        avg_gain = np.mean([r["compression_gain_bpp"] for r in rows])
        avg_ratio = np.mean([r["svd_to_cw_ratio"] for r in rows
                             if r["svd_to_cw_ratio"] < float('inf')])
        print(f"{ltype:<20} {n:>5} {avg_cw:>11.4f}% "
              f"{avg_svd:>11.4f}% {avg_gain:>11.5f} {avg_ratio:>7.2f}x")

    all_cw = [r["var_explained_cw"] for r in results]
    all_svd = [r["var_explained_svd1"] for r in results]
    print("-" * 72)
    print(f"{'OVERALL':<20} {len(results):>5} "
          f"{np.mean(all_cw):>11.4f}% {np.mean(all_svd):>11.4f}%")

    # ── Key finding ─────────────────────────────────────────────
    print()
    print("=" * 70)
    print("  KEY COMPARISON: TRAINED vs INITIALIZED")
    print("=" * 70)
    print(f"  Initialized (Xavier):        <0.30% crossword structure")
    print(f"  This analysis (trained):     {np.mean(all_cw):.4f}% avg")
    print(f"  CHILDES-1 co-occurrence:     24.1% crossword structure")
    print()

    ratio = np.mean(all_cw) / 0.27  # initialized baseline
    if ratio > 2:
        print(f"  FINDING: Trained weights show {ratio:.1f}x MORE crossword")
        print(f"  structure than initialized weights. Training induces")
        print(f"  additive row-column structure.")
    elif ratio > 0.8:
        print(f"  FINDING: Trained weights show similar crossword structure")
        print(f"  to initialized weights ({ratio:.1f}x). Training does NOT")
        print(f"  significantly induce additive structure.")
    else:
        print(f"  FINDING: Trained weights show LESS crossword structure")
        print(f"  than initialized weights ({ratio:.1f}x).")

    # ── Proposition 2 check ──────────────────────────────────────────────
    prop2_holds = all(r["var_explained_svd1"] >= r["var_explained_cw"]
                      for r in results)
    print()
    print(f"  Proposition 2 (SVD-r1 >= Crossword): "
          f"{'CONFIRMED' if prop2_holds else 'VIOLATED'} "
          f"on all {len(results)} matrices")

    # ── Generate figures ─────────────────────────────────────────────────
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        generate_figures(results, OUTPUT_DIR)
    except ImportError:
        print("\nmatplotlib not available, skipping figures")


def generate_figures(results, output_dir):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # ── Fig 1: rho^2 by layer type ──
    by_type = defaultdict(list)
    for r in results:
        by_type[r["layer_type"]].append(r["var_explained_cw"])

    types = sorted(by_type.keys())
    fig, ax = plt.subplots(figsize=(10, 6))
    positions = range(len(types))
    means = [np.mean(by_type[t]) for t in types]
    stds = [np.std(by_type[t]) for t in types]

    bars = ax.bar(positions, means, yerr=stds, capsize=4,
                  color="#407bff", alpha=0.85, edgecolor="black",
                  linewidth=0.5)
    ax.axhline(y=0.27, color="red", linestyle="--", alpha=0.7,
               label="Initialized baseline: 0.27%")
    ax.set_xticks(positions)
    ax.set_xticklabels(types, rotation=45, ha="right", fontsize=9)
    ax.set_ylabel("Variance Explained by Crossword Structure (%)")
    ax.set_title("Crossword Structure in Trained Qwen-3-8B Weights\n"
                 "(Layers 0–6)")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig1_cw_by_layer_type.png"),
                dpi=150)
    plt.close(fig)
    print(f"  Saved fig1_cw_by_layer_type.png")

    # ── Fig 2: CW vs SVD scatter ──
    fig, ax = plt.subplots(figsize=(7, 7))
    cw = [r["var_explained_cw"] for r in results]
    svd = [r["var_explained_svd1"] for r in results]
    colors = []
    type_color_map = {
        "attention_q": "#e41a1c", "attention_k": "#ff7f00",
        "attention_v": "#984ea3", "attention_o": "#a65628",
        "mlp_gate": "#377eb8", "mlp_up": "#4daf4a",
        "mlp_down": "#f781bf", "embedding": "#000000",
    }
    for r in results:
        colors.append(type_color_map.get(r["layer_type"], "#999999"))

    ax.scatter(cw, svd, c=colors, alpha=0.7, s=40, edgecolors="black",
               linewidth=0.3)
    lim = max(max(cw), max(svd)) * 1.15
    ax.plot([0, lim], [0, lim], "k--", alpha=0.3, label="y = x")
    ax.set_xlabel("Crossword Var. Explained (%)")
    ax.set_ylabel("SVD Rank-1 Var. Explained (%)")
    ax.set_title("Crossword vs SVD-r1 (Equal Parameter Budget)\n"
                 "Proposition 2: all points above the diagonal")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig2_cw_vs_svd.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig2_cw_vs_svd.png")

    # ── Fig 3: Depth profile ──
    layers = [(r["layer_index"], r["var_explained_cw"], r["layer_type"])
              for r in results if r["layer_index"] >= 0]

    if layers:
        fig, ax = plt.subplots(figsize=(12, 5))
        for idx, ve, lt in layers:
            c = type_color_map.get(lt, "#999999")
            ax.scatter(idx, ve, c=c, s=40, alpha=0.7,
                       edgecolors="black", linewidth=0.3)

        ax.axhline(y=0.27, color="red", linestyle="--", alpha=0.5,
                   label="Initialized baseline")
        ax.set_xlabel("Layer Index")
        ax.set_ylabel("Crossword Var. Explained (%)")
        ax.set_title("Crossword Structure Across Model Depth (Qwen-3-8B)")
        ax.grid(alpha=0.3)

        from matplotlib.lines import Line2D
        handles = [Line2D([0], [0], marker="o", color="w",
                          markerfacecolor=c, markersize=8, label=t)
                   for t, c in type_color_map.items()
                   if t in set(r["layer_type"] for r in results)]
        handles.append(Line2D([0], [0], color="red", linestyle="--",
                              label="Initialized baseline"))
        ax.legend(handles=handles, fontsize=8, ncol=2,
                  loc="upper right")
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, "fig3_depth_profile.png"),
                    dpi=150)
        plt.close(fig)
        print(f"  Saved fig3_depth_profile.png")

    # ── Fig 4: Comparison bar chart (trained vs initialized vs CHILDES) ──
    fig, ax = plt.subplots(figsize=(8, 5))
    categories = ["Initialized\n(Xavier)", "Trained\n(This Analysis)",
                  "CHILDES-1\nCo-occurrence"]
    values = [0.27, np.mean([r["var_explained_cw"] for r in results]), 24.1]
    bar_colors = ["#d62728", "#407bff", "#2ca02c"]

    ax.bar(categories, values, color=bar_colors, alpha=0.85,
           edgecolor="black", linewidth=0.5)
    ax.set_ylabel("Variance Explained by Crossword Structure (%)")
    ax.set_title("Crossword Structure: Where Does Qwen-3-8B Fall?")
    ax.grid(axis="y", alpha=0.3)

    for i, v in enumerate(values):
        ax.text(i, v + 0.3, f"{v:.2f}%", ha="center", fontsize=10,
                fontweight="bold")

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, "fig4_comparison.png"), dpi=150)
    plt.close(fig)
    print(f"  Saved fig4_comparison.png")


if __name__ == "__main__":
    main()
