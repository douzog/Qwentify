"""
Crossword Decomposition: Core Mathematics
==========================================

Implements the crossword encoding framework for matrix compression.

The decomposition:
    W = μ·11ᵀ + r·1ᵀ + 1·cᵀ + R

where:
    μ = grand mean
    r = row effects (zero-sum)
    c = column effects (zero-sum)
    R = residual matrix

This module is intentionally dependency-light (numpy only) so it can be:
    1. Called from the GGUF analysis pipeline
    2. Ported to R
    3. Reimplemented in Rust for high-performance parsing
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class CrosswordResult:
    """Result of a crossword decomposition on a single matrix."""
    # Decomposition components
    mu: float                    # Grand mean
    row_effects: np.ndarray      # r vector (m,)
    col_effects: np.ndarray      # c vector (n,)
    residual: np.ndarray         # R matrix (m, n)

    # Dimensions
    m: int
    n: int

    # Variance decomposition
    var_total: float
    var_row: float
    var_col: float
    var_residual: float
    var_explained: float         # ρ² = 1 - Var(R)/Var(W)

    # Compression metrics
    compression_gain_bpp: float  # ½ log₂(Var(W)/Var(R))
    overhead_bpp: float          # k(1+m+n)/(mn) for k=32
    crossword_rate_bpp: float    # overhead + H_k(R) estimate

    # Layer metadata (filled in by caller)
    layer_name: str = ""
    layer_type: str = ""


def crossword_decompose(W: np.ndarray, layer_name: str = "",
                        layer_type: str = "", bits: int = 32) -> CrosswordResult:
    """
    Compute the crossword decomposition of matrix W.

    Parameters
    ----------
    W : np.ndarray
        Weight matrix of shape (m, n).
    layer_name : str
        Human-readable name for this layer.
    layer_type : str
        Category: 'attention', 'mlp', 'embedding', 'other'.
    bits : int
        Precision for overhead calculation (default 32 for FP32).

    Returns
    -------
    CrosswordResult
        Full decomposition with variance and compression metrics.
    """
    if W.ndim != 2:
        raise ValueError(f"Expected 2D matrix, got shape {W.shape}")

    m, n = W.shape

    # --- Decomposition ---
    mu = float(np.mean(W))
    row_means = np.mean(W, axis=1)
    col_means = np.mean(W, axis=0)

    r = row_means - mu  # row effects, sum to ~0
    c = col_means - mu  # col effects, sum to ~0

    # Residual: R_ij = W_ij - μ - r_i - c_j
    R = W - mu - r[:, np.newaxis] - c[np.newaxis, :]

    # --- Variance decomposition (orthogonal) ---
    var_total = float(np.var(W))

    # Var of row effect broadcast: each r_i repeated n times
    var_row = float(np.var(np.repeat(r, n)))
    # Var of col effect broadcast: c tiled m times
    var_col = float(np.var(np.tile(c, m)))
    var_residual = float(np.var(R))

    # ρ²
    if var_total > 0:
        var_explained = 1.0 - var_residual / var_total
    else:
        var_explained = 0.0

    # --- Compression metrics ---
    # Compression gain (Corollary 1): ½ log₂(Var(W)/Var(R))
    if var_residual > 0 and var_total > 0:
        compression_gain = 0.5 * np.log2(var_total / var_residual)
    else:
        compression_gain = 0.0

    # Overhead (Theorem 1): k(1 + m + n) / (mn)
    overhead = bits * (1 + m + n) / (m * n)

    # Estimate residual entropy under Gaussian assumption
    # h(R) = ½ log₂(2πe·Var(R))
    if var_residual > 0:
        h_residual = 0.5 * np.log2(2 * np.pi * np.e * var_residual)
    else:
        h_residual = 0.0

    crossword_rate = overhead + max(0.0, h_residual)

    return CrosswordResult(
        mu=mu,
        row_effects=r,
        col_effects=c,
        residual=R,
        m=m,
        n=n,
        var_total=var_total,
        var_row=var_row,
        var_col=var_col,
        var_residual=var_residual,
        var_explained=var_explained,
        compression_gain_bpp=compression_gain,
        overhead_bpp=overhead,
        crossword_rate_bpp=crossword_rate,
        layer_name=layer_name,
        layer_type=layer_type,
    )


def compare_quantization_error(W_original: np.ndarray,
                                W_quantized: np.ndarray,
                                layer_name: str = "",
                                quant_label: str = "") -> dict:
    """
    Analyze the structure of quantization error using crossword decomposition.

    The key question: does quantization error (W_f16 - W_quant) have
    row-column structure? If so, the error is "additive" and predictable.
    If not, the error is distributed and harder to correct.

    Parameters
    ----------
    W_original : np.ndarray
        Original (F16) weight matrix.
    W_quantized : np.ndarray
        Quantized weight matrix (same shape).
    layer_name : str
        Layer identifier.
    quant_label : str
        Quantization method label (e.g., 'Q4_K_M', 'Q2_K').

    Returns
    -------
    dict with:
        - error_matrix: W_original - W_quantized
        - error_decomposition: CrosswordResult of the error matrix
        - original_decomposition: CrosswordResult of W_original
        - relative_error: Frobenius norm ratio
        - error_var_explained: ρ² of the error matrix
    """
    if W_original.shape != W_quantized.shape:
        raise ValueError(
            f"Shape mismatch: {W_original.shape} vs {W_quantized.shape}")

    error = W_original - W_quantized

    # Decompose the error itself
    error_decomp = crossword_decompose(
        error,
        layer_name=f"{layer_name}_error_{quant_label}",
        layer_type="quantization_error"
    )

    # Decompose the original for comparison
    orig_decomp = crossword_decompose(
        W_original,
        layer_name=layer_name,
        layer_type="original"
    )

    # Relative Frobenius error
    orig_norm = np.linalg.norm(W_original, 'fro')
    if orig_norm > 0:
        relative_error = np.linalg.norm(error, 'fro') / orig_norm
    else:
        relative_error = 0.0

    return {
        "layer_name": layer_name,
        "quant_label": quant_label,
        "error_matrix": error,
        "error_decomposition": error_decomp,
        "original_decomposition": orig_decomp,
        "relative_error": relative_error,
        "error_var_explained": error_decomp.var_explained,
        "error_compression_gain": error_decomp.compression_gain_bpp,
        "original_var_explained": orig_decomp.var_explained,
        "error_frobenius": float(np.linalg.norm(error, 'fro')),
        "error_mean": float(np.mean(np.abs(error))),
        "error_max": float(np.max(np.abs(error))),
    }


def svd_variance_explained(W: np.ndarray, rank: int) -> Tuple[float, int]:
    """
    Compute variance explained by rank-k SVD for comparison with crossword.

    Uses scipy's sparse SVD (svds) for speed — only computes the top-k
    singular values instead of the full decomposition.

    Returns (variance_explained, n_params).
    """
    from scipy.sparse.linalg import svds

    m, n = W.shape
    k = min(rank, min(m, n) - 1)

    # svds computes only the top-k singular triplets — much faster
    U, s, Vt = svds(W.astype(np.float64), k=k)

    W_approx = U @ np.diag(s) @ Vt
    residual = W - W_approx

    var_total = np.var(W)
    var_residual = np.var(residual)

    if var_total > 0:
        ve = 1.0 - var_residual / var_total
    else:
        ve = 0.0

    n_params = k * (m + n + 1)

    return ve, n_params


# ── Experiment helpers (paper §3.3) ──────────────────────────────────────────

def hadamard_matrix(n: int) -> np.ndarray:
    """
    Construct a normalized Hadamard-like matrix of size n.

    Uses the Walsh-Hadamard construction, padding to the next power of 2
    and truncating back to n if needed.
    """
    from scipy.linalg import hadamard as scipy_hadamard

    # Pad to next power of 2
    k = 1
    while k < n:
        k *= 2

    H = scipy_hadamard(k).astype(np.float64) / np.sqrt(k)
    return H[:n, :n]


def crossword_after_hadamard(W: np.ndarray) -> CrosswordResult:
    """
    Experiment 1 (§3.3): Apply Hadamard rotation to columns, then
    recompute crossword decomposition.

    If the additive structure is a coordinate artifact, rotation destroys
    it. If it is a genuine property, rotation preserves it.
    """
    m, n = W.shape
    H = hadamard_matrix(n)
    W_rotated = W @ H
    return crossword_decompose(W_rotated, layer_name="hadamard_rotated")


def crossword_activation_weighted(W: np.ndarray) -> CrosswordResult:
    """
    Experiment 2 (§3.3): Scale each column of W by its L2 norm (proxy
    for activation magnitude), then recompute crossword decomposition.

    Tests whether additive structure is concentrated in the salient
    channels that AWQ identifies as important.
    """
    col_norms = np.linalg.norm(W, axis=0)
    # Avoid division by zero; columns with zero norm stay zero
    col_norms = np.where(col_norms > 0, col_norms, 1.0)
    W_weighted = W * col_norms[np.newaxis, :]
    return crossword_decompose(W_weighted, layer_name="activation_weighted")


def row_dominance(decomp: CrosswordResult) -> float:
    """
    Experiment 4 (§3.3): Row dominance ratio.

    Returns Var(row) / (Var(row) + Var(col)).
    A value near 0 means column-dominated; near 1 means row-dominated;
    near 0.5 means balanced (typically at the noise floor).
    """
    total_additive = decomp.var_row + decomp.var_col
    if total_additive > 0:
        return decomp.var_row / total_additive
    return 0.5  # undefined → balanced
