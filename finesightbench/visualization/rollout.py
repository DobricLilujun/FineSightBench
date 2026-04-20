"""Attention Rollout: aggregate attention across layers.

References
----------
- Abnar & Zuidema (2020) "Quantifying Attention Flow in Transformers"
- Chefer et al. (2021) "Generic Attention-model Explainability for
  Interpreting Bi-Modal and Encoder-Decoder Transformers"
"""

from __future__ import annotations

import numpy as np

from .direct import _to_2d


# ────────────────────────────────────────────────────────────────────────────
# Self-attention rollout (ViT encoder)
# ────────────────────────────────────────────────────────────────────────────

def attention_rollout(
    attn: np.ndarray,
    *,
    grid_h: int,
    grid_w: int,
    has_cls: bool = True,
    head_reduction: str = "mean",
    discard_ratio: float = 0.0,
    layers: list[int] | None = None,
    query_token: int = 0,
) -> np.ndarray:
    """Compute attention rollout for a self-attention ViT encoder.

    The algorithm models the residual connection by adding an identity
    matrix to each layer's (head-reduced) attention, re-normalising
    rows, and then multiplying across layers::

        Â_l = 0.5 * A_l + 0.5 * I          (residual shortcut)
        R   = Â_L · Â_{L-1} · … · Â_1

    Parameters
    ----------
    attn : ndarray, shape (num_layers, num_heads, seq_len, seq_len)
        Raw attention weights from every layer of the encoder.
    grid_h, grid_w : int
        Patch grid dimensions.
    has_cls : bool
        Whether token 0 is a CLS token.
    head_reduction : str
        ``"mean"`` | ``"max"`` | ``"min"`` across heads.
    discard_ratio : float, 0–1
        Fraction of lowest-attention values to zero out per layer
        (helps suppress noise).  0 = keep all.
    layers : list[int] | None
        Which layers to include.  ``None`` → all layers.
    query_token : int
        Row of the rollout matrix to extract (usually CLS = 0).

    Returns
    -------
    ndarray, shape (grid_h, grid_w), normalised to [0, 1].
    """
    num_layers, num_heads, seq_len, _ = attn.shape
    if layers is None:
        layers = list(range(num_layers))

    rollout = np.eye(seq_len, dtype=np.float64)

    for l_idx in layers:
        # ── reduce heads ───────────────────────────────────────────
        if head_reduction == "mean":
            a = attn[l_idx].mean(axis=0).astype(np.float64)
        elif head_reduction == "max":
            a = attn[l_idx].max(axis=0).astype(np.float64)
        elif head_reduction == "min":
            a = attn[l_idx].min(axis=0).astype(np.float64)
        else:
            raise ValueError(f"Unknown head_reduction '{head_reduction}'.")

        # ── optional discard ───────────────────────────────────────
        if discard_ratio > 0:
            flat = a.flatten()
            threshold = np.quantile(flat, discard_ratio)
            a = np.where(a < threshold, 0.0, a)
            # re-normalise rows
            row_sums = a.sum(axis=-1, keepdims=True)
            row_sums = np.where(row_sums == 0, 1.0, row_sums)
            a = a / row_sums

        # ── add identity (model residual) and normalise ────────────
        a = 0.5 * a + 0.5 * np.eye(seq_len, dtype=np.float64)
        row_sums = a.sum(axis=-1, keepdims=True)
        a = a / row_sums

        rollout = a @ rollout

    # extract the query-token row, strip CLS column
    vec = rollout[query_token]
    if has_cls:
        vec = vec[1:]
    return _to_2d(vec, grid_h, grid_w)


# ────────────────────────────────────────────────────────────────────────────
# Cross-attention rollout (encoder-decoder / multimodal)
# ────────────────────────────────────────────────────────────────────────────

def cross_attention_rollout(
    self_attn_visual: np.ndarray,
    cross_attn_layers: list[np.ndarray],
    self_attn_text: np.ndarray | None = None,
    *,
    grid_h: int,
    grid_w: int,
    text_token_indices: list[int] | int | None = None,
    has_cls_visual: bool = True,
    head_reduction: str = "mean",
    discard_ratio: float = 0.0,
) -> np.ndarray:
    """Compute cross-attention rollout for multimodal models.

    This implements a simplified version of the Chefer et al. (2021)
    bi-modal relevance propagation.  It works in three steps:

    1. Compute visual self-attention rollout  →  R_v  (V × V)
    2. Aggregate cross-attention layers       →  C    (T × V)
    3. Combine:  relevance = C · R_v   (T × V)

    If *self_attn_text* is provided, text self-attention rollout R_t
    is also computed and applied:  relevance = R_tᵀ · C · R_v

    Parameters
    ----------
    self_attn_visual : ndarray, shape (L_v, H, V, V)
        Visual encoder self-attention weights.
    cross_attn_layers : list of ndarray, each shape (H, T, V)
        Cross-attention weights per layer (text queries, visual keys).
    self_attn_text : ndarray or None, shape (L_t, H, T, T)
        Text decoder self-attention weights (optional).
    grid_h, grid_w : int
    text_token_indices : int | list[int] | None
        Text tokens to visualise.  ``None`` → average over all.
    has_cls_visual : bool
    head_reduction : str
    discard_ratio : float

    Returns
    -------
    ndarray, shape (grid_h, grid_w), normalised to [0, 1].
    """
    if isinstance(text_token_indices, int):
        text_token_indices = [text_token_indices]

    # ── 1. visual self-attention rollout ───────────────────────────
    L_v, H_v, V, _ = self_attn_visual.shape
    R_v = np.eye(V, dtype=np.float64)

    for l_idx in range(L_v):
        a = _reduce_heads(self_attn_visual[l_idx], head_reduction)
        a = _apply_discard(a, discard_ratio)
        a = 0.5 * a + 0.5 * np.eye(V, dtype=np.float64)
        a = a / a.sum(axis=-1, keepdims=True)
        R_v = a @ R_v  # (V, V)

    # ── 2. aggregate cross-attention layers ────────────────────────
    T = cross_attn_layers[0].shape[1]
    C = np.zeros((T, V), dtype=np.float64)
    for ca in cross_attn_layers:
        ca_reduced = _reduce_heads(ca, head_reduction)  # (T, V)
        C += ca_reduced
    C /= len(cross_attn_layers)

    # ── 3. combine ─────────────────────────────────────────────────
    relevance = C @ R_v  # (T, V)

    # ── 4. optional text rollout ───────────────────────────────────
    if self_attn_text is not None:
        L_t, H_t, T2, _ = self_attn_text.shape
        R_t = np.eye(T2, dtype=np.float64)
        for l_idx in range(L_t):
            a = _reduce_heads(self_attn_text[l_idx], head_reduction)
            a = _apply_discard(a, discard_ratio)
            a = 0.5 * a + 0.5 * np.eye(T2, dtype=np.float64)
            a = a / a.sum(axis=-1, keepdims=True)
            R_t = a @ R_t
        relevance = R_t.T @ relevance  # (T, V)

    # ── 5. select text tokens & reduce to spatial map ──────────────
    if text_token_indices is not None:
        vec = relevance[text_token_indices].mean(axis=0)
    else:
        vec = relevance.mean(axis=0)

    if has_cls_visual:
        vec = vec[1:]

    return _to_2d(vec, grid_h, grid_w)


# ────────────────────────────────────────────────────────────────────────────
# internal helpers
# ────────────────────────────────────────────────────────────────────────────

def _reduce_heads(a: np.ndarray, reduction: str) -> np.ndarray:
    """Reduce the head dimension (axis 0)."""
    if reduction == "mean":
        return a.mean(axis=0).astype(np.float64)
    if reduction == "max":
        return a.max(axis=0).astype(np.float64)
    if reduction == "min":
        return a.min(axis=0).astype(np.float64)
    raise ValueError(f"Unknown reduction '{reduction}'.")


def _apply_discard(a: np.ndarray, discard_ratio: float) -> np.ndarray:
    """Zero out the lowest *discard_ratio* fraction of attention values."""
    if discard_ratio <= 0:
        return a
    flat = a.flatten()
    threshold = np.quantile(flat, discard_ratio)
    a = np.where(a < threshold, 0.0, a)
    row_sums = a.sum(axis=-1, keepdims=True)
    row_sums = np.where(row_sums == 0, 1.0, row_sums)
    return a / row_sums
