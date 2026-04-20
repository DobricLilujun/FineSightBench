"""Direct attention visualization: single-head, multi-head, multi-layer, text-conditioned.

All functions accept raw NumPy attention tensors and return 2-D heatmaps
(float64, values in [0, 1]) that can be passed to :func:`overlay.overlay_heatmap`.
"""

from __future__ import annotations

import numpy as np


# ────────────────────────────────────────────────────────────────────────────
# helpers
# ────────────────────────────────────────────────────────────────────────────

def _to_2d(attn_vec: np.ndarray, grid_h: int, grid_w: int) -> np.ndarray:
    """Reshape a 1-D attention vector over patches into a 2-D grid.

    Parameters
    ----------
    attn_vec : ndarray, shape (num_patches,)
        Attention weights for each spatial patch (CLS / special tokens
        should already be stripped).
    grid_h, grid_w : int
        Spatial dimensions of the patch grid (e.g. 14×14 for ViT-B/16
        with 224 px input).

    Returns
    -------
    ndarray, shape (grid_h, grid_w), dtype float64, values in [0, 1].
    """
    if attn_vec.size != grid_h * grid_w:
        raise ValueError(
            f"attn_vec length {attn_vec.size} != grid_h*grid_w "
            f"({grid_h}*{grid_w}={grid_h * grid_w})"
        )
    hmap = attn_vec.reshape(grid_h, grid_w).astype(np.float64)
    vmin, vmax = hmap.min(), hmap.max()
    if vmax - vmin > 1e-8:
        hmap = (hmap - vmin) / (vmax - vmin)
    else:
        hmap = np.zeros_like(hmap)
    return hmap


def _strip_cls(attn_matrix: np.ndarray, has_cls: bool) -> np.ndarray:
    """Remove the CLS column (index 0) from the last axis if present."""
    if has_cls:
        return attn_matrix[..., 1:]
    return attn_matrix


# ────────────────────────────────────────────────────────────────────────────
# public API
# ────────────────────────────────────────────────────────────────────────────

def single_head_attention_map(
    attn: np.ndarray,
    layer: int,
    head: int,
    query_token: int,
    grid_h: int,
    grid_w: int,
    *,
    has_cls: bool = True,
) -> np.ndarray:
    """Extract the attention map of one head at one layer.

    Parameters
    ----------
    attn : ndarray, shape (num_layers, num_heads, seq_len, seq_len)
        Raw attention weights from the model.
    layer : int
        Layer index (0-based).
    head : int
        Head index (0-based).
    query_token : int
        Index of the query token whose attention to visualise.
        For CLS-based pooling this is typically 0.
    grid_h, grid_w : int
        Patch grid dimensions.
    has_cls : bool
        Whether the first token in seq_len is a CLS token that should
        be excluded from the spatial map.

    Returns
    -------
    ndarray, shape (grid_h, grid_w), normalised to [0, 1].
    """
    # attn[layer, head, query_token, :] → 1-D over key tokens
    vec = attn[layer, head, query_token, :]
    vec = _strip_cls(vec, has_cls)
    return _to_2d(vec, grid_h, grid_w)


def multi_head_average_map(
    attn: np.ndarray,
    layer: int,
    query_token: int,
    grid_h: int,
    grid_w: int,
    *,
    has_cls: bool = True,
    reduction: str = "mean",
) -> np.ndarray:
    """Average (or max/min) across all heads at a given layer.

    Parameters
    ----------
    attn : ndarray, shape (num_layers, num_heads, seq_len, seq_len)
    layer : int
    query_token : int
    grid_h, grid_w : int
    has_cls : bool
    reduction : str
        ``"mean"`` (default), ``"max"``, or ``"min"``.

    Returns
    -------
    ndarray, shape (grid_h, grid_w), normalised to [0, 1].
    """
    # shape: (num_heads, seq_len)
    vecs = attn[layer, :, query_token, :]
    vecs = _strip_cls(vecs, has_cls)  # (num_heads, num_patches)
    if reduction == "mean":
        agg = vecs.mean(axis=0)
    elif reduction == "max":
        agg = vecs.max(axis=0)
    elif reduction == "min":
        agg = vecs.min(axis=0)
    else:
        raise ValueError(f"Unknown reduction '{reduction}', use mean/max/min.")
    return _to_2d(agg, grid_h, grid_w)


def multi_layer_aggregated_map(
    attn: np.ndarray,
    layers: list[int] | None,
    query_token: int,
    grid_h: int,
    grid_w: int,
    *,
    has_cls: bool = True,
    head_reduction: str = "mean",
    layer_reduction: str = "mean",
) -> np.ndarray:
    """Aggregate attention across multiple layers.

    For each selected layer the heads are first reduced (via
    *head_reduction*), then the resulting per-layer maps are reduced
    via *layer_reduction*.

    Parameters
    ----------
    attn : ndarray, shape (num_layers, num_heads, seq_len, seq_len)
    layers : list[int] or None
        Layer indices to include. ``None`` → all layers.
    query_token : int
    grid_h, grid_w : int
    has_cls : bool
    head_reduction : str
        Reduction across heads (``"mean"``/``"max"``/``"min"``).
    layer_reduction : str
        Reduction across layers (``"mean"``/``"max"``/``"min"``).

    Returns
    -------
    ndarray, shape (grid_h, grid_w), normalised to [0, 1].
    """
    num_layers = attn.shape[0]
    if layers is None:
        layers = list(range(num_layers))

    maps = []
    for l_idx in layers:
        m = multi_head_average_map(
            attn, l_idx, query_token, grid_h, grid_w,
            has_cls=has_cls, reduction=head_reduction,
        )
        maps.append(m)

    stack = np.stack(maps, axis=0)  # (L, grid_h, grid_w)
    if layer_reduction == "mean":
        agg = stack.mean(axis=0)
    elif layer_reduction == "max":
        agg = stack.max(axis=0)
    elif layer_reduction == "min":
        agg = stack.min(axis=0)
    else:
        raise ValueError(f"Unknown layer_reduction '{layer_reduction}'.")

    # re-normalise
    vmin, vmax = agg.min(), agg.max()
    if vmax - vmin > 1e-8:
        agg = (agg - vmin) / (vmax - vmin)
    else:
        agg = np.zeros_like(agg)
    return agg


def text_conditioned_attention_map(
    cross_attn: np.ndarray,
    text_token_indices: list[int] | int,
    grid_h: int,
    grid_w: int,
    *,
    has_cls: bool = False,
    head_reduction: str = "mean",
    token_reduction: str = "mean",
) -> np.ndarray:
    """Visualise cross-attention from specific text tokens to visual patches.

    Parameters
    ----------
    cross_attn : ndarray, shape (num_heads, num_text_tokens, num_visual_tokens)
        Cross-attention weights where text tokens attend to visual tokens.
        If your model stores the transpose, pass
        ``cross_attn.transpose(0, 2, 1)`` instead.
    text_token_indices : int or list[int]
        Which text-token positions to inspect (e.g. the token for "red dot").
    grid_h, grid_w : int
    has_cls : bool
        Whether the first visual token is a CLS token to discard.
    head_reduction : str
    token_reduction : str
        Reduction across the selected text tokens.

    Returns
    -------
    ndarray, shape (grid_h, grid_w), normalised to [0, 1].
    """
    if isinstance(text_token_indices, int):
        text_token_indices = [text_token_indices]

    # cross_attn: (H, T, V)
    selected = cross_attn[:, text_token_indices, :]  # (H, len(indices), V)
    selected = _strip_cls(selected, has_cls)  # strip visual CLS if present

    # reduce heads
    if head_reduction == "mean":
        h_agg = selected.mean(axis=0)  # (len(indices), V')
    elif head_reduction == "max":
        h_agg = selected.max(axis=0)
    else:
        raise ValueError(f"Unknown head_reduction '{head_reduction}'.")

    # reduce tokens
    if token_reduction == "mean":
        t_agg = h_agg.mean(axis=0)  # (V',)
    elif token_reduction == "max":
        t_agg = h_agg.max(axis=0)
    else:
        raise ValueError(f"Unknown token_reduction '{token_reduction}'.")

    return _to_2d(t_agg, grid_h, grid_w)
