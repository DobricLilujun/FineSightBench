"""Tests for finesightbench.visualization."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from finesightbench.visualization.direct import (
    single_head_attention_map,
    multi_head_average_map,
    multi_layer_aggregated_map,
    text_conditioned_attention_map,
)
from finesightbench.visualization.rollout import (
    attention_rollout,
    cross_attention_rollout,
)
from finesightbench.visualization.overlay import (
    overlay_heatmap,
    save_visualization,
    side_by_side,
)


# ────────────────────────────────────────────────────────────────────────────
# fixtures
# ────────────────────────────────────────────────────────────────────────────

GRID_H, GRID_W = 14, 14
NUM_PATCHES = GRID_H * GRID_W
SEQ_LEN = 1 + NUM_PATCHES  # CLS + patches
NUM_HEADS = 8
NUM_LAYERS = 6


def _random_attn(
    num_layers: int = NUM_LAYERS,
    num_heads: int = NUM_HEADS,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Create random but valid attention weights (rows sum to 1)."""
    rng = np.random.default_rng(42)
    raw = rng.random((num_layers, num_heads, seq_len, seq_len))
    # softmax-like normalisation along last axis
    raw = raw / raw.sum(axis=-1, keepdims=True)
    return raw


def _random_cross_attn(
    num_heads: int = NUM_HEADS,
    num_text: int = 10,
    num_visual: int = SEQ_LEN,
) -> np.ndarray:
    rng = np.random.default_rng(123)
    raw = rng.random((num_heads, num_text, num_visual))
    raw = raw / raw.sum(axis=-1, keepdims=True)
    return raw


@pytest.fixture
def attn():
    return _random_attn()


@pytest.fixture
def cross_attn():
    return _random_cross_attn()


@pytest.fixture
def sample_image():
    return Image.new("RGB", (224, 224), (100, 150, 200))


# ────────────────────────────────────────────────────────────────────────────
# direct.py tests
# ────────────────────────────────────────────────────────────────────────────

class TestSingleHeadAttentionMap:
    def test_shape_and_range(self, attn):
        hmap = single_head_attention_map(
            attn, layer=0, head=0, query_token=0,
            grid_h=GRID_H, grid_w=GRID_W,
        )
        assert hmap.shape == (GRID_H, GRID_W)
        assert hmap.min() >= 0.0
        assert hmap.max() <= 1.0

    def test_different_heads_differ(self, attn):
        h0 = single_head_attention_map(
            attn, layer=0, head=0, query_token=0,
            grid_h=GRID_H, grid_w=GRID_W,
        )
        h1 = single_head_attention_map(
            attn, layer=0, head=1, query_token=0,
            grid_h=GRID_H, grid_w=GRID_W,
        )
        assert not np.allclose(h0, h1)

    def test_no_cls(self, attn):
        # Without CLS, seq_len == num_patches
        a = _random_attn(seq_len=NUM_PATCHES)
        hmap = single_head_attention_map(
            a, layer=0, head=0, query_token=0,
            grid_h=GRID_H, grid_w=GRID_W, has_cls=False,
        )
        assert hmap.shape == (GRID_H, GRID_W)

    def test_size_mismatch_raises(self, attn):
        with pytest.raises(ValueError, match="attn_vec length"):
            single_head_attention_map(
                attn, layer=0, head=0, query_token=0,
                grid_h=10, grid_w=10,
            )


class TestMultiHeadAverageMap:
    @pytest.mark.parametrize("reduction", ["mean", "max", "min"])
    def test_shape_and_range(self, attn, reduction):
        hmap = multi_head_average_map(
            attn, layer=0, query_token=0,
            grid_h=GRID_H, grid_w=GRID_W,
            reduction=reduction,
        )
        assert hmap.shape == (GRID_H, GRID_W)
        assert hmap.min() >= 0.0
        assert hmap.max() <= 1.0

    def test_invalid_reduction(self, attn):
        with pytest.raises(ValueError, match="Unknown reduction"):
            multi_head_average_map(
                attn, layer=0, query_token=0,
                grid_h=GRID_H, grid_w=GRID_W,
                reduction="banana",
            )


class TestMultiLayerAggregatedMap:
    def test_all_layers(self, attn):
        hmap = multi_layer_aggregated_map(
            attn, layers=None, query_token=0,
            grid_h=GRID_H, grid_w=GRID_W,
        )
        assert hmap.shape == (GRID_H, GRID_W)

    def test_subset_layers(self, attn):
        hmap = multi_layer_aggregated_map(
            attn, layers=[3, 4, 5], query_token=0,
            grid_h=GRID_H, grid_w=GRID_W,
        )
        assert hmap.shape == (GRID_H, GRID_W)
        assert hmap.max() <= 1.0

    @pytest.mark.parametrize("lr", ["mean", "max", "min"])
    def test_layer_reductions(self, attn, lr):
        hmap = multi_layer_aggregated_map(
            attn, layers=[0, 1], query_token=0,
            grid_h=GRID_H, grid_w=GRID_W,
            layer_reduction=lr,
        )
        assert hmap.shape == (GRID_H, GRID_W)


class TestTextConditionedAttentionMap:
    def test_single_token(self, cross_attn):
        hmap = text_conditioned_attention_map(
            cross_attn, text_token_indices=3,
            grid_h=GRID_H, grid_w=GRID_W,
            has_cls=True,
        )
        assert hmap.shape == (GRID_H, GRID_W)

    def test_multiple_tokens(self, cross_attn):
        hmap = text_conditioned_attention_map(
            cross_attn, text_token_indices=[2, 3, 4],
            grid_h=GRID_H, grid_w=GRID_W,
            has_cls=True,
        )
        assert hmap.shape == (GRID_H, GRID_W)
        assert hmap.min() >= 0.0
        assert hmap.max() <= 1.0


# ────────────────────────────────────────────────────────────────────────────
# rollout.py tests
# ────────────────────────────────────────────────────────────────────────────

class TestAttentionRollout:
    def test_shape_and_range(self, attn):
        hmap = attention_rollout(
            attn, grid_h=GRID_H, grid_w=GRID_W,
        )
        assert hmap.shape == (GRID_H, GRID_W)
        assert hmap.min() >= 0.0
        assert hmap.max() <= 1.0

    def test_discard_ratio(self, attn):
        h0 = attention_rollout(
            attn, grid_h=GRID_H, grid_w=GRID_W,
            discard_ratio=0.0,
        )
        h9 = attention_rollout(
            attn, grid_h=GRID_H, grid_w=GRID_W,
            discard_ratio=0.9,
        )
        # should produce different results
        assert not np.allclose(h0, h9)

    def test_subset_layers(self, attn):
        hmap = attention_rollout(
            attn, grid_h=GRID_H, grid_w=GRID_W,
            layers=[3, 4, 5],
        )
        assert hmap.shape == (GRID_H, GRID_W)

    @pytest.mark.parametrize("reduction", ["mean", "max", "min"])
    def test_head_reductions(self, attn, reduction):
        hmap = attention_rollout(
            attn, grid_h=GRID_H, grid_w=GRID_W,
            head_reduction=reduction,
        )
        assert hmap.shape == (GRID_H, GRID_W)

    def test_no_cls(self):
        a = _random_attn(seq_len=NUM_PATCHES)
        hmap = attention_rollout(
            a, grid_h=GRID_H, grid_w=GRID_W,
            has_cls=False, query_token=0,
        )
        assert hmap.shape == (GRID_H, GRID_W)


class TestCrossAttentionRollout:
    def test_basic(self):
        V = SEQ_LEN  # CLS + patches
        T = 10
        self_attn_vis = _random_attn(num_layers=4, num_heads=8, seq_len=V)
        cross_layers = [
            _random_cross_attn(num_heads=8, num_text=T, num_visual=V)
            for _ in range(3)
        ]
        hmap = cross_attention_rollout(
            self_attn_vis, cross_layers,
            grid_h=GRID_H, grid_w=GRID_W,
        )
        assert hmap.shape == (GRID_H, GRID_W)
        assert hmap.min() >= 0.0
        assert hmap.max() <= 1.0

    def test_with_text_rollout(self):
        V = SEQ_LEN
        T = 10
        self_attn_vis = _random_attn(num_layers=4, num_heads=8, seq_len=V)
        self_attn_txt = _random_attn(num_layers=3, num_heads=8, seq_len=T)
        cross_layers = [
            _random_cross_attn(num_heads=8, num_text=T, num_visual=V)
            for _ in range(3)
        ]
        hmap = cross_attention_rollout(
            self_attn_vis, cross_layers,
            self_attn_text=self_attn_txt,
            grid_h=GRID_H, grid_w=GRID_W,
            text_token_indices=[0, 1, 2],
        )
        assert hmap.shape == (GRID_H, GRID_W)

    def test_single_text_token(self):
        V = SEQ_LEN
        T = 10
        self_attn_vis = _random_attn(num_layers=2, num_heads=4, seq_len=V)
        cross_layers = [
            _random_cross_attn(num_heads=4, num_text=T, num_visual=V)
        ]
        hmap = cross_attention_rollout(
            self_attn_vis, cross_layers,
            grid_h=GRID_H, grid_w=GRID_W,
            text_token_indices=5,
        )
        assert hmap.shape == (GRID_H, GRID_W)


# ────────────────────────────────────────────────────────────────────────────
# overlay.py tests
# ────────────────────────────────────────────────────────────────────────────

class TestOverlayHeatmap:
    def test_basic(self, sample_image):
        hmap = np.random.default_rng(0).random((14, 14))
        result = overlay_heatmap(sample_image, hmap)
        assert isinstance(result, Image.Image)
        assert result.size == sample_image.size
        assert result.mode == "RGB"

    @pytest.mark.parametrize("cmap", ["jet", "viridis", "turbo", "hot"])
    def test_colormaps(self, sample_image, cmap):
        hmap = np.random.default_rng(0).random((14, 14))
        result = overlay_heatmap(sample_image, hmap, colormap=cmap)
        assert result.size == sample_image.size

    def test_alpha_zero_is_original(self, sample_image):
        hmap = np.ones((14, 14))
        result = overlay_heatmap(sample_image, hmap, alpha=0.0)
        # alpha=0 means 100% original image
        np.testing.assert_array_equal(
            np.asarray(result), np.asarray(sample_image)
        )


class TestSideBySide:
    def test_creates_figure(self, sample_image):
        hmaps = {
            "map_a": np.random.default_rng(0).random((14, 14)),
            "map_b": np.random.default_rng(1).random((14, 14)),
        }
        fig = side_by_side(sample_image, hmaps)
        import matplotlib.pyplot as _plt
        assert isinstance(fig, _plt.Figure)
        assert len(fig.axes) == 3  # original + 2 heatmaps
        _plt.close(fig)


class TestSaveVisualization:
    def test_saves_file(self, sample_image, tmp_path):
        hmap = np.random.default_rng(0).random((14, 14))
        out = save_visualization(
            sample_image, hmap, tmp_path / "out.png",
        )
        assert out.exists()
        saved = Image.open(out)
        assert saved.size == sample_image.size


# ────────────────────────────────────────────────────────────────────────────
# integration: end-to-end pipeline
# ────────────────────────────────────────────────────────────────────────────

class TestEndToEnd:
    """Verify that the full pipeline (generate attention → heatmap → overlay)
    works without errors."""

    def test_single_head_overlay(self, sample_image):
        attn = _random_attn()
        hmap = single_head_attention_map(
            attn, layer=2, head=3, query_token=0,
            grid_h=GRID_H, grid_w=GRID_W,
        )
        result = overlay_heatmap(sample_image, hmap)
        assert result.size == (224, 224)

    def test_rollout_overlay(self, sample_image):
        attn = _random_attn()
        hmap = attention_rollout(
            attn, grid_h=GRID_H, grid_w=GRID_W,
            discard_ratio=0.9,
        )
        result = overlay_heatmap(sample_image, hmap, colormap="turbo")
        assert result.size == (224, 224)
