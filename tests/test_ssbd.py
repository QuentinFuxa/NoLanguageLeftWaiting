"""Tests for SSBD (Self-Speculative Biased Decoding) and NE (Normalized Erasure).

Tests the SSBD acceptance logic, NE metric computation, and integration
with the LA backend, without requiring llama.cpp.
"""

import math
import numpy as np
import pytest

from nllw.alignatt_la_backend import (
    ssbd_accept,
    _longest_common_prefix_tokens,
)
from nllw.backend_protocol import BackendConfig
from nllw.metrics import (
    compute_normalized_erasure,
    compute_normalized_erasure_text,
)


# ---------------------------------------------------------------------------
# SSBD acceptance tests
# ---------------------------------------------------------------------------

class TestSSBDAccept:
    """Test biased speculative decoding acceptance criterion."""

    def _make_logits(self, values: dict, n_vocab: int = 10) -> np.ndarray:
        """Create a logit array with specific tokens having specific values."""
        logits = np.full(n_vocab, -10.0, dtype=np.float32)
        for tok_id, val in values.items():
            logits[tok_id] = val
        return logits

    def test_draft_is_argmax(self):
        """Draft token IS the greedy argmax -> always accept."""
        logits = self._make_logits({3: 5.0, 7: 2.0})
        assert ssbd_accept(logits, draft_token=3, beta=0.0) is True
        assert ssbd_accept(logits, draft_token=3, beta=0.2) is True
        assert ssbd_accept(logits, draft_token=3, beta=0.5) is True

    def test_draft_not_argmax_no_bias(self):
        """Draft NOT argmax + beta=0 -> always reject."""
        logits = self._make_logits({3: 5.0, 7: 4.9})
        assert ssbd_accept(logits, draft_token=7, beta=0.0) is False

    def test_draft_close_with_bias(self):
        """Draft close to argmax + beta=0.2 -> accept due to bias.

        With beta=0.2, threshold = 0.2/0.8 = 0.25
        Accept if P(draft)/P(argmax) >= 1 - 0.25 = 0.75
        """
        # logits: tok 3 = 5.0, tok 7 = 4.7
        # P(7)/P(3) = exp(4.7 - 5.0) = exp(-0.3) ~ 0.74
        # With beta=0.2, threshold is 0.25, so need ratio >= 0.75
        # 0.74 < 0.75 -> reject
        logits = self._make_logits({3: 5.0, 7: 4.7})
        assert ssbd_accept(logits, draft_token=7, beta=0.2) is False

        # logits: tok 7 = 4.8
        # P(7)/P(3) = exp(4.8 - 5.0) = exp(-0.2) ~ 0.819
        # 0.819 >= 0.75 -> accept
        logits2 = self._make_logits({3: 5.0, 7: 4.8})
        assert ssbd_accept(logits2, draft_token=7, beta=0.2) is True

    def test_high_beta_more_lenient(self):
        """Higher beta -> more lenient acceptance."""
        logits = self._make_logits({3: 5.0, 7: 3.0})
        # P(7)/P(3) = exp(-2.0) ~ 0.135
        # beta=0.2: threshold 0.25, need >= 0.75 -> reject
        assert ssbd_accept(logits, draft_token=7, beta=0.2) is False
        # beta=0.5: threshold 1.0, need >= 0.0 -> accept everything
        assert ssbd_accept(logits, draft_token=7, beta=0.5) is True

    def test_beta_zero_is_pure_speculative(self):
        """beta=0 is pure speculative: only accept exact argmax match."""
        logits = self._make_logits({3: 5.0, 7: 4.99})
        assert ssbd_accept(logits, draft_token=7, beta=0.0) is False
        assert ssbd_accept(logits, draft_token=3, beta=0.0) is True

    def test_equal_logits(self):
        """Tied logits: draft is one of the argmax candidates."""
        logits = self._make_logits({3: 5.0, 7: 5.0})
        # np.argmax returns first index with max -> token 3
        # So draft=7 is NOT the argmax
        # But with any positive beta, the ratio is 1.0 which is >= threshold
        assert ssbd_accept(logits, draft_token=7, beta=0.0) is False
        assert ssbd_accept(logits, draft_token=7, beta=0.01) is True

    def test_very_negative_draft(self):
        """Draft with very low probability -> always rejected."""
        logits = self._make_logits({3: 10.0, 7: -5.0})
        # P(7)/P(3) = exp(-15) ~ 3e-7 -> way below any threshold
        assert ssbd_accept(logits, draft_token=7, beta=0.2) is False


class TestSSBDConfig:
    """Test SSBD configuration."""

    def test_default_disabled(self):
        config = BackendConfig()
        assert config.ssbd_beta is None

    def test_from_dict(self):
        config = BackendConfig.from_dict({"ssbd_beta": 0.2})
        assert config.ssbd_beta == 0.2

    def test_sweep_shortname(self):
        """SSBD can be swept via 'ssbd' shortname in bench."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("ssbd=0.0,0.1,0.2")
        assert "ssbd_beta" in grid
        assert grid["ssbd_beta"] == [0.0, 0.1, 0.2]


# ---------------------------------------------------------------------------
# Normalized Erasure (NE) tests
# ---------------------------------------------------------------------------

class TestNormalizedErasure:
    """Test NE metric for output stability."""

    def test_no_revisions(self):
        """Single translation -> NE = 0."""
        assert compute_normalized_erasure([[1, 2, 3]]) == 0.0

    def test_empty_history(self):
        assert compute_normalized_erasure([]) == 0.0

    def test_perfectly_stable(self):
        """Same translation every time -> NE = 0."""
        history = [[1, 2, 3], [1, 2, 3], [1, 2, 3]]
        assert compute_normalized_erasure(history) == 0.0

    def test_total_revision(self):
        """Completely different translations -> high NE."""
        history = [
            [1, 2, 3],  # 3 tokens
            [4, 5, 6],  # no common prefix -> erasure = 3
            [7, 8, 9],  # no common prefix -> erasure = 3
        ]
        ne = compute_normalized_erasure(history)
        # NE = (3 + 3) / 2 = 3.0
        assert ne == pytest.approx(3.0)

    def test_partial_revision(self):
        """Partial revision: first 2 tokens stable, last one changes."""
        history = [
            [1, 2, 3],    # step 1
            [1, 2, 4],    # LCP = [1, 2], erasure = 3 - 2 = 1
            [1, 2, 4, 5], # LCP = [1, 2, 4], erasure = 3 - 3 = 0
        ]
        ne = compute_normalized_erasure(history)
        # NE = (1 + 0) / 2 = 0.5
        assert ne == pytest.approx(0.5)

    def test_growing_stable(self):
        """Translation grows but prefix is always stable."""
        history = [
            [1, 2],
            [1, 2, 3],
            [1, 2, 3, 4],
        ]
        ne = compute_normalized_erasure(history)
        # LCP(step1, step2) = [1,2] len=2, erasure = 2 - 2 = 0
        # LCP(step2, step3) = [1,2,3] len=3, erasure = 3 - 3 = 0
        # NE = (0 + 0) / 2 = 0.0
        assert ne == pytest.approx(0.0)

    def test_threshold_low_revision(self):
        """NE < 0.2 is considered low revision."""
        # 10 steps, one small revision
        history = [[1, 2, 3, 4, 5]] * 10
        # Change one token in step 5
        history[5] = [1, 2, 3, 4, 6]  # erasure = 1
        history[6] = [1, 2, 3, 4, 5]  # erasure = 1 (reverts)
        ne = compute_normalized_erasure(history)
        # NE = 2 / 9 = 0.222...
        assert ne < 0.3

    def test_shrinking_translation(self):
        """Translation gets shorter -> measures erased tokens."""
        history = [
            [1, 2, 3, 4, 5],
            [1, 2, 3],  # LCP = 3, erasure = 5 - 3 = 2
        ]
        ne = compute_normalized_erasure(history)
        assert ne == pytest.approx(2.0)


class TestNormalizedErasureText:
    """Test word-level NE metric."""

    def test_stable(self):
        history = ["the cat sat", "the cat sat", "the cat sat"]
        assert compute_normalized_erasure_text(history) == 0.0

    def test_revision(self):
        history = ["the cat sat", "the dog sat"]
        # LCP = "the", len = 1, erasure = 3 - 1 = 2
        ne = compute_normalized_erasure_text(history)
        assert ne == pytest.approx(2.0)

    def test_empty(self):
        assert compute_normalized_erasure_text([]) == 0.0
        assert compute_normalized_erasure_text(["hello"]) == 0.0

    def test_growing(self):
        history = ["the", "the cat", "the cat sat"]
        ne = compute_normalized_erasure_text(history)
        # Both transitions: LCP covers all of previous -> erasure = 0
        assert ne == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Integration: LA backend state tests (without llama.cpp)
# ---------------------------------------------------------------------------

class TestLARevisionHistory:
    """Test that LA backend tracks revision history correctly."""

    def test_config_ssbd_roundtrip(self):
        """SSBD config survives from_dict -> field access."""
        config = BackendConfig.from_dict({
            "backend_type": "alignatt-la",
            "ssbd_beta": 0.2,
            "border_distance": 3,
        })
        assert config.ssbd_beta == 0.2
        assert config.border_distance == 3

    def test_ssbd_disabled_by_default(self):
        """SSBD is disabled when ssbd_beta is None."""
        config = BackendConfig(backend_type="alignatt-la")
        assert config.ssbd_beta is None


class TestDisplayMaskK:
    """Test display-only mask-k configuration."""

    def test_default_zero(self):
        config = BackendConfig()
        assert config.display_mask_k == 0

    def test_from_dict(self):
        config = BackendConfig.from_dict({"display_mask_k": 3})
        assert config.display_mask_k == 3

    def test_sweep_shortname(self):
        """display_mask_k can be swept via 'mask' shortname."""
        from nllw.bench import parse_sweep_spec
        grid = parse_sweep_spec("mask=0,1,2,3")
        assert "display_mask_k" in grid
        assert grid["display_mask_k"] == [0, 1, 2, 3]

    def test_combined_with_ssbd(self):
        """mask-k and SSBD can be used together."""
        config = BackendConfig.from_dict({
            "ssbd_beta": 0.2,
            "display_mask_k": 3,
        })
        assert config.ssbd_beta == 0.2
        assert config.display_mask_k == 3


class TestSSBDAcceptBoundaries:
    """Edge cases for SSBD acceptance."""

    def test_single_vocab_logits(self):
        """Edge case: single token vocabulary."""
        logits = np.array([5.0], dtype=np.float32)
        assert ssbd_accept(logits, draft_token=0, beta=0.0) is True

    def test_all_equal_logits(self):
        """All logits equal -> draft is accepted with any positive beta."""
        logits = np.full(100, 1.0, dtype=np.float32)
        # argmax returns 0 (first), so draft=50 is not argmax
        assert ssbd_accept(logits, draft_token=50, beta=0.0) is False
        # With any bias, ratio = 1.0 >= threshold -> accept
        assert ssbd_accept(logits, draft_token=50, beta=0.01) is True

    def test_large_logit_gap(self):
        """Large gap between argmax and draft -> never accept."""
        logits = np.full(100, -100.0, dtype=np.float32)
        logits[0] = 100.0  # argmax
        logits[50] = -100.0  # draft
        assert ssbd_accept(logits, draft_token=50, beta=0.2) is False
        assert ssbd_accept(logits, draft_token=50, beta=0.49) is False
