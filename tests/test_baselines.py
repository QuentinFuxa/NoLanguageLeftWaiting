"""Tests for baseline backends (unit tests, no GPU required).

These test the policy logic without requiring llama.cpp.
The inner AlignAtt backend is not instantiated in these tests.
"""

import pytest
from nllw.backend_protocol import BackendConfig, TranslationStep


class TestWaitKPolicy:
    """Test wait-k policy logic (without actual translation)."""

    def test_config_defaults(self):
        """BackendConfig has sensible defaults."""
        config = BackendConfig(backend_type="wait-k")
        assert config.backend_type == "wait-k"
        assert config.word_batch == 3
        assert config.border_distance == 3

    def test_config_from_dict(self):
        """BackendConfig.from_dict ignores unknown keys."""
        d = {"backend_type": "wait-k", "wait_k": 5, "unknown_param": 42}
        config = BackendConfig.from_dict(d)
        assert config.backend_type == "wait-k"

    def test_translation_step_dataclass(self):
        step = TranslationStep(text="hello", is_final=True, committed_tokens=3)
        assert step.text == "hello"
        assert step.is_final is True
        assert step.committed_tokens == 3
        assert step.stopped_at_border is False


class TestFixedRatePolicy:
    """Test fixed-rate policy config."""

    def test_config(self):
        config = BackendConfig(backend_type="fixed-rate", word_batch=4)
        assert config.backend_type == "fixed-rate"
        assert config.word_batch == 4


class TestBackendRegistry:
    """Test that baselines register correctly."""

    def test_baselines_registered(self):
        """Import baselines and check they're registered."""
        # Import triggers registration
        import nllw.baselines  # noqa: F401
        from nllw.backend_protocol import list_backends
        backends = list_backends()
        assert "wait-k" in backends
        assert "fixed-rate" in backends

    def test_alignatt_backends_registered(self):
        """AlignAtt backends register correctly."""
        import nllw.alignatt_backend  # noqa: F401
        from nllw.backend_protocol import list_backends
        backends = list_backends()
        assert "alignatt" in backends
        assert "full-sentence" in backends
        assert "eager" in backends
