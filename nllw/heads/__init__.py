"""
Alignment head detection and configuration.

Detects which attention heads in a decoder-only LLM perform token-level
source-target alignment during translation. These heads are used by AlignAtt
to decide when to stop generating (border detection).

Pre-computed head configs are stored in heads/configs/ as JSON files.
"""

from nllw.heads.detect import detect_alignment_heads, HeadDetectionConfig

__all__ = ["detect_alignment_heads", "HeadDetectionConfig"]
