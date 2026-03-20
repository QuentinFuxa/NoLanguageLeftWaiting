"""LoRA adapter utilities for NLLW.

Provides discovery, configuration, and documentation helpers for using
LoRA adapters with the AlignAtt translation backends.

LoRA (Low-Rank Adaptation) allows fine-tuning a base translation model for
specific domains, language pairs, or quality improvements without modifying
the original weights.  The adapter is stored as a small GGUF file that is
loaded alongside the base model at runtime.

Creating LoRA adapters for translation
--------------------------------------

The general workflow is:

1. **Prepare training data** -- Collect parallel sentence pairs (source, target)
   in JSONL chat format.  For simultaneous translation you should also include
   partial-source pairs (prefixes of the source with proportionally truncated
   targets).  See ``prepare_training_data.py`` in the iwslt reference for an
   example.

2. **Fine-tune with PEFT/LoRA** -- Use HuggingFace ``peft`` + ``trl`` to train
   a LoRA adapter on top of the base model (e.g. HY-MT 7B or Qwen 4B).
   Typical settings: rank=16, alpha=32, target_modules=["q_proj", "v_proj"].

   Example (abbreviated)::

       from peft import LoraConfig, get_peft_model
       from trl import SFTTrainer, SFTConfig

       lora_config = LoraConfig(
           r=16, lora_alpha=32, lora_dropout=0.05,
           target_modules=["q_proj", "v_proj"],
           task_type="CAUSAL_LM",
       )
       model = get_peft_model(base_model, lora_config)
       # ... train with SFTTrainer ...
       model.save_pretrained("checkpoints/final")

3. **Convert to GGUF** -- Use llama.cpp's ``convert_lora_to_gguf.py``::

       cd /path/to/llama.cpp
       python3 convert_lora_to_gguf.py checkpoints/final \\
           --outfile my_adapter.gguf \\
           --base "OriginalModelName"

4. **Use with NLLW** -- Pass the GGUF adapter path when creating a backend::

       from nllw import create_backend

       backend = create_backend(
           "alignatt",
           source_lang="en",
           target_lang="zh",
           model_path="/path/to/base_model.gguf",
           lora_path="/path/to/my_adapter.gguf",
           lora_scale=1.0,  # 0.0-1.0, controls adapter strength
       )

   Or via the web debug server::

       POST /load
       {
           "model_path": "/path/to/base_model.gguf",
           "lora_path": "/path/to/my_adapter.gguf",
           "lora_scale": 1.0,
           "target_lang": "zh"
       }
"""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class LoRAConfig:
    """Configuration for a LoRA adapter.

    Attributes:
        path: Filesystem path to the GGUF adapter file.
        scale: Adapter weight (0.0 to 1.0+).  1.0 applies the adapter at
            full strength; values < 1.0 blend with the base model; values
            > 1.0 amplify the adapter effect.
        description: Human-readable description of what the adapter does.
        source_lang: Source language the adapter was trained for (if specific).
        target_lang: Target language the adapter was trained for (if specific).
        base_model: Name or path of the base model this adapter was trained on.
        tags: Arbitrary tags for filtering/display (e.g. ["domain:medical"]).
    """

    path: str
    scale: float = 1.0
    description: str = ""
    source_lang: Optional[str] = None
    target_lang: Optional[str] = None
    base_model: Optional[str] = None
    tags: list[str] = field(default_factory=list)

    def exists(self) -> bool:
        """Return True if the adapter file exists on disk."""
        return os.path.isfile(self.path)

    @property
    def filename(self) -> str:
        """Return just the filename portion of the path."""
        return os.path.basename(self.path)

    @property
    def size_mb(self) -> float:
        """Return the adapter file size in megabytes, or 0 if missing."""
        if self.exists():
            return os.path.getsize(self.path) / (1024 * 1024)
        return 0.0


def list_lora_adapters(search_dir: str, recursive: bool = True) -> list[LoRAConfig]:
    """Find GGUF LoRA adapter files in a directory.

    Scans *search_dir* for files matching common LoRA adapter naming
    patterns (``*lora*.gguf``, ``*adapter*.gguf``).  Since any GGUF file
    *could* be a LoRA adapter, files that don't match the naming heuristic
    but end in ``.gguf`` are also included with a note in their description.

    Args:
        search_dir: Directory to search.
        recursive: If True, search subdirectories as well.

    Returns:
        A list of :class:`LoRAConfig` instances (with only ``path`` and
        ``description`` populated).  The caller should fill in ``scale``
        and language fields as appropriate.
    """
    if not os.path.isdir(search_dir):
        return []

    adapters: list[LoRAConfig] = []
    seen_paths: set[str] = set()

    def _is_likely_lora(name: str) -> bool:
        lower = name.lower()
        return any(kw in lower for kw in ("lora", "adapter", "finetun", "ft_"))

    for dirpath, dirnames, filenames in os.walk(search_dir):
        for fname in sorted(filenames):
            if not fname.endswith(".gguf"):
                continue
            full = os.path.abspath(os.path.join(dirpath, fname))
            if full in seen_paths:
                continue
            seen_paths.add(full)

            likely = _is_likely_lora(fname)
            desc = (
                f"LoRA adapter: {fname}"
                if likely
                else f"GGUF file (may be adapter or base model): {fname}"
            )
            adapters.append(LoRAConfig(path=full, description=desc))

        if not recursive:
            break  # only top-level

    return adapters
