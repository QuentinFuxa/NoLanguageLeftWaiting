"""Alignment head detection and pre-computed configs.

Pre-computed head configs are stored as JSON files in heads/configs/.
Use nllw.alignatt.load_head_config() to load them.
"""

import os

CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "configs")


def list_configs():
    """List all available head config files."""
    if not os.path.isdir(CONFIGS_DIR):
        return []
    return sorted(f for f in os.listdir(CONFIGS_DIR) if f.endswith(".json"))


def get_config_path(name: str) -> str:
    """Get full path to a head config file."""
    path = os.path.join(CONFIGS_DIR, name)
    if not os.path.exists(path):
        # Try with prefix
        for f in os.listdir(CONFIGS_DIR):
            if name in f:
                return os.path.join(CONFIGS_DIR, f)
        raise FileNotFoundError(f"Head config '{name}' not found in {CONFIGS_DIR}")
    return path
