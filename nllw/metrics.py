"""Standalone metrics module for NLLW translation evaluation.

Wraps sacrebleu (BLEU) and Unbabel COMET / xCOMET-XL with graceful
fallbacks when the heavy dependencies are not installed.

Usage:
    from nllw.metrics import compute_bleu, compute_comet, compute_xcomet

    bleu = compute_bleu(hypotheses, references)
    comet = compute_comet(sources, hypotheses, references)
    xcomet = compute_xcomet(sources, hypotheses, references)

Each function returns a dict:
    {
        "score": float,          # corpus-level average
        "scores": list[float],   # per-sentence scores
        "model": str,            # model name used (or "sacrebleu" for BLEU)
    }

Dependencies:
    - BLEU:   pip install sacrebleu
    - COMET:  pip install unbabel-comet>=2.2.0
    - xCOMET: pip install unbabel-comet>=2.2.0  (same package, larger model)
"""

from typing import Optional

# ---------------------------------------------------------------------------
# Optional imports
# ---------------------------------------------------------------------------

try:
    import sacrebleu as _sacrebleu

    _SACREBLEU_AVAILABLE = True
except ImportError:
    _sacrebleu = None
    _SACREBLEU_AVAILABLE = False

try:
    from comet import download_model as _comet_download, load_from_checkpoint as _comet_load

    _COMET_AVAILABLE = True
except ImportError:
    _COMET_AVAILABLE = False

# ---------------------------------------------------------------------------
# Model names
# ---------------------------------------------------------------------------

COMET_WMT22 = "Unbabel/wmt22-comet-da"
XCOMET_XL = "Unbabel/XCOMET-XL"

# ---------------------------------------------------------------------------
# Internal cache: avoid reloading the same model multiple times
# ---------------------------------------------------------------------------

_loaded_models: dict = {}


def _get_comet_model(model_name: str, device: Optional[str] = None):
    """Load (or return cached) COMET model.

    Parameters
    ----------
    model_name : str
        HuggingFace model identifier, e.g. ``"Unbabel/wmt22-comet-da"``
        or ``"Unbabel/XCOMET-XL"``.
    device : str or None
        ``"cuda"``, ``"mps"``, ``"cpu"``, or None (auto-detect).

    Returns
    -------
    The loaded COMET model and the resolved device string as a tuple.
    """
    if not _COMET_AVAILABLE:
        raise ImportError(
            "COMET is not installed. Install it with:  pip install unbabel-comet>=2.2.0"
        )

    # Auto-detect device
    if device is None:
        device = _auto_device()

    cache_key = (model_name, device)
    if cache_key in _loaded_models:
        return _loaded_models[cache_key], device

    model_path = _comet_download(model_name)
    model = _comet_load(model_path)
    model._eval_device = device

    _loaded_models[cache_key] = model
    return model, device


def _auto_device() -> str:
    """Pick the best available device: cuda > mps > cpu."""
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return "cpu"


def _score_with_comet(
    model,
    device: str,
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    batch_size: int = 8,
) -> dict:
    """Run COMET predict and normalise the return value."""
    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    if device == "cuda":
        gpus = 1
        accelerator = "auto"
    elif device == "mps":
        gpus = 0
        accelerator = "mps"
    else:
        gpus = 0
        accelerator = "cpu"

    output = model.predict(data, batch_size=batch_size, gpus=gpus, accelerator=accelerator)

    # COMET >=2.x returns a Prediction namedtuple; older versions return a tuple.
    if hasattr(output, "system_score"):
        return {
            "system_score": float(output.system_score),
            "segment_scores": [float(s) for s in output.scores],
        }
    elif isinstance(output, tuple):
        scores = output[0]
        sys_score = float(output[1]) if len(output) > 1 else sum(scores) / len(scores)
        return {
            "system_score": sys_score,
            "segment_scores": [float(s) for s in scores],
        }
    else:
        scores = list(output)
        return {
            "system_score": sum(scores) / len(scores) if scores else 0.0,
            "segment_scores": [float(s) for s in scores],
        }


# ============================================================================
# Public API
# ============================================================================


def compute_bleu(
    hypotheses: list[str],
    references: list[str],
) -> dict:
    """Compute corpus BLEU using sacrebleu.

    Parameters
    ----------
    hypotheses : list[str]
        System translations.
    references : list[str]
        Reference translations (same length as *hypotheses*).

    Returns
    -------
    dict
        ``score`` (corpus BLEU, 0-100 scale), ``scores`` (per-sentence BLEU),
        ``model`` (always ``"sacrebleu"``).

    Raises
    ------
    ImportError
        If sacrebleu is not installed.
    """
    if not _SACREBLEU_AVAILABLE:
        raise ImportError(
            "sacrebleu is not installed. Install it with:  pip install sacrebleu"
        )

    corpus_result = _sacrebleu.corpus_bleu(hypotheses, [references])
    corpus_score = round(corpus_result.score, 2)

    # Per-sentence BLEU
    per_sentence: list[float] = []
    for hyp, ref in zip(hypotheses, references):
        try:
            s = _sacrebleu.sentence_bleu(hyp, [ref]).score
            per_sentence.append(round(s, 2))
        except Exception:
            per_sentence.append(0.0)

    return {
        "score": corpus_score,
        "scores": per_sentence,
        "model": "sacrebleu",
    }


def compute_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    model_name: str = COMET_WMT22,
    *,
    device: Optional[str] = None,
    batch_size: int = 8,
) -> dict:
    """Compute COMET scores (source + hypothesis + reference).

    Parameters
    ----------
    sources : list[str]
        Source sentences.
    hypotheses : list[str]
        System translations.
    references : list[str]
        Reference translations.
    model_name : str
        COMET model identifier.  Default: ``Unbabel/wmt22-comet-da``
        (fast, ~400M params).  Use ``Unbabel/XCOMET-XL`` for the
        IWSLT-2026 official metric (~3.5B params, needs ~14 GB VRAM).
    device : str or None
        ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` (auto-detect).
    batch_size : int
        Batch size for model inference.

    Returns
    -------
    dict
        ``score`` (corpus average, 0-1 scale), ``scores`` (per-sentence),
        ``model`` (model name string).

    Raises
    ------
    ImportError
        If ``unbabel-comet`` is not installed.
    """
    model, resolved_device = _get_comet_model(model_name, device=device)
    result = _score_with_comet(
        model, resolved_device, sources, hypotheses, references,
        batch_size=batch_size,
    )

    return {
        "score": round(result["system_score"], 4),
        "scores": [round(s, 4) for s in result["segment_scores"]],
        "model": model_name,
    }


def compute_xcomet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    *,
    device: Optional[str] = None,
    batch_size: int = 8,
) -> dict:
    """Compute xCOMET-XL scores (IWSLT-2026 official metric).

    This is a convenience wrapper around :func:`compute_comet` with
    ``model_name="Unbabel/XCOMET-XL"``.

    Parameters
    ----------
    sources : list[str]
        Source sentences.
    hypotheses : list[str]
        System translations.
    references : list[str]
        Reference translations.
    device : str or None
        ``"cuda"``, ``"mps"``, ``"cpu"``, or ``None`` (auto-detect).
    batch_size : int
        Batch size for model inference.

    Returns
    -------
    dict
        ``score`` (corpus average, 0-1 scale), ``scores`` (per-sentence),
        ``model`` (always ``"Unbabel/XCOMET-XL"``).

    Raises
    ------
    ImportError
        If ``unbabel-comet`` is not installed.
    """
    return compute_comet(
        sources, hypotheses, references,
        model_name=XCOMET_XL,
        device=device,
        batch_size=batch_size,
    )


# ---------------------------------------------------------------------------
# Availability checks (for callers that want to skip gracefully)
# ---------------------------------------------------------------------------


def sacrebleu_available() -> bool:
    """Return True if sacrebleu is importable."""
    return _SACREBLEU_AVAILABLE


def comet_available() -> bool:
    """Return True if unbabel-comet is importable."""
    return _COMET_AVAILABLE
