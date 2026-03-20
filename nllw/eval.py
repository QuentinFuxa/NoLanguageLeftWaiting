"""Evaluation module for NLLW simultaneous translation backends.

Replays source sentences word-by-word through any backend that implements
the translate(text) -> (stable, buffer) / finish() / reset() interface
(AlignAttBackend or TranslationBackend), collecting per-step metrics.

Usage (standalone):
    python -m nllw.eval                             # default corpus, AlignAtt
    python -m nllw.eval --backend nllb              # NLLB backend
    python -m nllw.eval --model /path/to/model.gguf # custom model
    python -m nllw.eval --comet                     # + COMET (wmt22-comet-da)
    python -m nllw.eval --xcomet                    # + xCOMET-XL

Usage (as library):
    from nllw.eval import SimulMTEvaluator, DEFAULT_TEST_CASES
    evaluator = SimulMTEvaluator(backend)
    results = evaluator.evaluate_corpus(DEFAULT_TEST_CASES)

Dependencies:
    - sacrebleu is optional (BLEU is skipped if unavailable).
    - unbabel-comet is optional (COMET/xCOMET is skipped if unavailable).
"""

import itertools
import time
from typing import Any, Optional, Union

try:
    import sacrebleu

    _SACREBLEU_AVAILABLE = True
except ImportError:
    sacrebleu = None
    _SACREBLEU_AVAILABLE = False

try:
    from nllw.metrics import (
        compute_comet as _compute_comet_metric,
        compute_xcomet as _compute_xcomet_metric,
        comet_available as _comet_available,
        COMET_WMT22,
        XCOMET_XL,
    )

    _COMET_AVAILABLE = _comet_available()
except ImportError:
    _COMET_AVAILABLE = False


# ---------------------------------------------------------------------------
# Default test corpus  (~15 sentence pairs)
# ---------------------------------------------------------------------------

DEFAULT_TEST_CASES: list[dict] = [
    # ── EN → FR (5 sentences) ──────────────────────────────────────────
    {
        "source": "The weather is nice today",
        "reference": "Il fait beau aujourd'hui",
        "source_lang": "en",
        "target_lang": "fr",
        "tag": "en-fr/short",
    },
    {
        "source": "I would like to order a coffee with milk please",
        "reference": "Je voudrais commander un cafe au lait s'il vous plait",
        "source_lang": "en",
        "target_lang": "fr",
        "tag": "en-fr/medium",
    },
    {
        "source": "The president of France announced new economic reforms during the press conference held in Paris yesterday afternoon",
        "reference": "Le president de la France a annonce de nouvelles reformes economiques lors de la conference de presse tenue a Paris hier apres-midi",
        "source_lang": "en",
        "target_lang": "fr",
        "tag": "en-fr/long",
    },
    {
        "source": "My name is Jean-Pierre Dupont and I live in Marseille",
        "reference": "Je m'appelle Jean-Pierre Dupont et j'habite a Marseille",
        "source_lang": "en",
        "target_lang": "fr",
        "tag": "en-fr/names",
    },
    {
        "source": "The company reported revenue of 3.5 billion euros in 2024",
        "reference": "L'entreprise a declare un chiffre d'affaires de 3,5 milliards d'euros en 2024",
        "source_lang": "en",
        "target_lang": "fr",
        "tag": "en-fr/numbers",
    },
    # ── FR → EN (5 sentences) ──────────────────────────────────────────
    {
        "source": "Il fait beau aujourd'hui",
        "reference": "The weather is nice today",
        "source_lang": "fr",
        "target_lang": "en",
        "tag": "fr-en/short",
    },
    {
        "source": "Je voudrais commander un cafe au lait s'il vous plait",
        "reference": "I would like to order a coffee with milk please",
        "source_lang": "fr",
        "target_lang": "en",
        "tag": "fr-en/medium",
    },
    {
        "source": "Le president de la France a annonce de nouvelles reformes economiques hier",
        "reference": "The president of France announced new economic reforms yesterday",
        "source_lang": "fr",
        "target_lang": "en",
        "tag": "fr-en/long",
    },
    {
        "source": "Bonjour je m'appelle Quentin et je viens de Lyon",
        "reference": "Hello my name is Quentin and I come from Lyon",
        "source_lang": "fr",
        "target_lang": "en",
        "tag": "fr-en/names",
    },
    {
        "source": "L'entreprise a genere un benefice net de 250 millions d'euros",
        "reference": "The company generated a net profit of 250 million euros",
        "source_lang": "fr",
        "target_lang": "en",
        "tag": "fr-en/numbers",
    },
    # ── EN → DE (3 sentences) ──────────────────────────────────────────
    {
        "source": "Good morning how are you today",
        "reference": "Guten Morgen wie geht es Ihnen heute",
        "source_lang": "en",
        "target_lang": "de",
        "tag": "en-de/short",
    },
    {
        "source": "The train from Berlin to Munich departs at half past three",
        "reference": "Der Zug von Berlin nach Muenchen faehrt um halb vier ab",
        "source_lang": "en",
        "target_lang": "de",
        "tag": "en-de/medium",
    },
    {
        "source": "Artificial intelligence is transforming the way we work and communicate with each other",
        "reference": "Kuenstliche Intelligenz veraendert die Art und Weise wie wir arbeiten und miteinander kommunizieren",
        "source_lang": "en",
        "target_lang": "de",
        "tag": "en-de/long",
    },
    # ── EN → ES (2 sentences) ──────────────────────────────────────────
    {
        "source": "The children are playing in the park near the river",
        "reference": "Los ninos estan jugando en el parque cerca del rio",
        "source_lang": "en",
        "target_lang": "es",
        "tag": "en-es/medium",
    },
    {
        "source": "Climate change is one of the greatest challenges facing humanity in the twenty first century",
        "reference": "El cambio climatico es uno de los mayores desafios que enfrenta la humanidad en el siglo veintiuno",
        "source_lang": "en",
        "target_lang": "es",
        "tag": "en-es/long",
    },
]


# ---------------------------------------------------------------------------
# SimulMTEvaluator
# ---------------------------------------------------------------------------


class SimulMTEvaluator:
    """Evaluate simultaneous translation quality by replaying source words.

    Works with any backend that exposes:
      - translate(text) -> (stable, buffer)
      - finish() -> str
      - reset()

    Both ``AlignAttBackend`` and ``TranslationBackend`` satisfy this contract.
    Alternatively, *backend* can be a ``requests``-style API wrapper (see
    ``WebAPIBackend`` below) for evaluating via the web debug server.
    """

    def __init__(self, backend: Any) -> None:
        self.backend = backend

    # ------------------------------------------------------------------
    # Single-sentence evaluation
    # ------------------------------------------------------------------

    def evaluate_sentence(
        self,
        source_words: list[str],
        reference: str,
    ) -> dict:
        """Replay *source_words* one by one through the backend.

        Returns a dict with per-step details and aggregate sentence metrics.
        """
        self.backend.reset()

        steps: list[dict] = []
        all_stable = ""
        total_start = time.perf_counter()

        for word in source_words:
            chunk = word + " "
            step_start = time.perf_counter()
            stable, buffer = self.backend.translate(chunk)
            step_ms = (time.perf_counter() - step_start) * 1000.0

            all_stable += stable

            # Try to read committed_tokens from backend internals (best-effort)
            committed_tokens = self._get_committed_count()

            steps.append(
                {
                    "word": word,
                    "stable": stable,
                    "buffer": buffer,
                    "committed_tokens": committed_tokens,
                    "step_time_ms": round(step_ms, 2),
                }
            )

        committed_at_finish = self._get_committed_count()

        # Finish
        finish_start = time.perf_counter()
        remaining = self.backend.finish()
        finish_ms = (time.perf_counter() - finish_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        hypothesis = (all_stable + remaining).strip()

        return {
            "source": " ".join(source_words),
            "reference": reference,
            "hypothesis": hypothesis,
            "steps": steps,
            "total_time_ms": round(total_ms, 2),
            "finish_time_ms": round(finish_ms, 2),
            "committed_at_finish": committed_at_finish,
            "finish_remaining": remaining,
        }

    # ------------------------------------------------------------------
    # Corpus-level evaluation
    # ------------------------------------------------------------------

    def evaluate_corpus(
        self,
        test_cases: list[dict],
        *,
        verbose: bool = False,
        compute_comet: bool = False,
        compute_xcomet: bool = False,
        comet_device: Optional[str] = None,
        comet_batch_size: int = 8,
    ) -> dict:
        """Run ``evaluate_sentence`` on each test case and compute aggregates.

        Each entry in *test_cases* must have at least ``source`` (str) and
        ``reference`` (str).  Optional keys: ``source_lang``, ``target_lang``,
        ``tag``.

        Parameters
        ----------
        compute_comet : bool
            If True, compute COMET (wmt22-comet-da) scores.  Requires
            ``unbabel-comet`` to be installed; silently skipped otherwise.
        compute_xcomet : bool
            If True, compute xCOMET-XL scores.  Requires ``unbabel-comet``
            to be installed; silently skipped otherwise.
        comet_device : str or None
            Device for COMET inference (``"cuda"``, ``"mps"``, ``"cpu"``).
            ``None`` means auto-detect.
        comet_batch_size : int
            Batch size for COMET inference.

        Returns a dict with ``sentences`` (list of per-sentence results) and
        aggregate metrics (``bleu``, ``comet``, ``xcomet_xl``,
        ``avg_committed_ratio``, etc.).
        """
        sentences: list[dict] = []
        sources: list[str] = []
        hypotheses: list[str] = []
        references: list[str] = []

        current_lang_pair: Optional[tuple] = None

        for i, case in enumerate(test_cases):
            source_text: str = case["source"]
            reference_text: str = case["reference"]
            src_lang = case.get("source_lang")
            tgt_lang = case.get("target_lang")
            tag = case.get("tag", "")

            # Switch language pair if the backend supports it and the pair changed
            lang_pair = (src_lang, tgt_lang)
            if lang_pair != current_lang_pair and lang_pair != (None, None):
                self._switch_lang_pair(src_lang, tgt_lang)
                current_lang_pair = lang_pair

            words = source_text.strip().split()
            result = self.evaluate_sentence(words, reference_text)
            result["tag"] = tag
            result["index"] = i

            sentences.append(result)
            sources.append(source_text)
            hypotheses.append(result["hypothesis"])
            references.append(reference_text)

            if verbose:
                _print_sentence_result(result, i, len(test_cases))

        # Aggregate metrics
        aggregates = self._compute_aggregates(
            sentences, sources, hypotheses, references,
            compute_comet=compute_comet,
            compute_xcomet=compute_xcomet,
            comet_device=comet_device,
            comet_batch_size=comet_batch_size,
        )
        aggregates["sentences"] = sentences

        return aggregates

    # ------------------------------------------------------------------
    # Parameter sweep
    # ------------------------------------------------------------------

    def run_parameter_sweep(
        self,
        test_cases: list[dict],
        param_grid: dict[str, list],
        *,
        backend_factory: Any = None,
        verbose: bool = False,
    ) -> list[dict]:
        """Evaluate every combination in *param_grid*.

        *param_grid* maps parameter names (e.g. ``border_distance``,
        ``word_batch``) to lists of values to try.

        If *backend_factory* is provided it is called as
        ``backend_factory(**params)`` to create a fresh backend for each
        configuration.  Otherwise, the evaluator tries to set attributes on
        ``self.backend`` directly and calls ``reset()``.

        Returns a list of dicts sorted by BLEU (descending), each containing
        the parameter combination and aggregate metrics.
        """
        param_names = list(param_grid.keys())
        param_values = list(param_grid.values())
        sweep_results: list[dict] = []

        combos = list(itertools.product(*param_values))
        total = len(combos)

        for idx, combo in enumerate(combos):
            params = dict(zip(param_names, combo))

            if verbose:
                print(
                    f"\n{'='*60}\n"
                    f"  Sweep [{idx + 1}/{total}]  {params}\n"
                    f"{'='*60}"
                )

            # Apply parameters
            if backend_factory is not None:
                self.backend = backend_factory(**params)
            else:
                for name, value in params.items():
                    if hasattr(self.backend, name):
                        setattr(self.backend, name, value)
                    else:
                        print(f"  Warning: backend has no attribute '{name}', skipping")
                self.backend.reset()

            corpus_result = self.evaluate_corpus(test_cases, verbose=verbose)

            entry = dict(params)
            entry["bleu"] = corpus_result.get("bleu")
            entry["avg_committed_ratio"] = corpus_result["avg_committed_ratio"]
            entry["avg_time_per_word_ms"] = corpus_result["avg_time_per_word_ms"]
            entry["avg_finish_ratio"] = corpus_result["avg_finish_ratio"]
            entry["num_sentences"] = len(test_cases)
            sweep_results.append(entry)

        # Sort by BLEU descending (None sorts last)
        sweep_results.sort(
            key=lambda r: r["bleu"] if r["bleu"] is not None else -1,
            reverse=True,
        )
        return sweep_results

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_committed_count(self) -> int:
        """Best-effort attempt to read committed token count from the backend."""
        # AlignAttBackend
        if hasattr(self.backend, "_committed_ids"):
            return len(self.backend._committed_ids)
        # TranslationBackend
        if hasattr(self.backend, "stable_prefix_segments"):
            return sum(
                len(seg) if hasattr(seg, "__len__") else seg.numel()
                for seg in self.backend.stable_prefix_segments
            )
        # WebAPIBackend or unknown
        return -1

    def _switch_lang_pair(self, src_lang: Optional[str], tgt_lang: Optional[str]) -> None:
        """Try to switch the backend's language pair."""
        # AlignAttBackend: set_target_lang resets state
        if tgt_lang and hasattr(self.backend, "set_target_lang"):
            try:
                self.backend.set_target_lang(tgt_lang)
                return
            except (ValueError, AttributeError):
                pass
        # WebAPIBackend: load with new pair
        if hasattr(self.backend, "load"):
            try:
                self.backend.load(source_lang=src_lang, target_lang=tgt_lang)
                return
            except Exception:
                pass

    @staticmethod
    def _compute_aggregates(
        sentences: list[dict],
        sources: list[str],
        hypotheses: list[str],
        references: list[str],
        *,
        compute_comet: bool = False,
        compute_xcomet: bool = False,
        comet_device: Optional[str] = None,
        comet_batch_size: int = 8,
    ) -> dict:
        """Compute corpus-level aggregate metrics.

        Parameters
        ----------
        sources : list[str]
            Source sentences (needed for COMET, which scores src+hyp+ref).
        compute_comet : bool
            If True, run wmt22-comet-da scoring.
        compute_xcomet : bool
            If True, run xCOMET-XL scoring.
        comet_device : str or None
            Device for COMET (``None`` = auto-detect).
        comet_batch_size : int
            Batch size for COMET inference.
        """
        n = len(sentences)
        if n == 0:
            return {
                "bleu": None,
                "comet": None,
                "xcomet_xl": None,
                "avg_committed_ratio": 0.0,
                "avg_time_per_word_ms": 0.0,
                "avg_finish_ratio": 0.0,
            }

        # BLEU via sacrebleu
        bleu_score: Optional[float] = None
        if _SACREBLEU_AVAILABLE:
            try:
                result = sacrebleu.corpus_bleu(hypotheses, [references])
                bleu_score = round(result.score, 2)
            except Exception:
                pass

        # COMET (wmt22-comet-da)
        comet_score: Optional[float] = None
        comet_scores: Optional[list[float]] = None
        if compute_comet and _COMET_AVAILABLE:
            try:
                comet_result = _compute_comet_metric(
                    sources, hypotheses, references,
                    model_name=COMET_WMT22,
                    device=comet_device,
                    batch_size=comet_batch_size,
                )
                comet_score = comet_result["score"]
                comet_scores = comet_result["scores"]
            except Exception as e:
                import sys
                print(f"  Warning: COMET scoring failed: {e}", file=sys.stderr)

        # xCOMET-XL
        xcomet_score: Optional[float] = None
        xcomet_scores: Optional[list[float]] = None
        if compute_xcomet and _COMET_AVAILABLE:
            try:
                xcomet_result = _compute_xcomet_metric(
                    sources, hypotheses, references,
                    device=comet_device,
                    batch_size=comet_batch_size,
                )
                xcomet_score = xcomet_result["score"]
                xcomet_scores = xcomet_result["scores"]
            except Exception as e:
                import sys
                print(f"  Warning: xCOMET-XL scoring failed: {e}", file=sys.stderr)

        # Committed ratio: proportion of translation produced during
        # incremental steps (before finish) vs total hypothesis length
        committed_ratios: list[float] = []
        for s in sentences:
            hyp_len = len(s["hypothesis"]) if s["hypothesis"] else 1
            finish_len = len(s["finish_remaining"]) if s["finish_remaining"] else 0
            committed_len = max(0, hyp_len - finish_len)
            committed_ratios.append(committed_len / max(hyp_len, 1))

        # Time per source word
        time_per_word: list[float] = []
        for s in sentences:
            n_words = len(s["source"].split())
            if n_words > 0:
                time_per_word.append(s["total_time_ms"] / n_words)

        # Finish ratio: proportion of text produced by finish()
        finish_ratios: list[float] = []
        for s in sentences:
            hyp_len = len(s["hypothesis"]) if s["hypothesis"] else 1
            finish_len = len(s["finish_remaining"]) if s["finish_remaining"] else 0
            finish_ratios.append(finish_len / max(hyp_len, 1))

        aggregates: dict = {
            "bleu": bleu_score,
            "comet": comet_score,
            "xcomet_xl": xcomet_score,
            "avg_committed_ratio": round(
                sum(committed_ratios) / len(committed_ratios), 4
            ),
            "avg_time_per_word_ms": round(
                sum(time_per_word) / len(time_per_word), 2
            )
            if time_per_word
            else 0.0,
            "avg_finish_ratio": round(
                sum(finish_ratios) / len(finish_ratios), 4
            ),
        }

        # Attach per-sentence COMET/xCOMET scores if computed
        if comet_scores is not None:
            aggregates["comet_scores"] = comet_scores
        if xcomet_scores is not None:
            aggregates["xcomet_xl_scores"] = xcomet_scores

        return aggregates


# ---------------------------------------------------------------------------
# WebAPIBackend — thin wrapper that talks to the debug FastAPI server
# ---------------------------------------------------------------------------


class WebAPIBackend:
    """Backend adapter that calls the web debug server over HTTP.

    Implements the same translate/finish/reset contract so it can be
    passed directly to ``SimulMTEvaluator``.
    """

    def __init__(self, base_url: str = "http://localhost:8777") -> None:
        try:
            import requests as _requests
        except ImportError:
            raise ImportError(
                "WebAPIBackend requires the 'requests' package. "
                "Install it with: pip install requests"
            )
        self._requests = _requests
        self.base_url = base_url.rstrip("/")

    def load(
        self,
        *,
        model_path: Optional[str] = None,
        source_lang: str = "en",
        target_lang: str = "fr",
        border_distance: int = 3,
        word_batch: int = 3,
    ) -> dict:
        body: dict[str, Any] = {
            "target_lang": target_lang,
            "source_lang": source_lang,
            "border_distance": border_distance,
            "word_batch": word_batch,
        }
        if model_path:
            body["model_path"] = model_path
        r = self._requests.post(f"{self.base_url}/load", json=body, timeout=120)
        return r.json()

    def translate(self, text: str) -> tuple[str, str]:
        r = self._requests.post(
            f"{self.base_url}/translate", json={"text": text}, timeout=30
        )
        data = r.json()
        return data.get("stable", ""), data.get("buffer", "")

    def finish(self) -> str:
        r = self._requests.post(f"{self.base_url}/finish", timeout=30)
        data = r.json()
        return data.get("remaining", "")

    def reset(self) -> None:
        self._requests.post(f"{self.base_url}/reset", timeout=10)

    def set_target_lang(self, lang: str) -> None:
        r = self._requests.post(
            f"{self.base_url}/set_lang", json={"lang": lang}, timeout=10
        )
        data = r.json()
        if not data.get("ok", True):
            raise ValueError(data.get("error", "set_target_lang failed"))


# ---------------------------------------------------------------------------
# Pretty-printing helpers
# ---------------------------------------------------------------------------


def _print_sentence_result(result: dict, idx: int, total: int) -> None:
    """Print a single sentence evaluation result."""
    tag = result.get("tag", "")
    label = f"  [{idx + 1}/{total}]"
    if tag:
        label += f"  {tag}"

    print(f"\n{'='*64}")
    print(label)
    print(f"{'='*64}")
    print(f"  Source:     {result['source']}")
    print(f"  Reference:  {result['reference']}")
    print(f"  Hypothesis: {result['hypothesis']}")
    print(f"  {'-'*60}")

    for step in result["steps"]:
        parts = []
        if step["stable"]:
            parts.append(f"stable=\"{step['stable']}\"")
        if step["buffer"]:
            parts.append(f"buffer=\"{step['buffer']}\"")
        parts.append(f"tokens={step['committed_tokens']}")
        parts.append(f"{step['step_time_ms']:.1f}ms")
        print(f"    \"{step['word']}\" -> {' | '.join(parts)}")

    print(f"  {'-'*60}")
    print(f"  Finish remaining: \"{result['finish_remaining']}\"")
    print(f"  Total time: {result['total_time_ms']:.1f}ms")
    committed = result.get("committed_at_finish", -1)
    if committed >= 0:
        print(f"  Committed tokens before finish: {committed}")


def _print_corpus_results(results: dict) -> None:
    """Print corpus-level aggregate results as a table."""
    print(f"\n{'='*72}")
    print("  CORPUS RESULTS")
    print(f"{'='*72}")

    sentences = results.get("sentences", [])

    # Per-sentence table
    header = (
        f"  {'#':>3}  {'Tag':<20}  {'Time(ms)':>9}  "
        f"{'Committed%':>10}  {'Finish%':>8}  Hypothesis"
    )
    print(header)
    print(f"  {'-' * 68}")

    for s in sentences:
        tag = s.get("tag", "")[:20]
        hyp_len = max(len(s["hypothesis"]), 1)
        finish_len = len(s["finish_remaining"]) if s["finish_remaining"] else 0
        committed_pct = ((hyp_len - finish_len) / hyp_len) * 100
        finish_pct = (finish_len / hyp_len) * 100
        hyp_preview = s["hypothesis"][:40] + ("..." if len(s["hypothesis"]) > 40 else "")

        print(
            f"  {s.get('index', 0):>3}  {tag:<20}  "
            f"{s['total_time_ms']:>9.1f}  "
            f"{committed_pct:>9.1f}%  "
            f"{finish_pct:>7.1f}%  "
            f"{hyp_preview}"
        )

    # Aggregates
    print(f"\n  {'-' * 68}")
    bleu = results.get("bleu")
    bleu_str = f"{bleu:.2f}" if bleu is not None else "N/A (install sacrebleu)"
    print(f"  BLEU:                  {bleu_str}")

    comet = results.get("comet")
    if comet is not None:
        print(f"  COMET (wmt22):         {comet:.4f}")

    xcomet = results.get("xcomet_xl")
    if xcomet is not None:
        print(f"  xCOMET-XL:             {xcomet:.4f}")

    print(f"  Avg committed ratio:   {results['avg_committed_ratio']:.2%}")
    print(f"  Avg finish ratio:      {results['avg_finish_ratio']:.2%}")
    print(f"  Avg time/word:         {results['avg_time_per_word_ms']:.1f} ms")
    print()


def _print_sweep_results(sweep: list[dict]) -> None:
    """Print parameter sweep results as a ranked table."""
    if not sweep:
        print("  No sweep results.")
        return

    print(f"\n{'='*72}")
    print("  PARAMETER SWEEP RESULTS (ranked by BLEU)")
    print(f"{'='*72}")

    # Determine param columns
    fixed_keys = {"bleu", "avg_committed_ratio", "avg_time_per_word_ms", "avg_finish_ratio", "num_sentences"}
    param_keys = [k for k in sweep[0] if k not in fixed_keys]

    # Header
    parts = [f"{'Rank':>4}"]
    for pk in param_keys:
        parts.append(f"{pk:>16}")
    parts.append(f"{'BLEU':>8}")
    parts.append(f"{'Commit%':>8}")
    parts.append(f"{'Finish%':>8}")
    parts.append(f"{'ms/word':>8}")
    print("  " + "  ".join(parts))
    print(f"  {'-' * (len('  '.join(parts)) + 2)}")

    for rank, entry in enumerate(sweep, 1):
        parts = [f"{rank:>4}"]
        for pk in param_keys:
            parts.append(f"{entry[pk]!s:>16}")
        bleu = entry.get("bleu")
        parts.append(f"{bleu:>8.2f}" if bleu is not None else f"{'N/A':>8}")
        parts.append(f"{entry['avg_committed_ratio']:>7.2%}")
        parts.append(f"{entry['avg_finish_ratio']:>7.2%}")
        parts.append(f"{entry['avg_time_per_word_ms']:>8.1f}")
        print("  " + "  ".join(parts))

    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def _create_backend(args):
    """Create a backend from CLI arguments."""
    if args.backend == "web":
        backend = WebAPIBackend(args.web_url)
        # Ensure the server has a model loaded
        if not args.skip_load:
            result = backend.load(
                model_path=args.model,
                source_lang=args.source_lang,
                target_lang=args.target_lang,
                border_distance=args.border_distance,
                word_batch=args.word_batch,
            )
            if not result.get("ok", True):
                raise RuntimeError(f"Failed to load model via web API: {result}")
        return backend

    if args.backend == "alignatt":
        from nllw.alignatt_backend import AlignAttBackend

        return AlignAttBackend(
            source_lang=args.source_lang,
            target_lang=args.target_lang,
            model_path=args.model,
            border_distance=args.border_distance,
            word_batch=args.word_batch,
            verbose=args.verbose,
        )

    if args.backend == "nllb":
        from nllw.core import TranslationBackend, load_model
        from nllw.languages import convert_to_nllb_code

        src_nllb = convert_to_nllb_code(args.source_lang) or args.source_lang
        tgt_nllb = convert_to_nllb_code(args.target_lang) or args.target_lang
        model = load_model([src_nllb], nllb_backend="transformers", nllb_size=args.nllb_size)
        return TranslationBackend(
            source_lang=src_nllb,
            target_lang=tgt_nllb,
            model_name=model.model_name,
            model=model.translator,
            tokenizer=model.get_tokenizer(src_nllb),
            backend_type=model.backend_type,
            verbose=args.verbose,
        )

    raise ValueError(f"Unknown backend: {args.backend}")


def _filter_test_cases(
    test_cases: list[dict],
    source_lang: Optional[str],
    target_lang: Optional[str],
    lang_pair: Optional[str],
) -> list[dict]:
    """Filter test cases to those matching the given language pair."""
    if lang_pair:
        src, tgt = lang_pair.split("-", 1)
        return [
            tc for tc in test_cases
            if tc.get("source_lang") == src and tc.get("target_lang") == tgt
        ]

    if source_lang and target_lang:
        return [
            tc for tc in test_cases
            if tc.get("source_lang") == source_lang
            and tc.get("target_lang") == target_lang
        ]

    return test_cases


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Evaluate NLLW simultaneous translation backends.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nllw.eval                              # AlignAtt, default corpus\n"
            "  python -m nllw.eval --backend web                # via running web server\n"
            "  python -m nllw.eval --backend nllb --lang en-fr  # NLLB, EN->FR only\n"
            "  python -m nllw.eval --sweep                      # parameter sweep\n"
            "  python -m nllw.eval --comet                      # + COMET (wmt22-comet-da)\n"
            "  python -m nllw.eval --xcomet                     # + xCOMET-XL (IWSLT-2026)\n"
            "  python -m nllw.eval --comet --xcomet --comet-device cpu  # both metrics, on CPU\n"
        ),
    )

    parser.add_argument(
        "--backend",
        choices=["alignatt", "nllb", "web"],
        default="alignatt",
        help="Translation backend to evaluate (default: alignatt)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model path (GGUF for alignatt, ignored for nllb)",
    )
    parser.add_argument(
        "--source-lang", default="en", help="Source language (default: en)"
    )
    parser.add_argument(
        "--target-lang", default="fr", help="Target language (default: fr)"
    )
    parser.add_argument(
        "--lang",
        default=None,
        help="Language pair filter, e.g. 'en-fr'. Overrides --source-lang/--target-lang for filtering.",
    )
    parser.add_argument(
        "--border-distance", type=int, default=3, help="AlignAtt border distance"
    )
    parser.add_argument(
        "--word-batch", type=int, default=3, help="AlignAtt word batch size"
    )
    parser.add_argument(
        "--nllb-size", default="600M", help="NLLB model size (default: 600M)"
    )
    parser.add_argument(
        "--web-url",
        default="http://localhost:8777",
        help="Web debug server URL (for --backend web)",
    )
    parser.add_argument(
        "--skip-load",
        action="store_true",
        help="Skip model loading (for --backend web, assume model already loaded)",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Verbose per-sentence output"
    )
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="Run parameter sweep instead of single evaluation",
    )
    parser.add_argument(
        "--comet",
        action="store_true",
        help="Compute COMET (wmt22-comet-da) scores. Requires: pip install unbabel-comet",
    )
    parser.add_argument(
        "--xcomet",
        action="store_true",
        help="Compute xCOMET-XL scores (~3.5B params, ~14GB VRAM). Requires: pip install unbabel-comet",
    )
    parser.add_argument(
        "--comet-device",
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device for COMET inference (default: auto-detect)",
    )
    parser.add_argument(
        "--comet-batch-size",
        type=int,
        default=8,
        help="Batch size for COMET inference (default: 8)",
    )
    parser.add_argument(
        "--output", "-o", default=None, help="Save results to JSON file"
    )

    args = parser.parse_args()

    # Filter test cases
    test_cases = _filter_test_cases(
        DEFAULT_TEST_CASES, args.source_lang, args.target_lang, args.lang
    )
    if not test_cases:
        print(f"No test cases match the filter. Available language pairs:")
        pairs = sorted(set(
            (tc["source_lang"], tc["target_lang"]) for tc in DEFAULT_TEST_CASES
        ))
        for s, t in pairs:
            print(f"  {s}-{t}")
        return

    print(f"Backend:    {args.backend}")
    print(f"Test cases: {len(test_cases)}")
    if args.lang:
        print(f"Lang pair:  {args.lang}")
    else:
        print(f"Lang pair:  {args.source_lang}-{args.target_lang} (filter)")

    # Warn if COMET requested but not available
    if (args.comet or args.xcomet) and not _COMET_AVAILABLE:
        print(
            "\n  WARNING: unbabel-comet is not installed. "
            "COMET/xCOMET scores will be skipped.\n"
            "  Install with:  pip install unbabel-comet>=2.2.0\n"
        )

    backend = _create_backend(args)
    evaluator = SimulMTEvaluator(backend)

    if args.sweep:
        # Default parameter grid for AlignAtt
        grid = {
            "border_distance": [2, 3, 4, 5],
            "word_batch": [1, 2, 3],
        }
        print(f"\nParameter sweep: {grid}")
        results = evaluator.run_parameter_sweep(
            test_cases, grid, verbose=args.verbose
        )
        _print_sweep_results(results)

        if args.output:
            import json

            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False)
            print(f"Results saved to {args.output}")
    else:
        results = evaluator.evaluate_corpus(
            test_cases,
            verbose=args.verbose,
            compute_comet=args.comet,
            compute_xcomet=args.xcomet,
            comet_device=args.comet_device,
            comet_batch_size=args.comet_batch_size,
        )
        _print_corpus_results(results)

        if args.output:
            import json

            # Serialize for JSON (steps contain plain dicts, should be fine)
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2, ensure_ascii=False, default=str)
            print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
