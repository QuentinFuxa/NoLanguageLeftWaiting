"""Wrapper around SimulMT backends for the FastAPI debug server.

Uses create_backend() from backend_protocol.py so any registered backend
type (alignatt, alignatt-la, alignatt-kv, ...) can be loaded at runtime.

Thread-safety: AlignAttBackend already uses a threading.Lock internally
for its translate() and finish() methods. This wrapper does NOT add a
second lock -- it relies on the backend's own lock to serialise access.
"""

import os
import time
from typing import Any, Optional


# All backend types that the debug server can instantiate.
AVAILABLE_BACKENDS = ("alignatt", "alignatt-kv", "alignatt-la")


class TranslationService:
    """Singleton wrapper around any SimulMTBackend.

    Usage::

        svc = TranslationService.get_instance()
        svc.load(backend_type="alignatt", model_path="/path/model.gguf")
        result = svc.translate("hello world")
        final  = svc.finish()
        svc.reset()
    """

    _instance: Optional["TranslationService"] = None

    def __init__(self) -> None:
        self._backend: Any = None
        self._backend_type: str = "alignatt"
        self._source_lang: str = "en"
        self._target_lang: str = "fr"
        self._model_path: Optional[str] = None
        self._heads_path: Optional[str] = None
        self._prompt_format: str = "hymt"
        self._custom_template: Optional[str] = None
        self._border_distance: int = 3
        self._word_batch: int = 3
        self._entropy_veto_threshold: Optional[float] = None
        self._lora_path: Optional[str] = None
        self._lora_scale: float = 1.0

    @classmethod
    def get_instance(cls) -> "TranslationService":
        """Return the singleton TranslationService instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._backend is not None

    def load(
        self,
        model_path: Optional[str] = None,
        target_lang: str = "fr",
        source_lang: str = "en",
        *,
        backend_type: str = "alignatt",
        heads_path: Optional[str] = None,
        prompt_format: str = "hymt",
        custom_template: Optional[str] = None,
        border_distance: int = 3,
        word_batch: int = 3,
        n_ctx: int = 2048,
        top_k: int = 10,
        entropy_veto_threshold: Optional[float] = None,
        lora_path: Optional[str] = None,
        lora_scale: float = 1.0,
        verbose: bool = False,
    ) -> dict:
        """Load a translation backend via create_backend().

        If the requested model_path matches the currently loaded model and
        only backend_type / parameters differ, the model is NOT reloaded;
        instead a new backend is created (which reuses the cached llama.cpp
        model when the path is identical).

        Returns
        -------
        dict
            ``{"ok": True}`` on success, or ``{"ok": False, "error": str}``
            on failure.
        """
        if model_path is None:
            model_path = os.environ.get("HYMT_MODEL_PATH")
        if model_path is None:
            return {
                "ok": False,
                "error": (
                    "No model path provided.  Pass model_path or set the "
                    "HYMT_MODEL_PATH environment variable."
                ),
            }

        try:
            from nllw.backend_protocol import create_backend

            self._backend = create_backend(
                backend_type=backend_type,
                source_lang=source_lang,
                target_lang=target_lang,
                model_path=model_path,
                heads_path=heads_path,
                prompt_format=prompt_format,
                custom_template=custom_template,
                border_distance=border_distance,
                word_batch=word_batch,
                n_ctx=n_ctx,
                top_k=top_k,
                entropy_veto_threshold=entropy_veto_threshold,
                lora_path=lora_path,
                lora_scale=lora_scale,
                verbose=verbose,
            )
            self._backend_type = backend_type
            self._source_lang = source_lang
            self._target_lang = target_lang
            self._model_path = model_path
            self._heads_path = heads_path
            self._prompt_format = prompt_format
            self._custom_template = custom_template
            self._border_distance = border_distance
            self._word_batch = word_batch
            self._entropy_veto_threshold = entropy_veto_threshold
            self._lora_path = lora_path
            self._lora_scale = lora_scale
            return {"ok": True}
        except Exception as exc:
            self._backend = None
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Translation
    # ------------------------------------------------------------------

    def translate(self, text: str) -> dict:
        """Translate an incremental chunk of source text.

        Returns
        -------
        dict
            On success::

                {
                    "stable": str,
                    "buffer": str,
                    "source_words": list[str],
                    "committed_tokens": int,
                    "time_ms": float,
                }

            On error::

                {"error": str}
        """
        if not self.loaded:
            return {"error": "Backend not loaded. Call /load first."}

        try:
            t0 = time.perf_counter()
            stable, buffer = self._backend.translate(text)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            return {
                "stable": stable,
                "buffer": buffer,
                "source_words": list(getattr(self._backend, "_source_words", [])),
                "committed_tokens": self._committed_count(),
                "time_ms": round(elapsed_ms, 2),
            }
        except Exception as exc:
            return {"error": f"Translation failed: {exc}"}

    def finish(self) -> dict:
        """Flush remaining translation (generate until EOS).

        Returns
        -------
        dict
            On success::

                {
                    "remaining": str,
                    "full_translation": str,
                    "time_ms": float,
                }

            On error::

                {"error": str}
        """
        if not self.loaded:
            return {"error": "Backend not loaded. Call /load first."}

        try:
            t0 = time.perf_counter()
            remaining = self._backend.finish()
            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            # Reconstruct full translation from all committed tokens.
            full_translation = ""
            if hasattr(self._backend, "_committed_ids") and self._backend._committed_ids:
                from nllw import llama_backend as ll
                full_translation = ll.tokens_to_text(
                    self._backend._vocab,
                    self._backend._committed_ids,
                    errors="ignore",
                )

            return {
                "remaining": remaining,
                "full_translation": full_translation,
                "time_ms": round(elapsed_ms, 2),
            }
        except Exception as exc:
            return {"error": f"Finish failed: {exc}"}

    # ------------------------------------------------------------------
    # State management
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset the backend for a new sentence / utterance."""
        if self._backend is not None:
            self._backend.reset()

    def set_target_lang(self, lang: str) -> dict:
        """Change the target language (also resets translation state).

        Returns
        -------
        dict
            ``{"ok": True}`` on success, or ``{"ok": False, "error": str}``
            on failure.
        """
        if not self.loaded:
            return {"ok": False, "error": "Backend not loaded. Call /load first."}

        try:
            self._backend.set_target_lang(lang)
            self._target_lang = lang
            return {"ok": True}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ------------------------------------------------------------------
    # Comparison — run same text through multiple configs
    # ------------------------------------------------------------------

    def compare(
        self,
        text: str,
        configs: list[dict],
    ) -> list[dict]:
        """Run *text* through each config and return comparison results.

        Each config dict may contain: backend_type, border_distance,
        word_batch, entropy_veto_threshold, heads_path.
        The current model_path / source_lang / target_lang are reused.

        Returns a list of result dicts (one per config).
        """
        if not self._model_path:
            return [{"error": "No model loaded. Call /load first."}]

        from nllw.backend_protocol import create_backend

        results: list[dict] = []
        words = text.strip().split()

        for cfg in configs:
            bt = cfg.get("backend_type", self._backend_type)
            bd = cfg.get("border_distance", self._border_distance)
            wb = cfg.get("word_batch", self._word_batch)
            evt = cfg.get("entropy_veto_threshold", self._entropy_veto_threshold)
            hp = cfg.get("heads_path", self._heads_path)
            pf = cfg.get("prompt_format", self._prompt_format)
            ct = cfg.get("custom_template", self._custom_template)

            try:
                backend = create_backend(
                    backend_type=bt,
                    source_lang=self._source_lang,
                    target_lang=self._target_lang,
                    model_path=self._model_path,
                    heads_path=hp,
                    prompt_format=pf,
                    custom_template=ct,
                    border_distance=bd,
                    word_batch=wb,
                    entropy_veto_threshold=evt,
                )

                steps: list[dict] = []
                all_stable = ""
                total_start = time.perf_counter()

                for word in words:
                    chunk = word + " "
                    step_start = time.perf_counter()
                    stable, buffer = backend.translate(chunk)
                    step_ms = (time.perf_counter() - step_start) * 1000.0
                    all_stable += stable
                    steps.append({
                        "word": word,
                        "stable": stable,
                        "buffer": buffer,
                        "step_time_ms": round(step_ms, 2),
                    })

                finish_start = time.perf_counter()
                remaining = backend.finish()
                finish_ms = (time.perf_counter() - finish_start) * 1000.0
                total_ms = (time.perf_counter() - total_start) * 1000.0

                hypothesis = (all_stable + remaining).strip()

                results.append({
                    "config": {
                        "backend_type": bt,
                        "border_distance": bd,
                        "word_batch": wb,
                        "entropy_veto_threshold": evt,
                    },
                    "hypothesis": hypothesis,
                    "steps": steps,
                    "finish_remaining": remaining,
                    "total_time_ms": round(total_ms, 2),
                    "finish_time_ms": round(finish_ms, 2),
                })
            except Exception as exc:
                results.append({
                    "config": cfg,
                    "error": str(exc),
                })

        return results

    # ------------------------------------------------------------------
    # Evaluation — run the evaluator on provided test cases
    # ------------------------------------------------------------------

    def evaluate(
        self,
        test_cases: list[dict],
        backend_type: Optional[str] = None,
        params: Optional[dict] = None,
    ) -> dict:
        """Run SimulMTEvaluator on *test_cases*.

        Parameters
        ----------
        test_cases : list[dict]
            Each dict must have ``source``, ``reference``, ``source_lang``,
            ``target_lang``.
        backend_type : str, optional
            Override backend type (default: current).
        params : dict, optional
            Extra params: border_distance, word_batch, entropy_veto_threshold,
            heads_path.

        Returns
        -------
        dict
            Evaluator results including BLEU, committed_ratio, timing, and
            per-sentence details.
        """
        if not self._model_path:
            return {"error": "No model loaded. Call /load first."}

        from nllw.backend_protocol import create_backend
        from nllw.eval import SimulMTEvaluator

        bt = backend_type or self._backend_type
        p = params or {}

        try:
            backend = create_backend(
                backend_type=bt,
                source_lang=self._source_lang,
                target_lang=self._target_lang,
                model_path=self._model_path,
                heads_path=p.get("heads_path", self._heads_path),
                prompt_format=p.get("prompt_format", self._prompt_format),
                custom_template=p.get("custom_template", self._custom_template),
                border_distance=p.get("border_distance", self._border_distance),
                word_batch=p.get("word_batch", self._word_batch),
                entropy_veto_threshold=p.get(
                    "entropy_veto_threshold", self._entropy_veto_threshold
                ),
            )

            evaluator = SimulMTEvaluator(backend)
            results = evaluator.evaluate_corpus(test_cases)
            return results
        except Exception as exc:
            return {"error": str(exc)}

    # ------------------------------------------------------------------
    # Introspection
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        """Return current service status with full configuration details."""
        base = {
            "loaded": self.loaded,
            "source_lang": self._source_lang,
            "target_lang": self._target_lang,
            "backend_type": self._backend_type,
            "prompt_format": self._prompt_format,
            "border_distance": self._border_distance,
            "word_batch": self._word_batch,
            "entropy_veto_threshold": self._entropy_veto_threshold,
            "heads_file": os.path.basename(self._heads_path) if self._heads_path else "default",
            "lora_path": self._lora_path,
            "lora_scale": self._lora_scale,
        }

        if not self.loaded:
            base["source_words"] = []
            base["committed_tokens"] = 0
            return base

        base["target_lang"] = getattr(
            self._backend, "target_lang_iso", self._target_lang
        )
        base["source_words"] = list(getattr(self._backend, "_source_words", []))
        base["committed_tokens"] = self._committed_count()
        return base

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _committed_count(self) -> int:
        """Best-effort read of committed token count from the backend."""
        if hasattr(self._backend, "_committed_ids"):
            return len(self._backend._committed_ids)
        return 0
