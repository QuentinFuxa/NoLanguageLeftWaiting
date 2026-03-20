"""Research benchmarking module for NLLW simultaneous translation.

Ports the core evaluation workflow from iwslt26-sst into a clean research tool
that works over the web API (http://localhost:8777). No local model loading needed.

Provides:
  - ResearchBenchmark: rich per-step traces, compute-aware latency (CA-AL),
    multi-config comparison, full benchmark suites
  - FLORES_MINI: hardcoded 20-sentence test suite (en-fr, fr-en, en-de, en-zh)
  - CLI: python -m nllw.research --suite flores_mini --configs "bd=2,3,4 wb=2,3"

Usage (library):
    from nllw.research import ResearchBenchmark, FLORES_MINI
    rb = ResearchBenchmark("http://localhost:8777")
    trace = rb.benchmark_sentence("The weather is nice today".split(), ref="Il fait beau")
    ca = rb.compute_aware_metrics(trace, speech_rate_wps=2.5)

Usage (CLI):
    python -m nllw.research
    python -m nllw.research --suite flores_mini --configs "bd=2,3,4 wb=2,3"
    python -m nllw.research --sentence "The president announced new reforms"
"""

import itertools
import json
import time
from typing import Any, Optional

try:
    import sacrebleu

    _SACREBLEU = True
except ImportError:
    sacrebleu = None
    _SACREBLEU = False


# ---------------------------------------------------------------------------
# FLORES_MINI — 20 hardcoded sentence pairs (no external data needed)
# ---------------------------------------------------------------------------

FLORES_MINI: list[dict] = [
    # ── EN -> FR (5 sentences) ────────────────────────────────────────
    {
        "source": "The quick brown fox jumps over the lazy dog near the river bank",
        "reference": "Le rapide renard brun saute par-dessus le chien paresseux pres de la berge",
        "source_lang": "en", "target_lang": "fr", "tag": "en-fr/1",
    },
    {
        "source": "Scientists have discovered a new species of butterfly in the Amazon rainforest",
        "reference": "Les scientifiques ont decouvert une nouvelle espece de papillon dans la foret amazonienne",
        "source_lang": "en", "target_lang": "fr", "tag": "en-fr/2",
    },
    {
        "source": "The European Central Bank announced an interest rate decision yesterday afternoon in Frankfurt",
        "reference": "La Banque centrale europeenne a annonce une decision sur les taux d'interet hier apres-midi a Francfort",
        "source_lang": "en", "target_lang": "fr", "tag": "en-fr/3",
    },
    {
        "source": "My grandmother always used to say that patience is the mother of all virtues",
        "reference": "Ma grand-mere disait toujours que la patience est la mere de toutes les vertus",
        "source_lang": "en", "target_lang": "fr", "tag": "en-fr/4",
    },
    {
        "source": "The temperature in Paris reached thirty eight degrees Celsius during the heat wave last summer",
        "reference": "La temperature a Paris a atteint trente-huit degres Celsius pendant la canicule l'ete dernier",
        "source_lang": "en", "target_lang": "fr", "tag": "en-fr/5",
    },
    # ── FR -> EN (5 sentences) ────────────────────────────────────────
    {
        "source": "Le president de la Republique a prononce un discours devant l'Assemblee nationale",
        "reference": "The President of the Republic delivered a speech before the National Assembly",
        "source_lang": "fr", "target_lang": "en", "tag": "fr-en/1",
    },
    {
        "source": "Les chercheurs ont publie une etude sur les effets du changement climatique",
        "reference": "The researchers published a study on the effects of climate change",
        "source_lang": "fr", "target_lang": "en", "tag": "fr-en/2",
    },
    {
        "source": "Il est important de manger des fruits et des legumes chaque jour pour rester en bonne sante",
        "reference": "It is important to eat fruits and vegetables every day to stay healthy",
        "source_lang": "fr", "target_lang": "en", "tag": "fr-en/3",
    },
    {
        "source": "La Tour Eiffel a ete construite en mille huit cent quatre-vingt-neuf pour l'Exposition universelle",
        "reference": "The Eiffel Tower was built in eighteen eighty-nine for the World's Fair",
        "source_lang": "fr", "target_lang": "en", "tag": "fr-en/4",
    },
    {
        "source": "Le train a grande vitesse relie Paris a Marseille en environ trois heures",
        "reference": "The high-speed train connects Paris to Marseille in about three hours",
        "source_lang": "fr", "target_lang": "en", "tag": "fr-en/5",
    },
    # ── EN -> DE (5 sentences) ────────────────────────────────────────
    {
        "source": "The children played in the park until the sun went down behind the mountains",
        "reference": "Die Kinder spielten im Park bis die Sonne hinter den Bergen unterging",
        "source_lang": "en", "target_lang": "de", "tag": "en-de/1",
    },
    {
        "source": "Germany is the largest economy in Europe and a major exporter of automobiles",
        "reference": "Deutschland ist die groesste Volkswirtschaft in Europa und ein wichtiger Automobilexporteur",
        "source_lang": "en", "target_lang": "de", "tag": "en-de/2",
    },
    {
        "source": "The professor explained the theory of relativity to his students in a very clear manner",
        "reference": "Der Professor erklarte seinen Studenten die Relativitaetstheorie auf sehr klare Weise",
        "source_lang": "en", "target_lang": "de", "tag": "en-de/3",
    },
    {
        "source": "We need to reduce our carbon emissions significantly over the next decade",
        "reference": "Wir muessen unsere Kohlenstoffemissionen im naechsten Jahrzehnt deutlich reduzieren",
        "source_lang": "en", "target_lang": "de", "tag": "en-de/4",
    },
    {
        "source": "The museum exhibition features paintings from the nineteenth century romantic period",
        "reference": "Die Museumsausstellung zeigt Gemaelde aus der Epoche der Romantik des neunzehnten Jahrhunderts",
        "source_lang": "en", "target_lang": "de", "tag": "en-de/5",
    },
    # ── EN -> ZH (5 sentences) ────────────────────────────────────────
    {
        "source": "Artificial intelligence is transforming the way we work and communicate with each other",
        "reference": "\u4eba\u5de5\u667a\u80fd\u6b63\u5728\u6539\u53d8\u6211\u4eec\u5de5\u4f5c\u548c\u4ea4\u6d41\u7684\u65b9\u5f0f",
        "source_lang": "en", "target_lang": "zh", "tag": "en-zh/1",
    },
    {
        "source": "The Great Wall of China is one of the most impressive structures ever built by human civilization",
        "reference": "\u4e07\u91cc\u957f\u57ce\u662f\u4eba\u7c7b\u6587\u660e\u5efa\u9020\u7684\u6700\u4f1f\u5927\u7684\u5efa\u7b51\u4e4b\u4e00",
        "source_lang": "en", "target_lang": "zh", "tag": "en-zh/2",
    },
    {
        "source": "The company reported revenue of three point five billion dollars in the last quarter",
        "reference": "\u8be5\u516c\u53f8\u62a5\u544a\u4e0a\u5b63\u5ea6\u6536\u5165\u4e3a\u4e09\u5341\u4e94\u4ebf\u7f8e\u5143",
        "source_lang": "en", "target_lang": "zh", "tag": "en-zh/3",
    },
    {
        "source": "Climate change is one of the greatest challenges facing humanity in the twenty first century",
        "reference": "\u6c14\u5019\u53d8\u5316\u662f\u4e8c\u5341\u4e00\u4e16\u7eaa\u4eba\u7c7b\u9762\u4e34\u7684\u6700\u5927\u6311\u6218\u4e4b\u4e00",
        "source_lang": "en", "target_lang": "zh", "tag": "en-zh/4",
    },
    {
        "source": "The international space station orbits the earth approximately every ninety minutes",
        "reference": "\u56fd\u9645\u7a7a\u95f4\u7ad9\u5927\u7ea6\u6bcf\u4e5d\u5341\u5206\u949f\u7ed5\u5730\u7403\u8fd0\u884c\u4e00\u5468",
        "source_lang": "en", "target_lang": "zh", "tag": "en-zh/5",
    },
]


# ---------------------------------------------------------------------------
# WebResearchBackend — thin HTTP client for the debug server
# ---------------------------------------------------------------------------


class WebResearchBackend:
    """HTTP client for the NLLW web debug server at /load, /translate, /finish, /reset."""

    def __init__(self, base_url: str = "http://localhost:8777") -> None:
        try:
            import requests as _requests
        except ImportError:
            raise ImportError(
                "WebResearchBackend requires the 'requests' package. "
                "Install with: pip install requests"
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

    def translate(self, text: str) -> dict:
        """Returns full JSON dict: {stable, buffer, source_words, committed_tokens}."""
        r = self._requests.post(
            f"{self.base_url}/translate", json={"text": text}, timeout=30
        )
        return r.json()

    def finish(self) -> dict:
        """Returns full JSON dict: {remaining, full_translation}."""
        r = self._requests.post(f"{self.base_url}/finish", timeout=60)
        return r.json()

    def reset(self) -> None:
        self._requests.post(f"{self.base_url}/reset", timeout=10)

    def set_lang(self, lang: str) -> dict:
        r = self._requests.post(
            f"{self.base_url}/set_lang", json={"lang": lang}, timeout=10
        )
        return r.json()

    def status(self) -> dict:
        r = self._requests.get(f"{self.base_url}/status", timeout=10)
        return r.json()


# ---------------------------------------------------------------------------
# ResearchBenchmark
# ---------------------------------------------------------------------------


class ResearchBenchmark:
    """Research-grade benchmarking for NLLW simultaneous translation.

    Talks to the web debug server (default http://localhost:8777).
    All timing is measured client-side (includes network round-trip,
    matching real-world deployment latency).

    Parameters
    ----------
    base_url : str
        URL of the running NLLW debug server.
    """

    def __init__(self, base_url: str = "http://localhost:8777") -> None:
        self.api = WebResearchBackend(base_url)

    # ------------------------------------------------------------------
    # benchmark_sentence
    # ------------------------------------------------------------------

    def benchmark_sentence(
        self,
        words: list[str],
        *,
        ref: Optional[str] = None,
    ) -> dict:
        """Feed *words* one-by-one, recording per-step timing and committed tokens.

        Returns a rich trace dict:
            steps: list of per-word dicts with:
                word, stable, buffer, committed_tokens, step_time_ms,
                cumulative_committed, new_committed
            sentence: total_time_ms, finish_time_ms, committed_ratio,
                hypothesis, reference, source, num_words,
                committed_before_finish, finish_remaining
        """
        self.api.reset()

        steps: list[dict] = []
        cumulative_stable = ""
        prev_committed = 0
        total_start = time.perf_counter()

        for i, word in enumerate(words):
            chunk = word + " "
            step_start = time.perf_counter()
            data = self.api.translate(chunk)
            step_ms = (time.perf_counter() - step_start) * 1000.0

            stable = data.get("stable", "")
            buffer = data.get("buffer", "")
            committed = data.get("committed_tokens", 0)
            new_committed = max(0, committed - prev_committed)

            cumulative_stable += stable
            prev_committed = committed

            steps.append({
                "word": word,
                "word_index": i,
                "stable": stable,
                "buffer": buffer,
                "committed_tokens": committed,
                "new_committed": new_committed,
                "cumulative_committed": committed,
                "step_time_ms": round(step_ms, 2),
            })

        committed_before_finish = prev_committed

        # Finish: flush remaining translation
        finish_start = time.perf_counter()
        finish_data = self.api.finish()
        finish_ms = (time.perf_counter() - finish_start) * 1000.0

        total_ms = (time.perf_counter() - total_start) * 1000.0

        remaining = finish_data.get("remaining", "")
        full_translation = finish_data.get("full_translation", "")

        # If the server returned the full translation, use it; otherwise reconstruct
        if full_translation:
            hypothesis = full_translation.strip()
        else:
            hypothesis = (cumulative_stable + remaining).strip()

        # Committed ratio: chars committed during streaming vs total
        hyp_len = max(len(hypothesis), 1)
        remaining_len = len(remaining) if remaining else 0
        committed_char_len = max(0, hyp_len - remaining_len)
        committed_ratio = committed_char_len / hyp_len

        return {
            "steps": steps,
            "source": " ".join(words),
            "reference": ref,
            "hypothesis": hypothesis,
            "num_words": len(words),
            "total_time_ms": round(total_ms, 2),
            "finish_time_ms": round(finish_ms, 2),
            "committed_before_finish": committed_before_finish,
            "finish_remaining": remaining,
            "committed_ratio": round(committed_ratio, 4),
        }

    # ------------------------------------------------------------------
    # compute_aware_metrics
    # ------------------------------------------------------------------

    @staticmethod
    def compute_aware_metrics(
        trace: dict,
        speech_rate_wps: float = 2.5,
    ) -> dict:
        """Compute CA-AL (compute-aware average lagging) from a benchmark trace.

        Simulates source words arriving at *speech_rate_wps* words per second.
        Each translate call takes real measured time. The emission time of
        committed tokens from step w is:
            emission_time = max(arrival_time[w], prev_finish_time) + process_time[w]

        Parameters
        ----------
        trace : dict
            Output of benchmark_sentence().
        speech_rate_wps : float
            Source word arrival rate (words per second). Default 2.5
            (typical conversational speech).

        Returns
        -------
        dict
            ca_al_ms: compute-aware average lagging in milliseconds
            standard_al_words: standard average lagging in words
            realtime_ratio: total compute time / total speech time
            total_compute_ms: total wall-clock processing time
            avg_step_ms: average per-step processing time
        """
        steps = trace["steps"]
        num_words = trace["num_words"]
        if not steps or num_words == 0:
            return {
                "ca_al_ms": 0.0,
                "standard_al_words": 0.0,
                "realtime_ratio": 0.0,
                "total_compute_ms": 0.0,
                "avg_step_ms": 0.0,
            }

        word_interval_ms = 1000.0 / speech_rate_wps
        step_times_ms = [s["step_time_ms"] for s in steps]

        # --- Build per-token delays (word index when token was committed) ---
        delays: list[int] = []
        for step in steps:
            new = step["new_committed"]
            for _ in range(new):
                delays.append(step["word_index"])

        # Also account for tokens produced by finish() (assigned to last word)
        finish_tokens = 0
        if trace.get("committed_before_finish") is not None and delays:
            # Approximate: the server doesn't expose post-finish token count
            # directly, but we can infer from committed_ratio
            pass  # finish tokens get delay = last word index (handled below)

        num_tgt_tokens = len(delays)
        if num_tgt_tokens == 0:
            return {
                "ca_al_ms": 0.0,
                "standard_al_words": 0.0,
                "realtime_ratio": 0.0,
                "total_compute_ms": sum(step_times_ms),
                "avg_step_ms": sum(step_times_ms) / len(step_times_ms),
            }

        S = num_words
        T = num_tgt_tokens

        # --- Standard AL (word-level) ---
        mono: list[int] = []
        max_d = 0
        for d in delays:
            max_d = max(max_d, d)
            mono.append(max_d)
        ratio = S / T
        standard_al = sum(max(0, (mono[t] + 1) - t * ratio) for t in range(T)) / T

        # --- Compute-aware timeline ---
        # Source word w arrives at time w / speech_rate_wps
        # Processing of word w starts at max(arrival, prev_finish)
        finish_time_ms = [0.0] * S
        current_time = 0.0
        for w in range(S):
            arrival_ms = w * word_interval_ms
            start_ms = max(arrival_ms, current_time)
            finish_time_ms[w] = start_ms + step_times_ms[w]
            current_time = finish_time_ms[w]

        # Emission time for each target token = finish_time of the source word it was committed at
        emission_times_ms = [finish_time_ms[mono[t]] for t in range(T)]

        # CA-AL: average delay using time-domain AL formula
        total_time_ms = finish_time_ms[-1]
        time_ratio = total_time_ms / T if T > 0 else 0
        ca_al_ms = sum(
            max(0, emission_times_ms[t] - t * time_ratio) for t in range(T)
        ) / T

        speech_duration_ms = S * word_interval_ms
        total_compute = sum(step_times_ms)
        realtime_ratio = total_compute / speech_duration_ms if speech_duration_ms > 0 else 0

        return {
            "ca_al_ms": round(ca_al_ms, 1),
            "standard_al_words": round(standard_al, 3),
            "realtime_ratio": round(realtime_ratio, 3),
            "total_compute_ms": round(total_compute, 1),
            "avg_step_ms": round(total_compute / len(step_times_ms), 1),
        }

    # ------------------------------------------------------------------
    # compare_backends
    # ------------------------------------------------------------------

    def compare_backends(
        self,
        words: list[str],
        configs: list[dict],
        *,
        ref: Optional[str] = None,
        speech_rate_wps: float = 2.5,
    ) -> list[dict]:
        """Run the same sentence through multiple backend configurations.

        Each config dict can contain: border_distance, word_batch,
        source_lang, target_lang, model_path. The server is reconfigured
        via /load for each config.

        Returns a list of result dicts (one per config) with quality +
        latency metrics, ready for tabular display.
        """
        results = []

        for cfg in configs:
            label = cfg.get("label", _config_label(cfg))

            # Reconfigure the server
            load_resp = self.api.load(
                model_path=cfg.get("model_path"),
                source_lang=cfg.get("source_lang", "en"),
                target_lang=cfg.get("target_lang", "fr"),
                border_distance=cfg.get("border_distance", 3),
                word_batch=cfg.get("word_batch", 3),
            )
            if not load_resp.get("ok", True):
                results.append({
                    "label": label,
                    "error": load_resp.get("error", "load failed"),
                })
                continue

            trace = self.benchmark_sentence(words, ref=ref)
            ca = self.compute_aware_metrics(trace, speech_rate_wps=speech_rate_wps)

            # Sentence-level BLEU if reference is available
            bleu = None
            if ref and _SACREBLEU:
                try:
                    bleu = round(
                        sacrebleu.sentence_bleu(
                            trace["hypothesis"], [ref]
                        ).score,
                        1,
                    )
                except Exception:
                    pass

            results.append({
                "label": label,
                "config": {k: v for k, v in cfg.items() if k != "label"},
                "hypothesis": trace["hypothesis"],
                "bleu": bleu,
                "committed_ratio": trace["committed_ratio"],
                "total_time_ms": trace["total_time_ms"],
                "ca_al_ms": ca["ca_al_ms"],
                "standard_al_words": ca["standard_al_words"],
                "realtime_ratio": ca["realtime_ratio"],
                "avg_step_ms": ca["avg_step_ms"],
                "num_words": trace["num_words"],
                "num_steps_with_output": sum(
                    1 for s in trace["steps"] if s["new_committed"] > 0
                ),
            })

        return results

    # ------------------------------------------------------------------
    # run_benchmark_suite
    # ------------------------------------------------------------------

    def run_benchmark_suite(
        self,
        test_cases: list[dict],
        configs: list[dict],
        *,
        speech_rate_wps: float = 2.5,
        verbose: bool = False,
    ) -> dict:
        """Full benchmark: all test cases x all configs.

        Parameters
        ----------
        test_cases : list[dict]
            Each dict must have: source, reference, source_lang, target_lang.
            Optional: tag.
        configs : list[dict]
            Each dict can have: border_distance, word_batch, label, model_path.
            source_lang/target_lang are overridden per test case.
        speech_rate_wps : float
            Simulated speech rate for CA-AL.
        verbose : bool
            Print progress.

        Returns
        -------
        dict with keys:
            configs: list of config result dicts, each containing:
                label, bleu (corpus-level), avg_committed_ratio,
                avg_ca_al_ms, avg_step_ms, sentences (list of per-sentence traces)
            test_cases: the input test cases for reference
        """
        all_results: list[dict] = []

        for cfg_idx, cfg in enumerate(configs):
            label = cfg.get("label", _config_label(cfg))
            if verbose:
                print(f"\n{'='*60}")
                print(f"  Config [{cfg_idx + 1}/{len(configs)}]: {label}")
                print(f"{'='*60}")

            sentences: list[dict] = []
            hypotheses: list[str] = []
            references: list[str] = []
            committed_ratios: list[float] = []
            ca_al_values: list[float] = []
            step_times: list[float] = []
            al_words_values: list[float] = []

            # Group test cases by lang pair to minimize /load calls
            current_pair: Optional[tuple] = None

            for i, tc in enumerate(test_cases):
                src_lang = tc.get("source_lang", "en")
                tgt_lang = tc.get("target_lang", "fr")
                pair = (src_lang, tgt_lang)

                # Reconfigure if lang pair changed
                if pair != current_pair:
                    merged_cfg = dict(cfg)
                    merged_cfg["source_lang"] = src_lang
                    merged_cfg["target_lang"] = tgt_lang
                    load_resp = self.api.load(
                        model_path=merged_cfg.get("model_path"),
                        source_lang=src_lang,
                        target_lang=tgt_lang,
                        border_distance=merged_cfg.get("border_distance", 3),
                        word_batch=merged_cfg.get("word_batch", 3),
                    )
                    if not load_resp.get("ok", True):
                        if verbose:
                            print(f"  SKIP {pair}: {load_resp.get('error')}")
                        continue
                    current_pair = pair

                words = tc["source"].strip().split()
                ref = tc.get("reference")

                trace = self.benchmark_sentence(words, ref=ref)
                ca = self.compute_aware_metrics(trace, speech_rate_wps=speech_rate_wps)

                sentence_result = {
                    "tag": tc.get("tag", f"{src_lang}-{tgt_lang}/{i}"),
                    "source": tc["source"],
                    "reference": ref,
                    "hypothesis": trace["hypothesis"],
                    "committed_ratio": trace["committed_ratio"],
                    "total_time_ms": trace["total_time_ms"],
                    "ca_al_ms": ca["ca_al_ms"],
                    "standard_al_words": ca["standard_al_words"],
                    "avg_step_ms": ca["avg_step_ms"],
                    "num_words": trace["num_words"],
                    "steps": trace["steps"],
                }
                sentences.append(sentence_result)
                hypotheses.append(trace["hypothesis"])
                if ref:
                    references.append(ref)
                committed_ratios.append(trace["committed_ratio"])
                ca_al_values.append(ca["ca_al_ms"])
                al_words_values.append(ca["standard_al_words"])
                step_times.append(ca["avg_step_ms"])

                if verbose:
                    hyp_preview = trace["hypothesis"][:50]
                    if len(trace["hypothesis"]) > 50:
                        hyp_preview += "..."
                    print(
                        f"  [{i + 1}/{len(test_cases)}] "
                        f"{tc.get('tag', ''):<16} "
                        f"CA-AL={ca['ca_al_ms']:>6.0f}ms "
                        f"commit={trace['committed_ratio']:>5.1%} "
                        f"| {hyp_preview}"
                    )

            # Corpus BLEU
            bleu = None
            if references and hypotheses and _SACREBLEU and len(references) == len(hypotheses):
                try:
                    bleu = round(sacrebleu.corpus_bleu(hypotheses, [references]).score, 2)
                except Exception:
                    pass

            n = len(sentences)
            cfg_result = {
                "label": label,
                "config": {k: v for k, v in cfg.items() if k != "label"},
                "num_sentences": n,
                "bleu": bleu,
                "avg_committed_ratio": round(sum(committed_ratios) / n, 4) if n else 0,
                "avg_ca_al_ms": round(sum(ca_al_values) / n, 1) if n else 0,
                "avg_al_words": round(sum(al_words_values) / n, 3) if n else 0,
                "avg_step_ms": round(sum(step_times) / n, 1) if n else 0,
                "sentences": sentences,
            }
            all_results.append(cfg_result)

        return {
            "configs": all_results,
            "num_test_cases": len(test_cases),
            "speech_rate_wps": speech_rate_wps,
        }


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------


def _config_label(cfg: dict) -> str:
    """Generate a short label from a config dict."""
    parts = []
    if "border_distance" in cfg:
        parts.append(f"bd={cfg['border_distance']}")
    if "word_batch" in cfg:
        parts.append(f"wb={cfg['word_batch']}")
    if "model_path" in cfg and cfg["model_path"]:
        import os
        parts.append(os.path.basename(cfg["model_path"]).split(".")[0][:20])
    return " ".join(parts) if parts else "default"


def parse_configs_string(spec: str) -> list[dict]:
    """Parse a CLI config spec like ``"bd=2,3,4 wb=2,3"`` into a list of config dicts.

    Each key=value1,value2,... group defines a parameter with multiple values.
    The full cartesian product of all parameters is returned.

    Examples
    --------
    >>> parse_configs_string("bd=2,3 wb=2,3")
    [
        {"border_distance": 2, "word_batch": 2},
        {"border_distance": 2, "word_batch": 3},
        {"border_distance": 3, "word_batch": 2},
        {"border_distance": 3, "word_batch": 3},
    ]
    """
    ALIASES = {
        "bd": "border_distance",
        "wb": "word_batch",
        "src": "source_lang",
        "tgt": "target_lang",
        "model": "model_path",
    }

    param_grid: dict[str, list] = {}
    for part in spec.strip().split():
        if "=" not in part:
            continue
        key, vals_str = part.split("=", 1)
        key = ALIASES.get(key, key)
        vals: list = []
        for v in vals_str.split(","):
            v = v.strip()
            # Try int, then float, then keep as string
            try:
                vals.append(int(v))
            except ValueError:
                try:
                    vals.append(float(v))
                except ValueError:
                    vals.append(v)
        param_grid[key] = vals

    if not param_grid:
        return [{}]

    keys = list(param_grid.keys())
    value_lists = list(param_grid.values())
    configs = []
    for combo in itertools.product(*value_lists):
        configs.append(dict(zip(keys, combo)))
    return configs


# ---------------------------------------------------------------------------
# ASCII table output
# ---------------------------------------------------------------------------


def print_comparison_table(results: list[dict]) -> None:
    """Print a single-sentence comparison table."""
    if not results:
        print("  No results.")
        return

    print()
    header = (
        f"{'Config':<24} | {'BLEU':>6} | {'Commit%':>8} | "
        f"{'CA-AL(ms)':>9} | {'AL(words)':>9} | {'Time/word':>9} | "
        f"{'RT ratio':>8}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in results:
        if "error" in r:
            print(f"{r['label']:<24} | {'ERROR':>6} | {r['error']}")
            continue

        bleu_str = f"{r['bleu']:>6.1f}" if r.get("bleu") is not None else "   N/A"
        commit_str = f"{r['committed_ratio']:>7.1%}"
        ca_str = f"{r['ca_al_ms']:>9.0f}"
        al_str = f"{r['standard_al_words']:>9.2f}"
        tpw = r["total_time_ms"] / max(r["num_words"], 1)
        tpw_str = f"{tpw:>7.0f}ms"
        rt_str = f"{r['realtime_ratio']:>7.2f}x"
        print(f"{r['label']:<24} | {bleu_str} | {commit_str} | {ca_str} | {al_str} | {tpw_str} | {rt_str}")

    print()


def print_suite_results(suite_results: dict) -> None:
    """Print benchmark suite results as an ASCII table."""
    configs = suite_results.get("configs", [])
    if not configs:
        print("  No results.")
        return

    print()
    print(f"  Benchmark Suite  ({suite_results['num_test_cases']} sentences, "
          f"speech_rate={suite_results['speech_rate_wps']} wps)")
    print()

    header = (
        f"{'Config':<24} | {'BLEU':>6} | {'Commit%':>8} | "
        f"{'CA-AL(ms)':>9} | {'AL(words)':>9} | {'Time/word':>9}"
    )
    sep = "-" * len(header)
    print(header)
    print(sep)

    for r in configs:
        bleu_str = f"{r['bleu']:>6.1f}" if r.get("bleu") is not None else "   N/A"
        commit_str = f"{r['avg_committed_ratio']:>7.1%}"
        ca_str = f"{r['avg_ca_al_ms']:>9.0f}"
        al_str = f"{r['avg_al_words']:>9.2f}"
        tpw_str = f"{r['avg_step_ms']:>7.0f}ms"
        print(f"{r['label']:<24} | {bleu_str} | {commit_str} | {ca_str} | {al_str} | {tpw_str}")

    print()

    # Per-language-pair breakdown if there are multiple
    all_tags = set()
    for r in configs:
        for s in r.get("sentences", []):
            tag = s.get("tag", "")
            pair = tag.split("/")[0] if "/" in tag else tag
            if pair:
                all_tags.add(pair)

    if len(all_tags) > 1:
        print(f"  Per-language-pair breakdown:")
        print()
        for pair in sorted(all_tags):
            print(f"  --- {pair} ---")
            sub_header = f"  {'Config':<22} | {'Commit%':>8} | {'CA-AL(ms)':>9} | {'AL(words)':>9}"
            print(sub_header)
            print(f"  {'-' * (len(sub_header) - 2)}")
            for r in configs:
                pair_sentences = [
                    s for s in r.get("sentences", [])
                    if s.get("tag", "").startswith(pair + "/")
                ]
                if not pair_sentences:
                    continue
                n = len(pair_sentences)
                avg_commit = sum(s["committed_ratio"] for s in pair_sentences) / n
                avg_ca = sum(s["ca_al_ms"] for s in pair_sentences) / n
                avg_al = sum(s["standard_al_words"] for s in pair_sentences) / n
                print(
                    f"  {r['label']:<22} | {avg_commit:>7.1%} | "
                    f"{avg_ca:>9.0f} | {avg_al:>9.2f}"
                )
            print()


def print_trace(trace: dict) -> None:
    """Print a detailed per-step trace for a single sentence."""
    print()
    print(f"  Source:     {trace['source']}")
    if trace.get("reference"):
        print(f"  Reference:  {trace['reference']}")
    print(f"  Hypothesis: {trace['hypothesis']}")
    print(f"  {'-'*64}")

    for step in trace["steps"]:
        parts = []
        if step["stable"]:
            parts.append(f'stable="{step["stable"]}"')
        if step["buffer"]:
            parts.append(f'buf="{step["buffer"]}"')
        parts.append(f"tok={step['committed_tokens']}")
        parts.append(f"+{step['new_committed']}")
        parts.append(f"{step['step_time_ms']:.0f}ms")
        print(f"    +\"{step['word']}\" -> {' | '.join(parts)}")

    print(f"  {'-'*64}")
    print(f"  Finish: \"{trace['finish_remaining']}\" ({trace['finish_time_ms']:.0f}ms)")
    print(f"  Total: {trace['total_time_ms']:.0f}ms | "
          f"Committed: {trace['committed_ratio']:.1%} | "
          f"Words: {trace['num_words']}")
    print()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Research benchmarking for NLLW simultaneous translation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nllw.research                                # default suite, default config\n"
            "  python -m nllw.research --suite flores_mini             # 20-sentence FLORES mini\n"
            "  python -m nllw.research --configs 'bd=2,3,4 wb=2,3'    # parameter sweep\n"
            "  python -m nllw.research --sentence 'The president spoke yesterday'\n"
            "  python -m nllw.research --trace --sentence 'Hello world'\n"
        ),
    )

    parser.add_argument(
        "--url", default="http://localhost:8777",
        help="Web debug server URL (default: http://localhost:8777)",
    )
    parser.add_argument(
        "--suite", default=None, choices=["flores_mini"],
        help="Built-in test suite to run",
    )
    parser.add_argument(
        "--configs", default=None,
        help="Config spec, e.g. 'bd=2,3,4 wb=2,3'. Runs cartesian product.",
    )
    parser.add_argument(
        "--sentence", default=None,
        help="Single sentence to benchmark (instead of a suite)",
    )
    parser.add_argument(
        "--ref", default=None,
        help="Reference translation for --sentence mode",
    )
    parser.add_argument(
        "--lang", default=None,
        help="Language pair filter for suite mode, e.g. 'en-fr'",
    )
    parser.add_argument(
        "--speech-rate", type=float, default=2.5,
        help="Simulated speech rate in words/sec for CA-AL (default: 2.5)",
    )
    parser.add_argument(
        "--trace", action="store_true",
        help="Print detailed per-step trace (for --sentence mode)",
    )
    parser.add_argument(
        "--skip-load", action="store_true",
        help="Skip /load calls (assume server already configured)",
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Save results to JSON file",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Verbose per-sentence output",
    )

    args = parser.parse_args()
    rb = ResearchBenchmark(args.url)

    # --- Single sentence mode ---
    if args.sentence:
        words = args.sentence.strip().split()
        configs = parse_configs_string(args.configs) if args.configs else [{}]

        if len(configs) == 1 and not args.configs:
            # Single config, detailed trace
            if not args.skip_load:
                # Use default config
                pass
            trace = rb.benchmark_sentence(words, ref=args.ref)
            ca = rb.compute_aware_metrics(trace, speech_rate_wps=args.speech_rate)

            if args.trace:
                print_trace(trace)
            print(f"  CA-AL:      {ca['ca_al_ms']:.0f} ms")
            print(f"  AL (words): {ca['standard_al_words']:.2f}")
            print(f"  RT ratio:   {ca['realtime_ratio']:.2f}x")
            print(f"  Avg step:   {ca['avg_step_ms']:.0f} ms")
            print(f"  Committed:  {trace['committed_ratio']:.1%}")
        else:
            # Multi-config comparison
            results = rb.compare_backends(
                words, configs, ref=args.ref,
                speech_rate_wps=args.speech_rate,
            )
            print_comparison_table(results)

        return

    # --- Suite mode ---
    if args.suite == "flores_mini":
        test_cases = list(FLORES_MINI)
    else:
        # Default: use FLORES_MINI
        test_cases = list(FLORES_MINI)

    # Filter by language pair
    if args.lang:
        src, tgt = args.lang.split("-", 1)
        test_cases = [
            tc for tc in test_cases
            if tc.get("source_lang") == src and tc.get("target_lang") == tgt
        ]
        if not test_cases:
            print(f"No test cases for language pair '{args.lang}'.")
            print(f"Available pairs: en-fr, fr-en, en-de, en-zh")
            return

    configs = parse_configs_string(args.configs) if args.configs else [{"border_distance": 3, "word_batch": 3}]

    print(f"NLLW Research Benchmark")
    print(f"  Server:      {args.url}")
    print(f"  Test cases:  {len(test_cases)}")
    print(f"  Configs:     {len(configs)}")
    print(f"  Speech rate: {args.speech_rate} wps")
    if args.lang:
        print(f"  Lang filter: {args.lang}")

    suite_results = rb.run_benchmark_suite(
        test_cases, configs,
        speech_rate_wps=args.speech_rate,
        verbose=args.verbose,
    )

    print_suite_results(suite_results)

    # Save to JSON if requested
    if args.output:
        # Strip per-step data for cleaner JSON output (keep only aggregates)
        save_data = {
            "num_test_cases": suite_results["num_test_cases"],
            "speech_rate_wps": suite_results["speech_rate_wps"],
            "configs": [],
        }
        for cfg_result in suite_results["configs"]:
            cfg_save = {k: v for k, v in cfg_result.items() if k != "sentences"}
            # Keep sentence-level results but strip per-step data
            cfg_save["sentences"] = []
            for s in cfg_result.get("sentences", []):
                s_save = {k: v for k, v in s.items() if k != "steps"}
                cfg_save["sentences"].append(s_save)
            save_data["configs"].append(cfg_save)

        with open(args.output, "w") as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
