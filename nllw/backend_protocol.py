"""SimulMT backend protocol and factory.

All simultaneous translation backends implement the SimulMTBackend ABC.
The create_backend() factory instantiates the right backend from a config dict.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class TranslationStep:
    """Result of a single translate() call."""
    text: str                    # New committed text (may be empty if border hit early)
    is_final: bool = False       # True if this completes a sentence segment
    committed_tokens: int = 0    # Number of new tokens committed
    stopped_at_border: bool = False  # True if AlignAtt border was hit
    source_words_seen: int = 0   # Number of source words processed so far
    generation_time_ms: float = 0.0  # Wall-clock time for this step
    # Batch first emission time: ASR time of the FIRST word in this batch.
    # For correct LongYAAL: attribute output to when batch started, not completed.
    # None means "use the emission_time from the current call" (default).
    batch_first_emission_time: Optional[float] = None
    # Average log-probability of generated tokens (quality diagnostic).
    # Higher = more confident generation. None if no tokens generated.
    avg_logprob: Optional[float] = None


@dataclass
class BackendConfig:
    """Configuration for a SimulMT backend."""
    backend_type: str = "alignatt"
    model_path: str = ""
    heads_path: str = ""
    prompt_format: str = "hymt"
    direction: str = "en-zh"
    # AlignAtt parameters
    border_distance: int = 3
    word_batch: int = 3
    top_k_heads: int = 10
    # Context
    context_sentences: int = 0
    # Generation
    max_new_per_step: int = 50
    n_ctx: int = 2048
    # KV cache
    use_kvcache: bool = True
    # Entropy veto
    entropy_veto_threshold: Optional[float] = None
    # Attention aggregation method (ts_vote, softmax_mean, entropy_weighted, consensus, geomean, top_p)
    aggregation: str = "ts_vote"
    # Top-p threshold for the top_p aggregation method (0.0-1.0)
    # Controls what fraction of attention mass defines the "frontier".
    # Lower = tighter (closer to argmax), higher = broader (further right frontier).
    # Default 0.8. Range: 0.5-0.95.
    top_p_threshold: float = 0.8
    # Adaptive top_p threshold: dynamically adjust p_threshold per-sentence
    # based on source text complexity. Simple sentences use lower threshold
    # (faster latency, still good quality), complex sentences use higher
    # (more conservative, safer). Range mapped from complexity score:
    # simple -> base - 0.1, complex -> base + 0.1, capped at [0.5, 0.95].
    # Novel: no published work on adaptive aggregation thresholds for SimulMT.
    # False=disabled, True=enabled (adjusts around top_p_threshold as baseline).
    adaptive_top_p: bool = False
    # Dynamic border distance (adjusts per-token based on attention entropy)
    dynamic_border: bool = False
    # SSBD (Self-Speculative Biased Decoding) for LA backend
    # None = disabled, 0.0 = pure speculative (no bias), 0.2 = recommended
    ssbd_beta: Optional[float] = None
    # Display-only mask-k: hide last k tokens from display but keep as draft
    # Reduces output flicker (NE) while maintaining SSBD speedup. 0 = disabled.
    display_mask_k: int = 0
    # LA forced decoding: force-decode committed prefix before generating.
    # Instead of generating from scratch, conditions on committed tokens.
    # Reduces computation (only generate new tokens) and improves consistency.
    # CUNI approach (Polak et al., 2025). False = standard re-translation.
    la_forced_decode: bool = False
    # Adaptive SSBD beta: dynamically adjust beta based on model entropy.
    # When True and ssbd_beta is set, beta scales per-token:
    #   confident (low entropy) -> higher beta (more lenient)
    #   uncertain (high entropy) -> lower beta (stricter)
    # Combines SSBD (2509.21740) with entropy-modulated confidence (2508.15371).
    adaptive_ssbd: bool = False
    # LA Two-Pass Catch-up: run two re-translations per update, keep the one
    # with the longer stable prefix (more consistent with previous output).
    # CUNI approach: extra pass catches instability from attention drift.
    # Trades 2x compute for better output stability (lower NE).
    la_two_pass: bool = False
    # Adaptive Multi-Strategy (AMS): auto-select aggregation method per token
    # based on attention patterns. When True, overrides the 'aggregation' field.
    # Selects among ts_vote, entropy_weighted, consensus, geomean based on:
    #   - head agreement ratio (high -> ts_vote, low -> consensus)
    #   - attention entropy (low -> geomean, high -> entropy_weighted)
    adaptive_aggregation: bool = False
    # Per-head temperature normalization: normalize attention sharpness per head
    # before aggregation. Heads with naturally sharper distributions are scaled
    # to match heads with broader distributions, ensuring fair weighting.
    # Learned during head detection; at runtime uses a fixed reference entropy.
    head_temp_normalize: bool = False
    head_temp_reference: float = 1.5  # Reference entropy (nats) for normalization
    # Dynamic word_batch: adjust wb based on source sentence length.
    # Short sentences (< 8 words) -> wb - 1, long (> 20) -> wb + 1.
    # Gives faster latency on short inputs and safer quality on long ones.
    dynamic_word_batch: bool = False
    # Source-aware word batching: defer translation if the batch ends on a
    # function word (the, of, in, etc.). Ensures translation units don't
    # break mid-phrase. Allows up to max_defer extra words per batch.
    source_aware_batching: bool = False
    # Attention information gain: use KL divergence between consecutive
    # attention snapshots as secondary border signal. Large divergence = new
    # source info being processed, keep generating. Small divergence = source
    # exhausted, reinforce border stop. Threshold in nats.
    info_gain_threshold: Optional[float] = None  # None=disabled, 0.3=recommended
    # Shift-k adaptive border: shift border_distance per-token based on
    # cross-attention mass in the border region. Inspired by DrFrattn (EMNLP 2025).
    # When attention mass in border region is above this threshold, trigger stop.
    # None=disabled, overrides standard border check when set.
    shift_k_threshold: Optional[float] = None  # None=disabled, 0.4=recommended
    # Border confirmation: require N consecutive border hits before stopping.
    # Prevents false positive stops from transient attention patterns.
    # 1 = standard (stop on first hit), 2 = require 2 consecutive hits, etc.
    border_confirm: int = 1  # 1=disabled, 2=recommended for high quality
    # LSG logit KL divergence (arxiv 2501.00868, AAAI 2025):
    # Compare output logit distributions with full source vs reduced source
    # (last lsg_k source tokens removed via KV cache fork).
    # Low KL = source is exhausted -> reinforce border stop (WRITE).
    # High KL = source still matters -> inhibit border stop (READ more).
    # None=disabled, 7.0=recommended for 7B models. Range: 5.0-9.0.
    lsg_kl_threshold: Optional[float] = None
    # Number of source tokens to remove for the reduced-source probe.
    # Larger k tests more "distant" source dependency.
    # 3=default (matches word_batch), range: 1-5.
    lsg_k: int = 3
    # Complexity-adaptive parameters: estimate source sentence complexity from
    # text features (word count, avg length, numeral density, subword ratio)
    # and adjust border_distance, word_batch, and generation cap per-sentence.
    # Simple sentences -> aggressive (smaller bd/wb), complex -> conservative.
    complexity_adaptive: bool = False
    # Entropy change tracking (REINA-inspired, arxiv 2508.04946, AAAI 2026):
    # Track generation entropy across consecutive translate() calls. If adding
    # a new source word significantly reduces entropy (entropy_change < threshold),
    # the model is still learning from source -> inhibit border stop (READ more).
    # None=disabled, -0.5=recommended. Negative values: larger drop needed to inhibit.
    entropy_change_threshold: Optional[float] = None
    # Prediction stability tracking (novel): measure how much the model's top
    # predictions change between consecutive translate() calls. Stable predictions
    # = model has enough source context (supports WRITE). Volatile predictions
    # = model still adapting (supports READ). Combined with attention border check.
    # False=disabled, True=enabled.
    prediction_stability: bool = False
    # Source coverage guard (novel): track what fraction of source positions
    # receive significant attention from alignment heads. If coverage drops
    # below threshold, force stop (hallucination prevention). When the model
    # hallucinates, it attends to a narrow source region or ignores source entirely.
    # None=disabled, 0.3=recommended. Range: 0.1-0.5.
    coverage_threshold: Optional[float] = None
    # Attention monotonicity (novel): track how monotonically attention moves
    # through source during generation. Monotonic attention = straightforward
    # translation -> tighter border. Non-monotonic = reordering -> wider border.
    # False=disabled, True=enabled.
    attention_monotonicity: bool = False
    # N-gram repetition detection (novel): detect degenerate repetitive loops
    # during generation and force halt. When LLMs hallucinate, they often enter
    # repetitive patterns. This catches them early, preventing wasted compute
    # and garbage output. Orthogonal to attention-based signals (output-space).
    # None=disabled, 2=recommended (halt after 2 repeats of any 2-4gram).
    repetition_max_repeats: Optional[int] = None
    # Attention shift tracking (novel): cross-step, input-space signal.
    # Measures how much the model's source focus moved after adding a new
    # source word. Large shift = consuming source (WRITE). Small = stuck (READ).
    # False=disabled, True=enabled.
    attention_shift: bool = False
    # Language-pair-aware generation cap: adjust max tokens per step based
    # on the known source/target compression ratio. EN->ZH produces fewer
    # tokens (compact), EN->DE produces more (compounds). Prevents both
    # overgeneration and undergeneration. False=use fixed max_new_per_step.
    language_pair_gen_cap: bool = False
    # Confidence-adaptive word batching (novel): adjust effective word_batch
    # based on generation confidence from the PREVIOUS translate() call.
    # Uses avg_logprob (iter 22): confident generation -> reduce wb by 1
    # (faster emission, lower YAAL), uncertain -> increase wb by 1 (more
    # source context, better quality). No published work on confidence-based
    # batch size adaptation for SimulMT.
    # False=disabled, True=enabled.
    confidence_adaptive_wb: bool = False
    # Log-probability threshold for "confident" (reduce wb). Above this = wb-1.
    # Range: -2.0 to 0.0. Higher (closer to 0) = more selective.
    confidence_wb_high: float = -0.5
    # Log-probability threshold for "uncertain" (increase wb). Below this = wb+1.
    # Range: -5.0 to -1.0. Lower = more selective.
    confidence_wb_low: float = -2.0
    # Signal fusion mode: replace boolean cascade with weighted scoring.
    # When enabled, all signals produce continuous scores in [-1, +1] and
    # a weighted sum determines the border decision. Per-direction tunable.
    # False=disabled (use check_border_combined), True=use fused_border_check.
    signal_fusion: bool = False
    # Fusion decision threshold: fusion_score >= threshold -> WRITE (border hit).
    # 0.0 = balanced, positive = more conservative (fewer WRITE), negative = aggressive.
    fusion_threshold: float = 0.0
    # Entropy-gated top_p (novel): per-token top_p threshold modulation
    # based on the merged attention distribution's entropy during generation.
    # Focused attention (low entropy) -> lower threshold -> emit sooner (lower YAAL).
    # Spread attention (high entropy) -> higher threshold -> wait longer (better quality).
    # Different from adaptive_top_p (per-sentence, source complexity) and
    # entropy_veto (halts generation). This is per-TOKEN within the generation loop.
    # Only effective when aggregation="top_p". False=disabled, True=enabled.
    entropy_gated_top_p: bool = False
    # Perplexity-based adaptive border (Hibiki-inspired, novel for AlignAtt):
    # Adjust border_distance per translate() call based on generation confidence.
    # After each word batch, compute average perplexity from generation logits.
    # Low perplexity = model confident = tighten border (bd-1 or bd-2) for
    # lower latency. High perplexity = model uncertain = widen border (bd+1)
    # for better quality. Unlike entropy veto (which halts generation, a dead
    # end), this adjusts the READ/WRITE policy between steps.
    # False=disabled, True=enabled.
    perplexity_adaptive_bd: bool = False
    # Perplexity threshold for "confident" (tighten border). Below this = bd-1.
    # Range: 1.0-4.0. Lower = more selective (fewer bd reductions).
    perplexity_bd_low: float = 2.0
    # Perplexity threshold for "uncertain" (widen border). Above this = bd+1.
    # Range: 3.0-8.0.
    perplexity_bd_high: float = 5.0
    # Wait-k policy
    wait_k: int = 5
    # Target language (for output validation)
    target_lang: str = "zh"
    # Generation temperature: controls randomness in token selection.
    # 0.0 = pure greedy (argmax, default). Values 0.1-0.3 add slight
    # stochasticity that can help escape suboptimal greedy paths.
    # Research shows low-temperature sampling can improve MT quality
    # by exploring alternative translations the greedy path misses.
    # Novel for AlignAtt SimulMT: no published work on temperature
    # effects in attention-based simultaneous translation.
    # 0.0=greedy (default), 0.1-0.3=recommended for quality exploration.
    generation_temperature: float = 0.0
    # Entropy-based dynamic temperature (EDT, arxiv 2403.14541):
    # Instead of fixed temperature, dynamically adjust per token.
    # Confident tokens (low entropy) -> low temperature (near-greedy).
    # Uncertain tokens (high entropy) -> higher temperature (explore).
    # More principled than fixed temperature. Overrides generation_temperature.
    # False=disabled (use fixed temperature), True=use EDT.
    entropy_dynamic_temperature: bool = False
    # Confidence-gated token trimming (novel): after generation stops
    # (border hit or max tokens), trim trailing tokens that have
    # per-token logprob below a threshold. Prevents committing
    # low-confidence trailing tokens that may be hallucinated.
    # Especially important for XCOMET-XL which penalizes semantic errors.
    # None=disabled (commit all generated tokens).
    # -3.0=recommended (trim tokens with logprob < -3.0).
    confidence_trim_threshold: Optional[float] = None
    # Anti-LM contrastive decoding (Sia et al., NAACL 2024, arxiv 2311.08324):
    # Subtract source-language continuation penalty from translation logits.
    # Prevents hallucination and source copying by penalizing tokens that the
    # model would generate as source-language continuation rather than translation.
    # The anti-LM logits come from a single forward pass on source text only
    # (no translation instructions), computed once per translate() call.
    # False=disabled, True=enabled.
    anti_lm: bool = False
    # Decay rate for Anti-LM penalty. Penalty = gamma^step (strongest at first
    # generated token, decays exponentially). 0.3 recommended by Sia et al.
    # Lower values = faster decay. Range: 0.1-1.0.
    anti_lm_gamma: float = 0.3
    # Sentence-final refinement (novel): when is_final=True, discard partial
    # translations and regenerate from scratch with full source. During SimulMT,
    # partial translations committed word-by-word may lock in suboptimal prefixes.
    # Refinement re-translates the full sentence without this constraint, yielding
    # quality closer to the full-sentence baseline while keeping SimulMT latency
    # for intermediate steps. Only affects the FINAL output per segment.
    # False=disabled (continue from committed prefix, default).
    # True=re-translate from scratch on is_final.
    final_refinement: bool = False
    # GPU offloading: number of layers to offload. 0=CPU, 99=all layers.
    n_gpu_layers: int = 0

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BackendConfig":
        """Create config from a dict, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        return cls(**{k: v for k, v in d.items() if k in valid_keys})


class SimulMTBackend(ABC):
    """Abstract base class for simultaneous MT backends.

    Lifecycle:
        1. __init__(config) -- load model, set up state
        2. translate(word) -- called for each new source word
        3. reset() -- called at sentence boundaries
        4. close() -- free resources
    """

    def __init__(self, config: BackendConfig):
        self.config = config
        self._trace_collector = None

    @abstractmethod
    def translate(self, source_word: str, is_final: bool = False,
                  emission_time: float = 0.0) -> TranslationStep:
        """Process a new source word and return any new translation.

        Args:
            source_word: The new source word to add
            is_final: True if this word ends the current sentence/segment
            emission_time: ASR emission timestamp (for latency metrics)

        Returns:
            TranslationStep with new committed text
        """
        ...

    @abstractmethod
    def reset(self):
        """Reset state for next sentence segment.

        Called after is_final=True translation completes.
        Should preserve context if context_sentences > 0.
        """
        ...

    def close(self):
        """Free resources (model, context, etc.). Override if needed."""
        pass

    def get_full_translation(self) -> str:
        """Get the full committed translation so far."""
        return ""

    def set_trace_collector(self, collector) -> None:
        """Set a trace collector for fusion calibration.

        The collector's record_step() will be called at each border
        decision point with signal scores. Pass None to disable.
        """
        self._trace_collector = collector

    @property
    def name(self) -> str:
        return self.config.backend_type


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------
_BACKEND_REGISTRY: Dict[str, type] = {}


def register_backend(name: str):
    """Decorator to register a backend class."""
    def decorator(cls):
        _BACKEND_REGISTRY[name] = cls
        return cls
    return decorator


def _ensure_backends_imported():
    """Import backend modules so their @register_backend decorators fire."""
    if _BACKEND_REGISTRY:
        return  # already populated
    import importlib
    for mod_name in [
        "nllw.alignatt_backend",
        "nllw.alignatt_la_backend",
        "nllw.baselines",
    ]:
        try:
            importlib.import_module(mod_name)
        except Exception:
            pass


def create_backend(config: BackendConfig) -> SimulMTBackend:
    """Create a backend from config.

    Registered types:
        - "alignatt": AlignAtt border detection (primary)
        - "alignatt-kv": AlignAtt with KV cache (speed)
        - "wait-k": Wait-k baseline
        - "full-sentence": Full sentence baseline (quality upper bound)
        - "eager": Eager baseline (latency lower bound)
    """
    _ensure_backends_imported()
    backend_type = config.backend_type
    if backend_type not in _BACKEND_REGISTRY:
        available = ", ".join(sorted(_BACKEND_REGISTRY.keys()))
        raise ValueError(
            f"Unknown backend type '{backend_type}'. Available: {available}"
        )
    return _BACKEND_REGISTRY[backend_type](config)


def list_backends() -> List[str]:
    """List registered backend type names."""
    return sorted(_BACKEND_REGISTRY.keys())
