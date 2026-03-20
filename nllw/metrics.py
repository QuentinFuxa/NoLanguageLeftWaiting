"""
SimulMT latency metrics.

Implements standard metrics for evaluating simultaneous translation systems:
- Average Lagging (AL): How far behind the translation is on average
- Length-Adaptive Average Lagging (LAAL): Normalized for variable S/T ratio
- YAAL: Yet Another Average Lagging (IWSLT 2026 primary latency metric)
- Average Proportion (AP): Average source coverage at each output step
- Maximum Consecutive Wait (MaxCW): Longest source word sequence without output
- Differentiable Average Lagging (DAL): Smooth version of AL

References:
- Ma et al. "STACL" (ACL 2019) -- AL, AP
- Cherry & Foster (2019) -- DAL
- Polák et al. "Better Late Than Never" (https://arxiv.org/abs/2509.17349) -- YAAL
- OmniSTEval (https://github.com/pe-trik/OmniSTEval) -- reference implementation
"""

from dataclasses import dataclass
from typing import List, Optional


@dataclass
class LatencyMetrics:
    """Container for SimulMT latency metrics."""
    al: float = 0.0       # Average Lagging
    laal: float = 0.0     # Length-Adaptive AL
    yaal: float = 0.0     # Yet Another Average Lagging (IWSLT 2026 primary)
    ap: float = 0.0       # Average Proportion
    dal: float = 0.0      # Differentiable Average Lagging
    max_cw: int = 0       # Maximum Consecutive Wait

    def to_dict(self) -> dict:
        return {
            "al": round(self.al, 3),
            "laal": round(self.laal, 3),
            "yaal": round(self.yaal, 3),
            "ap": round(self.ap, 3),
            "dal": round(self.dal, 3),
            "max_cw": self.max_cw,
        }


def compute_latency_metrics(
    delays: List[int],
    num_src_words: int,
    num_tgt_tokens: int,
) -> LatencyMetrics:
    """Compute SimulMT latency metrics from delay sequence.

    Args:
        delays: For each target token t, the source word index that was read
                when token t was generated (0-indexed).
        num_src_words: Total number of source words (S).
        num_tgt_tokens: Total number of target tokens (T).

    Returns:
        LatencyMetrics with AL, LAAL, AP, MaxCW.
    """
    if not delays or num_tgt_tokens == 0 or num_src_words == 0:
        return LatencyMetrics()

    T = num_tgt_tokens
    S = num_src_words

    # Monotonize delays: g(t) = max(g(t-1), d(t))
    mono = []
    max_d = 0
    for d in delays:
        max_d = max(max_d, d)
        mono.append(max_d)

    # Average Lagging (AL) -- Ma et al. 2019
    # AL = (1/tau) * sum_{t=0}^{tau-1} (g(t) - t * S/T)
    # where tau = first t where g(t) >= S, or T if none
    gamma_al = T / S
    al_sum = 0.0
    al_tau = 0
    for t in range(T):
        al_sum += mono[t] - t / gamma_al
        al_tau = t + 1
        if mono[t] >= S:
            break
    al = al_sum / al_tau if al_tau > 0 else 0.0

    # LAAL -- uses max(|delays|, T) / S as gamma
    gamma_laal = max(len(delays), T) / S
    laal_sum = 0.0
    laal_tau = 0
    for t in range(len(delays)):
        laal_sum += delays[t] - t / gamma_laal
        laal_tau = t + 1
        if delays[t] >= S:
            break
    laal = laal_sum / laal_tau if laal_tau > 0 else 0.0

    # YAAL (Yet Another Average Lagging) -- Polák et al. 2025
    # Same formula as LAAL but using raw (non-monotonized) delays
    # and stopping when d >= source_length
    # This is the IWSLT 2026 primary latency metric
    gamma_yaal = max(len(delays), T) / S
    yaal_sum = 0.0
    yaal_tau = 0
    for t in range(len(delays)):
        if delays[t] >= S:
            break
        yaal_sum += delays[t] - t / gamma_yaal
        yaal_tau = t + 1
    yaal = yaal_sum / yaal_tau if yaal_tau > 0 else 0.0

    # Average Proportion (AP)
    # AP = (1/(S*T)) * sum_{t=0}^{T-1} (g(t) + 1)
    total_src_read = sum(mono[t] + 1 for t in range(T))
    ap = total_src_read / (S * T)

    # Differentiable Average Lagging (DAL) -- Cherry & Foster 2019
    gamma_dal = T / S
    dal_sum = 0.0
    g_prime_last = 0.0
    for t in range(T):
        if t == 0:
            g_prime = float(mono[t])
        else:
            g_prime = max(float(mono[t]), g_prime_last + 1.0 / gamma_dal)
        dal_sum += g_prime - t / gamma_dal
        g_prime_last = g_prime
    dal = dal_sum / T if T > 0 else 0.0

    # Maximum Consecutive Wait (MaxCW)
    # Longest gap between consecutive monotonized delays
    max_cw = mono[0] + 1  # initial wait
    for t in range(1, T):
        gap = mono[t] - mono[t - 1]
        if gap > 0:
            max_cw = max(max_cw, gap)

    return LatencyMetrics(
        al=al,
        laal=laal,
        yaal=yaal,
        ap=ap,
        dal=dal,
        max_cw=max_cw,
    )


def compute_yaal_ms(
    emission_times_ms: List[float],
    source_length_ms: float,
    reference_length: Optional[int] = None,
    is_longform: bool = True,
    recording_end_ms: Optional[float] = None,
) -> float:
    """Compute YAAL in milliseconds for speech translation.

    Implements the time-domain YAAL metric used in IWSLT 2026 evaluation
    (computation-unaware variant). Compatible with OmniSTEval's YAALScorer.

    In shortform mode, stops at segment boundary (d >= source_length_ms).
    In longform mode, continues past segment boundary but stops at recording end.
    This matches the LongYAAL behavior described in Polák et al. (2025).

    Args:
        emission_times_ms: Emission timestamps in ms (one per output segment/word).
        source_length_ms: Source segment duration in ms.
        reference_length: Number of reference tokens/words. If None, uses len(emission_times_ms).
        is_longform: If True, don't stop at segment boundary (stop at recording_end instead).
        recording_end_ms: End of full recording in ms (longform only). If None, no stream cutoff.

    Returns:
        YAAL in milliseconds.
    """
    if not emission_times_ms or source_length_ms <= 0:
        return 0.0

    delays = emission_times_ms
    T = reference_length if reference_length is not None else len(delays)
    gamma = max(len(delays), T) / source_length_ms
    rec_end = recording_end_ms if recording_end_ms is not None else float("inf")

    yaal_sum = 0.0
    tau = 0
    for t, d in enumerate(delays):
        if (not is_longform and d >= source_length_ms) or d >= rec_end:
            break
        yaal_sum += d - t / gamma
        tau = t + 1

    return yaal_sum / tau if tau > 0 else 0.0


def compute_average_lagging_ms(
    emission_times: List[float],
    word_durations: List[float],
    total_audio_duration: float,
) -> float:
    """Compute Average Lagging in milliseconds for speech translation.

    Legacy function. For IWSLT 2026 evaluation, use compute_yaal_ms() instead.

    Args:
        emission_times: Wall-clock time when each output segment was emitted (seconds).
        word_durations: Duration of each source word/segment in seconds.
        total_audio_duration: Total audio duration in seconds.

    Returns:
        Average lagging in milliseconds.
    """
    if not emission_times or not word_durations:
        return 0.0

    T = len(emission_times)
    ideal_times = [i * total_audio_duration / T for i in range(T)]

    total_lag = sum(
        max(0, emission_times[t] - ideal_times[t])
        for t in range(T)
    )
    return (total_lag / T) * 1000  # Convert to ms
