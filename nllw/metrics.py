"""
SimulMT latency metrics.

Implements standard metrics for evaluating simultaneous translation systems:
- Average Lagging (AL): How far behind the translation is on average
- Length-Adaptive Average Lagging (LAAL / LongYAAL): Normalized for variable S/T ratio
- Average Proportion (AP): Average source coverage at each output step
- Maximum Consecutive Wait (MaxCW): Longest source word sequence without output

Reference: Ma et al. "STACL: Simultaneous Translation with Implicit Anticipation
and Controllable Latency using Prefix-to-Prefix Framework" (ACL 2019)
"""

from dataclasses import dataclass
from typing import List


@dataclass
class LatencyMetrics:
    """Container for SimulMT latency metrics."""
    al: float = 0.0       # Average Lagging
    laal: float = 0.0     # Length-Adaptive AL (also called LongYAAL)
    ap: float = 0.0       # Average Proportion
    max_cw: int = 0       # Maximum Consecutive Wait

    def to_dict(self) -> dict:
        return {
            "al": round(self.al, 3),
            "laal": round(self.laal, 3),
            "ap": round(self.ap, 3),
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

    # Average Lagging (AL)
    # AL = (1/T) * sum_{t=0}^{T-1} max(0, g(t) + 1 - t * S/T)
    ratio = S / T
    total_lag = sum(max(0, (mono[t] + 1) - t * ratio) for t in range(T))
    al = total_lag / T

    # Length-Adaptive Average Lagging (LAAL)
    # LAAL = AL * tau, where tau = min(S/T, 1.0)
    tau = min(S / T, 1.0)
    laal = al * tau

    # Average Proportion (AP)
    # AP = (1/(S*T)) * sum_{t=0}^{T-1} (g(t) + 1)
    total_src_read = sum(mono[t] + 1 for t in range(T))
    ap = total_src_read / (S * T)

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
        ap=ap,
        max_cw=max_cw,
    )


def compute_average_lagging_ms(
    emission_times: List[float],
    word_durations: List[float],
    total_audio_duration: float,
) -> float:
    """Compute Average Lagging in milliseconds for speech translation.

    This is the time-domain version used in IWSLT evaluations (LongYAAL).

    Args:
        emission_times: Wall-clock time when each output segment was emitted.
        word_durations: Duration of each source word/segment in seconds.
        total_audio_duration: Total audio duration in seconds.

    Returns:
        Average lagging in milliseconds.
    """
    if not emission_times or not word_durations:
        return 0.0

    T = len(emission_times)
    # Compute cumulative ideal emission times (evenly spaced)
    ideal_times = [i * total_audio_duration / T for i in range(T)]

    total_lag = sum(
        max(0, emission_times[t] - ideal_times[t])
        for t in range(T)
    )
    return (total_lag / T) * 1000  # Convert to ms
