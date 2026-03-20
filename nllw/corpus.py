"""Categorized test corpus for SimulMT evaluation.

Provides a curated set of test sentences organized by difficulty level
and linguistic phenomena. Useful for quick sanity checks and targeted
debugging of translation quality.

Categories:
    - simple: Short, straightforward sentences (5-10 words)
    - medium: Standard conference talk sentences (10-20 words)
    - complex: Long, nested, or ambiguous sentences (20+ words)
    - numbers: Sentences with dates, numbers, statistics
    - names: Proper nouns, organization names
    - idioms: Idiomatic expressions, metaphors
    - technical: Domain-specific terminology
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


@dataclass
class TestSentence:
    """A categorized test sentence with optional reference translation."""
    text: str
    category: str
    direction: str = "en-fr"  # default
    reference: Optional[str] = None
    notes: str = ""


# ---------------------------------------------------------------------------
# EN -> FR corpus
# ---------------------------------------------------------------------------
_EN_FR_SENTENCES = [
    # Simple
    TestSentence("Hello, how are you today?", "simple", "en-fr",
                 "Bonjour, comment allez-vous aujourd'hui ?"),
    TestSentence("The weather is beautiful this morning.", "simple", "en-fr",
                 "Le temps est magnifique ce matin."),
    TestSentence("Thank you very much for your help.", "simple", "en-fr",
                 "Merci beaucoup pour votre aide."),
    TestSentence("I would like a cup of coffee please.", "simple", "en-fr",
                 "Je voudrais une tasse de cafe s'il vous plait."),
    TestSentence("The train arrives at three o'clock.", "simple", "en-fr",
                 "Le train arrive a trois heures."),

    # Medium
    TestSentence(
        "The president of France announced new economic reforms during yesterday's press conference.",
        "medium", "en-fr",
    ),
    TestSentence(
        "Researchers at the university have discovered a promising new treatment for Alzheimer's disease.",
        "medium", "en-fr",
    ),
    TestSentence(
        "The European Central Bank is expected to raise interest rates in the coming months.",
        "medium", "en-fr",
    ),
    TestSentence(
        "Climate change continues to accelerate, with global temperatures rising faster than predicted.",
        "medium", "en-fr",
    ),
    TestSentence(
        "The new artificial intelligence system can translate languages in real time with unprecedented accuracy.",
        "medium", "en-fr",
    ),

    # Complex
    TestSentence(
        "Despite the significant challenges posed by the ongoing pandemic, the international community "
        "has managed to accelerate vaccine development at a pace that would have been unimaginable "
        "just a decade ago.",
        "complex", "en-fr",
    ),
    TestSentence(
        "The relationship between economic growth and environmental sustainability remains one of "
        "the most contentious debates in modern policy making, with experts on both sides presenting "
        "compelling but contradictory evidence.",
        "complex", "en-fr",
    ),
    TestSentence(
        "While the initial results of the clinical trial were promising, further analysis revealed "
        "that the treatment's efficacy varied significantly across different demographic groups, "
        "raising questions about its universal applicability.",
        "complex", "en-fr",
    ),

    # Numbers
    TestSentence("The company reported revenues of 4.2 billion dollars in the third quarter of 2025.",
                 "numbers", "en-fr"),
    TestSentence("The population of the city has grown by 15.3 percent since the last census in 2020.",
                 "numbers", "en-fr"),
    TestSentence("The spacecraft traveled 384,400 kilometers to reach the moon in approximately 3 days.",
                 "numbers", "en-fr"),

    # Names
    TestSentence("Emmanuel Macron met with Chancellor Scholz at the Elysee Palace.",
                 "names", "en-fr"),
    TestSentence("The World Health Organization issued new guidelines on Monday.",
                 "names", "en-fr"),
    TestSentence("Apple, Google, and Microsoft are competing in the artificial intelligence market.",
                 "names", "en-fr"),

    # Technical
    TestSentence(
        "The transformer architecture uses self-attention mechanisms to process sequential data "
        "in parallel, significantly reducing training time compared to recurrent neural networks.",
        "technical", "en-fr",
    ),
    TestSentence(
        "Quantum computing leverages superposition and entanglement to perform calculations "
        "that would take classical computers thousands of years.",
        "technical", "en-fr",
    ),
]

# ---------------------------------------------------------------------------
# EN -> ZH corpus
# ---------------------------------------------------------------------------
_EN_ZH_SENTENCES = [
    TestSentence("Hello, how are you today?", "simple", "en-zh", "\u4f60\u597d\uff0c\u4eca\u5929\u4f60\u600e\u4e48\u6837\uff1f"),
    TestSentence("Thank you very much for your help.", "simple", "en-zh", "\u975e\u5e38\u611f\u8c22\u60a8\u7684\u5e2e\u52a9\u3002"),
    TestSentence(
        "The president of the United States announced new trade policies with China.",
        "medium", "en-zh",
    ),
    TestSentence(
        "Artificial intelligence is transforming every aspect of modern life, "
        "from healthcare to transportation.",
        "medium", "en-zh",
    ),
    TestSentence(
        "The simultaneous translation system processes audio in real time, "
        "detecting speech boundaries and generating translations incrementally "
        "as the speaker continues talking.",
        "complex", "en-zh",
    ),
    TestSentence("GDP growth reached 5.2 percent in the first quarter of 2025.",
                 "numbers", "en-zh"),
    TestSentence("Tsinghua University and MIT announced a joint research program.",
                 "names", "en-zh"),
    TestSentence(
        "The attention mechanism allows the model to focus on relevant parts "
        "of the input sequence when generating each output token.",
        "technical", "en-zh",
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_CORPUS_REGISTRY: Dict[str, List[TestSentence]] = {
    "en-fr": _EN_FR_SENTENCES,
    "en-zh": _EN_ZH_SENTENCES,
}


def get_corpus(
    direction: str = "en-fr",
    categories: Optional[List[str]] = None,
    n: Optional[int] = None,
) -> List[TestSentence]:
    """Get test sentences for a direction, optionally filtered by category.

    Args:
        direction: Language direction (e.g. "en-fr", "en-zh")
        categories: Filter to these categories (None = all)
        n: Max number of sentences

    Returns:
        List of TestSentence
    """
    sentences = _CORPUS_REGISTRY.get(direction, [])
    if categories:
        sentences = [s for s in sentences if s.category in categories]
    if n:
        sentences = sentences[:n]
    return sentences


def get_corpus_as_pairs(
    direction: str = "en-fr",
    categories: Optional[List[str]] = None,
    n: Optional[int] = None,
) -> List[Dict[str, str]]:
    """Get corpus in eval-compatible format (list of dicts with source/reference)."""
    sentences = get_corpus(direction, categories, n)
    return [
        {
            "source": s.text,
            "reference": s.reference or "",
            "category": s.category,
            "id": i,
        }
        for i, s in enumerate(sentences)
    ]


def list_directions() -> List[str]:
    """List available corpus directions."""
    return sorted(_CORPUS_REGISTRY.keys())


def list_categories(direction: str = "en-fr") -> List[str]:
    """List available categories for a direction."""
    sentences = _CORPUS_REGISTRY.get(direction, [])
    return sorted(set(s.category for s in sentences))


def corpus_stats(direction: str = "en-fr") -> Dict[str, int]:
    """Get sentence counts per category."""
    sentences = _CORPUS_REGISTRY.get(direction, [])
    stats = {}
    for s in sentences:
        stats[s.category] = stats.get(s.category, 0) + 1
    stats["total"] = len(sentences)
    return stats
