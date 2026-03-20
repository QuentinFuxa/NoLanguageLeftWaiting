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
    # Idioms
    TestSentence(
        "It's raining cats and dogs outside, so we should probably stay home.",
        "idioms", "en-fr",
    ),
    TestSentence(
        "The project was a piece of cake once we figured out the architecture.",
        "idioms", "en-fr",
    ),
    TestSentence(
        "He hit the nail on the head when he said the problem was communication.",
        "idioms", "en-fr",
    ),

    # More medium
    TestSentence(
        "The delegation from Japan arrived earlier this morning to discuss bilateral trade agreements.",
        "medium", "en-fr",
    ),
    TestSentence(
        "Renewable energy sources now account for more than thirty percent of global electricity production.",
        "medium", "en-fr",
    ),
    TestSentence(
        "The museum will host a special exhibition featuring works from the Italian Renaissance period.",
        "medium", "en-fr",
    ),

    # More complex
    TestSentence(
        "The newly proposed legislation, which has been the subject of intense lobbying from both "
        "the pharmaceutical industry and patient advocacy groups, seeks to establish a comprehensive "
        "framework for regulating drug pricing across the European Union.",
        "complex", "en-fr",
    ),
    TestSentence(
        "Although many scientists remain cautiously optimistic about the potential of gene therapy "
        "to treat previously incurable diseases, ethical concerns regarding germline modifications "
        "continue to spark heated debates in bioethics committees worldwide.",
        "complex", "en-fr",
    ),
]

# ---------------------------------------------------------------------------
# EN -> ZH corpus
# ---------------------------------------------------------------------------
_EN_ZH_SENTENCES = [
    # Simple
    TestSentence("Hello, how are you today?", "simple", "en-zh",
                 "\u4f60\u597d\uff0c\u4eca\u5929\u4f60\u600e\u4e48\u6837\uff1f"),
    TestSentence("Thank you very much for your help.", "simple", "en-zh",
                 "\u975e\u5e38\u611f\u8c22\u60a8\u7684\u5e2e\u52a9\u3002"),
    TestSentence("Good morning, nice to meet you.", "simple", "en-zh",
                 "\u65e9\u4e0a\u597d\uff0c\u5f88\u9ad8\u5174\u8ba4\u8bc6\u4f60\u3002"),
    TestSentence("Please open the window, it is too hot.", "simple", "en-zh",
                 "\u8bf7\u6253\u5f00\u7a97\u6237\uff0c\u592a\u70ed\u4e86\u3002"),
    TestSentence("I am studying Chinese at the university.", "simple", "en-zh",
                 "\u6211\u5728\u5927\u5b66\u5b66\u4e60\u4e2d\u6587\u3002"),

    # Medium
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
        "China's space program has made remarkable progress in recent years, "
        "including landing a rover on the far side of the moon.",
        "medium", "en-zh",
    ),
    TestSentence(
        "The Belt and Road Initiative continues to reshape international trade and "
        "infrastructure development across Asia and Africa.",
        "medium", "en-zh",
    ),
    TestSentence(
        "Electric vehicles have become increasingly popular in Chinese cities, "
        "driven by government subsidies and environmental concerns.",
        "medium", "en-zh",
    ),

    # Complex
    TestSentence(
        "The simultaneous translation system processes audio in real time, "
        "detecting speech boundaries and generating translations incrementally "
        "as the speaker continues talking.",
        "complex", "en-zh",
    ),
    TestSentence(
        "Despite facing criticism from Western nations regarding its human rights record, "
        "China has continued to assert that its economic development model provides a viable "
        "alternative to liberal democratic capitalism.",
        "complex", "en-zh",
    ),
    TestSentence(
        "The integration of traditional Chinese medicine with modern pharmaceutical research "
        "has yielded several promising compounds, though rigorous clinical trials remain essential "
        "to validate their therapeutic efficacy.",
        "complex", "en-zh",
    ),

    # Numbers
    TestSentence("GDP growth reached 5.2 percent in the first quarter of 2025.",
                 "numbers", "en-zh"),
    TestSentence("The population of Shanghai exceeded 24.87 million according to the latest census.",
                 "numbers", "en-zh"),
    TestSentence("China invested 320 billion dollars in renewable energy projects in 2024.",
                 "numbers", "en-zh"),

    # Names
    TestSentence("Tsinghua University and MIT announced a joint research program.",
                 "names", "en-zh"),
    TestSentence("Xi Jinping delivered a keynote speech at the World Economic Forum in Davos.",
                 "names", "en-zh"),
    TestSentence("Alibaba, Tencent, and ByteDance are reshaping the global technology landscape.",
                 "names", "en-zh"),

    # Technical
    TestSentence(
        "The attention mechanism allows the model to focus on relevant parts "
        "of the input sequence when generating each output token.",
        "technical", "en-zh",
    ),
    TestSentence(
        "Large language models based on the transformer architecture have demonstrated "
        "remarkable capabilities in natural language understanding and generation.",
        "technical", "en-zh",
    ),

    # Idioms
    TestSentence(
        "Rome was not built in a day, and neither will this ambitious project be completed overnight.",
        "idioms", "en-zh",
    ),
]


# ---------------------------------------------------------------------------
# EN -> DE corpus
# ---------------------------------------------------------------------------
_EN_DE_SENTENCES = [
    # Simple
    TestSentence("Good morning, how are you?", "simple", "en-de",
                 "Guten Morgen, wie geht es Ihnen?"),
    TestSentence("The train leaves at half past nine.", "simple", "en-de",
                 "Der Zug faehrt um halb zehn ab."),
    TestSentence("I would like to order a beer please.", "simple", "en-de",
                 "Ich moechte bitte ein Bier bestellen."),
    TestSentence("The children are playing in the garden.", "simple", "en-de",
                 "Die Kinder spielen im Garten."),
    TestSentence("Could you please tell me the way to the train station?", "simple", "en-de",
                 "Koennten Sie mir bitte den Weg zum Bahnhof sagen?"),

    # Medium
    TestSentence(
        "The German government has proposed new legislation to accelerate the transition to renewable energy.",
        "medium", "en-de",
    ),
    TestSentence(
        "Researchers at the Max Planck Institute have published a groundbreaking study on quantum computing.",
        "medium", "en-de",
    ),
    TestSentence(
        "The automotive industry in Germany is undergoing a fundamental transformation towards electric mobility.",
        "medium", "en-de",
    ),
    TestSentence(
        "Angela Merkel's legacy continues to influence European politics years after she left office.",
        "medium", "en-de",
    ),
    TestSentence(
        "The European Union is working on new regulations to ensure the responsible development of artificial intelligence.",
        "medium", "en-de",
    ),

    # Complex
    TestSentence(
        "The reunification of Germany in 1990, while celebrated as a triumph of democracy, "
        "also brought significant economic challenges that continue to shape the disparities "
        "between eastern and western federal states.",
        "complex", "en-de",
    ),
    TestSentence(
        "Despite the Energiewende's ambitious goals, the phase-out of nuclear power combined with "
        "the slow expansion of renewable infrastructure has raised concerns about energy security "
        "and the country's ability to meet its climate targets.",
        "complex", "en-de",
    ),
    TestSentence(
        "The Frankfurt Stock Exchange, as the largest in continental Europe, plays a crucial role "
        "in determining monetary policy direction, particularly through its influence on the "
        "European Central Bank's decision-making process.",
        "complex", "en-de",
    ),

    # Numbers
    TestSentence("Germany exported goods worth 1.66 trillion euros in the fiscal year 2024.",
                 "numbers", "en-de"),
    TestSentence("The unemployment rate fell to 3.1 percent in the third quarter, the lowest since 2019.",
                 "numbers", "en-de"),
    TestSentence("Over 83 million people live in Germany, making it the most populous EU member state.",
                 "numbers", "en-de"),

    # Names
    TestSentence("Chancellor Scholz met with President Macron at the Brandenburg Gate.",
                 "names", "en-de"),
    TestSentence("BMW, Mercedes-Benz, and Volkswagen are investing heavily in electric vehicle technology.",
                 "names", "en-de"),
    TestSentence("The Technical University of Munich was ranked among the top fifty universities worldwide.",
                 "names", "en-de"),

    # Technical
    TestSentence(
        "The new semiconductor fabrication plant in Dresden will produce chips using "
        "three-nanometer process technology, significantly advancing European chip sovereignty.",
        "technical", "en-de",
    ),
    TestSentence(
        "Machine learning algorithms trained on multilingual corpora can achieve near-human "
        "translation quality for closely related language pairs like English and German.",
        "technical", "en-de",
    ),

    # Idioms
    TestSentence(
        "You cannot have your cake and eat it too, especially when negotiating trade agreements.",
        "idioms", "en-de",
    ),
    TestSentence(
        "The early bird catches the worm, which is why our team starts meetings at seven in the morning.",
        "idioms", "en-de",
    ),
]


# ---------------------------------------------------------------------------
# EN -> IT corpus
# ---------------------------------------------------------------------------
_EN_IT_SENTENCES = [
    # Simple
    TestSentence("Good evening, where is the restaurant?", "simple", "en-it",
                 "Buonasera, dov'e il ristorante?"),
    TestSentence("I would like a table for two people.", "simple", "en-it",
                 "Vorrei un tavolo per due persone."),
    TestSentence("The museum opens at nine in the morning.", "simple", "en-it",
                 "Il museo apre alle nove di mattina."),
    TestSentence("Can you recommend a good hotel near here?", "simple", "en-it",
                 "Puo raccomandare un buon hotel qui vicino?"),
    TestSentence("The view from this window is absolutely beautiful.", "simple", "en-it",
                 "La vista da questa finestra e assolutamente bellissima."),

    # Medium
    TestSentence(
        "The Italian government has announced a comprehensive plan to restore historical monuments damaged by recent earthquakes.",
        "medium", "en-it",
    ),
    TestSentence(
        "The fashion industry in Milan continues to set global trends despite increasing competition from emerging markets.",
        "medium", "en-it",
    ),
    TestSentence(
        "Renewable energy installations in southern Italy have doubled over the past three years.",
        "medium", "en-it",
    ),
    TestSentence(
        "The archaeological excavations at Pompeii have revealed new insights into daily life in ancient Rome.",
        "medium", "en-it",
    ),
    TestSentence(
        "Italy's healthcare system is widely regarded as one of the most effective in the European Union.",
        "medium", "en-it",
    ),

    # Complex
    TestSentence(
        "The ongoing debate about immigration policy in Italy reflects deeper tensions between "
        "the country's humanitarian obligations under international law and the practical challenges "
        "of integrating large numbers of asylum seekers into the labor market.",
        "complex", "en-it",
    ),
    TestSentence(
        "While Italy's culinary tradition remains deeply rooted in regional diversity, the globalization "
        "of food culture has prompted a new generation of chefs to experiment with innovative fusion "
        "techniques that blend Mediterranean flavors with Asian and Latin American influences.",
        "complex", "en-it",
    ),
    TestSentence(
        "The restoration of the Sistine Chapel, which took over twenty years to complete, demonstrated "
        "that modern conservation techniques could reveal the original brilliance of Renaissance frescoes "
        "without compromising their historical integrity.",
        "complex", "en-it",
    ),

    # Numbers
    TestSentence("Tourism contributed 13 percent of Italy's GDP in 2024, with over 65 million visitors.",
                 "numbers", "en-it"),
    TestSentence("The Italian national debt reached 2.87 trillion euros at the end of the fiscal year.",
                 "numbers", "en-it"),
    TestSentence("Ferrari produced 14,004 cars in 2024, a new record for the luxury automaker.",
                 "numbers", "en-it"),

    # Names
    TestSentence("Prime Minister Meloni addressed the United Nations General Assembly in New York.",
                 "names", "en-it"),
    TestSentence("The University of Bologna, founded in 1088, is the oldest university in the Western world.",
                 "names", "en-it"),
    TestSentence("Fiat, Pirelli, and Leonardo are among Italy's most influential industrial conglomerates.",
                 "names", "en-it"),

    # Technical
    TestSentence(
        "The European Space Agency's mission to Jupiter, with significant Italian contributions, "
        "will study the gas giant's icy moons for signs of subsurface liquid water.",
        "technical", "en-it",
    ),
    TestSentence(
        "Advanced natural language processing models now handle the complexities of Italian morphology, "
        "including verb conjugations and gender agreement, with high accuracy.",
        "technical", "en-it",
    ),

    # Idioms
    TestSentence(
        "When in Rome, do as the Romans do, which means adapting your communication style to local customs.",
        "idioms", "en-it",
    ),
    TestSentence(
        "Every cloud has a silver lining, and even this economic downturn has created new opportunities.",
        "idioms", "en-it",
    ),
]


# ---------------------------------------------------------------------------
# CS -> EN corpus
# ---------------------------------------------------------------------------
_CS_EN_SENTENCES = [
    # Simple
    TestSentence("Dobry den, jak se mate?", "simple", "cs-en",
                 "Good afternoon, how are you?"),
    TestSentence("Prosim vas, kde je nejblizsi metro?", "simple", "cs-en",
                 "Excuse me, where is the nearest metro?"),
    TestSentence("Chtel bych platit, dekuji.", "simple", "cs-en",
                 "I would like to pay, thank you."),
    TestSentence("Pocasi je dnes velmi pekne.", "simple", "cs-en",
                 "The weather is very nice today."),
    TestSentence("Mam rad ceskou kuchyni, zejmena knedliky.", "simple", "cs-en",
                 "I like Czech cuisine, especially dumplings."),

    # Medium
    TestSentence(
        "Ceska republika je clenem Evropske unie od roku 2004 a aktivne se podili na formovani evropske politiky.",
        "medium", "cs-en",
    ),
    TestSentence(
        "Univerzita Karlova v Praze je jednou z nejstarsich univerzit v Evrope a byla zalozena v roce 1348.",
        "medium", "cs-en",
    ),
    TestSentence(
        "Cesky prumysl se v poslednich letech vyrazne transformoval smerem k high-tech vyrobu a automobilovemu prumyslu.",
        "medium", "cs-en",
    ),
    TestSentence(
        "Praha kazdy rok privita vice nez osm milionu turistu, coz z ni cini jednu z nejnavstevovanejsich mest v Evrope.",
        "medium", "cs-en",
    ),
    TestSentence(
        "Ceska narodni banka udrzuje stabilni menovou politiku, ktera podporuje ekonomicky rust zeme.",
        "medium", "cs-en",
    ),

    # Complex
    TestSentence(
        "Prechod Ceske republiky z centralne planovane ekonomiky na trzni hospodarstvi "
        "v devadesatych letech predstavoval jednu z nejuspesnejsich ekonomickych transformaci "
        "ve stredni a vychodni Evrope, prestoze byl doprovazen znacnymi socialnimi naklady.",
        "complex", "cs-en",
    ),
    TestSentence(
        "Navzdory soucasnym geopolitickym napetim v Evrope si Ceska republika udrzuje silne "
        "diplomaticke vztahy se svymi sousedy a aktivne podporuje rozsirovani transatlanticke "
        "spoluprace v oblasti bezpecnosti a obrany.",
        "complex", "cs-en",
    ),
    TestSentence(
        "Ceska veda a vyzkum zaznamenaly v poslednich letech vyrazny posun, predevsim v oblastech "
        "jako je nanotechnologie, informatika a biomedicina, coz vyznamne prispiva k inovacnimu "
        "potencialu zeme.",
        "complex", "cs-en",
    ),

    # Numbers
    TestSentence("HDP Ceske republiky vzrostl o 3,5 procenta v prvnim ctvrtleti roku 2025.",
                 "numbers", "cs-en"),
    TestSentence("V Ceske republice zije priblizne 10,8 milionu obyvatel na rozloze 78 866 kilometru ctverecnich.",
                 "numbers", "cs-en"),
    TestSentence("Nezamestnanost klesla na 2,7 procenta, coz je jedna z nejnizsich hodnot v Evropske unii.",
                 "numbers", "cs-en"),

    # Names
    TestSentence("Prezident Pavel se setkal s generalnim tajemnikem NATO v Bruselu.",
                 "names", "cs-en"),
    TestSentence("Skoda Auto, Ceske drahy a skupina CEZ patri k nejvyznamnejsim ceskym firmam.",
                 "names", "cs-en"),
    TestSentence("Karlstejn, Cesky Krumlov a Prazsky hrad jsou nejnavstevovanejsi pamatky v zemi.",
                 "names", "cs-en"),

    # Technical
    TestSentence(
        "Vyvojari na CVUT v Praze pracuji na novych algoritmech pro zpracovani prirozeneho jazyka, "
        "ktere vyuzivaji architekturu transformeru pro simultanni preklad.",
        "technical", "cs-en",
    ),
    TestSentence(
        "Ceske firmy v oblasti kyberneticke bezpecnosti vyvijeji pokrocile systemy pro detekci "
        "a prevenci kybernetickych utoku v realnem case.",
        "technical", "cs-en",
    ),

    # Idioms
    TestSentence(
        "Bez prace nejsou kolace, a proto musi kazdy student tvrde pracovat na svych projektech.",
        "idioms", "cs-en",
    ),
    TestSentence(
        "Rana ptace dale doskace, coz plati zejmena v konkurencnim prostredi technologickych startupu.",
        "idioms", "cs-en",
    ),
]


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------
_CORPUS_REGISTRY: Dict[str, List[TestSentence]] = {
    "en-fr": _EN_FR_SENTENCES,
    "en-zh": _EN_ZH_SENTENCES,
    "en-de": _EN_DE_SENTENCES,
    "en-it": _EN_IT_SENTENCES,
    "cs-en": _CS_EN_SENTENCES,
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
