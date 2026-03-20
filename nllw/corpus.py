"""Comprehensive test corpus for SimulMT evaluation.

Provides 120+ categorized sentence pairs across multiple language directions,
with edge cases specifically designed for simultaneous translation testing.

Usage:
    from nllw.corpus import TestCorpus, get_corpus, get_discourse_pairs

    # All pairs
    all_pairs = TestCorpus.FULL_CORPUS

    # Filtered
    en_fr = get_corpus(lang_pair="en-fr")
    idioms = get_corpus(category="idiom")
    hard = get_corpus(difficulty="hard")
    hard_en_fr_idioms = get_corpus(lang_pair="en-fr", category="idiom", difficulty="hard")

    # Discourse pairs (sentence2 needs context from sentence1)
    for s1, s2 in get_discourse_pairs():
        print(s1["source"], "->", s2["source"])
"""

from typing import Optional


class TestCorpus:
    """Curated test corpus for SimulMT evaluation.

    Each entry is a dict with:
        source       -- source sentence (plain text)
        reference    -- reference translation (plain text)
        source_lang  -- ISO 639-1 source language code
        target_lang  -- ISO 639-1 target language code
        category     -- one of: short, medium, long, pronoun_ambiguity, idiom,
                        numbers, named_entities, discourse_dependent, reordering,
                        technical, colloquial
        tags         -- list of additional descriptor tags
        difficulty   -- easy, medium, or hard
        notes        -- free-text note explaining why this pair matters
    """

    FULL_CORPUS: list[dict] = [
        # ==================================================================
        # EN -> FR  (30 pairs)  -- primary test direction
        # ==================================================================

        # ---- short (2-4 words) -------------------------------------------
        {
            "source": "Hello everyone",
            "reference": "Bonjour tout le monde",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Minimal input, checks that the system produces output quickly.",
        },
        {
            "source": "Thank you very much",
            "reference": "Merci beaucoup",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "short",
            "tags": ["politeness"],
            "difficulty": "easy",
            "notes": "Four source words collapse to two target words.",
        },
        {
            "source": "See you tomorrow",
            "reference": "A demain",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Short farewell; tests minimal latency path.",
        },
        # ---- medium (5-10 words) -----------------------------------------
        {
            "source": "I would like to order a coffee",
            "reference": "Je voudrais commander un cafe",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "medium",
            "tags": ["hallucination_prone"],
            "difficulty": "medium",
            "notes": "Partial prefix 'I would like to' is hallucination-prone in SimulMT.",
        },
        {
            "source": "The weather is nice today in Paris",
            "reference": "Il fait beau aujourd'hui a Paris",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "medium",
            "tags": ["weather", "named_entity"],
            "difficulty": "easy",
            "notes": "Simple declarative with a place name.",
        },
        {
            "source": "Can you please repeat that more slowly",
            "reference": "Pouvez-vous repeter cela plus lentement s'il vous plait",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "medium",
            "tags": ["question", "politeness"],
            "difficulty": "medium",
            "notes": "Question form; word order differs in French.",
        },
        {
            "source": "We are going to the airport right now",
            "reference": "Nous allons a l'aeroport maintenant",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "medium",
            "tags": ["travel"],
            "difficulty": "easy",
            "notes": "Near-future construction; tests progressive commit.",
        },
        {
            "source": "She has been working here for five years",
            "reference": "Elle travaille ici depuis cinq ans",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "medium",
            "tags": ["tense_mismatch"],
            "difficulty": "medium",
            "notes": "English present perfect continuous maps to French present tense.",
        },
        # ---- long (15+ words) --------------------------------------------
        {
            "source": "The president of France announced new economic reforms during the press conference held in Paris yesterday afternoon",
            "reference": "Le president de la France a annonce de nouvelles reformes economiques lors de la conference de presse tenue a Paris hier apres-midi",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "long",
            "tags": ["news", "named_entity", "politics"],
            "difficulty": "medium",
            "notes": "News-style sentence; tests sustained commit over 20+ words.",
        },
        {
            "source": "Scientists from the National Institute of Health have published a groundbreaking study on the long-term effects of air pollution on children",
            "reference": "Des scientifiques de l'Institut national de la sante ont publie une etude revolutionnaire sur les effets a long terme de la pollution atmospherique sur les enfants",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "long",
            "tags": ["science", "named_entity"],
            "difficulty": "hard",
            "notes": "22 source words; complex NP with prepositional chain.",
        },
        {
            "source": "The European Central Bank decided to raise interest rates by twenty-five basis points in order to combat rising inflation across the eurozone",
            "reference": "La Banque centrale europeenne a decide de relever les taux d'interet de vingt-cinq points de base afin de lutter contre l'inflation croissante dans la zone euro",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "long",
            "tags": ["economics", "numbers", "named_entity"],
            "difficulty": "hard",
            "notes": "Financial news with numbers and institutional name.",
        },
        {
            "source": "After several hours of intense negotiations the two parties finally reached an agreement that was satisfactory to both sides",
            "reference": "Apres plusieurs heures de negociations intenses les deux parties sont finalement parvenues a un accord satisfaisant pour les deux camps",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "long",
            "tags": ["news", "temporal"],
            "difficulty": "medium",
            "notes": "Fronted adverbial phrase; tests delayed main verb commit.",
        },
        # ---- pronoun_ambiguity -------------------------------------------
        {
            "source": "It is beautiful",
            "reference": "Il fait beau",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "pronoun_ambiguity",
            "tags": ["weather", "ambiguous_pronoun"],
            "difficulty": "hard",
            "notes": "'It' could be il/elle/ce; weather reading uses impersonal 'il fait'.",
        },
        {
            "source": "He left early this morning",
            "reference": "Il est parti tot ce matin",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "pronoun_ambiguity",
            "tags": ["ambiguous_pronoun"],
            "difficulty": "medium",
            "notes": "'Il' is masculine; contrast with 'It is beautiful' above.",
        },
        {
            "source": "It broke down again yesterday",
            "reference": "Elle est encore tombee en panne hier",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "pronoun_ambiguity",
            "tags": ["ambiguous_pronoun", "discourse_dependent"],
            "difficulty": "hard",
            "notes": "'It' referring to a feminine noun (la voiture); requires context.",
        },
        {
            "source": "They said they would come but they never showed up",
            "reference": "Ils ont dit qu'ils viendraient mais ils ne sont jamais venus",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "pronoun_ambiguity",
            "tags": ["ambiguous_pronoun"],
            "difficulty": "medium",
            "notes": "'They' is gender-ambiguous; default masculine 'ils' without context.",
        },
        # ---- idiom -------------------------------------------------------
        {
            "source": "It is raining cats and dogs",
            "reference": "Il pleut des cordes",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "idiom",
            "tags": ["weather", "figurative"],
            "difficulty": "hard",
            "notes": "Literal translation would be nonsensical; must wait for full idiom.",
        },
        {
            "source": "He kicked the bucket last night",
            "reference": "Il est mort hier soir",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "idiom",
            "tags": ["figurative", "colloquial"],
            "difficulty": "hard",
            "notes": "Idiom meaning 'he died'; literal translation is wrong.",
        },
        {
            "source": "Let us not beat around the bush",
            "reference": "Ne tournons pas autour du pot",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "idiom",
            "tags": ["figurative"],
            "difficulty": "hard",
            "notes": "Idiomatic expression; French has a different image (pot vs bush).",
        },
        {
            "source": "She really hit the nail on the head with that comment",
            "reference": "Elle a vraiment mis le doigt dessus avec ce commentaire",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "idiom",
            "tags": ["figurative"],
            "difficulty": "hard",
            "notes": "Idiomatic; different metaphor in French (finger vs nail).",
        },
        # ---- numbers -----------------------------------------------------
        {
            "source": "The company reported revenue of 3.5 billion euros in 2024",
            "reference": "L'entreprise a declare un chiffre d'affaires de 3,5 milliards d'euros en 2024",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "numbers",
            "tags": ["finance", "decimal"],
            "difficulty": "medium",
            "notes": "Decimal notation differs: English 3.5 -> French 3,5.",
        },
        {
            "source": "The meeting is scheduled for March 15th 2026 at 2:30 PM",
            "reference": "La reunion est prevue le 15 mars 2026 a 14 h 30",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "numbers",
            "tags": ["date", "time"],
            "difficulty": "medium",
            "notes": "Date format and 12h->24h clock conversion.",
        },
        {
            "source": "The population of the city grew from 1.2 million to 1.8 million in just ten years",
            "reference": "La population de la ville est passee de 1,2 million a 1,8 million en seulement dix ans",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "numbers",
            "tags": ["statistics", "decimal"],
            "difficulty": "medium",
            "notes": "Multiple numbers in one sentence with decimal points.",
        },
        # ---- named_entities ----------------------------------------------
        {
            "source": "My name is Jean-Pierre Dupont and I live in Marseille",
            "reference": "Je m'appelle Jean-Pierre Dupont et j'habite a Marseille",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "named_entities",
            "tags": ["person_name", "city"],
            "difficulty": "easy",
            "notes": "French proper names should pass through unchanged.",
        },
        {
            "source": "Emmanuel Macron met with Angela Merkel at the Elysee Palace",
            "reference": "Emmanuel Macron a rencontre Angela Merkel au palais de l'Elysee",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "named_entities",
            "tags": ["person_name", "politics", "place"],
            "difficulty": "medium",
            "notes": "Multiple named entities; 'Elysee Palace' becomes 'palais de l'Elysee'.",
        },
        {
            "source": "The World Health Organization issued a warning about the new virus variant detected in Southeast Asia",
            "reference": "L'Organisation mondiale de la sante a emis un avertissement concernant le nouveau variant du virus detecte en Asie du Sud-Est",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "named_entities",
            "tags": ["organization", "geography"],
            "difficulty": "hard",
            "notes": "Organization name translates; geographic compound translates.",
        },
        # ---- reordering --------------------------------------------------
        {
            "source": "I have never been to Japan",
            "reference": "Je ne suis jamais alle au Japon",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "reordering",
            "tags": ["negation"],
            "difficulty": "medium",
            "notes": "French negation wraps auxiliary: ne...jamais.",
        },
        {
            "source": "The book that she gave me yesterday was very interesting",
            "reference": "Le livre qu'elle m'a donne hier etait tres interessant",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "reordering",
            "tags": ["relative_clause"],
            "difficulty": "medium",
            "notes": "Relative clause with object pronoun fronting in French.",
        },
        # ---- technical ---------------------------------------------------
        {
            "source": "The neural network architecture uses multi-head attention with layer normalization and dropout regularization",
            "reference": "L'architecture du reseau de neurones utilise l'attention multi-tetes avec la normalisation par couche et la regularisation par dropout",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "technical",
            "tags": ["machine_learning", "jargon"],
            "difficulty": "hard",
            "notes": "Technical ML terminology; some terms borrowed, some translated.",
        },
        # ---- colloquial --------------------------------------------------
        {
            "source": "Honestly I have no idea what he is talking about",
            "reference": "Franchement je n'ai aucune idee de quoi il parle",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "colloquial",
            "tags": ["informal", "discourse_marker"],
            "difficulty": "medium",
            "notes": "Discourse marker 'honestly'; informal register.",
        },

        # ==================================================================
        # FR -> EN  (20 pairs)
        # ==================================================================

        # ---- short -------------------------------------------------------
        {
            "source": "Il fait beau",
            "reference": "The weather is nice",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "short",
            "tags": ["weather", "impersonal"],
            "difficulty": "easy",
            "notes": "Impersonal construction; classic SimulMT test.",
        },
        {
            "source": "Bonjour a tous",
            "reference": "Hello everyone",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Minimal greeting.",
        },
        {
            "source": "Merci beaucoup",
            "reference": "Thank you very much",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "short",
            "tags": ["politeness"],
            "difficulty": "easy",
            "notes": "Two source words expand to four target words.",
        },
        # ---- medium ------------------------------------------------------
        {
            "source": "Je voudrais commander un cafe au lait s'il vous plait",
            "reference": "I would like to order a coffee with milk please",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "medium",
            "tags": ["politeness", "food"],
            "difficulty": "easy",
            "notes": "Polite request; straightforward mapping.",
        },
        {
            "source": "Le train de Paris a Lyon part a quinze heures trente",
            "reference": "The train from Paris to Lyon departs at three thirty PM",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "medium",
            "tags": ["travel", "time", "named_entity"],
            "difficulty": "medium",
            "notes": "24h to 12h clock; city names pass through.",
        },
        {
            "source": "Il est important de manger des fruits et des legumes chaque jour",
            "reference": "It is important to eat fruits and vegetables every day",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "medium",
            "tags": ["health"],
            "difficulty": "easy",
            "notes": "Straightforward declarative.",
        },
        {
            "source": "Est-ce que tu peux m'aider a trouver la gare",
            "reference": "Can you help me find the train station",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "medium",
            "tags": ["question", "travel"],
            "difficulty": "medium",
            "notes": "French question form 'est-ce que' maps to English inversion.",
        },
        # ---- long --------------------------------------------------------
        {
            "source": "Le president de la Republique a prononce un discours devant l'Assemblee nationale sur les reformes du systeme de retraite",
            "reference": "The President of the Republic delivered a speech before the National Assembly on pension system reforms",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "long",
            "tags": ["politics", "named_entity"],
            "difficulty": "medium",
            "notes": "Political news; institutional names translate.",
        },
        {
            "source": "Les chercheurs de l'universite de la Sorbonne ont publie une etude demontrant les effets positifs de la meditation sur la sante mentale",
            "reference": "Researchers from the Sorbonne University published a study demonstrating the positive effects of meditation on mental health",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "long",
            "tags": ["science", "named_entity"],
            "difficulty": "medium",
            "notes": "Academic source; long prepositional chain.",
        },
        {
            "source": "Malgre les conditions meteorologiques difficiles les equipes de secours ont reussi a evacuer tous les habitants de la zone inondee avant la tombee de la nuit",
            "reference": "Despite the difficult weather conditions the rescue teams managed to evacuate all the inhabitants of the flooded area before nightfall",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "long",
            "tags": ["news", "temporal"],
            "difficulty": "hard",
            "notes": "Fronted concessive clause; 25+ source words.",
        },
        # ---- pronoun_ambiguity -------------------------------------------
        {
            "source": "Il est parti ce matin",
            "reference": "He left this morning",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "pronoun_ambiguity",
            "tags": ["ambiguous_pronoun"],
            "difficulty": "medium",
            "notes": "'Il' could be 'he' or 'it'; person reading requires context.",
        },
        {
            "source": "Elle ne sait pas encore si elle va accepter l'offre",
            "reference": "She does not know yet if she will accept the offer",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "pronoun_ambiguity",
            "tags": ["ambiguous_pronoun", "negation"],
            "difficulty": "medium",
            "notes": "Repeated 'elle'; negation ne...pas wraps verb.",
        },
        # ---- idiom -------------------------------------------------------
        {
            "source": "Il pleut des cordes depuis ce matin",
            "reference": "It has been raining cats and dogs since this morning",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "idiom",
            "tags": ["weather", "figurative"],
            "difficulty": "hard",
            "notes": "French weather idiom; literal 'raining ropes' -> English idiom.",
        },
        {
            "source": "Ca ne casse pas trois pattes a un canard",
            "reference": "It is nothing to write home about",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "idiom",
            "tags": ["figurative", "colloquial"],
            "difficulty": "hard",
            "notes": "Uniquely French idiom with no direct English parallel.",
        },
        # ---- numbers -----------------------------------------------------
        {
            "source": "L'entreprise a genere un benefice net de 250 millions d'euros au troisieme trimestre",
            "reference": "The company generated a net profit of 250 million euros in the third quarter",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "numbers",
            "tags": ["finance"],
            "difficulty": "medium",
            "notes": "Financial figure with ordinal number.",
        },
        {
            "source": "La Tour Eiffel mesure trois cent trente metres de haut et a ete construite en 1889",
            "reference": "The Eiffel Tower is three hundred and thirty meters tall and was built in 1889",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "numbers",
            "tags": ["named_entity", "measurement", "date"],
            "difficulty": "medium",
            "notes": "Spelled-out number plus numeric year.",
        },
        # ---- named_entities ----------------------------------------------
        {
            "source": "Bonjour je m'appelle Quentin et je viens de Lyon",
            "reference": "Hello my name is Quentin and I come from Lyon",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "named_entities",
            "tags": ["person_name", "city"],
            "difficulty": "easy",
            "notes": "Self-introduction with name and city.",
        },
        # ---- reordering --------------------------------------------------
        {
            "source": "Jamais je n'aurais imagine que cela puisse arriver ici",
            "reference": "Never would I have imagined that this could happen here",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "reordering",
            "tags": ["negation", "inversion"],
            "difficulty": "hard",
            "notes": "Fronted negative adverb triggers inversion in both languages.",
        },
        # ---- colloquial --------------------------------------------------
        {
            "source": "Franchement c'est n'importe quoi cette histoire",
            "reference": "Honestly this whole thing is nonsense",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "colloquial",
            "tags": ["informal", "discourse_marker"],
            "difficulty": "medium",
            "notes": "Colloquial dislocation; discourse marker 'franchement'.",
        },
        {
            "source": "T'inquiete on va se debrouiller comme d'habitude",
            "reference": "Do not worry we will figure it out as usual",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "colloquial",
            "tags": ["informal", "elision"],
            "difficulty": "hard",
            "notes": "Elided spoken French; 'se debrouiller' is hard to translate.",
        },

        # ==================================================================
        # EN -> DE  (20 pairs)
        # ==================================================================

        # ---- short -------------------------------------------------------
        {
            "source": "Good morning",
            "reference": "Guten Morgen",
            "source_lang": "en",
            "target_lang": "de",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Minimal German greeting.",
        },
        {
            "source": "Thank you",
            "reference": "Danke schoen",
            "source_lang": "en",
            "target_lang": "de",
            "category": "short",
            "tags": ["politeness"],
            "difficulty": "easy",
            "notes": "Two-word exchange.",
        },
        # ---- medium ------------------------------------------------------
        {
            "source": "The train from Berlin to Munich departs at half past three",
            "reference": "Der Zug von Berlin nach Muenchen faehrt um halb vier ab",
            "source_lang": "en",
            "target_lang": "de",
            "category": "medium",
            "tags": ["travel", "time", "named_entity"],
            "difficulty": "medium",
            "notes": "Separable verb 'abfahren'; German 'halb vier' = 3:30, not 4:30.",
        },
        {
            "source": "I have been living in Germany for three years",
            "reference": "Ich lebe seit drei Jahren in Deutschland",
            "source_lang": "en",
            "target_lang": "de",
            "category": "medium",
            "tags": ["tense_mismatch"],
            "difficulty": "medium",
            "notes": "English present perfect continuous -> German present with 'seit'.",
        },
        {
            "source": "Could you please tell me where the nearest hospital is",
            "reference": "Koennten Sie mir bitte sagen wo das naechste Krankenhaus ist",
            "source_lang": "en",
            "target_lang": "de",
            "category": "medium",
            "tags": ["question", "politeness"],
            "difficulty": "medium",
            "notes": "Embedded question; verb-final in subordinate clause.",
        },
        # ---- long --------------------------------------------------------
        {
            "source": "Artificial intelligence is transforming the way we work and communicate with each other across the globe",
            "reference": "Kuenstliche Intelligenz veraendert die Art und Weise wie wir weltweit arbeiten und miteinander kommunizieren",
            "source_lang": "en",
            "target_lang": "de",
            "category": "long",
            "tags": ["technology"],
            "difficulty": "medium",
            "notes": "Long declarative; tests word order in German subordinate clause.",
        },
        {
            "source": "Germany is the largest economy in Europe and a major exporter of automobiles and industrial machinery",
            "reference": "Deutschland ist die groesste Volkswirtschaft Europas und ein bedeutender Exporteur von Automobilen und Industriemaschinen",
            "source_lang": "en",
            "target_lang": "de",
            "category": "long",
            "tags": ["economics", "named_entity"],
            "difficulty": "medium",
            "notes": "German compound nouns: Volkswirtschaft, Industriemaschinen.",
        },
        {
            "source": "The professor explained the theory of relativity to his students using simple examples that everyone could understand",
            "reference": "Der Professor erklaerte seinen Studenten die Relativitaetstheorie anhand einfacher Beispiele die jeder verstehen konnte",
            "source_lang": "en",
            "target_lang": "de",
            "category": "long",
            "tags": ["academic"],
            "difficulty": "hard",
            "notes": "Dative object before accusative; relative clause verb-final.",
        },
        {
            "source": "We need to significantly reduce our carbon emissions over the next decade to meet the climate targets set by the Paris Agreement",
            "reference": "Wir muessen unsere Kohlenstoffemissionen im naechsten Jahrzehnt deutlich reduzieren um die im Pariser Abkommen festgelegten Klimaziele zu erreichen",
            "source_lang": "en",
            "target_lang": "de",
            "category": "long",
            "tags": ["environment", "named_entity"],
            "difficulty": "hard",
            "notes": "German infinitive clause with 'um...zu'; compound nouns.",
        },
        # ---- reordering --------------------------------------------------
        {
            "source": "Yesterday I went to the store because we needed milk",
            "reference": "Gestern bin ich zum Geschaeft gegangen weil wir Milch brauchten",
            "source_lang": "en",
            "target_lang": "de",
            "category": "reordering",
            "tags": ["verb_position", "temporal"],
            "difficulty": "medium",
            "notes": "German V2: time adverb fronted, verb second, past participle end.",
        },
        {
            "source": "The book that my friend recommended to me was incredibly interesting",
            "reference": "Das Buch das mir mein Freund empfohlen hat war unglaublich interessant",
            "source_lang": "en",
            "target_lang": "de",
            "category": "reordering",
            "tags": ["relative_clause", "verb_position"],
            "difficulty": "hard",
            "notes": "Relative clause with verb-final order in German.",
        },
        {
            "source": "Although the weather was bad we decided to go hiking in the mountains",
            "reference": "Obwohl das Wetter schlecht war haben wir beschlossen in den Bergen wandern zu gehen",
            "source_lang": "en",
            "target_lang": "de",
            "category": "reordering",
            "tags": ["concessive", "verb_position"],
            "difficulty": "hard",
            "notes": "Subordinate clause verb-final, main clause verb-second after fronted clause.",
        },
        # ---- numbers -----------------------------------------------------
        {
            "source": "The concert tickets cost forty-five euros and fifty cents each",
            "reference": "Die Konzertkarten kosten jeweils fuenfundvierzig Euro und fuenfzig Cent",
            "source_lang": "en",
            "target_lang": "de",
            "category": "numbers",
            "tags": ["currency"],
            "difficulty": "medium",
            "notes": "German number words reverse tens/ones: fuenfundvierzig (5-and-40).",
        },
        {
            "source": "The population of Berlin is approximately 3.7 million people",
            "reference": "Die Bevoelkerung von Berlin betraegt ungefaehr 3,7 Millionen Einwohner",
            "source_lang": "en",
            "target_lang": "de",
            "category": "numbers",
            "tags": ["statistics", "named_entity"],
            "difficulty": "medium",
            "notes": "Decimal notation differs; named city.",
        },
        # ---- named_entities ----------------------------------------------
        {
            "source": "Angela Merkel served as Chancellor of Germany for sixteen years from 2005 to 2021",
            "reference": "Angela Merkel war sechzehn Jahre lang Bundeskanzlerin von Deutschland von 2005 bis 2021",
            "source_lang": "en",
            "target_lang": "de",
            "category": "named_entities",
            "tags": ["person_name", "politics", "date"],
            "difficulty": "medium",
            "notes": "Gendered title: Bundeskanzlerin (feminine form).",
        },
        # ---- idiom -------------------------------------------------------
        {
            "source": "That is a piece of cake",
            "reference": "Das ist ein Kinderspiel",
            "source_lang": "en",
            "target_lang": "de",
            "category": "idiom",
            "tags": ["figurative"],
            "difficulty": "hard",
            "notes": "English 'piece of cake' -> German 'child's play'.",
        },
        # ---- technical ---------------------------------------------------
        {
            "source": "The database server experienced a critical failure resulting in two hours of downtime",
            "reference": "Der Datenbankserver hatte einen kritischen Ausfall der zu zwei Stunden Ausfallzeit fuehrte",
            "source_lang": "en",
            "target_lang": "de",
            "category": "technical",
            "tags": ["computing", "compound_noun"],
            "difficulty": "hard",
            "notes": "German compounds: Datenbankserver, Ausfallzeit.",
        },
        # ---- colloquial --------------------------------------------------
        {
            "source": "I am so tired I could sleep for a week",
            "reference": "Ich bin so muede ich koennte eine Woche lang schlafen",
            "source_lang": "en",
            "target_lang": "de",
            "category": "colloquial",
            "tags": ["informal", "hyperbole"],
            "difficulty": "easy",
            "notes": "Informal hyperbole; straightforward mapping.",
        },

        # ==================================================================
        # EN -> ZH  (20 pairs)  -- IWSLT competition direction
        # ==================================================================

        # ---- short -------------------------------------------------------
        {
            "source": "Good morning",
            "reference": "\u65e9\u4e0a\u597d",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Minimal greeting; two EN words -> three ZH characters.",
        },
        {
            "source": "Thank you",
            "reference": "\u8c22\u8c22",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "short",
            "tags": ["politeness"],
            "difficulty": "easy",
            "notes": "Minimal politeness expression.",
        },
        {
            "source": "See you later",
            "reference": "\u56de\u5934\u89c1",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Three words -> three characters.",
        },
        # ---- medium ------------------------------------------------------
        {
            "source": "Artificial intelligence is transforming the way we work",
            "reference": "\u4eba\u5de5\u667a\u80fd\u6b63\u5728\u6539\u53d8\u6211\u4eec\u7684\u5de5\u4f5c\u65b9\u5f0f",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "medium",
            "tags": ["technology"],
            "difficulty": "medium",
            "notes": "Technical topic; tests character-level output.",
        },
        {
            "source": "I would like to book a table for two people tonight",
            "reference": "\u6211\u60f3\u9884\u8ba2\u4eca\u665a\u4e24\u4eba\u7684\u4f4d\u5b50",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "medium",
            "tags": ["reservation"],
            "difficulty": "medium",
            "notes": "Word order rearrangement: time before action in Chinese.",
        },
        {
            "source": "The children are playing in the park near the lake",
            "reference": "\u5b69\u5b50\u4eec\u6b63\u5728\u6e56\u8fb9\u7684\u516c\u56ed\u91cc\u73a9\u800d",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "medium",
            "tags": ["description"],
            "difficulty": "easy",
            "notes": "Location phrase order differs in Chinese.",
        },
        # ---- long --------------------------------------------------------
        {
            "source": "The Great Wall of China is one of the most impressive structures ever built by human civilization spanning thousands of kilometers across northern China",
            "reference": "\u4e07\u91cc\u957f\u57ce\u662f\u4eba\u7c7b\u6587\u660e\u6240\u5efa\u9020\u7684\u6700\u4f1f\u5927\u7684\u5efa\u7b51\u4e4b\u4e00\u6a2a\u8de8\u4e2d\u56fd\u5317\u90e8\u6570\u5343\u516c\u91cc",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "long",
            "tags": ["culture", "named_entity", "geography"],
            "difficulty": "hard",
            "notes": "22+ source words; named entity, participial clause.",
        },
        {
            "source": "Climate change is one of the greatest challenges facing humanity in the twenty first century requiring immediate global cooperation",
            "reference": "\u6c14\u5019\u53d8\u5316\u662f\u4e8c\u5341\u4e00\u4e16\u7eaa\u4eba\u7c7b\u9762\u4e34\u7684\u6700\u5927\u6311\u6218\u4e4b\u4e00\u9700\u8981\u7acb\u5373\u5f00\u5c55\u5168\u7403\u5408\u4f5c",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "long",
            "tags": ["environment"],
            "difficulty": "hard",
            "notes": "Complex sentence; participial phrase.",
        },
        {
            "source": "The company reported revenue of three point five billion dollars in the last quarter which exceeded analysts expectations by a significant margin",
            "reference": "\u8be5\u516c\u53f8\u62a5\u544a\u4e0a\u5b63\u5ea6\u6536\u5165\u4e3a\u4e09\u5341\u4e94\u4ebf\u7f8e\u5143\u5927\u5e45\u8d85\u8fc7\u4e86\u5206\u6790\u5e08\u7684\u9884\u671f",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "long",
            "tags": ["finance", "numbers"],
            "difficulty": "hard",
            "notes": "Financial report; relative clause restructured in Chinese.",
        },
        {
            "source": "The international space station orbits the earth approximately every ninety minutes carrying astronauts from different countries",
            "reference": "\u56fd\u9645\u7a7a\u95f4\u7ad9\u5927\u7ea6\u6bcf\u4e5d\u5341\u5206\u949f\u7ed5\u5730\u7403\u8fd0\u884c\u4e00\u5468\u642d\u8f7d\u6765\u81ea\u4e0d\u540c\u56fd\u5bb6\u7684\u5b87\u822a\u5458",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "long",
            "tags": ["science", "numbers"],
            "difficulty": "medium",
            "notes": "Scientific fact with number and participial clause.",
        },
        # ---- reordering --------------------------------------------------
        {
            "source": "Because it was raining we decided to stay at home and watch a movie",
            "reference": "\u56e0\u4e3a\u4e0b\u96e8\u4e86\u6211\u4eec\u51b3\u5b9a\u5446\u5728\u5bb6\u91cc\u770b\u7535\u5f71",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "reordering",
            "tags": ["causal"],
            "difficulty": "medium",
            "notes": "Causal clause order is natural in both languages here.",
        },
        {
            "source": "The student who scored highest on the exam received a scholarship",
            "reference": "\u8003\u8bd5\u6210\u7ee9\u6700\u9ad8\u7684\u5b66\u751f\u83b7\u5f97\u4e86\u5956\u5b66\u91d1",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "reordering",
            "tags": ["relative_clause"],
            "difficulty": "hard",
            "notes": "English post-nominal relative clause -> Chinese pre-nominal modifier.",
        },
        # ---- numbers -----------------------------------------------------
        {
            "source": "China's GDP reached 17.96 trillion US dollars in 2023",
            "reference": "\u4e2d\u56fd2023\u5e74\u56fd\u5185\u751f\u4ea7\u603b\u503c\u8fbe\u523017.96\u4e07\u4ebf\u7f8e\u5143",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "numbers",
            "tags": ["economics", "statistics"],
            "difficulty": "hard",
            "notes": "Number system differs: trillion vs wan-yi; decimal notation.",
        },
        {
            "source": "The temperature dropped to minus fifteen degrees Celsius last night",
            "reference": "\u6628\u665a\u6c14\u6e29\u964d\u81f3\u96f6\u4e0b\u5341\u4e94\u5ea6",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "numbers",
            "tags": ["weather", "measurement"],
            "difficulty": "medium",
            "notes": "Temperature; time phrase moves to front in Chinese.",
        },
        # ---- named_entities ----------------------------------------------
        {
            "source": "Xi Jinping met with the President of the United States at the summit in San Francisco",
            "reference": "\u4e60\u8fd1\u5e73\u5728\u65e7\u91d1\u5c71\u5cf0\u4f1a\u4e0a\u4e0e\u7f8e\u56fd\u603b\u7edf\u4f1a\u9762",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "named_entities",
            "tags": ["politics", "person_name", "place"],
            "difficulty": "hard",
            "notes": "Named entities; San Francisco -> transliterated name.",
        },
        # ---- idiom -------------------------------------------------------
        {
            "source": "You cannot have your cake and eat it too",
            "reference": "\u4f60\u4e0d\u80fd\u9c7c\u4e0e\u718a\u638c\u517c\u5f97",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "idiom",
            "tags": ["figurative", "proverb"],
            "difficulty": "hard",
            "notes": "English proverb -> Chinese chengyu equivalent.",
        },
        # ---- technical ---------------------------------------------------
        {
            "source": "The new quantum computing chip can perform calculations millions of times faster than traditional processors",
            "reference": "\u65b0\u578b\u91cf\u5b50\u8ba1\u7b97\u82af\u7247\u7684\u8ba1\u7b97\u901f\u5ea6\u6bd4\u4f20\u7edf\u5904\u7406\u5668\u5feb\u6570\u767e\u4e07\u500d",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "technical",
            "tags": ["computing"],
            "difficulty": "hard",
            "notes": "Technical terminology; comparative restructured in Chinese.",
        },
        # ---- colloquial --------------------------------------------------
        {
            "source": "No way that is amazing I cannot believe it",
            "reference": "\u4e0d\u4f1a\u5427\u592a\u795e\u5947\u4e86\u6211\u4e0d\u6562\u76f8\u4fe1",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "colloquial",
            "tags": ["informal", "exclamation"],
            "difficulty": "medium",
            "notes": "Informal exclamations; discourse-like sequence.",
        },

        # ==================================================================
        # EN -> ES  (15 pairs)
        # ==================================================================

        # ---- short -------------------------------------------------------
        {
            "source": "Good afternoon",
            "reference": "Buenas tardes",
            "source_lang": "en",
            "target_lang": "es",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Minimal greeting; singular -> plural in Spanish.",
        },
        {
            "source": "How are you",
            "reference": "Como estas",
            "source_lang": "en",
            "target_lang": "es",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Informal register assumed.",
        },
        # ---- medium ------------------------------------------------------
        {
            "source": "The children are playing in the park near the river",
            "reference": "Los ninos estan jugando en el parque cerca del rio",
            "source_lang": "en",
            "target_lang": "es",
            "category": "medium",
            "tags": ["description"],
            "difficulty": "easy",
            "notes": "Straightforward present continuous.",
        },
        {
            "source": "I need to go to the supermarket to buy some groceries",
            "reference": "Necesito ir al supermercado a comprar algunas cosas",
            "source_lang": "en",
            "target_lang": "es",
            "category": "medium",
            "tags": ["daily_life"],
            "difficulty": "easy",
            "notes": "Purpose clause with 'a + infinitive' in Spanish.",
        },
        {
            "source": "My grandmother always said that patience is a virtue",
            "reference": "Mi abuela siempre decia que la paciencia es una virtud",
            "source_lang": "en",
            "target_lang": "es",
            "category": "medium",
            "tags": ["proverb"],
            "difficulty": "easy",
            "notes": "Reported speech with imperfect tense.",
        },
        # ---- long --------------------------------------------------------
        {
            "source": "Climate change is one of the greatest challenges facing humanity in the twenty first century and requires immediate action",
            "reference": "El cambio climatico es uno de los mayores desafios que enfrenta la humanidad en el siglo veintiuno y requiere accion inmediata",
            "source_lang": "en",
            "target_lang": "es",
            "category": "long",
            "tags": ["environment"],
            "difficulty": "medium",
            "notes": "Complex NP with relative clause.",
        },
        {
            "source": "The Spanish national football team won the European Championship after defeating England in the final match",
            "reference": "La seleccion espanola de futbol gano el Campeonato Europeo tras derrotar a Inglaterra en la final",
            "source_lang": "en",
            "target_lang": "es",
            "category": "long",
            "tags": ["sports", "named_entity"],
            "difficulty": "medium",
            "notes": "Sports news with named entities and personal 'a'.",
        },
        {
            "source": "Researchers at the University of Barcelona have discovered a new method for treating Alzheimer's disease using artificial intelligence",
            "reference": "Investigadores de la Universidad de Barcelona han descubierto un nuevo metodo para tratar la enfermedad de Alzheimer mediante inteligencia artificial",
            "source_lang": "en",
            "target_lang": "es",
            "category": "long",
            "tags": ["science", "named_entity", "medical"],
            "difficulty": "hard",
            "notes": "Academic; named entities; medical terminology.",
        },
        # ---- numbers -----------------------------------------------------
        {
            "source": "The flight from Madrid to Buenos Aires takes approximately twelve hours and costs around 800 euros",
            "reference": "El vuelo de Madrid a Buenos Aires dura aproximadamente doce horas y cuesta alrededor de 800 euros",
            "source_lang": "en",
            "target_lang": "es",
            "category": "numbers",
            "tags": ["travel", "currency", "named_entity"],
            "difficulty": "medium",
            "notes": "Duration and currency; two named cities.",
        },
        # ---- reordering --------------------------------------------------
        {
            "source": "The house that we bought last year has a beautiful garden",
            "reference": "La casa que compramos el ano pasado tiene un jardin hermoso",
            "source_lang": "en",
            "target_lang": "es",
            "category": "reordering",
            "tags": ["relative_clause"],
            "difficulty": "medium",
            "notes": "Adjective position differs: jardin hermoso (post-nominal).",
        },
        # ---- idiom -------------------------------------------------------
        {
            "source": "He let the cat out of the bag during the meeting",
            "reference": "El revelo el secreto durante la reunion",
            "source_lang": "en",
            "target_lang": "es",
            "category": "idiom",
            "tags": ["figurative"],
            "difficulty": "hard",
            "notes": "Idiom; literal translation would be nonsensical in Spanish.",
        },
        # ---- pronoun_ambiguity -------------------------------------------
        {
            "source": "She told her that her sister would arrive tomorrow",
            "reference": "Ella le dijo que su hermana llegaria manana",
            "source_lang": "en",
            "target_lang": "es",
            "category": "pronoun_ambiguity",
            "tags": ["ambiguous_pronoun"],
            "difficulty": "hard",
            "notes": "Triple 'her' ambiguity: whose sister? Spanish 'su' is equally ambiguous.",
        },
        # ---- named_entities ----------------------------------------------
        {
            "source": "Gabriel Garcia Marquez was born in Aracataca Colombia in 1927",
            "reference": "Gabriel Garcia Marquez nacio en Aracataca Colombia en 1927",
            "source_lang": "en",
            "target_lang": "es",
            "category": "named_entities",
            "tags": ["person_name", "place", "date"],
            "difficulty": "easy",
            "notes": "Latin American proper names and place; date.",
        },
        # ---- technical ---------------------------------------------------
        {
            "source": "The renewable energy sector generated over 300 gigawatts of new capacity worldwide last year",
            "reference": "El sector de energias renovables genero mas de 300 gigavatios de nueva capacidad a nivel mundial el ano pasado",
            "source_lang": "en",
            "target_lang": "es",
            "category": "technical",
            "tags": ["energy", "numbers"],
            "difficulty": "medium",
            "notes": "Technical unit (gigawatts -> gigavatios) plus number.",
        },
        # ---- colloquial --------------------------------------------------
        {
            "source": "Come on let us grab something to eat I am starving",
            "reference": "Venga vamos a comer algo me muero de hambre",
            "source_lang": "en",
            "target_lang": "es",
            "category": "colloquial",
            "tags": ["informal", "imperative"],
            "difficulty": "medium",
            "notes": "Informal imperative chain; hyperbolic expression.",
        },

        # ==================================================================
        # EN -> IT  (8 pairs)
        # ==================================================================

        # ---- short -------------------------------------------------------
        {
            "source": "Good evening",
            "reference": "Buona sera",
            "source_lang": "en",
            "target_lang": "it",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Minimal Italian greeting.",
        },
        # ---- medium ------------------------------------------------------
        {
            "source": "I would like a glass of red wine please",
            "reference": "Vorrei un bicchiere di vino rosso per favore",
            "source_lang": "en",
            "target_lang": "it",
            "category": "medium",
            "tags": ["food", "politeness"],
            "difficulty": "easy",
            "notes": "Polite request; adjective after noun in Italian.",
        },
        {
            "source": "The Colosseum in Rome is one of the most visited monuments in the world",
            "reference": "Il Colosseo a Roma e uno dei monumenti piu visitati al mondo",
            "source_lang": "en",
            "target_lang": "it",
            "category": "medium",
            "tags": ["culture", "named_entity"],
            "difficulty": "medium",
            "notes": "Named entities: Colosseum, Rome.",
        },
        # ---- long --------------------------------------------------------
        {
            "source": "The Italian government announced a new investment plan worth ten billion euros aimed at modernizing the country's railway infrastructure",
            "reference": "Il governo italiano ha annunciato un nuovo piano di investimenti da dieci miliardi di euro volto a modernizzare l'infrastruttura ferroviaria del paese",
            "source_lang": "en",
            "target_lang": "it",
            "category": "long",
            "tags": ["politics", "economics", "numbers"],
            "difficulty": "hard",
            "notes": "Complex NP with participial modifier; financial figure.",
        },
        # ---- reordering --------------------------------------------------
        {
            "source": "I have never eaten such delicious pasta in my entire life",
            "reference": "Non ho mai mangiato una pasta cosi deliziosa in tutta la mia vita",
            "source_lang": "en",
            "target_lang": "it",
            "category": "reordering",
            "tags": ["negation", "food"],
            "difficulty": "medium",
            "notes": "Italian negation 'non...mai'; adjective after noun.",
        },
        # ---- idiom -------------------------------------------------------
        {
            "source": "When in Rome do as the Romans do",
            "reference": "Paese che vai usanza che trovi",
            "source_lang": "en",
            "target_lang": "it",
            "category": "idiom",
            "tags": ["proverb", "figurative"],
            "difficulty": "hard",
            "notes": "Proverb; Italian equivalent uses completely different image.",
        },
        # ---- numbers -----------------------------------------------------
        {
            "source": "Leonardo da Vinci painted the Mona Lisa between 1503 and 1519",
            "reference": "Leonardo da Vinci dipinse la Gioconda tra il 1503 e il 1519",
            "source_lang": "en",
            "target_lang": "it",
            "category": "numbers",
            "tags": ["named_entity", "date", "art"],
            "difficulty": "medium",
            "notes": "Named entities; Mona Lisa -> Gioconda in Italian.",
        },
        # ---- colloquial --------------------------------------------------
        {
            "source": "This pizza is absolutely incredible you have to try it",
            "reference": "Questa pizza e assolutamente incredibile devi provarla",
            "source_lang": "en",
            "target_lang": "it",
            "category": "colloquial",
            "tags": ["informal", "food"],
            "difficulty": "easy",
            "notes": "Informal recommendation; clitic pronoun 'la' attaches to verb.",
        },

        # ==================================================================
        # CS -> EN  (7 pairs)
        # ==================================================================

        # ---- short -------------------------------------------------------
        {
            "source": "Dobry den",
            "reference": "Good day",
            "source_lang": "cs",
            "target_lang": "en",
            "category": "short",
            "tags": ["greeting"],
            "difficulty": "easy",
            "notes": "Minimal Czech greeting.",
        },
        # ---- medium ------------------------------------------------------
        {
            "source": "Praha je hlavni mesto Ceske republiky",
            "reference": "Prague is the capital city of the Czech Republic",
            "source_lang": "cs",
            "target_lang": "en",
            "category": "medium",
            "tags": ["geography", "named_entity"],
            "difficulty": "easy",
            "notes": "Named entities: Praha -> Prague, Ceske republiky -> Czech Republic.",
        },
        {
            "source": "Chtel bych si objednat svickovou s knedliky prosim",
            "reference": "I would like to order sirloin with dumplings please",
            "source_lang": "cs",
            "target_lang": "en",
            "category": "medium",
            "tags": ["food", "culture"],
            "difficulty": "medium",
            "notes": "Czech national dish; cultural term that may not translate well.",
        },
        # ---- long --------------------------------------------------------
        {
            "source": "Cesti vedci z Univerzity Karlovy publikovali novou studii o vlivu klimatickych zmen na biodiverzitu ve stredni Evrope",
            "reference": "Czech scientists from Charles University published a new study on the impact of climate change on biodiversity in Central Europe",
            "source_lang": "cs",
            "target_lang": "en",
            "category": "long",
            "tags": ["science", "named_entity"],
            "difficulty": "hard",
            "notes": "Named entity: Univerzita Karlova -> Charles University; technical topic.",
        },
        {
            "source": "Navzdory slozitym podnebinym podminkam se hasicum podarilo zachranit vsechny obyvatele zaplavene vesnice",
            "reference": "Despite the difficult weather conditions the firefighters managed to rescue all the inhabitants of the flooded village",
            "source_lang": "cs",
            "target_lang": "en",
            "category": "long",
            "tags": ["news"],
            "difficulty": "hard",
            "notes": "Czech free word order; dative reflexive construction.",
        },
        # ---- reordering --------------------------------------------------
        {
            "source": "Knihu kterou mi doporucil muj kamarad jsem precetl za dva dny",
            "reference": "I read the book that my friend recommended to me in two days",
            "source_lang": "cs",
            "target_lang": "en",
            "category": "reordering",
            "tags": ["relative_clause", "verb_position"],
            "difficulty": "hard",
            "notes": "Czech SOV-like ordering with verb at end; topic fronting.",
        },
        # ---- numbers -----------------------------------------------------
        {
            "source": "Plzensky Prazdroj se vari od roku 1842 a je nejznamejsi ceske pivo na svete",
            "reference": "Pilsner Urquell has been brewed since 1842 and is the most famous Czech beer in the world",
            "source_lang": "cs",
            "target_lang": "en",
            "category": "numbers",
            "tags": ["named_entity", "date", "culture"],
            "difficulty": "medium",
            "notes": "Named entity: brand name translates; historical date.",
        },

        # ==================================================================
        # Discourse-dependent pairs (sentence2 needs context from sentence1)
        # These have discourse_group and discourse_position fields.
        # ==================================================================

        # ---- Discourse group 1: pronoun resolution (EN->FR) --------------
        {
            "source": "The car would not start this morning",
            "reference": "La voiture ne demarrait pas ce matin",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "vehicle"],
            "difficulty": "medium",
            "notes": "Context sentence: establishes 'the car' (feminine in French).",
            "discourse_group": 1,
            "discourse_position": 1,
        },
        {
            "source": "It broke down again on the highway",
            "reference": "Elle est encore tombee en panne sur l'autoroute",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "pronoun_resolution"],
            "difficulty": "hard",
            "notes": "'It' refers to 'la voiture' (feminine); without context, model may use 'il'.",
            "discourse_group": 1,
            "discourse_position": 2,
        },

        # ---- Discourse group 2: topic continuity (EN->FR) ----------------
        {
            "source": "The prime minister gave a speech about education reform",
            "reference": "Le premier ministre a prononce un discours sur la reforme de l'education",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "politics"],
            "difficulty": "medium",
            "notes": "Context sentence: establishes topic and subject.",
            "discourse_group": 2,
            "discourse_position": 1,
        },
        {
            "source": "He also announced that teachers would receive a salary increase next year",
            "reference": "Il a egalement annonce que les enseignants recevraient une augmentation de salaire l'annee prochaine",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "pronoun_resolution"],
            "difficulty": "medium",
            "notes": "'He' refers to the prime minister; 'also' signals discourse continuation.",
            "discourse_group": 2,
            "discourse_position": 2,
        },

        # ---- Discourse group 3: anaphoric reference (FR->EN) -------------
        {
            "source": "Marie a adopte un chien au refuge la semaine derniere",
            "reference": "Marie adopted a dog from the shelter last week",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "named_entity"],
            "difficulty": "easy",
            "notes": "Context sentence: establishes Marie and the dog.",
            "discourse_group": 3,
            "discourse_position": 1,
        },
        {
            "source": "Elle l'a appele Rex et il est tres affectueux",
            "reference": "She named him Rex and he is very affectionate",
            "source_lang": "fr",
            "target_lang": "en",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "pronoun_resolution"],
            "difficulty": "medium",
            "notes": "'Elle' = Marie, 'l'' = the dog, 'il' = Rex; multiple anaphoric references.",
            "discourse_group": 3,
            "discourse_position": 2,
        },

        # ---- Discourse group 4: temporal sequence (EN->DE) ---------------
        {
            "source": "We arrived at the hotel at midnight after a long drive",
            "reference": "Wir kamen nach einer langen Fahrt um Mitternacht im Hotel an",
            "source_lang": "en",
            "target_lang": "de",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "travel", "temporal"],
            "difficulty": "medium",
            "notes": "Context sentence: establishes setting and time.",
            "discourse_group": 4,
            "discourse_position": 1,
        },
        {
            "source": "The next morning we discovered that someone had stolen our luggage from the car",
            "reference": "Am naechsten Morgen stellten wir fest dass jemand unser Gepaeck aus dem Auto gestohlen hatte",
            "source_lang": "en",
            "target_lang": "de",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "temporal"],
            "difficulty": "hard",
            "notes": "'The next morning' is relative to previous sentence's midnight; verb-final in 'dass' clause.",
            "discourse_group": 4,
            "discourse_position": 2,
        },

        # ---- Discourse group 5: contrastive (EN->ZH) --------------------
        {
            "source": "My older brother studied medicine and became a doctor",
            "reference": "\u6211\u54e5\u54e5\u5b66\u4e86\u533b\u5b66\u6210\u4e86\u4e00\u540d\u533b\u751f",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "family"],
            "difficulty": "easy",
            "notes": "Context sentence: establishes the older brother.",
            "discourse_group": 5,
            "discourse_position": 1,
        },
        {
            "source": "I on the other hand chose to study computer science and I have never regretted it",
            "reference": "\u800c\u6211\u5219\u9009\u62e9\u4e86\u5b66\u8ba1\u7b97\u673a\u79d1\u5b66\u6211\u4ece\u672a\u540e\u6094\u8fc7",
            "source_lang": "en",
            "target_lang": "zh",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "contrast"],
            "difficulty": "medium",
            "notes": "'On the other hand' signals contrast with previous sentence; 'it' = the choice.",
            "discourse_group": 5,
            "discourse_position": 2,
        },

        # ---- Discourse group 6: elaboration (EN->FR) ---------------------
        {
            "source": "The experiment produced unexpected results",
            "reference": "L'experience a produit des resultats inattendus",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "science"],
            "difficulty": "easy",
            "notes": "Context sentence: establishes the experiment and its results.",
            "discourse_group": 6,
            "discourse_position": 1,
        },
        {
            "source": "In particular the control group performed better than the treatment group which contradicted the initial hypothesis",
            "reference": "En particulier le groupe temoin a obtenu de meilleurs resultats que le groupe de traitement ce qui contredisait l'hypothese initiale",
            "source_lang": "en",
            "target_lang": "fr",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "science", "elaboration"],
            "difficulty": "hard",
            "notes": "'In particular' elaborates on previous results; 'which' refers to the whole proposition.",
            "discourse_group": 6,
            "discourse_position": 2,
        },

        # ---- Discourse group 7: cause-effect (EN->ES) --------------------
        {
            "source": "The factory released toxic chemicals into the river for several years",
            "reference": "La fabrica vertio productos quimicos toxicos al rio durante varios anos",
            "source_lang": "en",
            "target_lang": "es",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "environment"],
            "difficulty": "medium",
            "notes": "Context sentence: establishes the cause.",
            "discourse_group": 7,
            "discourse_position": 1,
        },
        {
            "source": "As a result the local fish population declined dramatically and residents could no longer use the water for drinking",
            "reference": "Como resultado la poblacion local de peces disminuyo drasticamente y los residentes ya no pudieron usar el agua para beber",
            "source_lang": "en",
            "target_lang": "es",
            "category": "discourse_dependent",
            "tags": ["discourse_context", "environment", "consequence"],
            "difficulty": "hard",
            "notes": "'As a result' links to previous sentence's cause; 'the water' = the contaminated river.",
            "discourse_group": 7,
            "discourse_position": 2,
        },
    ]


def get_corpus(
    *,
    lang_pair: Optional[str] = None,
    category: Optional[str] = None,
    difficulty: Optional[str] = None,
    tags: Optional[list[str]] = None,
) -> list[dict]:
    """Return a filtered subset of the full corpus.

    Parameters
    ----------
    lang_pair : str, optional
        Filter by language pair, e.g. ``"en-fr"``, ``"fr-en"``, ``"en-zh"``.
    category : str, optional
        Filter by category, e.g. ``"idiom"``, ``"numbers"``, ``"long"``.
    difficulty : str, optional
        Filter by difficulty: ``"easy"``, ``"medium"``, or ``"hard"``.
    tags : list[str], optional
        Filter entries that contain *all* of the specified tags.

    Returns
    -------
    list[dict]
        Matching corpus entries (copies of the originals).
    """
    results = list(TestCorpus.FULL_CORPUS)

    if lang_pair is not None:
        parts = lang_pair.split("-", 1)
        if len(parts) == 2:
            src, tgt = parts
            results = [
                e for e in results
                if e["source_lang"] == src and e["target_lang"] == tgt
            ]

    if category is not None:
        results = [e for e in results if e["category"] == category]

    if difficulty is not None:
        results = [e for e in results if e["difficulty"] == difficulty]

    if tags is not None:
        tag_set = set(tags)
        results = [e for e in results if tag_set.issubset(set(e.get("tags", [])))]

    return results


def get_discourse_pairs() -> list[tuple[dict, dict]]:
    """Return discourse-dependent sentence pairs.

    Each pair is a ``(context_sentence, dependent_sentence)`` tuple where
    the second sentence requires the first for correct pronoun resolution,
    topic continuity, or discourse coherence.

    Returns
    -------
    list[tuple[dict, dict]]
        Ordered pairs; sentence1 provides context for sentence2.
    """
    discourse_entries: dict[int, dict[int, dict]] = {}
    for entry in TestCorpus.FULL_CORPUS:
        group = entry.get("discourse_group")
        pos = entry.get("discourse_position")
        if group is not None and pos is not None:
            discourse_entries.setdefault(group, {})[pos] = entry

    pairs: list[tuple[dict, dict]] = []
    for group_id in sorted(discourse_entries):
        group = discourse_entries[group_id]
        if 1 in group and 2 in group:
            pairs.append((group[1], group[2]))

    return pairs


def corpus_summary() -> str:
    """Return a human-readable summary of the corpus contents."""
    corpus = TestCorpus.FULL_CORPUS
    total = len(corpus)

    # Count by language pair
    pair_counts: dict[str, int] = {}
    for e in corpus:
        pair = f"{e['source_lang']}-{e['target_lang']}"
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

    # Count by category
    cat_counts: dict[str, int] = {}
    for e in corpus:
        cat = e["category"]
        cat_counts[cat] = cat_counts.get(cat, 0) + 1

    # Count by difficulty
    diff_counts: dict[str, int] = {}
    for e in corpus:
        diff = e["difficulty"]
        diff_counts[diff] = diff_counts.get(diff, 0) + 1

    # Discourse pairs
    n_discourse = len(get_discourse_pairs())

    lines = [
        f"SimulMT Test Corpus: {total} sentence pairs",
        "",
        "Language pairs:",
    ]
    for pair in sorted(pair_counts, key=lambda p: -pair_counts[p]):
        lines.append(f"  {pair}: {pair_counts[pair]}")

    lines.append("")
    lines.append("Categories:")
    for cat in sorted(cat_counts, key=lambda c: -cat_counts[c]):
        lines.append(f"  {cat}: {cat_counts[cat]}")

    lines.append("")
    lines.append("Difficulty:")
    for diff in ["easy", "medium", "hard"]:
        lines.append(f"  {diff}: {diff_counts.get(diff, 0)}")

    lines.append("")
    lines.append(f"Discourse-dependent pairs: {n_discourse}")

    return "\n".join(lines)


if __name__ == "__main__":
    print(corpus_summary())
