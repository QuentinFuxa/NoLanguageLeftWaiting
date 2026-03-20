#!/usr/bin/env python3
"""Automatic detection of translation alignment heads for NLLW.

Probes every (layer, head) in a GGUF model to find which attention heads
track source-target token alignment during translation.  The output is a
JSON file compatible with the ``nllw/heads/*.json`` format consumed by
:class:`AlignAttBackend`.

The algorithm:
  1. For each (source, reference) sentence pair, build the translation
     prompt, encode it, and greedily generate the target tokens.
  2. At every generation step, extract attention weights restricted to
     the source token range for every head being probed.
  3. Compute word-level alignments between source and (generated)
     target using SimAlign (preferred) or a monotonic fallback heuristic.
  4. For each aligned (src_word, tgt_word) pair, count how often the
     argmax of a head's source attention falls on the correct source
     tokens.  This count, normalized by the total number of alignment
     points, gives the Translation Score (TS) for each head.
  5. Heads with TS above a threshold are reported as "translation
     alignment heads".

Usage::

    python -m nllw.detect_heads \\
        --model /path/to/model.gguf \\
        --prompt-format hymt \\
        --lang en-fr \\
        -n 50 \\
        -o translation_heads.json
"""

import argparse
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Optional dependency: simalign
# ---------------------------------------------------------------------------
try:
    from simalign import SentenceAligner as _SentenceAligner

    _HAS_SIMALIGN = True
except ImportError:
    _HAS_SIMALIGN = False

# ===========================================================================
# Prompt format registry
# ===========================================================================

# Each format has:
#   template     – must contain ``{source}`` placeholder
#   stop_strings – tokens that signal end of generation
#   description  – human-readable label
#
# The ``{target_lang}`` placeholder inside the template is resolved at
# runtime by ``_resolve_template`` so that one generic entry can serve
# many language pairs.

PROMPT_FORMATS = {
    # ------------------------------------------------------------------
    # HY-MT 1.5  (Tencent HunYuan)
    # ------------------------------------------------------------------
    "hymt": {
        "template": (
            "Translate the following text into {target_lang}, please only "
            "output the translated result without additional explanation:"
            "\n\n{source}<|extra_0|>"
        ),
        "description": "HY-MT1.5 (generic language)",
        "stop_strings": ["<|extra_0|>", "<|endoftext|>"],
    },
    # ------------------------------------------------------------------
    # Qwen3 chat
    # ------------------------------------------------------------------
    "qwen3": {
        "template": (
            "<|im_start|>user\n"
            "You are a professional translator. "
            "Produce only the {target_lang} translation, without any additional "
            "explanations or commentary. Please translate the following text "
            "into {target_lang}:\n\n\n"
            "{source}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        ),
        "description": "Qwen3 (no-think mode)",
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
    },
    # ------------------------------------------------------------------
    # Qwen3.5 chat (hybrid GDN architecture)
    # ------------------------------------------------------------------
    "qwen3.5": {
        "template": (
            "<|im_start|>user\n"
            "You are a professional translator. "
            "Produce only the {target_lang} translation, without any additional "
            "explanations or commentary. Please translate the following text "
            "into {target_lang}:\n\n\n"
            "{source}<|im_end|>\n"
            "<|im_start|>assistant\n"
            "<think>\n\n</think>\n\n"
        ),
        "description": "Qwen3.5 (no-think mode, hybrid GDN)",
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
    },
    # ------------------------------------------------------------------
    # EuroLLM-9B-Instruct
    # ------------------------------------------------------------------
    "eurollm": {
        "template": (
            "<|im_start|>system\n"
            "You are a professional simultaneous interpreter at an academic conference. "
            "Translate the following text into {target_lang}.<|im_end|>\n"
            "<|im_start|>user\n{source}<|im_end|>\n"
            "<|im_start|>assistant\n"
        ),
        "description": "EuroLLM-9B-Instruct",
        "stop_strings": ["<|im_end|>", "<|endoftext|>"],
    },
    # ------------------------------------------------------------------
    # Seed-X (ByteDance)
    # ------------------------------------------------------------------
    "seedx": {
        "template": "Translate the following sentence into {target_lang}:\n{source} <{target_tag}>",
        "description": "Seed-X-PPO-7B (ByteDance)",
        "stop_strings": ["</s>", "<|endoftext|>"],
    },
    # ------------------------------------------------------------------
    # Generic / custom – user provides full template via --custom-template
    # ------------------------------------------------------------------
    "custom": {
        "template": "{source}",
        "description": "User-provided custom template",
        "stop_strings": ["<|endoftext|>"],
    },
}

# ===========================================================================
# Language configuration
# ===========================================================================

# Mapping from ISO-639-1 code to human-readable name (for prompt templates)
_LANG_NAMES = {
    "en": "English", "fr": "French", "de": "German", "es": "Spanish",
    "pt": "Portuguese", "it": "Italian", "nl": "Dutch", "pl": "Polish",
    "tr": "Turkish", "vi": "Vietnamese", "id": "Indonesian", "cs": "Czech",
    "ro": "Romanian", "hu": "Hungarian", "sv": "Swedish", "da": "Danish",
    "fi": "Finnish", "el": "Greek", "bg": "Bulgarian", "hr": "Croatian",
    "sk": "Slovak", "sl": "Slovenian", "lt": "Lithuanian", "lv": "Latvian",
    "et": "Estonian", "zh": "Chinese", "ja": "Japanese", "ko": "Korean",
    "ru": "Russian", "ar": "Arabic", "uk": "Ukrainian", "th": "Thai",
    "he": "Hebrew", "hi": "Hindi", "bn": "Bengali", "ms": "Malay",
    "fa": "Persian",
}

# Seed-X uses short tags like <fr>, <de>, etc.
_LANG_TAGS = {k: k for k in _LANG_NAMES}
_LANG_TAGS["zh"] = "zh"

# FLORES+ identifiers (iso_639_3, iso_15924)
_FLORES_CODES = {
    "en": ("eng", "Latn"), "fr": ("fra", "Latn"), "de": ("deu", "Latn"),
    "es": ("spa", "Latn"), "pt": ("por", "Latn"), "it": ("ita", "Latn"),
    "nl": ("nld", "Latn"), "pl": ("pol", "Latn"), "tr": ("tur", "Latn"),
    "vi": ("vie", "Latn"), "id": ("ind", "Latn"), "cs": ("ces", "Latn"),
    "ro": ("ron", "Latn"), "hu": ("hun", "Latn"), "sv": ("swe", "Latn"),
    "da": ("dan", "Latn"), "fi": ("fin", "Latn"), "el": ("ell", "Grek"),
    "bg": ("bul", "Cyrl"), "hr": ("hrv", "Latn"), "sk": ("slk", "Latn"),
    "sl": ("slv", "Latn"), "lt": ("lit", "Latn"), "lv": ("lvs", "Latn"),
    "et": ("est", "Latn"), "zh": ("cmn", "Hans"), "ja": ("jpn", "Jpan"),
    "ko": ("kor", "Hang"), "ru": ("rus", "Cyrl"), "ar": ("arb", "Arab"),
    "uk": ("ukr", "Cyrl"), "th": ("tha", "Thai"), "he": ("heb", "Hebr"),
    "hi": ("hin", "Deva"), "bn": ("ben", "Beng"), "ms": ("zsm", "Latn"),
    "fa": ("pes", "Arab"),
}

# CJK languages where we align on character level rather than word level
_CJK_LANGS = {"zh", "ja", "ko"}

# ===========================================================================
# Built-in test sentence pairs (used when FLORES is not available)
# ===========================================================================

_BUILTIN_SENTENCES = {
    "en-fr": [
        ("The weather is nice today.", "Le temps est beau aujourd'hui."),
        ("I would like a cup of coffee.", "Je voudrais une tasse de cafe."),
        ("The train arrives at three o'clock.", "Le train arrive a trois heures."),
        ("She reads a book every evening.", "Elle lit un livre chaque soir."),
        ("We need to find a solution.", "Nous devons trouver une solution."),
        ("The children are playing in the park.", "Les enfants jouent dans le parc."),
        ("He works in a large hospital.", "Il travaille dans un grand hopital."),
        ("The meeting starts in ten minutes.", "La reunion commence dans dix minutes."),
        ("They traveled to Paris last summer.", "Ils ont voyage a Paris l'ete dernier."),
        ("The cat is sleeping on the sofa.", "Le chat dort sur le canape."),
    ],
    "en-de": [
        ("The weather is nice today.", "Das Wetter ist heute schoen."),
        ("I would like a cup of coffee.", "Ich moechte eine Tasse Kaffee."),
        ("The train arrives at three o'clock.", "Der Zug kommt um drei Uhr an."),
        ("She reads a book every evening.", "Sie liest jeden Abend ein Buch."),
        ("We need to find a solution.", "Wir muessen eine Loesung finden."),
        ("The children are playing in the park.", "Die Kinder spielen im Park."),
        ("He works in a large hospital.", "Er arbeitet in einem grossen Krankenhaus."),
        ("The meeting starts in ten minutes.", "Das Treffen beginnt in zehn Minuten."),
        ("They traveled to Berlin last summer.", "Sie sind letzten Sommer nach Berlin gereist."),
        ("The cat is sleeping on the sofa.", "Die Katze schlaeft auf dem Sofa."),
    ],
    "en-es": [
        ("The weather is nice today.", "El tiempo esta bonito hoy."),
        ("I would like a cup of coffee.", "Me gustaria una taza de cafe."),
        ("The train arrives at three o'clock.", "El tren llega a las tres."),
        ("She reads a book every evening.", "Ella lee un libro cada noche."),
        ("We need to find a solution.", "Necesitamos encontrar una solucion."),
        ("The children are playing in the park.", "Los ninos juegan en el parque."),
        ("He works in a large hospital.", "El trabaja en un gran hospital."),
        ("The meeting starts in ten minutes.", "La reunion empieza en diez minutos."),
        ("They traveled to Madrid last summer.", "Ellos viajaron a Madrid el verano pasado."),
        ("The cat is sleeping on the sofa.", "El gato esta durmiendo en el sofa."),
    ],
    "en-it": [
        ("The weather is nice today.", "Il tempo e bello oggi."),
        ("I would like a cup of coffee.", "Vorrei una tazza di caffe."),
        ("The train arrives at three o'clock.", "Il treno arriva alle tre."),
        ("She reads a book every evening.", "Lei legge un libro ogni sera."),
        ("We need to find a solution.", "Dobbiamo trovare una soluzione."),
        ("The children are playing in the park.", "I bambini giocano nel parco."),
        ("He works in a large hospital.", "Lui lavora in un grande ospedale."),
        ("The meeting starts in ten minutes.", "La riunione inizia tra dieci minuti."),
        ("They traveled to Rome last summer.", "Hanno viaggiato a Roma l'estate scorsa."),
        ("The cat is sleeping on the sofa.", "Il gatto dorme sul divano."),
    ],
    "en-zh": [
        ("The weather is nice today.", "今天天气很好。"),
        ("I would like a cup of coffee.", "我想要一杯咖啡。"),
        ("The train arrives at three o'clock.", "火车三点钟到达。"),
        ("She reads a book every evening.", "她每天晚上都读一本书。"),
        ("We need to find a solution.", "我们需要找到一个解决方案。"),
        ("The children are playing in the park.", "孩子们在公园里玩耍。"),
        ("He works in a large hospital.", "他在一家大医院工作。"),
        ("The meeting starts in ten minutes.", "会议十分钟后开始。"),
        ("They traveled to Beijing last summer.", "他们去年夏天去了北京旅行。"),
        ("The cat is sleeping on the sofa.", "猫在沙发上睡觉。"),
    ],
}


# ===========================================================================
# Prompt building and source range detection
# ===========================================================================

def _resolve_template(fmt, target_lang):
    """Resolve ``{target_lang}`` and ``{target_tag}`` in a format template."""
    tgt_name = _LANG_NAMES.get(target_lang, target_lang)
    tgt_tag = _LANG_TAGS.get(target_lang, target_lang)
    tpl = fmt["template"]
    tpl = tpl.replace("{target_lang}", tgt_name)
    tpl = tpl.replace("{target_tag}", tgt_tag)
    return tpl


def build_prompt(source_text, template):
    """Build prompt string by inserting source text into a resolved template."""
    return template.replace("{source}", source_text)


def find_source_range(ll, vocab, source_text, template):
    """Find the token range [src_start, src_end) of the source text."""
    marker = "{source}"
    idx = template.find(marker)
    if idx == -1:
        return 0, 0

    prefix = template[:idx]
    suffix = template[idx + len(marker):]

    full_prompt = prefix + source_text + suffix
    prefix_tokens = ll.tokenize(vocab, prefix, add_bos=True, special=True)
    full_tokens = ll.tokenize(vocab, full_prompt, add_bos=True, special=True)
    suffix_tokens = ll.tokenize(vocab, suffix, add_bos=False, special=True)

    src_start = len(prefix_tokens)
    src_end = len(full_tokens) - len(suffix_tokens)
    return (src_start, src_end) if src_end > src_start else (0, 0)


# ===========================================================================
# Token-to-word mapping utilities
# ===========================================================================

def tokens_to_word_map(token_strings):
    """Map subword tokens to word indices (word boundary = leading space)."""
    word2tokens = defaultdict(list)
    word_idx = -1
    for i, tok in enumerate(token_strings):
        if (tok.startswith("\u0120") or tok.startswith(" ")
                or tok.startswith("\u2581") or word_idx == -1):
            word_idx += 1
        word2tokens[word_idx].append(i)
    return dict(word2tokens)


def reconstruct_words(token_strings):
    """Reconstruct word list from sub-word token strings."""
    words = []
    current = ""
    for tok in token_strings:
        if tok.startswith("\u0120") or tok.startswith(" ") or tok.startswith("\u2581"):
            if current:
                words.append(current)
            current = tok.lstrip("\u0120 \u2581")
        else:
            current += tok
    if current:
        words.append(current)
    return words


def reconstruct_cjk_chars(token_strings):
    """Reconstruct character-level list for CJK output."""
    text = ""
    for tok in token_strings:
        clean = tok.lstrip("\u0120 \u2581")
        text += clean
    return [c for c in text if c.strip()]


def cjk_char_to_token_map(token_strings):
    """Map individual CJK characters back to token positions."""
    char2tokens = {}
    char_idx = 0
    for tok_idx, tok in enumerate(token_strings):
        clean = tok.lstrip("\u0120 \u2581")
        for _ in clean:
            char2tokens.setdefault(char_idx, []).append(tok_idx)
            char_idx += 1
    return char2tokens


# ===========================================================================
# Word alignment (SimAlign or monotonic fallback)
# ===========================================================================

class _MonotonicAligner:
    """Fallback aligner when SimAlign is not installed.

    Uses a simple monotonic heuristic: word *i* in the source roughly
    aligns to word ``i * (len_tgt / len_src)`` in the target.
    """

    @staticmethod
    def get_word_aligns(src_words, tgt_words):
        n_src = len(src_words)
        n_tgt = len(tgt_words)
        if n_src == 0 or n_tgt == 0:
            return {"itermax": []}
        ratio = n_tgt / n_src
        aligns = []
        for i in range(n_src):
            j = min(int(i * ratio + 0.5), n_tgt - 1)
            aligns.append((i, j))
        # Also add reverse mapping for coverage
        ratio_r = n_src / n_tgt
        for j in range(n_tgt):
            i = min(int(j * ratio_r + 0.5), n_src - 1)
            pair = (i, j)
            if pair not in aligns:
                aligns.append(pair)
        return {"itermax": aligns}


def _get_aligner(use_simalign=True):
    """Return a word aligner instance."""
    if use_simalign and _HAS_SIMALIGN:
        try:
            return _SentenceAligner(model="bert", token_type="bpe")
        except Exception:
            pass
    return _MonotonicAligner()


# ===========================================================================
# Data loading helpers
# ===========================================================================

def _load_flores(src_lang, tgt_lang, num_sentences):
    """Attempt to load sentence pairs from FLORES+ dataset.

    Returns list of (source_text, reference_text) or None if unavailable.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        return None

    src_flores = _FLORES_CODES.get(src_lang)
    tgt_flores = _FLORES_CODES.get(tgt_lang)
    if src_flores is None or tgt_flores is None:
        return None

    try:
        ds = load_dataset("openlanguagedata/flores_plus", split="dev")
    except Exception:
        return None

    src_ds = ds.filter(
        lambda x: x["iso_639_3"] == src_flores[0] and x["iso_15924"] == src_flores[1]
    )
    tgt_ds = ds.filter(
        lambda x: x["iso_639_3"] == tgt_flores[0] and x["iso_15924"] == tgt_flores[1]
    )
    src_map = {row["id"]: row["text"] for row in src_ds}
    tgt_map = {row["id"]: row["text"] for row in tgt_ds}
    common_ids = sorted(set(src_map) & set(tgt_map))
    if not common_ids:
        return None

    n = min(num_sentences, len(common_ids))
    return [(src_map[sid], tgt_map[sid]) for sid in common_ids[:n]]


def _load_sentences(src_lang, tgt_lang, num_sentences):
    """Load sentence pairs from FLORES+ or built-in fallback."""
    # Try FLORES first
    pairs = _load_flores(src_lang, tgt_lang, num_sentences)
    if pairs:
        return pairs, "FLORES+"

    # Fall back to built-in pairs
    key = f"{src_lang}-{tgt_lang}"
    if key in _BUILTIN_SENTENCES:
        builtin = _BUILTIN_SENTENCES[key]
        return builtin[:num_sentences], "built-in"

    # If we only have built-in for en-fr and en-de, allow en-* to fall back
    # to en-fr as a reasonable proxy (cross-lingual heads are often shared)
    if src_lang == "en" and "en-fr" in _BUILTIN_SENTENCES:
        builtin = _BUILTIN_SENTENCES["en-fr"]
        return builtin[:num_sentences], "built-in (en-fr proxy)"

    return [], "none"


# ===========================================================================
# Core detection algorithm
# ===========================================================================

def detect_translation_heads(
    model_path,
    source_sentences=None,
    reference_translations=None,
    prompt_format="hymt",
    source_lang="en",
    target_lang="fr",
    num_sentences=50,
    n_ctx=2048,
    max_gen=256,
    batch_heads=0,
    ts_threshold=0.1,
    top_k=20,
    skip_gdn_layers=False,
    use_simalign=True,
    custom_template=None,
    verbose=False,
    suppress_model_log=True,
):
    """Detect translation alignment heads in a GGUF model.

    Parameters
    ----------
    model_path : str
        Path to the GGUF model file.
    source_sentences : list[str], optional
        Source sentences.  If not provided, loaded from FLORES+ or built-in.
    reference_translations : list[str], optional
        Reference translations (used only for word alignment, not scoring).
        If not provided, the model's own greedy output is used.
    prompt_format : str
        One of the keys in :data:`PROMPT_FORMATS`.
    source_lang, target_lang : str
        ISO-639-1 language codes (e.g. ``"en"``, ``"fr"``).
    num_sentences : int
        Number of sentence pairs to process.
    n_ctx : int
        Context window size for llama.cpp.
    max_gen : int
        Maximum tokens to generate per sentence.
    batch_heads : int
        Process heads in batches of this size (0 = all at once).
    ts_threshold : float
        Minimum TS score to include a head in the output.
    top_k : int
        Maximum number of heads to return.
    skip_gdn_layers : bool
        Skip GDN/linear-attention layers (for Qwen3.5 hybrid models).
    use_simalign : bool
        Prefer SimAlign when available; fall back to monotonic otherwise.
    custom_template : str, optional
        Override prompt template (must contain ``{source}``).
    verbose : bool
        Print progress information.
    suppress_model_log : bool
        Suppress llama.cpp stderr logs.

    Returns
    -------
    dict
        JSON-serialisable result with keys:
        ``model``, ``prompt_format``, ``language_pair``, ``num_layers``,
        ``num_heads``, ``num_sentences``, ``total_alignable_tokens``,
        ``ts_threshold``, ``ts_matrix``, ``token_alignment_heads``.
    """
    from nllw import llama_backend as ll

    is_cjk = target_lang in _CJK_LANGS
    lang_pair = f"{source_lang}-{target_lang}"

    # -- Resolve prompt format -------------------------------------------
    if prompt_format not in PROMPT_FORMATS:
        raise ValueError(
            f"Unknown prompt format: {prompt_format}. "
            f"Choose from: {', '.join(PROMPT_FORMATS)}"
        )
    fmt = PROMPT_FORMATS[prompt_format]
    if custom_template is not None:
        fmt = dict(fmt)
        fmt["template"] = custom_template

    resolved_template = _resolve_template(fmt, target_lang)

    # -- Load sentence pairs ---------------------------------------------
    if source_sentences is not None:
        pairs = list(zip(
            source_sentences,
            reference_translations or [None] * len(source_sentences),
        ))
        data_source = "user-provided"
    else:
        loaded, data_source = _load_sentences(source_lang, target_lang, num_sentences)
        if not loaded:
            raise RuntimeError(
                f"No sentence pairs available for {lang_pair}. "
                "Install 'datasets' and 'openlanguagedata/flores_plus', "
                "or provide source_sentences/reference_translations."
            )
        pairs = loaded

    num_sentences = min(num_sentences, len(pairs))
    pairs = pairs[:num_sentences]

    if verbose:
        print(f"Data source: {data_source} ({num_sentences} pairs)")

    # -- Load word aligner -----------------------------------------------
    aligner = _get_aligner(use_simalign=use_simalign)
    aligner_name = type(aligner).__name__
    if verbose:
        print(f"Word aligner: {aligner_name}")

    # -- Init llama.cpp --------------------------------------------------
    if verbose:
        print(f"Loading model: {model_path}")

    if suppress_model_log:
        ll.suppress_stderr()

    ll.init(suppress_log=suppress_model_log)
    model = ll.load_model(model_path, suppress_log=suppress_model_log)

    if suppress_model_log:
        ll.restore_stderr()

    vocab = ll.get_vocab(model)
    nv = ll.n_vocab(vocab)
    eos_id = ll.vocab_eos(vocab)
    num_layers = ll.n_layer(model)
    num_heads_per_layer = ll.n_head(model)

    if verbose:
        print(f"  {num_layers} layers x {num_heads_per_layer} heads "
              f"= {num_layers * num_heads_per_layer} total")

    # Build stop IDs
    stop_ids = {eos_id}
    for tok_str in fmt["stop_strings"]:
        tok_ids = ll.tokenize(vocab, tok_str, add_bos=False, special=True)
        if len(tok_ids) == 1:
            stop_ids.add(tok_ids[0])
    # Common stop tokens across models
    for sid in [2, 151643, 151645]:
        stop_ids.add(sid)

    if verbose:
        print(f"  Stop IDs: {stop_ids}")

    # -- Set up head pairs -----------------------------------------------
    all_layers = []
    all_head_ids = []
    for layer in range(num_layers):
        if skip_gdn_layers and (layer % 4 != 3):
            # Qwen3.5 hybrid: full-attention layers at 3, 7, 11, ...
            continue
        for head in range(num_heads_per_layer):
            all_layers.append(layer)
            all_head_ids.append(head)

    total_heads = len(all_layers)
    batch_size = batch_heads if batch_heads > 0 else total_heads
    n_batches = (total_heads + batch_size - 1) // batch_size

    if verbose:
        if n_batches > 1:
            print(f"  Processing heads in {n_batches} batches of {batch_size}")
        else:
            print(f"  Processing all {total_heads} heads at once")

    # -- Process sentences -----------------------------------------------
    if verbose:
        print(f"\nProcessing {num_sentences} sentences...\n")

    g = np.zeros(total_heads, dtype=np.int64)
    m = 0  # total alignable token pairs

    t0 = time.time()
    for sent_idx, pair in enumerate(pairs):
        if isinstance(pair, (list, tuple)) and len(pair) >= 2:
            source_text, ref_text = pair[0], pair[1]
        else:
            source_text = pair
            ref_text = None

        # Build prompt
        prompt = build_prompt(source_text, resolved_template)
        prompt_tokens = ll.tokenize(vocab, prompt, add_bos=True, special=True)
        prompt_len = len(prompt_tokens)

        # Find source range
        src_start, src_end = find_source_range(ll, vocab, source_text, resolved_template)
        if src_end <= src_start:
            if verbose:
                print(f"  [{sent_idx+1}/{num_sentences}] SKIP (empty source range)")
            continue

        source_positions = list(range(src_start, src_end))
        source_token_strings = [
            ll.token_to_piece(vocab, prompt_tokens[i])
            for i in range(src_start, src_end)
        ]

        # Process in head batches
        # Cache alignment info across batches for the same sentence
        _word_aligns = None
        _src_word2tok = None
        _tgt_char2tok = None
        generated_ids = None

        for batch_idx in range(n_batches):
            b_start = batch_idx * batch_size
            b_end = min(b_start + batch_size, total_heads)
            b_layers = all_layers[b_start:b_end]
            b_heads = all_head_ids[b_start:b_end]
            n_pairs_batch = b_end - b_start

            # Create context with attention weights
            if suppress_model_log:
                ll.suppress_stderr()
            ctx = ll.create_context(model, n_ctx=n_ctx, n_batch=n_ctx)
            ll.set_attn_heads(ctx, b_layers, b_heads)
            if suppress_model_log:
                ll.restore_stderr()

            # Decode prompt
            ll.decode_batch(ctx, prompt_tokens)
            pos = prompt_len

            # Generate tokens greedily
            batch_generated_ids = []
            step_argmaxes = []

            for step in range(max_gen):
                # After decode_batch, logits at batch index (n-1) for first step,
                # then at batch index 0 after decode_single
                logit_idx = prompt_len - 1 if step == 0 else 0
                next_tok = ll.argmax_logits(ctx, logit_idx, nv)
                if next_tok in stop_ids or next_tok < 0:
                    break

                batch_generated_ids.append(next_tok)

                # Decode token to compute attention
                ll.decode_single(ctx, next_tok, pos)
                pos += 1

                # Extract attention weights for this step
                ctx_size = ll.n_ctx(ctx)
                attn = ll.get_attn_weights(ctx, 0, n_pairs_batch, ctx_size)
                if attn is not None and src_end <= attn.shape[1]:
                    src_attn = attn[:, src_start:src_end]
                    argmaxes = np.argmax(src_attn, axis=1)  # (n_pairs_batch,)
                    step_argmaxes.append(argmaxes)
                else:
                    step_argmaxes.append(None)

            ll.free_context(ctx)

            if not batch_generated_ids:
                continue

            num_gen = len(batch_generated_ids)

            # Use the generated IDs from the first batch (all batches produce
            # the same greedy output since the model is deterministic)
            if batch_idx == 0:
                generated_ids = batch_generated_ids

            # Get output token strings
            output_tokens = [ll.token_to_piece(vocab, t) for t in batch_generated_ids]

            # Word-level alignments (only on first batch to avoid redundant work)
            if batch_idx == 0:
                # Build source and target word/char lists
                src_words = reconstruct_words(source_token_strings)
                if is_cjk:
                    tgt_units = reconstruct_cjk_chars(output_tokens)
                else:
                    tgt_units = reconstruct_words(output_tokens)

                if not src_words or not tgt_units:
                    break  # skip all batches for this sentence

                try:
                    alignments = aligner.get_word_aligns(src_words, tgt_units)
                except Exception:
                    break

                word_aligns = alignments.get("itermax", alignments.get("inter", []))
                if not word_aligns:
                    break

                # Build token maps
                src_word2tok = tokens_to_word_map(source_token_strings)
                if is_cjk:
                    tgt_char2tok = cjk_char_to_token_map(output_tokens)
                else:
                    tgt_char2tok = tokens_to_word_map(output_tokens)

                # Cache for subsequent batches
                _word_aligns = word_aligns
                _src_word2tok = src_word2tok
                _tgt_char2tok = tgt_char2tok
            else:
                # Reuse alignment from first batch
                word_aligns = _word_aligns
                src_word2tok = _src_word2tok
                tgt_char2tok = _tgt_char2tok

                if word_aligns is None:
                    break  # alignment failed in first batch

            # Score heads based on word alignments
            for src_widx, tgt_cidx in word_aligns:
                if src_widx not in src_word2tok or tgt_cidx not in tgt_char2tok:
                    continue

                src_abs_positions = set(
                    source_positions[ti]
                    for ti in src_word2tok[src_widx]
                    if ti < len(source_positions)
                )
                if not src_abs_positions:
                    continue

                for tgt_step in tgt_char2tok[tgt_cidx]:
                    if tgt_step >= num_gen or tgt_step >= len(step_argmaxes):
                        continue
                    if step_argmaxes[tgt_step] is None:
                        continue

                    if batch_idx == 0:
                        m += 1

                    argmaxes = step_argmaxes[tgt_step]
                    for h_offset in range(n_pairs_batch):
                        abs_head_idx = b_start + h_offset
                        # argmax is relative to src_start, convert to absolute
                        abs_pos = src_start + int(argmaxes[h_offset])
                        if abs_pos in src_abs_positions:
                            g[abs_head_idx] += 1

        # Progress
        if verbose:
            elapsed = time.time() - t0
            avg = elapsed / (sent_idx + 1)
            eta = avg * (num_sentences - sent_idx - 1)
            preview = ""
            if generated_ids:
                preview = ll.tokens_to_text(vocab, generated_ids[:20])
            print(
                f"  [{sent_idx+1}/{num_sentences}] "
                f"m={m} | {preview}... | "
                f"{avg:.1f}s/sent | ETA: {eta:.0f}s",
                flush=True,
            )

    elapsed = time.time() - t0
    if verbose:
        print(f"\nDone in {elapsed:.1f}s. Total alignable tokens: m={m}")

    ll.free_model(model)
    ll.cleanup()

    # -- Compute Translation Scores --------------------------------------
    ts = g / max(m, 1)

    # Build full TS matrix (num_layers x num_heads_per_layer)
    probed_layers = sorted(set(all_layers))
    ts_matrix = np.zeros((num_layers, num_heads_per_layer))

    idx = 0
    for layer in probed_layers:
        for head in range(num_heads_per_layer):
            ts_matrix[layer, head] = ts[idx]
            idx += 1

    # Identify heads above threshold
    tah = []
    for layer in probed_layers:
        for head in range(num_heads_per_layer):
            score = ts_matrix[layer, head]
            if score > ts_threshold:
                tah.append({
                    "layer": int(layer),
                    "head": int(head),
                    "ts": round(float(score), 4),
                })

    tah.sort(key=lambda x: x["ts"], reverse=True)

    # Apply top-k limit
    if top_k > 0 and len(tah) > top_k:
        tah = tah[:top_k]

    if verbose:
        print(f"\n{'='*60}")
        print(f"TOKEN ALIGNMENT HEADS (TS > {ts_threshold}): "
              f"{len(tah)} / {total_heads}")
        print(f"{'='*60}")
        for entry in tah:
            bar = "#" * int(entry["ts"] * 50)
            print(f"  L{entry['layer']:2d} H{entry['head']:2d} : "
                  f"TS={entry['ts']:.4f}  {bar}")

        n_active = int(np.sum(ts > ts_threshold))
        n_low = int(np.sum((ts > 0) & (ts <= ts_threshold)))
        n_zero = int(np.sum(ts == 0))
        print(f"\nDistribution:")
        print(f"  TS > {ts_threshold} (alignment heads): "
              f"{n_active} ({100*n_active/max(total_heads,1):.1f}%)")
        print(f"  0 < TS <= {ts_threshold} (low activity): "
              f"{n_low} ({100*n_low/max(total_heads,1):.1f}%)")
        print(f"  TS = 0 (inactive): "
              f"{n_zero} ({100*n_zero/max(total_heads,1):.1f}%)")

    # -- Build result dict -----------------------------------------------
    result = {
        "model": os.path.basename(model_path),
        "prompt_format": prompt_format,
        "language_pair": lang_pair,
        "num_layers": int(num_layers),
        "num_heads": int(num_heads_per_layer),
        "num_sentences": int(num_sentences),
        "total_alignable_tokens": int(m),
        "ts_threshold": ts_threshold,
        "ts_matrix": ts_matrix.tolist(),
        "token_alignment_heads": tah,
    }
    return result


# ===========================================================================
# CLI entry point
# ===========================================================================

def _parse_lang_pair(lang_str):
    """Parse a language pair string like 'en-fr' into (src, tgt)."""
    parts = lang_str.lower().split("-")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError(
            f"Language pair must be in format 'src-tgt' (e.g. 'en-fr'), got: {lang_str}"
        )
    return parts[0], parts[1]


def main():
    parser = argparse.ArgumentParser(
        description="Detect translation alignment heads in a GGUF model for NLLW",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m nllw.detect_heads --model model.gguf --prompt-format hymt --lang en-fr\n"
            "  python -m nllw.detect_heads --model model.gguf --prompt-format qwen3 --lang en-de -n 100\n"
            "  python -m nllw.detect_heads --model model.gguf --prompt-format qwen3.5 --lang en-zh "
            "--skip-gdn-layers\n"
        ),
    )
    parser.add_argument(
        "--model", required=True,
        help="Path to GGUF model file",
    )
    parser.add_argument(
        "--prompt-format", default="hymt",
        choices=list(PROMPT_FORMATS.keys()),
        help="Prompt format to use (default: hymt)",
    )
    parser.add_argument(
        "--lang", default="en-fr",
        help="Language pair as 'src-tgt' (e.g. en-fr, en-de, en-zh). Default: en-fr",
    )
    parser.add_argument(
        "-n", "--num-sentences", type=int, default=50,
        help="Number of sentences to process (default: 50)",
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output JSON path (default: auto-derived from model and lang)",
    )
    parser.add_argument(
        "--ts-threshold", type=float, default=0.1,
        help="Minimum TS score to include a head (default: 0.1)",
    )
    parser.add_argument(
        "--top-k", type=int, default=20,
        help="Limit output to top K heads (default: 20, 0=unlimited)",
    )
    parser.add_argument(
        "--n-ctx", type=int, default=2048,
        help="Context window size (default: 2048)",
    )
    parser.add_argument(
        "--max-gen", type=int, default=256,
        help="Max tokens to generate per sentence (default: 256)",
    )
    parser.add_argument(
        "--batch-heads", type=int, default=0,
        help="Process heads in batches of this size (0=all at once, default: 0)",
    )
    parser.add_argument(
        "--skip-gdn-layers", action="store_true",
        help="Skip GDN/linear-attention layers (for Qwen3.5 hybrid models)",
    )
    parser.add_argument(
        "--no-simalign", action="store_true",
        help="Force monotonic fallback alignment even if simalign is installed",
    )
    parser.add_argument(
        "--custom-template", default=None,
        help="Custom prompt template (must contain {source})",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true",
        help="Show progress during processing",
    )
    args = parser.parse_args()

    # Parse language pair
    src_lang, tgt_lang = _parse_lang_pair(args.lang)

    # Auto-derive output path
    if args.output is None:
        model_base = os.path.basename(args.model).replace(".gguf", "").replace(".", "_")
        lang_tag = args.lang.replace("-", "_")
        args.output = f"translation_heads_{model_base}_{lang_tag}.json"

    print(f"Model:         {args.model}")
    print(f"Prompt format: {args.prompt_format}")
    print(f"Language pair: {src_lang} -> {tgt_lang}")
    print(f"Sentences:     {args.num_sentences}")
    print(f"Output:        {args.output}")
    if args.skip_gdn_layers:
        print(f"GDN skip:      enabled (hybrid model)")
    if args.no_simalign or not _HAS_SIMALIGN:
        aligner_info = "monotonic fallback"
        if not _HAS_SIMALIGN:
            aligner_info += " (simalign not installed)"
        print(f"Aligner:       {aligner_info}")
    else:
        print(f"Aligner:       simalign (mBERT)")
    print()

    # Run detection
    result = detect_translation_heads(
        model_path=args.model,
        prompt_format=args.prompt_format,
        source_lang=src_lang,
        target_lang=tgt_lang,
        num_sentences=args.num_sentences,
        n_ctx=args.n_ctx,
        max_gen=args.max_gen,
        batch_heads=args.batch_heads,
        ts_threshold=args.ts_threshold,
        top_k=args.top_k,
        skip_gdn_layers=args.skip_gdn_layers,
        use_simalign=not args.no_simalign,
        custom_template=args.custom_template,
        verbose=True,  # CLI always verbose
    )

    # Save JSON
    with open(args.output, "w") as f:
        json.dump(result, f, indent=2)
    print(f"\nResults saved to {args.output}")

    # Summary
    tah = result["token_alignment_heads"]
    print(f"\nTop alignment heads ({len(tah)} with TS > {args.ts_threshold}):")
    for entry in tah[:10]:
        print(f"  Layer {entry['layer']:2d}, Head {entry['head']:2d}: "
              f"TS = {entry['ts']:.4f}")
    if len(tah) > 10:
        print(f"  ... and {len(tah) - 10} more (see {args.output})")


if __name__ == "__main__":
    main()
