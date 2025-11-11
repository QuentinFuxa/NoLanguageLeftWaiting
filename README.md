<h1 align="center">NoLanguageLeftWaiting</h1>

<p align="center">
<img src="demo.gif"width="730">
</p>

<p align="center">
<img src="architecture_NLLW.png"width="730">
</p>

Converts [NoLanguageLeftBehind](https://arxiv.org/abs/2207.04672) translation model to a SimulMT (Simultaneous Machine Translation) model, optimized for live/streaming use cases.

> Based offline models such as NLLB suffer from eos token and punctuation insertion, inconsistent prefix handling and exponentially growing computational overhead as input length increases. This implementation aims at resolving that.


- [LocalAgreement policy](https://www.isca-archive.org/interspeech_2020/liu20s_interspeech.pdf)
- Backends: [HuggingFace transformers](https://huggingface.co/docs/transformers/model_doc/auto#transformers.AutoModelForSeq2SeqLM) / [Ctranslate2 Translator](https://opennmt.net/CTranslate2/python/ctranslate2.Translator.html#ctranslate2.Translator.translate_batch) (beta)
- Built for [WhisperLiveKit](https://github.com/QuentinFuxa/WhisperLiveKit)
- 200 languages. See [supported_languages.md](supported_languages.md) for the full list.
- Working on implementing a speculative/self-speculative decoding for a faster decoder, using 600M as draft model, and 1.3B as main model. Refs: https://arxiv.org/pdf/2211.17192:  https://arxiv.org/html/2509.21740v1, 

## Installation

```bash
pip install nllw
```
> The textual frontend is not installed by default.


## Quick Start

1. Demo interface :
```bash
python textual_interface.py
```

2. Use it as a package
```python
import nllw

model = nllw.load_model(
    src_langs=["fra_Latn"],
    nllb_backend="transformers",
    nllb_size="600M" #Alternative: 1.3B
)
translator = nllw.OnlineTranslation(
    model,
    input_languages=["fra_Latn"],
    output_languages=["eng_Latn"]
)

tokens = [nllw.timed_text.TimedText('Ceci est un test de traduction')]
translator.insert_tokens(tokens)
validated, buffer = translator.process()
print(f"{validated} | {buffer}")

tokens = [nllw.timed_text.TimedText('en temps réel')]
translator.insert_tokens(tokens)
validated, buffer = translator.process()
print(f"{validated} | {buffer}")
```

## Work In Progress : Partial Speculative Decoding

Local Agreement already locks a stable prefix for the committed translation, so we cannot directly adopt [Self-Speculative Biased Decoding for Faster Live Translation](https://arxiv.org/html/2509.21740v1). Our ongoing prototype instead borrows the speculative idea only for the *new* tokens that need to be validated by the larger model.

The flow tested in `speculative_decoding_v0.py`:
- Run the 600M draft decoder once to obtain the candidate continuation and its cache.
- Replay the draft tokens through the 1.3B model, but stop the forward pass as soon as the main model reproduces a token emitted by the draft (`predicted_tokens` matches the draft output). We keep those verified tokens and only continue generation from that point.
- On mismatch, resume full decoding with the 1.3B model until a match is reached again, instead of discarding the entire draft segment.

This “partial verification” trims the work the main decoder performs after each divergence, while keeping the responsiveness of the draft hypothesis. Early timing experiments from `speculative_decoding_v0.py` show the verification pass (~0.15 s in the example) is significantly cheaper than recomputing a full decoding step every time.

<p align="center">
<img src="partial_speculative_decoding.png"width="730">
</p>



## Input vs Output length:

Succesfully maintain output length, even if stable prefix tends to take time to grow.

<p align="center">
<img src="french_to_english.png"width="730">
</p>
