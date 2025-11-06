import logging
import time
import torch
import transformers
from dataclasses import dataclass, field
from typing import Optional, Union
import huggingface_hub
from nllw.timed_text import TimedText
# try:
#     import ctranslate2
#     CTRANSLATE2_AVAILABLE = True
# except ImportError:
#     CTRANSLATE2_AVAILABLE = False
#     ctranslate2 = None

from .languages import convert_to_nllb_code
from .core import TranslationBackend

"""
Interface for WhisperLiveKit. For other usages, it may be wiser to look at nllw.core directly.
"""


logger = logging.getLogger(__name__)
MIN_SILENCE_DURATION_DEL_BUFFER = 1.0


@dataclass
class TranslationModel():
    translator: Union['ctranslate2.Translator', 'transformers.AutoModelForSeq2SeqLM']
    device: str
    tokenizer: dict = field(default_factory=dict)
    backend_type: str = 'transformers'
    nllb_size: str = '600M'
    model_name: str = field(default='')

    def get_tokenizer(self, input_lang):
        if not self.tokenizer.get(input_lang, False):
            model_name = self.model_name or f"facebook/nllb-200-distilled-{self.nllb_size}"
            self.tokenizer[input_lang] = transformers.AutoTokenizer.from_pretrained(
                model_name,
                src_lang=input_lang,
                clean_up_tokenization_spaces=True
            )
        return self.tokenizer[input_lang]


def load_model(src_langs, nllb_backend='transformers', nllb_size='600M'):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Convert all source languages to NLLB codes
    converted_src_langs = []
    for lang in src_langs:
        if lang == 'auto':
            converted_src_langs.append('auto')
        else:
            nllb_code = convert_to_nllb_code(lang)
            if nllb_code is None:
                raise ValueError(f"Unknown language identifier: {lang}")
            converted_src_langs.append(nllb_code)

    if nllb_backend == 'ctranslate2': #BROKEN
        if not CTRANSLATE2_AVAILABLE:
            raise ImportError("ctranslate2 is not installed. Install it with: pip install ctranslate2")
        model = f'nllb-200-distilled-{nllb_size}-ctranslate2'
        MODEL_GUY = 'entai2965'
        huggingface_hub.snapshot_download(MODEL_GUY + '/' + model, local_dir=model)
        translator = ctranslate2.Translator(model, device=device)
        model_name = model
    elif nllb_backend == 'transformers':
        model_name = f"facebook/nllb-200-distilled-{nllb_size}"
        translator = transformers.AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
    else:
        raise ValueError(f"Unknown backend: {nllb_backend}. Use 'transformers' or 'ctranslate2'")

    tokenizer = dict()
    for src_lang in converted_src_langs:
        if src_lang != 'auto':
            tokenizer[src_lang] = transformers.AutoTokenizer.from_pretrained(
                model_name if nllb_backend == 'transformers' else f"facebook/nllb-200-distilled-{nllb_size}",
                src_lang=src_lang,
                clean_up_tokenization_spaces=True
            )

    translation_model = TranslationModel(
        translator=translator,
        tokenizer=tokenizer,
        backend_type=nllb_backend,
        device=device,
        nllb_size=nllb_size,
        model_name=model_name if nllb_backend == 'transformers' else f"facebook/nllb-200-distilled-{nllb_size}"
    )
    return translation_model

class OnlineTranslation:
    def __init__(self, translation_model: TranslationModel, input_languages: list, output_languages: list):
        self.translation_model = translation_model        
        self.input_languages = []
        for lang in input_languages:
            if lang == 'auto':
                self.input_languages.append('auto')
            else:
                nllb_code = convert_to_nllb_code(lang)
                if nllb_code is None:
                    raise ValueError(f"Unknown input language identifier: {lang}")
                self.input_languages.append(nllb_code)
        
        self.output_languages = []
        for lang in output_languages:
            nllb_code = convert_to_nllb_code(lang)
            if nllb_code is None:
                raise ValueError(f"Unknown output language identifier: {lang}")
            self.output_languages.append(nllb_code)

        self.last_buffer = ''
        self.commited = []
        self.last_end_time: float = 0.0 

        self.backend = TranslationBackend(
            source_lang=self.input_languages[0],
            target_lang=self.output_languages[0],
            model_name=translation_model.model_name,
            model=translation_model.translator,
            tokenizer=translation_model.get_tokenizer(self.input_languages[0])
        )

    def insert_tokens(self, tokens):
        self.backend.input_buffer.extend(tokens)
    
    def process(self):
        if self.backend.input_buffer:
            start_time = self.last_end_time
            end_time = self.backend.input_buffer[-1].end
            self.last_end_time = end_time
        else:
            start_time = end_time = 0.0
        self.last_end_time
        stable_translation, buffer_text = self.backend.translate()
        validated = TimedText(
            text=stable_translation,
            start=start_time,
            end=end_time
        )

        buffer = TimedText(
            text=buffer_text,
            start=start_time,
            end=end_time
        )
        self.last_buffer = buffer
        self.commited.append(validated)
        return self.commited, buffer

    def insert_silence(self, silence_duration: float):
        if silence_duration >= MIN_SILENCE_DURATION_DEL_BUFFER:
            if self.last_buffer:
                if isinstance(self.last_buffer, str):
                    self.last_buffer = TimedText(text=self.last_buffer)
                self.commited.append(self.last_buffer)
            self.backend.input_buffer = [] #maybe need to reprocess stuff before inserting silence
            self.last_buffer = ''
