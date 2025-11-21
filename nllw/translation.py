import logging
from nllw.timed_text import TimedText

from .languages import convert_to_nllb_code
from .core import TranslationBackend, TranslationModel, load_model

"""
Interface for WhisperLiveKit. For other usages, it may be wiser to look at nllw.core directly.
"""


logger = logging.getLogger(__name__)
MIN_SILENCE_DURATION_DEL_BUFFER = 1.0

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

        self.last_buffer = TimedText()
        self.last_end_time: float = 0.0 

        self.backend = TranslationBackend(
            source_lang=self.input_languages[0],
            target_lang=self.output_languages[0],
            model_name=translation_model.model_name,
            model=translation_model.translator,
            tokenizer=translation_model.get_tokenizer(self.input_languages[0]),
            backend_type=translation_model.backend_type
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
        new_validated_translation = TimedText(
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
        return new_validated_translation, buffer

    def validate_buffer_and_reset(self, duration: float = None):
        pass

    def insert_silence(self, duration: float = None):
        pass
