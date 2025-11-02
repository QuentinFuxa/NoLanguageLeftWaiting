import torch
import time
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.cache_utils import EncoderDecoderCache, DynamicCache
from typing import Tuple, Optional
from nllw.timed_text import TimedText

"""
nllb-200-distilled: pytorch_model.bin : 2.46 GB
"""

PUNCTUATION_MARKS = {'.', '!', '?', '。', '！', '？'}


class TranslationBackend:
    def __init__(
        self,
        source_lang,
        target_lang,
        model_name: str = "facebook/nllb-200-distilled-600M",
        model=None,
        tokenizer=None,
        verbose: bool = False,
        max_kept_sentences: int = 1
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.verbose = verbose
        self.max_kept_sentences = max_kept_sentences

        if model is not None:
            self.model = model
            if not hasattr(model, 'device') or str(model.device) != self.device:
                self.model = self.model.to(self.device)
        else:
            self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)

        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_lang)
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)

        self.sentence_end_token_ids = set()

        # find all tokens that decode to sentence-ending punctuation. For ex: token 81 and 248075 both represent '.')
        for token_id in range(min(300000, self.tokenizer.vocab_size)):
            try:
                decoded = self.tokenizer.decode([token_id])
                cleaned = decoded.strip().strip("'\"").strip()
                if cleaned in PUNCTUATION_MARKS:
                    self.sentence_end_token_ids.add(token_id)
            except:
                pass

        self.input_buffer = []
        self.previous_tokens = []
        self.stable_prefix_segments = []
        self.stable_prefix_tokens = torch.tensor([], dtype=torch.int64)
        self.n_remaining_input_punctuation = 0

    def _trim(self) -> bool:
        x = 5
        # print(f'Before trimming: {len(self.input_buffer)} {len(self.stable_prefix_segments)}')
        self.input_buffer = self.input_buffer[-x-1:]
        self.stable_prefix_segments = self.stable_prefix_segments[-x:]
        # print(f'After trimming: {len(self.input_buffer)} {len(self.stable_prefix_segments)}')
    
    
    def simple_translation(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        if self.verbose:
            start_time = time.time()
        encoder_outputs = self.model.get_encoder()(**inputs)
        if self.verbose:
            encoding_time = time.time() - start_time
        
        if self.verbose:
            start_time = time.time()
        output = self.generate(
                    encoder_outputs=encoder_outputs,
        )
        if self.verbose:
            decoding_time = time.time() - start_time
            print(f"Encoding time: {encoding_time:.4f}s | Decoding time: {decoding_time:.4f}s")
        
        result = self.tokenizer.decode(output[0], skip_special_tokens=True)
        return output, result

    def generate(
        self,
        encoder_outputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_length: Optional[int] = 200
        
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:    
        with torch.no_grad():
            generated_tokens = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                forced_bos_token_id=self.bos_token_id,
                max_length=max_length,
            )
        return generated_tokens

    def has_sentence_end_token(self, tokens):
        last_sentence_end = -1
        for i in range(len(tokens)):
            if tokens[i].item() in self.sentence_end_token_ids:
                last_sentence_end = i
        return last_sentence_end

    def compute_common_prefix_tokens(self):
        common_length = 0
        for i in range(min(len(self.previous_tokens), len(self.new_produced_tokens))):
            if self.previous_tokens[i] != self.new_produced_tokens[i]:
                common_length = i
                break
        else:
            common_length = min(len(self.previous_tokens), len(self.new_produced_tokens))

        last_sentence_end = self.has_sentence_end_token(self.new_produced_tokens[:common_length])

        if last_sentence_end >= 0:
            if self.n_remaining_input_punctuation > 0:
                print(f"\033[33mEOS detected. Remaining punctuation: {self.n_remaining_input_punctuation}\033[0m")
                self.n_remaining_input_punctuation -= 1
            else:
                print("\033[31mPrefix cut\033[0m")
                return self.new_produced_tokens[:last_sentence_end]
        return self.new_produced_tokens[:common_length], self.new_produced_tokens[common_length:]

    
    def translate(self, text: Optional[str | TimedText] = None) -> str:
        if type(text) == str:
            self.input_buffer.append(TimedText(text))
        elif type(text) == TimedText:
            self.input_buffer.append(text)
        
        # self._trim()
        buffer_text = ''.join([token.text for token in self.input_buffer])

        early_cut = False
        last_punct_pos = -1

        for i, char in enumerate(buffer_text):
            if char in PUNCTUATION_MARKS:
                last_punct_pos = i

        if last_punct_pos >= 0:
            text_to_process = buffer_text[:last_punct_pos + 1]
            remaining_text = buffer_text[last_punct_pos + 1:]

            char_count = 0
            split_index = len(self.input_buffer)
            for idx, timed_text in enumerate(self.input_buffer):
                char_count += len(timed_text.text)
                if char_count > last_punct_pos:
                    split_index = idx
                    if char_count - len(timed_text.text) <= last_punct_pos:
                        chars_before = last_punct_pos + 1 - (char_count - len(timed_text.text))
                        before_text = timed_text.text[:chars_before]
                        after_text = timed_text.text[chars_before:]
                        if after_text:
                            self.input_buffer = [TimedText(text=after_text, start=timed_text.start, end=timed_text.end)] + self.input_buffer[idx + 1:]
                        else:
                            self.input_buffer = self.input_buffer[idx + 1:]
                    break

            buffer_text = text_to_process
            early_cut = True
            print(f'\033[33mEarly cut. Processing: "{text_to_process}" | Remaining: "{remaining_text}"\033[0m')

        word_count = len(buffer_text.strip().split())
        if word_count < 3:
            return "", ""
        inputs = self.tokenizer(buffer_text, return_tensors="pt").to(self.device)
        self.n_remaining_input_punctuation += (self.has_sentence_end_token(inputs['input_ids'][0]) != -1)

        if self.stable_prefix_segments:
            self.stable_prefix_tokens = torch.cat(self.stable_prefix_segments, dim=0)
        if self.verbose:
            start_time = time.time()
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(**inputs)
        if self.verbose:
            encoding_time = time.time() - start_time
        
        if self.verbose:
            start_time = time.time()
        if len(self.stable_prefix_tokens):
            # from bertviz import model_view
            # html_obj = model_view(output.attentions, source_tokens, html_action='return')
            # with open("attention_viz.html", "w") as f:
            #     f.write(str(html_obj.data))
            translation_tokens = self._continue_generation_with_cache(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )
        else:
            with torch.no_grad():
                translation_tokens = self.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=inputs['attention_mask'],
                )
        if self.verbose:
            decoding_time = time.time() - start_time
            print(f"Encoding time: {encoding_time:.4f}s | Decoding time: {decoding_time:.4f}s")

        self.new_produced_tokens = translation_tokens[0][len(self.stable_prefix_tokens):]
        if self.verbose:
            prefix_used_tokens = translation_tokens[0][:len(self.stable_prefix_tokens)]
            prefix_used = self.tokenizer.decode(
                prefix_used_tokens,
                skip_special_tokens=True
            )
        if early_cut:
            print('\033[33mResetting stable prefix state after early cut\033[0m')
            new_stable_tokens = self.new_produced_tokens
            self.stable_prefix_segments = []
            self.stable_prefix_tokens = torch.tensor([], dtype=torch.int64)
            self.previous_tokens = []
            self.n_remaining_input_punctuation = 0
            buffer = ''
        else:
            new_stable_tokens, new_buffer = self.compute_common_prefix_tokens()
            self.stable_prefix_segments.append(new_stable_tokens)
            self.previous_tokens = new_buffer
            buffer = self.tokenizer.decode(
                    new_buffer,
                    skip_special_tokens=True
            )
        stable_translation = self.tokenizer.decode(
            new_stable_tokens,
            skip_special_tokens=True
        )
        if self.verbose:
            # print(f'{i}/{len(src_texts) + 1}: \033[36m{truncated_text}\033[0m')
            print(f' \033[36m{prefix_used}\033[0m\033[32m{stable_translation}\033[0m \033[35m{buffer}\033[0m')
        return stable_translation, buffer

    def _continue_generation_with_cache(
        self,
        encoder_hidden_states: torch.Tensor,
        max_new_tokens: int = 200
    ) -> torch.Tensor:
        eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            past_key_values = EncoderDecoderCache(
                DynamicCache(),
                DynamicCache()
            )

            decoder_out = self.model.model.decoder(
                input_ids=self.stable_prefix_tokens.unsqueeze(0),
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_out.past_key_values
            prefix_logits = self.model.lm_head(decoder_out.last_hidden_state)
            next_token_id = torch.argmax(prefix_logits[:, -1, :], dim=-1).unsqueeze(-1)

            generated_tokens = self.stable_prefix_tokens.unsqueeze(0).clone()

            for _ in range(max_new_tokens):
                if next_token_id.item() == eos_token_id:
                    break

                generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)

                decoder_out = self.model.model.decoder(
                    input_ids=next_token_id,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                past_key_values = decoder_out.past_key_values
                logits = self.model.lm_head(decoder_out.last_hidden_state)
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)

        return generated_tokens



if __name__ == '__main__':
    from nllw.test_strings import *
    import pandas as pd
    translation_backend = TranslationBackend(source_lang='fra_Latn', target_lang="eng_Latn", verbose=True)
    # input_text = " ".join(src_2_fr)
    src_texts = src_2_fr 

    l_vals_with_cache = []
    for i in range(0, len(src_texts)):
        truncated_text = src_texts[i]
        print(f'\n\n{i}/{len(src_texts) + 1}: {truncated_text}')
        stable_translation, buffer = translation_backend.translate(truncated_text)
        # print(f'\033[32m{stable_translation}\033[0m \033[35m{buffer}\033[0m')
        
        full_output = stable_translation + buffer
        l_vals_with_cache.append({
            "input": truncated_text,
            "stable_translation": stable_translation,
            # "buffer_tokens": translation_backend.buffer_tokens,
            "stable_prefix_tokens": translation_backend.stable_prefix_tokens,
            "input_word_count": len(truncated_text.split()),
            "stable_word_count": len(stable_translation.split()) if stable_translation else 0,
            "total_output_word_count": len(full_output.split()) if full_output else 0
        })
    pd.DataFrame(l_vals_with_cache).to_pickle('export_with_tokens.pkl')