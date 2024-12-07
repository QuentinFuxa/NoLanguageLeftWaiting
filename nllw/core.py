import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.cache_utils import EncoderDecoderCache, DynamicCache
from typing import Tuple, Optional


class TranslationBackend:
    def __init__(self, source_lang, target_lang, model_name: str = "facebook/nllb-200-distilled-600M", model=None, tokenizer=None):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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
                if cleaned in ['.', '!', '?']:
                    self.sentence_end_token_ids.add(token_id)
            except:
                pass

        self.previous_tokens = None
        self.buffer_tokens = None
        self.stable_prefix_tokens = None
        self.n_remaining_input_punctuation = 0
    
    def simple_translation(self, text):
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        encoder_outputs = self.model.get_encoder()(**inputs)
        output = self.generate(
                    encoder_outputs=encoder_outputs,
        )
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
        for i in range(len(tokens[0])):
            if tokens[0][i].item() in self.sentence_end_token_ids:
                last_sentence_end = i
        return last_sentence_end
        
        

    def compute_common_prefix_tokens(
        self, new_tokens
    ):
        common_length = 0
        for i in range(min(len(self.previous_tokens[0]), len(new_tokens[0]))):
            if self.previous_tokens[0][i] != new_tokens[0][i]:
                common_length = i
                break
        else:
            common_length = min(len(self.previous_tokens[0]), len(new_tokens[0]))


        last_sentence_end = self.has_sentence_end_token(new_tokens[:, :common_length])

        if last_sentence_end >= 0:
            if self.n_remaining_input_punctuation > 0:
                print(f"\033[33mEOS detected. Remaining punctuation: {self.n_remaining_input_punctuation}\033[0m")
                self.n_remaining_input_punctuation -= 1
            else:
                print("\033[31mPrefix cut\033[0m")
                return new_tokens[:, :last_sentence_end]
        return new_tokens[:, :common_length]
    
    def translate(self, text: str) -> str:
        word_count = len(text.strip().split())
        
        if word_count < 3:
            return "", ""
    
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        self.n_remaining_input_punctuation += (self.has_sentence_end_token(inputs['input_ids']) != -1)
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(**inputs)
        
        if (self.previous_tokens is not None and self.stable_prefix_tokens is not None):
            translation_tokens = self._continue_generation_with_cache(
                encoder_hidden_states=encoder_outputs.last_hidden_state,
            )
        else:
            with torch.no_grad():
                translation_tokens = self.generate(
                    encoder_outputs=encoder_outputs,
                    attention_mask=inputs['attention_mask'],
                )
        if self.previous_tokens is not None:
            self.stable_prefix_tokens = self.compute_common_prefix_tokens(new_tokens=translation_tokens)
        self.previous_tokens = translation_tokens
        self.buffer_tokens = translation_tokens[0][len(self.stable_prefix_tokens[0]) if self.stable_prefix_tokens is not None else 0:]
        
        
        buffer = self.tokenizer.decode(
                self.buffer_tokens, 
                skip_special_tokens=True
        )        
        if self.stable_prefix_tokens is not None:
            stable_translation = self.tokenizer.decode(
                self.stable_prefix_tokens[0], 
                skip_special_tokens=True
            )
            return stable_translation, buffer
        else:
            return "", buffer

    def _continue_generation_with_cache(
        self,
        encoder_hidden_states: torch.Tensor,
        max_new_tokens: int = 200
    ) -> torch.Tensor:
        eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            past_key_values = EncoderDecoderCache(
                self_attention_cache=DynamicCache(),
                cross_attention_cache=DynamicCache()
            )

            decoder_out = self.model.model.decoder(
                input_ids=self.stable_prefix_tokens,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_out.past_key_values
            prefix_logits = self.model.lm_head(decoder_out.last_hidden_state)
            next_token_id = torch.argmax(prefix_logits[:, -1, :], dim=-1).unsqueeze(-1)

            generated_tokens = self.stable_prefix_tokens.clone()

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
    translation_backend = TranslationBackend(source_lang='fra_Latn', target_lang="eng_Latn")
    # input_text = " ".join(src_2_fr)
    src_texts = src_2_fr 

    l_vals_with_cache = []
    for i in range(1, len(src_texts) + 1):
        truncated_text = " ".join(src_texts[:i])
        stable_translation, buffer = translation_backend.translate(truncated_text)
        print(f'{i}/{len(src_texts) + 1}: \033[36m{truncated_text}\033[0m')
        print(f'\033[32m{stable_translation}\033[0m \033[35m{buffer}\033[0m')
        
        full_output = stable_translation + buffer
        l_vals_with_cache.append({
            "input": truncated_text,
            "stable_translation": stable_translation,
            "buffer_tokens": translation_backend.buffer_tokens,
            "stable_prefix_tokens": translation_backend.stable_prefix_tokens,
            "input_word_count": len(truncated_text.split()),
            "stable_word_count": len(stable_translation.split()) if stable_translation else 0,
            "total_output_word_count": len(full_output.split()) if full_output else 0
        })
    pd.DataFrame(l_vals_with_cache).to_pickle('export_with_tokens.pkl')

    # l_vals = []
    # for i in range(1, len(src_texts)+1):
    #     truncated_text = " ".join(src_texts[:i])
    #     input_tokens = translation_backend.tokenizer(truncated_text, return_tensors="pt").to(translation_backend.device)
    #     encoder_outputs = translation_backend.model.get_encoder()(**input_tokens)
    #     output_tokens = translation_backend.model.generate(
    #         encoder_outputs=encoder_outputs,
    #         forced_bos_token_id=translation_backend.bos_token_id
    #     )
    #     output_text = translation_backend.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    #     l_vals.append(
    #         {
    #             "input": truncated_text,
    #             "output_tokens_shape": output_tokens.shape[1], 
    #             "output_text": output_text,
    #         }
    #     )
    # # input_tokens = translation_backend.tokenizer(text, return_tensors="pt").to(translation_backend.device)
    # # encoder_outputs = translation_backend.model.get_encoder()(**input_tokens)
    # # output_tokens = translation_backend.model.generate(
    # #     encoder_outputs=encoder_outputs,
    # #     forced_bos_token_id=translation_backend.bos_token_id
    # # )
    # # output_text = translation_backend.tokenizer.decode(output_tokens[0], skip_special_tokens=True)