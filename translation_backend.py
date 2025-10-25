import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.cache_utils import EncoderDecoderCache
from typing import Tuple, Optional



class TranslationBackend:
    def __init__(self, source_lang, target_lang, model_name: str = "facebook/nllb-200-distilled-1.3B"):

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_lang)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(self.device)
        
        self.source_lang = source_lang
        self.target_lang = target_lang
        self.bos_token_id = self.tokenizer.convert_tokens_to_ids(self.target_lang)
        
        self.previous_tokens = None
        self.stable_prefix = None
    
    def generate(
        self,
        encoder_outputs: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, Optional[Tuple]]:    
        with torch.no_grad():
            generated_tokens = self.model.generate(
                encoder_outputs=encoder_outputs,
                attention_mask=attention_mask,
                forced_bos_token_id=self.bos_token_id,
            )
        return generated_tokens

    def compute_common_prefix_tokens(
        self, new_tokens
    ):
        for i in range(min(len(self.previous_tokens[0]), len(new_tokens[0]))):
            if self.previous_tokens[0][i] != new_tokens[0][i]:
                return new_tokens[:i]
        return self.previous_tokens
    
    def translate(self, text: str) -> str:
        word_count = len(text.strip().split())
        
        if word_count < 3:
            return "", ""
    
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)            
        with torch.no_grad():
            encoder_outputs = self.model.get_encoder()(**inputs)
        
        if (self.previous_tokens is not None and self.stable_prefix is not None):
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
            self.stable_prefix = self.compute_common_prefix_tokens(new_tokens=translation_tokens)
        self.previous_tokens = translation_tokens
        
        buffer = self.tokenizer.decode(
                translation_tokens[0], 
                skip_special_tokens=True
        )        
        if self.stable_prefix is not None:
            stable_translation = self.tokenizer.decode(
                self.stable_prefix[0], 
                skip_special_tokens=True
            )
            return stable_translation, buffer[len(stable_translation):]            
        else:
            return "", buffer

    def _continue_generation_with_cache(
        self,
        encoder_hidden_states: torch.Tensor,
        max_new_tokens: int = 50
    ) -> torch.Tensor:
        eos_token_id = self.tokenizer.eos_token_id
                
        with torch.no_grad():
            decoder_out = self.model.model.decoder(
                input_ids=self.stable_prefix,
                encoder_hidden_states=encoder_hidden_states,
                use_cache=True,
                return_dict=True,
            )
            prefix_logits = self.model.lm_head(decoder_out.last_hidden_state)
            past_key_values = EncoderDecoderCache.from_legacy_cache(decoder_out.past_key_values)
            
            next_token_id = torch.argmax(prefix_logits[:, -1, :], dim=-1).unsqueeze(-1)
            
            if next_token_id.item() == eos_token_id:
                return self.stable_prefix
            
            generated_tokens = torch.cat([self.stable_prefix, next_token_id], dim=-1)
            
            tokens_to_generate = max_new_tokens - generated_tokens.shape[1]
            
            for _ in range(max(0, tokens_to_generate)):
                decoder_out = self.model.model.decoder(
                    input_ids=next_token_id,
                    encoder_hidden_states=encoder_hidden_states,
                    past_key_values=past_key_values,
                    use_cache=True,
                    return_dict=True,
                )
                logits = self.model.lm_head(decoder_out.last_hidden_state)
                
                next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
                past_key_values = EncoderDecoderCache.from_legacy_cache(decoder_out.past_key_values)
                
                if next_token_id.item() == eos_token_id:
                    break
                
                generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
        
        return generated_tokens



if __name__ == '__main__':
    # src_texts = [
    #     "Have you noticed how accurate",
    #     "LLM are now that GPU have became more powerful, ",
    #     "especially when we think at the difficulties we had in the 2010 era",
    #     "where online chatbots were not performant",
    #     "at all, and ofter doing strict rules worked better",
    #     "do you remember that era?"]

    src_texts = [
        "As-tu remarqué à quel point",
        "les LLM sont précis maintenant que les GPU sont devenus plus puissants, ",
        "surtout quand on pense aux difficultés qu'on avait dans les années 2010",
        "où les chatbots en ligne n'étaient pas performants",
        "du tout, et souvent faire des règles strictes fonctionnait mieux,",
        "tu te souviens de cette époque ?"]

    translation_backend = TranslationBackend(source_lang='fra_Latn', target_lang="eng_Latn")
    
    for i in range(len(src_texts)+1):
        translation, buffer = translation_backend.translate(" ".join(src_texts[:i]))
        print(translation, '|', buffer)
