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
        self.stable_prefix = None
    
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

        last_sentence_end = -1
        for i in range(common_length):
            if new_tokens[0][i].item() in self.sentence_end_token_ids:
                last_sentence_end = i

        if last_sentence_end >= 0:
            return new_tokens[:, :last_sentence_end]

        return new_tokens[:, :common_length]
    
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
        max_new_tokens: int = 200
    ) -> torch.Tensor:
        eos_token_id = self.tokenizer.eos_token_id

        with torch.no_grad():
            past_key_values = EncoderDecoderCache(
                self_attention_cache=DynamicCache(),
                cross_attention_cache=DynamicCache()
            )

            decoder_out = self.model.model.decoder(
                input_ids=self.stable_prefix,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            past_key_values = decoder_out.past_key_values
            prefix_logits = self.model.lm_head(decoder_out.last_hidden_state)
            next_token_id = torch.argmax(prefix_logits[:, -1, :], dim=-1).unsqueeze(-1)

            generated_tokens = self.stable_prefix.clone()

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
    # src_texts = [
    #     "Have you noticed how accurate",
    #     "LLM are now that GPU have became more powerful, ",
    #     "especially when we think at the difficulties we had in the 2010 era",
    #     "where online chatbots were not performant",
    #     "at all, and ofter doing strict rules worked better",
    #     "do you remember that era?"]

#     src_texts = [
#         "As-tu remarqué à quel point",
#         "les LLM sont précis maintenant que les GPU sont devenus plus puissants, ",
#         "surtout quand on pense aux difficultés qu'on avait dans les années 2010",
#         "où les chatbots en ligne n'étaient pas performants",
#         "du tout, et souvent faire des règles strictes fonctionnait mieux.",
#         "Tu te souviens de cette époque, ",
#         "c'était bien plus compliqué de travailler"]

    translation_backend = TranslationBackend(source_lang='fra_Latn', target_lang="eng_Latn")
    
#     # for i in range(len(src_texts)+1):
#     #     translation, buffer = translation_backend.translate(" ".join(src_texts[:i]))
#     #     print(translation, '|', buffer)

#     tokens, text = translation_backend.simple_translation("Ceci est un test de traduction. Nous avons fait tout notre possible. Jusqu'à partir de 0")[1]
    
#     # text = "Ceci est un test de traduction. Nous avons fait tout notre possible. Jusqu'à partir de 0, où nous avions pris la route, tout droit, et loin, loin, loin ! tu penses qu'on peut vraiment faire la longeur qu'on veut ?"
#     print(output_text) 
    

    src_texts = ["Il s’arrêtait par moments devant une villa",
    "coquettement nichée dans la verdure, il regardait",
    "par la grille et voyait au loin des femmes",
    "élégantes sur les balcons et les terrasses, des",
    "enfants couraient dans les jardins. Il s’intéressait",
    "surtout aux fleurs ; c’étaient elles qui attiraient",
    "particulièrement ses regards. De temps en temps,"
    "il voyait passer des cavaliers, des amazones et de"
    "belles voitures ; il les suivait d’un œil curieux et",
    "les oubliait avant qu’ils eussent disparu. "]

    text = " ".join(src_texts)
    

    target = """He would stop occasionally in front of a villa
    nestled charmingly in the greenery, look
    through the gate, and see elegant women
    on balconies and terraces in the distance,
    children running in the gardens. He was particularly interested
    in the flowers; they were what caught his eye. From time to time, 
    he would see horsemen, horsewomen, and beautiful carriages passing by; 
    he would follow them with a curious eye and
    forget them before they had disappeared.
    """
    l_vals = []
    for i in range(1, len(src_texts)+1):
        truncated_text = " ".join(src_texts[:i])
        input_tokens = translation_backend.tokenizer(truncated_text, return_tensors="pt").to(translation_backend.device)
        encoder_outputs = translation_backend.model.get_encoder()(**input_tokens)
        output_tokens = translation_backend.model.generate(
            encoder_outputs=encoder_outputs,
            forced_bos_token_id=translation_backend.bos_token_id
        )
        output_text = translation_backend.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
        l_vals.append(
            {
                "input": truncated_text,
                "output_tokens_shape": output_tokens.shape[1], 
                "output_text": output_text,
            }
        )
    # input_tokens = translation_backend.tokenizer(text, return_tensors="pt").to(translation_backend.device)
    # encoder_outputs = translation_backend.model.get_encoder()(**input_tokens)
    # output_tokens = translation_backend.model.generate(
    #     encoder_outputs=encoder_outputs,
    #     forced_bos_token_id=translation_backend.bos_token_id
    # )
    # output_text = translation_backend.tokenizer.decode(output_tokens[0], skip_special_tokens=True)

    
    print('end')
