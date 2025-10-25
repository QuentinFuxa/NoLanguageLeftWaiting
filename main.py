import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from transformers.cache_utils import EncoderDecoderCache
from typing import Tuple, Optional

model_name = "facebook/nllb-200-distilled-600M"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to("cuda" if torch.cuda.is_available() else "cpu")

device = model.device
bos_token_id = tokenizer.convert_tokens_to_ids("fra_Latn")


def compute_common_prefix_tokens(
    prev_tokens: torch.Tensor,
    new_tokens: torch.Tensor,
    tokenizer: AutoTokenizer,
    sep: str = " "
) -> torch.Tensor:
    if prev_tokens is None or len(prev_tokens) == 0:
        return new_tokens
    
    prev_text = tokenizer.decode(prev_tokens, skip_special_tokens=True)
    new_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    
    if not prev_text or not new_text:
        return new_tokens
    
    prev_words = prev_text.split(sep)
    new_words = new_text.split(sep)
    
    common_word_count = 0
    for i in range(min(len(prev_words), len(new_words))):
        if prev_words[i] == new_words[i]:
            common_word_count += 1
        else:
            break
    
    if common_word_count == 0:
        if len(new_tokens) > 0:
            return new_tokens[:1]
        return new_tokens
    
    if common_word_count == len(prev_words) and len(prev_words) == len(new_words):
        return new_tokens
    
    common_prefix_text = sep.join(new_words[:common_word_count])
    
    for token_idx in range(1, len(new_tokens) + 1):
        decoded = tokenizer.decode(new_tokens[:token_idx], skip_special_tokens=True)
        if decoded == common_prefix_text:
            return new_tokens[:token_idx]
        if len(decoded) > len(common_prefix_text):
            return new_tokens[:max(1, token_idx - 1)]
    
    return new_tokens


def manual_generate(
    encoder_outputs: torch.Tensor,
    attention_mask: torch.Tensor,
    forced_bos_token_id: int,
    eos_token_id: Optional[int] = None
) -> Tuple[torch.Tensor, Optional[Tuple]]:
    
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    with torch.no_grad():
        generated_tokens = model.generate(
            encoder_outputs=encoder_outputs,
            attention_mask=attention_mask,
            forced_bos_token_id=forced_bos_token_id,
        )
    
    return generated_tokens


def continue_generation_with_cache(
    encoder_hidden_states: torch.Tensor,
    prefix_tokens: torch.Tensor,
    max_new_tokens: int = 50,
    eos_token_id: Optional[int] = None
) -> torch.Tensor:
    if eos_token_id is None:
        eos_token_id = tokenizer.eos_token_id
    
    prefix_tokens = prefix_tokens.to(device)
    
    if prefix_tokens.dim() == 1:
        prefix_tokens = prefix_tokens.unsqueeze(0)
    
    with torch.no_grad():
        decoder_out = model.model.decoder(
            input_ids=prefix_tokens,
            encoder_hidden_states=encoder_hidden_states,
            use_cache=True,
            return_dict=True,
        )
        prefix_logits = model.lm_head(decoder_out.last_hidden_state)
        past_key_values = EncoderDecoderCache.from_legacy_cache(decoder_out.past_key_values)
        
        next_token_id = torch.argmax(prefix_logits[:, -1, :], dim=-1).unsqueeze(-1)
        
        if next_token_id.item() == eos_token_id:
            return prefix_tokens
        
        generated_tokens = torch.cat([prefix_tokens, next_token_id], dim=-1)
        
        tokens_to_generate = max_new_tokens - generated_tokens.shape[1]
        
        for _ in range(tokens_to_generate):
            decoder_out = model.model.decoder(
                input_ids=next_token_id,
                encoder_hidden_states=encoder_hidden_states,
                past_key_values=past_key_values,
                use_cache=True,
                return_dict=True,
            )
            logits = model.lm_head(decoder_out.last_hidden_state)
            
            next_token_id = torch.argmax(logits[:, -1, :], dim=-1).unsqueeze(-1)
            past_key_values = EncoderDecoderCache.from_legacy_cache(decoder_out.past_key_values)
            
            if next_token_id.item() == eos_token_id:
                break
            
            generated_tokens = torch.cat([generated_tokens, next_token_id], dim=-1)
    
    return generated_tokens


src_texts = [
    "Have you noticed how accurate",
    "LLM are now that GPU have became more powerful, ",
    "especially when we think at the difficulties we had in the 2010 era",
    "where online chatbots were not performant",
    "at all, and ofter doing strict rules worked better",
    "do you remember that era?"]

inputs_1 = tokenizer(" ".join(src_texts[:1]), return_tensors="pt").to(device)
encoder_outputs_1 = model.get_encoder()(**inputs_1)

translation_tokens_1 = manual_generate(
    encoder_outputs=encoder_outputs_1,
    attention_mask=inputs_1['attention_mask'],
    forced_bos_token_id=bos_token_id,
)

print(f"Translation 1: {tokenizer.decode(translation_tokens_1[0], skip_special_tokens=True)}")

inputs_2 = tokenizer(" ".join(src_texts[:2]), return_tensors="pt").to(device)
encoder_outputs_2 = model.get_encoder()(**inputs_2)

translation_tokens_2 = manual_generate(
    encoder_outputs=encoder_outputs_2,
    attention_mask=inputs_2['attention_mask'],
    forced_bos_token_id=bos_token_id,
)

print(f"Translation 2: {tokenizer.decode(translation_tokens_2[0], skip_special_tokens=True)}")


stable_prefix = compute_common_prefix_tokens(
    translation_tokens_1[0],
    translation_tokens_2[0],
    tokenizer
)

last_translation_tokens = translation_tokens_2
for i in range(3, len(src_texts)+1):
    inputs = tokenizer(" ".join(src_texts[:i]), return_tensors="pt").to(device)
    encoder_outputs = model.get_encoder()(**inputs)
    translation_tokens = continue_generation_with_cache(
        encoder_hidden_states=encoder_outputs.last_hidden_state,
        prefix_tokens=stable_prefix.clone(),
    )
    print(f"Translation {i}: {tokenizer.decode(translation_tokens[0], skip_special_tokens=True)}")

    stable_prefix = compute_common_prefix_tokens(
        last_translation_tokens[0],
        translation_tokens[0],
        tokenizer
    )
    last_translation_tokens = translation_tokens

    print(f"Common prefix: {tokenizer.decode(stable_prefix, skip_special_tokens=True)}")
