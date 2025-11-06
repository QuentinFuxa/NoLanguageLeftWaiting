import time
from typing import Optional

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.cache_utils import DynamicCache, EncoderDecoderCache

from nllw.test_strings import src_2_fr



source_lang = "fra_Latn"
target_lang = "eng_Latn"
model_name: str = "facebook/nllb-200-distilled-600M"
max_new_tokens = 200

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

draft_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
main_model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-distilled-1.3B").to(device)
draft_model.eval()
tokenizer = AutoTokenizer.from_pretrained(model_name, src_lang=source_lang)

eos_token_id: Optional[int] = tokenizer.eos_token_id
language_token_id = tokenizer.convert_tokens_to_ids(target_lang)

decoder_start_token_id: Optional[int] = draft_model.config.decoder_start_token_id
if decoder_start_token_id is None:
    decoder_start_token_id = tokenizer.bos_token_id
if decoder_start_token_id is None:
    decoder_start_token_id = tokenizer.cls_token_id
if decoder_start_token_id is None:
    decoder_start_token_id = language_token_id

def sequential_decoding(model, encoder_last_hidden_state):
        start_tokens = torch.tensor([[decoder_start_token_id, language_token_id]], device=device)
        generated_tokens = start_tokens.clone()
        decoder_input = generated_tokens
        token_cache = EncoderDecoderCache(DynamicCache(), DynamicCache())

        if device.type == "cuda":
            torch.cuda.synchronize()
        sequential_start = time.time()

        produced_tokens = 0
        while produced_tokens < max_new_tokens:
            decoder_out = model.model.decoder(
                input_ids=decoder_input,
                encoder_hidden_states=encoder_last_hidden_state,
                past_key_values=token_cache,
                use_cache=True,
                return_dict=True,
            )

            token_cache = decoder_out.past_key_values
            logits = model.lm_head(decoder_out.last_hidden_state)
            next_token = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)

            generated_tokens = torch.cat([generated_tokens, next_token], dim=-1)
            produced_tokens += 1

            if eos_token_id is not None and next_token.item() == eos_token_id:
                break

            decoder_input = next_token

        if device.type == "cuda":
            torch.cuda.synchronize()
        sequential_time = time.time() - sequential_start
        total_sequence_length = generated_tokens.shape[1]
        new_token_count = total_sequence_length - start_tokens.shape[1]
        print(f"Sequential decoding: {sequential_time:.4f}s for {new_token_count} new tokens")
        return generated_tokens, sequential_time

l_results = []

for i in range(3, len(src_2_fr)):
    text = ' '.join(src_2_fr[:i])

    inputs = tokenizer(text, return_tensors="pt").to(device)

    with torch.no_grad():

        encode_draft_start = time.time()
        encoder_outputs = draft_model.get_encoder()(**inputs)
        encode_draft_end = time.time()
        encode_time_draft = encode_draft_end - encode_draft_start
        encoder_outputs_main = main_model.get_encoder()(**inputs)
        encode_time_main = time.time() - encode_draft_end

    generated_tokens_draft, sequential_time_draft = sequential_decoding(draft_model, encoder_last_hidden_state=encoder_outputs.last_hidden_state)

    generated_tokens_main, sequential_time_main = sequential_decoding(main_model, encoder_last_hidden_state=encoder_outputs_main.last_hidden_state)

    decoder_inputs_for_verification = generated_tokens_draft[:, :-1]
    expected_tokens = generated_tokens_draft[:, 1:]

    if device.type == "cuda":
        torch.cuda.synchronize()
    verify_start = time.time()
    verification_out = main_model.model.decoder(
        input_ids=decoder_inputs_for_verification,
        encoder_hidden_states=encoder_outputs_main.last_hidden_state,
        use_cache=False,
        return_dict=True,
    )
    verification_logits = main_model.lm_head(verification_out.last_hidden_state)
    if device.type == "cuda":
        torch.cuda.synchronize()
    verification_time = time.time() - verify_start

    predicted_tokens = torch.argmax(verification_logits, dim=-1)

    generated_tokens_main[0, 2:]
    predicted_tokens[0, 1:]
    generated_tokens_draft[0, 2:]

    # for i in range(len(generated_tokens_main[0, 2:])):
    #     if predicted_tokens[0, 1:][i] != generated_tokens_main[0, 2:][i]:
    #         print('A', i)
    #     if generated_tokens_draft[0, 2:][i] != generated_tokens_main[0, 2:][i]:
    #         print('B', i)
    #     if predicted_tokens[0, 1:][i] != generated_tokens_draft[0, 2:][i]:
    #         print('C', i)


    matches = (predicted_tokens == expected_tokens)
    verified_tokens = matches.sum().item()
    total_verified = expected_tokens.numel()

    print(f"Verification pass: {verification_time:.4f}s for {total_verified} positions")
    print(f"Tokens matching main model: {verified_tokens}/{total_verified}")

    result = tokenizer.decode(generated_tokens_draft[0], skip_special_tokens=True)
    print("\n=== Translation Result ===")
    print(result)

    l_results.append({
        'text': text,
        'sequential_time_draft': sequential_time_draft,
        'sequential_time_main': sequential_time_main,
        'matches': matches,
        'total_verified': total_verified,
        'encode_time_draft': encode_time_draft,
        'encode_time_main': encode_time_main,
    })

print(l_results)

