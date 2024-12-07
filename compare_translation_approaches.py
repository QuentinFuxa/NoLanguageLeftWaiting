from nllw.core import TranslationBackend
from matplotlib.ticker import MaxNLocator
from nllw.test_strings import src_2_fr
import matplotlib.pyplot as plt
import pandas as pd

src_texts = src_2_fr

colors = {
    "dark_gray_1": "#2E2E2E",
    "dark_gray_2": "#4B4B4B",
    "orange": "#FF7F0E",
    "red": "#D62728",
}


# print("1: direct generation)")
# translation_backend_1 = TranslationBackend(source_lang='fra_Latn', target_lang="eng_Latn")
# l_vals_no_cache = []

# for i in range(1, len(src_texts) + 1):
#     print(f'{i}/{len(src_texts) + 1}')
#     truncated_text = " ".join(src_texts[:i])
#     input_tokens = translation_backend_1.tokenizer(truncated_text, return_tensors="pt").to(translation_backend_1.device)
#     encoder_outputs = translation_backend_1.model.get_encoder()(**input_tokens)
#     output_tokens = translation_backend_1.model.generate(
#         encoder_outputs=encoder_outputs,
#         forced_bos_token_id=translation_backend_1.bos_token_id
#     )
#     output_text = translation_backend_1.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
#     l_vals_no_cache.append({
#         "input": truncated_text,
#         "output_text": output_text,
#         "input_word_count": len(truncated_text.split()),
#         "output_word_count": len(output_text.split())
#     })

print("2: prefix reuse")
translation_backend_2 = TranslationBackend(source_lang='fra_Latn', target_lang="eng_Latn")
l_vals_with_cache = []

for i in range(1, len(src_texts) + 1):
    print(f'{i}/{len(src_texts) + 1}')
    truncated_text = " ".join(src_texts[:i])
    stable_translation, buffer = translation_backend_2.translate(truncated_text)
    
    full_output = stable_translation + buffer
    l_vals_with_cache.append({
        "input": truncated_text,
        "stable_translation": stable_translation,
        "buffer": buffer,
        "full_output": full_output,
        "input_word_count": len(truncated_text.split()),
        "stable_word_count": len(stable_translation.split()) if stable_translation else 0,
        "buffer_word_count": len(buffer.split()) if buffer else 0,
        "total_output_word_count": len(full_output.split()) if full_output else 0
    })

# df_base = pd.DataFrame(l_vals_no_cache)
df_prefix = pd.DataFrame(l_vals_with_cache)
# df_base.to_csv('output_base_analysis.csv')
df_prefix.to_csv('output_prefix_analysis.csv')

df_base = pd.read_csv('output_base_analysis.csv')
# df_prefix = pd.read_csv('output_prefix_analysis.csv')


iterations = list(range(1, len(df_prefix) + 1))
input_counts = df_prefix['input_word_count'].tolist()
output_with_prefix_and_buffer = df_prefix['total_output_word_count'].tolist()
stable_counts = df_prefix['stable_word_count'].tolist()
buffer_counts = df_prefix['buffer_word_count'].tolist()
base_output_wc = df_base['output_word_count'].tolist()

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
fig, (ax1) = plt.subplots(1, 1, figsize=(5, 5))


ax1.plot(iterations, input_counts, label='Input', marker='o', color=colors['dark_gray_1'])
ax1.plot(iterations, base_output_wc, label='Base output (no prefix, no stability)', marker='^', color=colors['dark_gray_2'])
ax1.plot(iterations, stable_counts, label='With prefix (Stable part)', marker='^', linewidth=2, color=colors['red'])
ax1.plot(iterations, output_with_prefix_and_buffer, label='With prefix (Stable part + Buffer)', marker='^', color=colors['orange'])
ax1.set_xlabel('Iteration')
ax1.set_ylabel('Word count')
ax1.set_title('SimulMT french to english Word Count')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.xaxis.set_major_locator(MaxNLocator(integer=True))

# # Second plot: Detailed view of cached approach
# ax2.plot(iterations, input_counts, label='Input', marker='o', linewidth=2)
# ax2.plot(iterations, stable_counts, label='Stable prefix (cached)', marker='D', linewidth=2)
# ax2.plot(iterations, buffer_counts, label='Buffer (new)', marker='x', linewidth=2)
# ax2.plot(iterations, output_with_cache, label='Total output', marker='^', linewidth=2, linestyle='--')
# ax2.set_xlabel('Iteration', fontsize=12)
# ax2.set_ylabel('Word count', fontsize=12)
# ax2.set_title('Detailed view: Stable prefix vs Buffer (with cache)', fontsize=14, fontweight='bold')
# ax2.legend(fontsize=10)
# ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('translation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()