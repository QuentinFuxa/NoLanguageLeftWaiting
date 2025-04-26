from translation_backend import TranslationBackend
import matplotlib.pyplot as plt

src_texts = [
    "Il s'arrêtait par moments devant une villa",
    "coquettement nichée dans la verdure, il regardait",
    "par la grille et voyait au loin des femmes",
    "élégantes sur les balcons et les terrasses, des",
    "enfants couraient dans les jardins. Il s'intéressait",
    "surtout aux fleurs ; c'étaient elles qui attiraient",
    "particulièrement ses regards. De temps en temps,",
    "il voyait passer des cavaliers, des amazones et de",
    "belles voitures ; il les suivait d'un œil curieux et",
    "les oubliait avant qu'ils eussent disparu. ",
    "Une fois, il s'arrêta et compta son argent ; il",
    "lui restait trente kopecks : « vingt au sergent de",
    "ville, trois à Nastassia pour la lettre, j'en ai donc",
    "donné hier à Marmeladov quarante-sept ou",
    "cinquante », se dit-il. Il devait avoir une raison de",
    "calculer ainsi, mais il l'oublia en tirant l'argent de",
    "sa poche et ne s'en souvint qu'un peu plus tard en",
    "passant devant un marchand de comestibles, une",
    "sorte de gargote plutôt ; il sentit alors qu'il avait",
    "faim. "
]

print("1: direct generation)")
translation_backend_1 = TranslationBackend(source_lang='fra_Latn', target_lang="eng_Latn")
l_vals_no_cache = []

for i in range(1, len(src_texts) + 1):
    truncated_text = " ".join(src_texts[:i])
    input_tokens = translation_backend_1.tokenizer(truncated_text, return_tensors="pt").to(translation_backend_1.device)
    encoder_outputs = translation_backend_1.model.get_encoder()(**input_tokens)
    output_tokens = translation_backend_1.model.generate(
        encoder_outputs=encoder_outputs,
        forced_bos_token_id=translation_backend_1.bos_token_id
    )
    output_text = translation_backend_1.tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    l_vals_no_cache.append({
        "input": truncated_text,
        "output_text": output_text,
        "input_word_count": len(truncated_text.split()),
        "output_word_count": len(output_text.split())
    })

print("2: prefix reuse")
translation_backend_2 = TranslationBackend(source_lang='fra_Latn', target_lang="eng_Latn")
l_vals_with_cache = []

for i in range(1, len(src_texts) + 1):
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

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
iterations = list(range(1, len(src_texts) + 1))
input_counts = [item["input_word_count"] for item in l_vals_no_cache]
output_no_cache = [item["output_word_count"] for item in l_vals_no_cache]
output_with_cache = [item["total_output_word_count"] for item in l_vals_with_cache]

ax1.plot(iterations, input_counts, label='Input', marker='o', linewidth=2)
ax1.plot(iterations, output_no_cache, label='Output (no cache)', marker='s', linewidth=2)
ax1.plot(iterations, output_with_cache, label='Output (with cache)', marker='^', linewidth=2)
ax1.set_xlabel('Iteration', fontsize=12)
ax1.set_ylabel('Word count', fontsize=12)
ax1.set_title('Comparison: Input vs Output (with and without prefix reuse)', fontsize=14, fontweight='bold')
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)

stable_counts = [item["stable_word_count"] for item in l_vals_with_cache]
buffer_counts = [item["buffer_word_count"] for item in l_vals_with_cache]

ax2.plot(iterations, input_counts, label='Input', marker='o', linewidth=2)
ax2.plot(iterations, stable_counts, label='Stable prefix (cached)', marker='D', linewidth=2)
ax2.plot(iterations, buffer_counts, label='Buffer (new)', marker='x', linewidth=2)
ax2.plot(iterations, output_with_cache, label='Total output', marker='^', linewidth=2, linestyle='--')
ax2.set_xlabel('Iteration', fontsize=12)
ax2.set_ylabel('Word count', fontsize=12)
ax2.set_title('Detailed view: Stable prefix vs Buffer (with cache)', fontsize=14, fontweight='bold')
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('translation_comparison.png', dpi=300, bbox_inches='tight')
plt.show()
