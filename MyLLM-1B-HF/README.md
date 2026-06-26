---
library_name: transformers
pipeline_tag: text-generation
tags:
  - math
  - custom_code
  - dpo
---

# MyLLM-1B

MyLLM-1B is a 1.055B-parameter decoder-only math language model trained on
OpenWebMath plus general educational English, supervised fine-tuned on mixed math/instruction data, and aligned
with GRPO, DPO, and English GRPO.

The model uses 28 layers, hidden size 1792, 14 attention heads, 2 KV heads,
SwiGLU, RMSNorm, interleaved RoPE (`theta=500000`), and an 8192-token context.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "MyLLM-1B-HF"
tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    path, trust_remote_code=True, torch_dtype="auto"
)
messages = [{"role": "user", "content": "Solve: 17 * 23"}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt", return_dict=True
)
output = model.generate(**inputs, max_new_tokens=128)
print(tokenizer.decode(output[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

`trust_remote_code=True` is required because the training implementation uses
interleaved rotary embeddings. Generation currently recomputes the prefix (no
KV cache), prioritizing exact checkpoint compatibility over decoding speed.

