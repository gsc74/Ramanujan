<div align="center">

# MyLLM-1B

**A tiny math and English tutor for students that runs on your laptop.**

[![Transformers](https://img.shields.io/badge/format-Transformers-blue)](#download) [![Params](https://img.shields.io/badge/params-1.05B-green)](#model-summary) [![Runs on](https://img.shields.io/badge/runs%20on-PyTorch-orange)](#usage) [![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](#license)

</div>

> [!NOTE]
> Educational outcome of the tutorial **[How to train your first LLM?](https://gsc74.github.io/blogs/train_index.html)**.
> It's a **basic high-school math model**.

## Model Summary

A **1.05B-parameter** decoder-only model for student tutoring in math and English: pretrained on math/general text, SFT on math and chat instructions, then aligned for clearer and more accurate answers.

| | |
|---|---|
| Parameters | 1.055B |
| Architecture | Llama-style, 28 layers, hidden 1792 |
| Attention | 14 query / 2 KV heads (GQA), head dim 128 |
| FFN / Norm | SwiGLU (4864) / RMSNorm |
| Positional | RoPE (θ = 500,000), 8,192 ctx |
| Vocabulary | 65,536 (byte-level BPE) |

## Download

| Format | Use for | Link |
|---|---|---|
| Transformers (safetensors) | Python / PyTorch | [MyLLM-1B-HF/](https://github.com/gsc74/MyLLM/tree/main/MyLLM-1B-HF) + [weights from Release](https://github.com/gsc74/MyLLM/releases/latest) |

> The model weights (`model-0000*.safetensors`, ~2 GB total) are attached as a **GitHub Release asset**. Clone the repo and drop them into `MyLLM-1B-HF/`.

## Prerequisites

- **Python 3.9+**
- **PyTorch** and **Transformers** (`pip install torch transformers`)
- The model files: clone this repo and place the `model-0000*.safetensors` weights from the [latest Release](https://github.com/gsc74/MyLLM/releases/latest) into `MyLLM-1B-HF/`.

```bash
git clone https://github.com/gsc74/MyLLM.git
cd MyLLM
# download the safetensors from the Release into MyLLM-1B-HF/
pip install torch transformers
```

Runs on **CPU** (default) or **GPU** (`DEVICE=cuda`). About 2 GB of RAM/VRAM is needed for the BF16 weights.

## Usage

### Chat with `infer.sh` (interactive)

The easiest way to chat with the model is the bundled script:

```bash
./infer.sh
```

It loads `MyLLM-1B-HF/` and starts an interactive prompt. Type a question, press Enter, and type `quit` (or Ctrl+D) to exit:

```
You: What is 2+2?
MyLLM: The answer is 4.
```

Optional settings (environment variables):

| Variable | Default | Description |
|---|---|---|
| `DEVICE` | `cpu` | torch device: `cpu` or `cuda` |
| `MAX_NEW` | `256` | max new tokens per reply |
| `THREADS` | `64` | CPU threads |
| `TEMPERATURE` | `0.7` | sampling temperature |
| `REPETITION_PENALTY` | `1.3` | discourages repetition |

```bash
# example: run on GPU with shorter replies
DEVICE=cuda MAX_NEW=128 ./infer.sh
```

### Transformers (PyTorch)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

path = "MyLLM-1B-HF"
tok = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype="auto")

msgs = [{"role": "user", "content": "What is 2+2?"}]
inputs = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt", return_dict=True)
out = model.generate(**inputs, max_new_tokens=128, temperature=0.7, repetition_penalty=1.3)
print(tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

## Example

**Prompt:** `Give me the integration of x^2.`

```
The integral is: ∫ x² dx

So, we have: (1/3) x³ + C₀

Therefore: \boxed{\frac{1}{3}x^{3}}
```

## Training Data & Licenses

| Stage | Dataset | Amount | License |
|---|---|---:|---|
| Pretraining Data 1 | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | 12,540,182,528 input tokens | ODC-By 1.0 |
| Pretraining Data 2 | FineWeb-Edu + Cosmopedia auto_math_text | 15,000,061,824 input tokens | Dataset-specific |
| Pretraining total | OpenWebMath + FineWeb-Edu + Cosmopedia auto_math_text | 27,540,244,352 input tokens | Dataset-specific |
| SFT | [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | 99,178 examples | MIT |
| SFT | [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 260,085 examples | Apache-2.0 |
| SFT | [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | 228,408 examples | CC-BY-4.0 |
| SFT | Orca Math word problems | 109,618 examples | Dataset-specific |
| SFT | UltraChat | 202,618 examples | Dataset-specific |
| SFT total | Mixed math + English chat SFT | 899,907 examples / 403,406,096 tokens | Dataset-specific |
| Math GRPO reward | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 7,473 prompts / 700 steps | MIT |
| DPO | Mixed preference pairs | 73,874 pairs / 1 epoch | Dataset-specific |
| English GRPO | UltraChat prompt bank | 8,000 prompts / 200 steps | Dataset-specific |

## License

Weights and code are released under **Apache-2.0**. Use must also comply with the dataset licenses above (including attribution for OpenWebMath and OpenMathInstruct-2).

## Acknowledgements

Capstone artifact of my tutorial **[How to train your first LLM?](https://gsc74.github.io/blogs/train_index.html)**.
