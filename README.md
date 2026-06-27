<div align="center">

# MyLLM-1B

**A tiny math and English tutor for students that runs on your laptop.**

[![Transformers](https://img.shields.io/badge/format-Transformers-blue)](#download) [![GGUF](https://img.shields.io/badge/format-GGUF-purple)](#download) [![Params](https://img.shields.io/badge/params-1.05B-green)](#model-summary) [![Runs on](https://img.shields.io/badge/runs%20on-PyTorch%20%7C%20llama.cpp-orange)](#usage) [![License](https://img.shields.io/badge/license-Apache--2.0-lightgrey)](#license)

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
| GGUF (BF16) | llama.cpp / LM Studio / Ollama | [MyLLM-1B-BF16.gguf](https://github.com/gsc74/MyLLM/releases/latest/download/MyLLM-1B-BF16.gguf) |

> The model weights (`model-0000*.safetensors`, ~2 GB total) and the single-file **`MyLLM-1B-BF16.gguf`** (~2 GB) are attached as **GitHub Release assets**. For PyTorch, clone the repo and drop the safetensors into `MyLLM-1B-HF/`. For llama.cpp / LM Studio, just download the `.gguf`.

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

Runs on **CPU** (default) or **Intel GPU** (`DEVICE=xpu`). About 2 GB of RAM/VRAM is needed for the BF16 weights.

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
| `DEVICE` | `cpu` | torch device: `cpu` or `xpu` (Intel GPU) |
| `MAX_NEW` | `256` | max new tokens per reply |
| `THREADS` | `64` | CPU threads |
| `TEMPERATURE` | `0` | sampling temperature (`0` = greedy, best for math; raise for more variety) |
| `REPETITION_PENALTY` | `1.3` | discourages repetition |

```bash
# example: run on an Intel GPU with shorter replies
DEVICE=xpu MAX_NEW=128 ./infer.sh
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

### Run with llama.cpp / LM Studio (GGUF)

Download **[MyLLM-1B-BF16.gguf](https://github.com/gsc74/MyLLM/releases/latest/download/MyLLM-1B-BF16.gguf)** - a single BF16 file (~2 GB) with the chat template and a default math system prompt already embedded, so it works out of the box.

**LM Studio** (cross-platform): drop the `.gguf` into your models folder (or use *My Models > Import*), select it, and chat. On a supported GPU it is far faster than CPU PyTorch. Recommended generation parameters (match the PyTorch defaults):

| LM Studio setting | Value | Notes |
|---|---|---|
| Temperature | `0` | greedy - best for math; raise to `0.7` for variety |
| Top K | `40` | |
| Top P | `0.9` | |
| Repeat Penalty | `1.3` | stops a small model from looping |
| Repeat Last N | `64` | window the repeat penalty looks back over |
| Max Tokens | `256` | length cap per reply |

> The system prompt and chat template are baked into the GGUF; you don't need to set them in LM Studio. Providing your own system message overrides the embedded one.

**Quick start with the bundled preset.** Instead of setting the values above by hand, import [`MyLLM.preset.json`](MyLLM.preset.json) (it sets temperature, top-k/top-p, repeat penalty, max tokens, and stop strings that trim the model's trailing boilerplate). The order matters:

1. Download `MyLLM-1B-BF16.gguf` from the [latest Release](https://github.com/gsc74/MyLLM/releases/latest) and import/load it in LM Studio first.
2. Then import `MyLLM.preset.json` (chat sidebar → preset selector → **Import preset**).
3. Select the **MyLLM** preset for the loaded model and start chatting.

**llama.cpp** command line (single-shot):

```bash
llama-cli -m MyLLM-1B-BF16.gguf --jinja -st --temp 0 -n 256 \
  --repeat-penalty 1.3 --repeat-last-n 64 \
  -p "What is the derivative of x^2?"
```

## Example

**Prompt:** `Give me the integration of x^2.`

```
The integral is: ∫ x² dx

So, we have: (1/3) x³ + C₀

Therefore: \boxed{\frac{1}{3}x^{3}}
```

**Prompt:** `What is the derivative of x^2?`

```
Using the power rule, d/dx(xⁿ) = n·x^(n-1).

Therefore, the derivative of x² is 2x^(2-1) = 2x.
```

**Prompt:** `What is a black hole?`

```
A black hole is an object in space that has not been fully formed, but has been heated to millions of degrees by its own gravity. The gravitational force between two objects with masses greater than or equal to that of the sun can cause them to collapse into each other, forming a singularity, a point where they become so dense that nothing, including light, can escape from it.
```

> MyLLM is a tiny model focused on high-school math. It can chat in English, but answers outside math may be incomplete or inaccurate.

## Training Data & Licenses

| Stage | Dataset | Amount | License |
|---|---|---:|---|
| Pretraining Data 1 | [OpenWebMath](https://huggingface.co/datasets/open-web-math/open-web-math) | 12,540,182,528 input tokens | ODC-By 1.0 |
| Pretraining Data 2 | [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) + [Cosmopedia](https://huggingface.co/datasets/HuggingFaceTB/cosmopedia) auto_math_text | 15,000,061,824 input tokens | ODC-By 1.0 / Apache-2.0 |
| Pretraining total | OpenWebMath + FineWeb-Edu + Cosmopedia auto_math_text | 27,540,244,352 input tokens | ODC-By 1.0 / Apache-2.0 |
| SFT | [MetaMathQA](https://huggingface.co/datasets/meta-math/MetaMathQA) | 99,178 examples | MIT |
| SFT | [NuminaMath-CoT](https://huggingface.co/datasets/AI-MO/NuminaMath-CoT) | 260,085 examples | Apache-2.0 |
| SFT | [OpenMathInstruct-2](https://huggingface.co/datasets/nvidia/OpenMathInstruct-2) | 228,408 examples | CC-BY-4.0 |
| SFT | [Orca Math word problems](https://huggingface.co/datasets/microsoft/orca-math-word-problems-200k) | 109,618 examples | MIT |
| SFT | [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) | 202,618 examples | MIT |
| SFT total | Mixed math + English chat SFT | 899,907 examples / 403,406,096 tokens | MIT / Apache-2.0 / CC-BY-4.0 |
| Math GRPO reward | [GSM8K](https://huggingface.co/datasets/openai/gsm8k) | 7,473 prompts / 700 steps | MIT |
| DPO | [distilabel-math-preference](https://huggingface.co/datasets/argilla/distilabel-math-preference-dpo) + [UltraFeedback](https://huggingface.co/datasets/HuggingFaceH4/ultrafeedback_binarized) + [Intel orca_dpo_pairs](https://huggingface.co/datasets/Intel/orca_dpo_pairs) | 73,874 pairs / 1 epoch | Apache-2.0 / MIT |
| English GRPO | [UltraChat](https://huggingface.co/datasets/HuggingFaceH4/ultrachat_200k) prompt bank | 8,000 prompts / 200 steps | MIT |

## License

Weights and code are released under **Apache-2.0**. Use must also comply with the dataset licenses above (including attribution for OpenWebMath and OpenMathInstruct-2).

## Acknowledgements

Capstone artifact of my tutorial **[How to train your first LLM?](https://gsc74.github.io/blogs/train_index.html)**.
