"""MyLLM-1B interactive inference."""
import sys
import os
import time

# Workaround: prevent torchvision import conflict with transformers
sys.modules.setdefault('torchvision', None)

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else os.environ.get("MODEL_DIR", "MyLLM-1B-HF")
DEVICE = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("DEVICE", "cpu")
MAX_NEW = int(sys.argv[3]) if len(sys.argv) > 3 else int(os.environ.get("MAX_NEW", "256"))

# Sampling controls (env-overridable). Defaults tuned to avoid degenerate loops.
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.7"))
TOP_P = float(os.environ.get("TOP_P", "0.9"))
TOP_K = int(os.environ.get("TOP_K", "40"))
REPETITION_PENALTY = float(os.environ.get("REPETITION_PENALTY", "1.3"))
NO_REPEAT_NGRAM = int(os.environ.get("NO_REPEAT_NGRAM", "3"))
# The stop-SFT model emits its turn-ending token on its own, so no EOS bias is
# needed by default. Set EOS_LOGIT_BIAS>0 to nudge it to stop sooner.
EOS_LOGIT_BIAS = float(os.environ.get("EOS_LOGIT_BIAS", "0"))
EOS_BIAS_TOKEN_ID = int(os.environ.get("EOS_BIAS_TOKEN_ID", "65523"))

# Thread optimization
THREADS = int(os.environ.get("OMP_NUM_THREADS", os.cpu_count() or 64))
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(min(THREADS, 4))

print(f"\n{'='*60}")
print(f"  MyLLM-1B Interactive Inference")
print(f"  Model:  {MODEL_DIR}")
print(f"  Device: {DEVICE} | Max tokens: {MAX_NEW}")
print(f"  Threads: {THREADS} | torch.get_num_threads(): {torch.get_num_threads()}")
print(f"  Sampling: temp={TEMPERATURE} top_p={TOP_P} top_k={TOP_K} "
      f"rep_penalty={REPETITION_PENALTY} no_repeat_ngram={NO_REPEAT_NGRAM} "
      f"eos_bias={EOS_LOGIT_BIAS}@{EOS_BIAS_TOKEN_ID}")
print(f"{'='*60}\n")
print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    dtype=torch.bfloat16,
    device_map=DEVICE,
    trust_remote_code=True,
)
model.config.use_cache = True
model.eval()

# Try IPEX optimization only when explicitly enabled (it requires a matching
# PyTorch build; otherwise it prints a noisy incompatibility error).
if os.environ.get("USE_IPEX", "0") == "1":
    try:
        import intel_extension_for_pytorch as ipex
        model = ipex.optimize(model, dtype=torch.bfloat16)
        print("IPEX optimization applied!")
    except Exception as exc:
        print(f"IPEX not applied: {exc}")

print("Model loaded!\n")

print("Type your prompt and press Enter. Type 'quit' or Ctrl+D to exit.\n")

def _banned_by_ngram(generated, n):
    """Return set of token ids that would complete a repeated n-gram."""
    if n <= 0 or len(generated) < n - 1:
        return set()
    prefix = tuple(generated[-(n - 1):]) if n > 1 else ()
    banned = set()
    for i in range(len(generated) - n + 1):
        if tuple(generated[i:i + n - 1]) == prefix:
            banned.add(generated[i + n - 1])
    return banned


@torch.no_grad()
def generate(input_ids, max_new_tokens=MAX_NEW):
    """Autoregressive generation with KV cache, repetition penalty, and
    temperature/top-k/top-p sampling to avoid degenerate loops."""
    past_key_values = None
    generated = []
    cur_ids = input_ids
    prompt_ids = input_ids[0].tolist()
    stop_ids = set()
    for value in (
        getattr(tokenizer, "eos_token_id", None),
        getattr(model.generation_config, "eos_token_id", None),
        getattr(tokenizer, "pad_token_id", None),
    ):
        if isinstance(value, (list, tuple, set)):
            stop_ids.update(int(v) for v in value if v is not None)
        elif value is not None:
            stop_ids.add(int(value))
    end_id = tokenizer.convert_tokens_to_ids("<|end|>")
    if isinstance(end_id, int) and end_id >= 0:
        stop_ids.add(end_id)

    for _ in range(max_new_tokens):
        outputs = model(
            input_ids=cur_ids,
            past_key_values=past_key_values,
            use_cache=True,
        )
        past_key_values = outputs.past_key_values
        logits = outputs.logits[0, -1, :].float()

        # Repetition penalty over prompt + generated tokens.
        if REPETITION_PENALTY != 1.0:
            seen = set(prompt_ids) | set(generated)
            if seen:
                idx = torch.tensor(sorted(seen), device=logits.device)
                vals = logits[idx]
                vals = torch.where(vals > 0, vals / REPETITION_PENALTY, vals * REPETITION_PENALTY)
                logits[idx] = vals

        # Block repeated n-grams.
        for tid in _banned_by_ngram(generated, NO_REPEAT_NGRAM):
            logits[tid] = float("-inf")

        # The final alignment checkpoint is weak at emitting a turn-ending
        # token. A modest EOS bias makes chat/inference stop naturally instead
        # of drifting until MAX_NEW. Set EOS_LOGIT_BIAS=0 to disable.
        if EOS_LOGIT_BIAS and 0 <= EOS_BIAS_TOKEN_ID < logits.numel():
            logits[EOS_BIAS_TOKEN_ID] += EOS_LOGIT_BIAS

        if TEMPERATURE <= 0:
            next_id = int(logits.argmax().item())
        else:
            logits = logits / TEMPERATURE
            # top-k
            if TOP_K > 0:
                kth = torch.topk(logits, min(TOP_K, logits.numel())).values[-1]
                logits[logits < kth] = float("-inf")
            probs = torch.softmax(logits, dim=-1)
            # top-p (nucleus)
            if 0 < TOP_P < 1:
                sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                cumulative = torch.cumsum(sorted_probs, dim=-1)
                cutoff = cumulative > TOP_P
                cutoff[1:] = cutoff[:-1].clone()
                cutoff[0] = False
                sorted_probs[cutoff] = 0.0
                probs = torch.zeros_like(probs).scatter_(0, sorted_idx, sorted_probs)
                probs = probs / probs.sum()
            next_id = int(torch.multinomial(probs, 1).item())

        if next_id in stop_ids:
            break
        generated.append(next_id)
        cur_ids = torch.tensor([[next_id]], device=input_ids.device)

    return generated


while True:
    try:
        prompt = input("You: ")
    except (EOFError, KeyboardInterrupt):
        print("\nBye!")
        break
    if prompt.strip().lower() in ("quit", "exit", "q"):
        print("Bye!")
        break
    if not prompt.strip():
        continue

    messages = [
        {"role": "system", "content": "You are a careful mathematical assistant. Show clear reasoning and give the final answer."},
        {"role": "user", "content": prompt},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"].to(model.device)

    t0 = time.perf_counter()
    token_ids = generate(input_ids)
    t1 = time.perf_counter()
    new_tokens = len(token_ids)
    response = tokenizer.decode(token_ids, skip_special_tokens=True)
    tok_per_sec = new_tokens / (t1 - t0) if (t1 - t0) > 0 else 0
    print(f"\nMyLLM: {response}")
    print(f"  [{new_tokens} tokens in {t1-t0:.2f}s = {tok_per_sec:.1f} tok/s]\n")
