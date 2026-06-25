import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import GenerationMixin, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast

from .configuration_ramanujan import RamanujanConfig


class RamanujanRMSNorm(nn.Module):
    def __init__(self, size, eps):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(size))
        self.eps = eps

    def forward(self, x):
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)


def rotate_half_interleaved(x):
    return torch.stack((-x[..., 1::2], x[..., ::2]), dim=-1).flatten(-2)


class RamanujanRotaryEmbedding(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.head_dim = config.head_dim
        self.rope_theta = config.rope_theta

    def forward(self, q, k, position_ids):
        # Compute inv_freq on the fly so it never relies on a registered buffer
        # (non-persistent buffers are not materialized under meta-device loading).
        inv_freq = 1.0 / (
            self.rope_theta
            ** (torch.arange(0, self.head_dim, 2, dtype=torch.float32, device=q.device) / self.head_dim)
        )
        frequencies = position_ids.float().unsqueeze(-1) * inv_freq.view(1, 1, -1)
        embedding = torch.repeat_interleave(frequencies, 2, dim=-1).unsqueeze(1)
        cos, sin = embedding.cos().to(q.dtype), embedding.sin().to(q.dtype)
        return q * cos + rotate_half_interleaved(q) * sin, k * cos + rotate_half_interleaved(k) * sin


class RamanujanAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_attention_heads
        self.num_kv_heads = config.num_key_value_heads
        self.head_dim = config.head_dim
        self.q_proj = nn.Linear(config.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, config.hidden_size, bias=False)
        self.rope = RamanujanRotaryEmbedding(config)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None,
                use_cache=False, cache=None, layer_idx=None):
        batch, length, _ = x.shape
        q = self.q_proj(x).view(batch, length, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch, length, self.num_kv_heads, self.head_dim).transpose(1, 2)
        if position_ids is None:
            position_ids = torch.arange(length, device=x.device).view(1, -1).expand(batch, -1)
        q, k = self.rope(q, k, position_ids)

        # KV cache: support both a transformers Cache object (model.generate) and
        # the legacy per-layer tuple format (manual decoding loops).
        if cache is not None:
            k, v = cache.update(k, v, layer_idx)
            new_past_key_value = None
        else:
            if past_key_value is not None:
                k = torch.cat([past_key_value[0], k], dim=2)
                v = torch.cat([past_key_value[1], v], dim=2)
            new_past_key_value = (k, v) if use_cache else None

        kv_len = k.shape[2]
        if attention_mask is None or bool(attention_mask.all()):
            if length == kv_len:
                y = F.scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
            else:
                # Decoding step: single query against full KV, no causal mask needed
                y = F.scaled_dot_product_attention(q, k, v, is_causal=False, enable_gqa=True)
        else:
            causal = torch.ones(length, kv_len, dtype=torch.bool, device=x.device)
            causal[:, :kv_len].tril_(diagonal=kv_len - length)
            allowed = causal.view(1, 1, length, kv_len) & attention_mask.bool().view(batch, 1, 1, kv_len)
            y = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, enable_gqa=True)
        return self.o_proj(y.transpose(1, 2).contiguous().view(batch, length, -1)), new_past_key_value


class RamanujanSwiGLU(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class RamanujanBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.input_norm = RamanujanRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.attention = RamanujanAttention(config)
        self.post_attention_norm = RamanujanRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.mlp = RamanujanSwiGLU(config)

    def forward(self, x, attention_mask=None, position_ids=None, past_key_value=None,
                use_cache=False, cache=None, layer_idx=None):
        attn_out, new_past_key_value = self.attention(
            self.input_norm(x), attention_mask, position_ids,
            past_key_value=past_key_value, use_cache=use_cache,
            cache=cache, layer_idx=layer_idx,
        )
        x = x + attn_out
        return x + self.mlp(self.post_attention_norm(x)), new_past_key_value


class RamanujanPreTrainedModel(PreTrainedModel):
    config_class = RamanujanConfig
    base_model_prefix = ""
    _no_split_modules = ["RamanujanBlock"]
    _supports_flash_attn = False
    _supports_sdpa = True
    _supports_cache_class = False
    _supports_static_cache = False

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)


class RamanujanForCausalLM(RamanujanPreTrainedModel, GenerationMixin):
    _tied_weights_keys = {"lm_head.weight": "embed_tokens.weight"}

    def __init__(self, config):
        super().__init__(config)
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList([RamanujanBlock(config) for _ in range(config.num_hidden_layers)])
        self.norm = RamanujanRMSNorm(config.hidden_size, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def get_input_embeddings(self):
        return self.embed_tokens

    def set_input_embeddings(self, value):
        self.embed_tokens = value

    def get_output_embeddings(self):
        return self.lm_head

    def set_output_embeddings(self, value):
        self.lm_head = value

    def prepare_inputs_for_generation(self, input_ids, attention_mask=None, past_key_values=None, **kwargs):
        past_length = 0
        if past_key_values is not None:
            if isinstance(past_key_values, (list, tuple)):
                if len(past_key_values) > 0 and past_key_values[0] is not None:
                    past_length = past_key_values[0][0].shape[2]
            else:
                past_length = past_key_values.get_seq_length()
        if past_length > 0:
            # Only feed the newest token once the prompt is already cached.
            input_ids = input_ids[:, -1:]
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "past_key_values": past_key_values,
            "use_cache": True,
        }

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        position_ids=None,
        labels=None,
        use_cache=False,
        past_key_values=None,
        return_dict=None,
        **kwargs,
    ):
        if input_ids is None:
            raise ValueError("input_ids must be provided")

        # Figure out how many tokens are already cached so positions line up. The
        # cache may be a transformers Cache object (model.generate) or a legacy
        # per-layer tuple (manual decoding loops).
        is_cache_obj = past_key_values is not None and not isinstance(past_key_values, (list, tuple))
        is_legacy_cache = isinstance(past_key_values, (list, tuple)) and len(past_key_values) > 0
        past_length = 0
        if is_cache_obj:
            past_length = past_key_values.get_seq_length()
        elif is_legacy_cache and past_key_values[0] is not None:
            past_length = past_key_values[0][0].shape[2]

        if position_ids is None:
            if attention_mask is not None:
                position_ids = attention_mask.long().cumsum(-1).sub(1).clamp_min(0)
                # Keep only the positions for the tokens processed in this step.
                if position_ids.shape[1] != input_ids.shape[1]:
                    position_ids = position_ids[:, -input_ids.shape[1]:]
            else:
                position_ids = torch.arange(
                    past_length, past_length + input_ids.shape[1], device=input_ids.device
                ).unsqueeze(0).expand(input_ids.shape[0], -1)

        hidden = self.embed_tokens(input_ids)
        new_past_key_values = [] if (use_cache and not is_cache_obj) else None
        for i, layer in enumerate(self.layers):
            if is_cache_obj:
                hidden, _ = layer(hidden, attention_mask, position_ids,
                                  use_cache=use_cache, cache=past_key_values, layer_idx=i)
            else:
                past_kv = past_key_values[i] if (is_legacy_cache and i < len(past_key_values)) else None
                hidden, new_past_kv = layer(hidden, attention_mask, position_ids,
                                            past_key_value=past_kv, use_cache=use_cache)
                if use_cache:
                    new_past_key_values.append(new_past_kv)

        logits = self.lm_head(self.norm(hidden))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].float().reshape(-1, self.config.vocab_size),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
        if return_dict is False:
            return ((loss,) if loss is not None else ()) + (logits,)
        if use_cache:
            present = past_key_values if is_cache_obj else tuple(new_past_key_values)
        else:
            present = None
        return CausalLMOutputWithPast(
            loss=loss, logits=logits,
            past_key_values=present,
        )

