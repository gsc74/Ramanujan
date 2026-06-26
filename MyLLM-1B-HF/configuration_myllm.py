from transformers import PretrainedConfig


class MyLLMConfig(PretrainedConfig):
    model_type = "myllm"
    keys_to_ignore_at_inference = ["past_key_values"]

    def __init__(
        self,
        vocab_size=65536,
        hidden_size=1792,
        num_hidden_layers=28,
        num_attention_heads=14,
        num_key_value_heads=2,
        head_dim=128,
        intermediate_size=4864,
        max_position_embeddings=8192,
        rope_theta=500000.0,
        rms_norm_eps=1e-5,
        tie_word_embeddings=True,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
        **kwargs,
    ):
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            is_encoder_decoder=False,
            **kwargs,
        )
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.intermediate_size = intermediate_size
        self.max_position_embeddings = max_position_embeddings
        self.rope_theta = rope_theta
        self.rms_norm_eps = rms_norm_eps
        self.hidden_act = "silu"
        self.use_cache = False

