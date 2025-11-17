MODEL_PRESETS = {
    "10M": {
        "n_layer": 6,
        "n_embd": 256,
        "n_head": 8,
        "learning_rate": 5e-4,
        "vocab_size": 50257,
    },
    "30M": {
        "n_layer": 6,
        "n_embd": 384,
        "n_head": 6,
        "learning_rate": 5e-4,
        "vocab_size": 50257,
    },
    "100M": {
        "n_layer": 12,
        "n_embd": 512,
        "n_head": 8,
        "learning_rate": 3e-4,
        "vocab_size": 50257,
    },
}

BATCH_CONFIGS = {
    "T4": {
        "10M": {"batch_size": 16, "grad_accum": 2},
        "30M": {"batch_size": 8, "grad_accum": 4},
        "100M": {"batch_size": 4, "grad_accum": 8},
    },
    "L4": {
        # L4 has ~50% more memory than T4 (24GB vs 16GB)
        "10M": {"batch_size": 24, "grad_accum": 2},
        "30M": {"batch_size": 12, "grad_accum": 4},
        "100M": {"batch_size": 6, "grad_accum": 8},
    },
    "A100": {
        "10M": {"batch_size": 32, "grad_accum": 2},
        "30M": {"batch_size": 16, "grad_accum": 4},
        "100M": {"batch_size": 8, "grad_accum": 8},
    },
}

CONTEXT_LENGTHS = [256, 512, 1024]

# LoRA configuration presets
LORA_PRESETS = {
    "small": {
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "target_modules": None,  # Auto-detect
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "medium": {
        "r": 16,
        "lora_alpha": 32,
        "lora_dropout": 0.05,
        "target_modules": None,  # Auto-detect
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
    "large": {
        "r": 32,
        "lora_alpha": 64,
        "lora_dropout": 0.1,
        "target_modules": None,  # Auto-detect
        "bias": "none",
        "task_type": "CAUSAL_LM",
    },
}

# Quantization configurations
QUANTIZATION_CONFIGS = {
    "4bit": {
        "load_in_4bit": True,
        "bnb_4bit_compute_dtype": "float16",
        "bnb_4bit_quant_type": "nf4",
        "bnb_4bit_use_double_quant": True,
    },
    "8bit": {
        "load_in_8bit": True,
    },
}

# LoRA target modules mapping for different model architectures
LORA_TARGET_MODULES = {
    "llama": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "mistral": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "deepseek": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "deepseek_v2": ["q_proj", "v_proj", "k_proj", "o_proj"],
    "gpt2": ["c_attn"],
    "gpt_neox": ["query_key_value"],
    "falcon": ["query_key_value"],
}