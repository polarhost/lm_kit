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
    "A100": {
        "10M": {"batch_size": 32, "grad_accum": 2},
        "30M": {"batch_size": 16, "grad_accum": 4},
        "100M": {"batch_size": 8, "grad_accum": 8},
    },
}

CONTEXT_LENGTHS = [256, 512, 1024]