import torch
import numpy as np
from transformers import GPT2Tokenizer

def detect_gpu():
    """Detect if we have T4 or A100"""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU detected! This package requires a GPU.")
    
    gpu_name = torch.cuda.get_device_name(0)
    
    if "A100" in gpu_name:
        return "A100"
    else:
        return "T4"

def suggest_context_length(dataset, tokenizer):
    """Figure out good context length based on dataset"""
    sample_size = min(1000, len(dataset))
    lengths = []
    
    for i in range(sample_size):
        tokens = tokenizer(dataset[i]['text'], truncation=False)['input_ids']
        lengths.append(len(tokens))
    
    median = np.median(lengths)
    
    # Pick the smallest context length that fits most texts
    if median <= 128:
        return 256
    elif median <= 256:
        return 512
    else:
        return 1024

def estimate_total_tokens(dataset, tokenizer):
    """Estimate total tokens in dataset"""
    # Sample 1% to estimate
    sample_size = max(100, len(dataset) // 100)
    sample_tokens = 0
    
    for i in range(sample_size):
        tokens = tokenizer(dataset[i]['text'], truncation=False)['input_ids']
        sample_tokens += len(tokens)
    
    # Extrapolate to full dataset
    avg_tokens = sample_tokens / sample_size
    total = int(avg_tokens * len(dataset))
    
    return total

def calculate_training_steps(total_tokens, model_size, batch_size, grad_accum, context_length):
    """Calculate how many steps to train"""
    target_tokens = {
        "10M": 10_000_000_000,
        "30M": 20_000_000_000,
        "100M": 30_000_000_000,
    }
    
    tokens_per_step = batch_size * grad_accum * context_length
    steps = target_tokens[model_size] // tokens_per_step
    
    # Clamp to reasonable range
    return max(5000, min(steps, 50000))