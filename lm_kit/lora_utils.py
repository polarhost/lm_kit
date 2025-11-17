"""Utility functions for LoRA fine-tuning"""

from .configs import LORA_TARGET_MODULES


def detect_lora_target_modules(model):
    """
    Auto-detect which modules to apply LoRA to based on model architecture.

    Args:
        model: The HuggingFace model instance

    Returns:
        list: List of module names to apply LoRA to, or None for PEFT defaults
    """
    model_type = model.config.model_type.lower()

    # Check if we have a predefined mapping
    if model_type in LORA_TARGET_MODULES:
        return LORA_TARGET_MODULES[model_type]

    # For unknown models, return None to use PEFT's automatic detection
    print(f"⚠ Unknown model type '{model_type}', using PEFT auto-detection for target modules")
    return None


def estimate_lora_params(model, lora_r, target_modules=None):
    """
    Estimate the number of trainable parameters with LoRA.

    Args:
        model: The base model
        lora_r: LoRA rank
        target_modules: List of modules to apply LoRA to

    Returns:
        dict: Statistics about trainable parameters
    """
    if target_modules is None:
        target_modules = detect_lora_target_modules(model)

    total_params = sum(p.numel() for p in model.parameters())

    # Rough estimation: for each target module, we add 2 * r * d parameters
    # where d is the hidden dimension
    hidden_dim = model.config.hidden_size
    num_layers = model.config.num_hidden_layers

    if target_modules:
        lora_params = len(target_modules) * num_layers * 2 * lora_r * hidden_dim
    else:
        # Conservative estimate if auto-detecting
        lora_params = 4 * num_layers * 2 * lora_r * hidden_dim

    return {
        "total_params": total_params,
        "lora_params": lora_params,
        "trainable_percentage": (lora_params / total_params) * 100,
    }


def prepare_model_for_kbit_training(model):
    """
    Prepare a quantized model for training with gradient checkpointing.

    Args:
        model: The quantized model

    Returns:
        model: The prepared model
    """
    from peft import prepare_model_for_kbit_training as peft_prepare

    model = peft_prepare(model)

    # Enable gradient checkpointing for memory efficiency
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    return model


def format_chat_template(messages, tokenizer):
    """
    Format messages using the model's chat template.

    Args:
        messages: List of message dicts with 'role' and 'content'
        tokenizer: The tokenizer with chat template

    Returns:
        str: Formatted text
    """
    if hasattr(tokenizer, "apply_chat_template") and tokenizer.chat_template:
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False
        )
    else:
        # Fallback: simple concatenation
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            if role == "system":
                formatted += f"System: {content}\n\n"
            elif role == "user":
                formatted += f"User: {content}\n\n"
            elif role == "assistant":
                formatted += f"Assistant: {content}\n\n"
        return formatted.strip()


def get_memory_footprint(model):
    """
    Calculate approximate memory footprint of a model.

    Args:
        model: The model

    Returns:
        dict: Memory statistics in GB
    """
    total_params = sum(p.numel() for p in model.parameters())

    # Check if model is quantized
    is_4bit = any(hasattr(p, 'quant_state') for p in model.parameters())

    if is_4bit:
        bytes_per_param = 0.5  # 4-bit quantization
    else:
        bytes_per_param = 2  # fp16

    model_size_gb = (total_params * bytes_per_param) / (1024**3)

    return {
        "model_size_gb": model_size_gb,
        "params_millions": total_params / 1e6,
        "quantized": is_4bit,
    }
