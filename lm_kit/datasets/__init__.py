"""
Built-in datasets for testing LoRA fine-tuning.

These are simple text files you can use with kit.encode_twoway_to_lora().

Usage:
    from lm_kit.datasets import pirate_speak

    data = kit.encode_twoway_to_lora(
        pirate_speak,
        system_prompt="You are a pirate."
    )
    dataset = kit.create_instruction_dataset(data)
"""

import os

# Path to this directory
_datasets_dir = os.path.dirname(__file__)

# Built-in dataset paths - just references to the text files
pirate_speak = os.path.join(_datasets_dir, "pirate_speak.txt")
