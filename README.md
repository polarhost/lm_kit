# lm_kit

A simple, powerful toolkit for training and fine-tuning language models.

## Features

### 🚀 Train Models from Scratch
Train GPT-2 style models (10M, 30M, 100M parameters) from scratch on your own data.

### 🎯 LoRA Fine-Tuning (NEW!)
Fine-tune any HuggingFace model (Llama, DeepSeek, Mistral, etc.) with LoRA:
- **Memory efficient**: 4-bit quantization fits 3B models in ~3GB
- **Fast**: Train only 1-2% of parameters
- **Universal**: Works with any HuggingFace model
- **Easy**: 5 lines of code to fine-tune

## Quick Start

### Training from Scratch
```python
from lm_kit import kit

# Load dataset
dataset = kit.get_dataset("hugging_face", "wikitext", "train[:1%]")

# Create and train model
model = kit.create_model(dataset, model_size="30M", steps=5000)
model.train()

# Generate text
print(model.complete("Once upon a time"))
```

### LoRA Fine-Tuning (NEW!)
```python
from lm_kit import kit

# Load any HuggingFace model with quantization
model = kit.load_hf_model(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization="4bit"  # Fits in ~3GB instead of ~12GB
)

# Create instruction dataset
dataset = kit.create_instruction_dataset([
    {
        "messages": [
            {"role": "system", "content": "You are Shakespeare"},
            {"role": "user", "content": "What is coding?"},
            {"role": "assistant", "content": "Hark! 'Tis the art of..."}
        ]
    },
    # Add 50-100+ examples for best results
])

# Configure LoRA and train
tuned_model = kit.lora_tune(model, dataset, steps=1000)
tuned_model.train()

# Use your fine-tuned model!
response = tuned_model.complete("Tell me about Python")
print(response)  # Responds in Shakespeare style!

# Save the adapter (small file, ~50MB)
tuned_model.save_adapter("./shakespeare_lora")
```

## Installation

```bash
# Clone the repository
git clone <repo-url>
cd lm_kit

# Install (basic)
pip install -e .

# Install with LoRA quantization support
pip install -e ".[quantization]"
```

## LoRA Supported Models

Works with **any HuggingFace model**, including:
- ✓ Llama 2, Llama 3.x (all sizes)
- ✓ DeepSeek V2, V3, DeepSeek-Coder
- ✓ Mistral, Mixtral
- ✓ GPT-2, GPT-NeoX
- ✓ Falcon
- ✓ And many more!

### DeepSeek Example
```python
model = kit.load_hf_model(
    "deepseek-ai/DeepSeek-Coder-V2-Instruct",
    quantization="4bit"
)
```

## Documentation

- **[LORA_GUIDE.md](LORA_GUIDE.md)**: Comprehensive LoRA fine-tuning guide
  - API reference
  - Examples
  - Best practices
  - Troubleshooting

- **[Examples](examples/)**: Working code examples
  - `lora_shakespeare_example.py` - Fine-tune Llama to speak like Shakespeare
  - `lora_deepseek_example.py` - Fine-tune DeepSeek for custom coding style
  - `quick_lora_test.py` - Quick test with GPT-2

## Memory Requirements

### LoRA Fine-Tuning
| Model Size | No Quantization | 4-bit Quantization |
|------------|-----------------|-------------------|
| 1B params  | ~4 GB           | ~2 GB            |
| 3B params  | ~12 GB          | ~3 GB            |
| 7B params  | ~28 GB          | ~7 GB            |
| 13B params | ~52 GB          | ~13 GB           |

### Training from Scratch
- 10M model: ~2 GB (T4, RTX 3090)
- 30M model: ~4 GB (T4, RTX 3090)
- 100M model: ~8 GB (A100)

## Examples

### Style Transfer
```python
# Fine-tune Llama to respond in any style
# - Shakespeare
# - Pirate speak
# - Professional business
# - Casual/friendly
```

### Custom Coding Styles
```python
# Fine-tune DeepSeek to:
# - Add extensive comments
# - Follow specific conventions
# - Use particular patterns
```

### Domain Adaptation
```python
# Adapt models to specific domains:
# - Medical terminology
# - Legal language
# - Technical documentation
```

## Features in Detail

### Train from Scratch
- Pre-configured model sizes (10M, 30M, 100M)
- Custom or standard tokenizers
- GPU-optimized training
- Automatic hyperparameter tuning

### LoRA Fine-Tuning
- Parameter-efficient (train 1-2% of weights)
- Memory-efficient (4-bit/8-bit quantization)
- Fast training (500-3000 steps typical)
- Small adapters (10-100MB vs GBs)
- Multiple presets (small/medium/large)
- Auto-detection for all model types

## Project Structure

```
lm_kit/
├── lm_kit/
│   ├── kit.py           # Main implementation
│   ├── configs.py       # Model and LoRA configs
│   ├── utils.py         # Helper functions
│   └── lora_utils.py    # LoRA utilities
├── examples/            # Working examples
├── LORA_GUIDE.md       # Comprehensive guide
└── setup.py            # Installation
```

## Requirements

- Python 3.7+
- PyTorch 2.0+
- transformers 4.30+
- datasets 2.0+
- peft 0.7+ (for LoRA)
- accelerate 0.24+ (for LoRA)
- bitsandbytes 0.41+ (optional, for quantization)

## License

[Your License Here]

## Contributing

Contributions welcome! Please feel free to submit a Pull Request.

## Citation

If you use this toolkit in your research, please cite:

```bibtex
@software{lm_kit,
  author = {polarhost},
  title = {lm_kit: A Simple Language Model Toolkit},
  year = {2025},
  url = {https://github.com/yourusername/lm_kit}
}
```

## Acknowledgments

- Built on HuggingFace Transformers
- LoRA implementation uses PEFT library
- Inspired by the excellent work of the open-source AI community
