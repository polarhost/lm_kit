from setuptools import setup, find_packages

setup(
    name="lm_kit",
    version="0.1.0",
    packages=find_packages(),
    package_data={
        "lm_kit": ["datasets/*.txt"],  # Include built-in dataset files
    },
    python_requires=">=3.7",
    description="A simple language model kit",
    author="polarhost",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
        "peft>=0.7.0",        # LoRA and parameter-efficient fine-tuning
        "accelerate>=0.24.0", # Memory optimization and multi-GPU support
    ],
    extras_require={
        "quantization": ["bitsandbytes>=0.41.0"],  # Optional: 4-bit/8-bit quantization
    },
)