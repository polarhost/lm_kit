from setuptools import setup, find_packages

setup(
    name="lm_kit",
    version="0.1.0",
    packages=find_packages(),
    python_requires=">=3.7",
    description="A simple language model kit",
    author="polarhost",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "datasets>=2.0.0",
    ],
)