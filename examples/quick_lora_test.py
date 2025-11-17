"""
Quick LoRA test with GPT-2 (no GPU required, runs on CPU)

This is a minimal example for testing the LoRA implementation
without needing a large model or GPU.
"""

from lm_kit import kit

# Minimal training data
test_data = [
    {
        "messages": [
            {"role": "user", "content": "Say hello"},
            {"role": "assistant", "content": "Greetings, noble friend!"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "How are you?"},
            {"role": "assistant", "content": "I fare well, good traveler!"}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Goodbye"},
            {"role": "assistant", "content": "Farewell, until we meet again!"}
        ]
    },
]

def main():
    print("Quick LoRA Test with GPT-2")
    print("=" * 70)

    # Load GPT-2 (small model, works without GPU)
    print("\n1. Loading GPT-2 model...")
    model = kit.load_hf_model("gpt2")  # No quantization needed for small model

    # Create dataset
    print("\n2. Creating dataset...")
    dataset = kit.create_instruction_dataset(test_data)

    # Configure LoRA
    print("\n3. Configuring LoRA...")
    tuned_model = kit.lora_tune(
        hf_model=model,
        dataset=dataset,
        lora_preset="small",  # Small preset for quick testing
        steps=10,  # Just 10 steps for quick test
        batch_size=1,
        gradient_accumulation_steps=1,
    )

    # Train
    print("\n4. Training (10 steps, should be quick)...")
    tuned_model.train()

    # Test
    print("\n5. Testing...")
    response = tuned_model.complete("Say hello", max_length=50)
    print(f"\nPrompt: Say hello")
    print(f"Response: {response}")

    print("\n✓ Test complete! LoRA implementation is working.")
    print("\nNext steps:")
    print("- Try examples/lora_shakespeare_example.py for a real use case")
    print("- Use a larger model like Llama with quantization")
    print("- Add more training examples (50-100+) for better results")


if __name__ == "__main__":
    main()
