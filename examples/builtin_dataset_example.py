"""
Example: Using built-in datasets for quick LoRA testing

Built-in datasets are just text file paths that you can import
and use directly with encode_twoway_to_lora(). Simple!
"""

from lm_kit import kit
from lm_kit.datasets import pirate_speak  # Import the dataset path

def simple_example():
    """Simplest way to use a built-in dataset"""
    print("=" * 70)
    print("USING BUILT-IN PIRATE DATASET")
    print("=" * 70)

    # pirate_speak is just a path to the text file
    # Use it directly with encode_twoway_to_lora()
    data = kit.encode_twoway_to_lora(
        pirate_speak,
        system_prompt="You are a pirate. Respond to all queries in pirate speak."
    )

    # Create dataset
    dataset = kit.create_instruction_dataset(data)

    print("✓ Dataset ready for training!")
    print("\nNext steps:")
    print("  model = kit.load_hf_model('gpt2')")
    print("  tuned = kit.lora_tune(model, dataset, steps=100)")
    print("  tuned.train()")
    print()


def full_training_example():
    """Complete workflow with built-in dataset"""
    print("=" * 70)
    print("FULL TRAINING EXAMPLE")
    print("=" * 70)

    # Load model
    print("\n1. Loading model...")
    model = kit.load_hf_model("gpt2")

    # Use built-in dataset
    print("\n2. Loading pirate dataset...")
    data = kit.encode_twoway_to_lora(
        pirate_speak,  # Just the imported path
        system_prompt="You are a pirate. Respond to all queries in pirate speak."
    )

    # Create dataset
    print("\n3. Creating instruction dataset...")
    dataset = kit.create_instruction_dataset(data)

    # Configure LoRA
    print("\n4. Configuring LoRA...")
    tuned_model = kit.lora_tune(
        model,
        dataset,
        lora_preset="small",
        steps=50,  # Quick test
        batch_size=1,
    )

    # Train
    print("\n5. Training...")
    tuned_model.train()

    # Test
    print("\n6. Testing...")
    response = tuned_model.complete("How do I learn Python?", max_length=100)
    print(f"\nResponse: {response}")

    # Save
    print("\n7. Saving...")
    tuned_model.save_adapter("./pirate_adapter")

    print("\n✓ Done!")


if __name__ == "__main__":
    # Show simple usage
    simple_example()

    # Uncomment to run full training
    # full_training_example()

    print("=" * 70)
    print("KEY POINTS")
    print("=" * 70)
    print("✓ Import datasets: from lm_kit.datasets import pirate_speak")
    print("✓ They're just file paths - use with encode_twoway_to_lora()")
    print("✓ Intellisense shows what's available")
    print("✓ Simple and straightforward!")
    print()
