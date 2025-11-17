"""
Example: Using encode_twoway_to_lora() for easy dataset creation

This shows how to create LoRA datasets from simple text files with Q: and A: markers.
Much easier than manually creating the nested dictionary structure!
"""

from lm_kit import kit

def example_from_file():
    """Load and encode from a text file"""
    print("=" * 70)
    print("EXAMPLE 1: Loading from file")
    print("=" * 70)

    # Simply encode the file - it handles everything!
    shakespeare_data = kit.encode_twoway_to_lora(
        "shakespeare_data.txt",
        input_marker="Q:",
        output_marker="A:",
        system_prompt="You are Shakespeare. Respond to all queries in Shakespearean English."
    )

    print(f"Loaded {len(shakespeare_data)} examples")
    print("\nFirst example:")
    print(shakespeare_data[0])
    print()

    return shakespeare_data


def example_from_string():
    """Encode from a text string"""
    print("=" * 70)
    print("EXAMPLE 2: Encoding from string")
    print("=" * 70)

    text = """
    Q: Say hello
    A: Good morrow to thee, noble friend!

    Q: How are you?
    A: I fare most excellently, I thank thee for thy kind inquiry!

    Q: Goodbye
    A: Farewell, good traveler! May fortune smile upon thy journey!
    """

    data = kit.encode_twoway_to_lora(
        text,
        system_prompt="You are Shakespeare"
    )

    print(f"Encoded {len(data)} examples")
    print("\nFirst example:")
    print(data[0])
    print()

    return data


def example_custom_markers():
    """Use custom markers instead of Q: and A:"""
    print("=" * 70)
    print("EXAMPLE 3: Custom markers")
    print("=" * 70)

    text = """
    Input: Translate to French: Hello
    Output: Bonjour

    Input: Translate to French: Goodbye
    Output: Au revoir

    Input: Translate to French: Thank you
    Output: Merci
    """

    data = kit.encode_twoway_to_lora(
        text,
        input_marker="Input:",
        output_marker="Output:",
        system_prompt="You are a French translator"
    )

    print(f"Encoded {len(data)} examples")
    print("\nFirst example:")
    print(data[0])
    print()

    return data


def example_completion_format():
    """Use completion format instead of conversational"""
    print("=" * 70)
    print("EXAMPLE 4: Completion format")
    print("=" * 70)

    text = """
    Q: What is 2+2?
    A: 4

    Q: What is the capital of France?
    A: Paris
    """

    data = kit.encode_twoway_to_lora(
        text,
        format_type="completion"  # No system prompt in completion format
    )

    print(f"Encoded {len(data)} examples")
    print("\nFirst example:")
    print(data[0])
    print()

    return data


def full_workflow_example():
    """Complete workflow: encode -> create dataset -> train"""
    print("=" * 70)
    print("EXAMPLE 5: Full workflow")
    print("=" * 70)

    # Step 1: Encode from file
    data = kit.encode_twoway_to_lora(
        "shakespeare_data.txt",
        system_prompt="You are Shakespeare. Respond to all queries in Shakespearean English."
    )

    # Step 2: Create dataset
    dataset = kit.create_instruction_dataset(data)

    print("✓ Dataset ready for training!")
    print("\nNext steps:")
    print("  model = kit.load_hf_model('meta-llama/Llama-3.2-3B-Instruct', quantization='4bit')")
    print("  tuned_model = kit.lora_tune(model, dataset, steps=1000)")
    print("  tuned_model.train()")
    print()


if __name__ == "__main__":
    # Run all examples
    example_from_file()
    example_from_string()
    example_custom_markers()
    example_completion_format()
    full_workflow_example()

    print("=" * 70)
    print("✓ All examples completed!")
    print("=" * 70)
    print("\nKey takeaways:")
    print("- Use encode_twoway_to_lora() to convert simple Q:/A: text to training format")
    print("- Works with files or strings")
    print("- Customize markers (Q:/A:, Input:/Output:, etc.)")
    print("- Much easier than manually creating nested dictionaries!")
    print("- Supports both conversational and completion formats")
