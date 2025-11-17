"""
Example: Fine-tune DeepSeek Coder to write code in a specific style using LoRA

This example demonstrates LoRA fine-tuning with DeepSeek models.
"""

from lm_kit import kit

# Example data: Teaching DeepSeek to write code with extensive comments
coding_style_data = [
    {
        "messages": [
            {"role": "user", "content": "Write a function to check if a number is prime"},
            {"role": "assistant", "content": """def is_prime(n):
    # First, handle edge cases for numbers less than 2
    # Numbers less than 2 are not considered prime by mathematical definition
    if n < 2:
        return False

    # Check divisibility from 2 up to sqrt(n)
    # This optimization works because if n has a divisor greater than sqrt(n),
    # it must also have a corresponding divisor less than sqrt(n)
    for i in range(2, int(n ** 0.5) + 1):
        # If n is divisible by i, it's not prime
        if n % i == 0:
            return False

    # If no divisors were found, n is prime
    return True"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Create a function to reverse a string"},
            {"role": "assistant", "content": """def reverse_string(s):
    # Python provides multiple ways to reverse a string
    # Using slicing with [::-1] is the most Pythonic approach

    # The slice notation [start:stop:step] with step=-1
    # traverses the string backwards, creating a reversed copy
    return s[::-1]

    # Alternative approaches:
    # - ''.join(reversed(s))  # Using built-in reversed()
    # - s[::-1]  # Most concise
    # - Manual iteration with a loop"""}
        ]
    },
    {
        "messages": [
            {"role": "user", "content": "Implement binary search"},
            {"role": "assistant", "content": """def binary_search(arr, target):
    # Binary search requires a sorted array
    # Time complexity: O(log n), Space complexity: O(1)

    # Initialize two pointers at the start and end of the array
    left, right = 0, len(arr) - 1

    # Continue searching while the search space is valid
    while left <= right:
        # Calculate middle index, avoiding potential integer overflow
        # Using (left + right) // 2 could overflow with very large indices
        mid = left + (right - left) // 2

        # Check if we found the target at the middle position
        if arr[mid] == target:
            return mid  # Target found, return its index

        # If target is greater, ignore left half by moving left pointer
        elif arr[mid] < target:
            left = mid + 1

        # If target is smaller, ignore right half by moving right pointer
        else:
            right = mid - 1

    # Target not found in the array
    return -1"""}
        ]
    },
    # Add more examples for better results
]

def main():
    print("=" * 70)
    print("DEEPSEEK CODER LORA FINE-TUNING EXAMPLE")
    print("=" * 70)
    print()

    # Load DeepSeek Coder model
    print("Loading DeepSeek Coder model...")
    model = kit.load_hf_model(
        "deepseek-ai/DeepSeek-Coder-V2-Instruct",
        quantization="4bit"
    )

    # Create dataset
    print("\nCreating instruction dataset...")
    dataset = kit.create_instruction_dataset(
        coding_style_data,
        format_type="conversational"
    )

    # Configure LoRA
    print("\nConfiguring LoRA fine-tuning...")
    tuned_model = kit.lora_tune(
        hf_model=model,
        dataset=dataset,
        lora_preset="medium",
        learning_rate=2e-4,
        steps=100,  # Increase for production
        batch_size=1,
        gradient_accumulation_steps=4,
        output_dir="./deepseek_lora"
    )

    # Train
    print("\nTraining...")
    tuned_model.train()

    # Test
    print("\nTesting fine-tuned model...")
    test_prompt = "Write a function to calculate factorial"
    response = tuned_model.complete(test_prompt, max_length=300)
    print(f"\nPrompt: {test_prompt}")
    print(f"Response:\n{response}")

    # Save
    tuned_model.save_adapter("./deepseek_coder_lora_adapter")
    print("\n✓ DeepSeek model fine-tuned and saved!")


if __name__ == "__main__":
    main()
