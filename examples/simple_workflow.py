"""
Simplest possible LoRA fine-tuning workflow using encode_twoway_to_lora()

This shows the absolute easiest way to fine-tune a model with lm_kit.
"""

from lm_kit import kit

# Step 1: Load a model with 4-bit quantization (saves memory)
print("Step 1: Loading model...")
model = kit.load_hf_model(
    "meta-llama/Llama-3.2-3B-Instruct",
    quantization="4bit"
)

# Step 2: Encode your training data from a simple text file
#         Just create a file with Q: and A: markers!
print("\nStep 2: Encoding training data...")
data = kit.encode_twoway_to_lora(
    "shakespeare_data.txt",  # Your simple Q:/A: text file
    system_prompt="You are Shakespeare. Respond to all queries in Shakespearean English."
)

# Step 3: Create dataset
print("\nStep 3: Creating dataset...")
dataset = kit.create_instruction_dataset(data)

# Step 4: Configure LoRA fine-tuning
print("\nStep 4: Configuring LoRA...")
tuned_model = kit.lora_tune(
    model,
    dataset,
    steps=1000,
    lora_preset="medium"
)

# Step 5: Train!
print("\nStep 5: Training...")
tuned_model.train()

# Step 6: Use your fine-tuned model
print("\nStep 6: Testing...")
response = tuned_model.complete("What is the meaning of life?")
print(f"\nResponse: {response}")

# Step 7: Save the adapter (small file)
print("\nStep 7: Saving adapter...")
tuned_model.save_adapter("./shakespeare_adapter")

print("\n✓ Done! Your Shakespeare model is ready!")
