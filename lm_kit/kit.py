import torch
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as HFDataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from .configs import MODEL_PRESETS, BATCH_CONFIGS
from .utils import detect_gpu, suggest_context_length, estimate_total_tokens, calculate_training_steps

class Dataset:
    """Wrapper around HuggingFace dataset with analysis"""
    
    def __init__(self, hf_dataset, tokenizer):
        self.raw = hf_dataset
        self.tokenizer = tokenizer
        self.total_tokens = None
        self.context_length = None
        
    def analyze(self):
        """Analyze dataset and print stats"""
        print("Analyzing dataset...")
        
        # Get stats
        self.context_length = suggest_context_length(self.raw, self.tokenizer)
        self.total_tokens = estimate_total_tokens(self.raw, self.tokenizer)
        
        # Detect GPU
        gpu = detect_gpu()
        
        print(f"\nDataset Analysis")
        print("-" * 50)
        print(f"Examples: {len(self.raw):,}")
        print(f"Total tokens: ~{self.total_tokens/1e6:.0f}M")
        print(f"Recommended context length: {self.context_length}")
        print(f"\nDetected GPU: {gpu}")
        print(f"\nAvailable model sizes: 10M, 30M, 100M")
        
        return self
    
    def tokenize(self, context_length):
        """Tokenize the dataset"""
        def tokenize_fn(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                max_length=context_length,
                padding=False
            )
        
        print(f"Tokenizing dataset (context length: {context_length})...")
        tokenized = self.raw.map(
            tokenize_fn,
            batched=True,
            remove_columns=["text"],
            desc="Tokenizing"
        )
        print(f"Tokenized {len(tokenized):,} examples\n")
        
        return tokenized


class Model:
    """Language model wrapper"""
    
    def __init__(self, model, tokenizer, trainer, model_size):
        self.model = model
        self.tokenizer = tokenizer
        self.trainer = trainer
        self.model_size = model_size
        
    def train(self):
        """Train the model"""
        print("=" * 70)
        print("STARTING TRAINING")
        print("=" * 70)
        
        result = self.trainer.train()
        
        print("\n" + "=" * 70)
        print("TRAINING COMPLETE!")
        print("=" * 70)
        print(f"Final loss: {result.training_loss:.4f}\n")
        
        self.model.eval()
        return self
    
    def complete(self, prompt, max_length=200, temperature=0.8):
        """Generate text completion"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=max_length,
                temperature=temperature,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                repetition_penalty=1.2,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def save(self, path):
        """Save model to disk"""
        self.trainer.save_model(path)
        self.tokenizer.save_pretrained(path)
        print(f"Model saved to {path}")


class Kit:
    """Main kit object with all methods"""
    
    @staticmethod
    def get_dataset(source, name, split):
        """Load dataset from HuggingFace"""
        if source != "hugging_face":
            raise ValueError("Only 'hugging_face' source supported for now")
        
        print(f"Loading dataset: {name} ({split})")
        
        try:
            raw_dataset = hf_load_dataset(name, split=split)
        except Exception as e:
            raise ValueError(f"Could not load dataset '{name}'. Error: {str(e)}")
        
        print(f"Loaded {len(raw_dataset):,} examples\n")
        
        # Load tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        
        return Dataset(raw_dataset, tokenizer)
    
    @staticmethod
    def create_model(dataset, model_size="30M"):
        """Create and configure model for training"""
        
        if model_size not in MODEL_PRESETS:
            raise ValueError(f"model_size must be one of: {list(MODEL_PRESETS.keys())}")
        
        # Analyze dataset if not done yet
        if dataset.context_length is None:
            dataset.analyze()
        
        # Get model config
        config_dict = MODEL_PRESETS[model_size].copy()
        config_dict['n_positions'] = dataset.context_length
        
        config = GPT2Config(**config_dict)
        model = GPT2LMHeadModel(config)
        
        num_params = sum(p.numel() for p in model.parameters())
        print(f"Creating {model_size} model")
        print("-" * 50)
        print(f"Parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"Layers: {config.n_layer}")
        print(f"Embedding dim: {config.n_embd}")
        print(f"Attention heads: {config.n_head}")
        print(f"Context length: {config.n_positions}\n")
        
        # Get GPU-specific batch config
        gpu = detect_gpu()
        batch_config = BATCH_CONFIGS[gpu][model_size]
        
        # Calculate training steps
        steps = calculate_training_steps(
            dataset.total_tokens,
            model_size,
            batch_config['batch_size'],
            batch_config['grad_accum'],
            dataset.context_length
        )
        
        print(f"Training Configuration")
        print("-" * 50)
        print(f"Batch size: {batch_config['batch_size']}")
        print(f"Gradient accumulation: {batch_config['grad_accum']}")
        print(f"Effective batch size: {batch_config['batch_size'] * batch_config['grad_accum']}")
        print(f"Training steps: {steps:,}")
        print(f"Warmup steps: {steps // 20}")
        print(f"Learning rate: {config_dict['learning_rate']}\n")
        
        # Tokenize dataset
        tokenized_dataset = dataset.tokenize(dataset.context_length)
        
        # Setup training
        training_args = TrainingArguments(
            output_dir=f"./lm_kit_{model_size}",
            overwrite_output_dir=True,
            per_device_train_batch_size=batch_config['batch_size'],
            gradient_accumulation_steps=batch_config['grad_accum'],
            max_steps=steps,
            warmup_steps=steps // 20,
            learning_rate=config_dict['learning_rate'],
            save_steps=steps // 3,
            save_total_limit=2,
            logging_steps=100,
            fp16=True,
            report_to="none",
            disable_tqdm=False,
        )
        
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=dataset.tokenizer,
            mlm=False
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )
        
        if torch.cuda.is_available():
            model = model.cuda()
        
        return Model(model, dataset.tokenizer, trainer, model_size)
    
    
@staticmethod
def create_paper_model():
    """
    Create a tiny model with minimal dataset for testing your setup.
    This trains in ~30 seconds and verifies everything works.
    """
    print("Creating paper model for testing.")
    
    fake_data = {
        "text": [
            "The quick brown fox jumps over the lazy dog.",
            "Hello world, this is a test sentence.",
            "A pimento cheese sandwich consists of one slice of bread, with the correct amount of cheese applied."
        ]
    }
    
    raw_dataset = HFDataset.from_dict(fake_data)
    
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    config = GPT2Config(
        vocab_size=50257,
        n_positions=128,
        n_embd=64,
        n_layer=2,
        n_head=2,
    )
    
    model = GPT2LMHeadModel(config)
    num_params = sum(p.numel() for p in model.parameters())
    
    print(f"Model Configuration")
    print("-" * 50)
    print(f"Parameters: {num_params:,} ({num_params/1e6:.2f}M)")
    print(f"Layers: 2")
    print(f"Embedding dim: 64")
    print(f"Dataset: 3 examples")
    print(f"Training steps: 10\n")
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=128,
            padding=False
        )
    
    tokenized_dataset = raw_dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
    )
    
    training_args = TrainingArguments(
        output_dir="./paper_model_test",
        overwrite_output_dir=True,
        per_device_train_batch_size=1,
        max_steps=10,
        logging_steps=5,
        fp16=torch.cuda.is_available(),
        report_to="none",
        disable_tqdm=False,
        learning_rate=5e-4,
    )
    
    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset,
        data_collator=data_collator,
    )
    
    if torch.cuda.is_available():
        model = model.cuda()
        print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")
    else:
        print("⚠ No GPU detected - will be slow")
    
    print("\nStarting quick training test...\n")
    
    trainer.train()
    
    print("Training succeeded. Setup works.")
    
    model.eval()
    
    class PaperModel:
        def __init__(self, model, tokenizer):
            self.model = model
            self.tokenizer = tokenizer
        
        def complete(self, prompt):
            """Generate text (won't be good, but proves it works)"""
            print("Testing text generation")
            
            inputs = self.tokenizer(prompt, return_tensors="pt")
            
            if torch.cuda.is_available():
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=50,
                    temperature=0.8,
                    do_sample=True,
                    top_k=50,
                    pad_token_id=self.tokenizer.eos_token_id,
                )
            
            result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            print(f"Prompt: '{prompt}'")
            print(f"Output: '{result}'\n")
            print("✓ Generation works!")
            
            return result
        
        def save(self, path):
            """You don't need to save paper models"""
            print("Paper models are just for testing. No need to save.")
    
    return PaperModel(model, tokenizer)

kit = Kit()