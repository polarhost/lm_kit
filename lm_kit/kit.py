import torch
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as HFDataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling
)
from .configs import MODEL_PRESETS, BATCH_CONFIGS
from .utils import detect_gpu, suggest_context_length, estimate_total_tokens, calculate_training_steps

def _train_custom_tokenizer(dataset, vocab_size=16384):
    """
    Train a custom BPE tokenizer on the dataset.
    Uses the same algorithm as GPT-2 for compatibility.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers
    from tokenizers.processors import TemplateProcessing
    from transformers import PreTrainedTokenizerFast
    
    print(f"Training custom BPE tokenizer ({vocab_size:,} tokens)...")
    print("This may take 1-2 minutes.\n")
    
    tokenizer = Tokenizer(models.BPE())
    
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=["<|endoftext|>"],
        show_progress=True,
        min_frequency=2,
    )
    
    # Train on the dataset
    def text_iterator():
        for item in dataset:
            yield item['text']
    
    tokenizer.train_from_iterator(text_iterator(), trainer=trainer)
    
    # Add post-processing for special tokens (like GPT-2)
    tokenizer.post_processor = TemplateProcessing(
        single="$A <|endoftext|>",
        special_tokens=[("<|endoftext|>", tokenizer.token_to_id("<|endoftext|>"))],
    )
    
    # Wrap in HuggingFace PreTrainedTokenizerFast for compatibility
    wrapped_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        eos_token="<|endoftext|>",
        bos_token="<|endoftext|>",
        unk_token="<|endoftext|>",
        pad_token="<|endoftext|>",
    )
    
    print("Tokenizer training complete!\n")
    return wrapped_tokenizer


class Dataset:
    """Wrapper around HuggingFace dataset with analysis"""
    
    def __init__(self, hf_dataset, tokenizer):
        self.raw = hf_dataset
        self.tokenizer = tokenizer
        self.total_tokens = None
        self.context_length = None
        
    def analyze(self):
        """Analyze dataset and print stats"""
        if self.tokenizer is None:
            raise ValueError("Tokenizer not set yet. This will be set when you call kit.create_model()")
            
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
    
    def __init__(self):
        self.tokenizer_mode = "standard"  # Default
    
    def set_tokenizer(self, mode="standard"):
        """
        Set tokenizer size for future models.
        
        Args:
            mode: "standard" (50K vocab) or "small" (16K vocab, trained on your data)
        """
        valid_modes = ["standard", "small"]
        if mode not in valid_modes:
            raise ValueError(f"mode must be one of {valid_modes}")
        
        self.tokenizer_mode = mode
        
        print(f"\nTokenizer mode set to: '{mode}'")
        print("-" * 50)
        
        if mode == "standard":
            print("Vocabulary: 50,257 tokens (GPT-2)")
            print("Impact on 10M model: ~13M params in embeddings (56%)")
            print("Impact on 30M model: ~19M params in embeddings (47%)")
            print("Impact on 100M model: ~26M params in embeddings (26%)")
            print("\nBest for: General use, broad vocabulary needs")
            
        elif mode == "small":
            print("Vocabulary: 16,384 tokens (custom trained)")
            print("Impact on 10M model: ~4M params in embeddings (29%)")
            print("Impact on 30M model: ~6M params in embeddings (20%)")
            print("Impact on 100M model: ~8M params in embeddings (8%)")
            print("\nNote: Tokenizer will be trained on your dataset (1-2 min)")
            print("Best for: Smaller models, domain-specific text")
        
        print()
        return self
    
    def _get_tokenizer(self, dataset=None):
        """Internal method to get the appropriate tokenizer"""
        
        if self.tokenizer_mode == "standard":
            # Use pretrained GPT-2 tokenizer
            tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
            tokenizer.pad_token = tokenizer.eos_token
            vocab_size = 50257
            
        elif self.tokenizer_mode == "small":
            # Train custom tokenizer
            if dataset is None:
                raise ValueError(
                    "Dataset required for 'small' tokenizer mode. "
                    "Call kit.get_dataset() first."
                )
            
            tokenizer = _train_custom_tokenizer(dataset, vocab_size=16384)
            vocab_size = 16384
        
        return tokenizer, vocab_size
    
    def get_dataset(self, source, name, split):
        """Load dataset from HuggingFace"""
        if source != "hugging_face":
            raise ValueError("Only 'hugging_face' source supported for now")
        
        print(f"Loading dataset: {name} ({split})")
        
        try:
            raw_dataset = hf_load_dataset(name, split=split)
        except Exception as e:
            raise ValueError(f"Could not load dataset '{name}'. Error: {str(e)}")
        
        print(f"Loaded {len(raw_dataset):,} examples\n")
        
        return Dataset(raw_dataset, tokenizer=None)
    
    def create_model(self, dataset, model_size="30M"):
        """Create and configure model for training"""
        
        if model_size not in MODEL_PRESETS:
            raise ValueError(f"model_size must be one of: {list(MODEL_PRESETS.keys())}")
        
        # Get the appropriate tokenizer based on mode
        tokenizer, vocab_size = self._get_tokenizer(dataset.raw)
        
        # Assign tokenizer to dataset
        dataset.tokenizer = tokenizer
        
        # Analyze dataset if not done yet
        if dataset.context_length is None:
            dataset.analyze()
        
        # Get model config and use the selected vocab size
        config_dict = MODEL_PRESETS[model_size].copy()
        config_dict['vocab_size'] = vocab_size
        config_dict['n_positions'] = dataset.context_length
        
        config = GPT2Config(**config_dict)
        model = GPT2LMHeadModel(config)
        
        num_params = sum(p.numel() for p in model.parameters())
        vocab_params = vocab_size * config_dict['n_embd']
        model_params = num_params - (vocab_params * 2)
        
        print(f"Creating {model_size} model")
        print("-" * 50)
        print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"  Vocabulary: {vocab_params*2:,} ({vocab_params*2/1e6:.1f}M)")
        print(f"  Model layers: {model_params:,} ({model_params/1e6:.1f}M)")
        print(f"Architecture:")
        print(f"  Layers: {config.n_layer}")
        print(f"  Embedding dim: {config.n_embd}")
        print(f"  Attention heads: {config.n_head}")
        print(f"  Context length: {config.n_positions}")
        print(f"  Vocabulary size: {vocab_size:,}\n")
        
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
        
        return Model(model, tokenizer, trainer, model_size)
    
    def create_paper_model(self):
        """
        Create a tiny model with minimal dataset for testing your setup.
        """
        print("Creating tiny paper model for testing.")
        
        fake_data = {
            "text": [
                "The quick brown fox jumps over the lazy dog.",
                "Hello world, this is a test sentence.",
                "Machine learning is fascinating and fun."
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
        
        print("\nStarting training test...\n")
        
        trainer.train()
        
        print("Training completed. Setup works properly.")
        
        model.eval()
        
        class PaperModel:
            def __init__(self, model, tokenizer):
                self.model = model
                self.tokenizer = tokenizer
            
            def complete(self, prompt):
                """Generate text"""
                print("Testing text completion:")
                
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
                print("Generation works!")
                
                return result
            
            def save(self, path):
                """You don't need to save paper models"""
                print("Paper models are just for testing. No need to save.")
        
        return PaperModel(model, tokenizer)

kit = Kit()