import torch
from datasets import load_dataset as hf_load_dataset
from datasets import Dataset as HFDataset
from transformers import (
    GPT2Config, GPT2LMHeadModel, GPT2Tokenizer,
    Trainer, TrainingArguments, DataCollatorForLanguageModeling,
    AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
)
from .configs import MODEL_PRESETS, BATCH_CONFIGS, LORA_PRESETS, QUANTIZATION_CONFIGS, LORA_TARGET_MODULES
from .utils import detect_gpu, suggest_context_length, estimate_total_tokens, calculate_training_steps
from .lora_utils import (
    detect_lora_target_modules, estimate_lora_params,
    prepare_model_for_kbit_training, format_chat_template, get_memory_footprint
)

def _train_custom_tokenizer(dataset, vocab_size=16384):
    """
    Train a custom BPE tokenizer on the dataset.
    Uses the same algorithm as GPT-2 for compatibility.
    """
    from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
    from transformers import PreTrainedTokenizerFast
    
    print(f"Training custom BPE tokenizer ({vocab_size:,} tokens)...")
    print("This may take 1-2 minutes.\n")
    
    tokenizer = Tokenizer(models.BPE())

    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)

    # Add ByteLevel decoder to properly decode byte-level tokens back to text
    tokenizer.decoder = decoders.ByteLevel()
    
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

    # Note: Unlike the GPT-2 tokenizer, we don't add a post-processor that automatically
    # appends <|endoftext|> to every sequence. The DataCollatorForLanguageModeling
    # handles special token placement during training, and explicit control during inference
    # is better than automatic appending.
    
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


class InstructionDataset:
    """Dataset for instruction/chat fine-tuning with LoRA"""

    def __init__(self, data, format_type="conversational"):
        """
        Initialize instruction dataset.

        Args:
            data: List of examples or HuggingFace dataset
                  For conversational format: [{"messages": [{"role": "user", "content": "..."}, ...]}, ...]
                  For completion format: [{"prompt": "...", "completion": "..."}, ...]
            format_type: "conversational" or "completion"
        """
        if format_type not in ["conversational", "completion"]:
            raise ValueError("format_type must be 'conversational' or 'completion'")

        self.format_type = format_type

        # Convert to HuggingFace dataset if it's a list
        if isinstance(data, list):
            self.raw = HFDataset.from_list(data)
        else:
            self.raw = data

    def prepare_for_training(self, tokenizer, max_length=512):
        """
        Prepare dataset for LoRA training by applying chat template and tokenization.

        Args:
            tokenizer: The model's tokenizer
            max_length: Maximum sequence length

        Returns:
            Tokenized HuggingFace dataset ready for training
        """
        def tokenize_conversational(examples):
            """Tokenize conversational format using chat template"""
            texts = []
            for messages in examples["messages"]:
                text = format_chat_template(messages, tokenizer)
                texts.append(text)

            return tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,
            )

        def tokenize_completion(examples):
            """Tokenize completion format"""
            # For completion format, we concatenate prompt and completion
            # The DataCollatorForCompletionOnlyLM will handle masking the prompt
            texts = [
                f"{prompt}\n{completion}"
                for prompt, completion in zip(examples["prompt"], examples["completion"])
            ]

            return tokenizer(
                texts,
                truncation=True,
                max_length=max_length,
                padding=False,
            )

        print(f"Tokenizing instruction dataset ({self.format_type} format)...")

        if self.format_type == "conversational":
            tokenized = self.raw.map(
                tokenize_conversational,
                batched=True,
                remove_columns=self.raw.column_names,
                desc="Tokenizing"
            )
        else:  # completion
            tokenized = self.raw.map(
                tokenize_completion,
                batched=True,
                remove_columns=self.raw.column_names,
                desc="Tokenizing"
            )

        print(f"✓ Tokenized {len(tokenized):,} examples")

        # Show sample of what the data looks like
        if len(tokenized) > 0:
            sample_text = tokenizer.decode(tokenized[0]['input_ids'][:200])
            print(f"\nSample tokenized text (first 200 tokens):")
            print(f"---")
            print(sample_text)
            print(f"---\n")

        return tokenized


class Dataset:
    """Wrapper around HuggingFace dataset with analysis (for pretraining)"""

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


class HFModel:
    """HuggingFace model wrapper with LoRA support"""

    def __init__(self, model, tokenizer, model_name, is_quantized=False):
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.is_quantized = is_quantized
        self.trainer = None
        self.is_lora_enabled = False

    def train(self):
        """Train the LoRA model"""
        if self.trainer is None:
            raise ValueError("No trainer configured. Call kit.lora_tune() first.")

        print("=" * 70)
        print("STARTING LORA FINE-TUNING")
        print("=" * 70)

        result = self.trainer.train()

        print("\n" + "=" * 70)
        print("LORA FINE-TUNING COMPLETE!")
        print("=" * 70)
        print(f"Final loss: {result.training_loss:.4f}\n")

        self.model.eval()
        return self

    def complete(self, prompt, max_new_tokens=100, temperature=0.8, stop_at_eos=True, system_prompt=None):
        """Generate text completion

        Args:
            prompt: Input text to complete (plain text or conversation messages)
            max_new_tokens: Maximum number of NEW tokens to generate (not including prompt)
            temperature: Sampling temperature (higher = more random)
            stop_at_eos: If True, stop generation at EOS token (recommended for chat models)
            system_prompt: Optional system prompt (for instruction-tuned models). If not provided,
                          uses the system prompt from training if available.
        """
        # Auto-detect if this is a chat model and format accordingly
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Format as chat if we have a chat template
            messages = []

            # Add system prompt if available
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            elif hasattr(self, '_training_system_prompt') and self._training_system_prompt:
                messages.append({"role": "system", "content": self._training_system_prompt})

            # Add user message
            messages.append({"role": "user", "content": prompt})

            # Format using chat template and add generation prompt
            formatted_prompt = self.tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )

            inputs = self.tokenizer(formatted_prompt, return_tensors="pt")
        else:
            # Plain text completion
            inputs = self.tokenizer(prompt, return_tensors="pt")

        if torch.cuda.is_available():
            inputs = {k: v.cuda() for k, v in inputs.items()}

        # Build generation kwargs
        gen_kwargs = {
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "do_sample": True,
            "top_k": 50,
            "top_p": 0.95,
            "repetition_penalty": 1.2,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Add stopping criteria for chat models
        if stop_at_eos and hasattr(self.tokenizer, 'eos_token_id'):
            # Stop at EOS token (like <|im_end|>)
            gen_kwargs["eos_token_id"] = self.tokenizer.eos_token_id

        with torch.no_grad():
            outputs = self.model.generate(**inputs, **gen_kwargs)

        result = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        # If we formatted as chat, extract just the assistant response
        if hasattr(self.tokenizer, 'chat_template') and self.tokenizer.chat_template:
            # Remove the prompt part to return only the generated response
            result = result[len(formatted_prompt):].strip()

        return result

    def save_adapter(self, path):
        """Save only the LoRA adapter weights (lightweight)"""
        if not self.is_lora_enabled:
            raise ValueError("No LoRA adapter to save. Call kit.lora_tune() first.")

        self.model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✓ LoRA adapter saved to {path}")
        print(f"  (Adapter is small, typically 10-100MB)")

    def save_merged(self, path):
        """Merge LoRA weights with base model and save (larger file)"""
        if not self.is_lora_enabled:
            raise ValueError("No LoRA adapter to merge. Call kit.lora_tune() first.")

        print("Merging LoRA weights with base model...")
        merged_model = self.model.merge_and_unload()

        merged_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)
        print(f"✓ Merged model saved to {path}")
        print(f"  (Full model, can be loaded without LoRA)")

    def load_adapter(self, adapter_path):
        """Load a LoRA adapter onto this model"""
        from peft import PeftModel

        print(f"Loading LoRA adapter from {adapter_path}...")
        self.model = PeftModel.from_pretrained(self.model, adapter_path)
        self.is_lora_enabled = True
        print("✓ LoRA adapter loaded successfully\n")


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
    
    def create_model(self, dataset, model_size="30M", steps=0):
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
        steps = steps or calculate_training_steps(
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
    
    def load_model(self, path, model_size=None):
        """
        Load a pre-trained model from disk or cloud storage.

        Args:
            path: Path to the saved model directory (local path) or URL (cloud storage).
                  Supported URL formats:
                  - HTTP/HTTPS: https://example.com/model
                  - S3: s3://bucket/path/to/model
                  - GCS: gs://bucket/path/to/model
                  - Azure: https://account.blob.core.windows.net/container/path
            model_size: Optional model size ("10M", "30M", or "100M").
                       If not provided, will be inferred from config.json

        Returns:
            Model instance ready for inference with complete() method
        """
        import os
        import re
        from transformers import AutoTokenizer

        # Determine if path is a URL or local path using regex
        url_pattern = re.compile(
            r'^(https?://|s3://|gs://|[a-zA-Z0-9]+://)',
            re.IGNORECASE
        )
        is_url = bool(url_pattern.match(path))

        if is_url:
            print(f"Loading model from URL: {path}")
            print("(Downloading from cloud storage...)")
        else:
            print(f"Loading model from disk: {path}")
            # Validate local path exists
            if not os.path.exists(path):
                raise ValueError(f"Model path does not exist: {path}")

        print("-" * 50)

        try:
            model = GPT2LMHeadModel.from_pretrained(path)
            print("✓ Model loaded successfully")
        except Exception as e:
            if is_url:
                raise ValueError(f"Failed to load model from URL {path}: {str(e)}\n"
                               f"Ensure the URL is accessible and contains valid model files.")
            else:
                raise ValueError(f"Failed to load model from {path}: {str(e)}")

        # Load the tokenizer
        try:
            tokenizer = AutoTokenizer.from_pretrained(path)
            # Ensure pad_token is set (should be saved in config, but set as fallback)
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            if is_url:
                raise ValueError(f"Failed to load tokenizer from URL {path}: {str(e)}")
            else:
                raise ValueError(f"Failed to load tokenizer from {path}: {str(e)}")

        # Infer model size if not provided
        if model_size is None:
            # Try to infer from config
            config = model.config
            n_layer = config.n_layer
            n_embd = config.n_embd

            # Match against known presets
            for size, preset in MODEL_PRESETS.items():
                if preset['n_layer'] == n_layer and preset['n_embd'] == n_embd:
                    model_size = size
                    break

            if model_size is None:
                model_size = "custom"

        # Get model info
        num_params = sum(p.numel() for p in model.parameters())

        print(f"\nModel Information")
        print("-" * 50)
        print(f"Model size: {model_size}")
        print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"Architecture:")
        print(f"  Layers: {model.config.n_layer}")
        print(f"  Embedding dim: {model.config.n_embd}")
        print(f"  Attention heads: {model.config.n_head}")
        print(f"  Context length: {model.config.n_positions}")
        print(f"  Vocabulary size: {model.config.vocab_size:,}")

        # Move to GPU if available
        if torch.cuda.is_available():
            model = model.cuda()
            print(f"\n✓ Model moved to GPU: {torch.cuda.get_device_name(0)}")
        else:
            print("\n⚠ No GPU detected - inference will be slower")

        # Set to evaluation mode
        model.eval()

        print("\n✓ Model ready for inference\n")

        # Return Model instance (trainer=None since we're not training)
        return Model(model, tokenizer, trainer=None, model_size=model_size)

    def load_hf_model(self, model_name, quantization=None, device_map="auto", trust_remote_code=False):
        """
        Load any HuggingFace model for LoRA fine-tuning.

        Args:
            model_name: HuggingFace model identifier (e.g., "meta-llama/Llama-3.2-3B-Instruct",
                       "deepseek-ai/DeepSeek-Coder-V2-Instruct", "gpt2")
            quantization: Quantization mode - None, "4bit", or "8bit"
                         Quantization reduces memory usage (e.g., 3B model in ~3GB instead of ~12GB)
            device_map: Device mapping strategy - "auto" (recommended), "cuda:0", or custom mapping
            trust_remote_code: Whether to trust remote code (needed for some models)

        Returns:
            HFModel instance ready for LoRA fine-tuning

        Examples:
            # Load Llama model with 4-bit quantization
            model = kit.load_hf_model("meta-llama/Llama-3.2-3B-Instruct", quantization="4bit")

            # Load DeepSeek model
            model = kit.load_hf_model("deepseek-ai/DeepSeek-Coder-V2-Instruct", quantization="4bit")

            # Load without quantization (requires more memory)
            model = kit.load_hf_model("gpt2")
        """
        print(f"Loading HuggingFace model: {model_name}")
        print("=" * 70)

        # Setup quantization config if requested
        quant_config = None
        is_quantized = False

        if quantization is not None:
            if quantization not in QUANTIZATION_CONFIGS:
                raise ValueError(f"quantization must be None, '4bit', or '8bit', got '{quantization}'")

            is_quantized = True
            quant_dict = QUANTIZATION_CONFIGS[quantization].copy()

            # Convert dtype string to torch dtype
            if "bnb_4bit_compute_dtype" in quant_dict:
                compute_dtype = quant_dict["bnb_4bit_compute_dtype"]
                if compute_dtype == "float16":
                    quant_dict["bnb_4bit_compute_dtype"] = torch.float16
                elif compute_dtype == "bfloat16":
                    quant_dict["bnb_4bit_compute_dtype"] = torch.bfloat16

            quant_config = BitsAndBytesConfig(**quant_dict)
            print(f"Quantization: {quantization}")
        else:
            print("Quantization: None (full precision)")

        print(f"Device mapping: {device_map}")
        print()

        # Load tokenizer
        try:
            print("Loading tokenizer...")
            tokenizer = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=trust_remote_code
            )

            # Ensure pad_token is set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token

            print("✓ Tokenizer loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load tokenizer for '{model_name}': {str(e)}")

        # Load model
        try:
            print("Loading model (this may take a few minutes)...")

            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                quantization_config=quant_config,
                device_map=device_map,
                trust_remote_code=trust_remote_code,
                torch_dtype=torch.float16 if not is_quantized else None,
            )

            print("✓ Model loaded successfully")
        except Exception as e:
            raise ValueError(f"Failed to load model '{model_name}': {str(e)}\n"
                           f"Make sure the model name is correct and you have access to it.")

        # Get model information
        num_params = sum(p.numel() for p in model.parameters())
        memory_stats = get_memory_footprint(model)

        print(f"\nModel Information")
        print("-" * 70)
        print(f"Model: {model_name}")
        print(f"Architecture: {model.config.model_type}")
        print(f"Total parameters: {num_params:,} ({num_params/1e6:.1f}M)")
        print(f"Hidden size: {model.config.hidden_size}")
        print(f"Num layers: {model.config.num_hidden_layers}")
        print(f"Vocab size: {model.config.vocab_size:,}")
        print(f"\nMemory footprint: ~{memory_stats['model_size_gb']:.2f} GB")
        print(f"Quantized: {memory_stats['quantized']}")

        # Set model to training mode for LoRA
        model.train()

        print("\n✓ Model ready for LoRA fine-tuning")
        print("  Next step: Use kit.lora_tune(model, dataset) to configure LoRA\n")

        return HFModel(
            model=model,
            tokenizer=tokenizer,
            model_name=model_name,
            is_quantized=is_quantized
        )

    def lora_tune(self, hf_model, dataset, lora_preset="medium", learning_rate=2e-4,
                  steps=1000, batch_size=4, gradient_accumulation_steps=4,
                  max_length=512, output_dir="./lora_output", response_template=None,
                  use_simple_chat_template=True):
        """
        Configure LoRA fine-tuning for a HuggingFace model.

        Args:
            hf_model: HFModel instance from load_hf_model()
            dataset: InstructionDataset instance with training data
            lora_preset: LoRA configuration preset - "small" (r=8), "medium" (r=16), or "large" (r=32)
            learning_rate: Learning rate for training (2e-4 is good default for LoRA)
            steps: Number of training steps
            batch_size: Batch size per device
            gradient_accumulation_steps: Gradient accumulation steps
            max_length: Maximum sequence length for training
            output_dir: Directory to save checkpoints and final model

        Returns:
            HFModel instance with LoRA configured and ready for training

        Example:
            model = kit.load_hf_model("meta-llama/Llama-3.2-3B-Instruct", quantization="4bit")
            dataset = kit.create_instruction_dataset([...])
            tuned_model = kit.lora_tune(model, dataset, steps=1000)
            tuned_model.train()
        """
        from peft import LoraConfig, get_peft_model, TaskType

        if not isinstance(hf_model, HFModel):
            raise ValueError("hf_model must be an HFModel instance from load_hf_model()")

        if not isinstance(dataset, InstructionDataset):
            raise ValueError("dataset must be an InstructionDataset instance from create_instruction_dataset()")

        print("=" * 70)
        print("CONFIGURING LORA FINE-TUNING")
        print("=" * 70)

        # Get LoRA configuration
        if lora_preset not in LORA_PRESETS:
            raise ValueError(f"lora_preset must be one of {list(LORA_PRESETS.keys())}, got '{lora_preset}'")

        lora_config_dict = LORA_PRESETS[lora_preset].copy()

        # Auto-detect target modules if not specified
        if lora_config_dict["target_modules"] is None:
            target_modules = detect_lora_target_modules(hf_model.model)
            lora_config_dict["target_modules"] = target_modules
        else:
            target_modules = lora_config_dict["target_modules"]

        print(f"\nLoRA Configuration")
        print("-" * 70)
        print(f"Preset: {lora_preset}")
        print(f"Rank (r): {lora_config_dict['r']}")
        print(f"Alpha: {lora_config_dict['lora_alpha']}")
        print(f"Dropout: {lora_config_dict['lora_dropout']}")
        print(f"Target modules: {target_modules}")

        # Estimate trainable parameters
        lora_stats = estimate_lora_params(
            hf_model.model,
            lora_config_dict['r'],
            target_modules
        )

        print(f"\nParameter Statistics")
        print("-" * 70)
        print(f"Total model parameters: {lora_stats['total_params']:,}")
        print(f"LoRA trainable parameters: ~{lora_stats['lora_params']:,}")
        print(f"Trainable: ~{lora_stats['trainable_percentage']:.2f}%")

        # Create LoRA config
        lora_config = LoraConfig(
            r=lora_config_dict['r'],
            lora_alpha=lora_config_dict['lora_alpha'],
            lora_dropout=lora_config_dict['lora_dropout'],
            target_modules=target_modules,
            bias=lora_config_dict['bias'],
            task_type=TaskType.CAUSAL_LM,
        )

        # Prepare model for training if quantized
        if hf_model.is_quantized:
            print("\nPreparing quantized model for training...")
            hf_model.model = prepare_model_for_kbit_training(hf_model.model)

        # Apply LoRA to model
        print("Applying LoRA to model...")
        hf_model.model = get_peft_model(hf_model.model, lora_config)
        hf_model.is_lora_enabled = True

        # Print trainable parameters
        trainable_params = sum(p.numel() for p in hf_model.model.parameters() if p.requires_grad)
        all_params = sum(p.numel() for p in hf_model.model.parameters())

        print(f"\n✓ LoRA applied successfully")
        print(f"Actual trainable parameters: {trainable_params:,} ({100 * trainable_params / all_params:.2f}%)")

        # Override chat template if requested
        if use_simple_chat_template:
            print(f"\n✓ Using simplified chat template (removes bloated system prompts)")
            # Store original template
            original_template = hf_model.tokenizer.chat_template

            # Set a simple template that doesn't include the massive default system prompt
            simple_template = (
                "{% for message in messages %}"
                "{% if message['role'] == 'system' %}"
                "<|im_start|>system\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'user' %}"
                "<|im_start|>user\n{{ message['content'] }}<|im_end|>\n"
                "{% elif message['role'] == 'assistant' %}"
                "<|im_start|>assistant\n{{ message['content'] }}<|im_end|>\n"
                "{% endif %}"
                "{% endfor %}"
            )
            hf_model.tokenizer.chat_template = simple_template
            print(f"  Original template had default system prompt - now using clean template\n")

        # Prepare dataset
        print(f"Preparing dataset...")
        tokenized_dataset = dataset.prepare_for_training(hf_model.tokenizer, max_length=max_length)

        # Setup training arguments
        print(f"\nTraining Configuration")
        print("-" * 70)
        print(f"Steps: {steps:,}")
        print(f"Batch size: {batch_size}")
        print(f"Gradient accumulation: {gradient_accumulation_steps}")
        print(f"Effective batch size: {batch_size * gradient_accumulation_steps}")
        print(f"Learning rate: {learning_rate}")
        print(f"Max sequence length: {max_length}")
        print(f"Output directory: {output_dir}")

        training_args = TrainingArguments(
            output_dir=output_dir,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            max_steps=steps,
            learning_rate=learning_rate,
            fp16=True if not hf_model.is_quantized else False,
            bf16=False,
            logging_steps=50,
            save_steps=max(steps // 3, 100),
            save_total_limit=2,
            optim="paged_adamw_8bit" if hf_model.is_quantized else "adamw_torch",
            warmup_steps=min(100, steps // 10),
            lr_scheduler_type="cosine",
            gradient_checkpointing=True,
            report_to="none",
            disable_tqdm=False,
        )

        # Data collator - use response masking for instruction tuning
        # This ensures we only train on the assistant's responses, not the prompts

        # Implement our own response masking data collator
        class ResponseMaskingDataCollator:
            """Data collator that masks prompt tokens, only training on responses"""

            def __init__(self, tokenizer, response_template, mlm=False):
                self.tokenizer = tokenizer
                self.response_template = response_template
                self.mlm = mlm

                # Encode the response template to find it in sequences
                self.response_template_ids = tokenizer.encode(
                    response_template,
                    add_special_tokens=False
                )

            def __call__(self, features):
                # Pad sequences to the same length within the batch
                import torch

                # Find max length in this batch
                max_length = max(len(f["input_ids"]) for f in features)

                # Pad each feature to max_length
                padded_input_ids = []
                padded_attention_masks = []

                for f in features:
                    # Pad input_ids
                    padding_length = max_length - len(f["input_ids"])
                    padded_input_ids.append(f["input_ids"] + [self.tokenizer.pad_token_id] * padding_length)

                    # Pad attention_mask (if it exists)
                    if "attention_mask" in f:
                        padded_attention_masks.append(f["attention_mask"] + [0] * padding_length)
                    else:
                        padded_attention_masks.append([1] * len(f["input_ids"]) + [0] * padding_length)

                # Convert to tensors
                batch = {
                    "input_ids": torch.tensor(padded_input_ids),
                    "attention_mask": torch.tensor(padded_attention_masks)
                }

                # Create labels with -100 for prompt tokens (ignored in loss)
                labels = batch["input_ids"].clone()

                # Diagnostic counters
                templates_found = 0
                templates_not_found = 0
                total_tokens = 0
                masked_tokens = 0

                # For each sequence in the batch
                for idx in range(labels.shape[0]):
                    sequence = batch["input_ids"][idx].tolist()

                    # Find where the response template starts
                    response_start = None
                    for i in range(len(sequence) - len(self.response_template_ids) + 1):
                        if sequence[i:i+len(self.response_template_ids)] == self.response_template_ids:
                            # Response starts after the template
                            response_start = i + len(self.response_template_ids)
                            break

                    # Mask everything before the response
                    if response_start is not None:
                        labels[idx, :response_start] = -100
                        templates_found += 1
                        masked_tokens += response_start
                    else:
                        # If template not found, mask everything (fallback)
                        labels[idx, :] = -100
                        templates_not_found += 1
                        masked_tokens += len(sequence)

                    # Also mask padding tokens (where input_ids == pad_token_id)
                    padding_mask = batch["input_ids"][idx] == self.tokenizer.pad_token_id
                    labels[idx][padding_mask] = -100

                    total_tokens += len(sequence)

                # Log diagnostics periodically (every 100 batches)
                if not hasattr(self, '_batch_count'):
                    self._batch_count = 0
                    self._show_sample = True

                self._batch_count += 1

                if self._batch_count % 100 == 0 or self._show_sample:
                    mask_pct = (masked_tokens / total_tokens * 100) if total_tokens > 0 else 0
                    print(f"\n[Masking Debug - Batch {self._batch_count}]")
                    print(f"  Templates found: {templates_found}/{len(features)}")
                    print(f"  Templates NOT found: {templates_not_found}/{len(features)}")
                    print(f"  Tokens masked: {masked_tokens}/{total_tokens} ({mask_pct:.1f}%)")

                    # Show a sample on first batch
                    if self._show_sample and len(features) > 0:
                        print(f"\n  Sample from first sequence:")
                        sample_ids = batch["input_ids"][0].tolist()
                        sample_labels = labels[0].tolist()

                        # Decode and show what's masked vs unmasked
                        masked_part = [token_id for token_id, label in zip(sample_ids, sample_labels) if label == -100]
                        unmasked_part = [token_id for token_id, label in zip(sample_ids, sample_labels) if label != -100]

                        print(f"  MASKED (not trained): {self.tokenizer.decode(masked_part)[:200]}")
                        print(f"  UNMASKED (trained): {self.tokenizer.decode(unmasked_part)[:200]}")
                        self._show_sample = False

                batch["labels"] = labels
                return batch

        # For conversational format, we need to identify where responses start
        # Most chat templates use specific tokens or patterns
        detected_template = None

        # Manual override takes precedence
        if response_template:
            detected_template = response_template
            print(f"✓ Using manual response template: '{response_template}'")
        else:
            # Try to detect the response template based on tokenizer chat template
            if hasattr(hf_model.tokenizer, 'chat_template') and hf_model.tokenizer.chat_template:
                # Common patterns in chat templates
                if 'assistant' in hf_model.tokenizer.chat_template.lower():
                    # Try common assistant markers
                    for marker in ['<|assistant|>', 'assistant\n', '[/INST]', '<|im_start|>assistant', '<|start_header_id|>assistant<|end_header_id|>']:
                        test_messages = [
                            {"role": "user", "content": "test"},
                            {"role": "assistant", "content": "response"}
                        ]
                        formatted = hf_model.tokenizer.apply_chat_template(
                            test_messages, tokenize=False, add_generation_prompt=False
                        )
                        if marker in formatted:
                            detected_template = marker
                            print(f"✓ Auto-detected response template: '{marker}'")
                            break

            # Fallback: check if using our simple fallback format
            if not detected_template:
                # Test if the tokenizer uses our fallback format
                test_messages = [
                    {"role": "user", "content": "test"},
                    {"role": "assistant", "content": "response"}
                ]
                try:
                    formatted = hf_model.tokenizer.apply_chat_template(
                        test_messages, tokenize=False, add_generation_prompt=False
                    )
                except:
                    # No chat template - using our fallback in format_chat_template
                    formatted = "User: test\n\nAssistant: response"

                # Check for our fallback format
                if "Assistant:" in formatted:
                    detected_template = "Assistant:"
                    print(f"✓ Using fallback response template: 'Assistant:'")

        if detected_template:
            print(f"  Response masking enabled - only training on assistant responses")

            # Diagnostic: Show what the template looks like in the formatted text
            print(f"\n{'='*70}")
            print("DIAGNOSTIC: Response Template Detection")
            print(f"{'='*70}")

            # Create a sample formatted message to show the template
            sample_messages = [
                {"role": "user", "content": "What is programming?"},
                {"role": "assistant", "content": "It be the art of instructin' machines."}
            ]

            from .lora_utils import format_chat_template
            sample_formatted = format_chat_template(sample_messages, hf_model.tokenizer)
            print(f"Sample formatted conversation:")
            print(f"---")
            print(sample_formatted)
            print(f"---")
            print(f"\nResponse template marker: '{detected_template}'")
            print(f"Template found in sample: {detected_template in sample_formatted}")

            # Show token IDs for the template
            template_tokens = hf_model.tokenizer.encode(detected_template, add_special_tokens=False)
            print(f"Template token IDs: {template_tokens}")

            # Tokenize a sample and show which tokens would be masked
            sample_tokenized = hf_model.tokenizer(sample_formatted, return_tensors="pt")
            print(f"\nSample sequence length: {sample_tokenized['input_ids'].shape[1]} tokens")

            # Try to find where the template appears in the tokenized sequence
            full_ids = sample_tokenized['input_ids'][0].tolist()
            template_positions = []
            for i in range(len(full_ids) - len(template_tokens) + 1):
                if full_ids[i:i+len(template_tokens)] == template_tokens:
                    template_positions.append(i)

            if template_positions:
                print(f"Template found at token positions: {template_positions}")
                print(f"This means tokens {template_positions[0]}+ will be trained on (assistant response)")
            else:
                print(f"⚠ WARNING: Template NOT found in tokenized sequence!")
                print(f"  This means response masking will NOT work correctly")
                print(f"  Full token IDs: {full_ids[:50]}... (showing first 50)")

            print(f"{'='*70}\n")

            data_collator = ResponseMaskingDataCollator(
                tokenizer=hf_model.tokenizer,
                response_template=detected_template,
                mlm=False
            )
        else:
            print("⚠ Could not detect response template, training on full sequence")
            print("  WARNING: This will train on both prompts and responses")
            print("  Consider specifying response_template parameter manually")
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=hf_model.tokenizer,
                mlm=False
            )

        # Create trainer
        trainer = Trainer(
            model=hf_model.model,
            args=training_args,
            train_dataset=tokenized_dataset,
            data_collator=data_collator,
        )

        # Assign trainer to model
        hf_model.trainer = trainer

        # Try to extract system prompt from the dataset for use during inference
        hf_model._training_system_prompt = None
        if hasattr(dataset, 'raw') and len(dataset.raw) > 0:
            first_example = dataset.raw[0]
            if 'messages' in first_example:
                # Conversational format - look for system message
                for msg in first_example['messages']:
                    if msg.get('role') == 'system':
                        hf_model._training_system_prompt = msg.get('content')
                        break

        print("\n✓ LoRA fine-tuning configured successfully")
        print("  Next step: Call model.train() to start fine-tuning\n")

        return hf_model

    def encode_twoway_to_lora(self, text_or_file, input_marker="Q:", output_marker="A:",
                               system_prompt=None, format_type="conversational"):
        """
        Encode plain text question-answer pairs into LoRA training format.

        This method makes it easy to create datasets from simple text files where
        input and output are marked with identifiers like "Q:" and "A:".

        Args:
            text_or_file: Either a string containing the text, or a file path to read from
            input_marker: Marker that identifies input/questions (default "Q:")
            output_marker: Marker that identifies output/answers (default "A:")
            system_prompt: Optional system prompt to add to each example (for conversational format)
            format_type: "conversational" (default) or "completion"

        Returns:
            List of formatted examples ready for create_instruction_dataset()

        Examples:
            # From text string
            text = '''
            Q: What is programming?
            A: Hark! 'Tis the art of instructing mechanical minds.

            Q: How do I learn Python?
            A: Marry, good scholar! Begin with variables.
            '''
            data = kit.encode_twoway_to_lora(text, system_prompt="You are Shakespeare")

            # From file
            data = kit.encode_twoway_to_lora("data.txt", input_marker="Q:", output_marker="A:")

            # Then create dataset
            dataset = kit.create_instruction_dataset(data)
        """
        import os

        # Read from file if path is provided
        if isinstance(text_or_file, str) and os.path.isfile(text_or_file):
            print(f"Reading from file: {text_or_file}")
            with open(text_or_file, 'r', encoding='utf-8') as f:
                text = f.read()
        else:
            text = text_or_file

        # Split text into lines and parse pairs
        lines = text.strip().split('\n')
        pairs = []
        current_input = None
        current_output = None

        for line in lines:
            line = line.strip()
            if not line:
                # Empty line - if we have a complete pair, save it
                if current_input is not None and current_output is not None:
                    pairs.append((current_input, current_output))
                    current_input = None
                    current_output = None
                continue

            # Check if line contains both markers on the same line
            if input_marker in line and output_marker in line:
                # Save previous pair if exists
                if current_input is not None and current_output is not None:
                    pairs.append((current_input, current_output))

                # Split by output marker to separate input and output
                input_part, output_part = line.split(output_marker, 1)

                # Extract input (remove input marker)
                if input_part.startswith(input_marker):
                    current_input = input_part[len(input_marker):].strip()
                else:
                    current_input = input_part.strip()

                # Extract output
                current_output = output_part.strip()

                # Save this pair immediately
                pairs.append((current_input, current_output))
                current_input = None
                current_output = None

            # Check if line starts with input marker
            elif line.startswith(input_marker):
                # Save previous pair if exists
                if current_input is not None and current_output is not None:
                    pairs.append((current_input, current_output))
                # Start new input
                current_input = line[len(input_marker):].strip()
                current_output = None

            # Check if line starts with output marker
            elif line.startswith(output_marker):
                current_output = line[len(output_marker):].strip()

            # Continuation of previous line
            else:
                if current_output is not None:
                    # Continue the output
                    current_output += " " + line
                elif current_input is not None:
                    # Continue the input
                    current_input += " " + line

        # Don't forget the last pair
        if current_input is not None and current_output is not None:
            pairs.append((current_input, current_output))

        if not pairs:
            raise ValueError(
                f"No pairs found! Make sure your text contains '{input_marker}' and '{output_marker}' markers.\n"
                f"Example format:\n"
                f"{input_marker} Your question here\n"
                f"{output_marker} Your answer here\n"
            )

        # Convert to appropriate format
        formatted_data = []

        if format_type == "conversational":
            for inp, out in pairs:
                example = {
                    "messages": []
                }

                # Add system prompt if provided
                if system_prompt:
                    example["messages"].append({
                        "role": "system",
                        "content": system_prompt
                    })

                # Add user input and assistant output
                example["messages"].append({
                    "role": "user",
                    "content": inp
                })
                example["messages"].append({
                    "role": "assistant",
                    "content": out
                })

                formatted_data.append(example)

        elif format_type == "completion":
            for inp, out in pairs:
                formatted_data.append({
                    "prompt": inp,
                    "completion": out
                })
        else:
            raise ValueError(f"format_type must be 'conversational' or 'completion', got '{format_type}'")

        print(f"✓ Encoded {len(formatted_data):,} question-answer pairs")
        print(f"  Input marker: '{input_marker}'")
        print(f"  Output marker: '{output_marker}'")
        print(f"  Format: {format_type}")
        if system_prompt:
            print(f"  System prompt: '{system_prompt}'")
        print()

        return formatted_data

    def create_instruction_dataset(self, data, format_type="conversational"):
        """
        Create an instruction dataset for LoRA fine-tuning.

        Args:
            data: List of training examples
                  For conversational format:
                    [{"messages": [{"role": "user", "content": "..."},
                                   {"role": "assistant", "content": "..."}]}, ...]
                  For completion format:
                    [{"prompt": "...", "completion": "..."}, ...]
            format_type: "conversational" (default) or "completion"

        Returns:
            InstructionDataset instance ready for lora_tune()

        Examples:
            # Conversational format (for chat models)
            dataset = kit.create_instruction_dataset([
                {
                    "messages": [
                        {"role": "system", "content": "You are Shakespeare"},
                        {"role": "user", "content": "What is programming?"},
                        {"role": "assistant", "content": "Hark! 'Tis the art of..."}
                    ]
                },
                # ... more examples
            ])

            # Completion format (simpler)
            dataset = kit.create_instruction_dataset([
                {"prompt": "Translate to Shakespeare: Hello",
                 "completion": "Good morrow to thee!"},
                # ... more examples
            ], format_type="completion")
        """
        print(f"Creating instruction dataset ({format_type} format)")
        print(f"Examples: {len(data):,}\n")

        return InstructionDataset(data, format_type=format_type)

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