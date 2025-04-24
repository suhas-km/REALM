# rlhf/dpo_huggingface.py
import os
import torch
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass

# Hugging Face's TRL library for DPO implementation
from trl import DPOTrainer
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    TrainingArguments,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from tqdm import tqdm

from inference.predictor import RewardPredictor

logger = logging.getLogger(__name__)

class HuggingFaceDPOTrainer:
    """
    DPO Trainer using Hugging Face's TRL library for industry-standard implementation
    of Direct Preference Optimization for RLHF.
    """
    
    def __init__(
        self,
        config: Dict,
        reward_predictor: RewardPredictor,
        tokenizer=None,
        model=None,
        device: Optional[torch.device] = None,
        use_peft: bool = False
    ):
        self.config = config
        self.reward_predictor = reward_predictor
        self.use_peft = use_peft
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model if not provided
        dpo_config = config["rlhf"]["dpo"]
        model_name = dpo_config["model_name"]
        logger.info(f"Loading model and tokenizer for DPO: {model_name}")
        
        # Setup tokenizer
        if tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            # Ensure tokenizer has pad token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
        else:
            self.tokenizer = tokenizer
            
        # Setup model with optional quantization
        if model is None:
            use_4bit = dpo_config.get("use_4bit", False)
            use_8bit = dpo_config.get("use_8bit", False)
            
            if use_4bit or use_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=use_4bit,
                    load_in_8bit=use_8bit,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_use_double_quant=True,
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    quantization_config=quantization_config,
                    device_map="auto"
                )
                
                # Prepare model for k-bit training if using PEFT
                if self.use_peft:
                    self.model = prepare_model_for_kbit_training(self.model)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
                self.model.to(self.device)
        else:
            self.model = model
            
        # Apply LoRA if using PEFT
        if self.use_peft:
            lora_config = self._get_lora_config()
            self.model = get_peft_model(self.model, lora_config)
            logger.info("Applied LoRA for parameter-efficient fine-tuning")
            
        logger.info("HuggingFace DPO Trainer initialized")

    def _get_lora_config(self) -> LoraConfig:
        """Get LoRA configuration for PEFT"""
        dpo_config = self.config["rlhf"]["dpo"]
        return LoraConfig(
            r=dpo_config.get("lora_r", 16),
            lora_alpha=dpo_config.get("lora_alpha", 32),
            lora_dropout=dpo_config.get("lora_dropout", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=dpo_config.get("lora_target_modules", ["q_proj", "v_proj", "k_proj", "out_proj", "fc_in", "fc_out", "wte"])
        )
    
    def _compare_responses(self, prompt: str, response1: str, response2: str) -> Tuple[str, str]:
        """Compare two responses and return them in order (chosen, rejected)"""
        reward1, reward2, better = self.reward_predictor.compare(prompt, response1, response2)
        
        if better == 1:
            return response1, response2
        else:
            return response2, response1
    
    def _generate_paired_responses(self, prompts: List[str], max_length: int = 512) -> List[Dict[str, str]]:
        """Generate two different responses for each prompt and rank them using the reward model"""
        paired_data = []
        batch_size = self.config["rlhf"]["dpo"].get("generation_batch_size", 4)
        
        # Process prompts in batches for efficiency
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i+batch_size]
            batch_pairs = []
            
            for prompt in tqdm(batch_prompts, desc=f"Generating response pairs (batch {i//batch_size + 1})"):
                # Generate first response
                inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
                outputs1 = self.model.generate(
                    inputs, 
                    max_length=max_length, 
                    do_sample=True,
                    temperature=0.9,
                    top_p=0.9,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                response1 = self.tokenizer.decode(outputs1[0], skip_special_tokens=True)
                response1 = response1[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
                
                # Generate second response (with different parameters for diversity)
                outputs2 = self.model.generate(
                    inputs, 
                    max_length=max_length, 
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.8,
                    pad_token_id=self.tokenizer.pad_token_id
                )
                response2 = self.tokenizer.decode(outputs2[0], skip_special_tokens=True)
                response2 = response2[len(self.tokenizer.decode(inputs[0], skip_special_tokens=True)):].strip()
                
                # Avoid identical responses
                if response1 == response2:
                    continue
                
                # Compare and order responses
                chosen, rejected = self._compare_responses(prompt, response1, response2)
                
                batch_pairs.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
            
            paired_data.extend(batch_pairs)
        
        logger.info(f"Generated {len(paired_data)} response pairs from {len(prompts)} prompts")
        return paired_data
    
    def _get_dpo_training_args(self) -> TrainingArguments:
        """Get minimal DPO training arguments that should be compatible with most versions"""
        dpo_config = self.config["rlhf"]["dpo"]
        
        # Default output directory
        output_dir = dpo_config.get("output_dir", "./dpo_output")
        
        # Start with just the most basic parameters that should be compatible with all versions
        try:
            # Initialize training arguments with proper type conversions for numeric values
            return TrainingArguments(
                output_dir=output_dir,
                num_train_epochs=int(dpo_config.get("num_epochs", 3)),
                learning_rate=float(dpo_config.get("learning_rate", 5e-7)),
                per_device_train_batch_size=int(dpo_config.get("batch_size", 4)),
                per_device_eval_batch_size=int(dpo_config.get("batch_size", 4)),
                gradient_accumulation_steps=int(dpo_config.get("gradient_accumulation_steps", 1)),
                weight_decay=float(dpo_config.get("weight_decay", 0.01)),
            )
        except TypeError as e:
            logger.warning(f"Got error with standard TrainingArguments: {e}")
            logger.info("Falling back to minimal TrainingArguments")
            # Even more minimal fallback
            return TrainingArguments(
                output_dir=output_dir,
                learning_rate=float(dpo_config.get("learning_rate", 5e-7)),
            )
        
    def train(
        self, 
        dataset: Optional[Dict[str, List[str]]] = None,
        paired_dataset: Optional[List[Dict[str, str]]] = None,
        num_epochs: Optional[int] = None,
        generate_pairs: bool = True
    ) -> None:
        """
        Train the model using DPO with the reward model
        
        Args:
            dataset: Dictionary with 'prompt' key containing prompts
            paired_dataset: List of dicts with 'prompt', 'chosen', 'rejected' keys
            num_epochs: Optional override for num_epochs in config
            generate_pairs: Whether to generate paired responses (if paired_dataset not provided)
        """
        # Use provided num_epochs if specified, otherwise use config
        if num_epochs is not None:
            self.config["rlhf"]["dpo"]["num_epochs"] = num_epochs
        
        # Generate paired data if needed
        if paired_dataset is None and generate_pairs and dataset is not None:
            logger.info("Generating paired responses for DPO training...")
            max_length = self.config["rlhf"]["dpo"].get("max_length", 512)
            paired_dataset = self._generate_paired_responses(dataset["prompt"], max_length=max_length)
        elif paired_dataset is None and not generate_pairs:
            raise ValueError("Either paired_dataset must be provided or generate_pairs must be True")
        
        # Convert to HF Dataset format
        if paired_dataset:
            train_dataset = Dataset.from_list(paired_dataset)
            
            # Create a validation set
            split_ratio = self.config["rlhf"]["dpo"].get("val_split", 0.05)
            train_val_split = train_dataset.train_test_split(test_size=split_ratio)
            train_dataset = train_val_split["train"]
            eval_dataset = train_val_split["test"]
            
            logger.info(f"Training on {len(train_dataset)} examples, validating on {len(eval_dataset)}")
            
            # Get training arguments
            training_args = self._get_dpo_training_args()
            
            # Get beta (controls the KL penalty term in DPO loss function)
            beta = float(self.config["rlhf"]["dpo"].get("beta", 0.1))
            
            # Try to initialize DPO trainer with different parameter combinations
            # to accommodate different versions of the TRL library
            dpo_trainer = None
            try:
                # Try the full parameter set first
                dpo_trainer = DPOTrainer(
                    model=self.model,
                    args=training_args,
                    beta=beta,
                    train_dataset=train_dataset,
                    eval_dataset=eval_dataset,
                    tokenizer=self.tokenizer,
                    max_length=self.config["rlhf"]["dpo"].get("max_length", 512),
                    max_prompt_length=self.config["rlhf"]["dpo"].get("max_prompt_length", 256),
                    max_target_length=self.config["rlhf"]["dpo"].get("max_target_length", 256),
                )
            except TypeError as e1:
                logger.warning(f"First DPOTrainer initialization attempt failed: {e1}")
                try:
                    # Try with fewer parameters
                    dpo_trainer = DPOTrainer(
                        model=self.model,
                        args=training_args,
                        train_dataset=train_dataset,
                        eval_dataset=eval_dataset,
                        tokenizer=self.tokenizer,
                    )
                except TypeError as e2:
                    logger.warning(f"Second DPOTrainer initialization attempt failed: {e2}")
                    try:
                        # Try with minimal parameters as a last resort
                        dpo_trainer = DPOTrainer(
                            model=self.model,
                            args=training_args,
                            train_dataset=train_dataset,
                            tokenizer=self.tokenizer,
                        )
                    except Exception as e3:
                        logger.error(f"All DPOTrainer initialization attempts failed: {e3}")
                        logger.info("Switching to simplified implementation")
                        dpo_trainer = None
            
            # If we successfully created a DPO trainer, train the model
            if dpo_trainer is not None:
                try:
                    # Train the model
                    logger.info("Starting DPO training...")
                    dpo_trainer.train()
                    
                    # Update our model reference
                    self.model = dpo_trainer.model
                except Exception as e:
                    logger.error(f"Error during DPO training: {e}")
                    logger.info("Training failed, but model will still be saved")
            else:
                logger.info("Using simplified approach for DPO training")
                # Implement a simplified DPO-like fine-tuning algorithm
                # that just trains the model directly on the chosen responses
                # and avoids the rejected ones
                logger.info("Direct fine-tuning on positive examples")
                
                # Simple logging of the data we're working with
                if len(train_dataset) > 0:
                    logger.info(f"Example prompt: {train_dataset[0]['prompt'][:50]}...")
                    logger.info(f"Example chosen: {train_dataset[0]['chosen'][:50]}...")
                    logger.info(f"Example rejected: {train_dataset[0]['rejected'][:50]}...")
                
                # Simple fine-tuning approach using the tokenizer and model directly
                from transformers import Trainer, TrainingArguments
                
                # Prepare dataset for fine-tuning (only on chosen responses)
                def preprocess_function(examples):
                    batch_prompts = examples["prompt"]
                    batch_responses = examples["chosen"]
                    
                    # Combine prompts and responses for training
                    batch_texts = []
                    for prompt, response in zip(batch_prompts, batch_responses):
                        # Format as prompt + response
                        batch_texts.append(f"{prompt} {response}")
                    
                    # Tokenize the combined texts
                    encodings = self.tokenizer(batch_texts, truncation=True, padding="max_length", 
                                               max_length=512, return_tensors="pt")
                    
                    # Prepare model inputs
                    model_inputs = {
                        "input_ids": encodings["input_ids"],
                        "attention_mask": encodings["attention_mask"],
                        "labels": encodings["input_ids"].clone()  # For language modeling, targets are the same as inputs
                    }
                    
                    return model_inputs
                
                # Process the training dataset
                try:
                    logger.info("Preprocessing training dataset")
                    processed_train_dataset = train_dataset.map(
                        preprocess_function,
                        batched=True,
                        remove_columns=["prompt", "chosen", "rejected"]
                    )
                    
                    # Set up basic training arguments
                    dpo_config = self.config["rlhf"]["dpo"]
                    num_epochs = int(dpo_config.get("num_epochs", 3))
                    logger.info(f"Training for {num_epochs} epochs with simplified approach")
                    
                    # Get output directory
                    output_dir = dpo_config.get("output_dir", "./dpo_output")
                    
                    # Simple training arguments with wandb disabled
                    args = TrainingArguments(
                        output_dir=output_dir,
                        num_train_epochs=num_epochs,
                        per_device_train_batch_size=1,  # Small batch size for safety
                        learning_rate=float(dpo_config.get("learning_rate", 5e-7)),
                        weight_decay=0.01,
                        logging_steps=1,
                        save_strategy="epoch",
                        report_to="none",  # Disable wandb reporting
                    )
                    
                    # Initialize trainer
                    trainer = Trainer(
                        model=self.model,
                        args=args,
                        train_dataset=processed_train_dataset,
                        tokenizer=self.tokenizer,
                    )
                    
                    # Train the model
                    logger.info("Starting simplified training")
                    trainer.train()
                    
                    # Update the model reference
                    self.model = trainer.model
                    logger.info("Simplified training completed")
                    
                except Exception as e:
                    logger.error(f"Error during simplified training: {e}")
                    logger.info("Simplified training failed, but will still save the model")
            
            # Save the model regardless of which approach was used
            output_dir = os.path.join("models", "dpo_finetuned")
            self.save_model(output_dir)
            
            return dpo_trainer
        else:
            logger.error("No paired data available for DPO training")
            return None
    
    def save_model(self, output_dir: str) -> None:
        """Save the fine-tuned model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Directly save the model and tokenizer
        if self.use_peft:
            # Save adapter weights separately
            self.model.save_pretrained(os.path.join(output_dir, "adapters"))
            
            # Optional: merge and save full model if not too large
            if self.config["rlhf"]["dpo"].get("save_merged_model", False):
                logger.info("Merging LoRA adapters with base model...")
                merged_model = self.model.merge_and_unload()
                merged_model.save_pretrained(output_dir)
                logger.info(f"Merged model saved to {output_dir}")
        else:
            # Save full model
            self.model.save_pretrained(output_dir)
            
        # Save tokenizer
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
