# rlhf/dpo_huggingface.py
import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from tqdm import tqdm

# Import standard HuggingFace libraries (not TRL)
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    BitsAndBytesConfig
)
from datasets import Dataset
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from inference.predictor import RewardPredictor

logger = logging.getLogger(__name__)

class CustomDPO:
    """
    Custom implementation of Direct Preference Optimization (DPO) without relying on 
    TRL library. Specifically designed for fine-tuning Llama 3.1 8B Instruct with 
    harmonic blend reward on the SHP dataset.
    
    Based on the paper: "Direct Preference Optimization: Your Language Model is 
    Secretly a Reward Model" by Rafailov et al.
    """
    
    def __init__(
        self,
        model,
        reference_model,
        tokenizer,
        device: torch.device,
        beta: float = 0.1,
        max_length: int = 512,
    ):
        self.model = model
        self.reference_model = reference_model
        self.tokenizer = tokenizer
        self.device = device
        self.beta = beta  # Controls regularization strength
        self.max_length = max_length
        
        # Initialize training components
        self.optimizer = None
        self.scheduler = None
        
    def _forward_pass(self, model, input_ids, attention_mask=None, labels=None) -> torch.Tensor:
        """Forward pass through the model to get logits"""
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            return_dict=True
        )
        
        return outputs.logits
    
    def _get_logps(self, logits, labels, attention_mask=None):
        """Calculate log probabilities over token sequences"""
        # Shift predictions for autoregressive log-probs
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        
        # Create mask for valid tokens
        if attention_mask is not None:
            shift_mask = attention_mask[..., 1:].contiguous()
        else:
            shift_mask = torch.ones_like(shift_labels)
        
        # Calculate log probabilities
        log_probs = F.log_softmax(shift_logits, dim=-1)
        
        # Gather logprobs corresponding to the tokens
        token_log_probs = log_probs.gather(
            dim=-1, 
            index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)
        
        # Mask out padding tokens
        token_log_probs = token_log_probs * shift_mask
        
        # Sum token log probs to get sequence log probs
        sequence_log_probs = token_log_probs.sum(dim=-1)
        
        return sequence_log_probs
        
    def dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        reference_chosen_logps: torch.Tensor,
        reference_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        """
        Calculate DPO loss as described in the original paper.
        
        Args:
            policy_chosen_logps: Log probs from policy model for chosen responses
            policy_rejected_logps: Log probs from policy model for rejected responses
            reference_chosen_logps: Log probs from reference model for chosen responses
            reference_rejected_logps: Log probs from reference model for rejected responses
            
        Returns:
            DPO loss
        """
        # Log ratios for chosen and rejected completions
        chosen_ratio = policy_chosen_logps - reference_chosen_logps
        rejected_ratio = policy_rejected_logps - reference_rejected_logps
        
        # The core DPO loss
        logits = chosen_ratio - rejected_ratio
        
        # Mathematically, this is DPO loss: -log_sigmoid(β * (r_θ(x,y_w) - r_θ(x,y_l)))
        # Where r_θ is the implied reward: log(π_θ(y|x)/π_ref(y|x))
        losses = -F.logsigmoid(self.beta * logits)
        
        # Return mean loss
        return losses.mean()
    
    def train_step(
        self,
        prompts_input_ids: torch.Tensor,
        prompts_attention_mask: torch.Tensor,
        chosen_input_ids: torch.Tensor,
        chosen_attention_mask: torch.Tensor,
        rejected_input_ids: torch.Tensor,
        rejected_attention_mask: torch.Tensor
    ) -> Dict:
        """Perform a single DPO training step"""
        # Move inputs to device
        prompts_input_ids = prompts_input_ids.to(self.device)
        prompts_attention_mask = prompts_attention_mask.to(self.device)
        chosen_input_ids = chosen_input_ids.to(self.device)
        chosen_attention_mask = chosen_attention_mask.to(self.device)
        rejected_input_ids = rejected_input_ids.to(self.device)
        rejected_attention_mask = rejected_attention_mask.to(self.device)
        
        # Get logprobs from policy model for chosen responses
        chosen_logits = self._forward_pass(self.model, chosen_input_ids, chosen_attention_mask)
        policy_chosen_logps = self._get_logps(
            chosen_logits,
            chosen_input_ids, 
            chosen_attention_mask
        )
        
        # Get logprobs from policy model for rejected responses  
        rejected_logits = self._forward_pass(self.model, rejected_input_ids, rejected_attention_mask)  
        policy_rejected_logps = self._get_logps(
            rejected_logits,  
            rejected_input_ids,
            rejected_attention_mask
        )
        
        # Get logprobs from reference model for chosen responses
        with torch.no_grad():
            ref_chosen_logits = self._forward_pass(self.reference_model, chosen_input_ids, chosen_attention_mask)
            reference_chosen_logps = self._get_logps(
                ref_chosen_logits,
                chosen_input_ids,
                chosen_attention_mask
            )
            
            # Get logprobs from reference model for rejected responses
            ref_rejected_logits = self._forward_pass(self.reference_model, rejected_input_ids, rejected_attention_mask)
            reference_rejected_logps = self._get_logps(
                ref_rejected_logits,
                rejected_input_ids,
                rejected_attention_mask
            )
        
        # Calculate the DPO loss
        loss = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps
        )
        
        # Backward pass
        if self.optimizer is not None:
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            if self.scheduler is not None:
                self.scheduler.step()
        
        return {
            "loss": loss.item(),
        }


class HuggingFaceDPOTrainer:
    """
    Custom DPO Trainer that doesn't rely on TRL library.
    Specifically designed for fine-tuning Llama 3.1 8B Instruct on SHP dataset
    with harmonic blend reward.
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
        
        # Only support huggingface-cli login for authentication
        logger.info("Using huggingface-cli login for authentication")
        logger.info("If you have not authenticated, please run: huggingface-cli login")
        
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
            
        # Create reference model (for DPO)
        logger.info("Creating reference model for DPO by copying policy model")
        if self.use_peft:
            # For PEFT, we need the original model for reference
            self.reference_model = AutoModelForCausalLM.from_pretrained(
                model_name,
                device_map="auto" if "cuda" in str(self.device) else None
            )
        else:
            # Clone the policy model and parameters for reference
            self.reference_model = AutoModelForCausalLM.from_pretrained(model_name)
            
        # Ensure reference model is on same device but not trainable
        self.reference_model.to(self.device)
        for param in self.reference_model.parameters():
            param.requires_grad = False
            
        # Set generation parameters
        self.max_length = self.config["rlhf"]["dpo"].get("max_length", 512)
        self.max_prompt_length = self.config["rlhf"]["dpo"].get("max_prompt_length", 256)
        self.batch_size = self.config["rlhf"]["dpo"].get("batch_size", 4)
        
        # Get beta for DPO
        self.beta = float(self.config["rlhf"]["dpo"].get("beta", 0.1))
        
        # Create DPO trainer
        self.dpo = CustomDPO(
            model=self.model,
            reference_model=self.reference_model,
            tokenizer=self.tokenizer,
            device=self.device,
            beta=self.beta,
            max_length=self.max_length
        )
            
        logger.info("Custom DPO Trainer initialized for Llama 3.1 8B")

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
    
    def _prepare_batch(self, examples):
        """
        Prepare batch for DPO training by tokenizing prompts, chosen, and rejected responses
        """
        prompts = examples["prompt"]
        chosen = examples["chosen"]
        rejected = examples["rejected"]
        
        # Tokenize prompts
        prompt_tokens = self.tokenizer(
            prompts,
            padding=True,
            truncation=True,
            max_length=self.max_prompt_length,
            return_tensors="pt"
        )
        
        # Tokenize chosen completions (with prompts)
        chosen_tokens = self.tokenizer(
            [f"{prompt} {resp}" for prompt, resp in zip(prompts, chosen)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Tokenize rejected completions (with prompts)
        rejected_tokens = self.tokenizer(
            [f"{prompt} {resp}" for prompt, resp in zip(prompts, rejected)],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            "prompts_input_ids": prompt_tokens.input_ids,
            "prompts_attention_mask": prompt_tokens.attention_mask,
            "chosen_input_ids": chosen_tokens.input_ids,
            "chosen_attention_mask": chosen_tokens.attention_mask,
            "rejected_input_ids": rejected_tokens.input_ids,
            "rejected_attention_mask": rejected_tokens.attention_mask
        }
    
    def train(
        self, 
        dataset: Optional[Dict[str, List[str]]] = None,
        paired_dataset: Optional[List[Dict[str, str]]] = None,
        num_epochs: Optional[int] = None,
        generate_pairs: bool = True
    ) -> None:
        """
        Train the model using custom DPO implementation
        
        Args:
            dataset: Dictionary with 'prompt' key containing prompts
            paired_dataset: List of dicts with 'prompt', 'chosen', 'rejected' keys
            num_epochs: Optional override for num_epochs in config
            generate_pairs: Whether to generate paired responses (if paired_dataset not provided)
        """
        # Use provided num_epochs if specified, otherwise use config
        if num_epochs is not None:
            self.config["rlhf"]["dpo"]["num_epochs"] = num_epochs
        else:
            num_epochs = self.config["rlhf"]["dpo"].get("num_epochs", 3)
            
        # Get optimization parameters from config
        dpo_config = self.config["rlhf"]["dpo"]
        learning_rate = float(dpo_config.get("learning_rate", 5e-7))
        
        # Generate paired data if needed
        if paired_dataset is None and generate_pairs and dataset is not None:
            logger.info("Generating paired responses for DPO training...")
            paired_dataset = self._generate_paired_responses(
                dataset["prompt"], 
                max_length=self.max_length
            )
        elif paired_dataset is None and not generate_pairs:
            raise ValueError("Either paired_dataset must be provided or generate_pairs must be True")
        
        # Convert to HF Dataset format
        if paired_dataset:
            train_dataset = Dataset.from_list(paired_dataset)
            
            # Create a validation set
            split_ratio = dpo_config.get("val_split", 0.05)
            train_val_split = train_dataset.train_test_split(test_size=split_ratio)
            train_dataset = train_val_split["train"]
            eval_dataset = train_val_split["test"]
            
            logger.info(f"Training on {len(train_dataset)} examples, validating on {len(eval_dataset)}")
            
            # Setup optimizer and scheduler
            optimizer = torch.optim.AdamW(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=dpo_config.get("weight_decay", 0.01),
                betas=(0.9, 0.999),
                eps=1e-8
            )
            
            # Linear learning rate scheduler with warmup
            warmup_steps = int(0.1 * len(train_dataset) / self.batch_size)
            total_steps = int(num_epochs * len(train_dataset) / self.batch_size)
            
            def lr_lambda(current_step):
                if current_step < warmup_steps:
                    return float(current_step) / float(max(1, warmup_steps))
                return max(0.0, float(total_steps - current_step) / float(max(1, total_steps - warmup_steps)))
            
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
            
            # Attach optimizer and scheduler to DPO
            self.dpo.optimizer = optimizer
            self.dpo.scheduler = scheduler
            
            # Training loop
            logger.info(f"Starting DPO training for {num_epochs} epochs")
            
            global_step = 0
            total_loss = 0.0
            
            for epoch in range(num_epochs):
                epoch_loss = 0.0
                epoch_steps = 0
                
                logger.info(f"Epoch {epoch+1}/{num_epochs}")
                
                # Process data in batches
                for i in range(0, len(train_dataset), self.batch_size):
                    batch_end = min(i + self.batch_size, len(train_dataset))
                    batch_size = batch_end - i
                    
                    # Get batch
                    batch = train_dataset.select(range(i, batch_end))
                    
                    # Prepare batch
                    prepared_batch = self._prepare_batch(batch)
                    
                    # Run DPO training step
                    metrics = self.dpo.train_step(
                        prompts_input_ids=prepared_batch["prompts_input_ids"],
                        prompts_attention_mask=prepared_batch["prompts_attention_mask"],
                        chosen_input_ids=prepared_batch["chosen_input_ids"],
                        chosen_attention_mask=prepared_batch["chosen_attention_mask"],
                        rejected_input_ids=prepared_batch["rejected_input_ids"],
                        rejected_attention_mask=prepared_batch["rejected_attention_mask"]
                    )
                    
                    # Track metrics
                    step_loss = metrics["loss"]
                    total_loss += step_loss
                    epoch_loss += step_loss
                    global_step += 1
                    epoch_steps += 1
                    
                    # Log metrics periodically
                    if global_step % 10 == 0:
                        logger.info(f"Step {global_step}: loss = {step_loss:.4f}, avg loss = {total_loss/global_step:.4f}")
                
                # End of epoch
                epoch_avg_loss = epoch_loss / epoch_steps
                logger.info(f"Epoch {epoch+1} complete. Average loss: {epoch_avg_loss:.4f}")
                
                # Save checkpoint
                checkpoint_path = os.path.join("models", f"dpo_checkpoint_epoch_{epoch+1}")
                self.save_checkpoint(checkpoint_path, {
                    "epoch": epoch + 1,
                    "avg_loss": epoch_avg_loss,
                    "timestamp": time.strftime("%Y-%m-%d-%H-%M-%S")
                })
            
            # Save the final model
            output_dir = os.path.join("models", "dpo_finetuned")
            self.save_model(output_dir)
            
            logger.info("DPO training completed")
        else:
            logger.error("No paired data available for DPO training")
    
    def save_checkpoint(self, checkpoint_path: str, metadata: Dict = None):
        """Save a checkpoint during training with metadata"""
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save the model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save metadata if provided
        if metadata:
            with open(os.path.join(checkpoint_path, "checkpoint_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
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
        
        # Save training configuration
        dpo_config = {
            "learning_rate": self.config["rlhf"]["dpo"].get("learning_rate", 5e-7),
            "batch_size": self.config["rlhf"]["dpo"].get("batch_size", 4),
            "beta": self.beta,
            "model_name": self.model.config.name_or_path,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(dpo_config, f, indent=2)
        
        logger.info(f"Model, tokenizer, and config saved to {output_dir}")
