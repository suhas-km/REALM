# rlhf/ppo_huggingface.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm
import json

from inference.predictor import RewardPredictor

logger = logging.getLogger(__name__)

class CustomPPO:
    """
    Custom PPO implementation for language models, specifically designed for
    Llama 3.1 8B Instruct with harmonic blend reward on the SHP dataset.
    """
    def __init__(
        self, 
        model,
        tokenizer,
        device: torch.device,
        config: Dict,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        
        # PPO hyperparameters
        self.cliprange = config.get("cliprange", 0.2)
        self.cliprange_value = config.get("cliprange_value", 0.2)
        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lam", 0.95)  # GAE lambda
        self.kl_coef = config.get("kl_penalty", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        
        # Setup optimizer
        self.learning_rate = float(config.get("learning_rate", 1.41e-5))
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), 
            lr=self.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Keep a reference model for KL divergence calculation
        self.ref_model = None
        self._create_reference_model()
        
        # Create value head
        hidden_size = self.model.config.hidden_size
        self.value_head = nn.Linear(hidden_size, 1, bias=False).to(device)
        self.optimizer.add_param_group({'params': self.value_head.parameters()})
        
    def _create_reference_model(self):
        """Create a reference model by copying the current model for KL calculation"""
        self.ref_model = type(self.model)(self.model.config)
        self.ref_model.load_state_dict(self.model.state_dict())
        self.ref_model.to(self.device)
        self.ref_model.eval()
        
        # Freeze reference model parameters
        for param in self.ref_model.parameters():
            param.requires_grad = False
    
    def _value_loss(self, values: torch.Tensor, returns: torch.Tensor, old_values: torch.Tensor):
        """Calculate value function loss with clipping"""
        values_clipped = old_values + torch.clamp(
            values - old_values, -self.cliprange_value, self.cliprange_value
        )
        vf_loss1 = (values - returns) ** 2
        vf_loss2 = (values_clipped - returns) ** 2
        return 0.5 * torch.mean(torch.max(vf_loss1, vf_loss2))
    
    def _policy_loss(
        self, 
        logprobs: torch.Tensor, 
        old_logprobs: torch.Tensor, 
        advantages: torch.Tensor, 
        mask: torch.Tensor
    ):
        """Calculate policy loss with clipping"""
        ratio = torch.exp(logprobs - old_logprobs)
        pg_loss1 = -advantages * ratio
        pg_loss2 = -advantages * torch.clamp(ratio, 1.0 - self.cliprange, 1.0 + self.cliprange)
        pg_loss = torch.mean(torch.max(pg_loss1, pg_loss2) * mask)
        return pg_loss
    
    def _calculate_kl(self, logprobs: torch.Tensor, ref_logprobs: torch.Tensor, mask: torch.Tensor):
        """Calculate KL divergence between current and reference policy"""
        kl = (ref_logprobs - logprobs) * mask
        return torch.mean(kl)
    
    def _get_logprobs(self, model, input_ids, attention_mask=None):
        """Get log probabilities from the model"""
        with torch.set_grad_enabled(model.training):
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True
            )
            logits = outputs.logits
            
            # Shift logits and input_ids for next token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_ids = input_ids[..., 1:].contiguous()
            
            # Create attention mask for shifted sequence if provided
            if attention_mask is not None:
                shift_mask = attention_mask[..., 1:].contiguous()
            else:
                shift_mask = torch.ones_like(shift_ids)
            
            # Calculate log probabilities
            log_probs = F.log_softmax(shift_logits, dim=-1)
            
            # Get log prob of each chosen token
            token_log_probs = log_probs.gather(-1, shift_ids.unsqueeze(-1)).squeeze(-1)
            
            # Apply mask to get valid log probs
            token_log_probs = token_log_probs * shift_mask
            
            return token_log_probs, shift_mask
    
    def _get_values(self, hidden_states: torch.Tensor, mask: torch.Tensor):
        """Get state values from the value head"""
        # Use the last hidden state if hidden_states is a tuple
        if isinstance(hidden_states, tuple):
            hidden_states = hidden_states[0]
            
        # Get the last token's hidden state for each sequence
        last_tokens = (mask.sum(dim=1) - 1).to(dtype=torch.long)
        batch_indices = torch.arange(hidden_states.size(0), device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, last_tokens]
        
        # Pass through value head
        values = self.value_head(last_hidden).squeeze(-1)
        return values

    def _calculate_returns_and_advantages(
        self, 
        rewards: torch.Tensor, 
        values: torch.Tensor, 
        dones: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Calculate returns and advantages using GAE"""
        advantages = torch.zeros_like(rewards)
        returns = torch.zeros_like(rewards)
        
        last_gae_lam = 0
        for t in reversed(range(len(rewards))):
            if t == len(rewards) - 1:
                next_value = 0
                next_done = 1
            else:
                next_value = values[t + 1]
                next_done = dones[t + 1]
            
            delta = rewards[t] + self.gamma * next_value * (1 - next_done) - values[t]
            advantages[t] = last_gae_lam = delta + self.gamma * self.lam * (1 - next_done) * last_gae_lam
            returns[t] = advantages[t] + values[t]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        return returns, advantages
    
    def train_step(
        self, 
        input_ids: torch.Tensor, 
        attention_mask: torch.Tensor, 
        old_logprobs: torch.Tensor,
        old_values: torch.Tensor,
        rewards: torch.Tensor,
        response_mask: torch.Tensor,
        n_updates: int = 4
    ) -> Dict:
        """Perform a PPO update on a batch of data"""
        # Get current values
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            values = self._get_values(outputs.hidden_states[-1], attention_mask)
        
        # All examples are individual sequences, so done is always 1
        dones = torch.ones_like(rewards)
        
        # Calculate returns and advantages
        returns, advantages = self._calculate_returns_and_advantages(rewards, values, dones)
        
        # Calculate metrics
        value_loss_epoch = 0
        policy_loss_epoch = 0
        kl_epoch = 0
        
        # Multiple optimization epochs
        for _ in range(n_updates):
            # Get current log probabilities
            current_logprobs, current_mask = self._get_logprobs(
                self.model, input_ids, attention_mask
            )
            
            # Get reference log probabilities for KL calculation
            with torch.no_grad():
                ref_logprobs, _ = self._get_logprobs(
                    self.ref_model, input_ids, attention_mask
                )
            
            # Calculate masked log probs
            response_mask_shifted = response_mask[:, 1:]  # Shift to match token predictions
            
            # Calculate policy loss (only for generated tokens, not for prompt)
            policy_loss = self._policy_loss(
                current_logprobs, 
                old_logprobs, 
                advantages.unsqueeze(1).expand_as(current_logprobs), 
                response_mask_shifted
            )
            
            # Calculate KL divergence
            kl = self._calculate_kl(current_logprobs, ref_logprobs, current_mask)
            
            # Get fresh values
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            values = self._get_values(outputs.hidden_states[-1], attention_mask)
            
            # Calculate value loss
            value_loss = self._value_loss(values, returns, old_values)
            
            # Calculate entropy for regularization
            entropy = -torch.mean((torch.exp(current_logprobs) * current_logprobs) * response_mask_shifted)
            
            # Total loss
            loss = policy_loss + 0.5 * value_loss - self.entropy_coef * entropy + self.kl_coef * kl
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
            self.optimizer.step()
            
            # Accumulate metrics
            value_loss_epoch += value_loss.item()
            policy_loss_epoch += policy_loss.item()
            kl_epoch += kl.item()
        
        # Average metrics
        metrics = {
            "policy_loss": policy_loss_epoch / n_updates,
            "value_loss": value_loss_epoch / n_updates,
            "kl": kl_epoch / n_updates,
        }
        
        return metrics
    
    def update_reference_model(self):
        """Update the reference model with current model weights"""
        self.ref_model.load_state_dict(self.model.state_dict())


class HuggingFacePPOTrainer:
    """
    Custom PPO Trainer for language models that works without relying on TRL library.
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
        checkpoint_dir: Optional[str] = None
    ):
        self.config = config
        self.reward_predictor = reward_predictor
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Set checkpoint directory
        self.checkpoint_dir = checkpoint_dir or os.path.join("models", "ppo_checkpoints")
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        logger.info(f"Checkpoints will be saved to: {self.checkpoint_dir}")
        
        # Model initialization
        model_name = config["rlhf"]["ppo"]["model_name"]
        logger.info(f"Loading model and tokenizer: {model_name}")
        
        # Only support huggingface-cli login for authentication
        # Remove any environment variable token options
        logger.info("Using huggingface-cli login for authentication")
        logger.info("If you have not authenticated, please run: huggingface-cli login")
        
        try:
            # Load tokenizer and model
            self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
            # Set pad_token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = model or AutoModelForCausalLM.from_pretrained(model_name)
            self.model.to(self.device)
        except Exception as e:
            # If loading fails, try different approach
            logger.warning(f"Failed to load model/tokenizer: {e}")
            logger.info("Attempting alternative loading method...")
            
            try:
                # Try again with specific options
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    padding_side="left"
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
                    device_map="auto" if "cuda" in str(self.device) else None
                )
                self.model.to(self.device)
            except Exception as e2:
                # If both loading attempts fail, raise error with login instruction
                logger.error(f"Failed to load model using alternative method: {e2}")
                raise ValueError(f"Could not load model {model_name}. Please ensure you've run 'huggingface-cli login' to authenticate")
        
        # Create PPO algorithm
        ppo_config = config["rlhf"]["ppo"]
        self.ppo = CustomPPO(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            config=ppo_config
        )
        
        # Set up generation args
        self.max_length = config["rlhf"]["ppo"].get("max_length", 256)
        self.batch_size = config["rlhf"]["ppo"].get("batch_size", 8)
        
        logger.info("Custom PPO Trainer initialized for Llama 3.1 8B")
    
    def _prepare_dataset(self, dataset: Dict) -> Dataset:
        """Convert dict dataset to HF Dataset for PPO Trainer"""
        prompts = dataset.get("prompt", [])
        if not prompts:
            logger.warning("No prompts found in dataset")
            return None
            
        # Create HF dataset with prompts
        hf_dataset = Dataset.from_dict({"query": prompts})
        return hf_dataset
    
    def _generate_responses(self, prompts: List[str]) -> List[torch.Tensor]:
        """Generate responses for a list of prompts"""
        tokenized_prompts = []
        max_prompt_length = 0
        
        # Tokenize all prompts
        for prompt in prompts:
            tokens = self.tokenizer(prompt, return_tensors="pt", padding=False).input_ids[0]
            tokenized_prompts.append(tokens)
            max_prompt_length = max(max_prompt_length, len(tokens))
        
        responses = []
        response_tensors = []
        
        # Generate responses
        for i, tokens in enumerate(tokenized_prompts):
            # Prepare input
            input_ids = tokens.unsqueeze(0).to(self.device)
            
            # Generate with sampling
            with torch.no_grad():
                output = self.model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_length,
                    do_sample=True,
                    top_p=0.9,
                    top_k=50,
                    temperature=0.8,
                    pad_token_id=self.tokenizer.pad_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                )
            
            # Get just the response part (exclude the prompt)
            response_tensor = output[0]
            response = self.tokenizer.decode(response_tensor[len(tokens):], skip_special_tokens=True)
            
            responses.append(response)
            response_tensors.append(response_tensor)
            
        return responses, response_tensors
    
    def _compute_rewards(self, prompts: List[str], responses: List[str]) -> torch.Tensor:
        """Compute rewards for prompt-response pairs using harmonic blend"""
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.reward_predictor.predict(prompt, response)
            rewards.append(reward)
        return torch.tensor(rewards, device=self.device)
    
    def _compute_logprobs_values(self, input_ids: List[torch.Tensor], response_start_indices: List[int]):
        """Compute log probs and values for each sequence"""
        # Pad sequences to the same length
        max_length = max([len(ids) for ids in input_ids])
        padded_ids = torch.ones((len(input_ids), max_length), dtype=torch.long, device=self.device) * self.tokenizer.pad_token_id
        
        # Create attention mask
        attention_mask = torch.zeros((len(input_ids), max_length), dtype=torch.long, device=self.device)
        
        # Create response mask (1 for response tokens, 0 for prompt tokens)
        response_mask = torch.zeros((len(input_ids), max_length), dtype=torch.long, device=self.device)
        
        # Fill in the tensors
        for i, (ids, start_idx) in enumerate(zip(input_ids, response_start_indices)):
            seq_len = len(ids)
            padded_ids[i, :seq_len] = ids
            attention_mask[i, :seq_len] = 1
            response_mask[i, start_idx:seq_len] = 1
        
        # Get log probs and values
        with torch.no_grad():
            # Get log probs
            logprobs, mask = self.ppo._get_logprobs(
                self.model, padded_ids, attention_mask
            )
            
            # Get values
            outputs = self.model(
                input_ids=padded_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            values = self.ppo._get_values(outputs.hidden_states[-1], attention_mask)
        
        return padded_ids, attention_mask, response_mask, logprobs, values
    
    def train(
        self, 
        dataset: Dict, 
        num_epochs: int = 1, 
        max_steps: int = 100, 
        checkpoint_interval: int = 1
    ):
        """Train the model using custom PPO implementation"""
        logger.info(f"Starting custom PPO training for {num_epochs} epochs, max {max_steps} steps")
        
        # Prepare dataset
        hf_dataset = self._prepare_dataset(dataset)
        if hf_dataset is None:
            logger.error("Cannot train with empty dataset")
            return self.model
        
        # Limit steps if needed (max_steps=0 means use full dataset)
        if max_steps and max_steps > 0 and max_steps < len(hf_dataset):
            logger.info(f"Limiting dataset to {max_steps} examples (from {len(hf_dataset)})")
            hf_dataset = hf_dataset.select(range(max_steps))
        else:
            logger.info(f"Using full dataset with {len(hf_dataset)} examples")
        
        # Training loop
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Track metrics
            epoch_rewards = []
            epoch_losses = []
            
            # Process data in batches
            for i in range(0, len(hf_dataset), self.batch_size):
                batch_end = min(i + self.batch_size, len(hf_dataset))
                batch = hf_dataset.select(range(i, batch_end))
                prompts = batch["query"]
                
                logger.info(f"Processing batch {i//self.batch_size + 1}/{len(hf_dataset)//self.batch_size + 1}")
                
                # Step 1: Generate responses
                responses, response_tensors = self._generate_responses(prompts)
                
                # Print example
                if len(responses) > 0:
                    logger.info(f"Example prompt: {prompts[0][:50]}...")
                    logger.info(f"Example response: {responses[0][:50]}...")
                
                # Step 2: Calculate rewards
                rewards = self._compute_rewards(prompts, responses)
                logger.info(f"Batch reward mean: {rewards.mean().item():.4f}")
                epoch_rewards.append(rewards.mean().item())
                
                # Step 3: Prepare inputs for PPO update
                input_tensors = []
                start_indices = []
                
                for j, prompt in enumerate(prompts):
                    prompt_tokens = self.tokenizer(prompt, return_tensors="pt").input_ids[0]
                    start_indices.append(len(prompt_tokens))
                    input_tensors.append(response_tensors[j])
                
                # Step 4: Compute logprobs and values
                padded_ids, attention_mask, response_mask, logprobs, values = self._compute_logprobs_values(
                    input_tensors, start_indices
                )
                
                # Step 5: PPO update
                metrics = self.ppo.train_step(
                    input_ids=padded_ids,
                    attention_mask=attention_mask,
                    old_logprobs=logprobs.detach(),
                    old_values=values.detach(),
                    rewards=rewards,
                    response_mask=response_mask,
                    n_updates=4
                )
                
                # Log metrics
                logger.info(f"Policy loss: {metrics['policy_loss']:.4f}, Value loss: {metrics['value_loss']:.4f}, KL: {metrics['kl']:.4f}")
                epoch_losses.append(metrics['policy_loss'])
                
                # Update reference model every few batches
                if i % (5 * self.batch_size) == 0:
                    self.ppo.update_reference_model()
                    logger.info("Updated reference model")
            
            # End of epoch
            avg_reward = sum(epoch_rewards) / len(epoch_rewards) if epoch_rewards else 0
            avg_loss = sum(epoch_losses) / len(epoch_losses) if epoch_losses else 0
            
            logger.info(f"Epoch {epoch+1} complete. Average reward: {avg_reward:.4f}, Average loss: {avg_loss:.4f}")
            
            # Save checkpoint
            if (epoch + 1) % checkpoint_interval == 0:
                checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-epoch-{epoch+1}")
                self.save_checkpoint(checkpoint_path, metadata={
                    "epoch": epoch + 1,
                    "avg_reward": avg_reward,
                    "timestamp": time.strftime("%Y-%m-%d-%H-%M-%S")
                })
                logger.info(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")
        
        logger.info("Custom PPO training completed")
        return self.model
    
    def save_checkpoint(self, checkpoint_path: str, metadata: Dict = None):
        """Save a checkpoint during training with metadata"""
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save the model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save value head
        torch.save(self.ppo.value_head.state_dict(), os.path.join(checkpoint_path, "value_head.pt"))
        
        # Save metadata if provided
        if metadata:
            with open(os.path.join(checkpoint_path, "checkpoint_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_model(self, output_dir: str):
        """Save the final fine-tuned model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save value head
        torch.save(self.ppo.value_head.state_dict(), os.path.join(output_dir, "value_head.pt"))
        
        # Save training configuration
        ppo_config = {
            "learning_rate": self.config["rlhf"]["ppo"].get("learning_rate", 1.41e-5),
            "batch_size": self.config["rlhf"]["ppo"].get("batch_size", 8),
            "model_name": self.model.config.name_or_path,
            "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(os.path.join(output_dir, "training_config.json"), "w") as f:
            json.dump(ppo_config, f, indent=2)
        
        logger.info(f"Model, tokenizer, and config saved to {output_dir}")
