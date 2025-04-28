# rlhf/ppo_huggingface.py
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import logging
import time
import random
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
    Supports multi-GPU model parallelism for efficient training.
    """
    def __init__(
        self, 
        model,
        tokenizer,
        device: torch.device,
        config: Dict,
        model_devices=None,
        ref_model_devices=None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.config = config
        self.model_devices = model_devices or [0]
        self.ref_model_devices = ref_model_devices or [0]
        
        # Enable gradient checkpointing for memory efficiency
        if hasattr(self.model, 'gradient_checkpointing_enable'):
            self.model.gradient_checkpointing_enable()
            logger.info("Gradient checkpointing enabled for policy model")
        
        # PPO hyperparameters
        self.cliprange = config.get("cliprange", 0.2)
        self.cliprange_value = config.get("cliprange_value", 0.2)
        self.gamma = config.get("gamma", 0.99)
        self.lam = config.get("lam", 0.95)  # GAE lambda
        self.kl_coef = config.get("kl_penalty", 0.2)
        self.entropy_coef = config.get("entropy_coef", 0.01)
        
        # Setup optimizer with GPU distribution if multiple devices are specified
        self.learning_rate = float(config.get("learning_rate", 1.41e-5))
        
        # Regular optimizer setup regardless of GPU count
        # Removed DeepSpeed as it's causing issues
        logger.info(f"Setting up optimizer for policy model")
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
        
        # Handle loading state dictionary depending on model type (distributed or not)
        if hasattr(self.model, 'module'):  # For DistributedDataParallel
            self.ref_model.load_state_dict(self.model.module.state_dict())
        else:
            self.ref_model.load_state_dict(self.model.state_dict())
        
        # Move reference model to appropriate device
        if len(self.ref_model_devices) > 0:
            # Simply move to the first specified device
            ref_device = f"cuda:{self.ref_model_devices[0]}"
            self.ref_model = self.ref_model.to(ref_device)
            logger.info(f"Moved reference model to {ref_device}")
        
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
        """Get log probabilities from the model with extreme memory optimization"""
        with torch.set_grad_enabled(model.training):
            # Process one sequence at a time to minimize memory usage
            batch_size = input_ids.shape[0]
            max_chunk_size = 1  # Process just 1 sequence at a time for extreme memory saving
            all_token_log_probs = []
            all_shift_masks = []
            
            # Ensure gradient checkpointing is enabled
            if hasattr(model, 'gradient_checkpointing') and not model.gradient_checkpointing:
                try:
                    model.gradient_checkpointing_enable()
                    logger.info("Enabled gradient checkpointing for logprob computation")
                except Exception as e:
                    logger.warning(f"Could not enable gradient checkpointing: {e}")
            
            for chunk_start in range(0, batch_size, max_chunk_size):
                chunk_end = min(chunk_start + max_chunk_size, batch_size)
                
                # Clear cache before each chunk
                torch.cuda.empty_cache()
                
                # Move chunk data to device and use CPU offloading where possible
                chunk_input_ids = input_ids[chunk_start:chunk_end].clone()
                
                if attention_mask is not None:
                    chunk_attention_mask = attention_mask[chunk_start:chunk_end].clone()
                else:
                    chunk_attention_mask = None
                
                try:
                    # Use mixed precision for inference with modern API
                    with torch.amp.autocast('cuda', dtype=torch.float32):
                        # Run forward pass
                        outputs = model(
                            input_ids=chunk_input_ids,
                            attention_mask=chunk_attention_mask,
                            return_dict=True
                        )
                        
                        logits = outputs.logits if hasattr(outputs, "logits") else outputs[0]
                        
                        # Move subsequent computation to CPU if possible to save GPU memory
                        cpu_logits = logits.detach().cpu()
                        cpu_input_ids = chunk_input_ids.cpu()
                        cpu_attn_mask = chunk_attention_mask.cpu() if chunk_attention_mask is not None else None
                        
                        # Free GPU memory immediately
                        del outputs, logits
                        torch.cuda.empty_cache()
                        
                        # Perform rest of computation on CPU
                        shift_logits = cpu_logits[:, :-1, :].contiguous()
                        shift_ids = cpu_input_ids[:, 1:].contiguous()
                        
                        # Create mask to identify where the tokens are (ignoring padding)
                        if cpu_attn_mask is not None:
                            shift_mask = cpu_attn_mask[:, 1:].contiguous()
                        else:
                            shift_mask = torch.ones_like(shift_ids)
                        
                        # Compute log probabilities from logits
                        log_probs = F.log_softmax(shift_logits, dim=-1)
                        
                        # Extract only the log probs of the actual token that appeared
                        token_log_probs = -torch.gather(
                            log_probs, 
                            dim=-1, 
                            index=shift_ids.unsqueeze(-1)
                        ).squeeze(-1)
                        
                        # Apply mask to get the actual token log probs
                        token_log_probs = token_log_probs * shift_mask
                    
                        # Store results from this chunk
                        all_token_log_probs.append(token_log_probs)
                        all_shift_masks.append(shift_mask)
                    
                except torch.cuda.OutOfMemoryError as e:
                    logger.error(f"OOM in logprob computation for chunk {chunk_start}: {e}")
                    # Create dummy outputs in case of OOM
                    seq_len = chunk_input_ids.size(1) - 1 if chunk_input_ids.size(1) > 1 else 1
                    dummy_log_probs = torch.zeros((chunk_end - chunk_start, seq_len))
                    dummy_mask = torch.ones((chunk_end - chunk_start, seq_len))
                    
                    all_token_log_probs.append(dummy_log_probs)
                    all_shift_masks.append(dummy_mask)
                    logger.warning("Using dummy values due to OOM")
                
                # Explicitly clean up memory
                if 'chunk_input_ids' in locals(): del chunk_input_ids
                if 'chunk_attention_mask' in locals(): del chunk_attention_mask
                if 'cpu_logits' in locals(): del cpu_logits
                if 'cpu_input_ids' in locals(): del cpu_input_ids
                if 'cpu_attn_mask' in locals(): del cpu_attn_mask
                if 'shift_logits' in locals(): del shift_logits
                if 'shift_ids' in locals(): del shift_ids
                if 'log_probs' in locals(): del log_probs
                if 'token_log_probs' in locals(): del token_log_probs
                if 'shift_mask' in locals(): del shift_mask
                torch.cuda.empty_cache()
            
            # Combine results from all chunks (all on CPU at this point)
            if len(all_token_log_probs) > 1:
                token_log_probs = torch.cat(all_token_log_probs, dim=0)
                shift_mask = torch.cat(all_shift_masks, dim=0)
            else:
                token_log_probs = all_token_log_probs[0]
                shift_mask = all_shift_masks[0]
            
            # Move results back to the original device only at the end
            device = input_ids.device
            token_log_probs = token_log_probs.to(device)
            shift_mask = shift_mask.to(device)
            
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
        # Device synchronization to avoid issues
        torch.cuda.synchronize(self.device)
        
        # Move everything to the policy model's device
        input_ids = input_ids.to(self.device)
        attention_mask = attention_mask.to(self.device)
        old_logprobs = old_logprobs.to(self.device)
        old_values = old_values.to(self.device)
        rewards = rewards.to(self.device)
        response_mask = response_mask.to(self.device)
        
        # Get current values and logprobs on policy device
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            values = self._get_values(outputs.hidden_states[-1], attention_mask)
        
        # Get reference model logprobs WITHOUT moving model (compute on ref model's device)
        with torch.no_grad():
            # Get ref model's device
            ref_device = next(self.ref_model.parameters()).device
            logger.info(f"Computing reference logprobs on {ref_device}")
            
            # Copy input tensors to ref device
            ref_input_ids = input_ids.to(ref_device)
            ref_attention_mask = attention_mask.to(ref_device)
            
            # Compute logprobs on ref device
            ref_logprobs, _ = self._get_logprobs(
                self.ref_model, ref_input_ids, ref_attention_mask
            )
            
            # Move only the resulting tensor back to policy device
            ref_logprobs = ref_logprobs.to(self.device)
        
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
            
            # Calculate masked log probs
            response_mask_shifted = response_mask[:, 1:]  # Shift to match token predictions
            
            # Calculate policy loss (only for generated tokens, not for prompt)
            policy_loss = self._policy_loss(
                current_logprobs, 
                old_logprobs, 
                advantages.unsqueeze(1).expand_as(current_logprobs), 
                response_mask_shifted
            )
            
            # Calculate KL divergence using the pre-computed ref_logprobs
            kl = self._calculate_kl(current_logprobs, ref_logprobs, current_mask)
            
            # Get fresh values
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
                return_dict=True
            )
            current_values = self._get_values(outputs.hidden_states[-1], attention_mask)
            
            # Calculate value loss
            value_loss = self._value_loss(current_values, returns, old_values)
            
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
    
    Features:
    - Multi-GPU model parallelism support for distributed training
    - Memory-efficient computation with mixed precision and chunking
    - Support for tensor parallelism using DeepSpeed (if available)
    """
    
    def __init__(
        self,
        config: Dict,
        reward_predictor: RewardPredictor,
        tokenizer=None,
        model=None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
        policy_model_devices=None,
        ref_model_devices=None,
        reward_model_device=None,
        embedding_device=None
    ):
        self.config = config
        self.reward_predictor = reward_predictor
        
        # Setup device allocation for multi-GPU training
        self.world_size = torch.cuda.device_count()
        
        # Set primary device
        if device is None:
            device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.device = device
        logger.info(f"Primary device: {self.device}")
        
        # Setup model parallelism
        self.policy_model_devices = policy_model_devices or [0]
        self.ref_model_devices = ref_model_devices or [0]
        self.reward_model_device = reward_model_device
        self.embedding_device = embedding_device
        
        # Log GPU allocation
        logger.info("GPU allocation for model parallelism:")
        logger.info(f"Policy model: GPUs {self.policy_model_devices}")
        logger.info(f"Reference model: GPUs {self.ref_model_devices}")
        logger.info(f"Reward model: GPU {self.reward_model_device}")
        logger.info(f"Embedding model: GPU {self.embedding_device}")
        
        # Disable distributed training completely
        self.use_distributed = False
        
        # No mixed precision - removed per user request
        
        # Load tokenizer and model
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
        
        # Initialize the custom PPO algorithm with model parallelism support
        logger.info("Custom PPO Trainer initialized for Llama 3.1 8B with model parallelism")
        self.ppo = CustomPPO(
            model=self.model,
            tokenizer=self.tokenizer,
            device=self.device,
            config=config["rlhf"]["ppo"],
            model_devices=self.policy_model_devices,
            ref_model_devices=self.ref_model_devices,
        )
        
        # Set up generation args
        self.max_length = config["rlhf"]["ppo"].get("max_length", 256)
        self.batch_size = config["rlhf"]["ppo"].get("batch_size", 8)
        
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
    
    def _compute_rewards(self, prompts: List[str], responses: List[str]):
        """Compute rewards for prompt-response pairs using harmonic blend"""
        # Move reward model to specified device
        original_device = None
        if self.reward_model_device is not None:
            reward_device = f"cuda:{self.reward_model_device}"
            logger.info(f"Moving reward computation to dedicated GPU: {reward_device}")
            
            try:
                if hasattr(self.reward_predictor, 'to'):
                    # Save original device to restore later
                    original_device = self.reward_predictor.device if hasattr(self.reward_predictor, 'device') else None
                    self.reward_predictor.to(reward_device)
                    
                    # Synchronize to ensure device transfer is complete
                    torch.cuda.synchronize(reward_device)
            except Exception as e:
                logger.error(f"Failed to move reward model to dedicated device: {e}")
        
        # Process in chunks for memory efficiency
        chunk_size = min(len(prompts), 4)  # Process max 4 prompt-response pairs at a time
        all_rewards = []
        
        logger.info(f"Computing rewards for {len(prompts)} examples with chunk size {chunk_size}")
        
        for i in range(0, len(prompts), chunk_size):
            chunk_prompts = prompts[i:i+chunk_size]
            chunk_responses = responses[i:i+chunk_size]
            
            # Compute rewards for this chunk
            try:
                # No mixed precision - standard computation
                chunk_rewards = []
                for p, r in zip(chunk_prompts, chunk_responses):
                    try:
                        # Get reward score using harmonic blend with the correct method name
                        reward = self.reward_predictor.predict(p, r)
                        chunk_rewards.append(reward)
                    except Exception as e:
                        logger.error(f"Error getting reward score: {e}")
                        # Use a default score of 0 in case of error
                        chunk_rewards.append(0.0)
                        
                all_rewards.extend(chunk_rewards)
                logger.info(f"Processed reward chunk {(i//chunk_size)+1}/{(len(prompts)+chunk_size-1)//chunk_size} with mean reward: {np.mean(chunk_rewards):.4f}")
            except Exception as e:
                logger.error(f"Error computing rewards for chunk {i//chunk_size + 1}: {e}")
                # Fallback to individual computation
                chunk_rewards = []
                for prompt, response in zip(chunk_prompts, chunk_responses):
                    try:
                        reward = self.reward_predictor.predict(prompt, response)
                        chunk_rewards.append(reward)
                    except Exception as inner_e:
                        logger.error(f"Error computing individual reward: {inner_e}")
                        chunk_rewards.append(0.0)  # Fallback value
                all_rewards.extend(chunk_rewards)
            
            # Explicitly free memory after each chunk
            torch.cuda.empty_cache()
        
        # Restore original device if we changed it
        if original_device is not None and reward_device is not None:
            logger.info(f"Restoring reward computation back to original device: {original_device}")
            try:
                # Properly move back using the to() methods we added
                if hasattr(self.reward_predictor, 'to'):
                    self.reward_predictor.to(original_device)
            except Exception as e:
                logger.error(f"Error restoring reward predictor to original device: {e}")
        
        # Convert rewards to tensor and move to the PPO device
        rewards_tensor = torch.tensor(all_rewards, dtype=torch.float32).to(self.device)
        return rewards_tensor
    
    def _compute_logprobs_values(self, input_tensors: List[torch.Tensor], response_start_indices: List[int]):
        """Compute log probs and values for each sequence with memory-efficient chunking"""
        # Determine maximum sequence length
        max_length = max([t.size(0) for t in input_tensors])
        batch_size = len(input_tensors)
        
        # Process in smaller chunks to avoid OOM - use single sample processing for huge models
        chunk_size = 1  # Process just 1 sequence at a time to avoid OOM
        all_logprobs = []
        all_values = []
        all_padded_ids = []
        all_attention_masks = []
        all_response_masks = []
        
        logger.info(f"Processing {batch_size} sequences with chunk size {chunk_size} (extreme memory saving mode)")
        
        for i in range(0, batch_size, chunk_size):
            chunk_end = min(i + chunk_size, batch_size)
            chunk_size_actual = chunk_end - i
            logger.info(f"Processing chunk {i//chunk_size + 1}/{(batch_size+chunk_size-1)//chunk_size}: items {i}-{chunk_end-1}")
            
            try:
                # Create padded tensors for this chunk
                chunk_padded = torch.ones((chunk_size_actual, max_length), dtype=torch.long) * self.tokenizer.pad_token_id
                chunk_attention_mask = torch.zeros((chunk_size_actual, max_length), dtype=torch.long)
                chunk_response_mask = torch.zeros((chunk_size_actual, max_length), dtype=torch.float)
                
                # Fill padded tensors for this chunk
                for j in range(chunk_size_actual):
                    idx = i + j
                    tensor = input_tensors[idx]
                    start_idx = response_start_indices[idx]
                    length = tensor.size(0)
                    
                    chunk_padded[j, :length] = tensor
                    chunk_attention_mask[j, :length] = 1
                    chunk_response_mask[j, start_idx:length] = 1
                
                # Move to CPU first, then to GPU in parts if needed to avoid memory spikes
                torch.cuda.empty_cache()  # Clear cache before processing
                
                # Move to device
                chunk_padded = chunk_padded.to(self.device)
                chunk_attention_mask = chunk_attention_mask.to(self.device)
                chunk_response_mask = chunk_response_mask.to(self.device)
                
                # Run forward pass with extreme memory optimization and gradient checkpointing
                # No mixed precision - standard computation
                # Make sure gradient checkpointing is enabled
                if not getattr(self.model, 'gradient_checkpointing', False):
                    try:
                        if hasattr(self.model, 'gradient_checkpointing_enable'):
                            self.model.gradient_checkpointing_enable()
                            logger.info("Enabled gradient checkpointing for model")
                    except Exception as e:
                        logger.warning(f"Could not enable gradient checkpointing: {e}")
                
                # Forward pass through model
                outputs = self.model(
                    input_ids=chunk_padded,
                    attention_mask=chunk_attention_mask,
                    output_hidden_states=True,
                    return_dict=True
                )
                hidden_states = outputs.hidden_states[-1]
                
                # Compute values
                chunk_values = self.ppo._get_values(hidden_states, chunk_attention_mask)
                
                # Compute log probabilities
                chunk_logprobs, _ = self.ppo._get_logprobs(
                    self.model, 
                    chunk_padded, 
                    chunk_attention_mask
                )
                
                # Move results to CPU to save GPU memory
                chunk_logprobs_cpu = chunk_logprobs.detach().cpu()
                chunk_values_cpu = chunk_values.detach().cpu()
                
                # Collect results on CPU
                all_logprobs.append(chunk_logprobs_cpu)
                all_values.append(chunk_values_cpu)
                all_padded_ids.append(chunk_padded.cpu())
                all_attention_masks.append(chunk_attention_mask.cpu())
                all_response_masks.append(chunk_response_mask.cpu())
                
            except torch.cuda.OutOfMemoryError as e:
                logger.error(f"Out of memory error in chunk {i}: {e}")
                
                # Create dummy outputs as fallback
                dummy_logprobs = torch.zeros((chunk_size_actual, max_length), dtype=torch.float32)
                dummy_values = torch.zeros((chunk_size_actual, 1), dtype=torch.float32)
                dummy_padded_ids = torch.zeros((chunk_size_actual, max_length), dtype=torch.long)
                dummy_attention_masks = torch.zeros((chunk_size_actual, max_length), dtype=torch.long)
                dummy_response_masks = torch.zeros((chunk_size_actual, max_length), dtype=torch.bool)
                
                all_logprobs.append(dummy_logprobs)
                all_values.append(dummy_values)
                all_padded_ids.append(dummy_padded_ids)
                all_attention_masks.append(dummy_attention_masks)
                all_response_masks.append(dummy_response_masks)
                
                logger.warning(f"Using dummy values for chunk {i} due to OOM")
            
            # Explicitly free memory
            if 'chunk_padded' in locals(): del chunk_padded
            if 'chunk_attention_mask' in locals(): del chunk_attention_mask
            if 'chunk_response_mask' in locals(): del chunk_response_mask
            if 'model_outputs' in locals(): del model_outputs
            if 'hidden_states' in locals(): del hidden_states
            if 'chunk_logprobs' in locals(): del chunk_logprobs
            if 'chunk_values' in locals(): del chunk_values
            torch.cuda.empty_cache()
        
        # Create combined tensors but keep on CPU first
        max_size = max([tensor.size(1) for tensor in all_logprobs if tensor.numel() > 0]) if all_logprobs else 0
        
        # Make sure all tensors have the same size in dimension 1
        if max_size > 0:
            for i in range(len(all_logprobs)):
                if all_logprobs[i].numel() > 0 and all_logprobs[i].size(1) < max_size:
                    pad_size = max_size - all_logprobs[i].size(1)
                    all_logprobs[i] = torch.nn.functional.pad(all_logprobs[i], (0, pad_size), 'constant', 0)
            
            # Ensure all other tensors match in size too
            for i in range(len(all_padded_ids)):
                if all_padded_ids[i].numel() > 0 and all_padded_ids[i].size(1) < max_size:
                    pad_size = max_size - all_padded_ids[i].size(1)
                    all_padded_ids[i] = torch.nn.functional.pad(all_padded_ids[i], (0, pad_size), 'constant', 0)
            
            for i in range(len(all_attention_masks)):
                if all_attention_masks[i].numel() > 0 and all_attention_masks[i].size(1) < max_size:
                    pad_size = max_size - all_attention_masks[i].size(1)
                    all_attention_masks[i] = torch.nn.functional.pad(all_attention_masks[i], (0, pad_size), 'constant', 0)
            
            for i in range(len(all_response_masks)):
                if all_response_masks[i].numel() > 0 and all_response_masks[i].size(1) < max_size:
                    pad_size = max_size - all_response_masks[i].size(1)
                    all_response_masks[i] = torch.nn.functional.pad(all_response_masks[i], (0, pad_size), 'constant', 0)
        
        # Now it's safe to concatenate
        if all(tensor.numel() > 0 for tensor in all_logprobs) and len(all_logprobs) > 0:
            logprobs_cpu = torch.cat(all_logprobs, dim=0)
            values_cpu = torch.cat(all_values, dim=0)
            padded_ids_cpu = torch.cat(all_padded_ids, dim=0)
            attention_mask_cpu = torch.cat(all_attention_masks, dim=0)
            response_mask_cpu = torch.cat(all_response_masks, dim=0)
        else:
            # Handle the case where we have empty tensors
            logger.warning("Empty tensors detected during computation, using dummy values")
            # Create dummy tensors with appropriate sizes
            batch_size = len(input_tensors)
            logprobs_cpu = torch.zeros((batch_size, 1), dtype=torch.float32)
            values_cpu = torch.zeros((batch_size, 1), dtype=torch.float32)
            padded_ids_cpu = torch.zeros((batch_size, 1), dtype=torch.long)
            attention_mask_cpu = torch.zeros((batch_size, 1), dtype=torch.long)
            response_mask_cpu = torch.zeros((batch_size, 1), dtype=torch.bool)
        
        # Now move everything to GPU in a single operation to avoid memory fragmentation
        logger.info("Moving final tensors to GPU...")
        torch.cuda.empty_cache()
        
        padded_ids = padded_ids_cpu.to(self.device)
        attention_mask = attention_mask_cpu.to(self.device)
        response_mask = response_mask_cpu.to(self.device)
        logprobs = logprobs_cpu.to(self.device)
        values = values_cpu.to(self.device)
        
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
