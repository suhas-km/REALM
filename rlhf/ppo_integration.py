# rlhf/ppo_integration.py
import os
import torch
import logging
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

from inference.predictor import RewardPredictor

logger = logging.getLogger(__name__)

class PPOTrainerWithCustomReward:
    """
    Simplified PPO-like Trainer with custom reward model for LLM fine-tuning
    """
    
    def __init__(
        self,
        config: Dict,
        reward_predictor: RewardPredictor,
        tokenizer=None,
        model=None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.reward_predictor = reward_predictor
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model if not provided
        if tokenizer is None or model is None:
            model_name = config["rlhf"]["ppo"]["model_name"]
            logger.info(f"Loading model and tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
            self.model = model
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # Move model to device
        self.model.to(self.device)
            
        # Initialize optimizer with explicit float conversion for learning rate
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=float(config["rlhf"]["ppo"]["learning_rate"])
        )
        
        logger.info("Simplified PPO Trainer with custom reward initialized")
    
    def _get_reward(self, prompt: str, response: str) -> float:
        """Get reward from the custom reward predictor"""
        return self.reward_predictor.predict(prompt, response)
        
    def train(self, dataset: Dict, num_epochs: int = 1, max_steps: int = 100):
        """Train the model using a simplified PPO-like approach"""
        logger.info(f"Starting simplified PPO training for {num_epochs} epochs, max {max_steps} steps")
        
        # Extract prompts from dataset
        prompts = dataset.get("prompt", [])
        if not prompts:
            logger.warning("No prompts found in dataset")
            return self.model
            
        # Limit steps for demonstration if needed
        prompts = prompts[:max_steps]
        
        self.model.train()
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            epoch_rewards = []
            for prompt_idx, prompt in enumerate(tqdm(prompts, desc="Processing prompts")):
                # Generate base response
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                with torch.no_grad():
                    outputs = self.model.generate(
                        input_ids,
                        max_length=self.config["rlhf"]["ppo"].get("max_length", 256),
                        do_sample=True,
                        temperature=0.7,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                
                response = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
                response = response[len(self.tokenizer.decode(input_ids[0], skip_special_tokens=True)):].strip()
                
                # Get reward for the response
                reward = self._get_reward(prompt, response)
                epoch_rewards.append(reward)
                
                # Skip optimization step for very low-reward responses
                if reward < 0.2:  # Arbitrary threshold
                    continue
                    
                # Train the model to maximize reward
                # This is a simplified approach - real PPO would track policy ratios, etc.
                self.optimizer.zero_grad()
                
                # Generate with gradient tracking for backprop
                input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.to(self.device)
                outputs = self.model(input_ids, labels=input_ids)
                loss = -outputs.loss * reward  # Invert loss direction to maximize reward
                
                loss.backward()
                self.optimizer.step()
                
                if prompt_idx % 10 == 0:
                    logger.info(f"Step {prompt_idx}, Reward: {reward:.4f}, Loss: {loss.item():.4f}")
            
            # Log epoch stats
            if epoch_rewards:
                avg_reward = sum(epoch_rewards) / len(epoch_rewards)
                logger.info(f"Epoch {epoch+1} average reward: {avg_reward:.4f}")
        
        logger.info("Simplified PPO training completed")
        return self.model
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned model"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
    
    def save_model(self, output_dir: str):
        """Save the fine-tuned model"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
