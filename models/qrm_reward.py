# models/qrm_reward.py
# Implementation of the Hugging Face QRM-Llama3.1-8B-v2 reward model
import os
import torch
import logging
from typing import List, Optional, Dict, Any, Tuple
from transformers import AutoModelForSequenceClassification, AutoTokenizer

logger = logging.getLogger(__name__)

class QRMRewardModel:
    """
    Implementation of the QRM-Llama3.1-8B-v2 reward model from Hugging Face
    This model is a Quantile Regression for Distributional Reward Model based on Llama 3.1
    """
    
    def __init__(
        self,
        model_id: str = "nicolinho/QRM-Llama3.1-8B-v2",
        device: Optional[str] = None,
        use_torch_dtype: Optional[torch.dtype] = torch.bfloat16,
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the QRM-Llama3.1-8B-v2 reward model from Hugging Face.
        
        Args:
            model_id: The model ID on Hugging Face (default: "nicolinho/QRM-Llama3.1-8B-v2")
            device: The device to use for inference (default: cuda if available, else cpu)
            use_torch_dtype: The torch dtype to use (default: torch.bfloat16)
            cache_dir: Optional directory to cache the model
        """
        self.model_id = model_id
        
        # Set device
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        # Store other parameters
        self.use_torch_dtype = use_torch_dtype
        self.cache_dir = cache_dir
        
        # Load the model and tokenizer
        logger.info(f"Loading QRM Reward model: {model_id} on {self.device}")
        try:
            self.model = AutoModelForSequenceClassification.from_pretrained(
                model_id, 
                torch_dtype=use_torch_dtype, 
                device_map=self.device, 
                trust_remote_code=True,
                cache_dir=cache_dir
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_id, 
                use_fast=True,
                cache_dir=cache_dir
            )
            logger.info(f"QRM Reward model loaded successfully")
            
            # Get reward attributes
            self.attributes = ['helpsteer-helpfulness', 'helpsteer-correctness', 
                              'helpsteer-coherence', 'helpsteer-complexity', 
                              'helpsteer-verbosity']
            logger.info(f"Model supports reward attributes: {self.attributes}")
            
        except Exception as e:
            logger.error(f"Failed to load QRM Reward model: {str(e)}")
            raise
    
    def format_prompt(self, prompt: str, response: str) -> List[Dict[str, str]]:
        """
        Format prompt and response into the chat template format.
        
        Args:
            prompt: The prompt (user message)
            response: The response to evaluate (assistant message)
            
        Returns:
            List of message dictionaries in the format expected by the tokenizer
        """
        messages = [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response}
        ]
        return messages
    
    def to(self, device):
        """Move the model to a different device properly
        
        Args:
            device: The device to move to, e.g., 'cuda:0', 'cuda:1', etc.
        """
        logger.info(f"Moving QRM reward model from {self.device} to {device}")
        self.device = device
        
        # Move the model to the specified device
        if hasattr(self, 'model'):
            self.model = self.model.to(device)
        
        return self
            
    def get_reward_score(self, prompt: str, response: str) -> float:
        """
        Get reward score for a prompt-response pair.
        
        Args:
            prompt: The prompt (user message)
            response: The response to evaluate (assistant message)
            
        Returns:
            Float reward score (expected value of the reward distribution)
        """
        try:
            # Format into messages
            messages = self.format_prompt(prompt, response)
            
            # Ensure model is on the right device and in eval mode
            self.model.eval()
            device_str = str(self.device)
            
            # Get actual device of model parameters
            for param in self.model.parameters():
                if param.device != self.device:
                    logger.warning(f"Moving model from {param.device} to {device_str}")
                    self.model = self.model.to(self.device)
                break
            
            # Apply tokenization and make sure it's on the right device 
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt"
            )
            
            # Double check device before inference
            input_ids = input_ids.to(self.device)
            
            # Get reward prediction with explicit device management
            with torch.no_grad():
                # Force all computation to happen on the specified device
                with torch.device(self.device):
                    output = self.model(input_ids)
                    # Make sure to fetch from the same device
                    reward = output.score.detach().cpu().float().item()
            
            logger.debug(f"QRM reward score: {reward}")
            return reward
            
        except Exception as e:
            logger.error(f"Error getting reward score: {str(e)}")
            # Return a default value on error
            return 0.0
    
    def get_reward_distribution(self, prompt: str, response: str) -> Dict[str, Any]:
        """
        Get the full reward distribution for a prompt-response pair.
        
        Args:
            prompt: The prompt (user message)
            response: The response to evaluate (assistant message)
            
        Returns:
            Dictionary containing reward score and quantiles
        """
        try:
            # Format into messages
            messages = self.format_prompt(prompt, response)
            
            # Apply tokenization
            input_ids = self.tokenizer.apply_chat_template(
                messages, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get reward prediction
            with torch.no_grad():
                output = self.model(input_ids)
                # Extract the score and quantiles
                reward = output.score.cpu().float().item()
                reward_quantiles = output.reward_quantiles.cpu().float().numpy().tolist()[0]
            
            # Create result dictionary
            quantiles = [0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
            result = {
                "score": reward,
                "quantiles": dict(zip(quantiles, reward_quantiles)),
                "attributes": self.attributes
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error getting reward distribution: {str(e)}")
            # Return a default value on error
            return {"score": 0.0, "quantiles": {}, "attributes": self.attributes}
    
    def batch_get_reward_scores(self, prompts: List[str], responses: List[str]) -> List[float]:
        """
        Get reward scores for batches of prompt-response pairs.
        
        Args:
            prompts: List of prompts
            responses: List of responses to evaluate
            
        Returns:
            List of reward scores
        """
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must be the same")
        
        # Process each prompt-response pair individually
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.get_reward_score(prompt, response)
            rewards.append(reward)
        
        return rewards
    
    def compare(self, prompt: str, response1: str, response2: str) -> Tuple[float, float, int]:
        """
        Compare two responses and return their rewards and which is better.
        Added for compatibility with the DPO trainer.
        
        Args:
            prompt: The prompt text
            response1: First response
            response2: Second response
            
        Returns:
            Tuple of (reward1, reward2, better) where better is 1 if response1 is better, 
            2 if response2 is better, and 0 if they're equal
        """
        # Get rewards for both responses
        reward1 = self.get_reward_score(prompt, response1)
        reward2 = self.get_reward_score(prompt, response2)
        
        # Determine which is better
        if reward1 > reward2:
            better = 1
        elif reward2 > reward1:
            better = 2
        else:
            better = 0
            
        return reward1, reward2, better
