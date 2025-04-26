# combination of harmonic mean function and Lajavaness embedding
# inference/predictor.py
import torch
import logging
from typing import Optional, Tuple
import numpy as np

from models.qrm_reward import QRMRewardModel
from utils.embedding_utils import LajavanessEmbedding, cosine_similarity

logger = logging.getLogger(__name__)

def harmonic_blend(sim: float, reward: float, alpha: float = 0.5) -> float:
    """
    Calculate harmonic mean between similarity and reward scores
    
    Args:
        sim: Similarity score between prompt and response
        reward: Reward model score
        alpha: Weight parameter (default: 0.5 for equal weighting)
        
    Returns:
        Harmonic mean of the two scores
    """
    epsilon = 1e-8  # Small value to prevent division by zero
    return 2 * (alpha * sim * (1 - alpha) * reward) / (alpha * sim + (1 - alpha) * reward + epsilon)

class RewardPredictor:
    """Predictor class for harmonic blend reward model inference"""
    
    def __init__(
        self,
        reward_model: QRMRewardModel,
        embedding_model: LajavanessEmbedding,
        device: Optional[torch.device] = None,
        cache_size: int = 1000,
        alpha: float = 0.5,
        model_path: Optional[str] = None  # Kept for backward compatibility but not used
    ):
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Models for feature extraction
        self.reward_model = reward_model
        self.embedding_model = embedding_model
        
        # Cache for scores and embeddings
        self.llama_score_cache = {}
        self.embedding_cache = {}
        self.cache_size = cache_size
        
        # Alpha parameter for harmonic blend
        self.alpha = alpha
        
        logger.info(f"Reward predictor initialized with harmonic blend (alpha={self.alpha})")
        
    def to(self, device):
        """Move all models to the specified device
        
        Args:
            device: The device to move models to (can be string or torch.device)
            
        Returns:
            self for method chaining
        """
        logger.info(f"Moving RewardPredictor from {self.device} to {device}")
        
        # Convert string to torch.device if needed
        if isinstance(device, str):
            device = torch.device(device)
            
        # Update the device attribute
        self.device = device
        
        # Move the reward model if it has a to() method
        if hasattr(self.reward_model, 'to'):
            try:
                self.reward_model.to(device)
                logger.info(f"Successfully moved reward model to {device}")
            except Exception as e:
                logger.error(f"Failed to move reward model to {device}: {e}")
        
        # Move the embedding model if it has a to() method
        if hasattr(self.embedding_model, 'to'):
            try:
                self.embedding_model.to(device)
                logger.info(f"Successfully moved embedding model to {device}")
            except Exception as e:
                logger.error(f"Failed to move embedding model to {device}: {e}")
        
        # Clear caches after changing device
        self.llama_score_cache = {}
        self.embedding_cache = {}
        
        return self
    
    def _get_llama_score(self, prompt: str, response: str) -> float:
        """Get Llama reward score with caching"""
        cache_key = f"{hash(prompt)}_{hash(response)}"
        if cache_key in self.llama_score_cache:
            return self.llama_score_cache[cache_key]
        
        score = self.reward_model.get_reward_score(prompt, response)
        
        # Cache the result
        if len(self.llama_score_cache) >= self.cache_size:
            # Remove a random item if cache is full
            self.llama_score_cache.pop(next(iter(self.llama_score_cache)))
        self.llama_score_cache[cache_key] = score
        
        return score
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        cache_key = hash(text)
        if cache_key in self.embedding_cache:
            return self.embedding_cache[cache_key]
        
        embedding = self.embedding_model.get_embedding(text)
        
        # Cache the result
        if len(self.embedding_cache) >= self.cache_size:
            # Remove a random item if cache is full
            self.embedding_cache.pop(next(iter(self.embedding_cache)))
        self.embedding_cache[cache_key] = embedding
        
        return embedding
    
    def predict(self, prompt: str, response: str) -> float:
        """Predict the reward for a prompt-response pair"""
        # Get Llama score (with caching)
        llama_score = self._get_llama_score(prompt, response)
        
        # Get embeddings (with caching)
        prompt_embedding = self._get_embedding(prompt)
        response_embedding = self._get_embedding(response)
        
        # Calculate similarity score
        similarity = cosine_similarity(prompt_embedding, response_embedding)
        
        # Apply harmonic blend instead of linear model
        reward = harmonic_blend(similarity, llama_score, self.alpha)
        
        return reward
    
    def get_reward_batch(self, prompts: list, responses: list) -> list:
        """Process a batch of prompt-response pairs in parallel
        
        Args:
            prompts: List of prompt texts
            responses: List of response texts
            
        Returns:
            List of reward scores for each prompt-response pair
        """
        if len(prompts) != len(responses):
            raise ValueError(f"Mismatch in batch sizes: {len(prompts)} prompts vs {len(responses)} responses")
        
        # Process examples in parallel
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.predict(prompt, response)
            rewards.append(reward)
            
        return rewards
    
    def compare(self, prompt: str, response1: str, response2: str) -> Tuple[float, float, int]:
        """
        Compare two responses and return their rewards and which is better
        
        Args:
            prompt: The prompt text
            response1: First response
            response2: Second response
            
        Returns:
            Tuple of (reward1, reward2, better) where better is 1 if response1 is better, 
            2 if response2 is better, and 0 if they're equal
        """
        reward1 = self.predict(prompt, response1)
        reward2 = self.predict(prompt, response2)
        
        if reward1 > reward2:
            better = 1
        elif reward2 > reward1:
            better = 2
        else:
            better = 0
            
        return reward1, reward2, better
