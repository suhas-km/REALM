# combination of linear reward model and Lajavaness embedding
# inference/predictor.py
# this script is only used after we combine the linear reward model with the Lajavaness embedding
import torch
import logging
from typing import Optional
import numpy as np

from models.linear_reward_model import LinearRewardModel
from models.nim_reward import NIMRewardModel
from utils.embedding_utils import LajavanessEmbedding, cosine_similarity

logger = logging.getLogger(__name__)

class RewardPredictor:
    """Predictor class for combined reward model inference"""
    
    def __init__(
        self,
        model_path: str,
        nim_reward_model: NIMRewardModel,
        embedding_model: LajavanessEmbedding,
        device: Optional[torch.device] = None,
        cache_size: int = 1000
    ):
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model
        self.model = LinearRewardModel.load(model_path, device=self.device)
        self.model.eval()
        
        # Models for feature extraction
        self.nim_reward_model = nim_reward_model
        self.embedding_model = embedding_model
        
        # Cache for scores and embeddings
        self.llama_score_cache = {}
        self.embedding_cache = {}
        self.cache_size = cache_size
        
        logger.info(f"Reward predictor initialized with model from {model_path}")
    
    def _get_llama_score(self, prompt: str, response: str) -> float:
        """Get Llama reward score with caching"""
        cache_key = f"{hash(prompt)}_{hash(response)}"
        if cache_key in self.llama_score_cache:
            return self.llama_score_cache[cache_key]
        
        score = self.nim_reward_model.get_reward_score(prompt, response)
        
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
        
        # Create feature tensor
        features = torch.tensor([llama_score, similarity], dtype=torch.float32).to(self.device)
        
        # Get model prediction
        with torch.no_grad():
            reward = self.model(features.unsqueeze(0))
        
        return reward.item()
