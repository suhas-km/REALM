# gets the Stanford Human Preferences dataset
# data/processors.py
import os
import logging
import torch
import numpy as np
import time
from typing import Dict, Tuple, Optional, Any
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

# Updated import for the NIM reward model
from models.nim_reward import NIMRewardModel
from utils.embedding_utils import LajavanessEmbedding, cosine_similarity

logger = logging.getLogger(__name__)

class SHPDataProcessor:
    """Process the Stanford Human Preferences dataset"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.dataset_name = config["data"]["dataset_name"]
        self.cache_dir = config["data"]["preprocessing"]["cache_dir"]
        self.max_length = config["data"]["preprocessing"]["max_length"]
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
    
    def load_dataset(self, splits=None) -> Dict[str, Any]:
        """Load the SHP dataset and return specified splits
        
        Args:
            splits: List of splits to load (e.g., ['train', 'validation', 'test'])
                   If None, all splits are loaded.
        
        Returns:
            Dictionary mapping split names to datasets
        """
        logger.info(f"Loading dataset: {self.dataset_name}")
        
        try:
            # Set default splits if not provided
            if splits is None:
                splits = ['train', 'validation', 'test']
                
            # Load dataset using Hugging Face datasets
            data_dir = self.config.get("data", {}).get("data_dir", None)
            if data_dir:
                logger.info(f"Loading specific data directory: {data_dir}")
                dataset = load_dataset(self.dataset_name, data_dir=data_dir)
            else:
                dataset = load_dataset(self.dataset_name)
            
            # Process and return the dataset splits
            result = {}
            for split in splits:
                if split in dataset:
                    result[split] = dataset[split]
                    logger.info(f"Loaded {split} split with {len(dataset[split])} examples")
                else:
                    logger.warning(f"Split '{split}' not found in dataset")
            
            # Verify the dataset has the expected SHP fields
            if len(result) > 0:
                sample_split = list(result.keys())[0]
                sample = result[sample_split][0]
                
                # These are the expected fields for SHP dataset from Hugging Face
                expected_fields = [
                    "post_id", "domain", "upvote_ratio", "history", 
                    "c_root_id_A", "c_root_id_B", "created_at_utc_A", "created_at_utc_B",
                    "score_A", "score_B", "human_ref_A", "human_ref_B", 
                    "labels", "seconds_difference", "score_ratio"
                ]
                
                missing_fields = [f for f in expected_fields if f not in sample]
                if missing_fields:
                    logger.warning(f"Dataset is missing expected SHP fields: {missing_fields}")
                    logger.warning(f"Available fields: {list(sample.keys())}")
                else:
                    logger.info("Dataset structure matches SHP format")
            
            return result
            
        except Exception as e:
            logger.error(f"Error loading SHP dataset: {str(e)}")
            raise


class SHPRewardDataset(Dataset):
    """Dataset for training a reward model on SHP data"""
    
    def __init__(
        self, 
        data,
        nim_reward_model: NIMRewardModel,
        embedding_model: LajavanessEmbedding,
        cache_dir: str,
        max_length: int = 1024,
        rebuild_cache: bool = False,
        cache_embeddings: bool = True
    ):
        self.data = data
        self.nim_reward_model = nim_reward_model
        self.embedding_model = embedding_model
        self.cache_dir = cache_dir
        self.max_length = max_length
        self.rebuild_cache = rebuild_cache
        self.cache_embeddings = cache_embeddings
        
        # Create cache directory if it doesn't exist
        os.makedirs(self.cache_dir, exist_ok=True)
        
        # Create embedding cache directory if enabled
        if self.cache_embeddings:
            self.embedding_cache_dir = os.path.join(self.cache_dir, "embeddings")
            os.makedirs(self.embedding_cache_dir, exist_ok=True)
        
        # Initialize in-memory caches
        self.reward_cache = {}
        self.embedding_cache = {}
    
    def _get_cache_path(self, idx: int) -> str:
        """Get the cache file path for a data item"""
        return os.path.join(self.cache_dir, f"item_{idx}.pt")
    
    def _get_embedding_cache_path(self, text_hash: str) -> str:
        """Get the cache file path for an embedding"""
        if self.cache_embeddings:
            return os.path.join(self.embedding_cache_dir, f"emb_{text_hash}.npy")
        return None
    
    def _get_embedding(self, text: str) -> np.ndarray:
        """Get embedding with caching"""
        # Use a deterministic hash for the text
        text_hash = str(hash(text))
        
        # Check in-memory cache first
        if text_hash in self.embedding_cache:
            return self.embedding_cache[text_hash]
        
        # Check disk cache if enabled
        if self.cache_embeddings:
            cache_path = self._get_embedding_cache_path(text_hash)
            if os.path.exists(cache_path) and not self.rebuild_cache:
                try:
                    embedding = np.load(cache_path)
                    # Store in memory cache
                    self.embedding_cache[text_hash] = embedding
                    return embedding
                except Exception as e:
                    logger.warning(f"Failed to load cached embedding: {str(e)}")
        
        # Generate new embedding with exponential backoff for rate limits
        max_retries = 5
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                embedding = self.embedding_model.get_embedding(text)
                
                # Cache the embedding
                if self.cache_embeddings:
                    cache_path = self._get_embedding_cache_path(text_hash)
                    try:
                        np.save(cache_path, embedding)
                    except Exception as e:
                        logger.warning(f"Failed to cache embedding: {str(e)}")
                
                # Store in memory cache (with limited size)
                if len(self.embedding_cache) > 1000:  # Limit cache size
                    self.embedding_cache.pop(next(iter(self.embedding_cache)))
                self.embedding_cache[text_hash] = embedding
                
                return embedding
                
            except Exception as e:
                logger.warning(f"Embedding attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retrying...")
                    time.sleep(wait_time)
        
        # If all attempts fail, return a zero vector
        logger.error(f"All embedding attempts failed for text hash {text_hash}")
        return np.zeros(768)  # Default embedding size
    
    def _get_reward_score(self, prompt: str, response: str) -> float:
        """Get reward score with caching"""
        # Use a deterministic hash for the prompt-response pair
        cache_key = f"{hash(prompt)}_{hash(response)}"
        
        # Check in-memory cache
        if cache_key in self.reward_cache:
            return self.reward_cache[cache_key]
        
        # Get score with exponential backoff for rate limits
        max_retries = 5
        retry_delay = 1.0
        
        for attempt in range(max_retries):
            try:
                score = self.nim_reward_model.get_reward_score(prompt, response)
                
                # Cache the score in memory (with limited size)
                if len(self.reward_cache) > 1000:  # Limit cache size
                    self.reward_cache.pop(next(iter(self.reward_cache)))
                self.reward_cache[cache_key] = score
                
                return score
                
            except Exception as e:
                logger.warning(f"Reward score attempt {attempt+1} failed: {str(e)}")
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)  # Exponential backoff
                    logger.info(f"Waiting {wait_time}s before retrying...")
                    time.sleep(wait_time)
        
        # If all attempts fail, return a default score
        logger.error(f"All reward score attempts failed for cache key {cache_key}")
        return 0.0
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """Get a data item"""
        cache_path = self._get_cache_path(idx)
        
        # Use cached data if available and not rebuilding cache
        if os.path.exists(cache_path) and not self.rebuild_cache:
            try:
                return torch.load(cache_path)
            except Exception as e:
                logger.warning(f"Failed to load cached item {idx}: {str(e)}")
        
        # Process the data item
        try:
            item = self.data[idx]
            
            # Map SHP dataset fields correctly
            # 'history' is the prompt (formerly 'post')
            prompt = item["history"]
            
            # For preferred/dispreferred, use human_ref_A/B based on 'labels'
            # In SHP dataset, 'labels' is 1 when human_ref_A is preferred, 0 when human_ref_B is preferred
            # Ensure labels is treated as an integer
            label_value = int(item["labels"]) if not isinstance(item["labels"], bool) else (1 if item["labels"] else 0)
            
            if label_value == 1:  # human_ref_A is preferred
                chosen = item["human_ref_A"]
                rejected = item["human_ref_B"]
            else:  # human_ref_B is preferred (label_value == 0)
                chosen = item["human_ref_B"] 
                rejected = item["human_ref_A"]
            
            # Truncate text if necessary
            if len(prompt) > self.max_length:
                prompt = prompt[:self.max_length]
            if len(chosen) > self.max_length:
                chosen = chosen[:self.max_length]
            if len(rejected) > self.max_length:
                rejected = rejected[:self.max_length]
            
            # Get reward scores (with caching)
            chosen_llama_score = self._get_reward_score(prompt, chosen)
            rejected_llama_score = self._get_reward_score(prompt, rejected)
            
            # Get embeddings (with caching)
            prompt_embedding = self._get_embedding(prompt)
            chosen_embedding = self._get_embedding(chosen)
            rejected_embedding = self._get_embedding(rejected)
            
            # Calculate similarity scores
            chosen_similarity = cosine_similarity(prompt_embedding, chosen_embedding)
            rejected_similarity = cosine_similarity(prompt_embedding, rejected_embedding)
            
            # Create feature tensors
            chosen_features = torch.tensor([chosen_llama_score, chosen_similarity], dtype=torch.float32)
            rejected_features = torch.tensor([rejected_llama_score, rejected_similarity], dtype=torch.float32)
            
            # Create result dictionary
            result = {
                "chosen_features": chosen_features,
                "rejected_features": rejected_features,
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            }
            
            # Cache the result
            try:
                torch.save(result, cache_path)
            except Exception as e:
                logger.warning(f"Failed to cache item {idx}: {str(e)}")
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing item {idx}: {str(e)}")
            # Return empty tensors as fallback
            return {
                "chosen_features": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "rejected_features": torch.tensor([0.0, 0.0], dtype=torch.float32),
                "prompt": "",
                "chosen": "",
                "rejected": ""
            }


def create_dataloaders(
    config: Dict,
    train_data,
    val_data,
    test_data,
    nim_reward_model: NIMRewardModel,
    embedding_model: LajavanessEmbedding
) -> Tuple[Optional[DataLoader], Optional[DataLoader], Optional[DataLoader]]:
    """Create dataloaders for training, validation, and testing"""
    batch_size = config["data"]["preprocessing"]["batch_size"]
    num_workers = config["data"]["preprocessing"]["num_workers"]
    cache_dir = config["data"]["preprocessing"]["cache_dir"]
    max_length = config["data"]["preprocessing"]["max_length"]
    
    train_dataloader, val_dataloader, test_dataloader = None, None, None
    
    # Create datasets and dataloaders if data is provided
    if train_data is not None:
        logger.info("Creating training dataset...")
        train_dataset = SHPRewardDataset(
            train_data, nim_reward_model, embedding_model, 
            os.path.join(cache_dir, "train"),
            max_length
        )
        
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Training dataloader created with {len(train_dataset)} examples")
    
    if val_data is not None:
        logger.info("Creating validation dataset...")
        val_dataset = SHPRewardDataset(
            val_data, nim_reward_model, embedding_model, 
            os.path.join(cache_dir, "validation"),
            max_length
        )
        
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Validation dataloader created with {len(val_dataset)} examples")
    
    if test_data is not None:
        logger.info("Creating test dataset...")
        test_dataset = SHPRewardDataset(
            test_data, nim_reward_model, embedding_model, 
            os.path.join(cache_dir, "test"),
            max_length
        )
        
        test_dataloader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )
        logger.info(f"Test dataloader created with {len(test_dataset)} examples")
    
    return train_dataloader, val_dataloader, test_dataloader

def safe_initialize_dataset(config, nim_reward_model, embedding_model):
    """
    Progressive initialization to avoid rate limiting and verify functionality
    """
    logger.info("Initializing with safe progressive approach...")
    
    # Load dataset
    data_processor = SHPDataProcessor(config)
    train_data, val_data, test_data = data_processor.load_dataset()
    
    # Start with a very small subset to verify everything works
    test_size = min(5, len(train_data))
    logger.info(f"Testing with {test_size} examples before full initialization")
    
    # Create a small test subset
    if hasattr(train_data, 'select'):
        test_subset = train_data.select(range(test_size))
    else:
        test_subset = [train_data[i] for i in range(test_size)]
    
    # Create a test dataset
    test_cache_dir = os.path.join(config["data"]["preprocessing"]["cache_dir"], "test")
    os.makedirs(test_cache_dir, exist_ok=True)
    
    test_dataset = SHPRewardDataset(
        test_subset, 
        nim_reward_model, 
        embedding_model,
        test_cache_dir,
        config["data"]["preprocessing"]["max_length"]
    )
    
    # Process a few examples to ensure everything works
    logger.info("Testing reward model and embedding API calls...")
    for i in range(min(3, len(test_dataset))):
        logger.info(f"Testing example {i}...")
        example = test_dataset[i]
        logger.info(f"Chosen features: {example['chosen_features']}")
        logger.info(f"Rejected features: {example['rejected_features']}")
    
    logger.info("✓ Initial test successful, proceeding with full dataset")
    return train_data, val_data, test_data
