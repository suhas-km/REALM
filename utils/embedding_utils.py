# utils/embedding_utils.py
# attempt to get the best SOTA embeddings
import logging
import numpy as np
from typing import List
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity as sklearn_cosine_similarity

logger = logging.getLogger(__name__)

class LajavanessEmbedding:
    """Wrapper for the Lajavaness/bilingual-embedding-large model to get embeddings
    Supports specifying a device for multi-GPU distribution"""
    
    def __init__(self, model_name: str = "Lajavaness/bilingual-embedding-large", device=None):
        """
        Initialize the Lajavaness Embedding model.
        
        Args:
            model_name: Hugging Face model ID for the embedding model
            device: PyTorch device to place the model on (e.g., 'cuda:0', 'cuda:1')
        """
        self.model_id = model_name
        self.device = device
        
        # Set device if provided
        device_name = None
        if device is not None:
            if hasattr(device, 'type') and hasattr(device, 'index'):
                # Handle torch.device objects
                if device.type == 'cuda':
                    device_name = f"cuda:{device.index}" if device.index is not None else "cuda:0"
                else:
                    device_name = str(device)
            else:
                # Handle string device specifications
                device_name = str(device)
        
        # Load the model
        try:
            # SentenceTransformer takes device name as a string
            if device_name:
                logger.info(f"Loading embedding model on device: {device_name}")
                self.model = SentenceTransformer(model_name, device=device_name, trust_remote_code=True)
            else:
                self.model = SentenceTransformer(model_name, trust_remote_code=True)
            
            logger.info(f"Initialized Lajavaness Embedding model: {model_name}")
        except Exception as e:
            logger.error(f"Failed to load embedding model {model_name}: {str(e)}")
            raise
    
    def get_embedding(self, text: str, max_retries: int = 3) -> List[float]:
        """Get embedding for text using Lajavaness model"""
        retries = 0
        
        while retries < max_retries:
            try:
                # Get embedding directly using the model
                embedding = self.model.encode(text)
                
                # Return as list
                return embedding.tolist() if isinstance(embedding, np.ndarray) else embedding
                
            except Exception as e:
                logger.warning(f"Embedding attempt {retries+1} failed: {str(e)}")
                retries += 1
                
                if retries < max_retries:
                    import time
                    wait_time = 2 ** retries  # Exponential backoff
                    logger.info(f"Retrying embedding in {wait_time}s...")
                    time.sleep(wait_time)
        
        # Return empty embedding if all retries failed
        logger.warning("All embedding attempts failed, returning zeros")
        return [0.0] * 1024  # Using 1024 as the embedding dimension for this model
    
    def batch_get_embeddings(self, texts: List[str]) -> List[List[float]]:
        """Get embeddings for a batch of texts"""
        try:
            # The SentenceTransformer.encode method can handle batches efficiently
            embeddings = self.model.encode(texts)
            
            # Convert to list of lists if it's a numpy array
            if isinstance(embeddings, np.ndarray):
                return embeddings.tolist()
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {str(e)}")
            # Fall back to individual encoding
            embeddings = []
            for text in texts:
                embedding = self.get_embedding(text)
                embeddings.append(embedding)
            return embeddings


def cosine_similarity(vec1: List[float], vec2: List[float]) -> float:
    """Calculate cosine similarity between two vectors"""
    if not vec1 or not vec2:
        return 0.0
    
    # Convert to numpy arrays
    v1 = np.array(vec1).reshape(1, -1)
    v2 = np.array(vec2).reshape(1, -1)
    
    # Calculate cosine similarity
    similarity = sklearn_cosine_similarity(v1, v2)[0][0]
    return float(similarity)
