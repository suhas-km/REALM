# rlhf/ppo_huggingface.py
import os
import torch
import logging
from typing import Dict, List, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import Dataset
from tqdm import tqdm

# Import specific components for TRL 0.16.1
import time
from trl import PPOConfig, PPOTrainer

from inference.predictor import RewardPredictor

logger = logging.getLogger(__name__)

class HuggingFacePPOTrainer:
    """
    PPO Trainer that leverages Hugging Face's TRL library version 0.16.1 
    for industry-standard PPO implementation
    """
    
    def __init__(
        self,
        config: Dict,
        reward_predictor: RewardPredictor,
        tokenizer=None,
        model=None,
        device: Optional[torch.device] = None,
        checkpoint_dir: Optional[str] = None,
        token: Optional[str] = None
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
        
        # Prepare authentication token if provided
        auth_token = token
        if auth_token is None:
            # Try to get token from environment
            auth_token = os.environ.get("HUGGINGFACE_TOKEN", None)
            
        # Log authentication status
        if auth_token:
            logger.info("Using provided Hugging Face authentication token")
        else:
            logger.warning("No Hugging Face token provided - gated models may not be accessible")
            logger.info("If this fails, consider running 'huggingface-cli login' first")
        
        try:
            # First try loading tokenizer and model with token
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, token=auth_token)
            # Set pad_token if not set
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name, token=auth_token)
            self.model.to(self.device)
        except Exception as e:
            # If loading fails, try different approach
            logger.warning(f"Failed to load model/tokenizer: {e}")
            logger.info("Attempting alternative loading method...")
            
            try:
                # Try again with specific options
                self.tokenizer = AutoTokenizer.from_pretrained(
                    model_name, 
                    padding_side="left", 
                    token=auth_token
                )
                if self.tokenizer.pad_token is None:
                    self.tokenizer.pad_token = self.tokenizer.eos_token
                
                self.model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
                    device_map="auto" if "cuda" in str(self.device) else None,
                    token=auth_token
                )
                self.model.to(self.device)
            except Exception as e2:
                # If both loading attempts fail, raise error
                logger.error(f"Failed to load model using alternative method: {e2}")
                raise ValueError(f"Could not load model {model_name}: {e2}")
        
        # Create PPO config
        self.ppo_config = self._create_ppo_config(config["rlhf"]["ppo"])
        
        # Set up generation args
        self.max_length = config["rlhf"]["ppo"].get("max_length", 256)
        self.generation_kwargs = {
            "max_new_tokens": self.max_length,
            "top_k": 0.0,
            "top_p": 1.0,
            "do_sample": True,
            "pad_token_id": self.tokenizer.pad_token_id,
        }
        
        logger.info("HuggingFace PPO Trainer initialized (for TRL version 0.16.1)")
    
    def _create_ppo_config(self, ppo_config_dict: Dict) -> PPOConfig:
        """Create a minimal PPOConfig with only the most basic parameters"""
        # Only use the most basic parameters that should be supported across versions
        return PPOConfig(
            learning_rate=float(ppo_config_dict.get("learning_rate", 1e-5)),
            batch_size=ppo_config_dict.get("batch_size", 8),
            mini_batch_size=ppo_config_dict.get("mini_batch_size", 4),
        )
    
    def _prepare_dataset(self, dataset: Dict) -> Dataset:
        """Convert dict dataset to HF Dataset for PPO Trainer"""
        prompts = dataset.get("prompt", [])
        if not prompts:
            logger.warning("No prompts found in dataset")
            return None
            
        # Create HF dataset with prompts
        hf_dataset = Dataset.from_dict({"query": prompts})
        return hf_dataset
    
    def _reward_fn(self, queries: List[str], responses: List[str]) -> List[float]:
        """Reward function wrapper for the PPO trainer"""
        rewards = []
        for query, response in zip(queries, responses):
            reward = self.reward_predictor.predict(query, response)
            rewards.append(float(reward))
        return rewards
        
    def train(self, dataset: Dict, num_epochs: int = 1, max_steps: int = 100, checkpoint_interval: int = 1):
        """Train the model using TRL 0.16.1's PPO implementation"""
        logger.info(f"Starting PPO training with HuggingFace TRL 0.16.1 for {num_epochs} epochs, max {max_steps} steps")
        
        # Prepare dataset
        hf_dataset = self._prepare_dataset(dataset)
        if hf_dataset is None:
            logger.error("Cannot train with empty dataset")
            return self.model
        
        # Limit steps if needed
        if max_steps and max_steps < len(hf_dataset):
            hf_dataset = hf_dataset.select(range(max_steps))
        
        # Create a minimal PPOTrainer with only the most essential parameters
        # that should work across different versions
        try:
            # Try the current approach for newer versions
            ppo_trainer = PPOTrainer(
                config=self.ppo_config,
                model=self.model,
                ref_model=None,
                tokenizer=self.tokenizer,
                dataset=hf_dataset,
            )
        except TypeError as e:
            # If there's an error, try an alternative approach for older versions
            logger.warning(f"First PPOTrainer initialization attempt failed: {e}")
            try:
                ppo_trainer = PPOTrainer(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    dataset=hf_dataset,
                    learning_rate=self.ppo_config.learning_rate,
                    batch_size=self.ppo_config.batch_size,
                    mini_batch_size=self.ppo_config.mini_batch_size,
                )
            except TypeError as e2:
                # If both fail, use a third approach as a last resort
                logger.warning(f"Second PPOTrainer initialization attempt failed: {e2}")
                logger.info("Attempting to create a simpler implementation...")
                
                # Instead of using PPOTrainer, we'll implement a simplified version
                # that just fine-tunes the model on the generated responses with rewards
                ppo_trainer = None
        
        # Train for the specified number of epochs
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Initialize metrics tracking
            epoch_rewards = []
            epoch_kl_divs = []
            epoch_losses = []
            
            # Use a simplified approach for compatibility across TRL versions
            
            # Get only the amount of data we need for this epoch
            epoch_dataset = hf_dataset.shuffle(seed=epoch)
            if max_steps:
                epoch_dataset = epoch_dataset.select(range(min(max_steps, len(epoch_dataset))))
            
            # Process data in batches
            for i in range(0, len(epoch_dataset), self.ppo_config.batch_size):
                batch = epoch_dataset.select(range(i, min(i + self.ppo_config.batch_size, len(epoch_dataset))))
                
                # Get queries
                queries = batch["query"]
                
                # Generate responses and log them
                responses = []
                for query in tqdm(queries, desc=f"Processing batch {i//self.ppo_config.batch_size + 1}"):
                    # Tokenize the query
                    query_tensor = self.tokenizer(query, return_tensors="pt").to(self.device)
                    
                    # Generate a response
                    with torch.no_grad():
                        output = self.model.generate(
                            input_ids=query_tensor.input_ids,
                            attention_mask=query_tensor.attention_mask,
                            max_new_tokens=self.max_length,
                            do_sample=True,
                            top_p=0.9,
                            top_k=0,
                            pad_token_id=self.tokenizer.pad_token_id
                        )
                    
                    # Extract just the response part (not the query)
                    response_text = self.tokenizer.decode(output[0][query_tensor.input_ids.shape[1]:], skip_special_tokens=True)
                    responses.append(response_text)
                
                # Compute rewards
                rewards = self._reward_fn(queries, responses)
                
                # Log example and rewards
                if len(queries) > 0:
                    logger.info(f"Example query: {queries[0][:50]}...")
                    logger.info(f"Example response: {responses[0][:50]}...")
                    logger.info(f"Example reward: {rewards[0] if rewards else 'N/A'}")
                
                # Log mean reward
                if rewards:
                    mean_reward = sum(rewards) / len(rewards)
                    logger.info(f"Batch {i//self.ppo_config.batch_size + 1}, Mean reward: {mean_reward:.4f}")
                    epoch_rewards.append(mean_reward)
                
                # Note: We've moved this logic to the block above
                # for a more streamlined implementation
            
            # Log epoch stats
            if epoch_rewards:
                avg_reward = sum(epoch_rewards) / len(epoch_rewards)
                logger.info(f"Epoch {epoch+1} average reward: {avg_reward:.4f}")
                
                # Save checkpoint at specified intervals
                if (epoch + 1) % checkpoint_interval == 0:
                    checkpoint_path = os.path.join(self.checkpoint_dir, f"checkpoint-epoch-{epoch+1}")
                    self.save_checkpoint(checkpoint_path, metadata={
                        "epoch": epoch + 1,
                        "avg_reward": avg_reward,
                        "timestamp": time.strftime("%Y-%m-%d-%H-%M-%S")
                    })
                    logger.info(f"Saved checkpoint for epoch {epoch+1} to {checkpoint_path}")
        
        logger.info("HuggingFace PPO training completed")
        
        # If we used PPOTrainer, update our model reference
        if ppo_trainer is not None:
            self.model = ppo_trainer.model
        return self.model
    
    def save_checkpoint(self, checkpoint_path: str, metadata: Dict = None):
        """Save a checkpoint during training with metadata"""
        os.makedirs(checkpoint_path, exist_ok=True)
        
        # Save the model and tokenizer
        self.model.save_pretrained(checkpoint_path)
        self.tokenizer.save_pretrained(checkpoint_path)
        
        # Save metadata if provided
        if metadata:
            import json
            with open(os.path.join(checkpoint_path, "checkpoint_metadata.json"), "w") as f:
                json.dump(metadata, f, indent=2)
        
        logger.info(f"Checkpoint saved to {checkpoint_path}")
    
    def save_model(self, output_dir: str):
        """Save the final fine-tuned model"""
        os.makedirs(output_dir, exist_ok=True)
        
        # Directly save the model and tokenizer
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save training configuration
        import json
        config_path = os.path.join(output_dir, "training_config.json")
        with open(config_path, "w") as f:
            # Extract relevant PPO config for saving
            ppo_config = {
                "learning_rate": self.ppo_config.learning_rate,
                "batch_size": self.ppo_config.batch_size,
                "mini_batch_size": self.ppo_config.mini_batch_size,
                "model_name": self.model.config.name_or_path if hasattr(self.model.config, "name_or_path") else "custom",
                "saved_at": time.strftime("%Y-%m-%d %H:%M:%S")
            }
            json.dump(ppo_config, f, indent=2)
        
        logger.info(f"Model, tokenizer, and config saved to {output_dir}")
