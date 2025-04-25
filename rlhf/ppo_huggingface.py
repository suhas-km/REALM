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
from trl import PPOConfig, PPOTrainer, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

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
        
        # Get authentication token from environment
        auth_token = os.environ.get("HUGGINGFACE_TOKEN", None)
            
        # Log authentication status
        if auth_token:
            logger.info("Using Hugging Face authentication token from environment")
            # Set the token in the Hugging Face hub library
            import huggingface_hub
            huggingface_hub.login(token=auth_token, add_to_git_credential=False)
        else:
            logger.warning("No Hugging Face token found in environment - gated models may not be accessible")
            logger.info("Consider running 'huggingface-cli login' or setting HUGGINGFACE_TOKEN environment variable")
        
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
        """Create a proper PPOConfig for TRL 0.16.1 compatible parameters"""
        return PPOConfig(
            # Required PPO parameters
            learning_rate=ppo_config_dict.get("learning_rate", 1.41e-5),
            batch_size=ppo_config_dict.get("batch_size", 8),
            mini_batch_size=ppo_config_dict.get("mini_batch_size", 1),
            
            # Additional PPO parameters with sensible defaults
            ppo_epochs=ppo_config_dict.get("ppo_epochs", 4),  # Number of PPO optimization steps per batch
            gradient_accumulation_steps=ppo_config_dict.get("gradient_accumulation_steps", 1),
            optimize_device_placement=True,  # Let TRL optimize tensor placement
            log_with=None,  # Disable tracking
            accelerator_kwargs={"logging_dir": os.path.join(self.checkpoint_dir, "logs")},
            project_kwargs={"logging_dir": os.path.join(self.checkpoint_dir, "logs")},
            tracker_project_name="ppo_finetune",
            remove_unused_columns=False,  # Keep all columns for logging
            
            # KL divergence control
            kl_penalty=ppo_config_dict.get("kl_penalty", 0.2),
            init_kl_coef=ppo_config_dict.get("init_kl_coef", 0.2),
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
        logger.info(f"Starting PPO training with HuggingFace TRL 0.16.1 for {num_epochs} epochs, max {max_steps if max_steps else 'all'} steps")
        
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
        
        # Create value head model for PPO (required for TRL 0.16.1)
        logger.info("Converting model to AutoModelForCausalLMWithValueHead for TRL 0.16.1 compatibility")
        # First ensure model is in evaluation mode before conversion
        self.model.eval()
        
        # Get model name from config
        model_name = self.config["rlhf"]["ppo"]["model_name"]
        
        # Get authentication token from environment
        auth_token = os.environ.get("HUGGINGFACE_TOKEN", None)
        
        # Convert standard model to one with value head
        ppo_model = AutoModelForCausalLMWithValueHead.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if "cuda" in str(self.device) else torch.float32,
            device_map="auto" if "cuda" in str(self.device) else None,
            token=auth_token,
            state_dict=self.model.state_dict(),  # Transfer existing weights
        )
        ppo_model.to(self.device)
        logger.info("Value head model created successfully")
        
        # Initialize PPOTrainer with TRL 0.16.1 API
        logger.info("Initializing PPOTrainer with TRL 0.16.1 API")
        response_length_sampler = LengthSampler(self.max_length // 2, self.max_length)  # Sample response lengths between half and full max length
        
        ppo_trainer = PPOTrainer(
            model=ppo_model,  # Model with value head
            tokenizer=self.tokenizer,
            dataset=hf_dataset,
            data_collator=None,  # Use default collator
            ppo_config=self.ppo_config,
            response_length_sampler=response_length_sampler,
        )
        
        logger.info("PPOTrainer successfully initialized with TRL 0.16.1 API")
        
        # Train for the specified number of epochs
        for epoch in range(num_epochs):
            logger.info(f"Epoch {epoch+1}/{num_epochs}")
            
            # Initialize metrics tracking
            epoch_rewards = []
            epoch_kl_divs = []
            epoch_losses = []
            
            # Use TRL PPOTrainer to train on the dataset
            logger.info(f"Training PPOTrainer on {len(hf_dataset)} examples for epoch {epoch+1}/{num_epochs}")
            
            # Configure steps: either train on entire dataset or limit to max_steps
            total_steps = len(hf_dataset) // self.ppo_config.batch_size
            if max_steps and max_steps > 0 and max_steps < total_steps:
                steps_to_run = max_steps
            else:
                steps_to_run = total_steps
            
            logger.info(f"Running PPO for {steps_to_run} steps in epoch {epoch+1}")
            
            # Train for this epoch
            step_results = []
            for step, batch in enumerate(ppo_trainer.dataloader):
                if step >= steps_to_run:
                    break
                    
                # Get query texts for logging
                queries = batch["query"]
                
                # Generate model responses
                response_tensors = ppo_trainer.generate(
                    batch["query"],
                    return_prompt=False,
                    **self.generation_kwargs
                )
                
                # Decode responses for reward computation and logging
                responses = [self.tokenizer.decode(r, skip_special_tokens=True) for r in response_tensors]
                
                # Compute rewards
                rewards = torch.tensor(self._reward_fn(queries, responses)).to(self.device)
                
                # Run PPO step
                stats = ppo_trainer.step(queries, response_tensors, rewards)
                step_results.append(stats)
                
                # Log example and rewards
                if len(queries) > 0:
                    logger.info(f"Step {step+1}/{steps_to_run}, Example query: {queries[0][:50]}...")
                    logger.info(f"Example response: {responses[0][:50]}...")
                    logger.info(f"Example reward: {rewards[0].item() if len(rewards) > 0 else 'N/A'}")
                    logger.info(f"Mean batch reward: {rewards.mean().item():.4f}")
                    epoch_rewards.append(rewards.mean().item())
                    
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
