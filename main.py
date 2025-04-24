# main.py
# 

import os
import argparse
import logging
import yaml
import torch
import random
import numpy as np
import time
from typing import Dict, Any
from dotenv import load_dotenv

from utils.validation import validate_environment
from utils.embedding_utils import LajavanessEmbedding
from data.processors import SHPDataProcessor, create_dataloaders
from inference.predictor import RewardPredictor
from rlhf.ppo_huggingface import HuggingFacePPOTrainer
from rlhf.ppo_integration import PPOTrainerWithCustomReward  # Keep for backward compatibility
from rlhf.dpo_huggingface import HuggingFaceDPOTrainer
from rlhf.dpo_integration import DPOTrainerWithCustomReward  # Keep for backward compatibility
from models.qrm_reward import QRMRewardModel

# Load environment variables
load_dotenv()

def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    # Replace environment variables in config
    for section in config:
        if isinstance(config[section], dict):
            for key, value in config[section].items():
                if isinstance(value, str) and value.startswith("${") and value.endswith("}"):
                    env_var = value[2:-1]
                    config[section][key] = os.environ.get(env_var, "")
    
    return config

def set_seed(seed: int) -> None:
    """Set random seed for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Combined Reward Model for RLHF")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["predict", "ppo", "dpo"], default="predict", help="Operation mode")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (required for PPO and DPO modes)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for prediction (used in predict mode)")
    parser.add_argument("--response", type=str, default=None, help="Response for prediction (used in predict mode)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset for RLHF (used in ppo/dpo mode)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the fine-tuned model (defaults to 'models/ppo_finetuned' for PPO)")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save checkpoints every N epochs (PPO mode only)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Validate environment first
    logger.info("Validating environment...")
    
    # Validate environment
    validate_environment()
    
    # Setup logging
    logger.info("Starting Combined Reward Model")
    logger.info(f"Running in {args.mode} mode")
    
    # Set random seed
    set_seed(config["general"]["seed"])
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Initialize QRM-Llama3.1-8B-v2 Reward model from Hugging Face
    reward_model = QRMRewardModel(
        model_id=config.get("qrm_reward", {}).get("model_id", "nicolinho/QRM-Llama3.1-8B-v2"),
        device=device
    )
    
    # Initialize Lajavaness Embedding
    embedding_model = LajavanessEmbedding(
        model_id=config["embedding"]["model_id"]
    )
    
    if args.mode == "predict":
        # Check if response is provided
        # No model_path needed since we're using harmonic_blend
        if args.response is None:
            raise ValueError("Response must be provided for predict mode")
        
        # Initialize predictor with alpha from config
        alpha = config["reward"]["alpha"]
        predictor = RewardPredictor(
            reward_model=reward_model,
            embedding_model=embedding_model,
            device=device,
            alpha=alpha
        )
        
        # Predict reward
        prompt = args.prompt or "Tell me about yourself."
        reward = predictor.predict(prompt, args.response)
        logger.info(f"Predicted reward: {reward}")
        
    elif args.mode == "ppo":
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for PPO mode")
        
        # Initialize predictor with alpha from config
        alpha = config["reward"]["alpha"]
        predictor = RewardPredictor(
            reward_model=reward_model,
            embedding_model=embedding_model,
            device=device,
            alpha=alpha
        )
        
        # Set up checkpoint and output directories
        checkpoint_dir = os.path.join("models", "ppo_checkpoints", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        output_dir = args.output_dir or os.path.join("models", f"ppo_finetuned_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Log checkpoint configuration
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        logger.info(f"Final model will be saved to: {output_dir}")
        
        # Use Hugging Face's PPO Trainer implementation
        logger.info("Using HuggingFace's PPO implementation")
        ppo_trainer = HuggingFacePPOTrainer(
            config=config,
            reward_predictor=predictor,
            device=device,
            checkpoint_dir=checkpoint_dir
        )
        
        # Load and process SHP dataset using the processors.py
        logger.info("Loading SHP dataset for PPO training using SHPDataProcessor")
        data_processor = SHPDataProcessor(config)
        train_data, val_data, test_data = data_processor.load_dataset()
        
        # Check if we have data
        if train_data is None or len(train_data) == 0:
            raise ValueError("Failed to load training data for PPO mode")
            
        # Create dataset in format expected by PPO trainer
        logger.info(f"Preparing dataset with {len(train_data)} examples for PPO training")
        
        # Extract prompts (history field) from SHP dataset
        prompts = [item["history"] for item in train_data]
        dataset = {"prompt": prompts}
        
        # Train with PPO
        ppo_trainer.train(
            dataset=dataset,
            num_epochs=config["rlhf"]["ppo"].get("num_epochs", 1),
            max_steps=config["rlhf"]["ppo"].get("max_steps", 100),
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Save the fine-tuned model
        logger.info(f"Saving final fine-tuned model to {output_dir}")
        ppo_trainer.save_model(output_dir)
        
        # Create a symlink to the latest model for convenience
        latest_link_path = os.path.join("models", "ppo_finetuned_latest")
        try:
            if os.path.exists(latest_link_path) or os.path.islink(latest_link_path):
                os.remove(latest_link_path)
            os.symlink(output_dir, latest_link_path, target_is_directory=True)
            logger.info(f"Created symlink 'ppo_finetuned_latest' pointing to {output_dir}")
        except Exception as e:
            logger.warning(f"Could not create symlink to latest model: {e}")
        
    elif args.mode == "dpo":
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for DPO mode")
        
        # Initialize predictor with alpha from config
        alpha = config["reward"]["alpha"]
        predictor = RewardPredictor(
            reward_model=reward_model,
            embedding_model=embedding_model,
            device=device,
            alpha=alpha
        )
        
        # Use Hugging Face's DPO Trainer implementation
        logger.info("Using HuggingFace's DPO implementation")
        
        # Check for PEFT/LoRA flag
        use_peft = config["rlhf"]["dpo"].get("use_peft", False)
        
        dpo_trainer = HuggingFaceDPOTrainer(
            config=config,
            reward_predictor=predictor,
            device=device,
            use_peft=use_peft
        )
        
        # Load and process SHP dataset using the processors.py
        logger.info("Loading SHP dataset for DPO training using SHPDataProcessor")
        data_processor = SHPDataProcessor(config)
        train_data, val_data, test_data = data_processor.load_dataset()
        
        # Check if we have data
        if train_data is None or len(train_data) == 0:
            raise ValueError("Failed to load training data for DPO mode")
            
        # Create dataset in format expected by DPO trainer
        logger.info(f"Preparing dataset with {len(train_data)} examples for DPO training")
        
        # Extract prompt/chosen/rejected from SHP dataset
        paired_dataset = []
        for item in train_data:
            # Extract data based on the labels field
            prompt = item["history"]
            if item["labels"] == 1:  # 1 means human_ref_A is preferred
                chosen = item["human_ref_A"]
                rejected = item["human_ref_B"]
            else:  # 0 means human_ref_B is preferred
                chosen = item["human_ref_B"]
                rejected = item["human_ref_A"]
            
            paired_dataset.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
        
        # Get num_epochs from config
        num_epochs = config["rlhf"]["dpo"].get("num_epochs", 3)
        
        # Train with DPO on the paired dataset
        dpo_trainer.train(
            paired_dataset=paired_dataset,
            num_epochs=num_epochs,
            generate_pairs=False  # We already created the pairs
        )
        
        # Save the fine-tuned model
        dpo_trainer.save_model(os.path.join("models", "dpo_finetuned"))
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
