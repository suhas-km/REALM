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
from models.linear_reward_model import LinearRewardModel
from training.trainer import RewardModelTrainer
from inference.predictor import RewardPredictor
from rlhf.ppo_huggingface import HuggingFacePPOTrainer
from rlhf.ppo_integration import PPOTrainerWithCustomReward  # Keep for backward compatibility
from rlhf.dpo_huggingface import HuggingFaceDPOTrainer
from rlhf.dpo_integration import DPOTrainerWithCustomReward  # Keep for backward compatibility
from models.nim_reward import NIMRewardModel

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

def create_static_test_dataset():
    """Create a small static dataset for testing that exactly matches the Stanford Human Preferences (SHP) dataset structure"""
    # Define a very small dataset with exactly the same field names as the SHP dataset
    # Structure based on stanfordnlp/SHP from Hugging Face
    
    # Build the samples with matching SHP structure
    train_samples = []
    for i in range(5):
        # Create sample with exact data types matching SHP dataset
        score_a = 300 + i*10  # BIGINT
        score_b = 150 + i*5   # BIGINT
        
        sample = {
            "post_id": f"test{i+1}",                # VARCHAR
            "domain": f"test_train",               # VARCHAR
            "upvote_ratio": 0.98,                  # DOUBLE
            "history": [
                "Explain the concept of reinforcement learning.",
                "What is the capital of France?",
                "How do neural networks learn?",
                "Write a short poem about AI.",
                "Summarize the history of machine learning."
            ][i],                                  # VARCHAR
            "c_root_id_A": f"comment_a_{i+1}",     # VARCHAR
            "c_root_id_B": f"comment_b_{i+1}",     # VARCHAR
            "created_at_utc_A": int(time.time()) - 100,  # BIGINT
            "created_at_utc_B": int(time.time()) - 200,  # BIGINT
            "score_A": score_a,                    # BIGINT
            "score_B": score_b,                    # BIGINT
            "human_ref_A": [
                "Reinforcement learning is a training method based on rewarding desired behaviors and punishing undesired ones. It's different from supervised learning because the system learns from its experiences rather than from a training dataset.",
                "The capital of France is Paris, which is also its largest city and a global center for art, fashion, gastronomy, and culture.",
                "Neural networks learn by adjusting weights between neurons through backpropagation, minimizing the difference between predicted and actual outputs using gradient descent algorithms.",
                "Silicon thoughts in digital streams,\nLearning patterns, chasing dreams.\nArtificial yet so real,\nProcessing more than humans feel.",
                "Machine learning evolved from pattern recognition to deep learning: 1950s perceptrons, 1980s backpropagation, 2010s deep neural networks, and today's transformer models, each advance expanding AI capabilities."
            ][i],                                  # VARCHAR
            "human_ref_B": [
                "Reinforcement learning is when a computer plays games to get better at stuff.",
                "I think the capital of France is Lyon or maybe Marseille.",
                "Neural networks learn by magic and nobody really understands them.",
                "Robots are cool\nAI rules the school\nHumans drool",
                "Machine learning started with computers and then got better over time with more algorithms and stuff."
            ][i],                                  # VARCHAR
            "labels": 1,                           # BIGINT - 1 means A is preferred (human_ref_A)
            "seconds_difference": 100.0,           # DOUBLE
            "score_ratio": float(score_a) / float(score_b)  # DOUBLE - explicit float conversion
        }
        train_samples.append(sample)
    
    # Create validation samples
    val_samples = []
    for i in range(2):
        # Create sample with exact data types matching SHP dataset
        score_a = 280 + i*10  # BIGINT
        score_b = 140 + i*5   # BIGINT
        
        sample = {
            "post_id": f"val{i+1}",                # VARCHAR
            "domain": f"test_validation",         # VARCHAR
            "upvote_ratio": 0.95,                 # DOUBLE
            "history": [
                "Describe quantum computing in simple terms.",
                "What are the benefits of regular exercise?"
            ][i],                                 # VARCHAR
            "c_root_id_A": f"val_comment_a_{i+1}", # VARCHAR
            "c_root_id_B": f"val_comment_b_{i+1}", # VARCHAR
            "created_at_utc_A": int(time.time()) - 150, # BIGINT
            "created_at_utc_B": int(time.time()) - 250, # BIGINT
            "score_A": score_a,                   # BIGINT
            "score_B": score_b,                   # BIGINT
            "human_ref_A": [
                "Quantum computing uses quantum bits or qubits that can exist in multiple states simultaneously, unlike classical bits which can only be 0 or 1. This allows quantum computers to process certain types of problems much faster.",
                "Regular exercise improves cardiovascular health, builds muscle strength, enhances mood through endorphin release, helps maintain healthy weight, reduces disease risk, and improves sleep quality and cognitive function."
            ][i],                                 # VARCHAR
            "human_ref_B": [
                "Quantum computing is very complicated and involves particles and physics that are impossible to understand without advanced degrees.",
                "Exercise is good for you because it makes you less fat and more healthy. You should do it a lot."
            ][i],                                 # VARCHAR
            "labels": 1,                          # BIGINT - 1 means A is preferred
            "seconds_difference": 100.0,          # DOUBLE
            "score_ratio": float(score_a) / float(score_b) # DOUBLE
        }
        val_samples.append(sample)
    
    # Create test samples
    test_samples = []
    for i in range(2):
        # Create sample with exact data types matching SHP dataset
        score_a = 320 + i*10  # BIGINT
        score_b = 160 + i*5   # BIGINT
        
        sample = {
            "post_id": f"test{i+1}",                # VARCHAR
            "domain": f"test_test",                # VARCHAR
            "upvote_ratio": 0.97,                  # DOUBLE
            "history": [
                "How does photosynthesis work?",
                "Give some tips for learning a new language."
            ][i],                                  # VARCHAR
            "c_root_id_A": f"test_comment_a_{i+1}", # VARCHAR
            "c_root_id_B": f"test_comment_b_{i+1}", # VARCHAR
            "created_at_utc_A": int(time.time()) - 120, # BIGINT
            "created_at_utc_B": int(time.time()) - 220, # BIGINT
            "score_A": score_a,                    # BIGINT
            "score_B": score_b,                    # BIGINT
            "human_ref_A": [
                "Photosynthesis is the process where plants convert sunlight, water, and carbon dioxide into glucose and oxygen. The chlorophyll in plant cells captures light energy, which is used to convert CO2 and water into glucose, releasing oxygen as a byproduct.",
                "To learn a new language effectively: practice daily, use spaced repetition for vocabulary, immerse yourself through media, find conversation partners, focus on common words first, use language learning apps, and don't be afraid to make mistakes."
            ][i],                                  # VARCHAR
            "human_ref_B": [
                "Plants make food using sunlight. They take in water and air and turn it into energy.",
                "Just download Duolingo and use it every day. Also maybe watch some foreign movies."
            ][i],                                  # VARCHAR
            "labels": 1,                           # BIGINT - 1 means A is preferred
            "seconds_difference": 100.0,           # DOUBLE
            "score_ratio": float(score_a) / float(score_b) # DOUBLE
        }
        test_samples.append(sample)
        
    # Organize data by split
    test_data = {
        "train": train_samples,
        "validation": val_samples,
        "test": test_samples
    }
    
    return test_data

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Combined Reward Model for RLHF")
    parser.add_argument("--config", type=str, default="config/config.yaml", help="Path to configuration file")
    parser.add_argument("--mode", type=str, choices=["train", "eval", "ppo", "dpo", "predict", "test"], default="train", help="Mode to run")
    parser.add_argument("--model_path", type=str, default=None, help="Path to model checkpoint (required for eval, ppo, dpo, predict)")
    parser.add_argument("--prompt", type=str, default=None, help="Prompt for prediction (used in predict mode)")
    parser.add_argument("--response", type=str, default=None, help="Response for prediction (used in predict mode)")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset for RLHF (used in ppo/dpo mode)")
    args = parser.parse_args()
    
    # Load configuration
    config = load_config(args.config)

    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)
    # Validate environment first
    logger.info("Validating environment...")
    
    # Skip API check if we're in test mode
    if args.mode == "test":
        logger.info("In test mode - skipping API key validation")
        validate_environment(skip_api_check=True)
    else:
        validate_environment()
    
    # Setup logging
    logger.info("Starting Combined Reward Model")
    logger.info(f"Running in {args.mode} mode")
    
    # Set random seed
    set_seed(config["training"]["seed"])
    
    # Initialize models
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    if args.mode == "test":
        # In test mode, create mock objects instead of real API calls
        logger.info("Creating mock NIMRewardModel and LajavanessEmbedding for test mode")
        
        # Create a mock NIMRewardModel class
        class MockNIMRewardModel:
            def __init__(self, **kwargs):
                logger.info("Initialized mock NIMRewardModel")
                
            def get_reward_score(self, prompt, response):
                # Return a random score between 0 and 1 for testing
                return random.random()
        
        # Create a mock LajavanessEmbedding class
        class MockLajavanessEmbedding:
            def __init__(self, **kwargs):
                logger.info("Initialized mock LajavanessEmbedding")
                
            def get_embedding(self, text):
                # Return a random embedding vector (10 dimensions) as a LIST for testing
                # Using a list instead of np.array to avoid boolean context errors
                return [random.random() for _ in range(10)]
        
        # Use the mock classes
        nim_reward_model = MockNIMRewardModel()
        embedding_model = MockLajavanessEmbedding()
    else:
        # Initialize Llama 3.1 Nemotron Reward model via NIM API
        nim_reward_model = NIMRewardModel(
            api_key=config["nim_reward"]["api_key"],
            base_url=config["nim_reward"]["base_url"],
            model_id=config["nim_reward"]["model_id"],
            max_retries=config["nim_reward"]["max_retries"],
            retry_delay=config["nim_reward"]["retry_delay"]
        )
        
        # Initialize Lajavaness Embedding
        embedding_model = LajavanessEmbedding(
            model_id=config["embedding"]["model_id"]
        )
    
    if args.mode == "test":
        # Use the static dataset for testing (already in SHP format)
        logger.info("Using static test dataset")
        dataset = create_static_test_dataset()
        
        # Use the dataset splits directly (already in SHP format with correct field names)
        train_data = dataset["train"]
        val_data = dataset["validation"]
        test_data = dataset["test"]
        
        # Log the dataset structure
        logger.info(f"Test dataset created with {len(train_data)} training examples, "
                   f"{len(val_data)} validation examples, and {len(test_data)} test examples")
        
        # Create a test-specific config with the necessary structure
        data_config = config.get("data", {})
        if "preprocessing" not in data_config:
            data_config["preprocessing"] = {}
            
        # Set default values for required preprocessing parameters
        if "batch_size" not in data_config.get("preprocessing", {}):
            data_config["preprocessing"]["batch_size"] = 4  # Default batch size
        if "num_workers" not in data_config.get("preprocessing", {}):
            data_config["preprocessing"]["num_workers"] = 0  # Default num_workers
        if "cache_dir" not in data_config.get("preprocessing", {}):
            data_config["preprocessing"]["cache_dir"] = "cache"  # Default cache directory
        if "max_length" not in data_config.get("preprocessing", {}):
            data_config["preprocessing"]["max_length"] = 512  # Default max length
            
        # Ensure the config has the data key properly set
        config["data"] = data_config
        
        # Create dataloaders from the static dataset
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
            config=config,
            train_data=train_data, 
            val_data=val_data, 
            test_data=test_data,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model
        )
        
        # Initialize and train the model with the small dataset using robust defaults
        model_config = config.get("model", {})
        model = LinearRewardModel(
            input_dim=model_config.get("input_dim", 2),  # Default input dimension is 2 (llama score + similarity)
            hidden_dims=model_config.get("hidden_dims", [64, 32]),  # Default hidden dimensions
            dropout=model_config.get("dropout", 0.1)  # Default dropout rate
        ).to(device)
        
        # Create a default training config if missing
        training_config = config.get("training", {})
        
        # Add wandb settings with defaults
        if "use_wandb" not in training_config:
            training_config["use_wandb"] = False  # Default to not using wandb for test mode
        
        # Add early stopping settings if missing
        if "early_stopping_patience" not in training_config:
            training_config["early_stopping_patience"] = 5  # Default early stopping patience
        
        # Add logging steps if missing
        if "logging_steps" not in training_config:
            training_config["logging_steps"] = 10  # Default logging steps
        
        # Add evaluation steps if missing
        if "evaluation_steps" not in training_config:
            training_config["evaluation_steps"] = 5  # Evaluate every 5 steps
            
        # Add save steps if missing
        if "save_steps" not in training_config:
            training_config["save_steps"] = 10  # Save every 10 steps
            
        # Add max gradient norm if missing
        if "max_grad_norm" not in training_config:
            training_config["max_grad_norm"] = 1.0  # Default max gradient norm
        
        # Add num_epochs if missing
        if "num_epochs" not in training_config:
            training_config["num_epochs"] = 3  # Default epochs
            
        # Limit to max 3 epochs for testing to avoid long training times
        training_config["num_epochs"] = min(training_config["num_epochs"], 3)
        
        # Create wandb config with defaults
        wandb_config = {
            "project": "realm-test",
            "entity": "test-user",
            "name": f"test-run-{int(time.time())}"
        }
        
        # Create optimizer config with defaults if missing
        optimizer_config = {
            "lr": training_config.get("learning_rate", 1e-4),
            "weight_decay": training_config.get("weight_decay", 0.01)
        }
        
        # Create a complete config with all required sections
        complete_config = {
            "training": training_config,
            "optimizer": optimizer_config,
            "wandb": wandb_config  # Add wandb config section
        }
        
        trainer = RewardModelTrainer(
            model=model,
            config=complete_config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            device=device
        )
        
        # Call train without parameters - it will use the config values
        trainer.train()
        
        # Save the final model (using the model's save method, not the trainer)
        output_path = os.path.join("models", "test_model.pt")
        trainer.model.save(output_path)
        logger.info(f"Test model saved to {output_path}")
        
    elif args.mode == "train":
        # Load and process dataset
        data_processor = SHPDataProcessor(config)
        train_data, val_data, test_data = data_processor.load_dataset()
        
        # Create dataloaders
        train_dataloader, val_dataloader, test_dataloader = create_dataloaders(
            config, train_data, val_data, test_data, nim_reward_model, embedding_model
        )
        
        # Initialize model
        model = LinearRewardModel(
            input_dim=config["model"]["input_dim"],
            hidden_dims=config["model"]["hidden_dims"],
            output_dim=config["model"]["output_dim"],
            dropout=config["model"]["dropout"]
        )
        
        # Initialize trainer
        trainer = RewardModelTrainer(
            model=model,
            config=config,
            train_dataloader=train_dataloader,
            val_dataloader=val_dataloader,
            test_dataloader=test_dataloader,
            device=device
        )
        
        # Train model
        trained_model = trainer.train()
        
        # Save final model
        final_model_path = os.path.join("models", "final_model.pt")
        trained_model.save(final_model_path)
        logger.info(f"Final model saved to {final_model_path}")
        
    elif args.mode == "eval":
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for eval mode")
        
        # Load and process dataset
        data_processor = SHPDataProcessor(config)
        _, _, test_data = data_processor.load_dataset()
        
        # Create test dataloader
        _, _, test_dataloader = create_dataloaders(
            config, None, None, test_data, nim_reward_model, embedding_model
        )
        
        # Load model
        model = LinearRewardModel.load(args.model_path, device=device)
        
        # Initialize trainer
        trainer = RewardModelTrainer(
            model=model,
            config=config,
            train_dataloader=None,
            val_dataloader=None,
            test_dataloader=test_dataloader,
            device=device
        )
        
        # Evaluate model
        test_metrics = trainer.test()
        logger.info(f"Test results: {test_metrics}")
        
    elif args.mode == "predict":
        # Check if model path, prompt, and response are provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for predict mode")
        if args.prompt is None or args.response is None:
            raise ValueError("Prompt and response must be provided for predict mode")
        
        # Initialize predictor
        predictor = RewardPredictor(
            model_path=args.model_path,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model,
            device=device
        )
        
        # Predict reward
        reward = predictor.predict(args.prompt, args.response)
        logger.info(f"Predicted reward: {reward}")
        
    elif args.mode == "ppo":
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for PPO mode")
        
        # Initialize predictor
        predictor = RewardPredictor(
            model_path=args.model_path,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model,
            device=device
        )
        
        # Use Hugging Face's PPO Trainer implementation
        logger.info("Using HuggingFace's PPO implementation")
        ppo_trainer = HuggingFacePPOTrainer(
            config=config,
            reward_predictor=predictor,
            device=device
        )
        
        # Load dataset
        if args.dataset_path:
            import json
            with open(args.dataset_path, "r") as f:
                dataset = json.load(f)
        else:
            # Use our static test dataset for demonstration
            logger.info("Using static test dataset for PPO training")
            test_data = create_static_test_dataset()
            # Extract the prompts/history from each sample in the train dataset
            prompts = [sample["history"] for sample in test_data["train"]]
            dataset = {"prompt": prompts}
        
        # Train with PPO
        ppo_trainer.train(
            dataset=dataset,
            num_epochs=config["rlhf"]["ppo"].get("num_epochs", 1),
            max_steps=config["rlhf"]["ppo"].get("max_steps", 100)
        )
        
        # Save the fine-tuned model
        ppo_trainer.save_model(os.path.join("models", "ppo_finetuned"))
        
    elif args.mode == "dpo":
        # Check if model path is provided
        if args.model_path is None:
            raise ValueError("Model path must be provided for DPO mode")
        
        # Initialize predictor
        predictor = RewardPredictor(
            model_path=args.model_path,
            nim_reward_model=nim_reward_model,
            embedding_model=embedding_model,
            device=device
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
        
        # Load dataset
        if args.dataset_path:
            import json
            with open(args.dataset_path, "r") as f:
                dataset = json.load(f)
            
            # Get num_epochs from config
            num_epochs = config["rlhf"]["dpo"].get("num_epochs", 3)
            
            dpo_trainer.train(
                dataset=dataset,
                num_epochs=num_epochs,
                generate_pairs=True
            )
        else:
            # Use our static test dataset for demonstration
            logger.info("Using static test dataset for DPO training")
            test_data = create_static_test_dataset()
            
            # Convert static dataset to the format expected by DPO trainer
            # Create paired dataset directly with prompt, chosen, and rejected responses
            paired_dataset = []
            for sample in test_data:
                # Extract data from each sample in our SHP-formatted test data
                # 'history' is the prompt, and we use human_ref_A/B based on the 'labels' field
                prompt = sample["history"]
                if sample["labels"] == 1:  # 1 means human_ref_A is preferred
                    chosen = sample["human_ref_A"]
                    rejected = sample["human_ref_B"]
                else:  # 0 means human_ref_B is preferred
                    chosen = sample["human_ref_B"]
                    rejected = sample["human_ref_A"]
                
                paired_dataset.append({
                    "prompt": prompt,
                    "chosen": chosen,
                    "rejected": rejected
                })
            
            # Get num_epochs from config
            num_epochs = config["rlhf"]["dpo"].get("num_epochs", 3)
            
            # Train with paired dataset
            dpo_trainer.train(
                paired_dataset=paired_dataset,
                num_epochs=num_epochs,
                generate_pairs=False  # We already provided the pairs
            )

if __name__ == "__main__":
    main()
