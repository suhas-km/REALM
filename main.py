# main.py

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
from rlhf.dpo_huggingface import HuggingFaceDPOTrainer
from models.qrm_reward import QRMRewardModel
from data.truthfulness_dataset import TruthfulQADataset, evaluate_model_truthfulness
from transformers import AutoTokenizer, AutoModelForCausalLM

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
    parser.add_argument("--mode", type=str, choices=["predict", "ppo", "dpo", "qrm_ppo", "qrm_dpo", "evaluate"], default="predict", help="Operation mode")
    parser.add_argument("--prompt", type=str, default=None, help="Input prompt for model")
    parser.add_argument("--response", type=str, default=None, help="Response to evaluate (for predict mode)")
    # Authentication is handled via huggingface-cli login
    parser.add_argument("--max_samples", type=int, default=1000, help="Maximum number of training samples to use")
    parser.add_argument("--output_file", type=str, default=None, help="Output file for evaluation results")
    parser.add_argument("--dataset_path", type=str, default=None, help="Path to dataset for RLHF (used in ppo/dpo mode)")
    parser.add_argument("--output_dir", type=str, default=None, help="Directory to save the fine-tuned model (defaults to 'models/ppo_finetuned' for PPO)")
    parser.add_argument("--checkpoint_interval", type=int, default=1, help="Save checkpoints every N epochs (PPO mode only)")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size (overrides config.yaml batch_size)")
    parser.add_argument("--model_type", type=str, default="base", help="Model type for evaluation mode")
    parser.add_argument("--max_new_tokens", type=int, default=128, help="Maximum new tokens for generation in evaluate mode")
    parser.add_argument("--multi_gpu", action="store_true", help="Enable multi-GPU training with model parallelism")
    parser.add_argument("--policy_gpus", type=str, default="0,1", help="Comma-separated list of GPU IDs for policy model")
    parser.add_argument("--ref_gpus", type=str, default="2,3", help="Comma-separated list of GPU IDs for reference model")
    parser.add_argument("--reward_gpu", type=int, default=4, help="GPU ID for reward model")
    parser.add_argument("--embedding_gpu", type=int, default=5, help="GPU ID for embedding model")
    parser.add_argument("--mixed_precision", action="store_true", default=True, help="Enable mixed precision training")
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
    
    # Set device and GPU configuration
    num_gpus = torch.cuda.device_count()
    if args.multi_gpu and num_gpus > 1:
        logger.info(f"Multi-GPU training enabled with {num_gpus} GPUs available")
        
        # Parse GPU ID lists
        policy_gpus = [int(gpu) for gpu in args.policy_gpus.split(',')]
        ref_gpus = [int(gpu) for gpu in args.ref_gpus.split(',')]
        reward_gpu = args.reward_gpu
        embedding_gpu = args.embedding_gpu
        
        # Validate GPU IDs
        all_gpus = policy_gpus + ref_gpus + [reward_gpu, embedding_gpu]
        for gpu in all_gpus:
            if gpu >= num_gpus:
                logger.warning(f"GPU ID {gpu} is not available (only {num_gpus} GPUs found). Adjusting to use available GPUs.")
                # Will be auto-adjusted by the trainer
        
        # Log GPU allocation strategy
        logger.info(f"GPU Allocation Strategy:")
        logger.info(f"  Policy Model: GPUs {policy_gpus}")
        logger.info(f"  Reference Model: GPUs {ref_gpus}")
        logger.info(f"  Reward Model: GPU {reward_gpu}")
        logger.info(f"  Embedding Model: GPU {embedding_gpu}")
        logger.info(f"  Mixed Precision: {args.mixed_precision}")
    else:
        policy_gpus = [0]
        ref_gpus = [0]
        reward_gpu = 0
        embedding_gpu = 0
        logger.info(f"Using single GPU mode on device: {device}")
    
    # Initialize QRM-Llama3.1-8B-v2 Reward model from Hugging Face
    reward_model = QRMRewardModel(
        model_id=config.get("qrm_reward", {}).get("model_id", "nicolinho/QRM-Llama3.1-8B-v2"),
        device=device
    )
    
    # Initialize embedding model for semantic similarity on dedicated GPU
    embedding_device = torch.device(f"cuda:{embedding_gpu}" if torch.cuda.is_available() else "cpu")
    logger.info(f"Loading embedding model on {embedding_device}")
    embedding_model = LajavanessEmbedding(
        model_name=config["reward_model"]["embedding_model"],
        device=embedding_device
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
        # Initialize predictor with alpha from config
        alpha = config["reward"]["alpha"]
        predictor = RewardPredictor(
            reward_model=reward_model,
            embedding_model=embedding_model,
            device=device,
            alpha=alpha
        )
        
        # Set up checkpoint and output directories
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = os.path.join("models", "ppo_checkpoints", f"run_{timestamp}")
        output_dir = os.path.join("models", f"ppo_finetuned_{timestamp}")
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        logger.info(f"Final model will be saved to: {output_dir}")
        
        # Get model path from config
        model_path = config["model"]["model_path"]
        logger.info(f"Using model path from config: {model_path}")
        
        # Add mixed precision setting to config
        if "mixed_precision" not in config["rlhf"]["ppo"]:
            config["rlhf"]["ppo"]["mixed_precision"] = args.mixed_precision
        
        # Initialize the PPO trainer with model parallelism support
        logger.info("Using HuggingFace's PPO implementation with model parallelism")
        primary_device = torch.device(f"cuda:{policy_gpus[0]}" if torch.cuda.is_available() else "cpu")
        
        ppo_trainer = HuggingFacePPOTrainer(
            config=config,
            reward_predictor=predictor,
            checkpoint_dir=checkpoint_dir,
            device=primary_device,
            policy_model_devices=policy_gpus,
            ref_model_devices=ref_gpus,
            reward_model_device=reward_gpu,
            embedding_device=embedding_gpu
        )
        
        # Load and process SHP dataset using the processors.py
        logger.info("Loading SHP dataset for PPO training using SHPDataProcessor")
        data_processor = SHPDataProcessor(config)
        train_data, val_data, test_data = data_processor.load_dataset()
        
        # Check if we have data
        if train_data is None or len(train_data) == 0:
            raise ValueError("Failed to load training data for PPO mode")
            
        # Limit samples if max_samples is provided
        if args.max_samples and args.max_samples > 0 and args.max_samples < len(train_data):
            logger.info(f"Limiting training data to {args.max_samples} examples (from {len(train_data)})")
            train_data = train_data.select(range(args.max_samples))
        else:
            # If no max_samples provided, use a substantial portion of the dataset
            # Use at least 1000 samples for PPO training
            logger.info(f"No max_samples specified, using entire dataset with {len(train_data)} examples")
            
            # If config has a very small number of steps, increase it for meaningful training
            min_steps = 1000
            current_steps = config["rlhf"]["ppo"].get("max_steps", 0)
            if current_steps < 100:
                config["rlhf"]["ppo"]["max_steps"] = min_steps
                logger.info(f"Increasing max_steps to {min_steps} for meaningful training")
            
        # Create dataset in format expected by PPO trainer
        logger.info(f"Preparing dataset with {len(train_data)} examples for PPO training")
        
        # Extract prompts (history field) from SHP dataset
        try:
            # Try accessing as dictionary with expected SHP fields
            logger.info("Attempting to extract prompts using standard SHP format")
            prompts = []
            for item in train_data:
                if isinstance(item, dict) and "history" in item:
                    prompts.append(item["history"])
                elif isinstance(item, str):
                    # If item is already a string, use it directly
                    prompts.append(item)
                else:
                    # Try to get the first field in the dictionary
                    if isinstance(item, dict) and len(item) > 0:
                        first_key = next(iter(item))
                        prompts.append(item[first_key])
                    else:
                        logger.warning(f"Could not extract prompt from item: {item}")
                        prompts.append("")
            
            logger.info(f"Successfully extracted {len(prompts)} prompts")
            dataset = {"prompt": prompts}
            
        except Exception as e:
            logger.error(f"Error extracting prompts: {e}")
            logger.info("Falling back to using the entire dataset as prompts")
            
            # As a fallback, if train_data is already a list of strings, use it directly
            if all(isinstance(item, str) for item in train_data):
                dataset = {"prompt": train_data}
            else:
                raise ValueError(f"Dataset format not supported: {type(train_data)}")
        
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
        # Initialize predictor with alpha from config
        alpha = config["reward"]["alpha"]
        predictor = RewardPredictor(
            reward_model=reward_model,
            embedding_model=embedding_model,
            device=device,
            alpha=alpha
        )
        
        # Use model from config
        logger.info(f"Using model path from config: {config['rlhf']['dpo']['model_name']}")
        
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
        try:
            # Try accessing as dictionary with expected SHP fields
            logger.info("Attempting to extract prompts and responses using standard SHP format")
            paired_dataset = []
            
            for item in train_data:
                if isinstance(item, dict) and "history" in item and "human_ref_A" in item and "human_ref_B" in item and "labels" in item:
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
                elif isinstance(item, dict) and len(item) >= 3:
                    # Try to extract from generic dictionary with at least 3 fields
                    keys = list(item.keys())
                    if len(keys) >= 3:
                        prompt_key = keys[0]
                        chosen_key = keys[1]
                        rejected_key = keys[2]
                        
                        paired_dataset.append({
                            "prompt": item[prompt_key],
                            "chosen": item[chosen_key],
                            "rejected": item[rejected_key]
                        })
                else:
                    logger.warning(f"Could not process item format: {type(item)}")
            
            if not paired_dataset:
                raise ValueError("Failed to extract paired dataset using standard format")
                
            logger.info(f"Successfully extracted {len(paired_dataset)} paired examples")
            
        except Exception as e:
            logger.error(f"Error extracting paired dataset: {e}")
            logger.warning("Dataset format not compatible with DPO format requirements")
            paired_dataset = []
            raise ValueError("Could not create paired dataset from the provided data format")
        
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
    
    elif args.mode == "qrm_ppo":
        # Set up checkpoint and output directories
        checkpoint_dir = os.path.join("models", "qrm_ppo_checkpoints", f"run_{time.strftime('%Y%m%d_%H%M%S')}")
        output_dir = args.output_dir or os.path.join("models", f"qrm_ppo_finetuned_{time.strftime('%Y%m%d_%H%M%S')}")
        
        # Log checkpoint configuration
        logger.info(f"Checkpoints will be saved to: {checkpoint_dir}")
        logger.info(f"Final model will be saved to: {output_dir}")
        
        # Use model from config
        logger.info(f"Using model path from config: {config['rlhf']['ppo']['model_name']}")
        
        # Use Hugging Face's PPO Trainer implementation directly with QRM reward model
        logger.info("Using HuggingFace's PPO implementation with direct QRM reward model")
        ppo_trainer = HuggingFacePPOTrainer(
            config=config,
            reward_predictor=reward_model,  # Use QRM reward model directly
            device=device,
            checkpoint_dir=checkpoint_dir
        )
        
        # Load and process SHP dataset using the processors.py
        logger.info("Loading SHP dataset for QRM-PPO training")
        data_processor = SHPDataProcessor(config)
        train_data, val_data, test_data = data_processor.load_dataset()
        
        # Check if we have data
        if train_data is None or len(train_data) == 0:
            raise ValueError("Failed to load training data for QRM-PPO mode")
            
        # Create dataset in format expected by PPO trainer
        logger.info(f"Preparing dataset with {len(train_data)} examples for QRM-PPO training")
        
        # Extract prompts (history field) from SHP dataset
        try:
            # Try accessing as dictionary with expected SHP fields
            logger.info("Attempting to extract prompts using standard SHP format")
            prompts = []
            for item in train_data:
                if isinstance(item, dict) and "history" in item:
                    prompts.append(item["history"])
                elif isinstance(item, str):
                    # If item is already a string, use it directly
                    prompts.append(item)
                else:
                    # Try to get the first field in the dictionary
                    if isinstance(item, dict) and len(item) > 0:
                        first_key = next(iter(item))
                        prompts.append(item[first_key])
                    else:
                        logger.warning(f"Could not extract prompt from item: {item}")
                        prompts.append("")
            
            logger.info(f"Successfully extracted {len(prompts)} prompts")
            dataset = {"prompt": prompts}
            
        except Exception as e:
            logger.error(f"Error extracting prompts: {e}")
            logger.info("Falling back to using the entire dataset as prompts")
            
            # As a fallback, if train_data is already a list of strings, use it directly
            if all(isinstance(item, str) for item in train_data):
                dataset = {"prompt": train_data}
            else:
                raise ValueError(f"Dataset format not supported: {type(train_data)}")
        
        # Train with PPO
        ppo_trainer.train(
            dataset=dataset,
            num_epochs=config["rlhf"]["ppo"].get("num_epochs", 1),
            max_steps=config["rlhf"]["ppo"].get("max_steps", 100),
            checkpoint_interval=args.checkpoint_interval
        )
        
        # Save the fine-tuned model
        logger.info(f"Saving final QRM-PPO fine-tuned model to {output_dir}")
        ppo_trainer.save_model(output_dir)
        
        # Create a symlink to the latest model for convenience
        latest_link_path = os.path.join("models", "qrm_ppo_finetuned_latest")
        try:
            if os.path.exists(latest_link_path) or os.path.islink(latest_link_path):
                os.remove(latest_link_path)
            os.symlink(output_dir, latest_link_path, target_is_directory=True)
            logger.info(f"Created symlink 'qrm_ppo_finetuned_latest' pointing to {output_dir}")
        except Exception as e:
            logger.warning(f"Could not create symlink to latest model: {e}")
    
    elif args.mode == "qrm_dpo":
        # Use model from config
        logger.info(f"Using model path from config: {config['rlhf']['dpo']['model_name']}")
        
        # Use Hugging Face's DPO Trainer implementation
        logger.info("Using HuggingFace's DPO implementation with direct QRM reward model")
        
        # Check for PEFT/LoRA flag
        use_peft = config["rlhf"]["dpo"].get("use_peft", False)
        
        dpo_trainer = HuggingFaceDPOTrainer(
            config=config,
            reward_predictor=reward_model,  # Use QRM reward model directly
            device=device,
            use_peft=use_peft
        )
        
        # Load and process SHP dataset using the processors.py
        logger.info("Loading SHP dataset for QRM-DPO training")
        data_processor = SHPDataProcessor(config)
        train_data, val_data, test_data = data_processor.load_dataset()
        
        # Check if we have data
        if train_data is None or len(train_data) == 0:
            raise ValueError("Failed to load training data for QRM-DPO mode")
            
        # Create dataset in format expected by DPO trainer
        logger.info(f"Preparing dataset with {len(train_data)} examples for QRM-DPO training")
        
        # Extract prompt/chosen/rejected from SHP dataset
        try:
            # Try accessing as dictionary with expected SHP fields
            logger.info("Attempting to extract prompts and responses using standard SHP format")
            paired_dataset = []
            
            for item in train_data:
                if isinstance(item, dict) and "history" in item and "human_ref_A" in item and "human_ref_B" in item and "labels" in item:
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
                elif isinstance(item, dict) and len(item) >= 3:
                    # Try to extract from generic dictionary with at least 3 fields
                    keys = list(item.keys())
                    if len(keys) >= 3:
                        prompt_key = keys[0]
                        chosen_key = keys[1]
                        rejected_key = keys[2]
                        
                        paired_dataset.append({
                            "prompt": item[prompt_key],
                            "chosen": item[chosen_key],
                            "rejected": item[rejected_key]
                        })
                else:
                    logger.warning(f"Could not process item format: {type(item)}")
            
            if not paired_dataset:
                raise ValueError("Failed to extract paired dataset using standard format")
                
            logger.info(f"Successfully extracted {len(paired_dataset)} paired examples")
            
        except Exception as e:
            logger.error(f"Error extracting paired dataset: {e}")
            logger.warning("Dataset format not compatible with DPO format requirements")
            paired_dataset = []
            raise ValueError("Could not create paired dataset from the provided data format")
        
        # Get num_epochs from config
        num_epochs = config["rlhf"]["dpo"].get("num_epochs", 3)
        
        # Train with DPO on the paired dataset
        dpo_trainer.train(
            paired_dataset=paired_dataset,
            num_epochs=num_epochs,
            generate_pairs=False  # We already created the pairs
        )
        
        # Save the fine-tuned model
        output_dir = os.path.join("models", f"qrm_dpo_finetuned_{time.strftime('%Y%m%d_%H%M%S')}")
        dpo_trainer.save_model(output_dir)
        
        # Create a symlink to the latest model for convenience
        latest_link_path = os.path.join("models", "qrm_dpo_finetuned_latest")
        try:
            if os.path.exists(latest_link_path) or os.path.islink(latest_link_path):
                os.remove(latest_link_path)
            os.symlink(output_dir, latest_link_path, target_is_directory=True)
            logger.info(f"Created symlink 'qrm_dpo_finetuned_latest' pointing to {output_dir}")
        except Exception as e:
            logger.warning(f"Could not create symlink to latest model: {e}")
        
    elif args.mode == "evaluate":
        # Create output directory for evaluation results if not specified
        if args.output_file is None:
            os.makedirs("evaluation_results", exist_ok=True)
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            args.output_file = os.path.join("evaluation_results", f"{args.model_type}_evaluation_{timestamp}.json")
        
        # Ensure the output directory exists
        os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
        
        # Use model from config based on model_type parameter
        if args.model_type == "ppo" or args.model_type == "qrm_ppo":
            model_path = config["rlhf"]["ppo"]["model_name"]
        elif args.model_type == "dpo" or args.model_type == "qrm_dpo":
            model_path = config["rlhf"]["dpo"]["model_name"]
        else:
            # Default to using PPO model path
            model_path = config["rlhf"]["ppo"]["model_name"]
        
        logger.info(f"Evaluating model type: {args.model_type}")
        logger.info(f"Using model path from config: {model_path}")
        logger.info(f"Results will be saved to: {args.output_file}")
        
        # Load model and tokenizer
        try:
            logger.info(f"Loading model and tokenizer from: {model_path}")
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if "cuda" in str(device) else None,
                torch_dtype=torch.float16 if "cuda" in str(device) else torch.float32
            )
            tokenizer = AutoTokenizer.from_pretrained(model_path)
            
            # Set pad token if not set
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            logger.info(f"Model and tokenizer loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model or tokenizer: {str(e)}")
            raise
        
        # Load TruthfulQA dataset
        logger.info("Loading TruthfulQA dataset")
        try:
            truthful_dataset = TruthfulQADataset()
            eval_data = truthful_dataset.load_dataset()
            logger.info(f"Loaded {len(eval_data)} TruthfulQA questions for evaluation")
        except Exception as e:
            logger.error(f"Failed to load TruthfulQA dataset: {str(e)}")
            raise
        
        # Evaluate model on TruthfulQA
        try:
            logger.info("Starting evaluation on TruthfulQA dataset")
            results = evaluate_model_truthfulness(
                model=model,
                tokenizer=tokenizer,
                eval_data=eval_data,
                max_new_tokens=args.max_new_tokens,
                batch_size=args.batch_size,
                device=device
            )
            
            # Log summary of results
            accuracy = results["metrics"]["accuracy"]
            correct_count = results["metrics"]["correct_count"]
            total_count = results["metrics"]["total_count"]
            logger.info(f"Evaluation completed - Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
            
            # Save results to file
            import json
            with open(args.output_file, "w") as f:
                # Add metadata to results
                results["metadata"] = {
                    "model_path": model_path,
                    "model_type": args.model_type,
                    "max_new_tokens": args.max_new_tokens,
                    "batch_size": args.batch_size,
                    "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
                }
                json.dump(results, f, indent=2)
                
            logger.info(f"Evaluation results saved to {args.output_file}")
            
        except Exception as e:
            logger.error(f"Error during evaluation: {str(e)}")
            raise
        
    else:
        raise ValueError(f"Invalid mode: {args.mode}")


if __name__ == "__main__":
    main()
