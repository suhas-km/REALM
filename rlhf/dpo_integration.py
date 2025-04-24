# rlhf/dpo_integration.py
import os
import torch
import logging
from typing import Dict, List, Optional, Tuple
from trl import DPOTrainer
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from datasets import Dataset
from tqdm import tqdm

from inference.predictor import RewardPredictor

logger = logging.getLogger(__name__)

class DPOTrainerWithCustomReward:
    """
    DPO Trainer with custom combined reward model for LLM fine-tuning
    """
    
    def __init__(
        self,
        config: Dict,
        reward_predictor: RewardPredictor,
        tokenizer=None,
        model=None,
        device: Optional[torch.device] = None
    ):
        self.config = config
        self.reward_predictor = reward_predictor
        
        # Set device
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize tokenizer and model if not provided
        if tokenizer is None or model is None:
            model_name = config["rlhf"]["dpo"]["model_name"]
            logger.info(f"Loading model and tokenizer: {model_name}")
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            self.tokenizer = tokenizer
            self.model = model
        
        # Ensure tokenizer has pad token
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        logger.info("DPO Trainer with custom reward initialized")
    
    def _compare_responses(self, prompt: str, response1: str, response2: str) -> Tuple[str, str]:
        """Compare two responses and return them in order (chosen, rejected)"""
        reward1, reward2, better = self.reward_predictor.compare(prompt, response1, response2)
        
        if better == 1:
            return response1, response2
        else:
            return response2, response1
    
    def _generate_paired_responses(self, prompts: List[str], max_length: int = 512) -> List[Dict[str, str]]:
        """Generate two different responses for each prompt and rank them"""
        paired_data = []
        
        for prompt in tqdm(prompts, desc="Generating response pairs"):
            # Generate first response
            inputs = self.tokenizer.encode(prompt, return_tensors="pt").to(self.device)
            outputs1 = self.model.generate(
                inputs, 
                max_length=max_length, 
                do_sample=True,
                temperature=0.9,  # Higher temperature for more diversity
                top_p=0.9,
                pad_token_id=self.tokenizer.pad_token_id
            )
            response1 = self.tokenizer.decode(outputs1[0], skip_special_tokens=True)
            response1 = response1[len(prompt):]  # Remove the prompt
            
            # Generate second response (with different parameters for diversity)
            outputs2 = self.model.generate(
                inputs, 
                max_length=max_length, 
                do_sample=True,
                temperature=1.0,  # Higher temperature for more diversity
                top_p=0.8,  # Different top_p
                pad_token_id=self.tokenizer.pad_token_id
            )
            response2 = self.tokenizer.decode(outputs2[0], skip_special_tokens=True)
            response2 = response2[len(prompt):]  # Remove the prompt
            
            # Avoid identical responses
            if response1 == response2:
                continue
            
            # Compare and order responses
            chosen, rejected = self._compare_responses(prompt, response1, response2)
            
            paired_data.append({
                "prompt": prompt,
                "chosen": chosen,
                "rejected": rejected
            })
        
        return paired_data
    
    def train(
        self, 
        dataset: Optional[Dict[str, List[str]]] = None,
        paired_dataset: Optional[List[Dict[str, str]]] = None,
        num_epochs: int = 3,
        generate_pairs: bool = True
    ) -> None:
        """
        Train the model using DPO with the custom reward model
        
        Args:
            dataset: Dictionary with 'prompt' key containing prompts
            paired_dataset: List of dicts with 'prompt', 'chosen', 'rejected' keys
            num_epochs: Number of epochs to train
            generate_pairs: Whether to generate paired responses (if paired_dataset not provided)
        """
        # Generate paired data if needed
        if paired_dataset is None and generate_pairs and dataset is not None:
            logger.info("Generating paired responses...")
            paired_dataset = self._generate_paired_responses(dataset["prompt"])
        elif paired_dataset is None and not generate_pairs:
            raise ValueError("Either paired_dataset must be provided or generate_pairs must be True")
        
        # Convert to Dataset
        if paired_dataset:
            train_dataset = Dataset.from_list(paired_dataset)
            
            # Create a small validation set
            train_val_split = train_dataset.train_test_split(test_size=0.05)
            train_dataset = train_val_split["train"]
            eval_dataset = train_val_split["test"]
            
            logger.info(f"Training on {len(train_dataset)} examples, validating on {len(eval_dataset)}")
            
            # Initialize DPO trainer
            dpo_trainer = DPOTrainer(
                model=self.model,
                ref_model=None,  # Will be initialized by DPOTrainer
                tokenizer=self.tokenizer,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                args=self._get_dpo_args(),
            )
            
            # Train the model
            logger.info("Starting DPO training...")
            dpo_trainer.train()
            
            # Save the model
            output_dir = os.path.join("models", "dpo_finetuned")
            dpo_trainer.save_model(output_dir)
            logger.info(f"Model saved to {output_dir}")
        else:
            logger.error("No paired data available for training")
    
    def _get_dpo_args(self):
        """Get DPO training arguments"""
        return TrainingArguments(
            output_dir="./dpo_output",
            num_train_epochs=self.config["rlhf"].get("dpo", {}).get("num_epochs", 3),
            learning_rate=self.config["rlhf"].get("dpo", {}).get("learning_rate", 5e-7),
            per_device_train_batch_size=self.config["rlhf"].get("dpo", {}).get("batch_size", 4),
            per_device_eval_batch_size=self.config["rlhf"].get("dpo", {}).get("batch_size", 4),
            gradient_accumulation_steps=self.config["rlhf"].get("dpo", {}).get("gradient_accumulation_steps", 1),
            evaluation_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=1000,
            logging_steps=100,
            remove_unused_columns=False,
            push_to_hub=False,
            label_names=["chosen_response_ids", "rejected_response_ids"],
            load_best_model_at_end=True,
            bf16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8,
            fp16=torch.cuda.is_available() and torch.cuda.get_device_capability()[0] < 8,
        )
    
    def save_model(self, output_dir: str) -> None:
        """Save the fine-tuned model and tokenizer"""
        os.makedirs(output_dir, exist_ok=True)
        self.model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        logger.info(f"Model and tokenizer saved to {output_dir}")
