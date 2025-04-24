# data/truthfulness_dataset.py
import os
import logging
import torch
import json
from typing import Dict, List, Tuple, Optional
from datasets import load_dataset
from torch.utils.data import Dataset, DataLoader

logger = logging.getLogger(__name__)

class TruthfulQADataset:
    """
    Dataset loader for TruthfulQA dataset for evaluation
    """
    
    def __init__(self, cache_dir: str = "./cache/truthfulqa"):
        self.cache_dir = cache_dir
        os.makedirs(self.cache_dir, exist_ok=True)
        logger.info(f"Initializing TruthfulQA dataset with cache at {self.cache_dir}")
    
    def load_dataset(self) -> Dict:
        """
        Load the TruthfulQA dataset
        
        Returns:
            Dictionary containing the processed dataset
        """
        logger.info("Loading TruthfulQA dataset")
        try:
            # Load the dataset using Hugging Face datasets
            dataset = load_dataset("truthful_qa", "multiple_choice")
            
            # Format for evaluation
            formatted_data = []
            
            for item in dataset["validation"]:
                question = item["question"]
                
                # Extract correct and incorrect answers
                correct_answers = [item["mc1_targets"][i] for i, label in enumerate(item["mc1_labels"]) if label == 1]
                incorrect_answers = [item["mc1_targets"][i] for i, label in enumerate(item["mc1_labels"]) if label == 0]
                
                formatted_data.append({
                    "question": question,
                    "correct_answers": correct_answers,
                    "incorrect_answers": incorrect_answers
                })
            
            # Save to cache
            cache_path = os.path.join(self.cache_dir, "processed_data.json")
            with open(cache_path, 'w') as f:
                json.dump(formatted_data, f)
            
            logger.info(f"Loaded and processed {len(formatted_data)} questions for truthfulness evaluation")
            return formatted_data
            
        except Exception as e:
            logger.error(f"Error loading TruthfulQA dataset: {str(e)}")
            
            # Check if cached data exists
            cache_path = os.path.join(self.cache_dir, "processed_data.json")
            if os.path.exists(cache_path):
                logger.info("Loading from cache instead")
                with open(cache_path, 'r') as f:
                    formatted_data = json.load(f)
                return formatted_data
            else:
                raise e

def evaluate_model_truthfulness(
    model,
    tokenizer,
    eval_data: List[Dict],
    max_new_tokens: int = 128,
    batch_size: int = 4,
    device: str = "cuda"
) -> Dict:
    """
    Evaluate a model on the TruthfulQA dataset
    
    Args:
        model: Model to evaluate
        tokenizer: Tokenizer for the model
        eval_data: Processed TruthfulQA data
        max_new_tokens: Maximum new tokens to generate
        batch_size: Batch size for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    model.to(device)
    
    correct_count = 0
    total_count = 0
    
    results = {
        "questions": [],
        "metrics": {}
    }
    
    logger.info(f"Evaluating model on {len(eval_data)} TruthfulQA questions")
    
    for i in range(0, len(eval_data), batch_size):
        batch = eval_data[i:i+batch_size]
        
        for item in batch:
            question = item["question"]
            correct_answers = item["correct_answers"]
            
            # Generate response
            inputs = tokenizer(question, return_tensors="pt").to(device)
            
            with torch.no_grad():
                outputs = model.generate(
                    input_ids=inputs.input_ids,
                    attention_mask=inputs.attention_mask,
                    max_new_tokens=max_new_tokens,
                    do_sample=False  # Use greedy decoding for deterministic evaluation
                )
            
            response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if question in response:
                # Remove the question from the response if it's included
                response = response[len(question):].strip()
            
            # Check if response contains any correct answer
            is_correct = any(correct_answer.lower() in response.lower() for correct_answer in correct_answers)
            
            if is_correct:
                correct_count += 1
            total_count += 1
            
            # Store individual result
            results["questions"].append({
                "question": question,
                "response": response,
                "is_correct": is_correct,
                "correct_answers": correct_answers
            })
    
    # Calculate overall metrics
    accuracy = correct_count / total_count if total_count > 0 else 0
    
    results["metrics"] = {
        "accuracy": accuracy,
        "correct_count": correct_count,
        "total_count": total_count
    }
    
    logger.info(f"TruthfulQA Evaluation - Accuracy: {accuracy:.4f} ({correct_count}/{total_count})")
    return results
