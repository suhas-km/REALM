# using the nvidia APIs with rate limits 
# models/nim_reward.py -> gets the rewards scores
import time
import logging
import threading
from typing import List
from openai import OpenAI

logger = logging.getLogger(__name__)

class RateLimiter:
    """Thread-safe rate limiter for API calls"""
    
    def __init__(self, max_calls: int, time_period: int = 60):
        """
        Initialize rate limiter.
        
        Args:
            max_calls: Maximum number of calls allowed in the time period
            time_period: Time period in seconds (default: 60 seconds = 1 minute)
        """
        self.max_calls = max_calls
        self.time_period = time_period
        self.calls = []
        self.lock = threading.Lock()
    
    def _cleanup_old_calls(self):
        """Remove calls that are outside the time period"""
        current_time = time.time()
        with self.lock:
            self.calls = [call_time for call_time in self.calls 
                         if current_time - call_time < self.time_period]
    
    def wait(self):
        """Wait until a call can be made"""
        while True:
            self._cleanup_old_calls()
            with self.lock:
                if len(self.calls) < self.max_calls:
                    # We can make a call
                    self.calls.append(time.time())
                    return
            
            # Calculate the optimal wait time based on the oldest call
            with self.lock:
                if self.calls:
                    oldest_call = min(self.calls)
                    wait_time = max(0.1, self.time_period - (time.time() - oldest_call))
                else:
                    wait_time = 0.1
            
            # Wait before checking again
            time.sleep(wait_time)
    
    def __call__(self, func):
        """Decorator for rate-limited functions"""
        def wrapper(*args, **kwargs):
            self.wait()
            return func(*args, **kwargs)
        return wrapper


class NIMRewardModel:
    """Wrapper for Llama 3.1 Nemotron 70B Reward model via NVIDIA NIM API"""
    
    # models/nim_reward.py
    def __init__(
        self, 
        api_key: str,
        base_url: str = "https://api.nim.nvidia.com/v1",
        model_id: str = "llama-3.1-nemotron-70b-reward",
        max_retries: int = 3,
        retry_delay: float = 2.0,
    ):
        """Initialize the Llama 3.1 Nemotron 70B Reward model via NIM API."""
        self.api_key = api_key
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        
        # Try to initialize the client with better error handling
        try:
            # Import httpx to create a custom client without proxy settings
            import httpx
            
            # Create a custom HTTP client with explicit timeout and no proxies
            custom_http_client = httpx.Client(
                timeout=60.0,  # Set an appropriate timeout
                follow_redirects=True
            )
            
            # Initialize the OpenAI client with the custom HTTP client
            self.client = OpenAI(
                base_url=base_url,
                api_key=api_key,
                http_client=custom_http_client
            )
                
            # Test the connection
            logger.info("Successfully connected to NIM API")
            
            # Find the correct model name
            self.model_id = self._find_working_model_name(model_id)
            logger.info(f"Using model: {self.model_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize NIM API client: {e}")
            raise

    def _find_working_model_name(self, primary_model):
        """Try different model name formats if the primary one fails"""
        model_variants = [
            primary_model,
            f"nvidia/{primary_model}",
            "llama31-nemotron-70b-reward",
            "nem70b-reward"
        ]
        
        for model_name in model_variants:
            try:
                # Test with a simple request
                self.client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
                return model_name
            except Exception as e:
                logger.warning(f"Failed with model name {model_name}: {e}")
        
        raise ValueError("Could not find a working reward model name")
    
    def format_prompt(self, prompt: str, response: str) -> str:
        """Format prompt and response for the reward model"""
        # Format for Llama 3.1 Nemotron reward model
        return f"<|user|>\n{prompt}\n<|assistant|>\n{response}"
    
    def get_reward_score(self, prompt: str, response: str) -> float:
        """Get reward score for a prompt-response pair"""
        try:
            # Format the input for chat completions API
            messages = [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": response}
            ]
            
            # Make the API call
            reward_response = self.client.chat.completions.create(
                model=self.model_id,
                messages=messages,
                temperature=0.0,
                max_tokens=1
            )
            
            # Extract reward score from response
            if hasattr(reward_response, "choices") and len(reward_response.choices) > 0:
                choice = reward_response.choices[0]
                
                if hasattr(choice, "message") and hasattr(choice.message, "content"):
                    content = choice.message.content.strip()
                    
                    # Parse the "reward:X" format
                    if "reward:" in content:
                        try:
                            # Extract the numeric part after "reward:"
                            reward_part = content.split("reward:")[1].strip()
                            return float(reward_part)
                        except (ValueError, IndexError) as e:
                            logger.warning(f"Failed to parse reward value: {content}. Error: {e}")
                            return 0.0
                
                # Try alternative reward locations if the content parsing fails
                if hasattr(choice, "reward"):
                    return float(choice.reward)
                    
            logger.warning(f"No valid reward found in response: {reward_response}")
            return 0.0
            
        except Exception as e:
            logger.error(f"Error getting reward score: {str(e)}")
            return 0.0

    
    def batch_get_reward_scores(self, prompts: List[str], responses: List[str]) -> List[float]:
        """Get reward scores for batches of prompt-response pairs"""
        if len(prompts) != len(responses):
            raise ValueError("Number of prompts and responses must be the same")
        
        # Process each prompt-response pair individually
        # This is because reward models typically don't support batch processing
        rewards = []
        for prompt, response in zip(prompts, responses):
            reward = self.get_reward_score(prompt, response)
            rewards.append(reward)
        
        return rewards


# class BatchProcessingNimLlamaRewardModel(NimLlamaRewardModel):
#     """
#     Enhanced version of NimLlamaRewardModel with parallel processing using a worker pool
#     for efficient handling of rate limits
#     """
    
#     def __init__(
#         self,
#         api_key: str,
#         api_url: str = "https://api.nim.nvidia.com/v1/llm",
#         model_id: str = "llama-3.1-nemotron-70b-reward",
#         max_calls_per_minute: int = 40,
#         max_retries: int = 3,
#         retry_delay: float = 2.0,
#         max_batch_size: int = 10,
#         num_workers: int = 4,
#     ):
#         """
#         Initialize with worker pool for parallel processing within rate limits
        
#         Args:
#             num_workers: Number of worker threads for parallel processing
#             (other args same as parent class)
#         """
#         super().__init__(
#             api_key=api_key,
#             api_url=api_url,
#             model_id=model_id,
#             max_calls_per_minute=max_calls_per_minute,
#             max_retries=max_retries,
#             retry_delay=retry_delay,
#             max_batch_size=max_batch_size,
#         )
        
#         self.num_workers = num_workers
        
#         # Initialize work queue and worker threads
#         self.queue = Queue()
#         self.result_queue = Queue()
#         self.workers = []
        
#         # Start worker threads
#         for _ in range(num_workers):
#             worker = threading.Thread(target=self._worker_loop, daemon=True)
#             worker.start()
#             self.workers.append(worker)
    
#     def _worker_loop(self):
#         """Worker thread that processes reward requests from the queue"""
#         while True:
#             try:
#                 # Get a task from the queue
#                 task_id, prompt, response = self.queue.get()
                
#                 # Process the task
#                 try:
#                     reward = super().get_reward_score(prompt, response)
#                     self.result_queue.put((task_id, reward))
#                 except Exception as e:
#                     logger.error(f"Error in worker thread: {str(e)}")
#                     self.result_queue.put((task_id, 0.0))
                
#                 # Mark the task as done
#                 self.queue.task_done()
            
#             except Empty:
#                 # Queue is empty, wait a bit
#                 time.sleep(0.1)
#             except Exception as e:
#                 logger.error(f"Unexpected error in worker thread: {str(e)}")
#                 time.sleep(1.0)
    
#     def batch_get_reward_scores(self, prompts: List[str], responses: List[str]) -> List[float]:
#         """
#         Get reward scores for batches of prompt-response pairs using worker pool.
#         This method enables parallel processing within the rate limits.
#         """
#         if len(prompts) != len(responses):
#             raise ValueError("Number of prompts and responses must be the same")
        
#         # Add all tasks to the queue
#         for i, (prompt, response) in enumerate(zip(prompts, responses)):
#             self.queue.put((i, prompt, response))
        
#         # Wait for all tasks to be processed
#         self.queue.join()
        
#         # Collect results
#         results = {}
#         while not self.result_queue.empty():
#             task_id, reward = self.result_queue.get()
#             results[task_id] = reward
        
#         # Organize results in the original order
#         return [results.get(i, 0.0) for i in range(len(prompts))]
