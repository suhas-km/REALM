# utils/validation.py
import os
import logging
import importlib
import torch

logger = logging.getLogger(__name__)

def validate_environment(skip_api_check=False):
    """Validate environment and dependencies before running
    
    Args:
        skip_api_check: If True, skips checking for API keys (useful for test mode)
    """
    checks = []
    
    # Check API keys (optional)
    if not skip_api_check and not os.environ.get("NVIDIA_NIM_API_KEY"):
        checks.append("NVIDIA_NIM_API_KEY is not set in environment")
    
    # Check required packages
    required_packages = {
        "openai": "openai",
        "google.genai": "google-genai",
        "torch": "torch",
        "datasets": "datasets",
        "fastapi": "fastapi",
        "trl": "trl"
    }
    
    for import_name, package_name in required_packages.items():
        try:
            importlib.import_module(import_name)
            logger.info(f"✓ Package '{package_name}' is installed")
        except ImportError:
            checks.append(f"✗ Required package '{package_name}' is not installed")
    
    # Check CUDA availability if using torch
    try:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ CUDA is available: {device_name}")
        else:
            logger.warning("! CUDA is not available, using CPU (this will be slow)")
    except Exception as e:
        checks.append(f"✗ Error checking CUDA availability: {e}")
        logger.warning("! Could not check CUDA availability")
    
    # Check for cache directory write permissions
    cache_dir = "cache"
    os.makedirs(cache_dir, exist_ok=True)
    try:
        test_file = os.path.join(cache_dir, "test_write.txt")
        with open(test_file, "w") as f:
            f.write("test")
        os.remove(test_file)
        logger.info("✓ Cache directory is writable")
    except Exception as e:
        checks.append(f"✗ Cannot write to cache directory: {e}")
    
    # If any checks failed, raise error
    if checks:
        error_msg = "Environment validation failed:\n" + "\n".join(checks)
        logger.error(error_msg)
        raise EnvironmentError(error_msg)
    
    logger.info("✓ All environment checks passed")
    return True
