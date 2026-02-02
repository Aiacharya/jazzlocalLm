"""
GPU utilities for VRAM management and device detection.
"""
import logging
import torch
from typing import Optional, Dict, Any

logger = logging.getLogger(__name__)


def get_device() -> torch.device:
    """
    Detect and return the best available device (CUDA if available, else CPU).
    
    Returns:
        torch.device: The device to use for model inference
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"CUDA Version: {torch.version.cuda}")
        return device
    else:
        logger.warning("CUDA not available, falling back to CPU")
        return torch.device("cpu")


def get_gpu_info() -> Dict[str, Any]:
    """
    Get current GPU memory statistics.
    
    Returns:
        Dict containing GPU memory information
    """
    if not torch.cuda.is_available():
        return {
            "available": False,
            "device_count": 0,
            "message": "CUDA not available"
        }
    
    device_count = torch.cuda.device_count()
    info = {
        "available": True,
        "device_count": device_count,
        "devices": []
    }
    
    for i in range(device_count):
        torch.cuda.set_device(i)
        allocated = torch.cuda.memory_allocated(i) / (1024 ** 3)  # GB
        reserved = torch.cuda.memory_reserved(i) / (1024 ** 3)  # GB
        total = torch.cuda.get_device_properties(i).total_memory / (1024 ** 3)  # GB
        free = total - reserved
        
        info["devices"].append({
            "index": i,
            "name": torch.cuda.get_device_name(i),
            "total_memory_gb": round(total, 2),
            "allocated_memory_gb": round(allocated, 2),
            "reserved_memory_gb": round(reserved, 2),
            "free_memory_gb": round(free, 2),
            "utilization_percent": round((reserved / total) * 100, 2) if total > 0 else 0
        })
    
    return info


def estimate_model_memory(model_size_gb: float, dtype: str = "float16") -> float:
    """
    Estimate VRAM required for a model.
    
    Args:
        model_size_gb: Model size in GB (approximate)
        dtype: Data type (float16, bfloat16, float32)
    
    Returns:
        Estimated memory requirement in GB
    """
    # Rough estimates: model weights + activations + overhead
    dtype_multiplier = {
        "float16": 1.0,
        "bfloat16": 1.0,
        "float32": 2.0
    }
    
    base_memory = model_size_gb * dtype_multiplier.get(dtype, 1.0)
    # Add overhead for activations and KV cache (rough estimate: 20-30%)
    estimated = base_memory * 1.25
    
    return estimated


def check_available_memory(required_gb: float) -> tuple[bool, Optional[str]]:
    """
    Check if enough VRAM is available for model loading.
    
    Args:
        required_gb: Required memory in GB
    
    Returns:
        Tuple of (is_available, error_message)
    """
    if not torch.cuda.is_available():
        return False, "CUDA not available"
    
    gpu_info = get_gpu_info()
    if not gpu_info["available"]:
        return False, "No GPU available"
    
    # Check primary device (device 0)
    device_info = gpu_info["devices"][0]
    free_memory = device_info["free_memory_gb"]
    
    if free_memory < required_gb:
        return False, (
            f"Insufficient VRAM. Required: {required_gb:.2f} GB, "
            f"Available: {free_memory:.2f} GB"
        )
    
    return True, None


def clear_gpu_cache() -> None:
    """
    Clear PyTorch CUDA cache to free up memory.
    """
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        logger.info("GPU cache cleared")


def get_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch.dtype.
    
    Args:
        dtype_str: String representation of dtype (float16, bfloat16, float32)
    
    Returns:
        torch.dtype
    """
    dtype_map = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    
    dtype = dtype_map.get(dtype_str.lower(), torch.float16)
    logger.debug(f"Converted '{dtype_str}' to {dtype}")
    return dtype
