"""
Model manager for loading, unloading, and switching transformer models.
"""
import logging
import yaml
from pathlib import Path
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.gpu_utils import (
    get_device,
    get_dtype,
    check_available_memory,
    estimate_model_memory,
    clear_gpu_cache,
)
from app.core.inference import InferenceEngine

logger = logging.getLogger(__name__)


class ModelManager:
    """
    Manages model lifecycle: loading, unloading, and switching models.
    """
    
    def __init__(self, config_path: str = "config/models.yaml"):
        """
        Initialize the model manager.
        
        Args:
            config_path: Path to the models configuration YAML file
        """
        self.config_path = Path(config_path)
        self.models_config: Dict[str, Any] = {}
        self.loaded_model: Optional[AutoModelForCausalLM] = None
        self.loaded_tokenizer: Optional[AutoTokenizer] = None
        self.current_model_name: Optional[str] = None
        self.current_model_config: Optional[Dict[str, Any]] = None
        self.device = get_device()
        self.inference_engine = InferenceEngine()
        
        self._load_config()
    
    def _load_config(self) -> None:
        """Load model configuration from YAML file."""
        try:
            with open(self.config_path, "r", encoding="utf-8") as f:
                config = yaml.safe_load(f)
                self.models_config = config.get("models", [])
                logger.info(f"Loaded {len(self.models_config)} models from config")
        except Exception as e:
            logger.error(f"Failed to load config: {e}", exc_info=True)
            self.models_config = []
    
    def list_models(self) -> List[Dict[str, Any]]:
        """
        List all available models from configuration.
        
        Returns:
            List of model configuration dictionaries
        """
        return [
            {
                "name": model["name"],
                "hf_id": model.get("hf_id", ""),
                "description": model.get("description", ""),
                "max_length": model.get("max_length", 4096),
                "dtype": model.get("dtype", "float16"),
            }
            for model in self.models_config
        ]
    
    def get_model_config(self, model_name: str) -> Optional[Dict[str, Any]]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model
        
        Returns:
            Model configuration dictionary or None if not found
        """
        for model in self.models_config:
            if model["name"] == model_name:
                return model
        return None
    
    def is_model_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.current_model_name is not None
    
    def get_loaded_model_name(self) -> Optional[str]:
        """Get the name of the currently loaded model."""
        return self.current_model_name
    
    def load_model(
        self,
        model_name: str,
        force_reload: bool = False
    ) -> Dict[str, Any]:
        """
        Load a model by name.
        
        Args:
            model_name: Name of the model to load
            force_reload: If True, reload even if already loaded
        
        Returns:
            Dictionary with load status and information
        """
        # Check if model is already loaded
        if self.current_model_name == model_name and not force_reload:
            return {
                "status": "already_loaded",
                "model_name": model_name,
                "message": f"Model '{model_name}' is already loaded"
            }
        
        # Get model configuration
        model_config = self.get_model_config(model_name)
        if not model_config:
            raise ValueError(f"Model '{model_name}' not found in configuration")
        
        # Unload current model if different
        if self.is_model_loaded() and self.current_model_name != model_name:
            logger.info(f"Unloading current model '{self.current_model_name}' to load '{model_name}'")
            self.unload_model()
        
        # Estimate memory requirements (rough estimate based on model size)
        # For Qwen models: 3B ~ 6GB, 7B ~ 14GB, 1.5B ~ 3GB
        model_size_map = {
            "3b": 6.0,
            "7b": 14.0,
            "1.5b": 3.0,
        }
        
        estimated_size = 6.0  # default
        for size_key, size_gb in model_size_map.items():
            if size_key in model_name.lower():
                estimated_size = size_gb
                break
        
        required_memory = estimate_model_memory(
            estimated_size,
            model_config.get("dtype", "float16")
        )
        
        # Check available memory
        is_available, error_msg = check_available_memory(required_memory)
        if not is_available:
            raise RuntimeError(f"Cannot load model: {error_msg}")
        
        try:
            logger.info(f"Loading model '{model_name}' from Hugging Face...")
            hf_id = model_config.get("hf_id", model_name)
            dtype = get_dtype(model_config.get("dtype", "float16"))
            
            # Load tokenizer
            logger.info(f"Loading tokenizer for {hf_id}")
            tokenizer = AutoTokenizer.from_pretrained(
                hf_id,
                trust_remote_code=True,
            )
            
            # Load model
            logger.info(f"Loading model {hf_id} with dtype {dtype}")
            model = AutoModelForCausalLM.from_pretrained(
                hf_id,
                dtype=dtype,  # Use dtype instead of deprecated torch_dtype
                device_map="auto",
                trust_remote_code=True,
                low_cpu_mem_usage=True,
            )
            
            # Move to device if not already there
            if not next(model.parameters()).is_cuda and self.device.type == "cuda":
                model = model.to(self.device)
            
            # Set model in manager
            self.loaded_model = model
            self.loaded_tokenizer = tokenizer
            self.current_model_name = model_name
            self.current_model_config = model_config
            
            # Set in inference engine
            self.inference_engine.set_model(
                model,
                tokenizer,
                model_name,
                model_config
            )
            
            logger.info(f"Successfully loaded model '{model_name}'")
            
            return {
                "status": "loaded",
                "model_name": model_name,
                "hf_id": hf_id,
                "dtype": str(dtype),
                "device": str(self.device),
                "message": f"Model '{model_name}' loaded successfully"
            }
            
        except Exception as e:
            logger.error(f"Failed to load model '{model_name}': {e}", exc_info=True)
            # Clean up on failure
            if self.loaded_model is not None:
                del self.loaded_model
                self.loaded_model = None
            if self.loaded_tokenizer is not None:
                del self.loaded_tokenizer
                self.loaded_tokenizer = None
            clear_gpu_cache()
            raise RuntimeError(f"Failed to load model '{model_name}': {str(e)}")
    
    def unload_model(self) -> Dict[str, Any]:
        """
        Unload the currently loaded model.
        
        Returns:
            Dictionary with unload status
        """
        if not self.is_model_loaded():
            return {
                "status": "no_model_loaded",
                "message": "No model is currently loaded"
            }
        
        model_name = self.current_model_name
        
        try:
            # Clear inference engine
            self.inference_engine.clear_model()
            
            # Delete model and tokenizer
            if self.loaded_model is not None:
                del self.loaded_model
                self.loaded_model = None
            
            if self.loaded_tokenizer is not None:
                del self.loaded_tokenizer
                self.loaded_tokenizer = None
            
            # Clear GPU cache
            clear_gpu_cache()
            
            self.current_model_name = None
            self.current_model_config = None
            
            logger.info(f"Unloaded model '{model_name}'")
            
            return {
                "status": "unloaded",
                "model_name": model_name,
                "message": f"Model '{model_name}' unloaded successfully"
            }
            
        except Exception as e:
            logger.error(f"Error unloading model: {e}", exc_info=True)
            raise RuntimeError(f"Failed to unload model: {str(e)}")
    
    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """
        Switch to a different model (unload current, load new).
        
        Args:
            model_name: Name of the model to switch to
        
        Returns:
            Dictionary with switch status
        """
        logger.info(f"Switching from '{self.current_model_name}' to '{model_name}'")
        
        # Unload current model
        if self.is_model_loaded():
            self.unload_model()
        
        # Load new model
        return self.load_model(model_name)
    
    def get_inference_engine(self) -> InferenceEngine:
        """Get the inference engine instance."""
        return self.inference_engine
