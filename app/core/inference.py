"""
Inference logic for text generation with transformer models.
"""
import logging
from typing import Optional, Dict, Any, List
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer
from app.core.gpu_utils import get_device, get_dtype

logger = logging.getLogger(__name__)


class InferenceEngine:
    """
    Handles text generation inference with loaded models.
    """
    
    def __init__(self):
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.device: torch.device = get_device()
        self.model_name: Optional[str] = None
        self.model_config: Optional[Dict[str, Any]] = None
    
    def set_model(
        self, 
        model: AutoModelForCausalLM, 
        tokenizer: AutoTokenizer,
        model_name: str,
        model_config: Dict[str, Any]
    ) -> None:
        """
        Set the active model and tokenizer for inference.
        
        Args:
            model: Loaded model instance
            tokenizer: Loaded tokenizer instance
            model_name: Name of the model
            model_config: Model configuration dictionary
        """
        self.model = model
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.model_config = model_config
        logger.info(f"Inference engine ready for model: {model_name}")
    
    def clear_model(self) -> None:
        """Clear the current model and tokenizer."""
        if self.model is not None:
            del self.model
        if self.tokenizer is not None:
            del self.tokenizer
        
        self.model = None
        self.tokenizer = None
        self.model_name = None
        self.model_config = None
        logger.info("Inference engine cleared")
    
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
        do_sample: bool = True,
        stop_sequences: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Penalty for repetition (1.0 = no penalty)
            do_sample: Whether to use sampling (False = greedy decoding)
            stop_sequences: List of strings to stop generation at
        
        Returns:
            Dictionary containing generated text and metadata
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        try:
            # Tokenize input
            inputs = self.tokenizer(
                prompt,
                return_tensors="pt",
                padding=False,
                truncation=True,
                max_length=self.model_config.get("max_length", 4096) if self.model_config else 4096
            ).to(self.device)
            
            # Prepare generation parameters
            generation_kwargs = {
                "max_new_tokens": max_new_tokens,
                "temperature": temperature,
                "top_p": top_p,
                "top_k": top_k,
                "repetition_penalty": repetition_penalty,
                "do_sample": do_sample,
                "pad_token_id": self.tokenizer.eos_token_id,
            }
            
            # Add stop sequences if provided
            if stop_sequences:
                stop_token_ids = [
                    self.tokenizer.encode(seq, add_special_tokens=False)[0]
                    for seq in stop_sequences
                    if len(self.tokenizer.encode(seq, add_special_tokens=False)) > 0
                ]
                if stop_token_ids:
                    generation_kwargs["stop_token_ids"] = stop_token_ids
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    **generation_kwargs
                )
            
            # Decode output
            generated_text = self.tokenizer.decode(
                outputs[0][inputs["input_ids"].shape[1]:],
                skip_special_tokens=True
            )
            
            # Calculate token counts
            input_tokens = inputs["input_ids"].shape[1]
            output_tokens = outputs[0].shape[0] - input_tokens
            total_tokens = outputs[0].shape[0]
            
            return {
                "generated_text": generated_text,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": total_tokens,
                "model": self.model_name,
                "prompt": prompt,
            }
            
        except Exception as e:
            logger.error(f"Generation error: {str(e)}", exc_info=True)
            raise RuntimeError(f"Generation failed: {str(e)}")
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_p: float = 0.9,
        top_k: int = 50,
        repetition_penalty: float = 1.1,
    ) -> Dict[str, Any]:
        """
        Generate chat completion from a list of messages.
        
        Args:
            messages: List of message dicts with 'role' and 'content' keys
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
            top_k: Top-k sampling parameter
            repetition_penalty: Repetition penalty
        
        Returns:
            Dictionary containing assistant response and metadata
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("No model loaded. Please load a model first.")
        
        # Format messages into prompt (Qwen format)
        prompt = self._format_messages(messages)
        
        # Generate response
        result = self.generate(
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            repetition_penalty=repetition_penalty,
        )
        
        return {
            "role": "assistant",
            "content": result["generated_text"],
            "model": self.model_name,
            "usage": {
                "prompt_tokens": result["input_tokens"],
                "completion_tokens": result["output_tokens"],
                "total_tokens": result["total_tokens"],
            }
        }
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """
        Format chat messages into a prompt string.
        Uses Qwen2.5 chat format.
        
        Args:
            messages: List of message dicts
        
        Returns:
            Formatted prompt string
        """
        # Qwen2.5 uses a specific chat format
        formatted = ""
        
        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            
            if role == "system":
                formatted += f"<|im_start|>system\n{content}<|im_end|>\n"
            elif role == "user":
                formatted += f"<|im_start|>user\n{content}<|im_end|>\n"
            elif role == "assistant":
                formatted += f"<|im_start|>assistant\n{content}<|im_end|>\n"
        
        # Add assistant prompt
        formatted += "<|im_start|>assistant\n"
        
        return formatted
    
    def is_ready(self) -> bool:
        """Check if inference engine is ready."""
        return self.model is not None and self.tokenizer is not None
