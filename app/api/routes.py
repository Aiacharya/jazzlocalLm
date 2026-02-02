"""
FastAPI routes for model management and inference.
"""
import logging
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from app.core.model_manager import ModelManager
from app.core.gpu_utils import get_gpu_info

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model manager instance (initialized in main.py)
model_manager: Optional[ModelManager] = None


def set_model_manager(manager: ModelManager) -> None:
    """Set the model manager instance."""
    global model_manager
    model_manager = manager


# Pydantic models for request/response
class LoadModelRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to load")
    force_reload: bool = Field(False, description="Force reload even if already loaded")


class SwitchModelRequest(BaseModel):
    model_name: str = Field(..., description="Name of the model to switch to")


class ChatMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class ChatRequest(BaseModel):
    messages: List[ChatMessage] = Field(..., description="List of chat messages")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(50, ge=1, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")


class GenerateRequest(BaseModel):
    prompt: str = Field(..., description="Input text prompt")
    max_new_tokens: int = Field(512, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: float = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: float = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    top_k: int = Field(50, ge=1, description="Top-k sampling parameter")
    repetition_penalty: float = Field(1.1, ge=1.0, le=2.0, description="Repetition penalty")


@router.get("/health")
async def health_check() -> Dict[str, Any]:
    """
    Health check endpoint.
    
    Returns:
        Health status and basic information
    """
    return {
        "status": "healthy",
        "model_loaded": model_manager.is_model_loaded() if model_manager else False,
        "loaded_model": model_manager.get_loaded_model_name() if model_manager else None,
    }


@router.get("/gpu")
async def get_gpu_status() -> Dict[str, Any]:
    """
    Get GPU memory and status information.
    
    Returns:
        GPU information dictionary
    """
    return get_gpu_info()


@router.get("/models/list")
async def list_models() -> Dict[str, Any]:
    """
    List all available models from configuration.
    
    Returns:
        Dictionary containing list of models
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    models = model_manager.list_models()
    return {
        "models": models,
        "count": len(models),
        "loaded_model": model_manager.get_loaded_model_name(),
    }


@router.post("/models/load")
async def load_model(request: LoadModelRequest) -> Dict[str, Any]:
    """
    Load a model by name.
    
    Args:
        request: Load model request with model_name and optional force_reload
    
    Returns:
        Load status and model information
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        result = model_manager.load_model(
            model_name=request.model_name,
            force_reload=request.force_reload
        )
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/unload")
async def unload_model() -> Dict[str, Any]:
    """
    Unload the currently loaded model.
    
    Returns:
        Unload status
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        result = model_manager.unload_model()
        return result
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/models/switch")
async def switch_model(request: SwitchModelRequest) -> Dict[str, Any]:
    """
    Switch to a different model.
    
    Args:
        request: Switch model request with model_name
    
    Returns:
        Switch status and model information
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    try:
        result = model_manager.switch_model(model_name=request.model_name)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/chat")
async def chat_completion(request: ChatRequest) -> Dict[str, Any]:
    """
    Generate chat completion from messages.
    
    Args:
        request: Chat request with messages and generation parameters
    
    Returns:
        Assistant response and metadata
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    if not model_manager.is_model_loaded():
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Please load a model first using /models/load"
        )
    
    try:
        inference_engine = model_manager.get_inference_engine()
        
        # Convert Pydantic models to dicts
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        result = inference_engine.chat_completion(
            messages=messages,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        
        return result
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")


@router.post("/generate")
async def generate_text(request: GenerateRequest) -> Dict[str, Any]:
    """
    Generate text from a prompt.
    
    Args:
        request: Generate request with prompt and generation parameters
    
    Returns:
        Generated text and metadata
    """
    if not model_manager:
        raise HTTPException(status_code=500, detail="Model manager not initialized")
    
    if not model_manager.is_model_loaded():
        raise HTTPException(
            status_code=400,
            detail="No model loaded. Please load a model first using /models/load"
        )
    
    try:
        inference_engine = model_manager.get_inference_engine()
        
        result = inference_engine.generate(
            prompt=request.prompt,
            max_new_tokens=request.max_new_tokens,
            temperature=request.temperature,
            top_p=request.top_p,
            top_k=request.top_k,
            repetition_penalty=request.repetition_penalty,
        )
        
        return result
        
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    except Exception as e:
        logger.error(f"Text generation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Generation failed: {str(e)}")
