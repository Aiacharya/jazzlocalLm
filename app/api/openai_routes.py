"""
OpenAI-compatible API routes for OpenClaw and other OpenAI-compatible clients.
"""
import logging
import time
import uuid
from typing import List, Dict, Any, Optional
from fastapi import APIRouter, HTTPException, Body
from pydantic import BaseModel, Field
from app.core.model_manager import ModelManager

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model manager instance (initialized in main.py)
model_manager: Optional[ModelManager] = None


def set_model_manager(manager: ModelManager) -> None:
    """Set the model manager instance."""
    global model_manager
    model_manager = manager


# OpenAI-compatible request models
class OpenAIMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")


class OpenAIChatRequest(BaseModel):
    model: Optional[str] = Field(None, description="Model name (optional, uses currently loaded model)")
    messages: List[OpenAIMessage] = Field(..., description="List of chat messages")
    max_tokens: Optional[int] = Field(None, ge=1, le=4096, description="Maximum tokens to generate")
    temperature: Optional[float] = Field(0.7, ge=0.0, le=2.0, description="Sampling temperature")
    top_p: Optional[float] = Field(0.9, ge=0.0, le=1.0, description="Nucleus sampling parameter")
    stop: Optional[List[str]] = Field(None, description="Stop sequences")
    stream: Optional[bool] = Field(False, description="Streaming (not yet supported)")
    presence_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Presence penalty")
    frequency_penalty: Optional[float] = Field(0.0, ge=-2.0, le=2.0, description="Frequency penalty")


class OpenAIModelInfo(BaseModel):
    id: str
    object: str = "model"
    created: int
    owned_by: str = "local"


class OpenAIChoice(BaseModel):
    index: int
    message: OpenAIMessage
    finish_reason: str = "stop"


class OpenAIUsage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class OpenAIChatResponse(BaseModel):
    id: str
    object: str = "chat.completion"
    created: int
    model: str
    choices: List[OpenAIChoice]
    usage: OpenAIUsage


@router.post("/v1/chat/completions")
async def openai_chat_completions(request: OpenAIChatRequest) -> Dict[str, Any]:
    """
    OpenAI-compatible chat completions endpoint.
    
    Args:
        request: OpenAI-compatible chat request
    
    Returns:
        OpenAI-compatible chat completion response
    """
    if not model_manager:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Model manager not initialized",
                    "type": "server_error",
                    "code": 500
                }
            }
        )
    
    if not model_manager.is_model_loaded():
        raise HTTPException(
            status_code=400,
            detail={
                "error": {
                    "message": "No model loaded. Please load a model first using /api/models/load",
                    "type": "invalid_request_error",
                    "code": 400
                }
            }
        )
    
    # Get currently loaded model
    loaded_model_name = model_manager.get_loaded_model_name()
    
    # Use requested model if specified, otherwise use loaded model
    if request.model and request.model != loaded_model_name:
        # Try to switch model
        try:
            model_manager.switch_model(request.model)
            loaded_model_name = request.model
        except Exception as e:
            raise HTTPException(
                status_code=400,
                detail={
                    "error": {
                        "message": f"Failed to load model '{request.model}': {str(e)}",
                        "type": "invalid_request_error",
                        "code": 400
                    }
                }
            )
    
    try:
        inference_engine = model_manager.get_inference_engine()
        
        # Convert to internal format
        messages = [{"role": msg.role, "content": msg.content} for msg in request.messages]
        
        # Map max_tokens to max_new_tokens
        max_new_tokens = request.max_tokens or 512
        
        # Map frequency_penalty to repetition_penalty (inverse relationship)
        # frequency_penalty of 0.0 = repetition_penalty of 1.0
        # frequency_penalty of 1.0 ≈ repetition_penalty of 1.1
        repetition_penalty = 1.0 + (request.frequency_penalty or 0.0) * 0.1
        
        # Generate response
        result = inference_engine.chat_completion(
            messages=messages,
            max_new_tokens=max_new_tokens,
            temperature=request.temperature or 0.7,
            top_p=request.top_p or 0.9,
            top_k=50,  # Default, not in OpenAI spec
            repetition_penalty=repetition_penalty,
        )
        
        # Format OpenAI-compatible response
        response_id = f"chatcmpl-{uuid.uuid4().hex[:12]}"
        created = int(time.time())
        
        choice = {
            "index": 0,
            "message": {
                "role": result.get("role", "assistant"),
                "content": result.get("content", "")
            },
            "finish_reason": "stop"
        }
        
        usage = result.get("usage", {})
        
        return {
            "id": response_id,
            "object": "chat.completion",
            "created": created,
            "model": loaded_model_name,
            "choices": [choice],
            "usage": {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0)
            }
        }
        
    except RuntimeError as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": str(e),
                    "type": "server_error",
                    "code": 500
                }
            }
        )
    except Exception as e:
        logger.error(f"Chat completion error: {e}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": f"Generation failed: {str(e)}",
                    "type": "server_error",
                    "code": 500
                }
            }
        )


@router.get("/v1/models")
async def openai_list_models() -> Dict[str, Any]:
    """
    OpenAI-compatible models list endpoint.
    
    Returns:
        OpenAI-compatible models list response
    """
    if not model_manager:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Model manager not initialized",
                    "type": "server_error",
                    "code": 500
                }
            }
        )
    
    models = model_manager.list_models()
    loaded_model = model_manager.get_loaded_model_name()
    
    # Format as OpenAI models list
    model_objects = []
    for model in models:
        model_objects.append({
            "id": model["name"],
            "object": "model",
            "created": int(time.time()),
            "owned_by": "local"
        })
    
    return {
        "object": "list",
        "data": model_objects
    }


@router.get("/v1/models/{model_id}")
async def openai_get_model(model_id: str) -> Dict[str, Any]:
    """
    Get information about a specific model.
    
    Args:
        model_id: Model identifier
    
    Returns:
        Model information
    """
    if not model_manager:
        raise HTTPException(
            status_code=500,
            detail={
                "error": {
                    "message": "Model manager not initialized",
                    "type": "server_error",
                    "code": 500
                }
            }
        )
    
    model_config = model_manager.get_model_config(model_id)
    if not model_config:
        raise HTTPException(
            status_code=404,
            detail={
                "error": {
                    "message": f"Model '{model_id}' not found",
                    "type": "invalid_request_error",
                    "code": 404
                }
            }
        )
    
    return {
        "id": model_id,
        "object": "model",
        "created": int(time.time()),
        "owned_by": "local"
    }
