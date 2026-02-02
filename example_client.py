"""
Example Python client for the Local LLM Inference Server.
Demonstrates how to interact with the API programmatically.
"""
import requests
import json
from typing import Dict, Any, List

API_BASE_URL = "http://127.0.0.1:8000/api"


def check_health() -> Dict[str, Any]:
    """Check server health."""
    response = requests.get(f"{API_BASE_URL}/health")
    response.raise_for_status()
    return response.json()


def get_gpu_status() -> Dict[str, Any]:
    """Get GPU status."""
    response = requests.get(f"{API_BASE_URL}/gpu")
    response.raise_for_status()
    return response.json()


def list_models() -> Dict[str, Any]:
    """List available models."""
    response = requests.get(f"{API_BASE_URL}/models/list")
    response.raise_for_status()
    return response.json()


def load_model(model_name: str, force_reload: bool = False) -> Dict[str, Any]:
    """Load a model."""
    response = requests.post(
        f"{API_BASE_URL}/models/load",
        json={"model_name": model_name, "force_reload": force_reload},
        timeout=300  # 5 minutes for model loading
    )
    response.raise_for_status()
    return response.json()


def unload_model() -> Dict[str, Any]:
    """Unload current model."""
    response = requests.post(f"{API_BASE_URL}/models/unload")
    response.raise_for_status()
    return response.json()


def switch_model(model_name: str) -> Dict[str, Any]:
    """Switch to a different model."""
    response = requests.post(
        f"{API_BASE_URL}/models/switch",
        json={"model_name": model_name},
        timeout=300
    )
    response.raise_for_status()
    return response.json()


def chat_completion(
    messages: List[Dict[str, str]],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> Dict[str, Any]:
    """Generate chat completion."""
    response = requests.post(
        f"{API_BASE_URL}/chat",
        json={
            "messages": messages,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def generate_text(
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    top_k: int = 50,
    repetition_penalty: float = 1.1,
) -> Dict[str, Any]:
    """Generate text from prompt."""
    response = requests.post(
        f"{API_BASE_URL}/generate",
        json={
            "prompt": prompt,
            "max_new_tokens": max_new_tokens,
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
        },
        timeout=120
    )
    response.raise_for_status()
    return response.json()


def main():
    """Example usage."""
    print("=== Local LLM Inference Server - Example Client ===\n")
    
    # Check health
    print("1. Checking server health...")
    health = check_health()
    print(f"   Status: {health['status']}")
    print(f"   Model loaded: {health['model_loaded']}")
    print(f"   Loaded model: {health.get('loaded_model', 'None')}\n")
    
    # GPU status
    print("2. GPU Status:")
    gpu_info = get_gpu_status()
    if gpu_info.get("available"):
        device = gpu_info["devices"][0]
        print(f"   Device: {device['name']}")
        print(f"   Total VRAM: {device['total_memory_gb']} GB")
        print(f"   Free VRAM: {device['free_memory_gb']} GB")
    else:
        print("   CUDA not available")
    print()
    
    # List models
    print("3. Available models:")
    models_data = list_models()
    for model in models_data["models"]:
        print(f"   - {model['name']}: {model['description']}")
    print()
    
    # Load model (if not already loaded)
    if not health["model_loaded"]:
        print("4. Loading model...")
        model_name = "qwen2.5-3b-instruct"
        load_result = load_model(model_name)
        print(f"   {load_result['message']}\n")
    else:
        print(f"4. Model already loaded: {health.get('loaded_model')}\n")
    
    # Chat completion example
    print("5. Chat completion example:")
    messages = [
        {"role": "user", "content": "What is machine learning in one sentence?"}
    ]
    chat_result = chat_completion(messages, max_new_tokens=100)
    print(f"   User: {messages[0]['content']}")
    print(f"   Assistant: {chat_result['content']}\n")
    
    # Text generation example
    print("6. Text generation example:")
    prompt = "The future of artificial intelligence"
    gen_result = generate_text(prompt, max_new_tokens=100)
    print(f"   Prompt: {prompt}")
    print(f"   Generated: {gen_result['generated_text']}\n")
    
    print("=== Example completed ===")


if __name__ == "__main__":
    try:
        main()
    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to server. Make sure the server is running on http://127.0.0.1:8000")
    except requests.exceptions.HTTPError as e:
        print(f"HTTP Error: {e}")
        if e.response is not None:
            print(f"Response: {e.response.text}")
    except Exception as e:
        print(f"Error: {e}")
