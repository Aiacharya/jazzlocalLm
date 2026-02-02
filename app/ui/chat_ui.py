"""
Gradio-based chat UI for testing models.
"""
import logging
import gradio as gr
from typing import Optional, Tuple
import requests
import json

logger = logging.getLogger(__name__)

# API base URL (default to localhost)
API_BASE_URL = "http://127.0.0.1:8000/api"


def get_models() -> list:
    """Fetch available models from API."""
    try:
        response = requests.get(f"{API_BASE_URL}/models/list", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return [model["name"] for model in data.get("models", [])]
    except requests.exceptions.Timeout:
        logger.error("Request timed out - server may be busy")
        return []
    except Exception as e:
        logger.error(f"Failed to fetch models: {e}")
        return []
    return []


def get_loaded_model() -> Optional[str]:
    """Get currently loaded model name."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            return data.get("loaded_model")
    except requests.exceptions.Timeout:
        logger.error("Request timed out - server may be busy")
        return None
    except Exception as e:
        logger.error(f"Failed to fetch loaded model: {e}")
        return None
    return None


def load_model(model_name: str) -> Tuple[str, str]:
    """Load a model via API."""
    try:
        response = requests.post(
            f"{API_BASE_URL}/models/load",
            json={"model_name": model_name, "force_reload": False},
            timeout=300  # 5 minutes for model loading
        )
        if response.status_code == 200:
            data = response.json()
            return f"✅ {data.get('message', 'Model loaded')}", model_name
        else:
            error = response.json().get("detail", "Unknown error")
            return f"❌ Error: {error}", get_loaded_model() or ""
    except Exception as e:
        return f"❌ Failed to load model: {str(e)}", get_loaded_model() or ""


def _validate_message_format(messages: list) -> list:
    """Validate and clean message format for Gradio."""
    validated = []
    for msg in messages:
        if isinstance(msg, dict):
            role = msg.get("role", "")
            content = msg.get("content", "")
            # Ensure both role and content are strings and valid
            if role in ["user", "assistant", "system"] and isinstance(content, (str, int, float)):
                validated.append({
                    "role": str(role),
                    "content": str(content)
                })
        elif isinstance(msg, (list, tuple)) and len(msg) == 2:
            # Convert tuple to message format
            user_msg, assistant_msg = msg
            if user_msg:
                validated.append({"role": "user", "content": str(user_msg)})
            if assistant_msg:
                validated.append({"role": "assistant", "content": str(assistant_msg)})
    return validated


def chat_response(message: str, history: list) -> Tuple[str, list]:
    """Generate chat response via API."""
    # Gradio 6.5.1 expects list of dicts with 'role' and 'content' keys
    if history is None:
        history = []
    
    # Convert and validate history to message format
    if not isinstance(history, list):
        history = []
    
    messages_list = _validate_message_format(history)
    
    if not message or not str(message).strip():
        # Return validated empty history
        validated = _validate_message_format(messages_list)
        return "", validated
    
    # Get currently loaded model
    try:
        loaded_model = get_loaded_model()
    except:
        loaded_model = None
    
    if not loaded_model:
        error_msg = "❌ No model loaded. Please select and load a model first."
        current_msg = str(message).strip()
        new_messages = messages_list + [
            {"role": "user", "content": str(current_msg)},
            {"role": "assistant", "content": str(error_msg)}
        ]
        validated_messages = _validate_message_format(new_messages)
        # Double-check format
        final_messages = []
        for msg in validated_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                final_messages.append({
                    "role": str(msg["role"]),
                    "content": str(msg["content"])
                })
        return "", final_messages
    
    # Format history into messages for API
    # Extract messages from history format
    api_messages = []
    for msg in messages_list:
        if isinstance(msg, dict) and msg.get("role") in ["user", "assistant"]:
            content = str(msg.get("content", "")).strip()
            if content and not (msg.get("role") == "assistant" and content.startswith("❌")):
                api_messages.append({"role": msg["role"], "content": content})
    
    # Add current message
    current_msg = str(message).strip()
    if current_msg:
        api_messages.append({"role": "user", "content": current_msg})
    
    try:
        response = requests.post(
            f"{API_BASE_URL}/chat",
            json={
                "messages": api_messages,
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.1,
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            # Handle both response formats
            if isinstance(data, dict):
                assistant_response = data.get("content", "")
                if not assistant_response:
                    assistant_response = data.get("generated_text", "")
            else:
                assistant_response = str(data)
            
            # Ensure response is a string
            assistant_response = str(assistant_response).strip() if assistant_response else ""
            
            if assistant_response:
                # Append in message format: list of dicts with 'role' and 'content'
                new_messages = messages_list + [
                    {"role": "user", "content": str(current_msg)},
                    {"role": "assistant", "content": str(assistant_response)}
                ]
            else:
                new_messages = messages_list + [
                    {"role": "user", "content": str(current_msg)},
                    {"role": "assistant", "content": "❌ Error: Empty response from server"}
                ]
            
            # Validate and return in message format - ensure all are proper dicts
            validated_messages = _validate_message_format(new_messages)
            # Double-check: ensure every message has both role and content as strings
            final_messages = []
            for msg in validated_messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    final_messages.append({
                        "role": str(msg["role"]),
                        "content": str(msg["content"])
                    })
            return "", final_messages
        else:
            try:
                error_data = response.json()
                error = error_data.get("detail", f"HTTP {response.status_code}: {response.text}")
            except:
                error = f"HTTP {response.status_code}: {response.text}"
            new_messages = messages_list + [
                {"role": "user", "content": str(current_msg)},
                {"role": "assistant", "content": str(f"❌ Error: {error}")}
            ]
            validated_messages = _validate_message_format(new_messages)
            # Double-check format
            final_messages = []
            for msg in validated_messages:
                if isinstance(msg, dict) and "role" in msg and "content" in msg:
                    final_messages.append({
                        "role": str(msg["role"]),
                        "content": str(msg["content"])
                    })
            return "", final_messages
            
    except requests.exceptions.Timeout:
        new_messages = messages_list + [
            {"role": "user", "content": str(current_msg)},
            {"role": "assistant", "content": "❌ Error: Request timed out. The server may be busy. Please try again."}
        ]
        validated_messages = _validate_message_format(new_messages)
        final_messages = []
        for msg in validated_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                final_messages.append({
                    "role": str(msg["role"]),
                    "content": str(msg["content"])
                })
        return "", final_messages
    except Exception as e:
        error_msg = f"❌ Failed to generate response: {str(e)}"
        new_messages = messages_list + [
            {"role": "user", "content": str(current_msg)},
            {"role": "assistant", "content": str(error_msg)}
        ]
        validated_messages = _validate_message_format(new_messages)
        final_messages = []
        for msg in validated_messages:
            if isinstance(msg, dict) and "role" in msg and "content" in msg:
                final_messages.append({
                    "role": str(msg["role"]),
                    "content": str(msg["content"])
                })
        return "", final_messages


def create_ui() -> gr.Blocks:
    """Create and return the Gradio chat interface."""
    
    with gr.Blocks(title="Local LLM Chat Interface", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🤖 Local LLM Chat Interface")
        gr.Markdown("Test your local Hugging Face transformer models")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Initialize with empty list, will be populated on load
                model_dropdown = gr.Dropdown(
                    choices=[],
                    label="Available Models",
                    interactive=True,
                    value=None,
                )
                load_btn = gr.Button("Load Model", variant="primary")
                status_text = gr.Textbox(
                    label="Status",
                    value="No model loaded",
                    interactive=False,
                )
                loaded_model_display = gr.Textbox(
                    label="Currently Loaded",
                    value=get_loaded_model() or "None",
                    interactive=False,
                )
                
                gr.Markdown("### GPU Status")
                gpu_status = gr.JSON(label="GPU Information")
                
                def update_gpu_status():
                    try:
                        response = requests.get(f"{API_BASE_URL}/gpu", timeout=10)
                        if response.status_code == 200:
                            return response.json()
                        else:
                            return {"error": f"HTTP {response.status_code}"}
                    except requests.exceptions.Timeout:
                        logger.error("GPU status request timed out")
                        return {"error": "Request timed out - server may be busy"}
                    except Exception as e:
                        logger.error(f"GPU status error: {e}")
                        return {"error": f"Failed to fetch GPU status: {str(e)}"}
                
                def update_ui():
                    models = get_models()
                    loaded = get_loaded_model() or "None"
                    gpu_info = update_gpu_status()
                    return (
                        gr.Dropdown(choices=models),
                        loaded,
                        gpu_info
                    )
                
                refresh_btn = gr.Button("Refresh", variant="secondary")
                
            with gr.Column(scale=3):
                # Initialize chatbot with empty list in message format
                chatbot = gr.Chatbot(
                    label="Chat",
                    height=500,
                    value=[],  # Empty list - will be populated with message format dicts
                )
                msg_input = gr.Textbox(
                    label="Message",
                    placeholder="Type your message here...",
                    lines=2,
                )
                with gr.Row():
                    submit_btn = gr.Button("Send", variant="primary")
                    clear_btn = gr.Button("Clear", variant="secondary")
        
        # Event handlers
        load_btn.click(
            fn=load_model,
            inputs=[model_dropdown],
            outputs=[status_text, loaded_model_display],
        )
        
        refresh_btn.click(
            fn=update_ui,
            outputs=[model_dropdown, loaded_model_display, gpu_status],
        )
        
        submit_btn.click(
            fn=chat_response,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        
        msg_input.submit(
            fn=chat_response,
            inputs=[msg_input, chatbot],
            outputs=[msg_input, chatbot],
        )
        
        def clear_chat():
            """Clear chat history."""
            return [], ""  # Return empty list for messages format
        
        clear_btn.click(
            fn=clear_chat,
            outputs=[chatbot, msg_input],
        )
        
        # Initial load - update all UI elements
        def initialize_ui():
            try:
                models = get_models()
                loaded = get_loaded_model() or "None"
                gpu_info = update_gpu_status()
                return (
                    gr.Dropdown(choices=models, value=models[0] if models else None),
                    loaded,
                    gpu_info
                )
            except Exception as e:
                logger.error(f"UI initialization error: {e}")
                return (
                    gr.Dropdown(choices=[], value=None),
                    "Error loading",
                    {"error": str(e)}
                )
        
        demo.load(
            fn=initialize_ui,
            outputs=[model_dropdown, loaded_model_display, gpu_status],
        )
    
    return demo


def launch_ui(server_name: str = "127.0.0.1", server_port: int = 7860, share: bool = False):
    """
    Launch the Gradio UI.
    
    Args:
        server_name: Server hostname
        server_port: Server port
        share: Whether to create a public share link
    """
    demo = create_ui()
    demo.launch(
        server_name=server_name,
        server_port=server_port,
        share=share,
    )
