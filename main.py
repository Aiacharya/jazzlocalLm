"""
Main entry point for the FastAPI inference server.
"""
import logging
import argparse
import uvicorn
from pathlib import Path
from contextlib import asynccontextmanager
from app.core.model_manager import ModelManager
from app.api.routes import router, set_model_manager
from app.api.openai_routes import router as openai_router, set_model_manager as set_openai_model_manager
from app.core.gpu_utils import get_device, get_gpu_info
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import yaml

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

# Global model manager
model_manager: ModelManager = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for startup and shutdown events."""
    global model_manager
    
    # Startup
    config_path = Path("config/models.yaml")
    if not config_path.exists():
        logger.warning(f"Config file not found at {config_path}, using default")
        config_path = Path(__file__).parent / "config" / "models.yaml"
    
    # Initialize model manager
    model_manager = ModelManager(config_path=str(config_path))
    set_model_manager(model_manager)
    set_openai_model_manager(model_manager)  # Also set for OpenAI routes
    
    # Log GPU info
    device = get_device()
    logger.info(f"Using device: {device}")
    gpu_info = get_gpu_info()
    if gpu_info.get("available"):
        logger.info(f"GPU: {gpu_info['devices'][0]['name']}")
        logger.info(f"Total VRAM: {gpu_info['devices'][0]['total_memory_gb']} GB")
    
    # Load default model if specified
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config = yaml.safe_load(f)
            default_model = config.get("default_model")
            if default_model:
                logger.info(f"Loading default model: {default_model}")
                try:
                    model_manager.load_model(default_model)
                    logger.info(f"Default model '{default_model}' loaded successfully")
                except Exception as e:
                    logger.warning(f"Failed to load default model: {e}")
    except Exception as e:
        logger.warning(f"Could not load default model: {e}")
    
    yield
    
    # Shutdown
    if model_manager and model_manager.is_model_loaded():
        logger.info("Unloading model on shutdown...")
        model_manager.unload_model()


# Create FastAPI app
app = FastAPI(
    title="Local LLM Inference Server",
    description="FastAPI server for local Hugging Face transformer models",
    version="1.0.0",
    lifespan=lifespan,
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API routes
app.include_router(router, prefix="/api", tags=["api"])
# Include OpenAI-compatible routes for OpenClaw and other clients
app.include_router(openai_router, tags=["openai"])

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Local LLM Inference Server",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "custom_api": "/api/*",
            "openai_compatible": "/v1/*",
            "openai_chat": "/v1/chat/completions",
            "openai_models": "/v1/models"
        }
    }


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Local LLM Inference Server")
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host to bind to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )
    parser.add_argument(
        "--ui",
        action="store_true",
        help="Launch Gradio UI in addition to API server",
    )
    parser.add_argument(
        "--ui-port",
        type=int,
        default=7860,
        help="Port for Gradio UI (default: 7860)",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=1,
        help="Number of uvicorn workers (default: 1)",
    )
    
    args = parser.parse_args()
    
    # Launch UI if requested
    if args.ui:
        import threading
        from app.ui.chat_ui import launch_ui
        
        def run_ui():
            launch_ui(server_name="127.0.0.1", server_port=args.ui_port)
        
        ui_thread = threading.Thread(target=run_ui, daemon=True)
        ui_thread.start()
        logger.info(f"Gradio UI starting on http://127.0.0.1:{args.ui_port}")
    
    # Run FastAPI server
    logger.info(f"Starting server on http://{args.host}:{args.port}")
    logger.info(f"API docs available at http://{args.host}:{args.port}/docs")
    
    uvicorn.run(
        "main:app",
        host=args.host,
        port=args.port,
        workers=args.workers,
        log_level="info",
    )


if __name__ == "__main__":
    main()
