"""
Pre-download models to local cache.
This ensures models are available locally and won't be downloaded on every run.
"""
import logging
import yaml
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForCausalLM
from app.core.gpu_utils import get_dtype

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def is_model_cached(hf_id: str) -> bool:
    """
    Check if a model is already cached locally.
    
    Args:
        hf_id: Hugging Face model identifier
    
    Returns:
        True if model appears to be cached
    """
    try:
        # Try to load tokenizer with local_files_only
        AutoTokenizer.from_pretrained(
            hf_id,
            local_files_only=True,
            trust_remote_code=True,
        )
        return True
    except Exception:
        return False


def download_model(hf_id: str, model_name: str, dtype: str = "float16", force: bool = False):
    """
    Download a model and tokenizer to local cache.
    
    Args:
        hf_id: Hugging Face model identifier
        model_name: Friendly name for logging
        dtype: Data type (float16, bfloat16, float32)
        force: Force re-download even if cached
    """
    # Check if already cached
    if not force and is_model_cached(hf_id):
        logger.info(f"✓ Model already cached: {model_name} ({hf_id})")
        logger.info(f"  Skipping download. Use --force to re-download.")
        return True
    
    logger.info(f"Downloading model: {model_name} ({hf_id})")
    logger.info(f"Model will be cached in: ~/.cache/huggingface/hub/")
    
    try:
        # Download tokenizer (small, fast)
        logger.info(f"Downloading tokenizer for {hf_id}...")
        tokenizer = AutoTokenizer.from_pretrained(
            hf_id,
            trust_remote_code=True,
        )
        logger.info(f"✓ Tokenizer downloaded for {model_name}")
        
        # Download model (this is the large download)
        logger.info(f"Downloading model {hf_id} (this may take a while)...")
        logger.info(f"Using dtype: {dtype}")
        
        model_dtype = get_dtype(dtype)
        
        # Download model without loading into memory (just cache it)
        # We use from_pretrained with local_files_only=False to ensure download
        model = AutoModelForCausalLM.from_pretrained(
            hf_id,
            dtype=model_dtype,  # Use dtype instead of deprecated torch_dtype
            trust_remote_code=True,
            low_cpu_mem_usage=True,
        )
        
        logger.info(f"✓ Model downloaded and cached: {model_name}")
        logger.info(f"Model cache location: {Path.home() / '.cache' / 'huggingface' / 'hub'}")
        
        # Clean up memory
        del model
        del tokenizer
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True
        
    except Exception as e:
        logger.error(f"Failed to download {model_name}: {e}", exc_info=True)
        return False


def main():
    """Download all models from configuration."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Pre-download models to local cache")
    parser.add_argument(
        "--model",
        type=str,
        help="Download only a specific model by name",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if model is already cached",
    )
    
    args = parser.parse_args()
    
    config_path = Path("config/models.yaml")
    
    if not config_path.exists():
        logger.error(f"Config file not found: {config_path}")
        return
    
    # Load configuration
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    
    models = config.get("models", [])
    
    if not models:
        logger.warning("No models found in configuration")
        return
    
    # Filter to specific model if requested
    if args.model:
        models = [m for m in models if m.get("name") == args.model]
        if not models:
            logger.error(f"Model '{args.model}' not found in configuration")
            return
    
    logger.info("=" * 60)
    logger.info("Model Pre-Download Script")
    logger.info("=" * 60)
    logger.info(f"Found {len(models)} model(s) to process")
    logger.info("")
    
    # Show cache location
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    logger.info(f"Models will be cached in: {cache_dir}")
    logger.info("")
    
    # Download each model
    results = []
    for i, model_config in enumerate(models, 1):
        model_name = model_config.get("name", "unknown")
        hf_id = model_config.get("hf_id", "")
        dtype = model_config.get("dtype", "float16")
        
        if not hf_id:
            logger.warning(f"Skipping {model_name}: no hf_id specified")
            continue
        
        logger.info(f"[{i}/{len(models)}] Processing: {model_name}")
        logger.info("-" * 60)
        
        success = download_model(hf_id, model_name, dtype, force=args.force)
        results.append((model_name, success))
        
        logger.info("")
    
    # Summary
    logger.info("=" * 60)
    logger.info("Download Summary")
    logger.info("=" * 60)
    
    for model_name, success in results:
        status = "[OK] SUCCESS" if success else "[FAIL] FAILED"
        logger.info(f"{status}: {model_name}")
    
    successful = sum(1 for _, success in results if success)
    logger.info("")
    logger.info(f"Successfully processed: {successful}/{len(results)} model(s)")
    logger.info("")
    logger.info("Models are now cached and will not be re-downloaded on server startup.")
    logger.info("You can now run the server with: python main.py")


if __name__ == "__main__":
    main()
