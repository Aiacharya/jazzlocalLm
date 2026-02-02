"""
Setup verification script.
Checks if all dependencies and CUDA are properly installed.
"""
import sys
import subprocess
import io

# Set UTF-8 encoding for Windows console
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

def check_python_version():
    """Check Python version."""
    version = sys.version_info
    print(f"Python version: {version.major}.{version.minor}.{version.micro}")
    
    if version.major == 3 and version.minor == 11:
        print("[OK] Python 3.11 detected")
        return True
    else:
        print(f"[WARN] Python 3.11.9 is recommended, but {version.major}.{version.minor}.{version.micro} detected")
        return False


def check_torch():
    """Check PyTorch installation and CUDA support."""
    try:
        import torch
        print(f"\nPyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"[OK] CUDA available: {torch.version.cuda}")
            print(f"[OK] GPU: {torch.cuda.get_device_name(0)}")
            print(f"[OK] CUDA device count: {torch.cuda.device_count()}")
            return True
        else:
            print("[WARN] CUDA not available - will fall back to CPU")
            return False
    except ImportError:
        print("[FAIL] PyTorch not installed")
        return False


def check_dependencies():
    """Check other required dependencies."""
    dependencies = [
        "transformers",
        "accelerate",
        "safetensors",
        "fastapi",
        "uvicorn",
        "pydantic",
        "yaml",
        "gradio",
    ]
    
    print("\nChecking dependencies:")
    all_ok = True
    
    for dep in dependencies:
        try:
            if dep == "yaml":
                import yaml
                print(f"[OK] {dep} (pyyaml)")
            else:
                module = __import__(dep)
                version = getattr(module, "__version__", "unknown")
                print(f"[OK] {dep} (version: {version})")
        except ImportError:
            print(f"[FAIL] {dep} not installed")
            all_ok = False
    
    return all_ok


def check_config():
    """Check if config file exists."""
    from pathlib import Path
    config_path = Path("config/models.yaml")
    
    if config_path.exists():
        print(f"\n[OK] Config file found: {config_path}")
        return True
    else:
        print(f"\n[FAIL] Config file not found: {config_path}")
        return False


def main():
    """Run all checks."""
    print("=" * 50)
    print("Local LLM Inference Server - Setup Verification")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("PyTorch & CUDA", check_torch),
        ("Dependencies", check_dependencies),
        ("Configuration", check_config),
    ]
    
    results = []
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n[FAIL] Error checking {name}: {e}")
            results.append((name, False))
    
    print("\n" + "=" * 50)
    print("Summary:")
    print("=" * 50)
    
    all_passed = True
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
        if not result:
            all_passed = False
    
    print("=" * 50)
    
    if all_passed:
        print("\n[SUCCESS] All checks passed! You're ready to run the server.")
        print("\nTo start the server:")
        print("  python main.py")
        print("\nTo start with UI:")
        print("  python main.py --ui")
    else:
        print("\n[WARN] Some checks failed. Please review the errors above.")
        print("\nInstallation help:")
        print("  1. Install PyTorch with CUDA: https://pytorch.org/get-started/locally/")
        print("  2. Install dependencies: pip install -r requirements.txt")
        print("  3. Verify config file exists: config/models.yaml")


if __name__ == "__main__":
    main()
