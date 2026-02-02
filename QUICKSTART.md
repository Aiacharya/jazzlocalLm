# Quick Start Guide

## 1. Verify Setup

```powershell
python verify_setup.py
```

This will check:
- Python version (3.11.9 recommended)
- PyTorch and CUDA installation
- All required dependencies
- Configuration file

## 2. Install PyTorch with CUDA

**Important:** Install PyTorch separately first, as it requires specific CUDA version matching.

Visit https://pytorch.org/get-started/locally/ and select:
- Your OS: Windows
- Package: Pip
- Language: Python
- Compute Platform: Your CUDA version (e.g., CUDA 11.8 or 12.1)

Example for CUDA 11.8:
```powershell
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

## 3. Install Other Dependencies

```powershell
pip install -r requirements.txt
```

## 4. Pre-Download Models (Optional but Recommended)

To avoid downloading models on every server run:

```powershell
# Download all models from config
python download_models.py

# Or download just the default model
python download_models.py --model qwen2.5-3b-instruct
```

Models are cached in `~/.cache/huggingface/hub/` and won't be re-downloaded.

## 5. Start the Server

**API only:**
```powershell
python main.py
```

**API + UI:**
```powershell
python main.py --ui
```

## 6. Test the API

### Using curl (PowerShell):

```powershell
# Check health
curl http://127.0.0.1:8000/api/health

# List models
curl http://127.0.0.1:8000/api/models/list

# Load a model
curl -X POST "http://127.0.0.1:8000/api/models/load" `
  -H "Content-Type: application/json" `
  -d '{\"model_name\": \"qwen2.5-3b-instruct\"}'

# Chat completion
curl -X POST "http://127.0.0.1:8000/api/chat" `
  -H "Content-Type: application/json" `
  -d '{\"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'
```

### Using Python:

```powershell
python example_client.py
```

## 7. Access Points

- **API Server:** http://127.0.0.1:8000
- **API Docs (Swagger):** http://127.0.0.1:8000/docs
- **Gradio UI:** http://127.0.0.1:7860 (if `--ui` flag used)

## 8. Troubleshooting

### CUDA Not Available

1. Check NVIDIA drivers: `nvidia-smi`
2. Verify PyTorch CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
3. Reinstall PyTorch with correct CUDA version

### Out of Memory

- Use a smaller model (e.g., 1.5B instead of 7B)
- Close other GPU applications
- Reduce `max_length` in `config/models.yaml`

### Model Download Issues

- Check internet connection
- Verify Hugging Face model ID in `config/models.yaml`
- Check disk space (models can be several GB)

## 9. Next Steps

- Customize `config/models.yaml` to add your models
- Adjust generation parameters in API calls
- Explore the API documentation at http://127.0.0.1:8000/docs

---

**Credits:**
- Learner way to provide brain to OpenClaw
- 90Xboot.com Learning in Public
- Developed by Pankaj Tiwari https://www.linkedin.com/in/genai-guru-pankaj/
