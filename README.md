# Jazz Local LM Inference Server
https://github.com/Aiacharya/jazzlocalLm

A production-ready FastAPI-based inference server for local Hugging Face transformer models, optimized for GPU usage on Windows.

**Learner way to provide brain to OpenClaw**  
**90Xboot.com Learning in Public**  
**Developed by Pankaj Tiwari** - https://www.linkedin.com/in/genai-guru-pankaj/

## ⚠️ Important Windows Requirements

- **Python 3.11.9** is explicitly required for this project
- **CUDA Compatibility**: If you have CUDA installed, you must install the matching PyTorch version for your CUDA version
  - Check your CUDA version: `nvidia-smi`
  - Visit [PyTorch Installation](https://pytorch.org/get-started/locally/) to get the correct PyTorch command for your CUDA version
  - Example: For CUDA 11.8, use: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`
  - Example: For CUDA 12.1, use: `pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121`

## Features

- 🚀 **FastAPI REST API** for model management and inference
- 🎯 **GPU-optimized** with CUDA support and VRAM management
- 🔄 **Dynamic model loading/unloading** with memory safety checks
- 📦 **Multiple model support** - load, unload, and switch models at runtime
- 🎨 **Lightweight Gradio UI** for testing and debugging
- ⚙️ **Configuration-driven** architecture
- 🛡️ **Production-ready** with proper error handling and logging
- 🔌 **OpenAI-Compatible API** - Works with OpenClaw and other OpenAI-compatible clients
- 🌐 **Dual API Support** - Custom API endpoints + OpenAI-compatible `/v1/*` endpoints

## Requirements

- **Python 3.11.9** (explicitly required - other versions may work but not tested)
- **Windows 10/11** (native support)
- **NVIDIA GPU** with CUDA support (recommended, CPU fallback available)
- **CUDA Toolkit** (for GPU acceleration)
  - **Important**: PyTorch version must match your CUDA version
  - Check CUDA version: `nvidia-smi`
  - Install matching PyTorch from [pytorch.org](https://pytorch.org/get-started/locally/)

## Installation

### 1. Install Python 3.11.9

Download and install Python 3.11.9 from [python.org](https://www.python.org/downloads/).

### 2. Verify CUDA Installation

Check if CUDA is available:
```powershell
nvidia-smi
```

### 3. Install PyTorch with CUDA Support

**⚠️ Critical**: You must install PyTorch separately with the correct CUDA version for your system.

**Step 1: Check your CUDA version**
```powershell
nvidia-smi
```
Look for "CUDA Version" in the output (e.g., 11.8, 12.1, 12.8)

**Step 2: Install PyTorch matching your CUDA version**

Visit https://pytorch.org/get-started/locally/ and select:
- OS: Windows
- Package: Pip
- Language: Python
- Compute Platform: Your CUDA version

**Examples:**
```powershell
# For CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

**Step 3: Verify PyTorch CUDA installation**
```powershell
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
```

### 4. Install Other Dependencies

```powershell
# Create virtual environment (recommended)
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install other dependencies (PyTorch already installed above)
pip install -r requirements.txt
```

**Note:** `requirements.txt` does not include PyTorch - you must install it separately with the correct CUDA version.

### 5. Verify Installation

```powershell
python verify_setup.py
```

This will check:
- Python version
- PyTorch and CUDA installation
- All required dependencies
- Configuration file

### 6. Pre-Download Models (Recommended)

To avoid downloading models on every server run, pre-download them to local cache:

```powershell
# Download all models from config
python download_models.py

# Download only the default model
python download_models.py --model qwen2.5-3b-instruct

# Force re-download (if needed)
python download_models.py --force
```

Models will be cached in `~/.cache/huggingface/hub/` and won't be re-downloaded on subsequent runs.

## Project Structure

```
jazzlocalLm/
├── app/
│   ├── api/
│   │   ├── routes.py          # Custom FastAPI routes
│   │   └── openai_routes.py   # OpenAI-compatible routes (NEW)
│   ├── core/
│   │   ├── model_manager.py   # Model lifecycle management
│   │   ├── inference.py       # Text generation logic
│   │   └── gpu_utils.py       # GPU utilities and VRAM checks
│   └── ui/
│       └── chat_ui.py         # Gradio chat interface
├── config/
│   └── models.yaml            # Model registry configuration
├── main.py                    # Application entry point
├── requirements.txt           # Python dependencies (PyTorch excluded - install separately)
├── download_models.py        # Pre-download models script
├── test_model_api.py         # API testing script
├── test_openai_compatibility.py  # OpenAI compatibility test (NEW)
└── README.md                  # This file
```

## Configuration

Edit `config/models.yaml` to add or modify models:

```yaml
models:
  - name: "qwen2.5-3b-instruct"
    hf_id: "Qwen/Qwen2.5-3B-Instruct"
    description: "Qwen 2.5 3B Instruct model"
    max_length: 4096
    dtype: "float16"  # float16, bfloat16, or float32

default_model: "qwen2.5-3b-instruct"  # Optional: auto-load on startup
```

## Usage

### Start the Server

**API only:**
```powershell
python main.py
```

**API + UI:**
```powershell
python main.py --ui
```

**Custom host/port:**
```powershell
python main.py --host 0.0.0.0 --port 8000 --ui --ui-port 7860
```

### Access Points

- **API Server:** http://127.0.0.1:8000
- **API Documentation:** http://127.0.0.1:8000/docs
- **Gradio UI:** http://127.0.0.1:7860 (if `--ui` flag is used)

## API Endpoints

### Custom API (Original Endpoints)

**Health & Status:**
- `GET /api/health` - Health check and model status
- `GET /api/gpu` - GPU memory and status information

**Model Management:**
- `GET /api/models/list` - List all available models
- `POST /api/models/load` - Load a model
- `POST /api/models/unload` - Unload current model
- `POST /api/models/switch` - Switch to a different model

**Inference:**
- `POST /api/chat` - Chat completion endpoint
- `POST /api/generate` - Text generation endpoint

### OpenAI-Compatible API (For OpenClaw & Other Clients)

**New endpoints for OpenAI-compatible clients:**

- `POST /v1/chat/completions` - OpenAI-compatible chat completion
- `GET /v1/models` - List models (OpenAI format)
- `GET /v1/models/{model_id}` - Get model information

**OpenClaw Configuration:**
- Base URL: `http://127.0.0.1:8000/v1`
- API Type: OpenAI-compatible
- Model: `qwen2.5-3b-instruct` (or any model from config)

See [OPENCLAW_IMPLEMENTATION.md](OPENCLAW_IMPLEMENTATION.md) for detailed integration guide.

## Example API Usage

### Load a Model

```powershell
curl -X POST "http://127.0.0.1:8000/api/models/load" `
  -H "Content-Type: application/json" `
  -d '{"model_name": "qwen2.5-3b-instruct"}'
```

### Chat Completion

```powershell
curl -X POST "http://127.0.0.1:8000/api/chat" `
  -H "Content-Type: application/json" `
  -d '{
    "messages": [
      {"role": "user", "content": "What is machine learning?"}
    ],
    "max_new_tokens": 512,
    "temperature": 0.7
  }'
```

### Text Generation

```powershell
curl -X POST "http://127.0.0.1:8000/api/generate" `
  -H "Content-Type: application/json" `
  -d '{
    "prompt": "Once upon a time",
    "max_new_tokens": 256,
    "temperature": 0.8
  }'
```

### Check GPU Status

```powershell
curl "http://127.0.0.1:8000/api/gpu"
```

### List Available Models

```powershell
curl "http://127.0.0.1:8000/api/models/list"
```

### OpenAI-Compatible Chat (For OpenClaw)

```powershell
curl -X POST "http://127.0.0.1:8000/v1/chat/completions" `
  -H "Content-Type: application/json" `
  -d '{
    "model": "qwen2.5-3b-instruct",
    "messages": [{"role": "user", "content": "Hello!"}],
    "max_tokens": 100
  }'
```

### List Models (OpenAI Format)

```powershell
curl "http://127.0.0.1:8000/v1/models"
```

## Using the Gradio UI

1. Start the server with `--ui` flag
2. Open http://127.0.0.1:7860 in your browser
3. Select a model from the dropdown
4. Click "Load Model" and wait for it to load
5. Start chatting!

The UI shows:
- Available models
- Currently loaded model
- GPU memory status
- Chat interface for testing

## Model Memory Requirements

Approximate VRAM requirements:
- **Qwen2.5-1.5B:** ~3-4 GB
- **Qwen2.5-3B:** ~6-8 GB
- **Qwen2.5-7B:** ~14-16 GB

The server automatically checks available VRAM before loading models to prevent OOM errors.

## Architecture

### Model Manager

- Maintains registry of available models
- Handles dynamic loading/unloading
- Prevents OOM by checking VRAM
- Supports fp16, bf16, and fp32

### Inference Engine

- Handles text generation
- Supports chat completion format
- Configurable generation parameters
- Proper tokenization and decoding

### GPU Utils

- Device detection (CUDA/CPU)
- VRAM monitoring
- Memory estimation
- Cache management

## Development

### Code Standards

- Type hints throughout
- Comprehensive logging
- Clean separation of concerns
- Async FastAPI where appropriate
- Error handling and validation

### Adding New Models

1. Add model configuration to `config/models.yaml`
2. Ensure you have sufficient VRAM
3. Models will auto-download from Hugging Face on first load

## Troubleshooting

### CUDA Not Available

- **Verify NVIDIA drivers**: `nvidia-smi` should show your GPU
- **Check CUDA toolkit installation**: Ensure CUDA toolkit matches your driver version
- **Verify PyTorch CUDA build**: 
  ```powershell
  python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"
  ```
- **Common issue**: PyTorch installed without CUDA support - reinstall with correct CUDA version
- **CUDA version mismatch**: PyTorch CUDA version must match your system CUDA version

### Out of Memory (OOM)

- Use a smaller model
- Reduce `max_length` in config
- Use `float16` or `bfloat16` instead of `float32`
- Close other GPU applications

### Model Download Issues

- Check internet connection
- Verify Hugging Face model ID is correct
- Check disk space for model cache

## OpenAI/OpenClaw Compatibility

This server now supports **OpenAI-compatible API endpoints** for integration with OpenClaw and other OpenAI-compatible clients.

### Quick Setup for OpenClaw

1. **Start the server:**
   ```powershell
   python main.py
   ```

2. **Configure OpenClaw:**
   - Base URL: `http://127.0.0.1:8000/v1`
   - API Type: OpenAI-compatible
   - Model: `qwen2.5-3b-instruct` (or any model from your config)

3. **Test compatibility:**
   ```powershell
   python test_openai_compatibility.py
   ```

### API Endpoints

- **Custom API**: `/api/*` - Original endpoints (still fully functional)
- **OpenAI API**: `/v1/*` - OpenAI-compatible endpoints for OpenClaw

Both API sets work simultaneously!

For detailed integration guide, see [OPENCLAW_IMPLEMENTATION.md](OPENCLAW_IMPLEMENTATION.md)

## License

This project is provided as-is for local inference use.

## Contributing

This is a production-ready template. Customize as needed for your use case.

---

**Credits:**
- Learner way to provide brain to OpenClaw
- 90Xboot.com Learning in Public
- Developed by Pankaj Tiwari - https://www.linkedin.com/in/genai-guru-pankaj/
