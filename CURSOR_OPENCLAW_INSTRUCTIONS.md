# OpenClaw + Local LLM Integration Instructions

## Goal
Integrate OpenClaw with the local LLM inference server to process crawled web content.

## Current Setup
- **LLM Server**: Running on `http://127.0.0.1:8000/api`
- **Model**: `qwen2.5-3b-instruct` (loaded and tested)
- **API Status**: ✅ All tests passing
- **Test Script**: `test_model_api.py` confirms API is working

## Integration Steps

### Step 1: Install OpenClaw (if not already installed)
```powershell
pip install openclaw
# OR if using a different package name, install accordingly
```

### Step 2: Create OpenClaw Integration Script

Create a script that:
1. Uses OpenClaw to fetch web content
2. Sends content to the LLM API for processing
3. Handles responses and stores results

**Key API Endpoints to Use:**
- `POST http://127.0.0.1:8000/api/chat` - For chat/completion
- `POST http://127.0.0.1:8000/api/generate` - For text generation
- `GET http://127.0.0.1:8000/api/health` - Check server status

### Step 3: Example Integration Pattern

```python
import requests
from openclaw import OpenClaw  # Adjust import based on actual package

LLM_API = "http://127.0.0.1:8000/api"

# 1. Crawl content with OpenClaw
crawler = OpenClaw()
urls = ["https://example.com/article"]
crawled_data = crawler.crawl(urls)

# 2. Process each crawled page with LLM
for page in crawled_data:
    content = page.content
    
    # Summarize with LLM
    response = requests.post(
        f"{LLM_API}/generate",
        json={
            "prompt": f"Summarize this article in 3 sentences:\n\n{content}",
            "max_new_tokens": 150,
            "temperature": 0.5
        },
        timeout=120
    )
    
    summary = response.json()["generated_text"]
    print(f"URL: {page.url}")
    print(f"Summary: {summary}\n")
```

### Step 4: Use Existing Integration Helper

The project already has `openclaw_integration.py` with helper functions:
- `OpenClawLLMClient` class
- `summarize_content()` method
- `extract_key_info()` method
- `answer_question()` method

**Quick Start:**
```python
from openclaw_integration import OpenClawLLMClient

client = OpenClawLLMClient()

# Get content from OpenClaw (replace with actual OpenClaw code)
content = "crawled web content here"

# Process with LLM
summary = client.summarize_content(content)
```

## What to Ask Cursor

**Copy this prompt to Cursor:**

```
I need to integrate OpenClaw with my local LLM server. 

The LLM API is running at http://127.0.0.1:8000/api and is tested and working.

Please:
1. Check if OpenClaw is installed, if not install it
2. Review the existing openclaw_integration.py file
3. Create or update a working script that:
   - Uses OpenClaw to crawl URLs
   - Sends crawled content to the LLM API for summarization/processing
   - Handles errors gracefully
   - Saves results to a file

The LLM API endpoints are:
- POST /api/chat - for chat completion
- POST /api/generate - for text generation

Test the integration with a simple example URL.
```

## Testing

After integration, test with:
```powershell
python test_model_api.py  # Verify LLM is working
python openclaw_example.py  # Test OpenClaw integration
```

## Expected Workflow

1. **Crawl**: OpenClaw fetches content from URLs
2. **Process**: Send content to LLM API (`/api/generate` or `/api/chat`)
3. **Extract**: Get summary/key info from LLM response
4. **Store**: Save processed results

## API Request Format

**For Text Generation:**
```json
{
  "prompt": "Your prompt here with {crawled_content}",
  "max_new_tokens": 200,
  "temperature": 0.7
}
```

**For Chat Completion:**
```json
{
  "messages": [
    {"role": "user", "content": "Summarize: {crawled_content}"}
  ],
  "max_new_tokens": 200
}
```

## Notes
- LLM server must be running: `python main.py`
- Model is already loaded and tested
- API response time: ~1-2 seconds per request
- Use timeout=120 for API requests
- Handle rate limiting if processing many URLs

---

**Credits:**
- Learner way to provide brain to OpenClaw
- 90Xboot.com Learning in Public
- Developed by Pankaj Tiwari https://www.linkedin.com/in/genai-guru-pankaj/
