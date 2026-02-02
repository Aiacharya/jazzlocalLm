# Copy This Prompt to Cursor Chat

```
I need to integrate OpenClaw with my local LLM inference server.

CONTEXT:
- LLM server is running at http://127.0.0.1:8000/api
- Model qwen2.5-3b-instruct is loaded and tested (all tests pass)
- API endpoints: POST /api/chat and POST /api/generate
- Existing helper class: OpenClawLLMClient in openclaw_integration.py

TASK:
1. Check if OpenClaw package is installed (pip list | findstr openclaw)
2. If not installed, determine the correct package name and install it
3. Create a working script that:
   - Uses OpenClaw to crawl URLs (start with 1-2 test URLs)
   - Extracts text content from crawled pages
   - Sends content to LLM API using the existing OpenClawLLMClient class
   - Processes responses (summarize, extract key info)
   - Saves results to JSON file

4. Test the integration with a simple example

REQUIREMENTS:
- Use the existing openclaw_integration.py helper functions
- Handle errors gracefully
- Show progress/logging
- Save results to openclaw_results.json

Start by checking what OpenClaw package is available and proceed from there.
```

---

**Credits:**
- Learner way to provide brain to OpenClaw
- 90Xboot.com Learning in Public
- Developed by Pankaj Tiwari https://www.linkedin.com/in/genai-guru-pankaj/
