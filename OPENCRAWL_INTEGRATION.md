# OpenCrawl Integration Guide

This guide shows how to connect your local LLM inference server with OpenCrawl for web content processing.

## Overview

Your LLM server exposes REST APIs that OpenCrawl can use to:
- Process crawled web content (summarization, extraction, Q&A)
- Generate prompts based on crawled data
- Analyze and transform web content

## Server Endpoints

Your server runs on `http://127.0.0.1:8000` with these key endpoints:

### 1. Chat Completion (Recommended for OpenCrawl)
```
POST http://127.0.0.1:8000/api/chat
```

### 2. Text Generation
```
POST http://127.0.0.1:8000/api/generate
```

### 3. Health Check
```
GET http://127.0.0.1:8000/api/health
```

## Integration Patterns

### Pattern 1: Process Crawled Content

Use OpenCrawl to fetch web content, then send it to your LLM for processing.

### Pattern 2: LLM-Guided Crawling

Use the LLM to generate search queries or analyze which URLs to crawl next.

### Pattern 3: Content Summarization Pipeline

Crawl → Extract → Summarize with LLM → Store results

## Quick Start

1. **Start your LLM server:**
   ```powershell
   python main.py
   ```

2. **Load a model (if not auto-loaded):**
   ```powershell
   curl -X POST "http://127.0.0.1:8000/api/models/load" `
     -H "Content-Type: application/json" `
     -d '{\"model_name\": \"qwen2.5-3b-instruct\"}'
   ```

3. **Test the integration:**
   ```powershell
   python opencrawl_integration.py
   ```

4. **Run full workflow example:**
   ```powershell
   python opencrawl_example.py
   ```

## Example Use Cases

- **Summarize crawled articles**
- **Extract key information from web pages**
- **Answer questions about crawled content**
- **Generate metadata from web content**
- **Translate or reformat crawled text**

## API Integration Code

### Basic Usage

```python
from opencrawl_integration import OpenCrawlLLMClient

# Initialize client
client = OpenCrawlLLMClient()

# Summarize content
summary = client.summarize_content(crawled_content)

# Answer questions
answer = client.answer_question(crawled_content, "What is the main topic?")

# Extract key info
info = client.extract_key_info(crawled_content)
```

### Direct API Calls

If you prefer direct HTTP calls:

```python
import requests

# Chat completion
response = requests.post(
    "http://127.0.0.1:8000/api/chat",
    json={
        "messages": [
            {"role": "user", "content": "Summarize this: [crawled content]"}
        ],
        "max_new_tokens": 200
    }
)
result = response.json()
print(result["content"])
```

## Integration Steps

1. **Ensure LLM server is running** (`python main.py`)
2. **Import the integration client** (`from opencrawl_integration import OpenCrawlLLMClient`)
3. **Get content from OpenCrawl** (via OpenCrawl API/SDK)
4. **Process with LLM** using the client methods
5. **Store/use results** as needed

## Configuration

Edit `opencrawl_integration.py` to change:
- `LLM_API_BASE`: Server URL (default: `http://127.0.0.1:8000/api`)
- Generation parameters (temperature, max_tokens, etc.)
- Processing logic for your specific use case

---

**Credits:**
- Learner way to provide brain to Openclaw
- 90Xboot.com Learning in Public
- Developed by Pankaj Tiwari https://www.linkedin.com/in/genai-guru-pankaj/
