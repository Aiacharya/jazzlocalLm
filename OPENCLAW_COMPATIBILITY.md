# OpenClaw Local Inference Compatibility Analysis

## Current API Status

Your API is **partially compatible** but needs adjustments to fully work with OpenClaw's local inference provider.

## Required Changes for OpenClaw Compatibility

### 1. **OpenAI-Compatible Endpoint Path**

**Current:** `POST /api/chat`  
**Required:** `POST /v1/chat/completions` (OpenAI standard)

OpenClaw expects the OpenAI API format. You need to add this endpoint.

### 2. **Request Format Changes**

**Current Request:**
```json
{
  "messages": [{"role": "user", "content": "Hello"}],
  "max_new_tokens": 512,
  "temperature": 0.7,
  "top_p": 0.9,
  "top_k": 50,
  "repetition_penalty": 1.1
}
```

**OpenAI-Compatible Request (Required):**
```json
{
  "model": "qwen2.5-3b-instruct",
  "messages": [{"role": "user", "content": "Hello"}],
  "max_tokens": 512,  // Note: max_tokens not max_new_tokens
  "temperature": 0.7,
  "top_p": 0.9,
  "stop": null,  // Optional stop sequences
  "stream": false  // Optional streaming
}
```

**Changes Needed:**
- ✅ `messages` format is already correct
- ❌ Rename `max_new_tokens` → `max_tokens`
- ❌ Add `model` parameter (can be optional, use currently loaded model)
- ❌ Remove `top_k` and `repetition_penalty` (not in OpenAI spec, but can be ignored)
- ❌ Add `stop` parameter support
- ❌ Add `stream` parameter support (optional, can default to false)

### 3. **Response Format Changes**

**Current Response:**
```json
{
  "role": "assistant",
  "content": "Hello! How can I help?",
  "model": "qwen2.5-3b-instruct",
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**OpenAI-Compatible Response (Required):**
```json
{
  "id": "chatcmpl-abc123",
  "object": "chat.completion",
  "created": 1677652288,
  "model": "qwen2.5-3b-instruct",
  "choices": [{
    "index": 0,
    "message": {
      "role": "assistant",
      "content": "Hello! How can I help?"
    },
    "finish_reason": "stop"
  }],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**Changes Needed:**
- ❌ Wrap response in `choices` array
- ❌ Add `id` field (generate unique ID)
- ❌ Add `object` field (set to "chat.completion")
- ❌ Add `created` timestamp
- ❌ Add `finish_reason` (usually "stop")
- ✅ `usage` format is correct
- ✅ `model` field is correct

### 4. **Additional Endpoints Required**

OpenClaw may also expect:

**Models List:**
- `GET /v1/models` - List available models

**Current:** `GET /api/models/list`  
**Needs:** OpenAI format response

### 5. **Error Response Format**

**Current:** FastAPI default error format  
**Required:** OpenAI-compatible error format

```json
{
  "error": {
    "message": "Error description",
    "type": "invalid_request_error",
    "code": 400
  }
}
```

## Implementation Priority

### High Priority (Required for Basic Compatibility)
1. ✅ Add `/v1/chat/completions` endpoint
2. ✅ Convert request format (max_tokens, model parameter)
3. ✅ Convert response format (choices array, id, object, created)
4. ✅ Add `/v1/models` endpoint

### Medium Priority (Better Compatibility)
5. Add `stop` parameter support
6. Add error response formatting
7. Add request validation

### Low Priority (Nice to Have)
8. Add streaming support (`stream: true`)
9. Add `logprobs` parameter
10. Add `presence_penalty` and `frequency_penalty` (map from repetition_penalty)

## Summary of Code Changes Needed

### Files to Modify:
1. **`app/api/routes.py`** - Add OpenAI-compatible endpoints
2. **`app/core/inference.py`** - Adjust response format
3. **`main.py`** - Add `/v1` route prefix

### New Endpoints to Add:
- `POST /v1/chat/completions` - Main chat endpoint
- `GET /v1/models` - List models
- `GET /v1/models/{model_id}` - Get model info (optional)

### Response Format Changes:
- Wrap in `choices` array
- Add `id`, `object`, `created` fields
- Add `finish_reason` field

## Testing OpenClaw Compatibility

After implementing changes, test with:

```bash
curl -X POST http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen2.5-3b-instruct",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100
  }'
```

Expected response should match OpenAI format exactly.

---

**Credits:**
- Learner way to provide brain to Openclaw
- 90Xboot.com Learning in Public
- Developed by Pankaj Tiwari https://www.linkedin.com/in/genai-guru-pankaj/
