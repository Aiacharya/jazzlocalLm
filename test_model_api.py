"""
Quick test script to verify model responses via API.
Bypasses Gradio UI to test the core functionality.
"""
import requests
import json
import time
import sys
import io

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

API_BASE = "http://127.0.0.1:8000/api"

def test_health():
    """Test server health."""
    print("=" * 60)
    print("1. Testing Server Health")
    print("=" * 60)
    try:
        response = requests.get(f"{API_BASE}/health", timeout=10)
        if response.status_code == 200:
            data = response.json()
            print(f"[OK] Server Status: {data.get('status')}")
            print(f"[OK] Model Loaded: {data.get('model_loaded')}")
            print(f"[OK] Loaded Model: {data.get('loaded_model')}")
            return True
        else:
            print(f"[FAIL] Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Health check error: {e}")
        return False

def test_chat_completion():
    """Test chat completion."""
    print("\n" + "=" * 60)
    print("2. Testing Chat Completion")
    print("=" * 60)
    
    messages = [
        {"role": "user", "content": "Hello! Can you introduce yourself briefly?"}
    ]
    
    print(f"Request: {messages[0]['content']}")
    print("\nSending request...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/chat",
            json={
                "messages": messages,
                "max_new_tokens": 200,
                "temperature": 0.7,
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Response received in {elapsed:.2f}s")
            print(f"\nResponse:")
            print("-" * 60)
            print(data.get("content", ""))
            print("-" * 60)
            print(f"\nModel: {data.get('model')}")
            if "usage" in data:
                usage = data["usage"]
                print(f"Tokens - Prompt: {usage.get('prompt_tokens')}, "
                      f"Completion: {usage.get('completion_tokens')}, "
                      f"Total: {usage.get('total_tokens')}")
            return True
        else:
            print(f"[FAIL] Chat failed: {response.status_code}")
            try:
                error = response.json()
                print(f"Error: {error.get('detail', error)}")
            except:
                print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Chat error: {e}")
        return False

def test_text_generation():
    """Test text generation."""
    print("\n" + "=" * 60)
    print("3. Testing Text Generation")
    print("=" * 60)
    
    prompt = "Explain machine learning in one sentence:"
    print(f"Prompt: {prompt}")
    print("\nSending request...")
    
    try:
        start_time = time.time()
        response = requests.post(
            f"{API_BASE}/generate",
            json={
                "prompt": prompt,
                "max_new_tokens": 100,
                "temperature": 0.7,
            },
            timeout=120
        )
        elapsed = time.time() - start_time
        
        if response.status_code == 200:
            data = response.json()
            print(f"\n✓ Response received in {elapsed:.2f}s")
            print(f"\nGenerated Text:")
            print("-" * 60)
            print(data.get("generated_text", ""))
            print("-" * 60)
            print(f"\nInput Tokens: {data.get('input_tokens')}")
            print(f"Output Tokens: {data.get('output_tokens')}")
            print(f"Total Tokens: {data.get('total_tokens')}")
            return True
        else:
            print(f"[FAIL] Generation failed: {response.status_code}")
            try:
                error = response.json()
                print(f"Error: {error.get('detail', error)}")
            except:
                print(f"Error: {response.text}")
            return False
    except Exception as e:
        print(f"[FAIL] Generation error: {e}")
        return False

def test_conversation():
    """Test multi-turn conversation."""
    print("\n" + "=" * 60)
    print("4. Testing Multi-Turn Conversation")
    print("=" * 60)
    
    conversation = [
        {"role": "user", "content": "What is Python?"},
    ]
    
    print("Turn 1:")
    print(f"User: {conversation[0]['content']}")
    
    try:
        response = requests.post(
            f"{API_BASE}/chat",
            json={
                "messages": conversation,
                "max_new_tokens": 150,
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            assistant_msg = data.get("content", "")
            print(f"Assistant: {assistant_msg[:100]}...")
            
            # Add to conversation
            conversation.append({"role": "assistant", "content": assistant_msg})
            conversation.append({"role": "user", "content": "Can you give me a code example?"})
            
            print("\nTurn 2:")
            print(f"User: {conversation[2]['content']}")
            
            response2 = requests.post(
                f"{API_BASE}/chat",
                json={
                    "messages": conversation,
                    "max_new_tokens": 200,
                },
                timeout=120
            )
            
            if response2.status_code == 200:
                data2 = response2.json()
                print(f"Assistant: {data2.get('content', '')[:150]}...")
                print("\n[OK] Multi-turn conversation works!")
                return True
            else:
                print(f"[FAIL] Turn 2 failed: {response2.status_code}")
                return False
        else:
            print(f"[FAIL] Turn 1 failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Conversation error: {e}")
        return False

def test_opencrawl_format():
    """Test format suitable for OpenCrawl integration."""
    print("\n" + "=" * 60)
    print("5. Testing OpenCrawl Integration Format")
    print("=" * 60)
    
    # Simulate crawled content
    crawled_content = """
    Artificial intelligence (AI) is transforming industries across the globe.
    From healthcare to finance, AI applications are revolutionizing how we work
    and live. Machine learning algorithms can now diagnose diseases, predict
    market trends, and even create art.
    """
    
    # Test summarization
    prompt = f"Summarize the following content in 2-3 sentences:\n\n{crawled_content.strip()}"
    
    print("Testing content summarization...")
    print(f"Content length: {len(crawled_content)} characters")
    
    try:
        response = requests.post(
            f"{API_BASE}/generate",
            json={
                "prompt": prompt,
                "max_new_tokens": 100,
                "temperature": 0.5,
            },
            timeout=120
        )
        
        if response.status_code == 200:
            data = response.json()
            summary = data.get("generated_text", "")
            print(f"\n[OK] Summary generated:")
            print("-" * 60)
            print(summary)
            print("-" * 60)
            print("\n[OK] Model is ready for OpenCrawl integration!")
            return True
        else:
            print(f"[FAIL] Summarization failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"[FAIL] Summarization error: {e}")
        return False

def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("Local LLM API Test Suite")
    print("=" * 60)
    print(f"Testing API at: {API_BASE}\n")
    
    results = []
    
    # Run tests
    results.append(("Health Check", test_health()))
    results.append(("Chat Completion", test_chat_completion()))
    results.append(("Text Generation", test_text_generation()))
    results.append(("Multi-Turn Conversation", test_conversation()))
    results.append(("OpenCrawl Format", test_opencrawl_format()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "[PASS]" if result else "[FAIL]"
        print(f"{status}: {name}")
    
    print(f"\nResults: {passed}/{total} tests passed")
    
    if passed == total:
        print("\n[SUCCESS] All tests passed! Model is ready for OpenCrawl integration.")
        print("\nYou can now use the model with OpenCrawl:")
        print("  python opencrawl_integration.py")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed. Please check the errors above.")
    
    return passed == total

if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted by user.")
        exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
