"""
Test OpenAI/OpenClaw compatibility.
"""
import requests
import json

API_BASE = "http://127.0.0.1:8000"

def test_openai_chat():
    """Test OpenAI-compatible chat endpoint."""
    print("=" * 60)
    print("Testing OpenAI-Compatible Chat Endpoint")
    print("=" * 60)
    
    request_data = {
        "model": "qwen2.5-3b-instruct",
        "messages": [
            {"role": "user", "content": "Hello! Say hi in one sentence."}
        ],
        "max_tokens": 50,
        "temperature": 0.7
    }
    
    print(f"\nRequest:")
    print(json.dumps(request_data, indent=2))
    
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            json=request_data,
            timeout=120
        )
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse:")
            print(json.dumps(data, indent=2))
            
            # Validate OpenAI format
            required_fields = ["id", "object", "created", "model", "choices", "usage"]
            missing = [f for f in required_fields if f not in data]
            
            if missing:
                print(f"\n[FAIL] Missing required fields: {missing}")
                return False
            
            if "choices" in data and len(data["choices"]) > 0:
                choice = data["choices"][0]
                if "message" in choice and "content" in choice["message"]:
                    print(f"\n[OK] OpenAI format validated!")
                    print(f"Response: {choice['message']['content']}")
                    return True
                else:
                    print(f"\n[FAIL] Invalid choice format")
                    return False
            else:
                print(f"\n[FAIL] No choices in response")
                return False
        else:
            print(f"\n[FAIL] Request failed")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_openai_models():
    """Test OpenAI-compatible models endpoint."""
    print("\n" + "=" * 60)
    print("Testing OpenAI-Compatible Models Endpoint")
    print("=" * 60)
    
    try:
        response = requests.get(f"{API_BASE}/v1/models", timeout=10)
        
        print(f"\nStatus Code: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"\nResponse:")
            print(json.dumps(data, indent=2))
            
            # Validate format
            if "object" in data and data["object"] == "list":
                if "data" in data and isinstance(data["data"], list):
                    print(f"\n[OK] Found {len(data['data'])} models")
                    for model in data["data"]:
                        print(f"  - {model.get('id', 'unknown')}")
                    return True
                else:
                    print(f"\n[FAIL] Invalid data format")
                    return False
            else:
                print(f"\n[FAIL] Invalid response format")
                return False
        else:
            print(f"\n[FAIL] Request failed")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"\n[FAIL] Error: {e}")
        return False


def test_backward_compatibility():
    """Test that old API endpoints still work."""
    print("\n" + "=" * 60)
    print("Testing Backward Compatibility (Old API)")
    print("=" * 60)
    
    try:
        # Test old chat endpoint
        response = requests.post(
            f"{API_BASE}/api/chat",
            json={
                "messages": [{"role": "user", "content": "Test"}],
                "max_new_tokens": 10
            },
            timeout=120
        )
        
        if response.status_code == 200:
            print("[OK] Old /api/chat endpoint still works")
            return True
        else:
            print(f"[FAIL] Old endpoint failed: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"[FAIL] Error: {e}")
        return False


def main():
    """Run all tests."""
    print("\n" + "=" * 60)
    print("OpenAI/OpenClaw Compatibility Test Suite")
    print("=" * 60)
    
    results = []
    
    results.append(("OpenAI Chat Endpoint", test_openai_chat()))
    results.append(("OpenAI Models Endpoint", test_openai_models()))
    results.append(("Backward Compatibility", test_backward_compatibility()))
    
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
        print("\n[SUCCESS] All tests passed! API is OpenAI/OpenClaw compatible.")
        print("\nYou can now use this server with OpenClaw:")
        print("  Base URL: http://127.0.0.1:8000/v1")
        print("  Chat: POST /v1/chat/completions")
        print("  Models: GET /v1/models")
    else:
        print(f"\n[WARN] {total - passed} test(s) failed.")
    
    return passed == total


if __name__ == "__main__":
    try:
        success = main()
        exit(0 if success else 1)
    except KeyboardInterrupt:
        print("\n\nTest interrupted.")
        exit(1)
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        import traceback
        traceback.print_exc()
        exit(1)
