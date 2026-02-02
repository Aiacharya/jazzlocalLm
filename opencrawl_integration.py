"""
OpenCrawl Integration Script
Connects OpenCrawl with the local LLM inference server for content processing.
"""
import requests
import json
from typing import List, Dict, Any, Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# LLM Server Configuration
LLM_API_BASE = "http://127.0.0.1:8000/api"


class OpenCrawlLLMClient:
    """
    Client for integrating OpenCrawl with the local LLM inference server.
    """
    
    def __init__(self, api_base: str = LLM_API_BASE):
        """
        Initialize the client.
        
        Args:
            api_base: Base URL of the LLM inference server
        """
        self.api_base = api_base
        self._check_server_health()
    
    def _check_server_health(self) -> bool:
        """Check if the LLM server is running and ready."""
        try:
            response = requests.get(f"{self.api_base}/health", timeout=5)
            if response.status_code == 200:
                health = response.json()
                if health.get("model_loaded"):
                    logger.info(f"✓ LLM server ready. Model: {health.get('loaded_model')}")
                    return True
                else:
                    logger.warning("⚠ LLM server running but no model loaded")
                    return False
            return False
        except Exception as e:
            logger.error(f"✗ Cannot connect to LLM server: {e}")
            logger.error(f"  Make sure the server is running: python main.py")
            raise ConnectionError(f"Cannot connect to LLM server at {self.api_base}")
    
    def summarize_content(self, content: str, max_length: int = 200) -> str:
        """
        Summarize crawled web content.
        
        Args:
            content: The web content to summarize
            max_length: Maximum length of summary
        
        Returns:
            Summarized text
        """
        prompt = f"""Please provide a concise summary of the following content in {max_length} words or less:

{content[:2000]}  # Limit content to avoid token limits

Summary:"""
        
        return self.generate_text(prompt, max_new_tokens=max_length)
    
    def extract_key_info(self, content: str) -> Dict[str, Any]:
        """
        Extract key information from crawled content.
        
        Args:
            content: The web content to analyze
        
        Returns:
            Dictionary with extracted information
        """
        prompt = f"""Extract key information from the following content. Return a structured summary with:
- Main topic
- Key points (3-5 bullet points)
- Important entities (people, places, organizations)
- Date/time if mentioned

Content:
{content[:2000]}

Extracted Information:"""
        
        result = self.generate_text(prompt, max_new_tokens=300)
        
        # Try to parse structured output (basic implementation)
        return {
            "summary": result,
            "raw_content": content[:500]  # First 500 chars
        }
    
    def answer_question(self, content: str, question: str) -> str:
        """
        Answer a question based on crawled content.
        
        Args:
            content: The web content to search
            question: The question to answer
        
        Returns:
            Answer to the question
        """
        messages = [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on provided content."
            },
            {
                "role": "user",
                "content": f"""Based on the following content, please answer this question: {question}

Content:
{content[:3000]}

Answer:"""
            }
        ]
        
        return self.chat_completion(messages, max_new_tokens=200)
    
    def generate_metadata(self, content: str) -> Dict[str, str]:
        """
        Generate metadata (title, description, tags) from content.
        
        Args:
            content: The web content
        
        Returns:
            Dictionary with metadata
        """
        prompt = f"""Generate metadata for the following content:
1. A concise title (max 10 words)
2. A brief description (max 50 words)
3. 3-5 relevant tags/keywords

Content:
{content[:2000]}

Format your response as:
Title: [title]
Description: [description]
Tags: [tag1, tag2, tag3]"""
        
        result = self.generate_text(prompt, max_new_tokens=150)
        
        # Parse the result (basic implementation)
        return {
            "generated_metadata": result,
            "raw_content_preview": content[:200]
        }
    
    def process_crawled_urls(self, urls: List[str], operation: str = "summarize") -> List[Dict[str, Any]]:
        """
        Process multiple crawled URLs.
        
        Args:
            urls: List of URLs that were crawled
            operation: Operation to perform (summarize, extract, analyze)
        
        Returns:
            List of processed results
        """
        results = []
        
        for url in urls:
            try:
                # In a real implementation, you would fetch content from OpenCrawl here
                # For now, this is a placeholder
                logger.info(f"Processing {url}...")
                
                # Placeholder: In real usage, get content from OpenCrawl
                # content = opencrawl.get_content(url)
                content = f"Content from {url}"  # Placeholder
                
                if operation == "summarize":
                    result = self.summarize_content(content)
                elif operation == "extract":
                    result = self.extract_key_info(content)
                elif operation == "analyze":
                    result = self.generate_metadata(content)
                else:
                    result = {"error": f"Unknown operation: {operation}"}
                
                results.append({
                    "url": url,
                    "operation": operation,
                    "result": result
                })
                
            except Exception as e:
                logger.error(f"Error processing {url}: {e}")
                results.append({
                    "url": url,
                    "operation": operation,
                    "error": str(e)
                })
        
        return results
    
    def chat_completion(
        self,
        messages: List[Dict[str, str]],
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Send a chat completion request to the LLM server.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated response text
        """
        try:
            response = requests.post(
                f"{self.api_base}/chat",
                json={
                    "messages": messages,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("content", "")
        except Exception as e:
            logger.error(f"Chat completion error: {e}")
            raise
    
    def generate_text(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: Input text prompt
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Generated text
        """
        try:
            response = requests.post(
                f"{self.api_base}/generate",
                json={
                    "prompt": prompt,
                    "max_new_tokens": max_new_tokens,
                    "temperature": temperature,
                },
                timeout=120
            )
            response.raise_for_status()
            result = response.json()
            return result.get("generated_text", "")
        except Exception as e:
            logger.error(f"Text generation error: {e}")
            raise


def example_usage():
    """Example usage of OpenCrawl integration."""
    
    # Initialize client
    client = OpenCrawlLLMClient()
    
    # Example 1: Summarize content
    print("\n=== Example 1: Summarize Content ===")
    sample_content = """
    Artificial intelligence (AI) is transforming industries across the globe.
    From healthcare to finance, AI applications are revolutionizing how we work
    and live. Machine learning algorithms can now diagnose diseases, predict
    market trends, and even create art. The future of AI holds immense potential
    for solving complex problems and improving human life.
    """
    summary = client.summarize_content(sample_content)
    print(f"Summary: {summary}\n")
    
    # Example 2: Answer question about content
    print("=== Example 2: Answer Question ===")
    answer = client.answer_question(
        content=sample_content,
        question="What are some applications of AI?"
    )
    print(f"Answer: {answer}\n")
    
    # Example 3: Extract key information
    print("=== Example 3: Extract Key Info ===")
    info = client.extract_key_info(sample_content)
    print(f"Extracted Info: {json.dumps(info, indent=2)}\n")
    
    # Example 4: Generate metadata
    print("=== Example 4: Generate Metadata ===")
    metadata = client.generate_metadata(sample_content)
    print(f"Metadata: {json.dumps(metadata, indent=2)}\n")


def integrate_with_opencrawl_api(opencrawl_urls: List[str]):
    """
    Integration function to use with OpenCrawl.
    
    Args:
        opencrawl_urls: List of URLs from OpenCrawl to process
    """
    client = OpenCrawlLLMClient()
    
    # Process each URL
    results = client.process_crawled_urls(
        urls=opencrawl_urls,
        operation="summarize"  # or "extract", "analyze"
    )
    
    return results


if __name__ == "__main__":
    # Run examples
    try:
        example_usage()
    except ConnectionError as e:
        print(f"\nError: {e}")
        print("\nMake sure your LLM server is running:")
        print("  python main.py")
    except Exception as e:
        print(f"\nUnexpected error: {e}")
