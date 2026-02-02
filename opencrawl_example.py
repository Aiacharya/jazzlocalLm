"""
Complete example: OpenCrawl → LLM Processing Pipeline

This shows a full workflow of how to integrate OpenCrawl with your LLM server.
"""
import requests
from opencrawl_integration import OpenCrawlLLMClient
import json

# LLM Server must be running: python main.py
LLM_API = "http://127.0.0.1:8000/api"


def crawl_and_process_workflow():
    """
    Complete workflow: Crawl → Process → Store
    
    This is a template - replace OpenCrawl API calls with actual OpenCrawl integration.
    """
    
    # Step 1: Initialize LLM client
    print("Initializing LLM client...")
    llm_client = OpenCrawlLLMClient()
    
    # Step 2: Simulate getting URLs from OpenCrawl
    # In real usage, you would call OpenCrawl API here
    print("\n=== Step 1: Get URLs from OpenCrawl ===")
    crawled_urls = [
        "https://example.com/article1",
        "https://example.com/article2",
        "https://example.com/article3",
    ]
    print(f"Found {len(crawled_urls)} URLs to process")
    
    # Step 3: Process each URL with LLM
    print("\n=== Step 2: Process Content with LLM ===")
    processed_results = []
    
    for url in crawled_urls:
        print(f"\nProcessing: {url}")
        
        # In real usage, fetch content from OpenCrawl
        # content = opencrawl.get_page_content(url)
        # For demo, using placeholder
        content = f"""
        This is sample content from {url}. 
        In a real implementation, this would be the actual crawled content
        from OpenCrawl. The LLM will process this content to extract insights,
        summarize, or answer questions about it.
        """
        
        # Process with LLM
        try:
            # Summarize
            summary = llm_client.summarize_content(content, max_length=100)
            
            # Extract key info
            key_info = llm_client.extract_key_info(content)
            
            # Generate metadata
            metadata = llm_client.generate_metadata(content)
            
            processed_results.append({
                "url": url,
                "summary": summary,
                "key_info": key_info,
                "metadata": metadata,
                "status": "success"
            })
            
            print(f"  ✓ Processed successfully")
            
        except Exception as e:
            print(f"  ✗ Error: {e}")
            processed_results.append({
                "url": url,
                "status": "error",
                "error": str(e)
            })
    
    # Step 4: Display results
    print("\n=== Step 3: Results ===")
    for result in processed_results:
        print(f"\nURL: {result['url']}")
        if result['status'] == 'success':
            print(f"Summary: {result['summary'][:100]}...")
        else:
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    # Step 5: Save results (optional)
    print("\n=== Step 4: Saving Results ===")
    with open("opencrawl_processed_results.json", "w", encoding="utf-8") as f:
        json.dump(processed_results, f, indent=2, ensure_ascii=False)
    print("Results saved to: opencrawl_processed_results.json")
    
    return processed_results


def real_opencrawl_integration_example():
    """
    Template for real OpenCrawl integration.
    
    Replace placeholder code with actual OpenCrawl API calls.
    """
    
    llm_client = OpenCrawlLLMClient()
    
    # Example: If OpenCrawl has a REST API
    # import opencrawl_client
    # 
    # # Start crawl job
    # crawl_job = opencrawl_client.start_crawl(
    #     urls=["https://example.com"],
    #     max_pages=10
    # )
    # 
    # # Wait for completion
    # results = opencrawl_client.get_results(crawl_job.id)
    # 
    # # Process each result with LLM
    # for page in results.pages:
    #     content = page.content
    #     summary = llm_client.summarize_content(content)
    #     print(f"URL: {page.url}")
    #     print(f"Summary: {summary}\n")
    
    print("Replace this with actual OpenCrawl API integration")


if __name__ == "__main__":
    try:
        # Run the workflow
        results = crawl_and_process_workflow()
        
        print("\n" + "="*60)
        print("Integration Complete!")
        print("="*60)
        print(f"\nProcessed {len(results)} URLs")
        print("Check 'opencrawl_processed_results.json' for full results")
        
    except ConnectionError as e:
        print(f"\n✗ Error: {e}")
        print("\nMake sure your LLM server is running:")
        print("  python main.py")
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
