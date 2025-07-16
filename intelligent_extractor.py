#!/usr/bin/env python3
"""
Intelligent Web Extractor - Simple Python Interface

A clean, simple interface for extracting data from websites with AI-powered understanding.

Usage:
    from intelligent_extractor import extract
    
    # Simple extraction
    result = await extract("https://example.com", "Get the main article content")
    
    # With custom format
    result = await extract(
        "https://example.com",
        "Get product information",
        output_format={
            "title": "string",
            "price": "number",
            "description": "string"
        }
    )
"""

import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path("src")))

async def extract(
    url: str,
    prompt: str,
    output_format: Union[str, Dict[str, str]] = "json",
    api_key: Optional[str] = None,
    include_raw_html: bool = False
) -> Dict[str, Any]:
    """
    Extract data from a URL based on your prompt
    
    Args:
        url: The URL to extract from
        prompt: What you want to extract (e.g., "Get product title and price")
        output_format: Either "json" for auto-format or dict specifying structure
        api_key: Optional API key for AI features
        include_raw_html: Whether to include raw HTML in response
        
    Returns:
        Dictionary with extracted data in specified format
        
    Example:
        result = await extract(
            "https://example.com",
            "Get the main article content and author",
            output_format="json"
        )
        print(result["data"])
    """
    
    try:
        from simple_extractor import SimpleExtractor
        
        async with SimpleExtractor(api_key=api_key) as extractor:
            return await extractor.extract(
                url=url,
                prompt=prompt,
                output_format=output_format,
                include_raw_html=include_raw_html
            )
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "timestamp": datetime.now().isoformat()
        }

async def extract_batch(
    urls: list,
    prompt: str,
    output_format: Union[str, Dict[str, str]] = "json",
    api_key: Optional[str] = None
) -> list:
    """
    Extract data from multiple URLs
    
    Args:
        urls: List of URLs to extract from
        prompt: What you want to extract from each URL
        output_format: Output format specification
        api_key: Optional API key for AI features
        
    Returns:
        List of extraction results
        
    Example:
        results = await extract_batch(
            ["https://example1.com", "https://example2.com"],
            "Get the main content"
        )
    """
    
    results = []
    
    for url in urls:
        result = await extract(url, prompt, output_format, api_key)
        results.append(result)
        
    return results

def extract_sync(
    url: str,
    prompt: str,
    output_format: Union[str, Dict[str, str]] = "json",
    api_key: Optional[str] = None,
    include_raw_html: bool = False
) -> Dict[str, Any]:
    """
    Synchronous version of extract function
    
    Args:
        url: The URL to extract from
        prompt: What you want to extract
        output_format: Output format specification
        api_key: Optional API key for AI features
        include_raw_html: Whether to include raw HTML
        
    Returns:
        Dictionary with extracted data
    """
    
    return asyncio.run(extract(url, prompt, output_format, api_key, include_raw_html))

def extract_batch_sync(
    urls: list,
    prompt: str,
    output_format: Union[str, Dict[str, str]] = "json",
    api_key: Optional[str] = None
) -> list:
    """
    Synchronous version of extract_batch function
    
    Args:
        urls: List of URLs to extract from
        prompt: What you want to extract from each URL
        output_format: Output format specification
        api_key: Optional API key for AI features
        
    Returns:
        List of extraction results
    """
    
    return asyncio.run(extract_batch(urls, prompt, output_format, api_key))

# Example usage
if __name__ == "__main__":
    async def demo():
        print("🧠 Intelligent Web Extractor Demo")
        print("=" * 40)
        
        # Example 1: Simple extraction
        print("\n📰 Example 1: Extract article content")
        result1 = await extract(
            "https://httpbin.org/html",
            "Extract the main article content and title"
        )
        print(f"✅ Success: {result1['success']}")
        if result1['success']:
            print(f"📊 Data: {json.dumps(result1['data'], indent=2)[:200]}...")
            
        # Example 2: Custom format
        print("\n🛒 Example 2: Extract with custom format")
        result2 = await extract(
            "https://books.toscrape.com",
            "Get product information",
            output_format={
                "title": "string",
                "price": "number",
                "description": "string"
            }
        )
        print(f"✅ Success: {result2['success']}")
        if result2['success']:
            print(f"📊 Data: {json.dumps(result2['data'], indent=2)[:200]}...")
            
        # Example 3: Batch extraction
        print("\n🔄 Example 3: Batch extraction")
        results = await extract_batch(
            ["https://httpbin.org/html", "https://example.com"],
            "Get the main content"
        )
        print(f"✅ Processed {len(results)} URLs")
        for i, result in enumerate(results):
            print(f"   URL {i+1}: {'✅' if result['success'] else '❌'}")
    
    asyncio.run(demo()) 