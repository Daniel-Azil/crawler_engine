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
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime
from dotenv import load_dotenv

# Load env early
load_dotenv()

# Configure root logger based on environment
import os
console_logging = os.getenv("INTELLIGENT_EXTRACTOR_ENABLE_CONSOLE_LOGGING", "true").lower() == "true"
log_level = os.getenv("INTELLIGENT_EXTRACTOR_LOG_LEVEL", "INFO").upper()

# Set up basic logging configuration if console logging is disabled
if not console_logging:
    # Disable all logging to console by setting level to CRITICAL and removing handlers
    logging.basicConfig(level=logging.CRITICAL, handlers=[])
    # Or set up a null handler
    logging.getLogger().handlers = []
else:
    logging.basicConfig(
        level=getattr(logging, log_level, logging.INFO),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

# Add src to path
sys.path.insert(0, str(Path("src")))

async def extract(
    url: str,
    prompt: str,
    output_format: Any = "json",
    api_key: Optional[str] = None,
    include_raw_html: bool = False,
    mode: Optional[str] = None,
    timeout: Optional[int] = None,
    max_workers: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Extract data from a URL based on your prompt with flexible output formatting.
    
    Args:
        url: The website URL to extract content from
        prompt: What you want to extract (e.g., "Get the main article content")
        output_format: ANY format/schema you want the output in:
            - {"title": "value", "author": "value"} - JSON schema
            - ["item1", "item2"] - Array schema  
            - "markdown format" - String template
            - "<h1>{title}</h1>" - HTML template
            - Any custom structure you need
            - None - Returns raw content without AI formatting
        api_key: Optional API key for AI features
        include_raw_html: Whether to include raw HTML in response
        mode: Extraction strategy ("adaptive", "semantic", "structured", "rule_based", "hybrid")
        timeout: Request timeout in seconds
        max_workers: Maximum concurrent workers
        
    Returns:
        Dictionary with extracted data matching your specified output_format
    """
    
    # Validate required parameters
    validation_errors = []
    
    if not url or not isinstance(url, str) or url.strip() == "":
        validation_errors.append("'url' parameter is required and must be a non-empty string")
    
    if not prompt or not isinstance(prompt, str) or prompt.strip() == "":
        validation_errors.append("'prompt' parameter is required and must be a non-empty string")
    
    # output_format can be anything the user wants - no validation needed
    
    if mode is not None and (not isinstance(mode, str) or mode.strip() == ""):
        validation_errors.append("'mode' parameter must be a non-empty string or None")
    
    # Check if URL looks valid
    if url and isinstance(url, str) and url.strip():
        url = url.strip()
        if not (url.startswith('http://') or url.startswith('https://')):
            validation_errors.append("'url' must start with 'http://' or 'https://'")
    
    # Check valid modes
    valid_modes = ["adaptive", "semantic", "structured", "rule_based", "hybrid"]
    if mode and mode.lower() not in valid_modes:
        validation_errors.append(f"'mode' must be one of: {', '.join(valid_modes)}")
    
    # If there are validation errors, return error response
    if validation_errors:
        return {
            "success": False,
            "error": "Validation failed",
            "validation_errors": validation_errors,
            "url": url if isinstance(url, str) else "",
            "timestamp": datetime.now().isoformat(),
        }
    
    try:
        from intelligent_web_extractor.core.extractor import AdaptiveContentExtractor
        from intelligent_web_extractor.models.extraction_result import ExtractionStrategy
        from intelligent_web_extractor.models.config import ExtractorConfig

        config = ExtractorConfig.from_env()
        extraction_mode = None
        if mode:
            try:
                extraction_mode = ExtractionStrategy(mode)
            except Exception:
                extraction_mode = None

        custom_config = {"performance": {}, "browser": {}}
        if timeout is not None:
            # Set both request timeout (in seconds) and browser timeout (in milliseconds)
            custom_config["performance"]["request_timeout"] = timeout
            custom_config["browser"]["timeout"] = timeout * 1000  # Convert to milliseconds
        if max_workers is not None:
            custom_config["performance"]["max_workers"] = max_workers
        if not custom_config["performance"] and not custom_config["browser"]:
            custom_config = None

        async with AdaptiveContentExtractor(config) as extractor:
            result_obj = await extractor.extract_content(
                url=url,
                user_query=prompt,
                extraction_mode=extraction_mode,
                output_format=output_format,  # Pass exactly what user specified
                custom_config=custom_config,
            )

            # Simple data processing - respect user's output_format specification
            formatted = result_obj.custom_fields.get("formatted_data")
            if formatted is None:
                # No formatter applied; fall back to raw content
                data_value = result_obj.content
            else:
                # Use whatever the AI formatter returned based on user's schema
                data_value = formatted

            response = {
                "success": result_obj.success,
                "url": result_obj.url,
                "data": data_value,
                "raw_html": result_obj.raw_html if include_raw_html else None,
                "meta": result_obj.metadata if isinstance(result_obj.metadata, dict) else result_obj.metadata.__dict__,
                "strategy": result_obj.strategy_info.strategy_name,
            }
            return response
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "url": url,
            "timestamp": datetime.now().isoformat(),
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