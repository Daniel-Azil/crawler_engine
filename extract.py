#!/usr/bin/env python3
"""
Simple CLI for Intelligent Web Extractor

Usage:
    python extract.py "https://example.com" "Extract the main article content"
    python extract.py "https://example.com" "Get product prices" --format json
    python extract.py "https://example.com" "Find contact info" --output-file result.json
"""

import asyncio
import json
import sys
import argparse
from pathlib import Path
from typing import Optional

# Add src to path
sys.path.insert(0, str(Path("src")))

async def main():
    """Main CLI function"""
    parser = argparse.ArgumentParser(description="Intelligent Web Extractor CLI")
    parser.add_argument("url", help="URL to extract from")
    parser.add_argument("prompt", help="What you want to extract (e.g., 'Get product prices')")
    parser.add_argument("--format", default="json", help="Output format (json, custom)")
    parser.add_argument("--output-file", help="Save result to file")
    parser.add_argument("--api-key", help="API key for AI features")
    parser.add_argument("--include-html", action="store_true", help="Include raw HTML in output")
    
    args = parser.parse_args()
    
    try:
        from simple_extractor import SimpleExtractor
        
        print(f"🧠 Intelligent Web Extractor")
        print(f"🌐 URL: {args.url}")
        print(f"📝 Prompt: {args.prompt}")
        print(f"📊 Format: {args.format}")
        print("-" * 50)
        
        # Initialize extractor
        extractor = SimpleExtractor(api_key=args.api_key)
        
        # Extract data
        result = await extractor.extract(
            url=args.url,
            prompt=args.prompt,
            output_format=args.format,
            include_raw_html=args.include_html
        )
        
        # Display result
        if result["success"]:
            print("✅ Extraction successful!")
            print(f"📊 Data: {json.dumps(result['data'], indent=2)}")
        else:
            print(f"❌ Extraction failed: {result.get('error', 'Unknown error')}")
            
        # Save to file if requested
        if args.output_file:
            with open(args.output_file, 'w', encoding='utf-8') as f:
                json.dump(result, f, indent=2, ensure_ascii=False)
            print(f"💾 Result saved to: {args.output_file}")
            
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main()) 