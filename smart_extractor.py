#!/usr/bin/env python3
"""
Smart Intelligent Web Extractor

This version can intelligently distinguish between:
- Main content (articles, posts, product descriptions)
- Sidebar content (ads, related links, widgets)
- Navigation elements
- Footer content
- Metadata (titles, authors, dates)
"""

import asyncio
import json
import sys
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path("src")))

class SmartExtractor:
    """
    Smart extractor that can distinguish between different types of content
    """
    
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key
        self._initialized = False
        
    async def __aenter__(self):
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()
        
    async def initialize(self):
        if self._initialized:
            return
            
        try:
            from intelligent_web_extractor.utils.web_utils import load_html, clean_text, extract_metadata
            self._initialized = True
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise
            
    async def close(self):
        self._initialized = False
        
    async def extract(
        self,
        url: str,
        prompt: str,
        extraction_type: str = "main_content",  # main_content, all_content, specific_elements
        output_format: Union[str, Dict[str, str]] = "json",
        include_raw_html: bool = False
    ) -> Dict[str, Any]:
        """
        Extract data from a URL based on your prompt
        
        Args:
            url: The URL to extract from
            prompt: What you want to extract
            extraction_type: 
                - "main_content": Only the main article/content
                - "all_content": Everything on the page
                - "specific_elements": Based on prompt keywords
            output_format: Output format specification
            include_raw_html: Whether to include raw HTML
        """
        
        if not self._initialized:
            await self.initialize()
            
        try:
            from intelligent_web_extractor.utils.web_utils import load_html, clean_text, extract_metadata
            from bs4 import BeautifulSoup
            
            print(f"🌐 Loading: {url}")
            html_content = load_html(url)
            
            if not html_content:
                return {
                    "success": False,
                    "error": "Failed to load webpage",
                    "url": url,
                    "timestamp": datetime.now().isoformat()
                }
                
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract based on type
            if extraction_type == "main_content":
                extracted_data = self._extract_main_content(soup, prompt)
            elif extraction_type == "all_content":
                extracted_data = self._extract_all_content(soup, prompt)
            elif extraction_type == "specific_elements":
                extracted_data = self._extract_specific_elements(soup, prompt)
            else:
                extracted_data = self._extract_main_content(soup, prompt)
                
            # Extract metadata
            metadata = extract_metadata(html_content)
            extracted_data.update(metadata)
            
            # Format output
            formatted_data = self._format_output(extracted_data, output_format)
            
            result = {
                "success": True,
                "url": url,
                "prompt": prompt,
                "extraction_type": extraction_type,
                "data": formatted_data,
                "timestamp": datetime.now().isoformat(),
                "extraction_method": "smart_content_analysis"
            }
            
            if include_raw_html:
                result["raw_html"] = html_content
                
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
            
    def _extract_main_content(self, soup, prompt: str) -> Dict[str, Any]:
        """Extract only the main content, excluding sidebars, nav, etc."""
        
        # Remove unwanted elements
        for element in soup(["script", "style", "nav", "header", "footer", "aside", "sidebar"]):
            element.decompose()
            
        # Try to find main content areas
        main_content = ""
        
        # Common main content selectors
        main_selectors = [
            "main",
            "article",
            ".main-content",
            ".content",
            ".post-content",
            ".entry-content",
            "#content",
            "#main",
            ".article-content",
            ".story-content"
        ]
        
        for selector in main_selectors:
            elements = soup.select(selector)
            if elements:
                main_content = " ".join([elem.get_text() for elem in elements])
                break
                
        # If no main content found, try to identify the largest text block
        if not main_content:
            main_content = self._find_largest_text_block(soup)
            
        # Clean the content
        cleaned_content = self._clean_text(main_content)
        
        return {
            "main_content": cleaned_content,
            "content_type": "main_article",
            "word_count": len(cleaned_content.split()),
            "character_count": len(cleaned_content)
        }
        
    def _extract_all_content(self, soup, prompt: str) -> Dict[str, Any]:
        """Extract all content from the page"""
        
        # Remove scripts and styles
        for element in soup(["script", "style"]):
            element.decompose()
            
        # Get all text
        all_text = soup.get_text()
        cleaned_text = self._clean_text(all_text)
        
        return {
            "all_content": cleaned_text,
            "content_type": "full_page",
            "word_count": len(cleaned_text.split()),
            "character_count": len(cleaned_text)
        }
        
    def _extract_specific_elements(self, soup, prompt: str) -> Dict[str, Any]:
        """Extract specific elements based on prompt keywords"""
        
        prompt_lower = prompt.lower()
        extracted = {}
        
        # Extract titles
        if any(word in prompt_lower for word in ["title", "headline", "name"]):
            titles = soup.find_all(["h1", "h2", "h3"])
            extracted["titles"] = [title.get_text().strip() for title in titles if title.get_text().strip()]
            
        # Extract paragraphs
        if any(word in prompt_lower for word in ["content", "article", "text", "body", "main"]):
            paragraphs = soup.find_all("p")
            extracted["paragraphs"] = [p.get_text().strip() for p in paragraphs if p.get_text().strip()]
            
        # Extract links
        if any(word in prompt_lower for word in ["link", "url", "href"]):
            links = soup.find_all("a", href=True)
            extracted["links"] = [{"text": a.get_text().strip(), "url": a["href"]} for a in links if a.get_text().strip()]
            
        # Extract images
        if any(word in prompt_lower for word in ["image", "photo", "picture", "img"]):
            images = soup.find_all("img")
            extracted["images"] = [{"alt": img.get("alt", ""), "src": img.get("src", "")} for img in images]
            
        # Extract tables
        if any(word in prompt_lower for word in ["table", "data", "list"]):
            tables = soup.find_all("table")
            extracted["tables"] = []
            for table in tables:
                rows = table.find_all("tr")
                table_data = []
                for row in rows:
                    cells = row.find_all(["td", "th"])
                    table_data.append([cell.get_text().strip() for cell in cells])
                extracted["tables"].append(table_data)
                
        return extracted
        
    def _find_largest_text_block(self, soup) -> str:
        """Find the largest text block on the page (likely main content)"""
        
        # Get all text blocks
        text_blocks = []
        
        # Look for divs with substantial text
        for div in soup.find_all("div"):
            text = div.get_text().strip()
            if len(text) > 100:  # Only consider substantial blocks
                text_blocks.append((len(text), text))
                
        # Sort by length and return the largest
        if text_blocks:
            text_blocks.sort(reverse=True)
            return text_blocks[0][1]
            
        # Fallback to all text
        return soup.get_text()
        
    def _clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        if not text:
            return ""
            
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
        
    def _format_output(self, data: Dict[str, Any], output_format: Union[str, Dict[str, str]]) -> Dict[str, Any]:
        """Format the output according to specification"""
        
        if output_format == "json":
            return data
        elif isinstance(output_format, dict):
            formatted = {}
            for key, expected_type in output_format.items():
                if key in data:
                    value = data[key]
                    try:
                        if expected_type == "number":
                            formatted[key] = float(value) if isinstance(value, str) and value.replace(".", "").isdigit() else value
                        elif expected_type == "string":
                            formatted[key] = str(value)
                        else:
                            formatted[key] = value
                    except:
                        formatted[key] = value
                else:
                    formatted[key] = None
            return formatted
        else:
            return data

# Simple usage examples
async def demo():
    """Demonstrate the smart extractor"""
    
    print("🧠 Smart Intelligent Web Extractor Demo")
    print("=" * 50)
    
    async with SmartExtractor() as extractor:
        
        # Test 1: Extract main content only
        print("\n📰 Test 1: Extract main content only")
        result1 = await extractor.extract(
            url="https://httpbin.org/html",
            prompt="Extract the main article content",
            extraction_type="main_content"
        )
        print(f"✅ Success: {result1['success']}")
        if result1['success']:
            content = result1['data'].get('main_content', '')
            print(f"📊 Content length: {len(content)} characters")
            print(f"📝 Preview: {content[:200]}...")
            
        # Test 2: Extract all content
        print("\n📄 Test 2: Extract all content")
        result2 = await extractor.extract(
            url="https://httpbin.org/html",
            prompt="Get everything on the page",
            extraction_type="all_content"
        )
        print(f"✅ Success: {result2['success']}")
        if result2['success']:
            content = result2['data'].get('all_content', '')
            print(f"📊 Content length: {len(content)} characters")
            
        # Test 3: Extract specific elements
        print("\n🎯 Test 3: Extract specific elements")
        result3 = await extractor.extract(
            url="https://httpbin.org/html",
            prompt="Get all titles and headings",
            extraction_type="specific_elements"
        )
        print(f"✅ Success: {result3['success']}")
        if result3['success']:
            titles = result3['data'].get('titles', [])
            print(f"📊 Found {len(titles)} titles/headings")

if __name__ == "__main__":
    asyncio.run(demo()) 