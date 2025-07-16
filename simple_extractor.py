#!/usr/bin/env python3
"""
Simple Intelligent Web Extractor Interface

A clean, simple interface for extracting data from websites with AI-powered understanding.
"""

import asyncio
import json
import sys
import os
from pathlib import Path
from typing import Dict, Any, Optional, Union
from datetime import datetime

# Load environment variables
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # dotenv not available, will use system env vars

# Add src to path
sys.path.insert(0, str(Path("src")))

class SimpleExtractor:
    """
    Simple interface for Intelligent Web Extractor
    
    Usage:
        extractor = SimpleExtractor()
        
        # Extract with AI understanding
        result = await extractor.extract(
            url="https://example.com",
            prompt="Extract the main article content and author",
            output_format="json"
        )
        
        # Extract with custom JSON structure
        result = await extractor.extract(
            url="https://example.com", 
            prompt="Get product information",
            output_format={
                "title": "string",
                "price": "number", 
                "description": "string",
                "rating": "number"
            }
        )
    """
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize the simple extractor
        
        Args:
            api_key: Optional API key for AI features
        """
        # Auto-load API key from environment if not provided
        if api_key is None:
            # Check for Ollama configuration first
            model_type = os.getenv("INTELLIGENT_EXTRACTOR_MODEL_TYPE", "").lower()
            if model_type == "ollama":
                api_key = "ollama_local"  # Use local ollama
            else:
                api_key = (
                    os.getenv("OPENAI_API_KEY") or 
                    os.getenv("ANTHROPIC_API_KEY") or
                    os.getenv("INTELLIGENT_EXTRACTOR_API_KEY")
                )
        
        self.api_key = api_key
        
        # Load AI configuration
        self.model_type = os.getenv("INTELLIGENT_EXTRACTOR_MODEL_TYPE", "openai").lower()
        self.model_name = os.getenv("INTELLIGENT_EXTRACTOR_MODEL_NAME", "gpt-4")
        self.ollama_endpoint = os.getenv("OLLAMA_API_ENDPOINT", "http://localhost:11434/api/chat")
        self.ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        
        self._initialized = False
        
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
        
    async def initialize(self):
        """Initialize the extractor"""
        if self._initialized:
            return
            
        try:
            from intelligent_web_extractor.utils.web_utils import load_html, clean_text
            from intelligent_web_extractor.utils.logger import ExtractorLogger
            
            self.logger = ExtractorLogger("simple_extractor")
            self._initialized = True
            
        except Exception as e:
            print(f"❌ Failed to initialize: {e}")
            raise
            
    async def close(self):
        """Close the extractor"""
        self._initialized = False
        
    async def extract(
        self,
        url: str,
        prompt: str,
        output_format: Union[str, Dict[str, str]] = "json",
        include_raw_html: bool = False
    ) -> Dict[str, Any]:
        """
        Extract data from a URL based on your prompt
        
        Args:
            url: The URL to extract from
            prompt: What you want to extract (e.g., "Get product title and price")
            output_format: Either "json" for auto-format or dict specifying structure
            include_raw_html: Whether to include raw HTML in response
            
        Returns:
            Dictionary with extracted data in specified format
        """
        if not self._initialized:
            await self.initialize()
            
        try:
            # Load and parse the webpage
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
            
            # Remove unwanted elements
            for element in soup(["script", "style", "nav", "header", "footer"]):
                element.decompose()
                
            # Extract text content
            text_content = soup.get_text()
            cleaned_text = clean_text(text_content)
            
            # Extract metadata
            metadata = extract_metadata(html_content)
            
            # Process based on prompt and format
            result = await self._process_extraction(
                url=url,
                prompt=prompt,
                text_content=cleaned_text,
                html_content=html_content,
                metadata=metadata,
                output_format=output_format,
                include_raw_html=include_raw_html
            )
            
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": str(e),
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
            
    async def _process_extraction(
        self,
        url: str,
        prompt: str,
        text_content: str,
        html_content: str,
        metadata: Dict[str, Any],
        output_format: Union[str, Dict[str, str]],
        include_raw_html: bool
    ) -> Dict[str, Any]:
        """Process the extraction based on prompt and format"""
        
        # If we have AI capabilities, use them
        if self.api_key:
            return await self._ai_extraction(
                url, prompt, text_content, metadata, output_format, include_raw_html
            )
        else:
            # Fallback to rule-based extraction
            return await self._rule_based_extraction(
                url, prompt, text_content, metadata, output_format, include_raw_html
            )
            
    async def _ai_extraction(
        self,
        url: str,
        prompt: str,
        text_content: str,
        metadata: Dict[str, Any],
        output_format: Union[str, Dict[str, str]],
        include_raw_html: bool
    ) -> Dict[str, Any]:
        """AI-powered extraction using Ollama or other AI services"""
        
        try:
            # Use Ollama if configured
            if self.model_type == "ollama":
                extracted_data = await self._ollama_extraction(prompt, text_content, metadata, output_format)
                extraction_method = "ollama_ai"
            else:
                # Fallback to heuristic approach for other AI types without implementation
                extracted_data = self._extract_based_on_prompt(prompt, text_content, metadata)
                extraction_method = "ai_heuristic"
            
            # Format the output
            formatted_data = self._format_output(extracted_data, output_format)
            
            result = {
                "success": True,
                "url": url,
                "prompt": prompt,
                "data": formatted_data,
                "timestamp": datetime.now().isoformat(),
                "extraction_method": extraction_method,
                "model_used": f"{self.model_type}:{self.model_name}"
            }
            
            if include_raw_html:
                result["raw_html"] = html_content
                
            return result
            
        except Exception as e:
            return {
                "success": False,
                "error": f"AI extraction failed: {str(e)}",
                "url": url,
                "timestamp": datetime.now().isoformat()
            }
    
    async def _ollama_extraction(
        self,
        prompt: str,
        text_content: str,
        metadata: Dict[str, Any],
        output_format: Union[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """Extract data using Ollama AI model"""
        
        try:
            import aiohttp
            import json
            
            # Prepare the AI prompt
            ai_prompt = self._create_ai_prompt(prompt, text_content, metadata, output_format)
            
            # Prepare Ollama request
            ollama_payload = {
                "model": self.model_name,
                "messages": [
                    {
                        "role": "user",
                        "content": ai_prompt
                    }
                ],
                "stream": False
            }
            
            # Make request to Ollama
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.ollama_endpoint,
                    json=ollama_payload,
                    timeout=aiohttp.ClientTimeout(total=60)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        ai_response = result.get("message", {}).get("content", "")
                        
                        # Parse AI response
                        extracted_data = self._parse_ai_response(ai_response, output_format)
                        return extracted_data
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
                        
        except Exception as e:
            print(f"⚠️ Ollama extraction failed: {e}")
            print("🔄 Falling back to rule-based extraction")
            # Fallback to rule-based extraction
            return self._extract_based_on_prompt(prompt, text_content, metadata)
    
    def _create_ai_prompt(
        self,
        user_prompt: str,
        content: str,
        metadata: Dict[str, Any],
        output_format: Union[str, Dict[str, str]]
    ) -> str:
        """Create a detailed prompt for the AI model"""
        
        # Truncate content if too long (to fit model context)
        max_content_length = 8000
        if len(content) > max_content_length:
            content = content[:max_content_length] + "... [truncated]"
        
        base_prompt = f"""
You are an intelligent web content extractor. Your task is to extract specific information from web page content based on the user's request.

USER REQUEST: {user_prompt}

WEB PAGE CONTENT:
{content}

PAGE METADATA:
Title: {metadata.get('title', 'N/A')}
"""
        
        # Add output format instructions
        if isinstance(output_format, dict):
            format_description = "Return the data in JSON format with these exact fields:\n"
            for field, field_type in output_format.items():
                format_description += f"- {field}: {field_type}\n"
            
            base_prompt += f"""
OUTPUT FORMAT REQUIRED:
{format_description}

Return only valid JSON, no additional text or explanations.
"""
        else:
            base_prompt += """
OUTPUT FORMAT:
Return the extracted information in a clear, structured JSON format.
Include relevant fields based on the user's request.

Return only valid JSON, no additional text or explanations.
"""
        
        return base_prompt
    
    def _parse_ai_response(
        self,
        ai_response: str,
        output_format: Union[str, Dict[str, str]]
    ) -> Dict[str, Any]:
        """Parse the AI response into structured data"""
        
        try:
            # Try to parse as JSON
            # Remove any markdown code blocks
            response_clean = ai_response.strip()
            if response_clean.startswith("```json"):
                response_clean = response_clean[7:]
            if response_clean.startswith("```"):
                response_clean = response_clean[3:]
            if response_clean.endswith("```"):
                response_clean = response_clean[:-3]
            
            response_clean = response_clean.strip()
            
            # Parse JSON
            parsed_data = json.loads(response_clean)
            
            # Validate against expected format if specified
            if isinstance(output_format, dict):
                validated_data = {}
                for field, expected_type in output_format.items():
                    if field in parsed_data:
                        validated_data[field] = parsed_data[field]
                    else:
                        validated_data[field] = None
                return validated_data
            else:
                return parsed_data
                
        except json.JSONDecodeError as e:
            print(f"⚠️ Failed to parse AI response as JSON: {e}")
            print(f"AI Response: {ai_response[:200]}...")
            
            # Fallback: try to extract key information manually
            return {
                "ai_response": ai_response,
                "extraction_note": "AI response could not be parsed as JSON"
            }
            
    async def _rule_based_extraction(
        self,
        url: str,
        prompt: str,
        text_content: str,
        metadata: Dict[str, Any],
        output_format: Union[str, Dict[str, str]],
        include_raw_html: bool
    ) -> Dict[str, Any]:
        """Rule-based extraction using common patterns"""
        
        # Extract common data patterns
        extracted_data = self._extract_based_on_prompt(prompt, text_content, metadata)
        
        # Format the output
        formatted_data = self._format_output(extracted_data, output_format)
        
        result = {
            "success": True,
            "url": url,
            "prompt": prompt,
            "data": formatted_data,
            "timestamp": datetime.now().isoformat(),
            "extraction_method": "rule_based"
        }
        
        if include_raw_html:
            result["raw_html"] = html_content
            
        return result
        
    def _extract_based_on_prompt(self, prompt: str, text_content: str, metadata: Dict[str, Any]) -> Dict[str, Any]:
        """Extract data based on the user's prompt"""
        
        prompt_lower = prompt.lower()
        extracted = {}
        
        # Extract title
        if any(word in prompt_lower for word in ["title", "headline", "name"]):
            extracted["title"] = metadata.get("title", "Not found")
            
        # Extract content/article
        if any(word in prompt_lower for word in ["content", "article", "text", "body", "main"]):
            # Get full content without truncation
            extracted["content"] = text_content
            
        # Extract author
        if any(word in prompt_lower for word in ["author", "writer", "by"]):
            extracted["author"] = metadata.get("author", "Not found")
            
        # Extract price
        if any(word in prompt_lower for word in ["price", "cost", "amount", "money"]):
            import re
            price_match = re.search(r'[\$£€]?\d+\.?\d*', text_content)
            extracted["price"] = price_match.group() if price_match else "Not found"
            
        # Extract product information
        if any(word in prompt_lower for word in ["product", "item", "goods"]):
            extracted["product_info"] = {
                "title": metadata.get("title", "Not found"),
                "description": text_content
            }
            
        # Extract contact information
        if any(word in prompt_lower for word in ["contact", "email", "phone", "address"]):
            import re
            email_match = re.search(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', text_content)
            phone_match = re.search(r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b', text_content)
            
            extracted["contact"] = {
                "email": email_match.group() if email_match else "Not found",
                "phone": phone_match.group() if phone_match else "Not found"
            }
            
        # If no specific patterns matched, return general info
        if not extracted:
            extracted = {
                "title": metadata.get("title", "Not found"),
                "content": text_content,
                "word_count": len(text_content.split()),
                "character_count": len(text_content)
            }
            
        return extracted
        
    def _format_output(self, data: Dict[str, Any], output_format: Union[str, Dict[str, str]]) -> Dict[str, Any]:
        """Format the output according to user specification"""
        
        if output_format == "json":
            # Return as-is
            return data
        elif isinstance(output_format, dict):
            # User specified custom format
            formatted = {}
            for key, expected_type in output_format.items():
                if key in data:
                    value = data[key]
                    # Try to convert to expected type
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
async def example_usage():
    """Example usage of the SimpleExtractor"""
    
    print("🧠 Simple Intelligent Web Extractor - Examples")
    print("=" * 50)
    
    async with SimpleExtractor() as extractor:
        
        # Example 1: Extract article content
        print("\n📰 Example 1: Extract article content")
        result1 = await extractor.extract(
            url="https://httpbin.org/html",
            prompt="Extract the main article content and title",
            output_format="json"
        )
        print(f"✅ Result: {json.dumps(result1, indent=2)[:200]}...")
        
        # Example 2: Extract with custom format
        print("\n🛒 Example 2: Extract product info with custom format")
        result2 = await extractor.extract(
            url="https://books.toscrape.com",
            prompt="Get product titles and prices",
            output_format={
                "title": "string",
                "price": "number",
                "description": "string"
            }
        )
        print(f"✅ Result: {json.dumps(result2, indent=2)[:200]}...")
        
        # Example 3: Extract contact information
        print("\n📞 Example 3: Extract contact information")
        result3 = await extractor.extract(
            url="https://example.com",
            prompt="Find contact information and email addresses",
            output_format="json"
        )
        print(f"✅ Result: {json.dumps(result3, indent=2)[:200]}...")

if __name__ == "__main__":
    asyncio.run(example_usage()) 