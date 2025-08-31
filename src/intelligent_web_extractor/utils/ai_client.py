"""
AI Client

Manages AI model interactions for intelligent content analysis and
extraction strategy selection.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import logging
import json
from pathlib import Path
import os

# Handle optional imports gracefully
try:
    import openai
except ImportError:
    openai = None

try:
    import anthropic
except ImportError:
    anthropic = None

try:
    from sentence_transformers import SentenceTransformer
    import numpy as np
except ImportError:
    SentenceTransformer = None
    np = None

try:
    import httpx
except ImportError:
    httpx = None

from ..models.config import AIModelConfig, AIModelType


class AIClient:
    """
    Manages AI model interactions for intelligent content analysis.
    
    Provides unified interface for different AI models including OpenAI,
    Anthropic, and local models for content analysis and strategy selection.
    """
    
    def __init__(self, config: AIModelConfig, logging_config: Optional[Dict[str, Any]] = None):
        """
        Initialize the AI client.
        
        Args:
            config: AI model configuration settings
            logging_config: Logging configuration settings
        """
        self.config = config
        
        # Setup logging with provided config or default
        if logging_config:
            from ..utils.logger import ExtractorLogger
            self.logger = ExtractorLogger(__name__, logging_config)
        else:
            self.logger = logging.getLogger(__name__)
        
        # Model clients
        self._openai_client = None
        self._anthropic_client = None
        self._embedding_model = None
        
        # Performance tracking
        self._total_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._start_time = time.time()
        
        # State management
        self._initialized = False
        self._closed = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the AI client"""
        if self._initialized:
            return
        self.logger.info("Initializing AI Client")
        try:
            # Service selection
            service = os.getenv("SERVICE_TO_USE", "ollama").lower()
            self._service = service
            if service == "openai":
                await self._initialize_openai()
            elif service == "gemini":
                await self._initialize_gemini()
            elif service == "anthropic":
                await self._initialize_anthropic()
            elif service == "ollama":
                await self._initialize_ollama()
            else:
                raise ValueError(f"Unsupported SERVICE_TO_USE: {service}")
            # Initialize embedding model
            await self._initialize_embeddings()
            self._initialized = True
            self.logger.info(f"AI Client initialized with service={service}")
        except Exception as e:
            self.logger.error(f"Failed to initialize AI Client: {str(e)}")
            raise

    async def close(self):
        """Close the AI client and cleanup resources"""
        if self._closed:
            return
        
        self.logger.info("Closing AI Client")
        
        # Cleanup resources
        self._embedding_model = None
        self._openai_client = None
        self._anthropic_client = None
        
        self._closed = True
        self.logger.info("AI Client closed successfully")
    
    async def _initialize_openai(self):
        """Initialize OpenAI client (placeholder-ready)."""
        if not self.config.api_key:
            self.config.api_key = os.getenv("OPENAI_API_KEY")
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        self._openai_client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=os.getenv("OPENAI_API_BASE", self.config.base_url),
        )
        try:
            await self._openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {str(e)}")
            raise

    async def _initialize_anthropic(self):
        """Initialize Anthropic client"""
        # Prefer explicit env for service-based selection
        if not self.config.api_key:
            self.config.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.config.base_url:
            self.config.base_url = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com")
        if not self.config.api_key:
            raise ValueError("Anthropic API key is required")
        self._anthropic_client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url,
        )
        # Light connectivity check
        try:
            await self._anthropic_client.messages.create(
                model=self.config.model_name,
                max_tokens=5,
                messages=[{"role": "user", "content": "Hello"}],
            )
        except Exception as e:
            self.logger.error(f"Anthropic connection test failed: {str(e)}")
            raise
    
    async def _initialize_local(self):
        """Initialize local model (placeholder for future implementation)"""
        if not self.config.local_model_path:
            raise ValueError("Local model path is required for local model type")
        
        # Placeholder for local model initialization
        self.logger.info(f"Local model initialization for: {self.config.local_model_path}")
        # TODO: Implement local model loading
    
    async def _initialize_hybrid(self):
        """Initialize hybrid model setup"""
        # Initialize both OpenAI and Anthropic for hybrid approach
        if self.config.api_key:
            if "openai" in self.config.model_name.lower():
                await self._initialize_openai()
            elif "claude" in self.config.model_name.lower():
                await self._initialize_anthropic()
            else:
                # Default to OpenAI for hybrid
                await self._initialize_openai()
        else:
            raise ValueError("API key is required for hybrid model type")

    async def _initialize_gemini(self):
        """Initialize Gemini client (placeholder)."""
        # Placeholder: adapt to google-generativeai SDK if added
        api_key = os.getenv("GEMINI_API_KEY")
        base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
        if not api_key:
            raise ValueError("Gemini API key is required")
        # Store minimal info; real client setup would go here
        self._gemini = {"api_key": api_key, "base_url": base, "model": self.config.model_name}
        self.logger.debug("Gemini placeholder initialized")

    async def _initialize_ollama(self):
        """Initialize Ollama client (placeholder via OpenAI-compatible if available)."""
        base = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        model = os.getenv("OLLAMA_MODEL_NAME", self.config.model_name)
        # Some Ollama deployments expose an OpenAI-compatible endpoint; if not, adapt _get_ai_response
        try:
            self._openai_client = openai.AsyncOpenAI(api_key="ollama", base_url=f"{base}/v1")
            # Minimal test (may fail if no compat layer)
            await self._openai_client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=5,
            )
            self.config.model_name = model
            self.logger.debug("Ollama (OpenAI-compatible) initialized")
        except Exception:
            # Fallback: mark as custom and use raw HTTP if needed in _get_ai_response
            self._openai_client = None
            self._ollama = {"base_url": base, "model": model}
            self.logger.debug("Ollama raw HTTP mode initialized")
    
    async def _initialize_embeddings(self):
        """Initialize embedding model"""
        if SentenceTransformer is None:
            self.logger.warning("SentenceTransformer not available. Embeddings disabled.")
            self._embedding_model = None
            return
            
        try:
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Test embedding generation
            test_embedding = self._embedding_model.encode("test")
            self.logger.debug(f"Embedding model initialized with dimension: {len(test_embedding)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            self._embedding_model = None
    
    async def analyze_content_strategy(
        self,
        url: str,
        content: str,
        user_query: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Analyze content and recommend extraction strategy.
        
        Args:
            url: The URL being analyzed
            content: Page content for analysis
            user_query: Optional user query for context
            
        Returns:
            Dictionary with strategy recommendation and analysis
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare analysis prompt
            prompt = self._create_strategy_analysis_prompt(url, content, user_query)
            
            # Get AI response
            response = await self._get_ai_response(prompt)
            
            # Parse response
            analysis = self._parse_strategy_analysis(response)
            
            self._total_requests += 1
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Strategy analysis failed: {str(e)}")
            self._failed_requests += 1
            
            # Return default analysis
            return {
                "recommended_strategy": "adaptive",
                "confidence": 0.5,
                "reasoning": f"Analysis failed: {str(e)}",
                "content_type": "unknown",
                "complexity_score": 0.5
            }
    
    async def extract_semantic_content(
        self,
        content: str,
        user_query: str,
        max_chunks: int = 10
    ) -> Dict[str, Any]:
        """
        Extract semantic content based on user query.
        
        Args:
            content: Raw page content
            user_query: User's query for content extraction
            max_chunks: Maximum number of content chunks to return
            
        Returns:
            Dictionary with extracted semantic content
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Prepare extraction prompt
            prompt = self._create_semantic_extraction_prompt(content, user_query, max_chunks)
            
            # Get AI response
            response = await self._get_ai_response(prompt)
            
            # Parse response
            extraction_result = self._parse_semantic_extraction(response)
            
            self._total_requests += 1
            
            return extraction_result
            
        except Exception as e:
            self.logger.error(f"Semantic extraction failed: {str(e)}")
            self._failed_requests += 1
            
            return {
                "extracted_content": "",
                "confidence": 0.0,
                "chunks": [],
                "error": str(e)
            }
    
    async def get_embeddings(self, texts: List[str]) -> Any:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings or None if not available
        """
        if not self._initialized:
            await self.initialize()
        
        if self._embedding_model is None:
            self.logger.warning("Embedding model not available")
            return None
        
        try:
            # Process in batches
            embeddings = []
            batch_size = self.config.embedding_batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._embedding_model.encode(batch)
                embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            if np is not None:
                all_embeddings = np.vstack(embeddings)
                return all_embeddings
            else:
                return embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            return None
    
    async def calculate_similarity(
        self,
        query_embedding: Any,
        content_embeddings: Any
    ) -> Any:
        """
        Calculate similarity between query and content embeddings.
        
        Args:
            query_embedding: Query embedding
            content_embeddings: Content embeddings
            
        Returns:
            Array of similarity scores
        """
        try:
            # Normalize embeddings
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            content_norms = content_embeddings / np.linalg.norm(content_embeddings, axis=1, keepdims=True)
            
            # Calculate cosine similarity
            similarities = np.dot(content_norms, query_norm)
            
            return similarities
            
        except Exception as e:
            self.logger.error(f"Similarity calculation failed: {str(e)}")
            raise
    
    async def _get_ai_response(self, prompt: str) -> str:
        """Get response from the selected AI service."""
        start_time = time.time()
        try:
            service = getattr(self, "_service", os.getenv("SERVICE_TO_USE", "ollama").lower())
            if service == "openai" and self._openai_client:
                response = await self._openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout,
                )
                result = response.choices[0].message.content
            elif service == "gemini":
                # Use aiohttp instead of httpx
                import aiohttp
                headers = {"x-goog-api-key": os.getenv("GEMINI_API_KEY", "")}
                base = os.getenv("GEMINI_API_BASE", "https://generativelanguage.googleapis.com/v1beta")
                model = self.config.model_name or "gemini-pro"
                url = f"{base}/models/{model}:generateContent"
                payload = {"contents": [{"parts": [{"text": prompt}]}]}
                timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.post(url, json=payload, headers=headers) as resp:
                        if resp.status != 200:
                            error_text = await resp.text()
                            raise RuntimeError(f"Gemini API error: {resp.status} - {error_text}")
                        data = await resp.json()
                        candidates = data.get("candidates", [])
                    if candidates and "content" in candidates[0]:
                        parts = candidates[0]["content"].get("parts", [])
                        result = "\n".join(p.get("text", "") for p in parts)
                    else:
                        result = json.dumps(data)
            elif service == "anthropic" and self._anthropic_client:
                response = await self._anthropic_client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}],
                )
                # anthropic SDK returns content parts
                result = response.content[0].text if getattr(response, "content", None) else str(response)
            else:  # ollama
                if self._openai_client:  # OpenAI-compatible mode
                    response = await self._openai_client.chat.completions.create(
                        model=self.config.model_name,
                        messages=[{"role": "user", "content": prompt}],
                        temperature=self.config.temperature,
                        max_tokens=self.config.max_tokens,
                        timeout=self.config.timeout,
                    )
                    result = response.choices[0].message.content
                else:
                    # Raw HTTP to Ollama using aiohttp
                    import aiohttp
                    base = self._ollama.get("base_url")
                    model = self._ollama.get("model")
                    url = f"{base}/api/generate"
                    payload = {"model": model, "prompt": prompt, "stream": False}
                    timeout = aiohttp.ClientTimeout(total=self.config.timeout)
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(url, json=payload) as resp:
                            if resp.status != 200:
                                error_text = await resp.text()
                                raise RuntimeError(f"Ollama API error: {resp.status} - {error_text}")
                            data = await resp.json()
                            result = data.get("response", "") or data.get("text", "") or json.dumps(data)
            self._total_tokens += len(prompt.split()) + len(result.split())
            self.logger.debug(f"AI response received in {time.time() - start_time:.2f}s via {service}")
            return result
        except Exception as e:
            self.logger.error(f"AI request failed: {str(e)}")
            raise
    
    def _create_strategy_analysis_prompt(
        self,
        url: str,
        content: str,
        user_query: Optional[str]
    ) -> str:
        """Create prompt for strategy analysis"""
        prompt = f"""
Analyze the following web page content and recommend the best extraction strategy.

URL: {url}
User Query: {user_query or 'General content extraction'}

Content Preview (first 2000 characters):
{content[:2000]}...

Based on the content structure and user query, recommend the best extraction strategy from:
1. semantic - For content-heavy pages with natural language
2. structured - For data-rich pages with tables, lists, forms
3. hybrid - For complex pages with mixed content types
4. rule_based - For pages with consistent structure
5. adaptive - Let the system choose dynamically

Provide your response in JSON format:
{{
    "recommended_strategy": "strategy_name",
    "confidence": 0.0-1.0,
    "reasoning": "explanation of choice",
    "content_type": "article|product|listing|form|other",
    "complexity_score": 0.0-1.0
}}
"""
        return prompt
    
    def _create_semantic_extraction_prompt(
        self,
        content: str,
        user_query: str,
        max_chunks: int
    ) -> str:
        """Create prompt for semantic content extraction"""
        prompt = f"""
Extract relevant content from the following web page based on the user query.

User Query: {user_query}

Page Content:
{content}

Extract the most relevant content that answers the user's query. Focus on:
- Direct answers to the query
- Supporting information
- Key facts and details
- Important context

Provide your response in JSON format:
{{
    "extracted_content": "main extracted text",
    "confidence": 0.0-1.0,
    "chunks": [
        {{
            "content": "chunk text",
            "relevance": 0.0-1.0,
            "type": "main|supporting|context"
        }}
    ],
    "summary": "brief summary of extracted content"
}}

Limit to {max_chunks} chunks maximum.
"""
        return prompt
    
    def _parse_strategy_analysis(self, response: str) -> Dict[str, Any]:
        """Parse strategy analysis response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            analysis = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["recommended_strategy", "confidence"]
            for field in required_fields:
                if field not in analysis:
                    raise ValueError(f"Missing required field: {field}")
            
            return analysis
            
        except Exception as e:
            self.logger.warning(f"Failed to parse strategy analysis: {str(e)}")
            return {
                "recommended_strategy": "adaptive",
                "confidence": 0.5,
                "reasoning": f"Parse error: {str(e)}",
                "content_type": "unknown",
                "complexity_score": 0.5
            }
    
    def _parse_semantic_extraction(self, response: str) -> Dict[str, Any]:
        """Parse semantic extraction response"""
        try:
            # Extract JSON from response
            json_start = response.find('{')
            json_end = response.rfind('}') + 1
            
            if json_start == -1 or json_end == 0:
                raise ValueError("No JSON found in response")
            
            json_str = response[json_start:json_end]
            extraction = json.loads(json_str)
            
            # Validate required fields
            required_fields = ["extracted_content", "confidence"]
            for field in required_fields:
                if field not in extraction:
                    raise ValueError(f"Missing required field: {field}")
            
            return extraction
            
        except Exception as e:
            self.logger.warning(f"Failed to parse semantic extraction: {str(e)}")
            return {
                "extracted_content": "",
                "confidence": 0.0,
                "chunks": [],
                "error": f"Parse error: {str(e)}"
            }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on AI client"""
        health_status = {
            "initialized": self._initialized,
            "closed": self._closed,
            "model_type": self.config.model_type.value,
            "model_name": self.config.model_name,
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "total_tokens": self._total_tokens,
            "success_rate": 0.0,
            "uptime_seconds": 0.0
        }
        
        if self._start_time:
            health_status["uptime_seconds"] = time.time() - self._start_time
        
        if self._total_requests > 0:
            health_status["success_rate"] = (self._total_requests - self._failed_requests) / self._total_requests
        
        # Test AI functionality
        if self._initialized and not self._closed:
            try:
                # Test with simple prompt
                test_prompt = "Respond with 'OK' if you can read this."
                response = await self._get_ai_response(test_prompt)
                health_status["ai_functional"] = "OK" in response
            except Exception as e:
                self.logger.warning(f"AI health check failed: {str(e)}")
                health_status["ai_functional"] = False
        else:
            health_status["ai_functional"] = False
        
        return health_status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (self._total_requests - self._failed_requests) / max(self._total_requests, 1),
            "total_tokens": self._total_tokens,
            "average_tokens_per_request": self._total_tokens / max(self._total_requests, 1),
            "uptime_seconds": time.time() - self._start_time,
            "average_requests_per_minute": self._total_requests / max((time.time() - self._start_time) / 60, 1)
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self._total_requests = 0
        self._failed_requests = 0
        self._total_tokens = 0
        self._start_time = time.time()
        self.logger.info("AI Client statistics reset") 

    async def format_to_schema(self, content: str, output_format: Any, url: Optional[str] = None, user_query: Optional[str] = None) -> Any:
        """Format raw extracted content into the user-specified schema using AI.
        
        Supported output_format forms (schema-first):
        - dict: JSON object schema. Keys map to types (e.g., string|number|list|object). Can be nested.
        - list:
          - Single item schema: [ { ...object schema... } ] or ["string"] → return a JSON array (pure list).
          - Any length list of strings (e.g., ["heading1", "heading2"]) → treat as request for a list of strings.
        - string template: Any string containing {placeholders} will be treated as a deterministic template.
          We'll ask the LLM for a JSON object with exactly those keys, then render the template.
        - Back-compat literals: "json" → infer a reasonable JSON object. "string" → infer relevant text.
        
        Returns:
        - dict for object schemas
        - list for array schemas
        - str for templates (rendered text/markdown/HTML)
        - On failure, returns raw content or a minimal fallback
        """
        if not self._initialized:
            await self.initialize()
        
        # Local helpers for robust JSON extraction
        import re
        def _extract_json_obj(txt: str) -> Optional[Dict[str, Any]]:
            try:
                start = txt.find('{'); end = txt.rfind('}') + 1
                if start == -1 or end <= start:
                    # Try fenced blocks ```json ... ```
                    fence = re.search(r"```json\s*(\{[\s\S]*?\})\s*```", txt, re.IGNORECASE)
                    if fence:
                        return json.loads(fence.group(1))
                    return None
                return json.loads(txt[start:end])
            except Exception:
                return None
        
        def _extract_json_arr(txt: str) -> Optional[List[Any]]:
            try:
                start = txt.find('['); end = txt.rfind(']') + 1
                if start == -1 or end <= start:
                    # Try to unwrap common wrappers
                    obj = _extract_json_obj(txt)
                    if isinstance(obj, dict):
                        for k in ("items", "data", "results", "list"):
                            if isinstance(obj.get(k), list):
                                return obj.get(k)
                    return None
                return json.loads(txt[start:end])
            except Exception:
                # Try to unwrap object wrapper as last resort
                try:
                    obj = _extract_json_obj(txt)
                    if isinstance(obj, dict):
                        for k in ("items", "data", "results", "list"):
                            if isinstance(obj.get(k), list):
                                return obj.get(k)
                except Exception:
                    pass
                return None
        
        try:
            # 1) Dict schema → strict JSON object
            if isinstance(output_format, dict):
                schema_json = json.dumps(output_format, ensure_ascii=False)
                prompt = (
                    "You are a data extraction and formatting assistant. Given the page content and a target schema, "
                    "return ONLY a JSON object that matches the schema. Fill missing fields with best-effort from the content. "
                    "Respect types: string, number, list. Do not include any text outside the JSON.\n\n"
                    f"URL: {url or ''}\n"
                    f"User Query: {user_query or ''}\n\n"
                    f"Target Schema:\n{schema_json}\n\n"
                    "Page Content (truncated to 4000 chars):\n" + content[:4000]
                )
                response = await self._get_ai_response(prompt)
                obj = _extract_json_obj(response)
                if obj is None:
                    raise ValueError("No JSON object found in AI response")
                return obj
            
            # 2) List schema → pure list
            if isinstance(output_format, list) and len(output_format) >= 1:
                # Determine item schema intent
                single = (len(output_format) == 1)
                item_schema = output_format[0]
                if single and isinstance(item_schema, dict):
                    # List of objects with schema
                    schema_json = json.dumps(item_schema, ensure_ascii=False)
                    prompt = (
                        "Extract a homogeneous list of items from the page according to the item schema below. "
                        "Return ONLY a JSON array of objects matching the schema (no wrapper object).\n\n"
                        f"URL: {url or ''}\n"
                        f"User Query: {user_query or ''}\n\n"
                        f"Item Schema:\n{schema_json}\n\n"
                        "Page Content (truncated to 4000 chars):\n" + content[:4000]
                    )
                else:
                    # Treat as request for a list of strings
                    prompt = (
                        "Extract a list of the most relevant items from the page as strings. "
                        "Return ONLY a JSON array of strings (no wrapper object).\n\n"
                        f"URL: {url or ''}\n"
                        f"User Query: {user_query or ''}\n\n"
                        "Page Content (truncated to 4000 chars):\n" + content[:4000]
                    )
                response = await self._get_ai_response(prompt)
                arr = _extract_json_arr(response)
                if arr is None:
                    raise ValueError("No JSON array found in AI response")
                return arr
            
            # 3) String handling
            if isinstance(output_format, str):
                template_str = output_format
                # 3a) Template mode if placeholders exist
                placeholders = re.findall(r"\{([^{}]+)\}", template_str)
                if placeholders:
                    # Ask LLM for exactly these keys
                    fields_desc = json.dumps({k: "string" for k in placeholders}, ensure_ascii=False)
                    prompt = (
                        "Extract the minimal set of values required to render the provided template. "
                        "Return ONLY a JSON object with exactly these keys.\n\n"
                        f"URL: {url or ''}\n"
                        f"User Query: {user_query or ''}\n\n"
                        f"Required Keys Schema:\n{fields_desc}\n\n"
                        "Page Content (truncated to 4000 chars):\n" + content[:4000]
                    )
                    response = await self._get_ai_response(prompt)
                    obj = _extract_json_obj(response) or {}
                    # Safe deterministic rendering
                    class _SafeDict(dict):
                        def __missing__(self, key):
                            return ""
                    try:
                        rendered = template_str.format_map(_SafeDict(obj))
                    except Exception:
                        # As fallback, try replacing individually
                        rendered = template_str
                        for k in placeholders:
                            rendered = rendered.replace("{" + k + "}", str(obj.get(k, "")))
                    return rendered
                
                # 3b) Back-compat literals
                of = template_str.strip().lower()
                if of == "json":
                    prompt = (
                        "Extract the most relevant information for the user query from the page. "
                        "Return ONLY a JSON object with reasonable keys and values inferred from the content.\n\n"
                        f"User Query: {user_query or ''}\n\n"
                        "Page Content (truncated to 4000 chars):\n" + content[:4000]
                    )
                    response = await self._get_ai_response(prompt)
                    obj = _extract_json_obj(response)
                    if obj is None:
                        raise ValueError("No JSON object found in AI response")
                    return obj
                if of == "string":
                    prompt = (
                        "Extract and synthesize the most relevant text that answers the query. "
                        "Return ONLY the text content (no JSON).\n\n"
                        f"User Query: {user_query or ''}\n\n"
                        "Page Content (truncated to 4000 chars):\n" + content[:4000]
                    )
                    text = await self._get_ai_response(prompt)
                    return text.strip()
            
            # 4) Fallbacks
            # If content looks like JSON, try to parse it to return structured data
            obj = _extract_json_obj(content)
            if obj is not None:
                return obj
            arr = _extract_json_arr(content)
            if arr is not None:
                return arr
            return content
        except Exception as e:
            self.logger.warning(f"Schema formatting failed: {str(e)}")
            return content  # last-resort fallback to raw content