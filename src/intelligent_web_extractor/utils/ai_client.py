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

import openai
import anthropic
from sentence_transformers import SentenceTransformer
import numpy as np

from ..models.config import AIModelConfig, AIModelType


class AIClient:
    """
    Manages AI model interactions for intelligent content analysis.
    
    Provides unified interface for different AI models including OpenAI,
    Anthropic, and local models for content analysis and strategy selection.
    """
    
    def __init__(self, config: AIModelConfig):
        """
        Initialize the AI client.
        
        Args:
            config: AI model configuration settings
        """
        self.config = config
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
            # Initialize based on model type
            if self.config.model_type == AIModelType.OPENAI:
                await self._initialize_openai()
            elif self.config.model_type == AIModelType.ANTHROPIC:
                await self._initialize_anthropic()
            elif self.config.model_type == AIModelType.LOCAL:
                await self._initialize_local()
            elif self.config.model_type == AIModelType.HYBRID:
                await self._initialize_hybrid()
            elif self.config.model_type == AIModelType.OLLAMA:
                await self._initialize_ollama()
            else:
                raise ValueError(f"Unsupported model type: {self.config.model_type}")
            
            # Initialize embedding model
            await self._initialize_embeddings()
            
            self._initialized = True
            self.logger.info(f"AI Client initialized with {self.config.model_type.value}")
            
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
        """Initialize OpenAI client"""
        if not self.config.api_key:
            raise ValueError("OpenAI API key is required")
        
        self._openai_client = openai.AsyncOpenAI(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Test connection
        try:
            response = await self._openai_client.chat.completions.create(
                model=self.config.model_name,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10
            )
            self.logger.debug("OpenAI connection test successful")
        except Exception as e:
            self.logger.error(f"OpenAI connection test failed: {str(e)}")
            raise
    
    async def _initialize_anthropic(self):
        """Initialize Anthropic client"""
        if not self.config.api_key:
            raise ValueError("Anthropic API key is required")
        
        self._anthropic_client = anthropic.AsyncAnthropic(
            api_key=self.config.api_key,
            base_url=self.config.base_url
        )
        
        # Test connection
        try:
            response = await self._anthropic_client.messages.create(
                model=self.config.model_name,
                max_tokens=10,
                messages=[{"role": "user", "content": "Hello"}]
            )
            self.logger.debug("Anthropic connection test successful")
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
    
    async def _initialize_ollama(self):
        """Initialize Ollama client"""
        # Ollama doesn't require API key authentication, just endpoint access
        self.logger.info(f"Initializing Ollama client for: {self.config.model_name}")
        self.logger.info(f"Ollama endpoint: {self.config.ollama_endpoint}")
        
        # Test connection to Ollama
        try:
            import aiohttp
            
            test_payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": "Hello"}],
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.ollama_endpoint,
                    json=test_payload,
                    timeout=aiohttp.ClientTimeout(total=10)
                ) as response:
                    if response.status == 200:
                        self.logger.debug("Ollama connection test successful")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama connection test failed: {response.status} - {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Ollama initialization failed: {str(e)}")
            raise
    
    async def _initialize_embeddings(self):
        """Initialize embedding model"""
        try:
            self._embedding_model = SentenceTransformer(self.config.embedding_model)
            
            # Test embedding generation
            test_embedding = self._embedding_model.encode("test")
            self.logger.debug(f"Embedding model initialized with dimension: {len(test_embedding)}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize embedding model: {str(e)}")
            raise
    
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
    
    async def get_embeddings(self, texts: List[str]) -> np.ndarray:
        """
        Get embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            Numpy array of embeddings
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Process in batches
            embeddings = []
            batch_size = self.config.embedding_batch_size
            
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                batch_embeddings = self._embedding_model.encode(batch)
                embeddings.append(batch_embeddings)
            
            # Combine all embeddings
            all_embeddings = np.vstack(embeddings)
            
            return all_embeddings
            
        except Exception as e:
            self.logger.error(f"Embedding generation failed: {str(e)}")
            raise
    
    async def calculate_similarity(
        self,
        query_embedding: np.ndarray,
        content_embeddings: np.ndarray
    ) -> np.ndarray:
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
        """Get response from AI model"""
        start_time = time.time()
        
        try:
            if self._openai_client:
                response = await self._openai_client.chat.completions.create(
                    model=self.config.model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=self.config.temperature,
                    max_tokens=self.config.max_tokens,
                    timeout=self.config.timeout
                )
                result = response.choices[0].message.content
                
            elif self._anthropic_client:
                response = await self._anthropic_client.messages.create(
                    model=self.config.model_name,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    messages=[{"role": "user", "content": prompt}]
                )
                result = response.content[0].text
                
            elif self.config.model_type == AIModelType.OLLAMA:
                # Handle Ollama requests
                result = await self._get_ollama_response(prompt)
                
            else:
                raise ValueError("No AI client available")
            
            # Track tokens (approximate)
            self._total_tokens += len(prompt.split()) + len(result.split())
            
            response_time = time.time() - start_time
            self.logger.debug(f"AI response received in {response_time:.2f}s")
            
            return result
            
        except Exception as e:
            self.logger.error(f"AI request failed: {str(e)}")
            raise
    
    async def _get_ollama_response(self, prompt: str) -> str:
        """Get response from Ollama model"""
        try:
            import aiohttp
            
            payload = {
                "model": self.config.model_name,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False
            }
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.ollama_endpoint,
                    json=payload,
                    timeout=aiohttp.ClientTimeout(total=self.config.timeout)
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        return result.get("message", {}).get("content", "")
                    else:
                        error_text = await response.text()
                        raise Exception(f"Ollama API error {response.status}: {error_text}")
                        
        except Exception as e:
            self.logger.error(f"Ollama request failed: {str(e)}")
            raise
    
    def _create_strategy_analysis_prompt(
        self,
        url: str,
        content: str,
        user_query: Optional[str]
    ) -> str:
        """Create prompt for strategy analysis"""
        # Handle None or empty content
        if not content:
            content = "No content available"
        
        # Safely truncate content
        content_preview = content[:2000] if len(content) > 2000 else content
        
        prompt = f"""
Analyze the following web page content and recommend the best extraction strategy.

URL: {url}
User Query: {user_query or 'General content extraction'}

Content Preview (first 2000 characters):
{content_preview}

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