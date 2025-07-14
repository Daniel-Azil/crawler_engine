"""
Semantic Extractor Core Component

Provides semantic content extraction capabilities using AI to understand
and extract relevant content based on context and user queries.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from ..models.extraction_result import ExtractionResult
from ..models.config import ExtractorConfig
from ..utils.ai_client import AIClient
from ..strategies.semantic_strategy import SemanticExtractionStrategy


class SemanticExtractor:
    """
    High-level semantic content extractor that uses AI to understand
    and extract relevant content from web pages.
    """
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        """
        Initialize the semantic extractor.
        
        Args:
            config: Configuration settings
        """
        self.config = config or ExtractorConfig()
        self.ai_client = AIClient(self.config.ai_model)
        self.strategy = SemanticExtractionStrategy(self.ai_client, self.config)
        
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the semantic extractor"""
        if self._initialized:
            return
        
        await self.ai_client.initialize()
        await self.strategy.initialize()
        self._initialized = True
    
    async def close(self):
        """Close the semantic extractor"""
        await self.ai_client.close()
        await self.strategy.close()
        self._initialized = False
    
    async def extract_with_context(
        self,
        url: str,
        context: str,
        include_metadata: bool = True
    ) -> ExtractionResult:
        """
        Extract content with semantic context understanding.
        
        Args:
            url: The URL to extract from
            context: Context description for extraction
            include_metadata: Whether to include metadata
            
        Returns:
            ExtractionResult with extracted content
        """
        if not self._initialized:
            await self.initialize()
        
        # Extract content using semantic strategy
        result_dict = await self.strategy.extract(url, context)
        
        # Convert to ExtractionResult
        result = ExtractionResult(
            url=url,
            content=result_dict.get("content", ""),
            raw_html=result_dict.get("raw_html"),
            metadata=result_dict.get("metadata"),
            metrics=result_dict.get("metrics"),
            strategy_info=result_dict.get("strategy_info"),
            success=result_dict.get("success", False),
            error_message=result_dict.get("error_message"),
            extraction_started=datetime.now(),
            extraction_completed=datetime.now()
        )
        
        return result 