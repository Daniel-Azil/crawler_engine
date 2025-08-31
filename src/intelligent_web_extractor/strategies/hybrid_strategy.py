"""
Hybrid Extraction Strategy

Combines semantic and structured extraction approaches for optimal
content extraction from complex pages with mixed content types.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import logging
from datetime import datetime

from bs4 import BeautifulSoup

from ..models.extraction_result import ExtractionResult, ContentMetadata, ExtractionMetrics, StrategyInfo
from ..models.config import ExtractorConfig
from ..utils.ai_client import AIClient
from .semantic_strategy import SemanticExtractionStrategy
from .structured_strategy import StructuredExtractionStrategy


class HybridExtractionStrategy:
    """
    Hybrid strategy combines semantic and structured parsing for mixed pages.
    Note: Advanced AI reasoning and interactive navigation are NOT performed here; these are handled by Adaptive mode.
    """
    
    def __init__(self, ai_client: AIClient, config: ExtractorConfig):
        """
        Initialize the hybrid extraction strategy.
        
        Args:
            ai_client: AI client for content analysis
            config: Configuration settings
        """
        self.ai_client = ai_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Initialize sub-strategies
        self.semantic_strategy = SemanticExtractionStrategy(ai_client, config)
        self.structured_strategy = StructuredExtractionStrategy(config)
        
        # Performance tracking
        self._total_extractions = 0
        self._successful_extractions = 0
        self._average_confidence = 0.0
        self._average_time = 0.0
        
        # Strategy selection tracking
        self._semantic_usage = 0
        self._structured_usage = 0
        self._hybrid_usage = 0
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the hybrid extraction strategy"""
        if self._initialized:
            return
        
        self.logger.info("Initializing Hybrid Extraction Strategy")
        
        # Initialize sub-strategies
        await self.semantic_strategy.initialize()
        await self.structured_strategy.initialize()
        
        self._initialized = True
        self.logger.info("Hybrid Extraction Strategy initialized successfully")
    
    async def close(self):
        """Close the strategy and cleanup resources"""
        self.logger.info("Closing Hybrid Extraction Strategy")
        
        await self.semantic_strategy.close()
        await self.structured_strategy.close()
        
        self._initialized = False
    
    async def extract(
        self,
        url: str,
        user_query: Optional[str] = None,
        html_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract content using hybrid approach.
        
        Args:
            url: The URL being extracted
            user_query: Optional user query for context
            html_content: Raw HTML content
            
        Returns:
            Dictionary containing extraction results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Analyze page content to determine strategy mix
            soup = BeautifulSoup(html_content, 'html.parser')
            content_analysis = await self._analyze_content(soup, user_query)
            
            # Determine extraction approach
            approach = self._determine_approach(content_analysis)
            
            # Extract content using determined approach
            if approach == "semantic":
                result = await self._extract_semantic(url, user_query, html_content)
                self._semantic_usage += 1
            elif approach == "structured":
                result = await self._extract_structured(url, user_query, html_content)
                self._structured_usage += 1
            else:  # hybrid
                result = await self._extract_hybrid(url, user_query, html_content, content_analysis)
                self._hybrid_usage += 1
            
            # Calculate metrics
            extraction_time = time.time() - start_time
            result["metrics"].extraction_time_ms = extraction_time * 1000
            
            # Update strategy info
            result["strategy_info"].strategy_name = "HybridExtractionStrategy"
            result["strategy_info"].strategy_parameters["approach_used"] = approach
            result["strategy_info"].strategy_parameters["content_analysis"] = content_analysis
            
            # Update performance tracking
            self._update_performance_tracking(extraction_time, result["metrics"].confidence_score)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Hybrid extraction failed for {url}: {str(e)}")
            
            return {
                "content": "",
                "raw_html": html_content,
                "metadata": ContentMetadata(),
                "metrics": ExtractionMetrics(),
                "strategy_info": StrategyInfo(strategy_name="HybridExtractionStrategy"),
                "success": False,
                "error_message": str(e)
            }
    
    async def _analyze_content(self, soup: BeautifulSoup, user_query: Optional[str]) -> Dict[str, Any]:
        """Analyze page content to determine extraction approach"""
        analysis = {
            "content_type": "unknown",
            "complexity_score": 0.0,
            "structured_elements": {},
            "text_content_ratio": 0.0,
            "recommended_approach": "hybrid"
        }
        
        try:
            # Analyze structured elements
            analysis["structured_elements"] = {
                "tables": len(soup.find_all('table')),
                "lists": len(soup.find_all(['ul', 'ol'])),
                "forms": len(soup.find_all('form')),
                "links": len(soup.find_all('a')),
                "images": len(soup.find_all('img')),
                "videos": len(soup.find_all(['video', 'iframe']))
            }
            
            # Calculate text content ratio
            text_content = soup.get_text()
            total_elements = len(soup.find_all())
            text_ratio = len(text_content) / max(total_elements, 1)
            analysis["text_content_ratio"] = min(1.0, text_ratio)
            
            # Determine content type
            if analysis["structured_elements"]["tables"] > 2:
                analysis["content_type"] = "data_heavy"
            elif analysis["structured_elements"]["forms"] > 0:
                analysis["content_type"] = "interactive"
            elif analysis["text_content_ratio"] > 0.7:
                analysis["content_type"] = "text_heavy"
            elif analysis["structured_elements"]["lists"] > 3:
                analysis["content_type"] = "list_heavy"
            else:
                analysis["content_type"] = "mixed"
            
            # Calculate complexity score
            complexity = 0.0
            complexity += analysis["structured_elements"]["tables"] * 0.2
            complexity += analysis["structured_elements"]["forms"] * 0.3
            complexity += analysis["structured_elements"]["lists"] * 0.1
            complexity += analysis["text_content_ratio"] * 0.4
            analysis["complexity_score"] = min(1.0, complexity)
            
            # Use AI to analyze content if available
            if self.ai_client and user_query:
                ai_analysis = await self._get_ai_content_analysis(soup, user_query)
                analysis.update(ai_analysis)
            
        except Exception as e:
            self.logger.warning(f"Content analysis failed: {str(e)}")
        
        return analysis
    
    async def _get_ai_content_analysis(self, soup: BeautifulSoup, user_query: str) -> Dict[str, Any]:
        """Get AI analysis of content"""
        try:
            # Extract key content for AI analysis
            text_content = soup.get_text()[:2000]  # Limit for AI analysis
            
            prompt = f"""
Analyze the following web page content and determine the best extraction approach.

User Query: {user_query}

Content Preview:
{text_content[:1000]}...

Based on the content structure and user query, recommend:
1. content_type: "text_heavy", "data_heavy", "interactive", "list_heavy", or "mixed"
2. recommended_approach: "semantic", "structured", or "hybrid"
3. reasoning: brief explanation of the recommendation

Provide your response in JSON format.
"""
            
            response = await self.ai_client._get_ai_response(prompt)
            
            # Parse AI response
            try:
                import json
                ai_analysis = json.loads(response)
                return ai_analysis
            except:
                return {}
                
        except Exception as e:
            self.logger.warning(f"AI content analysis failed: {str(e)}")
            return {}
    
    def _determine_approach(self, content_analysis: Dict[str, Any]) -> str:
        """Determine the best extraction approach based on content analysis"""
        content_type = content_analysis.get("content_type", "mixed")
        complexity_score = content_analysis.get("complexity_score", 0.5)
        structured_elements = content_analysis.get("structured_elements", {})
        
        # Use AI recommendation if available
        if "recommended_approach" in content_analysis:
            return content_analysis["recommended_approach"]
        
        # Rule-based approach selection
        if content_type == "text_heavy" and complexity_score < 0.3:
            return "semantic"
        elif content_type == "data_heavy" and structured_elements.get("tables", 0) > 2:
            return "structured"
        elif content_type == "interactive" and structured_elements.get("forms", 0) > 0:
            return "structured"
        else:
            return "hybrid"
    
    async def _extract_semantic(self, url: str, user_query: Optional[str], html_content: str) -> Dict[str, Any]:
        """Extract content using semantic approach"""
        return await self.semantic_strategy.extract(url, user_query, html_content)
    
    async def _extract_structured(self, url: str, user_query: Optional[str], html_content: str) -> Dict[str, Any]:
        """Extract content using structured approach"""
        return await self.structured_strategy.extract(url, user_query, html_content)
    
    async def _extract_hybrid(self, url: str, user_query: Optional[str], html_content: str, content_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Extract content using hybrid approach"""
        try:
            # Extract using both strategies
            semantic_result = await self.semantic_strategy.extract(url, user_query, html_content)
            structured_result = await self.structured_strategy.extract(url, user_query, html_content)
            
            # Combine results
            combined_content = self._combine_results(semantic_result, structured_result, content_analysis)
            
            # Calculate combined metrics
            combined_metrics = self._combine_metrics(semantic_result["metrics"], structured_result["metrics"])
            
            # Create combined metadata
            combined_metadata = self._combine_metadata(semantic_result["metadata"], structured_result["metadata"])
            
            # Create strategy info
            strategy_info = StrategyInfo(
                strategy_name="HybridExtractionStrategy",
                strategy_version="1.0.0",
                strategy_parameters={
                    "semantic_confidence": semantic_result["metrics"].confidence_score,
                    "structured_confidence": structured_result["metrics"].confidence_score,
                    "content_type": content_analysis.get("content_type", "mixed"),
                    "approach": "hybrid"
                }
            )
            
            return {
                "content": combined_content,
                "raw_html": html_content,
                "metadata": combined_metadata,
                "structured_data": structured_result.get("structured_data", {}),
                "metrics": combined_metrics,
                "strategy_info": strategy_info,
                "success": semantic_result["success"] or structured_result["success"],
                "semantic_content_length": len(semantic_result.get("content", "")),
                "structured_content_length": len(structured_result.get("content", "")),
                "combined_content_length": len(combined_content)
            }
            
        except Exception as e:
            self.logger.error(f"Hybrid extraction failed: {str(e)}")
            
            # Fallback to semantic extraction
            return await self._extract_semantic(url, user_query, html_content)
    
    def _combine_results(self, semantic_result: Dict[str, Any], structured_result: Dict[str, Any], content_analysis: Dict[str, Any]) -> str:
        """Combine results from both strategies"""
        semantic_content = semantic_result.get("content", "")
        structured_content = structured_result.get("content", "")
        
        # Determine combination strategy based on content analysis
        content_type = content_analysis.get("content_type", "mixed")
        
        if content_type == "text_heavy":
            # Prefer semantic content for text-heavy pages
            if semantic_content and len(semantic_content) > len(structured_content):
                return semantic_content
            else:
                return structured_content
        elif content_type == "data_heavy":
            # Prefer structured content for data-heavy pages
            if structured_content and len(structured_content) > len(semantic_content):
                return structured_content
            else:
                return semantic_content
        else:
            # For mixed content, combine both
            combined_parts = []
            
            if semantic_content:
                combined_parts.append(semantic_content)
            
            if structured_content:
                combined_parts.append(structured_content)
            
            return "\n\n--- Structured Data ---\n\n".join(combined_parts)
    
    def _combine_metrics(self, semantic_metrics: ExtractionMetrics, structured_metrics: ExtractionMetrics) -> ExtractionMetrics:
        """Combine metrics from both strategies"""
        combined_metrics = ExtractionMetrics()
        
        # Weighted average of confidence scores
        semantic_weight = 0.6
        structured_weight = 0.4
        
        combined_metrics.confidence_score = (
            semantic_metrics.confidence_score * semantic_weight +
            structured_metrics.confidence_score * structured_weight
        )
        
        combined_metrics.relevance_score = (
            semantic_metrics.relevance_score * semantic_weight +
            structured_metrics.relevance_score * structured_weight
        )
        
        # Use the better completeness score
        combined_metrics.completeness_score = max(
            semantic_metrics.completeness_score,
            structured_metrics.completeness_score
        )
        
        # Use the better accuracy score
        combined_metrics.accuracy_score = max(
            semantic_metrics.accuracy_score,
            structured_metrics.accuracy_score
        )
        
        # Average the processing times
        combined_metrics.processing_time_ms = (
            semantic_metrics.processing_time_ms + structured_metrics.processing_time_ms
        ) / 2
        
        return combined_metrics
    
    def _combine_metadata(self, semantic_metadata: ContentMetadata, structured_metadata: ContentMetadata) -> ContentMetadata:
        """Combine metadata from both strategies"""
        combined_metadata = ContentMetadata()
        
        # Prefer non-empty values
        combined_metadata.title = semantic_metadata.title or structured_metadata.title
        combined_metadata.author = semantic_metadata.author or structured_metadata.author
        combined_metadata.publish_date = semantic_metadata.publish_date or structured_metadata.publish_date
        combined_metadata.last_modified = semantic_metadata.last_modified or structured_metadata.last_modified
        combined_metadata.language = semantic_metadata.language or structured_metadata.language
        combined_metadata.content_type = semantic_metadata.content_type or structured_metadata.content_type
        
        # Combine tags and categories
        combined_metadata.tags = list(set(semantic_metadata.tags + structured_metadata.tags))
        combined_metadata.categories = list(set(semantic_metadata.categories + structured_metadata.categories))
        
        # Use the higher word and character counts
        combined_metadata.word_count = max(semantic_metadata.word_count, structured_metadata.word_count)
        combined_metadata.character_count = max(semantic_metadata.character_count, structured_metadata.character_count)
        combined_metadata.reading_time_minutes = max(semantic_metadata.reading_time_minutes, structured_metadata.reading_time_minutes)
        
        return combined_metadata
    
    def _update_performance_tracking(self, extraction_time: float, confidence_score: float):
        """Update performance tracking metrics"""
        self._total_extractions += 1
        
        if confidence_score >= self.config.extraction.confidence_threshold:
            self._successful_extractions += 1
        
        # Update averages
        self._average_confidence = (
            (self._average_confidence * (self._total_extractions - 1) + confidence_score) / 
            self._total_extractions
        )
        
        self._average_time = (
            (self._average_time * (self._total_extractions - 1) + extraction_time) / 
            self._total_extractions
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on hybrid strategy"""
        semantic_health = await self.semantic_strategy.health_check()
        structured_health = await self.structured_strategy.health_check()
        
        return {
            "initialized": self._initialized,
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "average_confidence": self._average_confidence,
            "average_time": self._average_time,
            "semantic_usage": self._semantic_usage,
            "structured_usage": self._structured_usage,
            "hybrid_usage": self._hybrid_usage,
            "semantic_strategy_healthy": semantic_health.get("initialized", False),
            "structured_strategy_healthy": structured_health.get("initialized", False)
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "average_confidence": self._average_confidence,
            "average_time": self._average_time,
            "strategy_name": "HybridExtractionStrategy",
            "semantic_usage": self._semantic_usage,
            "structured_usage": self._structured_usage,
            "hybrid_usage": self._hybrid_usage,
            "total_usage": self._semantic_usage + self._structured_usage + self._hybrid_usage
        } 