"""
Adaptive Content Extractor

The main intelligent content extraction engine that adapts its strategy
based on the content and user requirements.
"""

import asyncio
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging

from ..models.extraction_result import ExtractionResult, ExtractionStrategy, ConfidenceLevel
from ..models.config import ExtractorConfig, AIModelType
from ..utils.logger import ExtractorLogger
from ..utils.browser_manager import BrowserManager
from ..utils.ai_client import AIClient
from ..utils.hidden_content import merge_iframe_content
from ..utils.interaction import ai_navigation_loop, basic_scroll, click_first_expander
from ..utils.errors import NavigationError, IframeExtractionError, PageLoadTimeout
from ..strategies.semantic_strategy import SemanticExtractionStrategy
from ..strategies.structured_strategy import StructuredExtractionStrategy
from ..strategies.hybrid_strategy import HybridExtractionStrategy
from ..strategies.rule_based_strategy import RuleBasedExtractionStrategy
from ..strategies.adaptive_strategy import AdaptiveExtractionStrategy


class AdaptiveContentExtractor:
    """
    Intelligent content extraction engine that adapts its strategy
    based on content analysis and user requirements.
    
    This extractor uses AI to understand the content structure and
    automatically selects the best extraction strategy for each page.
    """
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        """
        Initialize the adaptive content extractor.
        
        Args:
            config: Configuration object. If None, uses default configuration.
        """
        self.config = config or ExtractorConfig()
        
        # Setup logging with configuration from env
        logging_config = {
            "log_level": self.config.logging.log_level,
            "enable_console_logging": self.config.logging.enable_console_logging,
            "enable_file_logging": self.config.logging.enable_file_logging,
            "log_format": self.config.logging.log_format,
            "log_directory": self.config.logging.log_directory,
            "log_performance_metrics": self.config.logging.log_performance_metrics,
            "log_extraction_details": self.config.logging.log_extraction_details,
            "log_ai_interactions": self.config.logging.log_ai_interactions,
        }
        self.logger = ExtractorLogger(__name__, logging_config)
        
        # Initialize components
        self.browser_manager = BrowserManager(self.config.browser)
        self.ai_client = AIClient(self.config.ai_model, logging_config)
        
        # Initialize strategies
        self.strategies = {
            ExtractionStrategy.SEMANTIC: SemanticExtractionStrategy(self.ai_client, self.config),
            ExtractionStrategy.STRUCTURED: StructuredExtractionStrategy(self.config),
            ExtractionStrategy.HYBRID: HybridExtractionStrategy(self.ai_client, self.config),
            ExtractionStrategy.RULE_BASED: RuleBasedExtractionStrategy(self.config),
            ExtractionStrategy.ADAPTIVE: AdaptiveExtractionStrategy(self.ai_client, self.config, self.browser_manager, logging_config),
        }
        
        # Performance tracking
        self.extraction_history: List[Dict[str, Any]] = []
        self.strategy_performance: Dict[str, Dict[str, float]] = {}
        
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
        """Initialize the extractor components"""
        if self._initialized:
            return
        
        self.logger.info("Initializing Adaptive Content Extractor")
        
        # Initialize browser
        await self.browser_manager.initialize()
        
        # Initialize AI client
        await self.ai_client.initialize()
        
        # Initialize strategies
        for strategy in self.strategies.values():
            await strategy.initialize()
        
        self._initialized = True
        self.logger.info("Adaptive Content Extractor initialized successfully")
    
    async def close(self):
        """Close the extractor and cleanup resources"""
        if self._closed:
            return
        
        self.logger.info("Closing Adaptive Content Extractor")
        
        # Close browser
        await self.browser_manager.close()
        
        # Close AI client
        await self.ai_client.close()
        
        # Close strategies
        for strategy in self.strategies.values():
            await strategy.close()
        
        self._closed = True
        self.logger.info("Adaptive Content Extractor closed successfully")
    
    async def extract_content(
        self,
        url: str,
        user_query: Optional[str] = None,
        extraction_mode: Optional[ExtractionStrategy] = None,
        custom_config: Optional[Dict[str, Any]] = None,
        output_format: Optional[Any] = None,
    ) -> ExtractionResult:
        """
        Extract content from a URL using adaptive intelligence.
        
        Args:
            url: The URL to extract content from
            user_query: Optional query to guide extraction
            extraction_mode: Optional specific extraction mode
            custom_config: Optional custom configuration overrides
            output_format: Optional schema (dict) or string indicating desired output shaping
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = datetime.now()
        self.logger.info(f"Starting content extraction from: {url}")
        
        try:
            # Create result object
            result = ExtractionResult(
                url=url,
                content="",
                extraction_started=start_time,
                success=False
            )
            
            # Apply custom configuration
            if custom_config:
                self._apply_custom_config(custom_config)
            
            # Determine extraction strategy
            strategy = await self._select_strategy(url, user_query, extraction_mode)
            result.strategy_info.strategy_name = strategy.__class__.__name__
            
            # Fetch full HTML with hidden content handling; interactive discovery only if adaptive
            html_content = await self._get_prepared_html(url, interactive=(extraction_mode == ExtractionStrategy.ADAPTIVE or (extraction_mode is None)))
            
            # Extract content using selected strategy
            extraction_result = await strategy.extract(url, user_query, html_content)
            
            # Update result with extraction data
            result.content = extraction_result.get("content", "")
            result.raw_html = extraction_result.get("raw_html", html_content)
            
            # Handle metadata conversion from dict to ContentMetadata object
            metadata_dict = extraction_result.get("metadata", {})
            if isinstance(metadata_dict, dict) and metadata_dict:
                # Update existing metadata object with dict values
                for key, value in metadata_dict.items():
                    if hasattr(result.metadata, key):
                        setattr(result.metadata, key, value)
            else:
                result.metadata = extraction_result.get("metadata", result.metadata)
            
            result.structured_data = extraction_result.get("structured_data", result.structured_data)
            result.metrics = extraction_result.get("metrics", result.metrics)
            result.success = extraction_result.get("success", True)
            result.error_message = extraction_result.get("error_message")
            
            # Optional AI-based output shaping - ONLY if output_format is explicitly provided and not raw
            if output_format is not None and result.content and output_format != "raw":
                try:
                    formatted = await self.ai_client.format_to_schema(
                        content=result.content,
                        output_format=output_format,
                        url=url,
                        user_query=user_query,
                    )
                    # Store formatted content in custom_fields while keeping raw content
                    result.custom_fields["formatted_data"] = formatted
                except Exception as _:
                    pass
            
            # Calculate final metrics
            result.extraction_completed = datetime.now()
            self._calculate_final_metrics(result)
            
            # Update performance tracking
            self._update_performance_tracking(result)
            
            # Log results
            self.logger.info(
                f"Extraction completed: {result.success}, "
                f"Confidence: {result.metrics.confidence_score:.2f}, "
                f"Content length: {len(result.content)}"
            )
            
            return result
            
        except Exception as e:
            self.logger.error(f"Extraction failed for {url}: {str(e)}")
            
            # Create error result
            error_result = ExtractionResult(
                url=url,
                content="",
                extraction_started=start_time,
                extraction_completed=datetime.now(),
                success=False,
                error_message=str(e),
                error_type=type(e).__name__
            )
            
            return error_result
    
    async def extract_batch(
        self,
        urls: List[str],
        user_queries: Optional[List[str]] = None,
        extraction_modes: Optional[List[ExtractionStrategy]] = None
    ) -> List[ExtractionResult]:
        """
        Extract content from multiple URLs in batch.
        
        Args:
            urls: List of URLs to extract from
            user_queries: Optional list of queries (one per URL)
            extraction_modes: Optional list of extraction modes (one per URL)
            
        Returns:
            List of ExtractionResult objects
        """
        if not self._initialized:
            await self.initialize()
        
        self.logger.info(f"Starting batch extraction for {len(urls)} URLs")
        
        # Prepare queries and modes
        if user_queries is None:
            user_queries = [None] * len(urls)
        
        if extraction_modes is None:
            extraction_modes = [None] * len(urls)
        
        # Ensure lists have same length
        if len(user_queries) != len(urls):
            user_queries.extend([None] * (len(urls) - len(user_queries)))
        
        if len(extraction_modes) != len(urls):
            extraction_modes.extend([None] * (len(urls) - len(extraction_modes)))
        
        # Create tasks for concurrent extraction
        tasks = []
        for i, url in enumerate(urls):
            task = self.extract_content(
                url=url,
                user_query=user_queries[i],
                extraction_mode=extraction_modes[i]
            )
            tasks.append(task)
        
        # Execute tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_requests)
        
        async def limited_extract(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_extract(task) for task in tasks], return_exceptions=True)
        
        # Process results
        extraction_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed extractions
                error_result = ExtractionResult(
                    url=urls[i],
                    content="",
                    success=False,
                    error_message=str(result),
                    error_type=type(result).__name__
                )
                extraction_results.append(error_result)
            else:
                extraction_results.append(result)
        
        self.logger.info(f"Batch extraction completed: {len([r for r in extraction_results if r.success])}/{len(extraction_results)} successful")
        
        return extraction_results
    
    async def _select_strategy(
        self,
        url: str,
        user_query: Optional[str],
        extraction_mode: Optional[ExtractionStrategy]
    ) -> Any:
        """
        Select the extraction strategy - ADAPTIVE MODE ALWAYS WINS.
        
        The adaptive strategy is the supreme controller that iterates until success.
        All other strategies are just tools for the adaptive strategy to use.
        """
        # If adaptive mode is requested OR no mode specified, use adaptive (it's the boss)
        if extraction_mode == ExtractionStrategy.ADAPTIVE or extraction_mode is None:
            self.logger.info("ADAPTIVE MODE ACTIVATED - Taking full control of extraction")
            return self.strategies[ExtractionStrategy.ADAPTIVE]
        
        # Only use other strategies if explicitly forced (but adaptive is still better)
        if extraction_mode and extraction_mode in self.strategies:
            self.logger.warning(f"Using inferior strategy: {extraction_mode} (adaptive would be better)")
            return self.strategies[extraction_mode]
        
        # Default to adaptive - it's the master
        self.logger.info("Defaulting to ADAPTIVE MODE - the superior extraction engine")
        return self.strategies[ExtractionStrategy.ADAPTIVE]
    
    def _apply_custom_config(self, custom_config: Dict[str, Any]):
        """Apply custom configuration overrides"""
        # Update extraction config
        if "extraction" in custom_config:
            extraction_config = custom_config["extraction"]
            for key, value in extraction_config.items():
                if hasattr(self.config.extraction, key):
                    setattr(self.config.extraction, key, value)
        
        # Update performance config
        if "performance" in custom_config:
            performance_config = custom_config["performance"]
            for key, value in performance_config.items():
                if hasattr(self.config.performance, key):
                    setattr(self.config.performance, key, value)
    
    def _calculate_final_metrics(self, result: ExtractionResult):
        """Calculate final metrics for the extraction result"""
        if result.extraction_started and result.extraction_completed:
            duration = (result.extraction_completed - result.extraction_started).total_seconds() * 1000
            result.metrics.extraction_time_ms = duration
        
        # Calculate content metrics
        if result.content:
            result.metadata.word_count = len(result.content.split())
            result.metadata.character_count = len(result.content)
            result.metadata.reading_time_minutes = result.metadata.word_count / 200  # Average reading speed
        
        # Calculate quality scores
        if result.content:
            result.metrics.completeness_score = min(len(result.content) / 1000, 1.0)  # Simple completeness metric
            result.metrics.accuracy_score = result.metrics.confidence_score  # Use confidence as accuracy proxy
    
    def _update_performance_tracking(self, result: ExtractionResult):
        """Update performance tracking data"""
        # Add to extraction history
        self.extraction_history.append({
            "url": result.url,
            "strategy": result.strategy_info.strategy_name,
            "success": result.success,
            "confidence": result.metrics.confidence_score,
            "extraction_time": result.metrics.extraction_time_ms,
            "content_length": len(result.content),
            "timestamp": datetime.now().isoformat()
        })
        
        # Update strategy performance
        strategy_name = result.strategy_info.strategy_name
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                "total_extractions": 0,
                "successful_extractions": 0,
                "average_confidence": 0.0,
                "average_time": 0.0
            }
        
        perf = self.strategy_performance[strategy_name]
        perf["total_extractions"] += 1
        
        if result.success:
            perf["successful_extractions"] += 1
        
        # Update averages
        current_avg_conf = perf["average_confidence"]
        current_avg_time = perf["average_time"]
        total = perf["total_extractions"]
        
        perf["average_confidence"] = (current_avg_conf * (total - 1) + result.metrics.confidence_score) / total
        perf["average_time"] = (current_avg_time * (total - 1) + result.metrics.extraction_time_ms) / total
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        if not self.extraction_history:
            return {}
        
        total_extractions = len(self.extraction_history)
        successful_extractions = len([h for h in self.extraction_history if h["success"]])
        
        return {
            "total_extractions": total_extractions,
            "successful_extractions": successful_extractions,
            "success_rate": successful_extractions / total_extractions if total_extractions > 0 else 0,
            "average_confidence": sum(h["confidence"] for h in self.extraction_history) / total_extractions,
            "average_extraction_time": sum(h["extraction_time"] for h in self.extraction_history) / total_extractions,
            "strategy_performance": self.strategy_performance,
            "recent_extractions": self.extraction_history[-10:]  # Last 10 extractions
        }
    
    def reset_performance_tracking(self):
        """Reset performance tracking data"""
        self.extraction_history.clear()
        self.strategy_performance.clear()
        self.logger.info("Performance tracking data reset")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on all components"""
        health_status = {
            "extractor_initialized": self._initialized,
            "browser_healthy": False,
            "ai_client_healthy": False,
            "strategies_healthy": {},
            "overall_healthy": False
        }
        
        try:
            # Check browser
            health_status["browser_healthy"] = await self.browser_manager.health_check()
            
            # Check AI client
            health_status["ai_client_healthy"] = await self.ai_client.health_check()
            
            # Check strategies
            for name, strategy in self.strategies.items():
                try:
                    health_status["strategies_healthy"][name.value] = await strategy.health_check()
                except Exception as e:
                    self.logger.warning(f"Strategy {name} health check failed: {str(e)}")
                    health_status["strategies_healthy"][name.value] = False
            
            # Overall health
            all_healthy = (
                health_status["browser_healthy"] and
                health_status["ai_client_healthy"] and
                all(health_status["strategies_healthy"].values())
            )
            health_status["overall_healthy"] = all_healthy
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            health_status["overall_healthy"] = False
        
        return health_status 

    async def _get_prepared_html(self, url: str, interactive: bool) -> str:
        """Load page and return merged HTML including hidden iframe content. Optionally perform AI-guided discovery."""
        merged_html = ""
        async with self.browser_manager.get_page(url) as page:
            # Optional interactive discovery (adaptive only): AI-guided loop
            if interactive:
                try:
                    await ai_navigation_loop(self.ai_client, page, user_query=None, max_steps=5)
                except Exception as e:
                    self.logger.warning(f"Interactive discovery failed: {e}")
            # Gather main + iframe content (parallel iframe merge)
            try:
                merged_html = await merge_iframe_content(page)
            except IframeExtractionError as e:
                self.logger.warning(f"Iframe merge failed: {e}")
        return merged_html or "" 