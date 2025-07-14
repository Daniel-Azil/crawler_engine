"""
Custom Extractor Core Component

Provides customizable content extraction capabilities with user-defined
rules, selectors, and extraction patterns.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from ..models.extraction_result import ExtractionResult, ContentMetadata, ExtractionMetrics, StrategyInfo
from ..models.config import ExtractorConfig
from ..strategies.rule_based_strategy import RuleBasedExtractionStrategy


class CustomExtractor:
    """
    Customizable content extractor that allows users to define their own
    extraction rules, selectors, and patterns.
    
    Provides flexibility for specific use cases and domain-specific
    content extraction requirements.
    """
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        """
        Initialize the custom extractor.
        
        Args:
            config: Configuration settings
        """
        self.config = config or ExtractorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Initialize rule-based strategy for custom rules
        self.strategy = RuleBasedExtractionStrategy(self.config)
        
        # Custom extraction rules
        self._custom_rules: Dict[str, Dict[str, Any]] = {}
        self._custom_selectors: List[str] = []
        self._custom_exclude_selectors: List[str] = []
        
        # Performance tracking
        self._total_extractions = 0
        self._successful_extractions = 0
        
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the custom extractor"""
        if self._initialized:
            return
        
        await self.strategy.initialize()
        self._initialized = True
        self.logger.info("Custom Extractor initialized")
    
    async def close(self):
        """Close the custom extractor"""
        await self.strategy.close()
        self._initialized = False
        self.logger.info("Custom Extractor closed")
    
    def add_rule(
        self,
        rule_name: str,
        selector: str,
        extraction_type: str = "text",
        fields: Optional[List[str]] = None,
        priority: int = 10
    ):
        """
        Add a custom extraction rule.
        
        Args:
            rule_name: Name of the rule
            selector: CSS selector for the element
            extraction_type: Type of extraction (text, attribute, structured)
            fields: List of fields to extract (for structured extraction)
            priority: Priority of the rule (lower = higher priority)
        """
        rule_config = {
            "selector": selector,
            "extraction_type": extraction_type,
            "fields": fields or [],
            "priority": priority
        }
        
        self._custom_rules[rule_name] = rule_config
        
        # Add to strategy
        self.strategy.add_custom_rule(rule_name, [selector], priority)
        
        self.logger.info(f"Added custom rule: {rule_name}")
    
    def remove_rule(self, rule_name: str):
        """
        Remove a custom extraction rule.
        
        Args:
            rule_name: Name of the rule to remove
        """
        if rule_name in self._custom_rules:
            del self._custom_rules[rule_name]
            self.strategy.remove_rule(rule_name)
            self.logger.info(f"Removed custom rule: {rule_name}")
        else:
            self.logger.warning(f"Rule not found: {rule_name}")
    
    def add_selector(self, selector: str):
        """
        Add a custom CSS selector for content extraction.
        
        Args:
            selector: CSS selector to add
        """
        self._custom_selectors.append(selector)
        self.logger.info(f"Added custom selector: {selector}")
    
    def add_exclude_selector(self, selector: str):
        """
        Add a CSS selector to exclude from extraction.
        
        Args:
            selector: CSS selector to exclude
        """
        self._custom_exclude_selectors.append(selector)
        self.logger.info(f"Added exclude selector: {selector}")
    
    def clear_rules(self):
        """Clear all custom rules"""
        self._custom_rules.clear()
        self._custom_selectors.clear()
        self._custom_exclude_selectors.clear()
        self.logger.info("Cleared all custom rules")
    
    async def extract(
        self,
        url: str,
        user_query: Optional[str] = None,
        html_content: Optional[str] = None
    ) -> ExtractionResult:
        """
        Extract content using custom rules.
        
        Args:
            url: The URL to extract from
            user_query: Optional user query for context
            html_content: Raw HTML content
            
        Returns:
            ExtractionResult with extracted content
        """
        if not self._initialized:
            await self.initialize()
        
        try:
            # Update config with custom selectors
            self.config.extraction.content_selectors = self._custom_selectors
            self.config.extraction.exclude_selectors = self._custom_exclude_selectors
            
            # Extract using rule-based strategy
            result_dict = await self.strategy.extract(url, user_query, html_content)
            
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
            
            # Update statistics
            self._total_extractions += 1
            if result.success:
                self._successful_extractions += 1
            
            return result
            
        except Exception as e:
            self.logger.error(f"Custom extraction failed for {url}: {str(e)}")
            
            return ExtractionResult(
                url=url,
                content="",
                success=False,
                error_message=str(e),
                error_type=type(e).__name__,
                extraction_started=datetime.now(),
                extraction_completed=datetime.now()
            )
    
    def get_rules(self) -> Dict[str, Dict[str, Any]]:
        """Get all custom rules"""
        return self._custom_rules.copy()
    
    def get_selectors(self) -> List[str]:
        """Get all custom selectors"""
        return self._custom_selectors.copy()
    
    def get_exclude_selectors(self) -> List[str]:
        """Get all exclude selectors"""
        return self._custom_exclude_selectors.copy()
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get extraction statistics"""
        return {
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "custom_rules_count": len(self._custom_rules),
            "custom_selectors_count": len(self._custom_selectors),
            "exclude_selectors_count": len(self._custom_exclude_selectors)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on custom extractor"""
        return {
            "initialized": self._initialized,
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "custom_rules_count": len(self._custom_rules),
            "strategy_healthy": await self.strategy.health_check()
        } 