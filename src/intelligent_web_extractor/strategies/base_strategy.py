"""
Base Strategy Module

This module defines the base class for all extraction strategies.
"""

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any, Union
import logging

from ..models.config import ExtractorConfig


class BaseExtractionStrategy(ABC):
    """
    Abstract base class for all extraction strategies.
    
    This class defines the interface that all extraction strategies
    must implement. It provides a common structure for different
    approaches to content extraction.
    """
    
    def __init__(self, config: ExtractorConfig):
        """
        Initialize the strategy with configuration.
        
        Args:
            config: Configuration settings for the strategy
        """
        self.config = config
        self.logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")
        self._initialized = False
        
    @abstractmethod
    async def initialize(self) -> bool:
        """
        Initialize the strategy.
        
        This method should perform any necessary setup,
        such as loading models, establishing connections,
        or preparing resources.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def close(self) -> bool:
        """
        Clean up resources used by the strategy.
        
        This method should release any resources,
        close connections, or perform other cleanup tasks.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        pass
    
    @abstractmethod
    async def extract(
        self,
        url: str,
        user_query: Optional[str] = None,
        html_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract content from a URL or HTML content.
        
        Args:
            url: The URL to extract from
            user_query: Optional user query for context
            html_content: Optional raw HTML content
            
        Returns:
            Dictionary containing extraction results with keys:
            - content: Extracted text content
            - success: Boolean indicating success
            - metadata: Additional metadata
            - metrics: Performance metrics
            - strategy_name: Name of the strategy used
        """
        pass
    
    @abstractmethod
    async def health_check(self) -> Dict[str, Any]:
        """
        Perform a health check on the strategy.
        
        Returns:
            Dictionary containing health information
        """
        pass
    
    def get_strategy_name(self) -> str:
        """Get the name of this strategy"""
        return self.__class__.__name__
    
    def get_strategy_version(self) -> str:
        """Get the version of this strategy"""
        return "1.0.0"
    
    def is_initialized(self) -> bool:
        """Check if the strategy is initialized"""
        return self._initialized
    
    def validate_config(self) -> List[str]:
        """
        Validate the configuration for this strategy.
        
        Returns:
            List of validation errors (empty if valid)
        """
        errors = []
        
        # Basic validation
        if not self.config:
            errors.append("Configuration is required")
        
        return errors
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    def get_capabilities(self) -> Dict[str, Any]:
        """
        Get the capabilities of this strategy.
        
        Returns:
            Dictionary describing strategy capabilities
        """
        return {
            "strategy_name": self.get_strategy_name(),
            "strategy_version": self.get_strategy_version(),
            "supports_ai": False,
            "supports_structured_data": False,
            "supports_custom_rules": False,
            "supports_batch_processing": False,
            "max_content_length": 100000,
            "supported_content_types": ["text/html"],
        }
    
    def get_performance_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics for this strategy.
        
        Returns:
            Dictionary containing performance metrics
        """
        return {
            "strategy_name": self.get_strategy_name(),
            "initialized": self._initialized,
            "total_extractions": 0,
            "successful_extractions": 0,
            "average_extraction_time_ms": 0.0,
            "error_count": 0,
        }
    
    def update_metrics(self, extraction_time_ms: float, success: bool):
        """
        Update performance metrics.
        
        Args:
            extraction_time_ms: Time taken for extraction
            success: Whether extraction was successful
        """
        # This is a base implementation
        # Subclasses should override to track their own metrics
        pass 