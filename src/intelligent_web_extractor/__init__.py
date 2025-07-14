"""
Intelligent Web Extractor - AI-Powered Web Content Extraction Engine

A cutting-edge web scraping library that uses artificial intelligence to intelligently
extract and process web content based on user queries and context.
"""

from .core.extractor import AdaptiveContentExtractor
from .core.semantic_extractor import SemanticExtractor
from .core.batch_processor import BatchProcessor
from .core.custom_extractor import CustomExtractor
from .models.extraction_result import ExtractionResult
from .models.config import ExtractorConfig
from .utils.logger import ExtractorLogger

__version__ = "0.1.0"
__author__ = "Intelligent Web Extractor Team"
__email__ = "team@intelligent-extractor.com"

__all__ = [
    "AdaptiveContentExtractor",
    "SemanticExtractor", 
    "BatchProcessor",
    "CustomExtractor",
    "ExtractionResult",
    "ExtractorConfig",
    "ExtractorLogger",
] 