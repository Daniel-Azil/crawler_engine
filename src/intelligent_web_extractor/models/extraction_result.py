"""
Extraction Result Models

This module defines the data structures for extraction results and related metadata.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
from enum import Enum
import json


class ExtractionMode(str, Enum):
    """Available extraction modes"""
    SEMANTIC = "semantic"
    STRUCTURED = "structured"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"


class ExtractionStrategy(str, Enum):
    """Available extraction strategies"""
    SEMANTIC = "semantic"
    STRUCTURED = "structured"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    RULE_BASED = "rule_based"


class ConfidenceLevel(str, Enum):
    """Confidence levels for extraction results"""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    UNKNOWN = "unknown"


@dataclass
class ContentMetadata:
    """Metadata about extracted content"""
    title: Optional[str] = None
    author: Optional[str] = None
    publish_date: Optional[datetime] = None
    last_modified: Optional[datetime] = None
    language: Optional[str] = None
    word_count: int = 0
    character_count: int = 0
    reading_time_minutes: float = 0.0
    content_type: Optional[str] = None
    tags: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)


@dataclass
class ExtractionMetrics:
    """Performance and quality metrics for extraction"""
    extraction_time_ms: float = 0.0
    processing_time_ms: float = 0.0
    confidence_score: float = 0.0
    relevance_score: float = 0.0
    completeness_score: float = 0.0
    accuracy_score: float = 0.0
    strategy_effectiveness: float = 0.0
    retry_count: int = 0
    error_count: int = 0


@dataclass
class StrategyInfo:
    """Information about the extraction strategy used"""
    strategy_name: str = ""
    strategy_version: str = ""
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    fallback_strategies: List[str] = field(default_factory=list)
    adaptation_reasons: List[str] = field(default_factory=list)


@dataclass
class StructuredData:
    """Structured data extracted from the page"""
    tables: List[Dict[str, Any]] = field(default_factory=list)
    lists: List[Dict[str, Any]] = field(default_factory=list)
    forms: List[Dict[str, Any]] = field(default_factory=list)
    navigation: List[Dict[str, Any]] = field(default_factory=list)
    buttons: List[Dict[str, Any]] = field(default_factory=list)
    links: List[Dict[str, Any]] = field(default_factory=list)
    images: List[Dict[str, Any]] = field(default_factory=list)
    videos: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ExtractionResult:
    """
    Result of a content extraction operation
    
    This class encapsulates all information about a successful or failed
    content extraction, including the extracted content, metadata,
    performance metrics, and strategy information.
    """
    
    # Core content
    url: str
    content: str
    raw_html: Optional[str] = None
    
    # Metadata
    metadata: ContentMetadata = field(default_factory=ContentMetadata)
    structured_data: StructuredData = field(default_factory=StructuredData)
    
    # Performance and quality
    metrics: ExtractionMetrics = field(default_factory=ExtractionMetrics)
    strategy_info: StrategyInfo = field(default_factory=StrategyInfo)
    
    # Status and errors
    success: bool = True
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    
    # Timestamps
    extraction_started: Optional[datetime] = None
    extraction_completed: Optional[datetime] = None
    
    # Additional data
    custom_fields: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        if self.extraction_started and self.extraction_completed:
            duration = (self.extraction_completed - self.extraction_started).total_seconds() * 1000
            self.metrics.extraction_time_ms = duration
    
    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level based on confidence score"""
        if self.metrics.confidence_score >= 0.8:
            return ConfidenceLevel.HIGH
        elif self.metrics.confidence_score >= 0.6:
            return ConfidenceLevel.MEDIUM
        elif self.metrics.confidence_score >= 0.4:
            return ConfidenceLevel.LOW
        else:
            return ConfidenceLevel.UNKNOWN
    
    @property
    def is_high_quality(self) -> bool:
        """Check if the extraction result is high quality"""
        return (
            self.success and
            self.metrics.confidence_score >= 0.7 and
            self.metrics.relevance_score >= 0.6 and
            len(self.content.strip()) > 100
        )
    
    @property
    def summary(self) -> Dict[str, Any]:
        """Get a summary of the extraction result"""
        return {
            "url": self.url,
            "success": self.success,
            "content_length": len(self.content),
            "confidence_score": self.metrics.confidence_score,
            "relevance_score": self.metrics.relevance_score,
            "extraction_time_ms": self.metrics.extraction_time_ms,
            "strategy_used": self.strategy_info.strategy_name,
            "error_message": self.error_message,
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "url": self.url,
            "content": self.content,
            "raw_html": self.raw_html,
            "metadata": {
                "title": self.metadata.title,
                "author": self.metadata.author,
                "publish_date": self.metadata.publish_date.isoformat() if self.metadata.publish_date else None,
                "last_modified": self.metadata.last_modified.isoformat() if self.metadata.last_modified else None,
                "language": self.metadata.language,
                "word_count": self.metadata.word_count,
                "character_count": self.metadata.character_count,
                "reading_time_minutes": self.metadata.reading_time_minutes,
                "content_type": self.metadata.content_type,
                "tags": self.metadata.tags,
                "categories": self.metadata.categories,
            },
            "metrics": {
                "extraction_time_ms": self.metrics.extraction_time_ms,
                "processing_time_ms": self.metrics.processing_time_ms,
                "confidence_score": self.metrics.confidence_score,
                "relevance_score": self.metrics.relevance_score,
                "completeness_score": self.metrics.completeness_score,
                "accuracy_score": self.metrics.accuracy_score,
                "strategy_effectiveness": self.metrics.strategy_effectiveness,
                "retry_count": self.metrics.retry_count,
                "error_count": self.metrics.error_count,
            },
            "strategy_info": {
                "strategy_name": self.strategy_info.strategy_name,
                "strategy_version": self.strategy_info.strategy_version,
                "strategy_parameters": self.strategy_info.strategy_parameters,
                "fallback_strategies": self.strategy_info.fallback_strategies,
                "adaptation_reasons": self.strategy_info.adaptation_reasons,
            },
            "success": self.success,
            "error_message": self.error_message,
            "error_type": self.error_type,
            "extraction_started": self.extraction_started.isoformat() if self.extraction_started else None,
            "extraction_completed": self.extraction_completed.isoformat() if self.extraction_completed else None,
            "custom_fields": self.custom_fields,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractionResult':
        """Create from dictionary"""
        # Parse dates
        extraction_started = None
        if data.get("extraction_started"):
            extraction_started = datetime.fromisoformat(data["extraction_started"])
        
        extraction_completed = None
        if data.get("extraction_completed"):
            extraction_completed = datetime.fromisoformat(data["extraction_completed"])
        
        publish_date = None
        if data.get("metadata", {}).get("publish_date"):
            publish_date = datetime.fromisoformat(data["metadata"]["publish_date"])
        
        last_modified = None
        if data.get("metadata", {}).get("last_modified"):
            last_modified = datetime.fromisoformat(data["metadata"]["last_modified"])
        
        # Create metadata
        metadata = ContentMetadata(
            title=data.get("metadata", {}).get("title"),
            author=data.get("metadata", {}).get("author"),
            publish_date=publish_date,
            last_modified=last_modified,
            language=data.get("metadata", {}).get("language"),
            word_count=data.get("metadata", {}).get("word_count", 0),
            character_count=data.get("metadata", {}).get("character_count", 0),
            reading_time_minutes=data.get("metadata", {}).get("reading_time_minutes", 0.0),
            content_type=data.get("metadata", {}).get("content_type"),
            tags=data.get("metadata", {}).get("tags", []),
            categories=data.get("metadata", {}).get("categories", []),
        )
        
        # Create metrics
        metrics = ExtractionMetrics(
            extraction_time_ms=data.get("metrics", {}).get("extraction_time_ms", 0.0),
            processing_time_ms=data.get("metrics", {}).get("processing_time_ms", 0.0),
            confidence_score=data.get("metrics", {}).get("confidence_score", 0.0),
            relevance_score=data.get("metrics", {}).get("relevance_score", 0.0),
            completeness_score=data.get("metrics", {}).get("completeness_score", 0.0),
            accuracy_score=data.get("metrics", {}).get("accuracy_score", 0.0),
            strategy_effectiveness=data.get("metrics", {}).get("strategy_effectiveness", 0.0),
            retry_count=data.get("metrics", {}).get("retry_count", 0),
            error_count=data.get("metrics", {}).get("error_count", 0),
        )
        
        # Create strategy info
        strategy_info = StrategyInfo(
            strategy_name=data.get("strategy_info", {}).get("strategy_name", ""),
            strategy_version=data.get("strategy_info", {}).get("strategy_version", ""),
            strategy_parameters=data.get("strategy_info", {}).get("strategy_parameters", {}),
            fallback_strategies=data.get("strategy_info", {}).get("fallback_strategies", []),
            adaptation_reasons=data.get("strategy_info", {}).get("adaptation_reasons", []),
        )
        
        return cls(
            url=data["url"],
            content=data["content"],
            raw_html=data.get("raw_html"),
            metadata=metadata,
            metrics=metrics,
            strategy_info=strategy_info,
            success=data.get("success", True),
            error_message=data.get("error_message"),
            error_type=data.get("error_type"),
            extraction_started=extraction_started,
            extraction_completed=extraction_completed,
            custom_fields=data.get("custom_fields", {}),
        )
    
    def save_to_file(self, filepath: str) -> None:
        """Save extraction result to file"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=2, ensure_ascii=False)
    
    @classmethod
    def load_from_file(cls, filepath: str) -> 'ExtractionResult':
        """Load extraction result from file"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return cls.from_dict(data) 