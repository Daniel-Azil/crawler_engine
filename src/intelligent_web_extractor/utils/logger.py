"""
Logger Utility

Provides intelligent logging functionality for the web content extraction engine
with configurable levels, formatting, and output destinations.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import json


class ExtractorLogger:
    """
    Intelligent logger for the web content extraction engine.
    
    Provides structured logging with configurable levels, multiple output
    destinations, and performance tracking capabilities.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the extractor logger.
        
        Args:
            name: Logger name
            config: Optional logging configuration
        """
        self.name = name
        self.config = config or {}
        
        # Get logger instance
        self.logger = logging.getLogger(name)
        
        # Configure logger if not already configured
        if not self.logger.handlers:
            self._configure_logger()
        
        # Performance tracking
        self._log_count = 0
        self._error_count = 0
        self._warning_count = 0
        self._start_time = datetime.now()
    
    def _configure_logger(self):
        """Configure the logger with handlers and formatters"""
        # Set log level
        log_level = self.config.get("log_level", "INFO")
        self.logger.setLevel(getattr(logging, log_level.upper()))
        
        # Create formatter
        formatter = logging.Formatter(
            fmt=self.config.get("log_format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        
        # Console handler
        if self.config.get("enable_console_logging", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.get("enable_file_logging", False):
            log_file = self.config.get("log_file")
            log_directory = self.config.get("log_directory", "./logs")
            
            if log_file:
                file_path = Path(log_file)
            else:
                # Create default log file
                log_dir = Path(log_directory)
                log_dir.mkdir(parents=True, exist_ok=True)
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                file_path = log_dir / f"extractor_{timestamp}.log"
            
            # Create rotating file handler
            file_handler = logging.handlers.RotatingFileHandler(
                file_path,
                maxBytes=10 * 1024 * 1024,  # 10MB
                backupCount=5
            )
            file_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
        
        # Prevent propagation to root logger
        self.logger.propagate = False
    
    def debug(self, message: str, **kwargs):
        """Log debug message"""
        self._log(logging.DEBUG, message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message"""
        self._log(logging.INFO, message, **kwargs)
    
    def warning(self, message: str, **kwargs):
        """Log warning message"""
        self._log(logging.WARNING, message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message"""
        self._log(logging.ERROR, message, **kwargs)
    
    def critical(self, message: str, **kwargs):
        """Log critical message"""
        self._log(logging.CRITICAL, message, **kwargs)
    
    def _log(self, level: int, message: str, **kwargs):
        """Internal logging method with performance tracking"""
        # Track log counts
        self._log_count += 1
        if level >= logging.ERROR:
            self._error_count += 1
        elif level >= logging.WARNING:
            self._warning_count += 1
        
        # Add extra context if provided
        if kwargs:
            extra_info = " | ".join(f"{k}={v}" for k, v in kwargs.items())
            message = f"{message} | {extra_info}"
        
        # Log the message
        self.logger.log(level, message)
    
    def log_extraction_start(self, url: str, strategy: str):
        """Log extraction start"""
        self.info(f"Starting extraction", url=url, strategy=strategy)
    
    def log_extraction_success(self, url: str, content_length: int, confidence: float, time_ms: float):
        """Log successful extraction"""
        self.info(
            f"Extraction completed successfully",
            url=url,
            content_length=content_length,
            confidence=f"{confidence:.2f}",
            time_ms=f"{time_ms:.0f}"
        )
    
    def log_extraction_failure(self, url: str, error: str):
        """Log extraction failure"""
        self.error(f"Extraction failed", url=url, error=error)
    
    def log_ai_interaction(self, model: str, prompt_length: int, response_length: int, time_ms: float):
        """Log AI interaction"""
        if self.config.get("log_ai_interactions", False):
            self.debug(
                f"AI interaction completed",
                model=model,
                prompt_length=prompt_length,
                response_length=response_length,
                time_ms=f"{time_ms:.0f}"
            )
    
    def log_performance_metrics(self, metrics: Dict[str, Any]):
        """Log performance metrics"""
        if self.config.get("log_performance_metrics", True):
            self.info(f"Performance metrics", **metrics)
    
    def log_extraction_details(self, details: Dict[str, Any]):
        """Log detailed extraction information"""
        if self.config.get("log_extraction_details", False):
            self.debug(f"Extraction details", **details)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get logger statistics"""
        uptime = datetime.now() - self._start_time
        
        return {
            "total_logs": self._log_count,
            "error_count": self._error_count,
            "warning_count": self._warning_count,
            "info_count": self._log_count - self._error_count - self._warning_count,
            "error_rate": self._error_count / max(self._log_count, 1),
            "uptime_seconds": uptime.total_seconds(),
            "logs_per_minute": self._log_count / max(uptime.total_seconds() / 60, 1)
        }
    
    def reset_stats(self):
        """Reset logger statistics"""
        self._log_count = 0
        self._error_count = 0
        self._warning_count = 0
        self._start_time = datetime.now()


class StructuredLogger(ExtractorLogger):
    """
    Structured logger that outputs JSON-formatted logs.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._configure_structured_logging()
    
    def _configure_structured_logging(self):
        """Configure structured JSON logging"""
        # Remove existing handlers
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # Create JSON formatter
        class JSONFormatter(logging.Formatter):
            def format(self, record):
                log_entry = {
                    "timestamp": datetime.fromtimestamp(record.created).isoformat(),
                    "level": record.levelname,
                    "logger": record.name,
                    "message": record.getMessage(),
                    "module": record.module,
                    "function": record.funcName,
                    "line": record.lineno
                }
                
                # Add extra fields if present
                if hasattr(record, 'extras'):
                    log_entry.update(record.extras)
                
                return json.dumps(log_entry)
        
        # Set up handlers with JSON formatter
        formatter = JSONFormatter()
        
        # Console handler
        if self.config.get("enable_console_logging", True):
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(formatter)
            self.logger.addHandler(console_handler)
        
        # File handler
        if self.config.get("enable_file_logging", False):
            log_file = self.config.get("log_file")
            if log_file:
                file_handler = logging.handlers.RotatingFileHandler(
                    log_file,
                    maxBytes=10 * 1024 * 1024,
                    backupCount=5
                )
                file_handler.setFormatter(formatter)
                self.logger.addHandler(file_handler)
    
    def _log(self, level: int, message: str, **kwargs):
        """Log with structured data"""
        # Create log record with extra data
        record = self.logger.makeRecord(
            self.name, level, "", 0, message, (), None
        )
        record.extras = kwargs
        
        self.logger.handle(record)
        
        # Update statistics
        self._log_count += 1
        if level >= logging.ERROR:
            self._error_count += 1
        elif level >= logging.WARNING:
            self._warning_count += 1


class PerformanceLogger(ExtractorLogger):
    """
    Performance-focused logger with detailed timing and metrics.
    """
    
    def __init__(self, name: str, config: Optional[Dict[str, Any]] = None):
        super().__init__(name, config)
        self._timings = {}
        self._metrics = {}
    
    def start_timer(self, operation: str):
        """Start timing an operation"""
        self._timings[operation] = datetime.now()
    
    def end_timer(self, operation: str) -> float:
        """End timing an operation and return duration in seconds"""
        if operation in self._timings:
            start_time = self._timings[operation]
            duration = (datetime.now() - start_time).total_seconds()
            del self._timings[operation]
            
            self.debug(f"Operation completed", operation=operation, duration_seconds=f"{duration:.3f}")
            return duration
        return 0.0
    
    def log_metric(self, name: str, value: Any, unit: str = ""):
        """Log a performance metric"""
        self._metrics[name] = {"value": value, "unit": unit, "timestamp": datetime.now()}
        self.debug(f"Metric recorded", metric_name=name, value=value, unit=unit)
    
    def get_metrics(self) -> Dict[str, Any]:
        """Get all recorded metrics"""
        return {
            name: {
                "value": data["value"],
                "unit": data["unit"],
                "timestamp": data["timestamp"].isoformat()
            }
            for name, data in self._metrics.items()
        }
    
    def clear_metrics(self):
        """Clear all recorded metrics"""
        self._metrics.clear()
        self._timings.clear()


def get_logger(name: str, logger_type: str = "standard", config: Optional[Dict[str, Any]] = None) -> ExtractorLogger:
    """
    Get a logger instance of the specified type.
    
    Args:
        name: Logger name
        logger_type: Type of logger ("standard", "structured", "performance")
        config: Optional configuration
        
    Returns:
        Logger instance
    """
    if logger_type == "structured":
        return StructuredLogger(name, config)
    elif logger_type == "performance":
        return PerformanceLogger(name, config)
    else:
        return ExtractorLogger(name, config) 