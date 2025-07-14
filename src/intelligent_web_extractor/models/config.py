"""
Configuration Models

This module defines configuration classes for the intelligent web extractor.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Union
from enum import Enum
from pathlib import Path
import os


class AIModelType(str, Enum):
    """Available AI model types"""
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    LOCAL = "local"
    HYBRID = "hybrid"
    OLLAMA = "ollama"


class ExtractionStrategy(str, Enum):
    """Available extraction strategies"""
    SEMANTIC = "semantic"
    STRUCTURED = "structured"
    HYBRID = "hybrid"
    ADAPTIVE = "adaptive"
    RULE_BASED = "rule_based"


class BrowserType(str, Enum):
    """Available browser types"""
    CHROMIUM = "chromium"
    FIREFOX = "firefox"
    WEBKIT = "webkit"


@dataclass
class AIModelConfig:
    """Configuration for AI model settings"""
    model_type: AIModelType = AIModelType.OPENAI
    model_name: str = "gpt-4"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.1
    max_tokens: int = 4000
    timeout: int = 60
    retry_attempts: int = 3
    retry_delay: float = 1.0
    
    # Embedding model settings
    embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_dimension: int = 384
    embedding_batch_size: int = 32
    
    # Local model settings
    local_model_path: Optional[str] = None
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Ollama settings
    ollama_base_url: Optional[str] = None
    ollama_endpoint: Optional[str] = None
    
    def __post_init__(self):
        """Post-initialization validation"""
        if self.api_key is None:
            if self.model_type == AIModelType.OLLAMA:
                self.api_key = "ollama_local"  # Placeholder for local ollama
            else:
                self.api_key = os.getenv("INTELLIGENT_EXTRACTOR_API_KEY")
        
        if self.base_url is None:
            if self.model_type == AIModelType.OPENAI:
                self.base_url = os.getenv("OPENAI_API_BASE", "https://api.openai.com/v1")
            elif self.model_type == AIModelType.ANTHROPIC:
                self.base_url = os.getenv("ANTHROPIC_API_BASE", "https://api.anthropic.com")
            elif self.model_type == AIModelType.OLLAMA:
                self.base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
                
        # Set Ollama-specific configurations
        if self.model_type == AIModelType.OLLAMA:
            self.ollama_base_url = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
            self.ollama_endpoint = os.getenv("OLLAMA_API_ENDPOINT", f"{self.ollama_base_url}/api/chat")


@dataclass
class BrowserConfig:
    """Configuration for browser settings"""
    browser_type: BrowserType = BrowserType.CHROMIUM
    headless: bool = True
    user_agent: Optional[str] = None
    viewport_width: int = 1920
    viewport_height: int = 1080
    timeout: int = 30000
    wait_for_load_state: str = "networkidle"
    ignore_https_errors: bool = True
    bypass_csp: bool = True
    
    # Performance settings
    max_concurrent_pages: int = 5
    memory_limit_mb: int = 512
    cpu_limit_percent: int = 50
    
    # Proxy settings
    proxy_server: Optional[str] = None
    proxy_username: Optional[str] = None
    proxy_password: Optional[str] = None
    
    # Additional arguments
    browser_args: List[str] = field(default_factory=lambda: [
        "--no-sandbox",
        "--disable-dev-shm-usage",
        "--disable-gpu",
        "--disable-web-security",
        "--disable-features=VizDisplayCompositor"
    ])


@dataclass
class ExtractionConfig:
    """Configuration for extraction settings"""
    strategy: ExtractionStrategy = ExtractionStrategy.ADAPTIVE
    confidence_threshold: float = 0.7
    relevance_threshold: float = 0.6
    max_retries: int = 3
    retry_delay: float = 1.0
    
    # Content processing
    min_content_length: int = 100
    max_content_length: int = 100000
    remove_ads: bool = True
    remove_navigation: bool = True
    remove_footers: bool = True
    remove_headers: bool = False
    
    # Semantic extraction
    semantic_chunk_size: int = 1000
    semantic_overlap: int = 200
    semantic_max_chunks: int = 10
    
    # Structured extraction
    extract_tables: bool = True
    extract_lists: bool = True
    extract_forms: bool = False
    extract_links: bool = True
    extract_images: bool = False
    
    # Custom selectors
    content_selectors: List[str] = field(default_factory=list)
    exclude_selectors: List[str] = field(default_factory=list)
    title_selectors: List[str] = field(default_factory=lambda: ["h1", "title"])
    author_selectors: List[str] = field(default_factory=lambda: [".author", "[data-author]"])
    date_selectors: List[str] = field(default_factory=lambda: [".date", "[data-date]", "time"])


@dataclass
class PerformanceConfig:
    """Configuration for performance settings"""
    max_workers: int = 10
    max_concurrent_requests: int = 5
    request_timeout: int = 30
    connection_timeout: int = 10
    
    # Rate limiting
    requests_per_second: float = 2.0
    requests_per_minute: int = 60
    delay_between_requests: float = 0.5
    
    # Memory management
    max_memory_usage_mb: int = 1024
    enable_garbage_collection: bool = True
    gc_frequency: int = 100  # Every N requests
    
    # Caching
    enable_caching: bool = True
    cache_ttl_seconds: int = 3600
    cache_max_size_mb: int = 100
    cache_directory: Optional[str] = None


@dataclass
class LoggingConfig:
    """Configuration for logging settings"""
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: Optional[str] = None
    log_directory: Optional[str] = None
    enable_console_logging: bool = True
    enable_file_logging: bool = False
    
    # Performance logging
    log_performance_metrics: bool = True
    log_extraction_details: bool = False
    log_ai_interactions: bool = False
    
    # Error logging
    log_errors: bool = True
    log_warnings: bool = True
    log_debug: bool = False


@dataclass
class ExtractorConfig:
    """
    Main configuration class for the intelligent web extractor
    
    This class combines all configuration settings into a single,
    easy-to-use configuration object.
    """
    
    # Core configurations
    ai_model: AIModelConfig = field(default_factory=AIModelConfig)
    browser: BrowserConfig = field(default_factory=BrowserConfig)
    extraction: ExtractionConfig = field(default_factory=ExtractionConfig)
    performance: PerformanceConfig = field(default_factory=PerformanceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    
    # Global settings
    debug_mode: bool = False
    test_mode: bool = False
    dry_run: bool = False
    
    # Output settings
    output_format: str = "json"  # json, markdown, text, html
    output_directory: Optional[str] = None
    include_metadata: bool = True
    include_raw_html: bool = False
    include_screenshots: bool = False
    
    # User agent rotation
    enable_user_agent_rotation: bool = True
    user_agents: List[str] = field(default_factory=lambda: [
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    ])
    
    # Custom settings
    custom_settings: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        """Post-initialization processing"""
        # Set default output directory
        if self.output_directory is None:
            self.output_directory = os.getenv("INTELLIGENT_EXTRACTOR_OUTPUT_DIR", "./extractions")
        
        # Set default cache directory
        if self.performance.cache_directory is None:
            self.performance.cache_directory = os.getenv("INTELLIGENT_EXTRACTOR_CACHE_DIR", "./cache")
        
        # Set default log directory
        if self.logging.log_directory is None:
            self.logging.log_directory = os.getenv("INTELLIGENT_EXTRACTOR_LOG_DIR", "./logs")
    
    @classmethod
    def from_env(cls) -> 'ExtractorConfig':
        """Create configuration from environment variables"""
        config = cls()
        
        # AI Model settings
        if os.getenv("INTELLIGENT_EXTRACTOR_MODEL_TYPE"):
            config.ai_model.model_type = AIModelType(os.getenv("INTELLIGENT_EXTRACTOR_MODEL_TYPE"))
        
        if os.getenv("INTELLIGENT_EXTRACTOR_MODEL_NAME"):
            config.ai_model.model_name = os.getenv("INTELLIGENT_EXTRACTOR_MODEL_NAME")
        
        if os.getenv("INTELLIGENT_EXTRACTOR_API_KEY"):
            config.ai_model.api_key = os.getenv("INTELLIGENT_EXTRACTOR_API_KEY")
            
        # Ollama-specific settings
        if os.getenv("OLLAMA_API_BASE"):
            config.ai_model.ollama_base_url = os.getenv("OLLAMA_API_BASE")
            
        if os.getenv("OLLAMA_API_ENDPOINT"):
            config.ai_model.ollama_endpoint = os.getenv("OLLAMA_API_ENDPOINT")
        
        # Browser settings
        if os.getenv("INTELLIGENT_EXTRACTOR_BROWSER_TYPE"):
            config.browser.browser_type = BrowserType(os.getenv("INTELLIGENT_EXTRACTOR_BROWSER_TYPE"))
        
        if os.getenv("INTELLIGENT_EXTRACTOR_HEADLESS"):
            config.browser.headless = os.getenv("INTELLIGENT_EXTRACTOR_HEADLESS").lower() == "true"
        
        # Performance settings
        if os.getenv("INTELLIGENT_EXTRACTOR_MAX_WORKERS"):
            config.performance.max_workers = int(os.getenv("INTELLIGENT_EXTRACTOR_MAX_WORKERS"))
        
        if os.getenv("INTELLIGENT_EXTRACTOR_TIMEOUT"):
            config.performance.request_timeout = int(os.getenv("INTELLIGENT_EXTRACTOR_TIMEOUT"))
        
        # Logging settings
        if os.getenv("INTELLIGENT_EXTRACTOR_LOG_LEVEL"):
            config.logging.log_level = os.getenv("INTELLIGENT_EXTRACTOR_LOG_LEVEL")
        
        return config
    
    @classmethod
    def from_file(cls, filepath: str) -> 'ExtractorConfig':
        """Create configuration from file"""
        import yaml
        import json
        
        filepath = Path(filepath)
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        elif filepath.suffix.lower() == '.json':
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")
        
        return cls.from_dict(data)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ExtractorConfig':
        """Create configuration from dictionary"""
        config = cls()
        
        # Update AI model config
        if "ai_model" in data:
            ai_data = data["ai_model"]
            config.ai_model.model_type = AIModelType(ai_data.get("model_type", "openai"))
            config.ai_model.model_name = ai_data.get("model_name", "gpt-4")
            config.ai_model.temperature = ai_data.get("temperature", 0.1)
            config.ai_model.max_tokens = ai_data.get("max_tokens", 4000)
        
        # Update browser config
        if "browser" in data:
            browser_data = data["browser"]
            config.browser.browser_type = BrowserType(browser_data.get("browser_type", "chromium"))
            config.browser.headless = browser_data.get("headless", True)
            config.browser.viewport_width = browser_data.get("viewport_width", 1920)
            config.browser.viewport_height = browser_data.get("viewport_height", 1080)
        
        # Update extraction config
        if "extraction" in data:
            extraction_data = data["extraction"]
            config.extraction.strategy = ExtractionStrategy(extraction_data.get("strategy", "adaptive"))
            config.extraction.confidence_threshold = extraction_data.get("confidence_threshold", 0.7)
            config.extraction.relevance_threshold = extraction_data.get("relevance_threshold", 0.6)
        
        # Update performance config
        if "performance" in data:
            perf_data = data["performance"]
            config.performance.max_workers = perf_data.get("max_workers", 10)
            config.performance.request_timeout = perf_data.get("request_timeout", 30)
            config.performance.enable_caching = perf_data.get("enable_caching", True)
        
        # Update logging config
        if "logging" in data:
            log_data = data["logging"]
            config.logging.log_level = log_data.get("log_level", "INFO")
            config.logging.enable_console_logging = log_data.get("enable_console_logging", True)
            config.logging.enable_file_logging = log_data.get("enable_file_logging", False)
        
        # Update global settings
        if "debug_mode" in data:
            config.debug_mode = data["debug_mode"]
        
        if "output_format" in data:
            config.output_format = data["output_format"]
        
        if "output_directory" in data:
            config.output_directory = data["output_directory"]
        
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary"""
        return {
            "ai_model": {
                "model_type": self.ai_model.model_type.value,
                "model_name": self.ai_model.model_name,
                "temperature": self.ai_model.temperature,
                "max_tokens": self.ai_model.max_tokens,
                "timeout": self.ai_model.timeout,
                "retry_attempts": self.ai_model.retry_attempts,
            },
            "browser": {
                "browser_type": self.browser.browser_type.value,
                "headless": self.browser.headless,
                "viewport_width": self.browser.viewport_width,
                "viewport_height": self.browser.viewport_height,
                "timeout": self.browser.timeout,
                "max_concurrent_pages": self.browser.max_concurrent_pages,
            },
            "extraction": {
                "strategy": self.extraction.strategy.value,
                "confidence_threshold": self.extraction.confidence_threshold,
                "relevance_threshold": self.extraction.relevance_threshold,
                "max_retries": self.extraction.max_retries,
                "min_content_length": self.extraction.min_content_length,
                "max_content_length": self.extraction.max_content_length,
            },
            "performance": {
                "max_workers": self.performance.max_workers,
                "max_concurrent_requests": self.performance.max_concurrent_requests,
                "request_timeout": self.performance.request_timeout,
                "enable_caching": self.performance.enable_caching,
                "cache_ttl_seconds": self.performance.cache_ttl_seconds,
            },
            "logging": {
                "log_level": self.logging.log_level,
                "enable_console_logging": self.logging.enable_console_logging,
                "enable_file_logging": self.logging.enable_file_logging,
                "log_performance_metrics": self.logging.log_performance_metrics,
            },
            "debug_mode": self.debug_mode,
            "output_format": self.output_format,
            "output_directory": self.output_directory,
            "include_metadata": self.include_metadata,
            "include_raw_html": self.include_raw_html,
        }
    
    def save_to_file(self, filepath: str) -> None:
        """Save configuration to file"""
        import yaml
        
        filepath = Path(filepath)
        
        if filepath.suffix.lower() in ['.yaml', '.yml']:
            with open(filepath, 'w', encoding='utf-8') as f:
                yaml.dump(self.to_dict(), f, default_flow_style=False, indent=2)
        elif filepath.suffix.lower() == '.json':
            import json
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(self.to_dict(), f, indent=2)
        else:
            raise ValueError(f"Unsupported configuration file format: {filepath.suffix}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of errors"""
        errors = []
        
        # Validate AI model settings
        if not self.ai_model.api_key and self.ai_model.model_type in [AIModelType.OPENAI, AIModelType.ANTHROPIC]:
            errors.append("API key is required for OpenAI and Anthropic models")
        
        if self.ai_model.temperature < 0 or self.ai_model.temperature > 2:
            errors.append("Temperature must be between 0 and 2")
        
        if self.ai_model.max_tokens < 1:
            errors.append("Max tokens must be positive")
        
        # Validate browser settings
        if self.browser.viewport_width < 1 or self.browser.viewport_height < 1:
            errors.append("Viewport dimensions must be positive")
        
        if self.browser.timeout < 1000:
            errors.append("Browser timeout must be at least 1000ms")
        
        # Validate extraction settings
        if self.extraction.confidence_threshold < 0 or self.extraction.confidence_threshold > 1:
            errors.append("Confidence threshold must be between 0 and 1")
        
        if self.extraction.relevance_threshold < 0 or self.extraction.relevance_threshold > 1:
            errors.append("Relevance threshold must be between 0 and 1")
        
        if self.extraction.max_retries < 0:
            errors.append("Max retries must be non-negative")
        
        # Validate performance settings
        if self.performance.max_workers < 1:
            errors.append("Max workers must be positive")
        
        if self.performance.request_timeout < 1:
            errors.append("Request timeout must be positive")
        
        if self.performance.cache_ttl_seconds < 0:
            errors.append("Cache TTL must be non-negative")
        
        return errors 