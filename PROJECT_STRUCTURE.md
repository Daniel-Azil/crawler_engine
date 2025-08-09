# Intelligent Web Extractor - Project Structure

This document provides an overview of the Intelligent Web Extractor project structure and architecture.

## 📁 Project Structure

```
intelligent-web-extractor/
├── 📄 README.md                    # Main project documentation
├── 📄 pyproject.toml              # Project configuration and dependencies
├── 📄 requirements.txt             # Python dependencies
├── 📄 setup.py                    # Installation setup script
├── 📄 env.example                 # Environment variables template
├── 📄 example_usage.py            # Usage examples
├── 📄 PROJECT_STRUCTURE.md        # This file
│
├── 📁 src/                        # Source code directory
│   └── 📁 intelligent_web_extractor/
│       ├── 📄 __init__.py         # Package initialization
│       │
│       ├── 📁 core/               # Core components
│       │   ├── 📄 __init__.py
│       │   ├── 📄 extractor.py    # Main adaptive extractor
│       │   ├── 📄 semantic_extractor.py
│       │   ├── 📄 batch_processor.py
│       │   └── 📄 custom_extractor.py
│       │
│       ├── 📁 models/             # Data models
│       │   ├── 📄 __init__.py
│       │   ├── 📄 extraction_result.py
│       │   └── 📄 config.py
│       │
│       ├── 📁 strategies/         # Extraction strategies
│       │   ├── 📄 __init__.py
│       │   ├── 📄 semantic_strategy.py
│       │   ├── 📄 structured_strategy.py
│       │   ├── 📄 hybrid_strategy.py
│       │   └── 📄 rule_based_strategy.py
│       │
│       ├── 📁 utils/              # Utility components
│       │   ├── 📄 __init__.py
│       │   ├── 📄 browser_manager.py
│       │   ├── 📄 ai_client.py
│       │   └── 📄 logger.py
│       │
│       └── 📄 cli.py              # Command line interface
│
├── 📁 tests/                      # Test files
│   ├── 📄 __init__.py
│   ├── 📄 test_extractor.py
│   ├── 📄 test_strategies.py
│   └── 📄 test_utils.py
│
├── 📁 docs/                       # Documentation
│   ├── 📄 user_guide.md
│   ├── 📄 api_reference.md
│   └── 📄 advanced_usage.md
│
├── 📁 examples/                   # Example scripts
│   ├── 📄 basic_extraction.py
│   ├── 📄 batch_processing.py
│   └── 📄 custom_rules.py
│
└── 📁 scripts/                    # Utility scripts
    ├── 📄 install_browsers.py
    └── 📄 benchmark.py
```

## 🏗️ Architecture Overview

### Core Components

#### 1. **AdaptiveContentExtractor** (`core/extractor.py`)
- Main entry point for content extraction
- Intelligently selects the best extraction strategy
- Manages browser instances and AI clients
- Provides unified interface for all extraction operations

#### 2. **Extraction Strategies** (`strategies/`)
- **SemanticStrategy**: AI-powered content understanding
- **StructuredStrategy**: Data-rich page extraction
- **HybridStrategy**: Combines multiple approaches
- **RuleBasedStrategy**: Custom rule-based extraction

#### 3. **Browser Manager** (`utils/browser_manager.py`)
- Manages Playwright browser instances
- Handles page navigation and content retrieval
- Provides browser lifecycle management
- Supports multiple browser types (Chromium, Firefox, WebKit)

#### 4. **AI Client** (`utils/ai_client.py`)
- Unified interface for AI model interactions
- Supports OpenAI, Anthropic, and local models
- Handles embedding generation and similarity calculations
- Manages API rate limiting and error handling

### Data Models

#### 1. **ExtractionResult** (`models/extraction_result.py`)
- Comprehensive result container
- Includes content, metadata, metrics, and strategy information
- Supports serialization and export formats

#### 2. **ExtractorConfig** (`models/config.py`)
- Centralized configuration management
- Supports environment variables and file-based config
- Validates configuration parameters

### Key Features

#### 🔄 **Adaptive Intelligence**
- Automatically selects the best extraction strategy
- Learns from previous extractions
- Adapts to different website structures

#### 🎯 **Multiple Strategies**
- **Semantic**: AI-powered content understanding
- **Structured**: Data-rich page extraction
- **Hybrid**: Combines multiple approaches
- **Rule-based**: Custom extraction rules

#### ⚡ **High Performance**
- Asynchronous architecture
- Concurrent processing
- Intelligent caching
- Resource optimization

#### 🔧 **Developer Friendly**
- Clean, intuitive API
- Comprehensive documentation
- Extensive configuration options
- Command-line interface

## 🚀 Usage Patterns

### Basic Usage
```python
from intelligent_web_extractor import AdaptiveContentExtractor

async with AdaptiveContentExtractor() as extractor:
    result = await extractor.extract_content(
        url="https://example.com",
        user_query="Find pricing information"
    )
    print(result.content)
```

### Batch Processing
```python
from intelligent_web_extractor import BatchProcessor

processor = BatchProcessor()
results = await processor.process_urls([
    "https://site1.com",
    "https://site2.com"
])
```

### Custom Rules
```python
from intelligent_web_extractor import CustomExtractor

extractor = CustomExtractor()
extractor.add_rule("main_content", ".content", "text")
result = await extractor.extract("https://example.com")
```

## 🔧 Configuration

### Environment Variables
- `INTELLIGENT_EXTRACTOR_API_KEY`: Main API key
- `OPENAI_API_KEY`: OpenAI API key
- `ANTHROPIC_API_KEY`: Anthropic API key
- `INTELLIGENT_EXTRACTOR_MODEL_TYPE`: AI model type
- `INTELLIGENT_EXTRACTOR_STRATEGY`: Default strategy

### Configuration File
```yaml
ai_model:
  model_type: openai
  model_name: gpt-4
  temperature: 0.1

browser:
  browser_type: chromium
  headless: true

extraction:
  strategy: adaptive
  confidence_threshold: 0.7
```

## 🧪 Testing

### Running Tests
```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_extractor.py

# Run with coverage
pytest --cov=intelligent_web_extractor
```

### Test Structure
- **Unit Tests**: Test individual components
- **Integration Tests**: Test component interactions
- **Performance Tests**: Test extraction performance
- **Strategy Tests**: Test different extraction strategies

## 📚 Documentation

### User Documentation
- **README.md**: Quick start and basic usage
- **User Guide**: Comprehensive usage guide
- **API Reference**: Complete API documentation
- **Advanced Usage**: Advanced features and techniques

### Developer Documentation
- **Architecture Guide**: System design and architecture
- **Contributing Guide**: How to contribute to the project
- **Development Setup**: Setting up development environment

## 🔄 Development Workflow

### 1. **Setup Development Environment**
```bash
# Clone repository
git clone <repository-url>
cd intelligent-web-extractor

# Install dependencies
pip install -r requirements.txt

# Setup environment
python setup.py
```

### 2. **Running Tests**
```bash
# Run all tests
pytest

# Run specific tests
pytest tests/test_extractor.py -v
```

### 3. **Code Quality**
```bash
# Format code
black src/
isort src/

# Lint code
flake8 src/
mypy src/
```

### 4. **Building Documentation**
```bash
# Build docs
mkdocs build

# Serve docs locally
mkdocs serve
```

## 🎯 Key Design Principles

### 1. **Modularity**
- Each component has a single responsibility
- Clear interfaces between components
- Easy to extend and modify

### 2. **Configurability**
- Extensive configuration options
- Environment variable support
- File-based configuration

### 3. **Performance**
- Asynchronous operations
- Intelligent caching
- Resource optimization

### 4. **Reliability**
- Comprehensive error handling
- Graceful degradation
- Health monitoring

### 5. **Usability**
- Simple, intuitive API
- Comprehensive documentation
- Multiple usage patterns

## 🔮 Future Enhancements

### Planned Features
- **Advanced AI Models**: Support for more AI providers
- **Distributed Processing**: Multi-node extraction
- **Real-time Monitoring**: Live extraction monitoring
- **Advanced Caching**: Redis-based caching
- **Plugin System**: Extensible architecture

### Performance Improvements
- **GPU Acceleration**: CUDA support for embeddings
- **Memory Optimization**: Better memory management
- **Concurrent Processing**: Improved parallelism

### Integration Features
- **Database Support**: SQL and NoSQL databases
- **Message Queues**: RabbitMQ and Redis support
- **Cloud Deployment**: Docker and Kubernetes support

This architecture provides a solid foundation for intelligent web content extraction with room for growth and enhancement. 