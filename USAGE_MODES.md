# 🎯 **Usage Modes Guide**

## **Quick Decision Matrix**

| **Use Case** | **Recommended Mode** | **Why** |
|-------------|---------------------|---------|
| **Quick testing/prototyping** | Simple Extractor | Fast setup, minimal config |
| **Production web scraping** | Full System | AI strategy selection, robust error handling |
| **Complex websites** | Full System + Custom Config | Tailored settings, performance optimization |
| **Batch processing** | Full System | Built-in batch processing, concurrent handling |
| **Content analysis** | Full System with queries | AI understands context and user intent |
| **Monitoring/Analytics** | Full System | Health monitoring, performance metrics |

## **🔧 Configuration Levels**

### **Level 1: Zero Config (Simple)**
```python
from simple_extractor import SimpleExtractor
extractor = SimpleExtractor()
result = await extractor.extract_content(url)
```
- **Best for:** Quick scripts, testing, learning
- **Features:** Basic extraction, rule-based fallback

### **Level 2: Default Config (Full System)**
```python
from src.intelligent_web_extractor import AdaptiveContentExtractor
async with AdaptiveContentExtractor() as extractor:
    result = await extractor.extract_content(url)
```
- **Best for:** Production use with environment variables
- **Features:** AI strategy selection, full capabilities

### **Level 3: Custom Config (Full System)**
```python
config = ExtractorConfig()
config.ai_model.model_type = AIModelType.OLLAMA
config.extraction.confidence_threshold = 0.9
async with AdaptiveContentExtractor(config) as extractor:
    result = await extractor.extract_content(url)
```
- **Best for:** Fine-tuned performance, specific requirements
- **Features:** Full control over all settings

## **⚡ Performance Comparison**

| **Mode** | **Speed** | **Quality** | **Features** | **Resource Usage** |
|----------|-----------|-------------|--------------|-------------------|
| **Simple** | ⚡⚡⚡ Fast | ⭐⭐ Good | ⭐ Basic | 🔋 Low |
| **Full System** | ⚡⚡ Moderate | ⭐⭐⭐⭐ Excellent | ⭐⭐⭐⭐⭐ Complete | 🔋🔋 Medium |

## **📊 Strategy Selection Guide**

The Full System AI will automatically choose, but you can also force strategies:

- **🧠 SEMANTIC**: News articles, blogs, content-heavy pages
- **📊 STRUCTURED**: E-commerce, data tables, forms, listings  
- **🔄 HYBRID**: Complex pages with mixed content types
- **📋 RULE_BASED**: Consistent page structures, simple sites
- **🎯 ADAPTIVE**: Let AI decide (recommended default)

## **🚀 Getting Started Recommendations**

1. **Start with Simple Extractor** to understand basic functionality
2. **Move to Full System** when you need better quality/features
3. **Add Custom Config** when you have specific requirements
4. **Use Batch Processing** for multiple URLs
5. **Monitor Health** in production environments
