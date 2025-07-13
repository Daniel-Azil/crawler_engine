# 🧠 Intelligent Web Extractor

> **"Just tell it what you want in plain English, and it extracts it"** 🤯

The world's most intuitive web scraper. No CSS selectors. No HTML parsing. Just natural language.

[![PyPI version](https://badge.fury.io/py/intelligent-web-extractor.svg)](https://pypi.org/project/intelligent-web-extractor/)
[![Downloads](https://pepy.tech/badge/intelligent-web-extractor)](https://pepy.tech/project/intelligent-web-extractor)
[![Stars](https://img.shields.io/github/stars/username/intelligent-web-extractor)](https://github.com/username/intelligent-web-extractor)

## 🔥 Why This Changes Everything

```python
# OLD WAY (100+ lines of code)
soup = BeautifulSoup(html)
title = soup.find('h1', class_='article-title').text
content = soup.find('div', class_='content').get_text()
# ... 95 more lines of fragile CSS selectors

# NEW WAY (1 line of code)
result = await extract("https://any-site.com", "Get article title and content")
```

## ✨ Features That Matter

- **🎯 One-Line Extraction**: `extract(url, "what you want")`
- **🤖 AI-Powered**: Understands context, not just structure
- **📊 Smart Output**: Returns exactly the format you need
- **⚡ Lightning Fast**: Async processing + intelligent caching
- **🔧 Zero Setup**: Works out of the box

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd intelligent-web-extractor

# Install dependencies
pip install -r requirements-windows.txt

# Install Playwright browsers
playwright install chromium
```

### Basic Usage

```python
from intelligent_extractor import extract

# Simple extraction
result = await extract(
    "https://example.com",
    "Get the main article content and title"
)

print(result["data"])
```

### CLI Usage

```bash
# Extract article content
python extract.py "https://example.com" "Get the main article content"

# Extract with custom format
python extract.py "https://example.com" "Get product prices" --format json

# Save to file
python extract.py "https://example.com" "Find contact info" --output-file result.json
```

## 📖 Examples

### 1. Extract Article Content

```python
from intelligent_extractor import extract

result = await extract(
    "https://www.bbc.com/pidgin/articles/cqle6xr4qzzo",
    "Extract the main article content, title, and key information about the helicopter crash"
)

print(result["data"])
# Output:
# {
#   "title": "Ghana Airforce helicopter crash victims...",
#   "content": "Ghana Defence minister, Environment minister, six odas die for helicopter crash...",
#   "word_count": 1309,
#   "character_count": 8173
# }
```

### 2. Extract with Custom Format

```python
result = await extract(
    "https://books.toscrape.com",
    "Get product information",
    output_format={
        "title": "string",
        "price": "number", 
        "description": "string"
    }
)

print(result["data"])
# Output:
# {
#   "title": "A Light in the Attic",
#   "price": 51.77,
#   "description": "Book description..."
# }
```

### 3. Extract Contact Information

```python
result = await extract(
    "https://example.com",
    "Find contact information and email addresses"
)

print(result["data"])
# Output:
# {
#   "contact": {
#     "email": "contact@example.com",
#     "phone": "123-456-7890"
#   }
# }
```

### 4. Batch Processing

```python
from intelligent_extractor import extract_batch

urls = [
    "https://example1.com",
    "https://example2.com",
    "https://example3.com"
]

results = await extract_batch(
    urls,
    "Get the main content from each page"
)

for i, result in enumerate(results):
    print(f"URL {i+1}: {'✅' if result['success'] else '❌'}")
```

### 5. Synchronous Usage

```python
from intelligent_extractor import extract_sync

result = extract_sync(
    "https://example.com",
    "Get the main article content"
)

print(result["data"])
```

## 🎯 Advanced Usage

### Custom Output Formats

```python
# Define your own structure
result = await extract(
    "https://example.com",
    "Extract victim information from the crash report",
    output_format={
        "victims": "list",
        "total_deaths": "number", 
        "crash_location": "string",
        "crash_date": "string"
    }
)
```

### Include Raw HTML

```python
result = await extract(
    "https://example.com",
    "Get the main content",
    include_raw_html=True
)

# Access raw HTML
html_content = result["raw_html"]
```

### With API Key (AI Features)

```python
result = await extract(
    "https://example.com",
    "Extract the main article content and summarize it",
    api_key="your_openai_api_key"
)
```

## 🔧 Configuration

### Environment Variables

Create a `.env` file:

```env
# AI Configuration
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here

# Browser Configuration  
INTELLIGENT_EXTRACTOR_BROWSER_TYPE=chromium
INTELLIGENT_EXTRACTOR_HEADLESS=true

# Performance Configuration
INTELLIGENT_EXTRACTOR_MAX_WORKERS=10
INTELLIGENT_EXTRACTOR_TIMEOUT=30
```

## 📊 Output Format

The extractor returns a structured response:

```python
{
    "success": True,
    "url": "https://example.com",
    "prompt": "Get the main content",
    "data": {
        "title": "Page Title",
        "content": "Extracted content...",
        "author": "Author Name",
        # ... other extracted data
    },
    "timestamp": "2025-08-06T19:36:31.499935",
    "extraction_method": "ai_heuristic"
}
```

## 🎯 Supported Prompts

The extractor understands various types of requests:

- **Content Extraction**: "Get the main article content"
- **Product Information**: "Extract product titles and prices"
- **Contact Information**: "Find email addresses and phone numbers"
- **Structured Data**: "Get the table data with columns"
- **Metadata**: "Extract the title, author, and publish date"
- **Custom Queries**: "Find all mentions of specific keywords"

## 🚀 Performance

- **Speed**: ~2-3 seconds per URL
- **Memory**: Minimal memory usage
- **Concurrency**: Supports batch processing
- **Reliability**: Handles various content types

## 🔧 Development

### Project Structure

```
intelligent-web-extractor/
├── src/
│   └── intelligent_web_extractor/
│       ├── core/           # Core extraction logic
│       ├── models/         # Data models
│       ├── strategies/     # Extraction strategies
│       └── utils/          # Utility functions
├── intelligent_extractor.py    # Simple interface
├── extract.py              # CLI interface
├── simple_extractor.py     # Core extractor class
└── requirements-windows.txt
```

### Adding New Features

1. **New Extraction Strategy**: Add to `src/intelligent_web_extractor/strategies/`
2. **New Output Format**: Modify `simple_extractor.py`
3. **New CLI Options**: Update `extract.py`

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

MIT License - see LICENSE file for details

## 🆘 Support

- **Issues**: Create an issue on GitHub
- **Documentation**: Check the examples above
- **Questions**: Open a discussion

---

**Made with ❤️ for intelligent web extraction** 