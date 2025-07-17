# 🧠 Intelligent Web Extractor - Usage Guide

## 🚀 Simple Interface

The Intelligent Web Extractor provides a **simple, clean interface** where you just tell it what you want and it extracts it for you.

## 📝 Basic Usage

### 1. Simple Extraction

```python
from intelligent_extractor import extract

# Tell it what you want
result = await extract(
    "https://example.com",
    "Get the main article content"
)

print(result["data"])
```

### 2. Extract with Custom Format

```python
# Specify exactly how you want the data
result = await extract(
    "https://example.com",
    "Get product information",
    output_format={
        "title": "string",
        "price": "number",
        "description": "string"
    }
)
```

### 3. CLI Usage

```bash
# Simple command line usage
python extract.py "https://example.com" "Get the main content"

# Save to file
python extract.py "https://example.com" "Get product prices" --output-file result.json
```

## 🎯 Real Examples

### Extract News Article

```python
result = await extract(
    "https://www.bbc.com/pidgin/articles/cqle6xr4qzzo",
    "Extract the main article content, title, and key information about the helicopter crash"
)

# Output:
# {
#   "title": "Ghana Airforce helicopter crash victims...",
#   "content": "Ghana Defence minister, Environment minister, six odas die for helicopter crash...",
#   "word_count": 1309,
#   "character_count": 8173
# }
```

### Extract Product Information

```python
result = await extract(
    "https://books.toscrape.com",
    "Get product titles and prices",
    output_format={
        "title": "string",
        "price": "number",
        "description": "string"
    }
)
```

### Extract Contact Information

```python
result = await extract(
    "https://example.com",
    "Find contact information and email addresses"
)

# Output:
# {
#   "contact": {
#     "email": "contact@example.com",
#     "phone": "123-456-7890"
#   }
# }
```

## 🔧 Advanced Features

### Batch Processing

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
```

### Synchronous Usage

```python
from intelligent_extractor import extract_sync

result = extract_sync(
    "https://example.com",
    "Get the main article content"
)
```

### Include Raw HTML

```python
result = await extract(
    "https://example.com",
    "Get the main content",
    include_raw_html=True
)

html_content = result["raw_html"]
```

## 🎯 What You Can Extract

The extractor understands various prompts:

- **📰 Articles**: "Get the main article content and title"
- **🛒 Products**: "Extract product titles and prices"
- **📞 Contact**: "Find email addresses and phone numbers"
- **📊 Data**: "Get the table data with columns"
- **👤 People**: "Extract names and positions"
- **📅 Dates**: "Find publication dates and times"
- **📍 Locations**: "Extract addresses and locations"
- **💰 Prices**: "Get all prices and costs"
- **⭐ Ratings**: "Find ratings and reviews"

## 📊 Output Formats

### Default JSON Format

```python
{
    "title": "Page Title",
    "content": "Main content...",
    "author": "Author Name",
    "word_count": 1500,
    "character_count": 8500
}
```

### Custom Format

```python
# Define your own structure
output_format = {
    "victims": "list",
    "total_deaths": "number",
    "crash_location": "string",
    "crash_date": "string"
}

result = await extract(url, prompt, output_format)
```

## 🚀 Quick Start Examples

### 1. Extract Article Content

```python
from intelligent_extractor import extract

result = await extract(
    "https://example.com",
    "Extract the main article content and author"
)

print(result["data"]["content"])
```

### 2. Extract Product Data

```python
result = await extract(
    "https://books.toscrape.com",
    "Get all book titles and prices",
    output_format={
        "title": "string",
        "price": "number"
    }
)
```

### 3. Extract Contact Info

```python
result = await extract(
    "https://example.com",
    "Find all contact information"
)
```

### 4. Extract Structured Data

```python
result = await extract(
    "https://example.com",
    "Extract the table data with product information",
    output_format={
        "products": "list",
        "total_items": "number"
    }
)
```

## 🎯 Tips for Better Results

1. **Be Specific**: "Get product prices" vs "Get data"
2. **Use Clear Language**: "Extract contact information" vs "Find stuff"
3. **Specify Format**: Use `output_format` for structured data
4. **Test Different Prompts**: Try variations to get better results

## 🔧 Troubleshooting

### Common Issues

1. **No Data Extracted**: Try a more specific prompt
2. **Wrong Format**: Check your `output_format` specification
3. **Slow Performance**: Use batch processing for multiple URLs
4. **API Errors**: Check your API key configuration

### Debug Mode

```python
result = await extract(
    "https://example.com",
    "Get the main content",
    include_raw_html=True  # Get raw HTML for debugging
)
```

## 🚀 That's It!

The Intelligent Web Extractor is designed to be **simple and intuitive**. Just tell it what you want, and it extracts the data for you in the format you need.

**No complex configuration. No difficult setup. Just simple, powerful web extraction.** 