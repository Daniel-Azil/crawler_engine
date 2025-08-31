# Output Format System - Schema-First Approach

The intelligent extractor now uses a **schema-first** approach for output formatting. You specify exactly the structure you want, and the system returns data in that structure.

## Supported Output Formats

### 1. **Dict Schema (JSON Objects)**
Pass a dictionary describing the structure you want:

```python
# Input
output_format = {"title": "string", "price": "number", "tags": ["string"]}

# Output → dict
{
    "title": "Product Name",
    "price": 29.99,
    "tags": ["electronics", "gadgets"]
}
```

### 2. **List Schema (JSON Arrays)**
Pass a list with one item describing the array element structure:

```python
# Input - homogeneous list of objects
output_format = [{"name": "string", "url": "string"}]

# Output → list (pure list, no wrapper)
[
    {"name": "Link 1", "url": "https://example.com/1"},
    {"name": "Link 2", "url": "https://example.com/2"}
]

# Input - simple list of strings
output_format = ["string"]

# Output → list of strings
["item1", "item2", "item3"]
```

**Important**: Returns a **pure list**, NOT `{"items": [...]}` like the old system.

### 3. **Template Strings (Markdown/HTML/Text)**
Pass a string with `{placeholder}` variables:

```python
# Input - Markdown template
output_format = "# {title}\n\n{content}\n\nPrice: ${price}"

# Output → rendered string
"# Product Name\n\nThis is the product description.\n\nPrice: $29.99"

# Input - HTML template  
output_format = "<article><h1>{title}</h1><p>{content}</p></article>"

# Output → rendered HTML
"<article><h1>Product Name</h1><p>This is the content.</p></article>"
```

### 4. **Raw Content**
Pass `None` to get raw extracted content without AI formatting:

```python
# Input
output_format = None

# Output → raw string
"Raw webpage content as extracted by the strategy..."
```

### 5. **Back-Compatibility Literals**
For backward compatibility, these special strings are still supported:

```python
# "json" → inferred JSON object
output_format = "json"
# Returns: {"title": "...", "content": "...", ...}

# "string" → plain text
output_format = "string"  
# Returns: "Main content as plain text"
```

## Key Changes from Old System

### ✅ **What's New**
- **Schema-first**: You specify the exact structure you want
- **Pure lists**: Array schemas return `[...]` not `{"items": [...]}`
- **Template rendering**: Deterministic placeholder replacement
- **Type safety**: Dict schemas return dicts, list schemas return lists

### ❌ **What's Removed**
- ~~`"list"` format name~~ → Use `["string"]` instead
- ~~`"list_of_dict"` format name~~ → Use `[{"key": "type"}]` instead
- ~~`{"items": [...]}` wrappers~~ → Returns pure lists now

## Examples

```python
from intelligent_extractor import extract

# 1. Get structured product data
product = await extract(
    "https://shop.example.com/product/123",
    "Extract product information",
    output_format={
        "name": "string",
        "price": "number", 
        "description": "string",
        "features": ["string"]
    }
)
# Returns: {"name": "...", "price": 29.99, "description": "...", "features": [...]}

# 2. Get list of links
links = await extract(
    "https://news.example.com",
    "Extract all article links",
    output_format=[{"title": "string", "url": "string"}]
)
# Returns: [{"title": "Article 1", "url": "..."}, {"title": "Article 2", "url": "..."}]

# 3. Generate markdown summary
markdown = await extract(
    "https://blog.example.com/post/123",
    "Summarize this blog post",
    output_format="# {title}\n\n## Summary\n\n{summary}\n\n**Author**: {author}"
)
# Returns: "# Blog Title\n\n## Summary\n\nThis post discusses...\n\n**Author**: John Doe"

# 4. Get raw content
raw = await extract(
    "https://example.com",
    "Extract everything",
    output_format=None
)
# Returns: "Raw extracted content..."
```

## Migration Guide

If you're updating from the old system:

```python
# OLD WAY
output_format = "list"                    # ❌ No longer supported
output_format = "list_of_dict"            # ❌ No longer supported  
# Returned: {"items": [...]}              # ❌ Old wrapper format

# NEW WAY  
output_format = ["string"]                # ✅ List of strings
output_format = [{"key": "string"}]       # ✅ List of objects
# Returns: [...]                          # ✅ Pure list
```

The new system is more predictable: **you get exactly the structure you specify**.
