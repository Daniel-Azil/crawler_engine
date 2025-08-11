# 🧠 Intelligent Web Extractor — Usage Cookbook (Examples First)

A practical, example-heavy guide for real projects. Copy, paste, adapt.

- Programmatic API: import helpers from `intelligent_extractor.py` for the fastest start.
- Advanced control: use the core `AdaptiveContentExtractor` for per-URL queries/modes and fine-grained config.
- CLI: `intelligent-extractor` for single runs, batches, and diagnostics.

---

## Essentials

- Python ≥ 3.8
- Install deps and Playwright Chromium
- Configure providers via env (see `env.example`): `SERVICE_TO_USE` = openai | anthropic | gemini | ollama
- Quick verification: `intelligent-extractor doctor`

---

## API quickstart (simple helpers)

```python
from intelligent_extractor import extract

result = await extract(
    "https://example.com",
    "Extract the main article content and title"
)
print(result["data"])             # shaped result
print(result["meta"])             # metadata (title, etc.)
print(result["strategy"])         # selected strategy name
```

Parameters you can pass:
- `output_format`: shape response to a string, list, json object, or list-of-objects (see below)
- `include_raw_html`: include page HTML in the result
- `mode`: "semantic" | "structured" | "hybrid" | "adaptive" (default)
- `timeout`: per-run request timeout (seconds)
- `max_workers`: per-run concurrency cap

---

## Output formatting (deterministic shaping)

The engine can map extracted content into your requested structure:

- String: `"string"`
- JSON-ish passthrough: `"json"`
- Flat list: `"list"` (e.g., list of strings)
- List of dicts: `[ { ... field → type ... } ]`
- Object schema: `{ ... field → type ... }`

Types: "string", "number", "boolean", "list", "list_of_dict"

Examples:

```python
# 1) Summary as a string
await extract(url, "Summarize the article in 3 sentences", output_format="string")

# 2) Object schema
schema = {"title": "string", "author": "string", "published": "string"}
await extract(url, "Extract metadata", output_format=schema)

# 3) List of strings
await extract(url, "List all product names", output_format="list")

# 4) List of objects (catalog)
row = {"title": "string", "price": "number", "product_url": "string"}
await extract(url, "Extract all product cards with title, price, and URL", output_format=[row])
```

---

## Scenario cookbook

Below are focused, real-world recipes you can adapt.

### 1) News article (content + metadata)
```python
schema = {
  "headline": "string",
  "author": "string",
  "published": "string",
  "summary": "string"
}
await extract(
  "https://www.bbc.com/news/...",
  "Extract the headline, author, date, and a 3-sentence summary",
  output_format=schema
)
```

### 2) Blog index → list of posts
```python
post = {"title": "string", "url": "string", "snippet": "string"}
await extract(
  index_url,
  "List recent posts with title, URL and 1-sentence snippet",
  output_format=[post]
)
```

### 3) Product listing (cards)
```python
item = {"title": "string", "price": "number", "rating": "number", "url": "string"}
await extract(
  "https://books.toscrape.com/",
  "Extract all product cards (title, price, rating, URL)",
  output_format=[item],
  mode="structured"   # lists/tables respond well to Structured
)
```

### 4) Product detail page
```python
schema = {
  "title": "string",
  "price": "number",
  "availability": "string",
  "description": "string",
  "images": "list"
}
await extract(pdp_url, "Extract product details", output_format=schema)
```

### 5) Job board (cards)
```python
job = {"title": "string", "company": "string", "location": "string", "salary": "string", "url": "string"}
await extract(jobs_url, "Extract job cards with title, company, location, salary, URL", output_format=[job], mode="structured")
```

### 6) Real estate listing
```python
home = {"title": "string", "price": "number", "beds": "number", "baths": "number", "address": "string", "url": "string"}
await extract(listing_url, "Extract property cards (title, price, beds, baths, address, URL)", output_format=[home], mode="structured")
```

### 7) Events calendar → JSON rows
```python
row = {"name": "string", "date": "string", "time": "string", "venue": "string"}
await extract(events_url, "Extract the events table as JSON rows", output_format=[row], mode="structured")
```

### 8) FAQ page → Q&A pairs
```python
qa = {"question": "string", "answer": "string"}
await extract(url, "Extract all FAQs as question/answer pairs", output_format=[qa])
```

### 9) Contact page → emails and phones
```python
schema = {"emails": "list", "phones": "list", "address": "string"}
await extract(url, "Find all emails, phone numbers, and address", output_format=schema)
```

### 10) Documentation site → sidebar ToC
```python
entry = {"title": "string", "url": "string"}
await extract(docs_url, "List sidebar/table-of-contents entries with their URLs", output_format=[entry], mode="structured")
```

### 11) University publications list
```python
paper = {"title": "string", "authors": "list", "year": "number", "pdf": "string"}
await extract(pub_url, "Extract publications as title, authors, year, and pdf link", output_format=[paper], mode="structured")
```

### 12) Quotes page → list + stats
```python
schema = {"quotes": "list", "authors": "list", "tags": "list"}
await extract(url, "Extract notable quotes with authors and tags", output_format=schema)
```

### 13) Tables anywhere → rows
```python
row = {"column_1": "string", "column_2": "string", "column_3": "string"}
await extract(url, "Extract the main table as JSON rows", output_format=[row], mode="structured")
```

### 14) Extract all links (for follow-up crawling)
```python
await extract(url, "List all links on the page", output_format="list", mode="structured")
```

### 15) Long-form article → English summary
```python
await extract(url, "Summarize the article in English in 5 sentences", output_format="string", mode="semantic")
```

### 16) Pricing page → feature comparison
```python
plan = {"name": "string", "price_per_month": "number", "features": "list"}
await extract(url, "Extract pricing plans with price and feature bullet points", output_format=[plan], mode="structured")
```

### 17) Press releases → list of dicts
```python
press = {"title": "string", "date": "string", "url": "string"}
await extract(url, "Extract press releases with title, date, and link", output_format=[press])
```

### 18) Knowledge base article → steps
```python
step = {"order": "number", "text": "string"}
await extract(url, "Extract troubleshooting steps as numbered items", output_format=[step])
```

### 19) Careers page → teams and openings
```python
team = {"team": "string", "open_roles": "list"}
await extract(url, "Extract teams and their open roles", output_format=[team])
```

### 20) App release notes → versions
```python
ver = {"version": "string", "date": "string", "changes": "list"}
await extract(url, "Extract release notes as versioned entries", output_format=[ver])
```

---

## Include raw HTML and tune per call

```python
result = await extract(
  url,
  "Extract the main content",
  include_raw_html=True,
  mode="adaptive",
  timeout=60,
  max_workers=4,
)
print(len(result["raw_html"] or ""))
```

---

## Batch usage (simple helpers)

One prompt for all URLs:
```python
from intelligent_extractor import extract_batch

urls = ["https://site1.com", "https://site2.com", "https://site3.com"]
results = await extract_batch(urls, "Extract the main content")
```

---

## Advanced: core extractor (per-URL queries/modes, deep overrides)

Use the core engine for maximum control.

```python
import asyncio
from intelligent_web_extractor import AdaptiveContentExtractor
from intelligent_web_extractor.models.config import ExtractorConfig
from intelligent_web_extractor.models.extraction_result import ExtractionStrategy

async def run():
    config = ExtractorConfig()
    async with AdaptiveContentExtractor(config) as extractor:
        # Per-URL queries
        urls = [
            "https://news.site/article-1",
            "https://shop.site/listing",
        ]
        queries = [
            "Extract title and a 3-sentence summary",
            "Extract product cards with title and price",
        ]
        modes = [ExtractionStrategy.SEMANTIC, ExtractionStrategy.STRUCTURED]

        # Single URL with schema + overrides
        row = {"title": "string", "price": "number"}
        result = await extractor.extract_content(
            url=urls[1],
            user_query=queries[1],
            extraction_mode=modes[1],
            output_format=[row],
            custom_config={
                "performance": {"request_timeout": 45, "max_workers": 6},
                "extraction": {"semantic_chunk_size": 1200, "extract_tables": True}
            }
        )
        data = result.custom_fields.get("formatted_data") or result.content
        print(data)

        # Batch with per-URL queries/modes
        batch_results = await extractor.extract_batch(urls, queries, modes)
        for r in batch_results:
            print(r.url, r.success, r.strategy_info.strategy_name)

asyncio.run(run())
```

Notes:
- `extract_content` returns an `ExtractionResult`. If you requested `output_format`, the shaped payload is in `result.custom_fields["formatted_data"]`; otherwise use `result.content`.
- `custom_config` may override `performance` and `extraction` sub-configs for that call.

---

## CLI examples

Single URL with query and schema:
```powershell
intelligent-extractor extract "https://example.com" --query "Extract the main content" --schema schema.json --mode adaptive --format json --include-raw-html
```

Batch with progress:
```powershell
intelligent-extractor batch .\urls.txt --format json --max-workers 8 --progress
```

Interactive session:
```powershell
intelligent-extractor interactive
```

Diagnostics and config template:
```powershell
intelligent-extractor doctor
intelligent-extractor init --format yaml > extractor.config.yaml
```

---

## Validating results (recommended)

Use Pydantic (already a dependency) to validate AI-shaped data.

```python
from pydantic import BaseModel, Field
from typing import List
from intelligent_extractor import extract

class Product(BaseModel):
    title: str
    price: float

schema = {"title": "string", "price": "number"}
res = await extract(url, "Extract product title and price", output_format=schema)
validated = Product(**res["data"])     # raises if invalid
```

For list-of-dicts:
```python
class Row(BaseModel):
    title: str
    price: float

rows_schema = [{"title": "string", "price": "number"}]
res = await extract(url, "Extract products", output_format=rows_schema)
validated_rows = [Row(**it) for it in (res["data"] or [])]
```

---

## Performance tips

- Prefer `mode="structured"` for catalog-like pages (tables/lists/grids)
- Use `max_workers` judiciously (I/O bound, but model usage may rate-limit)
- Set `timeout` for flaky sites; retry upstream if needed
- Batch requests where possible

---

## Troubleshooting

- Empty/partial data: try a more specific prompt or force `mode="structured"`
- Schema mismatch: start simple, then add fields; validate with Pydantic
- Provider/API issues: check `SERVICE_TO_USE` and API keys, then run `intelligent-extractor doctor`
- Playwright/browser errors: ensure browsers installed and site reachable

---

## Return shape (simple helpers)

```python
{
  "success": bool,
  "url": str,
  "data": Any,            # shaped to your output_format (or raw content)
  "raw_html": Optional[str],
  "meta": dict,           # title, description, counts
  "strategy": str         # chosen strategy name
}
```

Need more examples? Open an issue with the target site type and desired fields.