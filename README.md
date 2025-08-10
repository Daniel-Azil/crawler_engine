# Intelligent Web Extractor

A production‑grade, prompt‑first crawler/scraper. You describe what you want in plain language; the system selects the right extraction strategy, navigates pages when needed, and returns data in your requested format.

- Multiple AI providers: Ollama, OpenAI, Anthropic, Gemini
- Hidden content handling (iframes) is enabled globally
- Adaptive mode performs interactive discovery (scroll/click) and AI reasoning
- Prompt → structured data using AI schema formatting

---

## Quick Start

### Python API
```python
from intelligent_extractor import extract

# Minimal example
result = await extract(
  url="https://example.com",
  prompt="Extract the main article title and content",
  output_format={"title": "string", "content": "string"},
  mode="adaptive",      # semantic | structured | hybrid | adaptive
  timeout=60,
  max_workers=5,
)
# `data` is a single value (dict/string/list) depending on output_format
print(result["data"])
```

- `output_format`: schema dict → AI formats the result into this structure
- `mode`: choose strategy; `adaptive` adds AI navigation and reasoning
- `timeout`, `max_workers`: per‑run performance overrides

### CLI
```bash
intelligent-extractor extract \
  "https://example.com" \
  --query "Extract the main article title and content" \
  --mode adaptive \
  --schema schema.json \
  --timeout 60 \
  --max-workers 5
```

`schema.json` example:
```json
{
  "title": "string",
  "content": "string",
  "author": "string",
  "publish_date": "string"
}
```

---

## Use Cases & Recipes

### Choosing the right strategy
- Semantic
  - Use when: extracting narratives (articles, blogs, docs)
  - Behavior: AI ranks text segments semantically; no interactive navigation
- Structured
  - Use when: parsing rigid structure (tables, lists, forms, product pages)
  - Behavior: DOM parsing heuristics; no interactive navigation
- Rule‑based
  - Use when: you know exact CSS selectors and want deterministic rules
  - Behavior: deterministic and fast; no interactive navigation
- Hybrid
  - Use when: mixed pages (narratives + structure) without complex navigation
  - Behavior: combines semantic + structured; no AI reasoning navigation
- Adaptive (recommended default)
  - Use when: unknown, dynamic, paginated, infinite scroll, “show more”
  - Behavior: AI reasoning + interactive discovery (scroll/click) + best‑strategy selection

Notes
- Hidden content handling (iframes, embedded HTML) is globally enabled across modes.
- Only Adaptive performs interactive actions (scroll/click/expand) using AI guidance.

Below are common scenarios with ready‑to‑run Python snippets.

### 1) News / Articles (narrative content)
Use semantic for stable pages; use adaptive if there are expandable sections or dynamic loading.
```python
from intelligent_extractor import extract

result = await extract(
  url="https://www.bbc.com/news",
  prompt="Extract the article title, author, publish date, and main story",
  output_format={
    "title": "string",
    "author": "string",
    "publish_date": "string",
    "content": "string"
  },
  mode="semantic"  # or "adaptive" if the page needs interactions
)
print(result["data"])
```

### 2) Product Pages (structured content)
Use structured for tables/lists/attributes. Hybrid is useful for mixed narrative + specs.
```python
from intelligent_extractor import extract

result = await extract(
  url="https://books.toscrape.com/catalogue/a-light-in-the-attic_1000/index.html",
  prompt="Extract product title, price, availability, rating, description",
    output_format={
        "title": "string",
    "price": "string",
    "availability": "string",
    "rating": "string",
        "description": "string"
  },
  mode="structured"  # or "hybrid"
)
print(result["data"])
```

### 3) Listings / Infinite Scroll / “Load more”
Use adaptive so the system can scroll/click and aggregate.
```python
from intelligent_extractor import extract

result = await extract(
  url="https://example.com/listings",
  prompt=(
    "Extract the top 100 items with fields: title, price, link; handle load more or infinite scroll"
  ),
  output_format={
    "items": "list",
    "total": "number"
  },
  mode="adaptive",
  timeout=120
)
print(result["data"])
```

### 4) Tables and Forms
Use structured to parse tables or capture form schemas.
```python
from intelligent_extractor import extract

result = await extract(
  url="https://example.com/table",
  prompt="Extract all tables as rows and headers; include row_count and column_count",
    output_format={
    "tables": "list"
  },
  mode="structured"
)
print(result["data"]) 
```

### 5) Hidden Content (iframes)
Iframes are merged automatically across modes. Use adaptive if content also requires clicks.
```python
from intelligent_extractor import extract

result = await extract(
  url="https://example.com/embedded",
  prompt="Extract the embedded article title and body from the iframe",
  output_format={"title": "string", "content": "string"},
  mode="semantic"  # switch to "adaptive" if the page requires interactions
)
print(result["data"]) 
```

---

## Providers and .env
Use `env.example` as a catalog of server‑level settings you can copy to your `.env`. The application loads `.env` automatically.

Minimal provider configuration:
```env
SERVICE_TO_USE=ollama  # or openai | anthropic | gemini

# Ollama
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL_NAME=llama3

# OpenAI
OPENAI_API_KEY=...

# Anthropic
ANTHROPIC_API_KEY=...

# Gemini
GEMINI_API_KEY=...
```

Also see `env.example` for server defaults (directories, logging, performance, caching, proxy, security).

---

## Choosing parameters
- `mode`
  - Start with `adaptive` for unknown/dynamic sites
  - Use `semantic` for text‑heavy content without interactions
  - Use `structured` for tables/lists/forms and rigid layouts
- `output_format`
  - Accepts: dict schema, "json", "string", [schema] for lists, "list", or "list_of_dict"
  - Dict schema example: {"title": "string", "price": "number"}
  - List item schema example: [{"title": "string", "price": "number"}] → returns items array
  - "string" → returns best single string in {"content": str}
  - "json" → returns best‑effort JSON object
  - "list" → returns {"items": [str, ...]}
  - "list_of_dict" → returns {"items": [{...}, ...]}
- `timeout` and `max_workers`
  - Per‑run performance tuning; server defaults are in `.env`/`env.example`

---

## Operational settings (server‑level)
Set these in `.env` (documented in `env.example`) so operations can manage deployment defaults:

- Provider selection & credentials (`SERVICE_TO_USE`, provider keys)
- Directories (output/cache/log)
- Logging defaults (level, console/file, performance logs)
- Performance defaults (max workers, timeouts, rate limits)
- Caching policy (enable, TTL, size)
- Proxy settings (server, username/password)
- Security defaults (SSL verify, cert path)
- Monitoring (opt‑in metrics)

Users then override per‑run knobs via CLI/API (`mode`, `schema`, `timeout`, `max_workers`, include flags).

---

## Best practices
- Start with `mode="adaptive"` for unknown sites; narrow strategy if stable
- Always specify `output_format` for predictable JSON shapes
- Use `--timeout` and `--max-workers` to control long pages/large batches
- Configure server‑level rate limits and caching in `.env` for throughput
- Prefer `mode="structured"` for rigid, table‑heavy pages for speed and determinism

---

## Troubleshooting
- Provider errors: confirm `SERVICE_TO_USE` and API keys in `.env`
- Incomplete results on dynamic pages: switch to `mode="adaptive"` and increase `--timeout`
- Schema not filled: simplify schema or use a more specific prompt
- Frame‑only content: already merged; if still missing, use `adaptive` for interactions
- Rate limits/timeouts: adjust server defaults (env) and/or per‑run flags

---

## FAQ
- Does it support non‑login hidden content? Yes. Iframes are merged; Adaptive can scroll/click to reveal content.
- Can I avoid AI reasoning? Yes. Use semantic/structured/rule‑based/hybrid.
- Can I choose a provider? Yes. Set `SERVICE_TO_USE` and corresponding keys in `.env`.

---

## Summary
- Prompt‑first interface with production‑grade internals
- Quick to start with Python/CLI examples
- Clear separation of per‑run parameters (CLI/API) and server defaults (`.env`)
- Multiple modes for different site types; Adaptive adds AI navigation when needed
- Robust handling of hidden content and schema‑driven outputs for consistent results 