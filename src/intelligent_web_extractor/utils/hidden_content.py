import asyncio
from typing import Optional
from playwright.async_api import Page
from .errors import IframeExtractionError

async def merge_iframe_content(page: Page) -> str:
    """Return current page.content() plus concatenated iframe HTML, gathered in parallel.
    Does not throw on individual iframe failures; aggregates best-effort.
    """
    try:
        base_html = await page.content()
    except Exception as e:
        base_html = ""

    try:
        frames = [f for f in page.frames if f != page.main_frame]
        async def _get(f):
            try:
                return await f.content()
            except Exception:
                return ""
        iframe_html_list = await asyncio.gather(*[_get(f) for f in frames], return_exceptions=False)
        iframe_html = "\n".join([h for h in iframe_html_list if h])
        if iframe_html:
            return base_html + "\n<!--iframe-content-->\n" + iframe_html
        return base_html
    except Exception as e:
        raise IframeExtractionError(str(e)) 