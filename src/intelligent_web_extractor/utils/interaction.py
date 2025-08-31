import asyncio
from typing import List, Optional, Dict, Any
from playwright.async_api import Page

DEFAULT_EXPANDER_SELECTORS = [
    "text=Load more",
    "text=Show more",
    "text=Continue reading",
    "text=More",
    "button:has-text('more')",
    "button[aria-label*='more']",
    "a[role='button']:has-text('more')",
]

async def basic_scroll(page: Page, times: int = 5, delay_ms: int = 500) -> None:
    for _ in range(max(0, times)):
        await page.mouse.wheel(0, 2000)
        await page.wait_for_timeout(delay_ms)

async def click_first_expander(page: Page, selectors: Optional[List[str]] = None) -> bool:
    selectors = selectors or DEFAULT_EXPANDER_SELECTORS
    for sel in selectors:
        try:
            el = await page.query_selector(sel)
            if el:
                await el.click()
                await page.wait_for_timeout(800)
                return True
        except Exception:
            continue
    return False

async def snapshot_page_state(page: Page) -> Dict[str, Any]:
    try:
        return await page.evaluate("""
        () => {
          const buttons = Array.from(document.querySelectorAll('button, a[role="button"], a, span'))
            .map(el => (el.innerText || el.textContent || '').trim())
            .filter(t => t && t.length < 120)
            .slice(0, 200);
          const height = document.body?.scrollHeight || 0;
          const linkCount = document.querySelectorAll('a[href]').length;
          const hasInfiniteScrollHints = /infinite|load more|show more|continue/i.test(document.body.innerText || '');
          return { buttons, height, linkCount, hasInfiniteScrollHints };
        }
        """)
    except Exception:
        return {"buttons": [], "height": 0, "linkCount": 0, "hasInfiniteScrollHints": False}

async def ai_navigation_loop(ai_client, page: Page, user_query: Optional[str], max_steps: int = 5) -> None:
    """Run a small AI-guided loop to decide between scroll / click expander / stop."""
    for _ in range(max_steps):
        state = await snapshot_page_state(page)
        prompt = (
            "You are controlling a headless browser to reveal hidden content needed to satisfy a user query.\n"
            "Decide ONE action: 'scroll', 'click_expander', or 'stop'.\n"
            "Return strict JSON: {\"action\": \"scroll|click_expander|stop\", \"reason\": \"...\"}.\n\n"
            f"User query: {user_query or 'General extraction'}\n"
            f"State summary: {state}\n"
        )
        try:
            decision_str = await ai_client._get_ai_response(prompt)
        except Exception:
            # Fallback: simple heuristic
            decision_str = '{"action": "scroll", "reason": "fallback"}'
        action = "scroll"
        if "click" in decision_str:
            action = "click_expander"
        elif "stop" in decision_str:
            action = "stop"
        if action == "scroll":
            await basic_scroll(page, times=2, delay_ms=400)
        elif action == "click_expander":
            clicked = await click_first_expander(page)
            if not clicked:
                # no expander found; perform a scroll instead
                await basic_scroll(page, times=1, delay_ms=400)
        else:
            break
        await page.wait_for_timeout(500) 