"""
Browser Manager

Manages browser instances for web content extraction with intelligent
resource management and performance optimization.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
import logging
from contextlib import asynccontextmanager

from playwright.async_api import async_playwright, Browser, BrowserContext, Page
from playwright.async_api import TimeoutError as PlaywrightTimeoutError

from ..models.config import BrowserConfig, BrowserType


class BrowserManager:
    """
    Manages browser instances for intelligent web content extraction.
    
    Provides browser lifecycle management, page handling, and performance
    optimization for content extraction operations.
    """
    
    def __init__(self, config: BrowserConfig):
        """
        Initialize the browser manager.
        
        Args:
            config: Browser configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Browser state
        self._playwright = None
        self._browser = None
        self._contexts: Dict[str, BrowserContext] = {}
        self._pages: Dict[str, Page] = {}
        
        # Performance tracking
        self._page_count = 0
        self._total_requests = 0
        self._failed_requests = 0
        self._start_time = None
        
        # Resource management
        self._semaphore = asyncio.Semaphore(config.max_concurrent_pages)
        self._initialized = False
        self._closed = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the browser manager"""
        if self._initialized:
            return
        
        self.logger.info("Initializing Browser Manager")
        self._start_time = time.time()
        
        try:
            # Launch playwright
            self._playwright = await async_playwright().start()
            
            # Launch browser
            browser_options = {
                "headless": self.config.headless,
                "args": self.config.browser_args,
            }
            
            if self.config.browser_type == BrowserType.CHROMIUM:
                self._browser = await self._playwright.chromium.launch(**browser_options)
            elif self.config.browser_type == BrowserType.FIREFOX:
                self._browser = await self._playwright.firefox.launch(**browser_options)
            elif self.config.browser_type == BrowserType.WEBKIT:
                self._browser = await self._playwright.webkit.launch(**browser_options)
            else:
                raise ValueError(f"Unsupported browser type: {self.config.browser_type}")
            
            self._initialized = True
            self.logger.info(f"Browser Manager initialized with {self.config.browser_type.value}")
            
        except Exception as e:
            self.logger.error(f"Failed to initialize Browser Manager: {str(e)}")
            raise
    
    async def close(self):
        """Close the browser manager and cleanup resources"""
        if self._closed:
            return
        
        self.logger.info("Closing Browser Manager")
        
        try:
            # Close all pages
            for page_id, page in self._pages.items():
                try:
                    await page.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close page {page_id}: {str(e)}")
            
            # Close all contexts
            for context_id, context in self._contexts.items():
                try:
                    await context.close()
                except Exception as e:
                    self.logger.warning(f"Failed to close context {context_id}: {str(e)}")
            
            # Close browser
            if self._browser:
                await self._browser.close()
            
            # Stop playwright
            if self._playwright:
                await self._playwright.stop()
            
            self._closed = True
            self.logger.info("Browser Manager closed successfully")
            
        except Exception as e:
            self.logger.error(f"Error closing Browser Manager: {str(e)}")
            raise
    
    @asynccontextmanager
    async def get_page(self, url: str, context_id: Optional[str] = None):
        """
        Get a browser page for content extraction.
        
        Args:
            url: The URL to navigate to
            context_id: Optional context identifier for session management
            
        Yields:
            Page object ready for content extraction
        """
        if not self._initialized:
            await self.initialize()
        
        page_id = f"{context_id or 'default'}_{int(time.time() * 1000)}"
        
        async with self._semaphore:
            try:
                # Create or get context
                if context_id and context_id in self._contexts:
                    context = self._contexts[context_id]
                else:
                    context = await self._create_context(context_id)
                    if context_id:
                        self._contexts[context_id] = context
                
                # Create page
                page = await context.new_page()
                self._pages[page_id] = page
                self._page_count += 1
                
                # Configure page
                await self._configure_page(page)
                
                # Navigate to URL
                await self._navigate_to_url(page, url)
                
                self.logger.debug(f"Page {page_id} ready for {url}")
                
                try:
                    yield page
                finally:
                    # Cleanup page
                    try:
                        await page.close()
                        del self._pages[page_id]
                    except Exception as e:
                        self.logger.warning(f"Failed to close page {page_id}: {str(e)}")
                        
            except Exception as e:
                self.logger.error(f"Failed to get page for {url}: {str(e)}")
                self._failed_requests += 1
                raise
    
    async def get_page_content(self, url: str, context_id: Optional[str] = None) -> Dict[str, Any]:
        """
        Get page content with metadata.
        
        Args:
            url: The URL to extract content from
            context_id: Optional context identifier
            
        Returns:
            Dictionary containing page content and metadata
        """
        async with self.get_page(url, context_id) as page:
            try:
                # Wait for page to load
                await page.wait_for_load_state(self.config.wait_for_load_state, timeout=self.config.timeout)
                
                # Get page content
                content = await page.content()
                
                # Get page metadata
                metadata = await self._get_page_metadata(page)
                
                # Get page title
                title = await page.title()
                
                # Get page URL (in case of redirects)
                current_url = page.url
                
                self._total_requests += 1
                
                return {
                    "content": content,
                    "title": title,
                    "url": current_url,
                    "metadata": metadata,
                    "success": True
                }
                
            except PlaywrightTimeoutError:
                self.logger.warning(f"Timeout while loading {url}")
                return {
                    "content": "",
                    "title": "",
                    "url": url,
                    "metadata": {},
                    "success": False,
                    "error": "Page load timeout"
                }
            except Exception as e:
                self.logger.error(f"Failed to get content from {url}: {str(e)}")
                self._failed_requests += 1
                return {
                    "content": "",
                    "title": "",
                    "url": url,
                    "metadata": {},
                    "success": False,
                    "error": str(e)
                }
    
    async def take_screenshot(self, url: str, path: str, context_id: Optional[str] = None) -> bool:
        """
        Take a screenshot of the page.
        
        Args:
            url: The URL to screenshot
            path: Path to save the screenshot
            context_id: Optional context identifier
            
        Returns:
            True if screenshot was taken successfully
        """
        async with self.get_page(url, context_id) as page:
            try:
                # Wait for page to load
                await page.wait_for_load_state(self.config.wait_for_load_state, timeout=self.config.timeout)
                
                # Take screenshot
                await page.screenshot(path=path, full_page=True)
                
                self.logger.debug(f"Screenshot saved to {path}")
                return True
                
            except Exception as e:
                self.logger.error(f"Failed to take screenshot of {url}: {str(e)}")
                return False
    
    async def _create_context(self, context_id: Optional[str] = None) -> BrowserContext:
        """Create a new browser context"""
        context_options = {
            "viewport": {
                "width": self.config.viewport_width,
                "height": self.config.viewport_height
            },
            "ignore_https_errors": self.config.ignore_https_errors,
            "bypass_csp": self.config.bypass_csp,
        }
        
        # Add proxy if configured
        if self.config.proxy_server:
            context_options["proxy"] = {
                "server": self.config.proxy_server,
                "username": self.config.proxy_username,
                "password": self.config.proxy_password,
            }
        
        return await self._browser.new_context(**context_options)
    
    async def _configure_page(self, page: Page):
        """Configure page settings"""
        # Set user agent
        if self.config.user_agent:
            await page.set_extra_http_headers({"User-Agent": self.config.user_agent})
        
        # Set viewport
        await page.set_viewport_size({
            "width": self.config.viewport_width,
            "height": self.config.viewport_height
        })
        
        # Add event listeners for performance tracking
        page.on("request", self._on_request)
        page.on("response", self._on_response)
        page.on("pageerror", self._on_page_error)
    
    async def _navigate_to_url(self, page: Page, url: str):
        """Navigate to URL with error handling"""
        try:
            await page.goto(url, timeout=self.config.timeout)
        except PlaywrightTimeoutError:
            self.logger.warning(f"Navigation timeout for {url}")
            raise
        except Exception as e:
            self.logger.error(f"Navigation failed for {url}: {str(e)}")
            raise
    
    async def _get_page_metadata(self, page: Page) -> Dict[str, Any]:
        """Extract page metadata"""
        try:
            metadata = {}
            
            # Get meta tags
            meta_tags = await page.eval_on_selector_all("meta", """
                (elements) => {
                    const meta = {};
                    elements.forEach(el => {
                        const name = el.getAttribute('name') || el.getAttribute('property');
                        const content = el.getAttribute('content');
                        if (name && content) {
                            meta[name] = content;
                        }
                    });
                    return meta;
                }
            """)
            metadata["meta_tags"] = meta_tags
            
            # Get structured data
            structured_data = await page.eval_on_selector_all("script[type='application/ld+json']", """
                (elements) => {
                    const data = [];
                    elements.forEach(el => {
                        try {
                            data.push(JSON.parse(el.textContent));
                        } catch (e) {
                            // Ignore invalid JSON
                        }
                    });
                    return data;
                }
            """)
            metadata["structured_data"] = structured_data
            
            # Get page statistics
            page_stats = await page.evaluate("""
                () => {
                    return {
                        title: document.title,
                        description: document.querySelector('meta[name="description"]')?.content || '',
                        keywords: document.querySelector('meta[name="keywords"]')?.content || '',
                        language: document.documentElement.lang || '',
                        characterCount: document.body?.textContent?.length || 0,
                        wordCount: document.body?.textContent?.split(/\s+/).length || 0,
                        linkCount: document.querySelectorAll('a').length,
                        imageCount: document.querySelectorAll('img').length,
                        scriptCount: document.querySelectorAll('script').length,
                        styleCount: document.querySelectorAll('style, link[rel="stylesheet"]').length
                    };
                }
            """)
            metadata["page_stats"] = page_stats
            
            return metadata
            
        except Exception as e:
            self.logger.warning(f"Failed to extract page metadata: {str(e)}")
            return {}
    
    def _on_request(self, request):
        """Handle page request events"""
        self.logger.debug(f"Request: {request.method} {request.url}")
    
    def _on_response(self, response):
        """Handle page response events"""
        self.logger.debug(f"Response: {response.status} {response.url}")
    
    def _on_page_error(self, error):
        """Handle page error events"""
        self.logger.warning(f"Page error: {error}")
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on browser manager"""
        health_status = {
            "initialized": self._initialized,
            "closed": self._closed,
            "browser_type": self.config.browser_type.value,
            "active_pages": len(self._pages),
            "active_contexts": len(self._contexts),
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": 0.0,
            "uptime_seconds": 0.0
        }
        
        if self._start_time:
            health_status["uptime_seconds"] = time.time() - self._start_time
        
        if self._total_requests > 0:
            health_status["success_rate"] = (self._total_requests - self._failed_requests) / self._total_requests
        
        # Test browser functionality
        if self._initialized and not self._closed:
            try:
                # Try to create a test context
                test_context = await self._create_context("health_check")
                test_page = await test_context.new_page()
                await test_page.close()
                await test_context.close()
                health_status["browser_functional"] = True
            except Exception as e:
                self.logger.warning(f"Browser health check failed: {str(e)}")
                health_status["browser_functional"] = False
        else:
            health_status["browser_functional"] = False
        
        return health_status
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_requests": self._total_requests,
            "failed_requests": self._failed_requests,
            "success_rate": (self._total_requests - self._failed_requests) / max(self._total_requests, 1),
            "active_pages": len(self._pages),
            "active_contexts": len(self._contexts),
            "uptime_seconds": time.time() - self._start_time if self._start_time else 0,
            "average_requests_per_minute": self._total_requests / max((time.time() - self._start_time) / 60, 1) if self._start_time else 0
        }
    
    def reset_stats(self):
        """Reset performance statistics"""
        self._total_requests = 0
        self._failed_requests = 0
        self._page_count = 0
        self._start_time = time.time()
        self.logger.info("Browser Manager statistics reset") 