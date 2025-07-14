"""
Web Utilities

This module provides utility functions for web-related operations.
"""

import re
import asyncio
from typing import List, Optional, Dict, Any
from urllib.parse import urlparse, urljoin, urlunparse
import logging

logger = logging.getLogger(__name__)


def sanitize_url(url: str) -> str:
    """
    Sanitize and normalize a URL.
    
    Args:
        url: The URL to sanitize
        
    Returns:
        Sanitized URL
    """
    if not url:
        return ""
    
    # Remove whitespace
    url = url.strip()
    
    # Add protocol if missing
    if not url.startswith(('http://', 'https://')):
        url = 'https://' + url
    
    # Parse and reconstruct to normalize
    parsed = urlparse(url)
    return urlunparse((
        parsed.scheme,
        parsed.netloc.lower(),
        parsed.path,
        parsed.params,
        parsed.query,
        parsed.fragment
    ))


def clean_text(text: str) -> str:
    """
    Clean and normalize text content.
    
    Args:
        text: Raw text to clean
        
    Returns:
        Cleaned text
    """
    if not text:
        return ""
    
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text)
    
    # Remove leading/trailing whitespace
    text = text.strip()
    
    return text


def extract_links(html_content: str, base_url: str = "") -> List[str]:
    """
    Extract links from HTML content.
    
    Args:
        html_content: HTML content to parse
        base_url: Base URL for relative links
        
    Returns:
        List of extracted URLs
    """
    try:
        from bs4 import BeautifulSoup
        
        soup = BeautifulSoup(html_content, 'html.parser')
        links = []
        
        for link in soup.find_all('a', href=True):
            href = link['href']
            
            # Skip javascript and mailto links
            if href.startswith(('javascript:', 'mailto:', 'tel:')):
                continue
            
            # Make relative URLs absolute
            if base_url and not href.startswith(('http://', 'https://')):
                href = urljoin(base_url, href)
            
            links.append(href)
        
        return links
        
    except ImportError:
        logger.warning("BeautifulSoup not available, using regex fallback")
        # Fallback to regex
        import re
        pattern = r'href=["\']([^"\']+)["\']'
        matches = re.findall(pattern, html_content)
        return [urljoin(base_url, match) for match in matches]


def parse_html(html_content: str) -> Optional[Any]:
    """
    Parse HTML content using BeautifulSoup.
    
    Args:
        html_content: HTML content to parse
        
    Returns:
        BeautifulSoup object or None
    """
    try:
        from bs4 import BeautifulSoup
        return BeautifulSoup(html_content, 'html.parser')
    except ImportError:
        logger.error("BeautifulSoup not available")
        return None


def load_html(url: str) -> Optional[str]:
    """
    Load HTML content from URL.
    
    Args:
        url: URL to load
        
    Returns:
        HTML content or None
    """
    try:
        import requests
        response = requests.get(url, timeout=30)
        response.raise_for_status()
        return response.text
    except Exception as e:
        logger.error(f"Failed to load HTML from {url}: {e}")
        return None


async def take_screenshot(url: str, output_path: str = "screenshot.png") -> bool:
    """
    Take a screenshot of a webpage.
    
    Args:
        url: URL to screenshot
        output_path: Path to save screenshot
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle')
            await page.screenshot(path=output_path)
            await browser.close()
            return True
            
    except Exception as e:
        logger.error(f"Failed to take screenshot: {e}")
        return False


async def generate_pdf(url: str, output_path: str = "page.pdf") -> bool:
    """
    Generate PDF from webpage.
    
    Args:
        url: URL to convert to PDF
        output_path: Path to save PDF
        
    Returns:
        True if successful, False otherwise
    """
    try:
        from playwright.async_api import async_playwright
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()
            await page.goto(url, wait_until='networkidle')
            await page.pdf(path=output_path)
            await browser.close()
            return True
            
    except Exception as e:
        logger.error(f"Failed to generate PDF: {e}")
        return False


def extract_metadata(html_content: str) -> Dict[str, Any]:
    """
    Extract metadata from HTML content.
    
    Args:
        html_content: HTML content to parse
        
    Returns:
        Dictionary of metadata
    """
    soup = parse_html(html_content)
    if not soup:
        return {}
    
    metadata = {}
    
    # Title
    title_tag = soup.find('title')
    if title_tag:
        metadata['title'] = title_tag.get_text().strip()
    
    # Meta tags
    meta_tags = soup.find_all('meta')
    for meta in meta_tags:
        name = meta.get('name', meta.get('property', ''))
        content = meta.get('content', '')
        if name and content:
            metadata[name] = content
    
    # Open Graph tags
    og_tags = soup.find_all('meta', property=re.compile(r'^og:'))
    for tag in og_tags:
        property_name = tag.get('property', '')
        content = tag.get('content', '')
        if property_name and content:
            metadata[property_name] = content
    
    return metadata


def is_valid_url(url: str) -> bool:
    """
    Check if a URL is valid.
    
    Args:
        url: URL to validate
        
    Returns:
        True if valid, False otherwise
    """
    try:
        result = urlparse(url)
        return all([result.scheme, result.netloc])
    except Exception:
        return False


def normalize_url(url: str) -> str:
    """
    Normalize a URL by removing fragments and query parameters.
    
    Args:
        url: URL to normalize
        
    Returns:
        Normalized URL
    """
    parsed = urlparse(url)
    return urlunparse((
        parsed.scheme,
        parsed.netloc,
        parsed.path,
        '',  # params
        '',  # query
        ''   # fragment
    )) 