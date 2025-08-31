"""
Structured Extraction Strategy

Implements structured content extraction for data-rich pages with
tables, lists, forms, and other structured elements.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup, Tag
import pandas as pd

from ..models.extraction_result import ExtractionResult, ContentMetadata, ExtractionMetrics, StrategyInfo, StructuredData
from ..models.config import ExtractorConfig


class StructuredExtractionStrategy:
    """
    Structured content extraction strategy for data-rich pages.
    
    This strategy is optimized for pages containing tables, lists,
    forms, and other structured data elements.
    """
    
    def __init__(self, config: ExtractorConfig):
        """
        Initialize the structured extraction strategy.
        
        Args:
            config: Configuration settings
        """
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self._total_extractions = 0
        self._successful_extractions = 0
        self._average_confidence = 0.0
        self._average_time = 0.0
        
        # Structured data extraction settings
        self._table_selectors = [
            'table',
            '[role="table"]',
            '.table',
            '.data-table',
            '[class*="table"]'
        ]
        
        self._list_selectors = [
            'ul', 'ol',
            '[role="list"]',
            '.list',
            '[class*="list"]'
        ]
        
        self._form_selectors = [
            'form',
            '[role="form"]',
            '.form',
            '[class*="form"]'
        ]
        
        self._link_selectors = [
            'a[href]',
            '[role="link"]',
            '.link',
            '[class*="link"]'
        ]
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the structured extraction strategy"""
        if self._initialized:
            return
        
        self.logger.info("Initializing Structured Extraction Strategy")
        self._initialized = True
        self.logger.info("Structured Extraction Strategy initialized successfully")
    
    async def close(self):
        """Close the strategy and cleanup resources"""
        self.logger.info("Closing Structured Extraction Strategy")
        self._initialized = False
    
    async def extract(
        self,
        url: str,
        user_query: Optional[str] = None,
        html_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract structured content from the page.
        
        Args:
            url: The URL being extracted
            user_query: Optional user query for context
            html_content: Raw HTML content
            
        Returns:
            Dictionary containing extraction results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Parse HTML
            soup = BeautifulSoup(html_content, 'html.parser')
            
            # Extract structured data
            structured_data = await self._extract_structured_data(soup)
            
            # Extract text content
            text_content = self._extract_text_content(soup)
            
            # Combine structured and text content
            combined_content = self._combine_content(text_content, structured_data)
            
            # Calculate metrics
            extraction_time = time.time() - start_time
            confidence_score = self._calculate_confidence(structured_data, text_content)
            
            # Create metadata
            metadata = self._extract_metadata(soup, url)
            
            # Create metrics
            metrics = ExtractionMetrics(
                extraction_time_ms=extraction_time * 1000,
                confidence_score=confidence_score,
                relevance_score=confidence_score,
                completeness_score=min(len(combined_content) / 1000, 1.0),
                accuracy_score=confidence_score
            )
            
            # Create strategy info
            strategy_info = StrategyInfo(
                strategy_name="StructuredExtractionStrategy",
                strategy_version="1.0.0",
                strategy_parameters={
                    "extract_tables": self.config.extraction.extract_tables,
                    "extract_lists": self.config.extraction.extract_lists,
                    "extract_forms": self.config.extraction.extract_forms,
                    "extract_links": self.config.extraction.extract_links,
                    "user_query": user_query
                }
            )
            
            # Update performance tracking
            self._update_performance_tracking(extraction_time, confidence_score)
            
            return {
                "content": combined_content,
                "raw_html": html_content,
                "metadata": metadata,
                "structured_data": structured_data,
                "metrics": metrics,
                "strategy_info": strategy_info,
                "success": True,
                "tables_extracted": len(structured_data.tables),
                "lists_extracted": len(structured_data.lists),
                "forms_extracted": len(structured_data.forms),
                "links_extracted": len(structured_data.links)
            }
            
        except Exception as e:
            self.logger.error(f"Structured extraction failed for {url}: {str(e)}")
            
            return {
                "content": "",
                "raw_html": html_content,
                "metadata": ContentMetadata(),
                "structured_data": StructuredData(),
                "metrics": ExtractionMetrics(),
                "strategy_info": StrategyInfo(strategy_name="StructuredExtractionStrategy"),
                "success": False,
                "error_message": str(e)
            }
    
    async def _extract_structured_data(self, soup: BeautifulSoup) -> StructuredData:
        """Extract structured data from the page"""
        structured_data = StructuredData()
        
        try:
            # Extract tables
            if self.config.extraction.extract_tables:
                structured_data.tables = await self._extract_tables(soup)
            
            # Extract lists
            if self.config.extraction.extract_lists:
                structured_data.lists = await self._extract_lists(soup)
            
            # Extract forms
            if self.config.extraction.extract_forms:
                structured_data.forms = await self._extract_forms(soup)
            
            # Extract links
            if self.config.extraction.extract_links:
                structured_data.links = await self._extract_links(soup)
            
            # Extract images
            if self.config.extraction.extract_images:
                structured_data.images = await self._extract_images(soup)
            
            # Extract videos
            structured_data.videos = await self._extract_videos(soup)
            
        except Exception as e:
            self.logger.warning(f"Structured data extraction failed: {str(e)}")
        
        return structured_data
    
    async def _extract_tables(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract tables from the page"""
        tables = []
        
        for selector in self._table_selectors:
            table_elements = soup.select(selector)
            
            for table in table_elements:
                try:
                    table_data = self._parse_table(table)
                    if table_data:
                        tables.append(table_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse table: {str(e)}")
                    continue
        
        return tables
    
    def _parse_table(self, table: Tag) -> Optional[Dict[str, Any]]:
        """Parse a table element into structured data"""
        try:
            # Extract headers
            headers = []
            header_row = table.find('thead')
            if header_row:
                header_cells = header_row.find_all(['th', 'td'])
                headers = [cell.get_text(strip=True) for cell in header_cells]
            
            # Extract rows
            rows = []
            body = table.find('tbody') or table
            for row in body.find_all('tr'):
                cells = row.find_all(['td', 'th'])
                row_data = [cell.get_text(strip=True) for cell in cells]
                if row_data:
                    rows.append(row_data)
            
            # If no headers found, use first row as headers
            if not headers and rows:
                headers = rows[0]
                rows = rows[1:]
            
            # Create table data
            table_data = {
                "headers": headers,
                "rows": rows,
                "row_count": len(rows),
                "column_count": len(headers) if headers else 0,
                "caption": self._get_table_caption(table)
            }
            
            return table_data
            
        except Exception as e:
            self.logger.warning(f"Table parsing failed: {str(e)}")
            return None
    
    def _get_table_caption(self, table: Tag) -> str:
        """Get table caption"""
        caption = table.find('caption')
        if caption:
            return caption.get_text(strip=True)
        return ""
    
    async def _extract_lists(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract lists from the page"""
        lists = []
        
        for selector in self._list_selectors:
            list_elements = soup.select(selector)
            
            for list_elem in list_elements:
                try:
                    list_data = self._parse_list(list_elem)
                    if list_data:
                        lists.append(list_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse list: {str(e)}")
                    continue
        
        return lists
    
    def _parse_list(self, list_elem: Tag) -> Optional[Dict[str, Any]]:
        """Parse a list element into structured data"""
        try:
            list_type = list_elem.name  # ul or ol
            items = []
            
            for item in list_elem.find_all('li', recursive=False):
                item_text = item.get_text(strip=True)
                if item_text:
                    items.append(item_text)
            
            list_data = {
                "type": list_type,
                "items": items,
                "item_count": len(items),
                "id": list_elem.get('id', ''),
                "class": list_elem.get('class', [])
            }
            
            return list_data
            
        except Exception as e:
            self.logger.warning(f"List parsing failed: {str(e)}")
            return None
    
    async def _extract_forms(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract forms from the page"""
        forms = []
        
        for selector in self._form_selectors:
            form_elements = soup.select(selector)
            
            for form in form_elements:
                try:
                    form_data = self._parse_form(form)
                    if form_data:
                        forms.append(form_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse form: {str(e)}")
                    continue
        
        return forms
    
    def _parse_form(self, form: Tag) -> Optional[Dict[str, Any]]:
        """Parse a form element into structured data"""
        try:
            form_data = {
                "action": form.get('action', ''),
                "method": form.get('method', 'get'),
                "id": form.get('id', ''),
                "class": form.get('class', []),
                "fields": []
            }
            
            # Extract form fields
            for field in form.find_all(['input', 'textarea', 'select']):
                field_data = {
                    "type": field.get('type', field.name),
                    "name": field.get('name', ''),
                    "id": field.get('id', ''),
                    "placeholder": field.get('placeholder', ''),
                    "required": field.get('required') is not None,
                    "value": field.get('value', '')
                }
                
                if field.name == 'select':
                    options = []
                    for option in field.find_all('option'):
                        options.append({
                            "value": option.get('value', ''),
                            "text": option.get_text(strip=True),
                            "selected": option.get('selected') is not None
                        })
                    field_data["options"] = options
                
                form_data["fields"].append(field_data)
            
            return form_data
            
        except Exception as e:
            self.logger.warning(f"Form parsing failed: {str(e)}")
            return None
    
    async def _extract_links(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract links from the page"""
        links = []
        
        for selector in self._link_selectors:
            link_elements = soup.select(selector)
            
            for link in link_elements:
                try:
                    link_data = self._parse_link(link)
                    if link_data:
                        links.append(link_data)
                except Exception as e:
                    self.logger.warning(f"Failed to parse link: {str(e)}")
                    continue
        
        return links
    
    def _parse_link(self, link: Tag) -> Optional[Dict[str, Any]]:
        """Parse a link element into structured data"""
        try:
            href = link.get('href', '')
            if not href:
                return None
            
            link_data = {
                "href": href,
                "text": link.get_text(strip=True),
                "title": link.get('title', ''),
                "target": link.get('target', ''),
                "rel": link.get('rel', []),
                "id": link.get('id', ''),
                "class": link.get('class', [])
            }
            
            return link_data
            
        except Exception as e:
            self.logger.warning(f"Link parsing failed: {str(e)}")
            return None
    
    async def _extract_images(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract images from the page"""
        images = []
        
        for img in soup.find_all('img'):
            try:
                image_data = {
                    "src": img.get('src', ''),
                    "alt": img.get('alt', ''),
                    "title": img.get('title', ''),
                    "width": img.get('width', ''),
                    "height": img.get('height', ''),
                    "id": img.get('id', ''),
                    "class": img.get('class', [])
                }
                
                if image_data["src"]:
                    images.append(image_data)
                    
            except Exception as e:
                self.logger.warning(f"Image parsing failed: {str(e)}")
                continue
        
        return images
    
    async def _extract_videos(self, soup: BeautifulSoup) -> List[Dict[str, Any]]:
        """Extract videos from the page"""
        videos = []
        
        # Extract video elements
        for video in soup.find_all(['video', 'iframe']):
            try:
                video_data = {
                    "type": video.name,
                    "src": video.get('src', ''),
                    "title": video.get('title', ''),
                    "width": video.get('width', ''),
                    "height": video.get('height', ''),
                    "id": video.get('id', ''),
                    "class": video.get('class', [])
                }
                
                if video_data["src"]:
                    videos.append(video_data)
                    
            except Exception as e:
                self.logger.warning(f"Video parsing failed: {str(e)}")
                continue
        
        return videos
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract text content from the page"""
        # Remove unwanted elements
        removable = ['script', 'style', 'noscript']
        if not self.config.extraction.enable_hidden_content_handling:
            removable += ['iframe', 'object', 'embed', 'applet']
        for element in soup.find_all(removable):
            element.decompose()
        
        # Remove navigation and footer elements
        for element in soup.find_all(['nav', 'footer', 'header']):
            if self.config.extraction.remove_navigation:
                element.decompose()
        
        # Remove ad-related elements
        if self.config.extraction.remove_ads:
            ad_selectors = [
                '[class*="ad"]', '[class*="advertisement"]', '[id*="ad"]',
                '[class*="banner"]', '[class*="sponsor"]', '[class*="promo"]'
            ]
            for selector in ad_selectors:
                for element in soup.select(selector):
                    element.decompose()
        
        # Extract text
        text_content = soup.get_text(separator=' ', strip=True)
        
        # Clean up text
        text_content = re.sub(r'\s+', ' ', text_content)
        text_content = re.sub(r'\n\s*\n', '\n', text_content)
        
        return text_content.strip()
    
    def _combine_content(self, text_content: str, structured_data: StructuredData) -> str:
        """Combine text and structured content"""
        content_parts = []
        
        # Add text content
        if text_content:
            content_parts.append(text_content)
        
        # Add structured data summaries
        if structured_data.tables:
            content_parts.append(f"\n\nTables found: {len(structured_data.tables)}")
            for i, table in enumerate(structured_data.tables[:3]):  # Limit to first 3 tables
                if table.get("caption"):
                    content_parts.append(f"Table {i+1}: {table['caption']}")
        
        if structured_data.lists:
            content_parts.append(f"\nLists found: {len(structured_data.lists)}")
        
        if structured_data.forms:
            content_parts.append(f"\nForms found: {len(structured_data.forms)}")
        
        if structured_data.links:
            content_parts.append(f"\nLinks found: {len(structured_data.links)}")
        
        return "\n".join(content_parts)
    
    def _calculate_confidence(self, structured_data: StructuredData, text_content: str) -> float:
        """Calculate confidence score based on structured data"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on structured data found
        if structured_data.tables:
            confidence += 0.2
        if structured_data.lists:
            confidence += 0.1
        if structured_data.forms:
            confidence += 0.1
        if structured_data.links:
            confidence += 0.05
        
        # Boost confidence based on text content length
        if len(text_content) > 1000:
            confidence += 0.1
        elif len(text_content) > 500:
            confidence += 0.05
        
        return min(1.0, confidence)
    
    def _extract_metadata(self, soup: BeautifulSoup, url: str) -> ContentMetadata:
        """Extract metadata from the page"""
        metadata = ContentMetadata()
        
        try:
            # Extract title
            title_tag = soup.find('title')
            if title_tag:
                metadata.title = title_tag.get_text().strip()
            
            # Extract meta description
            desc_tag = soup.find('meta', attrs={'name': 'description'})
            if desc_tag:
                metadata.content_type = desc_tag.get('content', '').strip()
            
            # Extract author
            author_selectors = [
                'meta[name="author"]',
                '[class*="author"]',
                '[data-author]',
                '.author',
                '[rel="author"]'
            ]
            
            for selector in author_selectors:
                author_element = soup.select_one(selector)
                if author_element:
                    if author_element.name == 'meta':
                        metadata.author = author_element.get('content', '').strip()
                    else:
                        metadata.author = author_element.get_text().strip()
                    break
            
            # Extract publish date
            date_selectors = [
                'meta[property="article:published_time"]',
                'meta[name="publish_date"]',
                'time[datetime]',
                '[data-date]',
                '.date',
                '.published'
            ]
            
            for selector in date_selectors:
                date_element = soup.select_one(selector)
                if date_element:
                    date_str = date_element.get('datetime') or date_element.get('content') or date_element.get_text()
                    if date_str:
                        try:
                            from dateutil import parser
                            metadata.publish_date = parser.parse(date_str)
                        except:
                            pass
                    break
            
            # Extract language
            lang_element = soup.find('html')
            if lang_element:
                metadata.language = lang_element.get('lang', '')
            
            # Calculate content statistics
            metadata.word_count = len(text_content.split()) if 'text_content' in locals() else 0
            metadata.character_count = len(text_content) if 'text_content' in locals() else 0
            metadata.reading_time_minutes = metadata.word_count / 200  # Average reading speed
            
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed: {str(e)}")
        
        return metadata
    
    def _update_performance_tracking(self, extraction_time: float, confidence_score: float):
        """Update performance tracking metrics"""
        self._total_extractions += 1
        
        if confidence_score >= self.config.extraction.confidence_threshold:
            self._successful_extractions += 1
        
        # Update averages
        self._average_confidence = (
            (self._average_confidence * (self._total_extractions - 1) + confidence_score) / 
            self._total_extractions
        )
        
        self._average_time = (
            (self._average_time * (self._total_extractions - 1) + extraction_time) / 
            self._total_extractions
        )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on structured strategy"""
        return {
            "initialized": self._initialized,
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "average_confidence": self._average_confidence,
            "average_time": self._average_time
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "average_confidence": self._average_confidence,
            "average_time": self._average_time,
            "strategy_name": "StructuredExtractionStrategy"
        } 