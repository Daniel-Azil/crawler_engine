"""
Semantic Extraction Strategy

Implements semantic content extraction using AI to understand and extract
relevant content based on user queries and context.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup
import numpy as np

from ..models.extraction_result import ExtractionResult, ContentMetadata, ExtractionMetrics, StrategyInfo
from ..models.config import ExtractorConfig
from ..utils.ai_client import AIClient


class SemanticExtractionStrategy:
    """
    Semantic content extraction strategy that uses AI to understand
    content context and extract relevant information.
    
    This strategy is best suited for content-heavy pages with natural
    language content like articles, blog posts, and documentation.
    """
    
    def __init__(self, ai_client: AIClient, config: ExtractorConfig):
        """
        Initialize the semantic extraction strategy.
        
        Args:
            ai_client: AI client for content analysis
            config: Configuration settings
        """
        self.ai_client = ai_client
        self.config = config
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self._total_extractions = 0
        self._successful_extractions = 0
        self._average_confidence = 0.0
        self._average_time = 0.0
        
        # Content processing settings
        self._min_chunk_size = 100
        self._max_chunk_size = 2000
        self._chunk_overlap = 200
        
        # HTML cleaning patterns
        self._remove_patterns = [
            r'<script[^>]*>.*?</script>',
            r'<style[^>]*>.*?</style>',
            r'<noscript[^>]*>.*?</noscript>',
            r'<iframe[^>]*>.*?</iframe>',
            r'<object[^>]*>.*?</object>',
            r'<embed[^>]*>.*?</embed>',
            r'<applet[^>]*>.*?</applet>',
        ]
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the semantic extraction strategy"""
        if self._initialized:
            return
        
        self.logger.info("Initializing Semantic Extraction Strategy")
        
        # Compile regex patterns
        self._compiled_patterns = [re.compile(pattern, re.DOTALL | re.IGNORECASE) 
                                 for pattern in self._remove_patterns]
        
        self._initialized = True
        self.logger.info("Semantic Extraction Strategy initialized successfully")
    
    async def close(self):
        """Close the strategy and cleanup resources"""
        self.logger.info("Closing Semantic Extraction Strategy")
        self._initialized = False
    
    async def extract(
        self,
        url: str,
        user_query: Optional[str] = None,
        html_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract semantic content from the page.
        
        Args:
            url: The URL being extracted
            user_query: Optional user query for context
            html_content: Raw HTML content (if not provided, will be fetched)
            
        Returns:
            Dictionary containing extraction results
        """
        if not self._initialized:
            await self.initialize()
        
        start_time = time.time()
        
        try:
            # Clean and parse HTML
            if html_content:
                cleaned_html = self._clean_html(html_content)
                soup = BeautifulSoup(cleaned_html, 'html.parser')
            else:
                # This would typically come from browser manager
                # For now, we'll assume it's provided
                raise ValueError("HTML content must be provided")
            
            # Extract text content
            text_content = self._extract_text_content(soup)
            
            # Chunk content for processing
            chunks = self._create_content_chunks(text_content)
            
            # Analyze chunks with AI
            relevant_chunks = await self._analyze_chunks_with_ai(chunks, user_query)
            
            # Combine relevant content
            extracted_content = self._combine_chunks(relevant_chunks)
            
            # Calculate metrics
            extraction_time = time.time() - start_time
            confidence_score = self._calculate_confidence(relevant_chunks)
            
            # Create metadata
            metadata = self._extract_metadata(soup, url)
            
            # Create metrics
            metrics = ExtractionMetrics(
                extraction_time_ms=extraction_time * 1000,
                confidence_score=confidence_score,
                relevance_score=confidence_score,  # Use confidence as relevance proxy
                completeness_score=min(len(extracted_content) / 1000, 1.0),
                accuracy_score=confidence_score
            )
            
            # Create strategy info
            strategy_info = StrategyInfo(
                strategy_name="SemanticExtractionStrategy",
                strategy_version="1.0.0",
                strategy_parameters={
                    "min_chunk_size": self._min_chunk_size,
                    "max_chunk_size": self._max_chunk_size,
                    "chunk_overlap": self._chunk_overlap,
                    "user_query": user_query
                }
            )
            
            # Update performance tracking
            self._update_performance_tracking(extraction_time, confidence_score)
            
            return {
                "content": extracted_content,
                "raw_html": html_content,
                "metadata": metadata,
                "metrics": metrics,
                "strategy_info": strategy_info,
                "success": True,
                "chunks_analyzed": len(chunks),
                "relevant_chunks": len(relevant_chunks)
            }
            
        except Exception as e:
            self.logger.error(f"Semantic extraction failed for {url}: {str(e)}")
            
            return {
                "content": "",
                "raw_html": html_content,
                "metadata": ContentMetadata(),
                "metrics": ExtractionMetrics(),
                "strategy_info": StrategyInfo(strategy_name="SemanticExtractionStrategy"),
                "success": False,
                "error_message": str(e)
            }
    
    def _clean_html(self, html: str) -> str:
        """Clean HTML content by removing unwanted elements"""
        cleaned_html = html
        
        # Remove unwanted patterns
        for pattern in self._compiled_patterns:
            cleaned_html = pattern.sub('', cleaned_html)
        
        # Remove comments
        cleaned_html = re.sub(r'<!--.*?-->', '', cleaned_html, flags=re.DOTALL)
        
        # Remove extra whitespace
        cleaned_html = re.sub(r'\s+', ' ', cleaned_html)
        
        return cleaned_html.strip()
    
    def _extract_text_content(self, soup: BeautifulSoup) -> str:
        """Extract text content from BeautifulSoup object"""
        # Remove unwanted elements
        for element in soup.find_all(['script', 'style', 'noscript', 'iframe', 'object', 'embed', 'applet']):
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
    
    def _create_content_chunks(self, text: str) -> List[str]:
        """Create content chunks for processing"""
        if len(text) <= self._max_chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + self._max_chunk_size
            
            # Try to break at sentence boundary
            if end < len(text):
                # Look for sentence endings
                sentence_end = text.rfind('.', start, end)
                paragraph_end = text.rfind('\n', start, end)
                
                if sentence_end > start and sentence_end > paragraph_end:
                    end = sentence_end + 1
                elif paragraph_end > start:
                    end = paragraph_end + 1
            
            chunk = text[start:end].strip()
            if len(chunk) >= self._min_chunk_size:
                chunks.append(chunk)
            
            start = end - self._chunk_overlap
            if start >= len(text):
                break
        
        return chunks
    
    async def _analyze_chunks_with_ai(
        self,
        chunks: List[str],
        user_query: Optional[str]
    ) -> List[Dict[str, Any]]:
        """Analyze chunks with AI to find relevant content"""
        if not chunks:
            return []
        
        relevant_chunks = []
        
        for i, chunk in enumerate(chunks):
            try:
                # Analyze chunk relevance
                relevance_score = await self._analyze_chunk_relevance(chunk, user_query)
                
                if relevance_score >= self.config.extraction.relevance_threshold:
                    relevant_chunks.append({
                        "content": chunk,
                        "relevance_score": relevance_score,
                        "chunk_index": i,
                        "length": len(chunk)
                    })
                
            except Exception as e:
                self.logger.warning(f"Failed to analyze chunk {i}: {str(e)}")
                continue
        
        # Sort by relevance score
        relevant_chunks.sort(key=lambda x: x["relevance_score"], reverse=True)
        
        # Limit to maximum chunks
        max_chunks = self.config.extraction.semantic_max_chunks
        if len(relevant_chunks) > max_chunks:
            relevant_chunks = relevant_chunks[:max_chunks]
        
        return relevant_chunks
    
    async def _analyze_chunk_relevance(
        self,
        chunk: str,
        user_query: Optional[str]
    ) -> float:
        """Analyze chunk relevance using AI"""
        try:
            # Create analysis prompt
            prompt = f"""
Analyze the relevance of the following content chunk to the user query.

User Query: {user_query or 'General content extraction'}

Content Chunk:
{chunk[:1000]}...

Rate the relevance from 0.0 to 1.0 where:
- 0.0: Completely irrelevant
- 0.5: Somewhat relevant
- 1.0: Highly relevant

Provide only a number between 0.0 and 1.0 as your response.
"""
            
            # Get AI response
            response = await self.ai_client._get_ai_response(prompt)
            
            # Parse relevance score
            try:
                relevance_score = float(response.strip())
                return max(0.0, min(1.0, relevance_score))
            except ValueError:
                # Fallback to simple keyword matching
                return self._calculate_keyword_relevance(chunk, user_query)
                
        except Exception as e:
            self.logger.warning(f"AI chunk analysis failed: {str(e)}")
            # Fallback to keyword matching
            return self._calculate_keyword_relevance(chunk, user_query)
    
    def _calculate_keyword_relevance(
        self,
        chunk: str,
        user_query: Optional[str]
    ) -> float:
        """Calculate relevance using keyword matching"""
        if not user_query:
            return 0.5  # Default relevance for general extraction
        
        # Extract keywords from query
        query_words = set(re.findall(r'\b\w+\b', user_query.lower()))
        
        # Count keyword matches in chunk
        chunk_words = set(re.findall(r'\b\w+\b', chunk.lower()))
        
        if not query_words:
            return 0.5
        
        # Calculate keyword overlap
        matches = len(query_words.intersection(chunk_words))
        relevance = matches / len(query_words)
        
        return min(1.0, relevance)
    
    def _combine_chunks(self, relevant_chunks: List[Dict[str, Any]]) -> str:
        """Combine relevant chunks into final content"""
        if not relevant_chunks:
            return ""
        
        # Sort by chunk index to maintain order
        sorted_chunks = sorted(relevant_chunks, key=lambda x: x["chunk_index"])
        
        # Combine chunks with separators
        combined_content = []
        for chunk_data in sorted_chunks:
            content = chunk_data["content"]
            combined_content.append(content)
        
        return "\n\n".join(combined_content)
    
    def _calculate_confidence(self, relevant_chunks: List[Dict[str, Any]]) -> float:
        """Calculate confidence score based on relevant chunks"""
        if not relevant_chunks:
            return 0.0
        
        # Calculate average relevance score
        avg_relevance = sum(chunk["relevance_score"] for chunk in relevant_chunks) / len(relevant_chunks)
        
        # Boost confidence if we have multiple high-relevance chunks
        if len(relevant_chunks) >= 3:
            avg_relevance *= 1.1
        
        return min(1.0, avg_relevance)
    
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
                            # Try to parse date
                            from dateutil import parser
                            metadata.publish_date = parser.parse(date_str)
                        except:
                            pass
                    break
            
            # Extract language
            lang_element = soup.find('html')
            if lang_element:
                metadata.language = lang_element.get('lang', '')
            
            # Extract tags/categories
            tag_selectors = [
                'meta[name="keywords"]',
                '[class*="tag"]',
                '[class*="category"]',
                '.tags',
                '.categories'
            ]
            
            for selector in tag_selectors:
                tag_elements = soup.select(selector)
                for element in tag_elements:
                    if element.name == 'meta':
                        content = element.get('content', '')
                        if content:
                            metadata.tags.extend([tag.strip() for tag in content.split(',')])
                    else:
                        text = element.get_text().strip()
                        if text:
                            metadata.tags.append(text)
            
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
        """Perform health check on semantic strategy"""
        return {
            "initialized": self._initialized,
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "average_confidence": self._average_confidence,
            "average_time": self._average_time,
            "ai_client_available": self.ai_client is not None
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "average_confidence": self._average_confidence,
            "average_time": self._average_time,
            "strategy_name": "SemanticExtractionStrategy"
        } 