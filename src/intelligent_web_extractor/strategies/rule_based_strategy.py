"""
Rule-Based Extraction Strategy

Implements rule-based content extraction using predefined patterns
and selectors for consistent content extraction from structured pages.
"""

import asyncio
import time
from typing import Dict, List, Optional, Any, Union
import logging
import re
from datetime import datetime

from bs4 import BeautifulSoup, Tag

from ..models.extraction_result import ExtractionResult, ContentMetadata, ExtractionMetrics, StrategyInfo
from ..models.config import ExtractorConfig


class RuleBasedExtractionStrategy:
    """
    Rule-based content extraction strategy using predefined patterns.
    
    This strategy is best suited for pages with consistent structure
    where content can be reliably extracted using CSS selectors and
    predefined rules.
    """
    
    def __init__(self, config: ExtractorConfig):
        """
        Initialize the rule-based extraction strategy.
        
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
        
        # Rule definitions
        self._content_rules = self._initialize_content_rules()
        self._metadata_rules = self._initialize_metadata_rules()
        self._cleaning_rules = self._initialize_cleaning_rules()
        
        # Rule performance tracking
        self._rule_performance = {}
        
        self._initialized = False
    
    async def initialize(self):
        """Initialize the rule-based extraction strategy"""
        if self._initialized:
            return
        
        self.logger.info("Initializing Rule-Based Extraction Strategy")
        
        # Initialize rule performance tracking
        for rule_name in self._content_rules.keys():
            self._rule_performance[rule_name] = {
                "usage_count": 0,
                "success_count": 0,
                "average_confidence": 0.0
            }
        
        self._initialized = True
        self.logger.info("Rule-Based Extraction Strategy initialized successfully")
    
    async def close(self):
        """Close the strategy and cleanup resources"""
        self.logger.info("Closing Rule-Based Extraction Strategy")
        self._initialized = False
    
    def _initialize_content_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize content extraction rules"""
        return {
            "main_content": {
                "selectors": [
                    "main",
                    "[role='main']",
                    ".main-content",
                    ".content",
                    ".post-content",
                    ".article-content",
                    ".entry-content",
                    "#content",
                    "#main"
                ],
                "priority": 1,
                "confidence_boost": 0.2
            },
            "article_content": {
                "selectors": [
                    "article",
                    ".article",
                    ".post",
                    ".entry",
                    ".story"
                ],
                "priority": 2,
                "confidence_boost": 0.15
            },
            "section_content": {
                "selectors": [
                    "section",
                    ".section",
                    ".block",
                    ".widget"
                ],
                "priority": 3,
                "confidence_boost": 0.1
            },
            "paragraph_content": {
                "selectors": [
                    "p",
                    ".paragraph",
                    ".text"
                ],
                "priority": 4,
                "confidence_boost": 0.05
            },
            "div_content": {
                "selectors": [
                    "div[class*='content']",
                    "div[class*='text']",
                    "div[class*='body']"
                ],
                "priority": 5,
                "confidence_boost": 0.05
            }
        }
    
    def _initialize_metadata_rules(self) -> Dict[str, Dict[str, Any]]:
        """Initialize metadata extraction rules"""
        return {
            "title": {
                "selectors": [
                    "h1",
                    ".title",
                    ".headline",
                    "[class*='title']",
                    "title"
                ],
                "attribute": "text"
            },
            "subtitle": {
                "selectors": [
                    "h2",
                    ".subtitle",
                    ".sub-headline",
                    "[class*='subtitle']"
                ],
                "attribute": "text"
            },
            "author": {
                "selectors": [
                    ".author",
                    "[class*='author']",
                    "[data-author]",
                    "[rel='author']",
                    "meta[name='author']"
                ],
                "attribute": "text"
            },
            "date": {
                "selectors": [
                    ".date",
                    ".published",
                    ".timestamp",
                    "time",
                    "[data-date]",
                    "meta[property='article:published_time']"
                ],
                "attribute": "datetime"
            },
            "category": {
                "selectors": [
                    ".category",
                    ".tag",
                    ".topic",
                    "[class*='category']",
                    "[class*='tag']"
                ],
                "attribute": "text"
            }
        }
    
    def _initialize_cleaning_rules(self) -> Dict[str, List[str]]:
        """Initialize content cleaning rules"""
        selectors = [
            "script",
            "style",
            "noscript",
            # iframe/object/embed/applet are only removed if hidden content handling is disabled
        ]
        if not self.config.extraction.enable_hidden_content_handling:
            selectors.extend(["iframe", "object", "embed", "applet"])
        selectors.extend([
            "nav",
            "footer",
            "header",
            ".advertisement",
            ".ad",
            ".banner",
            ".sponsor",
            ".promo",
            ".sidebar",
            ".navigation",
            ".menu",
        ])
        return {
            "remove_selectors": selectors,
            "remove_classes": [
                "ad",
                "advertisement",
                "banner",
                "sponsor",
                "promo",
                "sidebar",
                "navigation",
                "menu",
                "footer",
                "header",
            ],
            "remove_ids": [
                "ad",
                "advertisement",
                "banner",
                "sponsor",
                "promo",
                "sidebar",
                "navigation",
                "menu",
            ],
        }
    
    async def extract(
        self,
        url: str,
        user_query: Optional[str] = None,
        html_content: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Extract content using rule-based approach.
        
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
            
            # Clean HTML
            cleaned_soup = self._clean_html(soup)
            
            # Extract content using rules
            extracted_content = await self._extract_with_rules(cleaned_soup, user_query)
            
            # Extract metadata
            metadata = self._extract_metadata_with_rules(cleaned_soup)
            
            # Calculate metrics
            extraction_time = time.time() - start_time
            confidence_score = self._calculate_confidence(extracted_content, metadata)
            
            # Create metrics
            metrics = ExtractionMetrics(
                extraction_time_ms=extraction_time * 1000,
                confidence_score=confidence_score,
                relevance_score=confidence_score,
                completeness_score=min(len(extracted_content) / 1000, 1.0),
                accuracy_score=confidence_score
            )
            
            # Create strategy info
            strategy_info = StrategyInfo(
                strategy_name="RuleBasedExtractionStrategy",
                strategy_version="1.0.0",
                strategy_parameters={
                    "rules_used": list(self._rule_performance.keys()),
                    "user_query": user_query,
                    "custom_selectors": self.config.extraction.content_selectors
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
                "rules_applied": len([r for r in self._rule_performance.values() if r["usage_count"] > 0]),
                "content_length": len(extracted_content)
            }
            
        except Exception as e:
            self.logger.error(f"Rule-based extraction failed for {url}: {str(e)}")
            
            return {
                "content": "",
                "raw_html": html_content,
                "metadata": ContentMetadata(),
                "metrics": ExtractionMetrics(),
                "strategy_info": StrategyInfo(strategy_name="RuleBasedExtractionStrategy"),
                "success": False,
                "error_message": str(e)
            }
    
    def _clean_html(self, soup: BeautifulSoup) -> BeautifulSoup:
        """Clean HTML by removing unwanted elements"""
        # Remove elements by selector
        for selector in self._cleaning_rules["remove_selectors"]:
            for element in soup.select(selector):
                element.decompose()
        
        # Remove elements by class
        for class_name in self._cleaning_rules["remove_classes"]:
            for element in soup.find_all(class_=re.compile(class_name, re.IGNORECASE)):
                element.decompose()
        
        # Remove elements by ID
        for id_name in self._cleaning_rules["remove_ids"]:
            for element in soup.find_all(id=re.compile(id_name, re.IGNORECASE)):
                element.decompose()
        
        # Remove comments
        for comment in soup.find_all(string=lambda text: isinstance(text, str) and text.strip().startswith('<!--')):
            comment.extract()
        
        return soup
    
    async def _extract_with_rules(self, soup: BeautifulSoup, user_query: Optional[str]) -> str:
        """Extract content using predefined rules"""
        extracted_parts = []
        applied_rules = []
        
        # Sort rules by priority
        sorted_rules = sorted(
            self._content_rules.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for rule_name, rule_config in sorted_rules:
            try:
                content = self._apply_rule(soup, rule_config)
                if content:
                    extracted_parts.append(content)
                    applied_rules.append(rule_name)
                    
                    # Update rule performance
                    self._rule_performance[rule_name]["usage_count"] += 1
                    self._rule_performance[rule_name]["success_count"] += 1
                    
                    # If we have enough content, stop
                    if len(" ".join(extracted_parts)) > self.config.extraction.min_content_length:
                        break
                        
            except Exception as e:
                self.logger.warning(f"Rule {rule_name} failed: {str(e)}")
                self._rule_performance[rule_name]["usage_count"] += 1
        
        # Apply custom selectors if provided
        if self.config.extraction.content_selectors:
            custom_content = self._apply_custom_selectors(soup)
            if custom_content:
                extracted_parts.append(custom_content)
        
        # Combine extracted content
        combined_content = "\n\n".join(extracted_parts)
        
        # Clean up the combined content
        combined_content = self._clean_text_content(combined_content)
        
        return combined_content
    
    def _apply_rule(self, soup: BeautifulSoup, rule_config: Dict[str, Any]) -> str:
        """Apply a specific rule to extract content"""
        selectors = rule_config["selectors"]
        content_parts = []
        
        for selector in selectors:
            elements = soup.select(selector)
            
            for element in elements:
                # Extract text content
                text_content = element.get_text(separator=' ', strip=True)
                
                if text_content and len(text_content) > 50:  # Minimum content length
                    content_parts.append(text_content)
        
        return "\n\n".join(content_parts)
    
    def _apply_custom_selectors(self, soup: BeautifulSoup) -> str:
        """Apply custom selectors from configuration"""
        content_parts = []
        
        for selector in self.config.extraction.content_selectors:
            try:
                elements = soup.select(selector)
                
                for element in elements:
                    text_content = element.get_text(separator=' ', strip=True)
                    
                    if text_content and len(text_content) > 50:
                        content_parts.append(text_content)
                        
            except Exception as e:
                self.logger.warning(f"Custom selector '{selector}' failed: {str(e)}")
        
        return "\n\n".join(content_parts)
    
    def _clean_text_content(self, text: str) -> str:
        """Clean extracted text content"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove empty lines
        text = re.sub(r'\n\s*\n', '\n', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        return text
    
    def _extract_metadata_with_rules(self, soup: BeautifulSoup) -> ContentMetadata:
        """Extract metadata using predefined rules"""
        metadata = ContentMetadata()
        
        try:
            # Extract title
            title = self._extract_metadata_field(soup, "title")
            if title:
                metadata.title = title
            
            # Extract author
            author = self._extract_metadata_field(soup, "author")
            if author:
                metadata.author = author
            
            # Extract date
            date_str = self._extract_metadata_field(soup, "date")
            if date_str:
                try:
                    from dateutil import parser
                    metadata.publish_date = parser.parse(date_str)
                except:
                    pass
            
            # Extract category
            category = self._extract_metadata_field(soup, "category")
            if category:
                metadata.categories.append(category)
            
            # Extract language
            lang_element = soup.find('html')
            if lang_element:
                metadata.language = lang_element.get('lang', '')
            
            # Calculate content statistics
            text_content = soup.get_text()
            metadata.word_count = len(text_content.split())
            metadata.character_count = len(text_content)
            metadata.reading_time_minutes = metadata.word_count / 200
            
        except Exception as e:
            self.logger.warning(f"Metadata extraction failed: {str(e)}")
        
        return metadata
    
    def _extract_metadata_field(self, soup: BeautifulSoup, field_name: str) -> Optional[str]:
        """Extract a specific metadata field using rules"""
        if field_name not in self._metadata_rules:
            return None
        
        rule_config = self._metadata_rules[field_name]
        selectors = rule_config["selectors"]
        attribute = rule_config.get("attribute", "text")
        
        for selector in selectors:
            try:
                elements = soup.select(selector)
                
                for element in elements:
                    if attribute == "text":
                        value = element.get_text(strip=True)
                    else:
                        value = element.get(attribute, '')
                    
                    if value:
                        return value
                        
            except Exception as e:
                self.logger.warning(f"Metadata field '{field_name}' extraction failed: {str(e)}")
                continue
        
        return None
    
    def _calculate_confidence(self, content: str, metadata: ContentMetadata) -> float:
        """Calculate confidence score based on extracted content"""
        confidence = 0.5  # Base confidence
        
        # Boost confidence based on content length
        if len(content) > 1000:
            confidence += 0.2
        elif len(content) > 500:
            confidence += 0.1
        
        # Boost confidence based on metadata completeness
        metadata_score = 0
        if metadata.title:
            metadata_score += 0.1
        if metadata.author:
            metadata_score += 0.1
        if metadata.publish_date:
            metadata_score += 0.1
        if metadata.language:
            metadata_score += 0.05
        
        confidence += metadata_score
        
        # Boost confidence based on rule success rate
        successful_rules = sum(1 for rule in self._rule_performance.values() if rule["success_count"] > 0)
        total_rules = len(self._rule_performance)
        
        if total_rules > 0:
            rule_success_rate = successful_rules / total_rules
            confidence += rule_success_rate * 0.1
        
        return min(1.0, confidence)
    
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
        
        # Update rule performance averages
        for rule_name, rule_stats in self._rule_performance.items():
            if rule_stats["usage_count"] > 0:
                rule_stats["average_confidence"] = (
                    (rule_stats["average_confidence"] * (rule_stats["usage_count"] - 1) + confidence_score) /
                    rule_stats["usage_count"]
                )
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on rule-based strategy"""
        return {
            "initialized": self._initialized,
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "average_confidence": self._average_confidence,
            "average_time": self._average_time,
            "total_rules": len(self._content_rules),
            "active_rules": len([r for r in self._rule_performance.values() if r["usage_count"] > 0])
        }
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        return {
            "total_extractions": self._total_extractions,
            "successful_extractions": self._successful_extractions,
            "success_rate": self._successful_extractions / max(self._total_extractions, 1),
            "average_confidence": self._average_confidence,
            "average_time": self._average_time,
            "strategy_name": "RuleBasedExtractionStrategy",
            "rule_performance": self._rule_performance
        }
    
    def add_custom_rule(self, rule_name: str, selectors: List[str], priority: int = 10):
        """Add a custom extraction rule"""
        self._content_rules[rule_name] = {
            "selectors": selectors,
            "priority": priority,
            "confidence_boost": 0.05
        }
        
        self._rule_performance[rule_name] = {
            "usage_count": 0,
            "success_count": 0,
            "average_confidence": 0.0
        }
        
        self.logger.info(f"Added custom rule: {rule_name}")
    
    def remove_rule(self, rule_name: str):
        """Remove a custom extraction rule"""
        if rule_name in self._content_rules:
            del self._content_rules[rule_name]
            del self._rule_performance[rule_name]
            self.logger.info(f"Removed rule: {rule_name}")
        else:
            self.logger.warning(f"Rule not found: {rule_name}")
    
    def get_rule_stats(self) -> Dict[str, Any]:
        """Get detailed rule performance statistics"""
        return {
            "total_rules": len(self._content_rules),
            "rule_performance": self._rule_performance,
            "most_used_rule": max(self._rule_performance.items(), key=lambda x: x[1]["usage_count"])[0] if self._rule_performance else None,
            "most_successful_rule": max(self._rule_performance.items(), key=lambda x: x[1]["success_count"])[0] if self._rule_performance else None
        } 