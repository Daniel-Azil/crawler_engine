"""
Adaptive Extraction Strategy

A true adaptive strategy that uses AI reasoning to analyze page structure,
interact with elements, and dynamically adapt browser settings to extract
data from even the most stubborn websites.
"""

import asyncio
import json
import time
from typing import Dict, List, Optional, Any, Union
from datetime import datetime

from playwright.async_api import Page, TimeoutError as PlaywrightTimeoutError

from .base_strategy import BaseExtractionStrategy
from ..models.extraction_result import ExtractionMetrics, ContentMetadata, StructuredData
from ..models.config import ExtractorConfig
from ..utils.ai_client import AIClient
from ..utils.browser_manager import BrowserManager
from ..utils.logger import ExtractorLogger


class AdaptiveExtractionStrategy(BaseExtractionStrategy):
    """
    Advanced adaptive strategy that uses AI reasoning to overcome
    stubborn websites through intelligent interaction and browser adaptation.
    """
    
    def __init__(self, ai_client: AIClient, config: ExtractorConfig, browser_manager: BrowserManager, logging_config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.ai_client = ai_client
        self.browser_manager = browser_manager
        
        # Setup logging with config
        if logging_config:
            self.logger = ExtractorLogger(__name__, logging_config)
        else:
            self.logger = ExtractorLogger(__name__)
        
        # Adaptive state
        self.max_reasoning_steps = 20  # Increased for thorough extraction
        self.max_interaction_attempts = 15  # More attempts for stubborn sites
        self.max_scroll_attempts = 10  # Aggressive scrolling
        self.min_content_threshold = 2000  # Require substantial content
        self.reasoning_chain = []
        
    async def initialize(self) -> bool:
        """
        Initialize the adaptive extraction strategy.
        
        Returns:
            True if initialization was successful, False otherwise
        """
        try:
            # Initialize AI client if not already done
            if not self.ai_client._initialized:
                await self.ai_client.initialize()
            
            # Initialize browser manager if not already done
            if not self.browser_manager._initialized:
                await self.browser_manager.initialize()
                
            self._initialized = True
            self.logger.info("Adaptive extraction strategy initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize adaptive strategy: {str(e)}")
            return False
    
    async def close(self) -> bool:
        """
        Clean up resources used by the adaptive strategy.
        
        Returns:
            True if cleanup was successful, False otherwise
        """
        try:
            # Reset reasoning chain
            self.reasoning_chain = []
            
            self.logger.info("Adaptive extraction strategy closed successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to close adaptive strategy: {str(e)}")
            return False
        
    async def extract(self, url: str, user_query: Optional[str] = None, html_content: Optional[str] = None) -> Dict[str, Any]:
        """
        Extract content using adaptive AI reasoning and dynamic interaction.
        
        Args:
            url: The URL to extract from
            user_query: User's extraction query
            html_content: Pre-loaded HTML content (optional)
            
        Returns:
            Dictionary with extracted content and metadata
        """
        start_time = time.time()
        self.logger.info(f"Starting adaptive extraction for: {url}")
        
        try:
            # Initialize reasoning chain
            self.reasoning_chain = []
            
            # Phase 1: Initial Analysis and Planning
            async with self.browser_manager.get_page(url) as page:
                # Wait for initial load - use config timeout or default
                browser_timeout = getattr(self.browser_manager.config, 'timeout', 30000)
                await page.wait_for_load_state('networkidle', timeout=browser_timeout)
                
                # Brief wait to ensure dynamic content loads
                dynamic_wait = min(3000, browser_timeout // 10)  # 3s max, or 10% of browser timeout
                await page.wait_for_timeout(dynamic_wait)
                
                # Perform multi-phase adaptive extraction
                extraction_result = await self._perform_adaptive_extraction(page, url, user_query)
                
                # Calculate metrics
                metrics = ExtractionMetrics(
                    confidence_score=extraction_result.get("confidence", 0.8),
                    relevance_score=extraction_result.get("relevance", 0.7),
                    completeness_score=extraction_result.get("completeness", 0.8),
                    extraction_time_ms=(time.time() - start_time) * 1000
                )
                
                return {
                    "content": extraction_result.get("selected_content", extraction_result.get("content", "")),
                    "metadata": extraction_result.get("metadata", {}),
                    "structured_data": extraction_result.get("structured_data", {}),
                    "metrics": metrics,
                    "success": True,
                    "reasoning_chain": self.reasoning_chain
                }
                
        except Exception as e:
            self.logger.error(f"Adaptive extraction failed: {str(e)}")
            return {
                "content": "",
                "metadata": {},
                "structured_data": {},
                "metrics": ExtractionMetrics(confidence_score=0.0),
                "success": False,
                "error_message": str(e),
                "reasoning_chain": self.reasoning_chain
            }
    
    async def _perform_adaptive_extraction(self, page: Page, url: str, user_query: Optional[str]) -> Dict[str, Any]:
        """
        Perform TRUE ADAPTIVE EXTRACTION with ML/AI content analysis.
        This is the new system that sees ALL content and uses AI reasoning.
        """
        self._add_reasoning_step("adaptive_start", "Starting TRUE ADAPTIVE EXTRACTION with ML/AI analysis")
        
        # STEP 1: GET ALL CONTENT - NO LIMITS, NO RESTRICTIONS
        self._add_reasoning_step("phase_1", "PHASE 1: Extracting ALL content from page")
        full_page_content = await self._extract_absolutely_everything(page)
        self._add_reasoning_step("full_extraction", f"Extracted {len(full_page_content)} chars of COMPLETE page content")
        
        # STEP 2: AI ANALYZES USER INTENT AND CONTENT RELEVANCE
        self._add_reasoning_step("phase_2", "PHASE 2: AI analyzing user intent and content relevance")
        user_intent_analysis = await self._analyze_user_intent_and_content_relevance(user_query, full_page_content)
        self._add_reasoning_step("intent_analysis", f"User Intent: {user_intent_analysis.get('intent_type', 'unknown')}")
        
        # STEP 3: ITERATIVE AI PROCESSING BASED ON USER NEEDS
        self._add_reasoning_step("phase_3", "PHASE 3: Iterative AI processing based on user needs")
        final_result = await self._iterative_ai_processing(user_query, full_page_content, user_intent_analysis)
        
        self._add_reasoning_step("adaptive_complete", f"TRUE ADAPTIVE EXTRACTION COMPLETE: {len(str(final_result.get('selected_content', '')))} chars")
        return final_result
    
    async def _analyze_page_structure(self, page: Page, user_query: Optional[str]) -> Dict[str, Any]:
        """
        Use AI to deeply analyze the page structure and identify extraction challenges.
        """
        # Get comprehensive page information
        page_info = await page.evaluate("""
            () => {
                const getElementInfo = (el) => ({
                    tag: el.tagName.toLowerCase(),
                    classes: Array.from(el.classList),
                    id: el.id,
                    text: (el.textContent || '').trim().slice(0, 200),
                    visible: window.getComputedStyle(el).display !== 'none',
                    interactive: el.tagName.toLowerCase() in ['button', 'a', 'input', 'select', 'textarea']
                });
                
                return {
                    title: document.title,
                    url: window.location.href,
                    bodyText: (document.body.textContent || '').slice(0, 5000),
                    
                    // Structural elements
                    headers: Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6')).map(getElementInfo).slice(0, 20),
                    articles: Array.from(document.querySelectorAll('article,main,.content,.post,.article')).map(getElementInfo).slice(0, 10),
                    sections: Array.from(document.querySelectorAll('section,.section')).map(getElementInfo).slice(0, 15),
                    
                    // Interactive elements
                    buttons: Array.from(document.querySelectorAll('button,input[type="button"],.btn,.button')).map(getElementInfo).slice(0, 20),
                    links: Array.from(document.querySelectorAll('a[href]')).map(getElementInfo).slice(0, 30),
                    forms: Array.from(document.querySelectorAll('form')).map(getElementInfo).slice(0, 5),
                    
                    // Data containers
                    tables: Array.from(document.querySelectorAll('table')).map(getElementInfo).slice(0, 10),
                    lists: Array.from(document.querySelectorAll('ul,ol,.list')).map(getElementInfo).slice(0, 15),
                    
                    // Hidden/dynamic content indicators
                    hiddenElements: Array.from(document.querySelectorAll('[style*="display: none"],[style*="visibility: hidden"],.hidden')).length,
                    iframes: Array.from(document.querySelectorAll('iframe')).map(getElementInfo).slice(0, 5),
                    
                    // Navigation and interaction hints
                    loadMoreButtons: Array.from(document.querySelectorAll('*')).filter(el => 
                        /load more|show more|view more|see more|continue|next|expand/i.test(el.textContent || '')
                    ).map(getElementInfo).slice(0, 10),
                    
                    // Technical indicators
                    hasInfiniteScroll: document.body.innerHTML.includes('infinite') || 
                                      document.body.innerHTML.includes('lazy') ||
                                      Array.from(document.scripts).some(s => /infinite|lazy|scroll/i.test(s.textContent || '')),
                    
                    // Meta information
                    metaTags: Array.from(document.querySelectorAll('meta')).map(m => ({
                        name: m.getAttribute('name') || m.getAttribute('property'),
                        content: m.getAttribute('content')
                    })).filter(m => m.name && m.content).slice(0, 20),
                    
                    // Performance indicators
                    scriptCount: document.querySelectorAll('script').length,
                    imageCount: document.querySelectorAll('img').length,
                    totalElements: document.querySelectorAll('*').length
                };
            }
        """)
        
        # Use AI to analyze the structure and plan extraction strategy
        analysis_prompt = f"""
        You are an expert web scraper analyzing a page to extract: "{user_query or 'general content'}"
        
        Page Information:
        {json.dumps(page_info, indent=2)[:4000]}
        
        IMPORTANT: Respond with ONLY valid JSON. Do not include any text before or after the JSON.
        
        Analyze this page and provide a comprehensive extraction strategy in JSON format:
        {{
            "content_location": {{
                "primary_selectors": ["list of CSS selectors where target content likely exists"],
                "fallback_selectors": ["backup selectors if primary fails"],
                "content_type": "article|product|listing|table|form|mixed",
                "extraction_difficulty": "easy|medium|hard|extreme"
            }},
            "interaction_requirements": {{
                "needs_interaction": true/false,
                "interaction_types": ["scroll", "click_buttons", "expand_sections", "wait_for_load", "handle_popups"],
                "specific_elements": ["CSS selectors of elements to interact with"],
                "interaction_sequence": ["ordered list of interaction steps"]
            }},
            "technical_challenges": {{
                "has_dynamic_content": true/false,
                "requires_javascript": true/false,
                "has_anti_scraping": true/false,
                "needs_authentication": true/false,
                "has_rate_limiting": true/false,
                "content_behind_paywall": true/false
            }},
            "browser_optimizations": {{
                "recommended_wait_time": 1000,
                "needs_custom_headers": true/false,
                "requires_mobile_viewport": true/false,
                "disable_images": true/false,
                "bypass_cloudflare": true/false
            }},
            "extraction_confidence": 0.0-1.0,
            "reasoning": "Detailed explanation of analysis and strategy"
        }}"""
        
        try:
            analysis_response = await self.ai_client._get_ai_response(analysis_prompt)
            
            # Parse AI response
            json_start = analysis_response.find('{')
            json_end = analysis_response.rfind('}') + 1
            
            if json_start != -1 and json_end > json_start:
                analysis = json.loads(analysis_response[json_start:json_end])
                analysis['page_info'] = page_info
                return analysis
            else:
                raise ValueError("No valid JSON in AI response")
                
        except Exception as e:
            self.logger.warning(f"AI analysis failed: {str(e)}")
            # Fallback analysis based on heuristics
            return self._fallback_page_analysis(page_info, user_query)
    
    async def _optimize_browser_settings(self, page: Page, analysis: Dict[str, Any]) -> None:
        """
        Dynamically optimize browser settings based on AI analysis.
        """
        optimizations = analysis.get("browser_optimizations", {})
        
        try:
            # Apply recommended optimizations
            if optimizations.get("requires_mobile_viewport"):
                await page.set_viewport_size({"width": 375, "height": 812})
                await page.set_user_agent("Mozilla/5.0 (iPhone; CPU iPhone OS 14_0 like Mac OS X) AppleWebKit/605.1.15")
                self._add_reasoning_step("browser_optimization", "Switched to mobile viewport")
            
            if optimizations.get("needs_custom_headers"):
                await page.set_extra_http_headers({
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                    "Accept-Language": "en-US,en;q=0.5",
                    "Accept-Encoding": "gzip, deflate",
                    "Cache-Control": "no-cache",
                    "Pragma": "no-cache"
                })
                self._add_reasoning_step("browser_optimization", "Added custom headers")
            
            if optimizations.get("disable_images"):
                await page.route("**/*.{png,jpg,jpeg,gif,webp,svg}", lambda route: route.abort())
                self._add_reasoning_step("browser_optimization", "Disabled image loading for performance")
            
            # Wait for recommended time
            wait_time = optimizations.get("recommended_wait_time", 1000)
            await page.wait_for_timeout(wait_time)
            
        except Exception as e:
            self.logger.warning(f"Browser optimization failed: {str(e)}")
    
    async def _discover_content_intelligently(self, page: Page, user_query: Optional[str], analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Use AI-guided exploration to discover all relevant content on the page.
        """
        content_location = analysis.get("content_location", {})
        interaction_requirements = analysis.get("interaction_requirements", {})
        
        content_map = {
            "discovered_content": [],
            "interaction_history": [],
            "extraction_attempts": []
        }
        
        # Try primary selectors first
        primary_selectors = content_location.get("primary_selectors", [])
        for selector in primary_selectors:
            try:
                elements = await page.query_selector_all(selector)
                if elements:
                    for i, element in enumerate(elements[:10]):  # Limit to first 10
                        try:
                            content = await element.text_content()
                            if content and len(content.strip()) > 50:  # Meaningful content
                                content_map["discovered_content"].append({
                                    "selector": selector,
                                    "index": i,
                                    "content": content.strip()[:500],
                                    "length": len(content.strip()),
                                    "source": "primary_selector"
                                })
                        except Exception:
                            continue
            except Exception as e:
                self.logger.debug(f"Primary selector {selector} failed: {str(e)}")
        
        # If interaction is needed, perform intelligent interactions
        if interaction_requirements.get("needs_interaction"):
            await self._perform_intelligent_interactions(page, interaction_requirements, content_map)
        
        # Try fallback selectors if primary didn't yield enough content
        if len(content_map["discovered_content"]) < 3:
            fallback_selectors = content_location.get("fallback_selectors", [])
            for selector in fallback_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    if elements:
                        for i, element in enumerate(elements[:5]):
                            try:
                                content = await element.text_content()
                                if content and len(content.strip()) > 30:
                                    content_map["discovered_content"].append({
                                        "selector": selector,
                                        "index": i,
                                        "content": content.strip()[:500],
                                        "length": len(content.strip()),
                                        "source": "fallback_selector"
                                    })
                            except Exception:
                                continue
                except Exception as e:
                    self.logger.debug(f"Fallback selector {selector} failed: {str(e)}")
        
        self._add_reasoning_step("content_discovery", f"Discovered {len(content_map['discovered_content'])} content blocks")
        return content_map
    
    async def _perform_intelligent_interactions(self, page: Page, interaction_requirements: Dict[str, Any], content_map: Dict[str, Any]) -> None:
        """
        Perform AI-guided interactions to reveal hidden content.
        """
        interaction_types = interaction_requirements.get("interaction_types", [])
        specific_elements = interaction_requirements.get("specific_elements", [])
        interaction_sequence = interaction_requirements.get("interaction_sequence", [])
        
        for step_description in interaction_sequence:
            try:
                if "scroll" in step_description.lower():
                    await self._intelligent_scroll(page)
                    content_map["interaction_history"].append(f"Performed scroll: {step_description}")
                    
                elif "click" in step_description.lower():
                    clicked = await self._intelligent_click(page, specific_elements)
                    content_map["interaction_history"].append(f"Performed click: {step_description} - Success: {clicked}")
                    
                elif "wait" in step_description.lower():
                    await page.wait_for_load_state('networkidle', timeout=15000)
                    content_map["interaction_history"].append(f"Waited for load: {step_description}")
                    
                elif "expand" in step_description.lower():
                    expanded = await self._expand_sections(page)
                    content_map["interaction_history"].append(f"Expanded sections: {step_description} - Count: {expanded}")
                
                # Brief pause between interactions
                await page.wait_for_timeout(1000)
                
            except Exception as e:
                self.logger.warning(f"Interaction failed: {step_description} - {str(e)}")
                content_map["interaction_history"].append(f"Failed interaction: {step_description} - Error: {str(e)}")
    
    async def _intelligent_scroll(self, page: Page, times: int = 3) -> None:
        """Perform intelligent scrolling to reveal content."""
        for i in range(times):
            # Get current page height
            current_height = await page.evaluate("document.body.scrollHeight")
            
            # Scroll down
            await page.mouse.wheel(0, 1000)
            await page.wait_for_timeout(1500)
            
            # Check if new content loaded
            new_height = await page.evaluate("document.body.scrollHeight")
            if new_height <= current_height:
                break  # No new content, stop scrolling
    
    async def _intelligent_click(self, page: Page, specific_elements: List[str]) -> bool:
        """Intelligently click on relevant elements."""
        # Try specific elements first
        for selector in specific_elements:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    await page.wait_for_timeout(2000)
                    return True
            except Exception:
                continue
        
        # Fallback to common interactive elements
        common_selectors = [
            "button:has-text('Load more')",
            "button:has-text('Show more')",
            "a:has-text('Read more')",
            ".load-more",
            ".show-more",
            "[data-toggle='collapse']"
        ]
        
        for selector in common_selectors:
            try:
                element = await page.query_selector(selector)
                if element and await element.is_visible():
                    await element.click()
                    await page.wait_for_timeout(2000)
                    return True
            except Exception:
                continue
        
        return False
    
    async def _expand_sections(self, page: Page) -> int:
        """Expand collapsible sections to reveal hidden content."""
        expanded_count = 0
        
        expand_selectors = [
            "details:not([open])",
            ".collapse:not(.show)",
            ".accordion-item .accordion-button[aria-expanded='false']",
            "[data-bs-toggle='collapse']",
            ".expandable:not(.expanded)"
        ]
        
        for selector in expand_selectors:
            try:
                elements = await page.query_selector_all(selector)
                for element in elements[:5]:  # Limit to first 5
                    try:
                        if await element.is_visible():
                            await element.click()
                            await page.wait_for_timeout(500)
                            expanded_count += 1
                    except Exception:
                        continue
            except Exception:
                continue
        
        return expanded_count
    
    async def _extract_with_adaptive_interaction(self, page: Page, user_query: Optional[str], content_map: Dict[str, Any]) -> Dict[str, Any]:
        """
        TRULY ADAPTIVE EXTRACTION - AI sees ALL data and reasons about user intent.
        
        This is where the AI becomes truly intelligent:
        1. Gets ALL content from the page (no limits)
        2. Uses ML/AI to analyze relevance to user query
        3. Iteratively processes user intent
        4. Provides FULL data in user's desired format
        """
        self._add_reasoning_step("adaptive_start", f"Starting TRULY ADAPTIVE extraction for query: '{user_query}'")
        
        # STEP 1: GET ALL CONTENT - NO LIMITS, NO RESTRICTIONS
        full_page_content = await self._extract_absolutely_everything(page)
        self._add_reasoning_step("full_extraction", f"Extracted {len(full_page_content)} chars of COMPLETE page content")
        
        # STEP 2: AI ANALYZES USER INTENT AND CONTENT RELEVANCE
        user_intent_analysis = await self._analyze_user_intent_and_content_relevance(user_query, full_page_content)
        self._add_reasoning_step("intent_analysis", f"User Intent: {user_intent_analysis.get('intent_type', 'unknown')}")
        
        # STEP 3: ITERATIVE AI PROCESSING BASED ON USER NEEDS
        final_result = await self._iterative_ai_processing(user_query, full_page_content, user_intent_analysis)
        
        return final_result

    async def _extract_absolutely_everything(self, page: Page) -> str:
        """
        Extract EVERYTHING from the page - no limits, no filtering.
        The AI will decide what's relevant, not pre-filtering.
        """
        try:
            # Get ALL text content using multiple comprehensive strategies
            all_content = await page.evaluate("""
                () => {
                    // Strategy 1: Get all text from body
                    const bodyText = document.body.innerText || '';
                    
                    // Strategy 2: Get text from all elements
                    const allElements = Array.from(document.querySelectorAll('*'));
                    const allTexts = allElements.map(el => {
                        const text = el.textContent || '';
                        return text.trim();
                    }).filter(text => text.length > 0);
                    
                    // Strategy 3: Get HTML content for structure awareness
                    const htmlContent = document.documentElement.innerHTML || '';
                    
                    // Strategy 4: Get metadata and structured data
                    const metadata = {
                        title: document.title || '',
                        url: window.location.href,
                        metaTags: Array.from(document.querySelectorAll('meta')).map(meta => ({
                            name: meta.name || meta.property,
                            content: meta.content
                        })),
                        headings: Array.from(document.querySelectorAll('h1,h2,h3,h4,h5,h6')).map(h => ({
                            level: h.tagName,
                            text: h.textContent
                        })),
                        links: Array.from(document.querySelectorAll('a[href]')).map(a => ({
                            text: a.textContent,
                            href: a.href
                        })).slice(0, 50), // Limit links to prevent overflow
                        images: Array.from(document.querySelectorAll('img[src]')).map(img => ({
                            alt: img.alt,
                            src: img.src
                        })).slice(0, 20) // Limit images
                    };
                    
                    // Combine everything
                    const combinedText = [
                        bodyText,
                        ...allTexts
                    ].join(' ').replace(/\\s+/g, ' ').trim();
                    
                    return {
                        fullText: combinedText,
                        htmlStructure: htmlContent.slice(0, 10000), // First 10k chars of HTML
                        metadata: metadata,
                        textLength: combinedText.length
                    };
                }
            """)
            
            # Combine all content into one comprehensive string
            full_content = f"""
PAGE METADATA:
Title: {all_content.get('metadata', {}).get('title', '')}
URL: {all_content.get('metadata', {}).get('url', '')}

HEADINGS STRUCTURE:
{chr(10).join([f"{h.get('level', '')}: {h.get('text', '')}" for h in all_content.get('metadata', {}).get('headings', [])])}

FULL TEXT CONTENT:
{all_content.get('fullText', '')}

LINKS FOUND:
{chr(10).join([f"- {link.get('text', '')}: {link.get('href', '')}" for link in all_content.get('metadata', {}).get('links', [])])}
"""
            
            self._add_reasoning_step("content_extraction", f"Extracted {len(full_content)} characters of complete content")
            return full_content
            
        except Exception as e:
            self._add_reasoning_step("extraction_error", f"Error in full extraction: {str(e)}")
            # Fallback to simple text extraction
            try:
                fallback_content = await page.evaluate("() => document.body.innerText || document.body.textContent || ''")
                return fallback_content
            except:
                return ""

    async def _analyze_user_intent_and_content_relevance(self, user_query: Optional[str], full_content: str) -> Dict[str, Any]:
        """
        AI analyzes user intent and determines what they really want from the content.
        This is the ML/AI relevance analysis step.
        """
        analysis_prompt = f"""
        ANALYZE USER INTENT AND CONTENT RELEVANCE

        USER QUERY: "{user_query or 'Extract content'}"
        CONTENT LENGTH: {len(full_content)} characters
        CONTENT PREVIEW: {full_content[:2000]}...

        ANALYZE AND DETERMINE:
        1. What does the user REALLY want? (intent analysis)
        2. What format should the response be in?
        3. Which parts of the content are most relevant?
        4. Should this be raw content, structured data, or specific extraction?
        5. What's the user's goal with this data?

        Respond with ONLY JSON:
        {{
            "intent_type": "raw_content|structured_data|specific_extraction|data_analysis",
            "user_goal": "what the user wants to achieve",
            "relevance_score": 0.0-1.0,
            "recommended_format": "string|json|list|custom",
            "key_content_areas": ["list", "of", "relevant", "sections"],
            "extraction_strategy": "how to best extract this",
            "reasoning": "why this analysis"
        }}
        """
        
        try:
            intent_response = await self.ai_client.generate_response(analysis_prompt, format="json")
            intent_analysis = json.loads(intent_response)
            self._add_reasoning_step("intent_determined", f"Intent: {intent_analysis.get('intent_type')} - {intent_analysis.get('user_goal')}")
            return intent_analysis
        except Exception as e:
            self._add_reasoning_step("intent_error", f"Intent analysis failed: {str(e)}")
            return {
                "intent_type": "raw_content",
                "user_goal": "extract all content",
                "relevance_score": 0.7,
                "recommended_format": "string",
                "key_content_areas": ["all"],
                "extraction_strategy": "return_full_content"
            }

    async def _iterative_ai_processing(self, user_query: Optional[str], full_content: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        ITERATIVE AI PROCESSING - The AI processes the user request through multiple reasoning steps.
        """
        intent_type = intent_analysis.get("intent_type", "raw_content")
        recommended_format = intent_analysis.get("recommended_format", "string")
        
        self._add_reasoning_step("processing_start", f"Starting iterative processing for intent: {intent_type}")
        
        if intent_type == "raw_content":
            # User wants ALL the content
            return await self._process_raw_content_request(user_query, full_content, intent_analysis)
        
        elif intent_type == "structured_data":
            # User wants structured/organized data
            return await self._process_structured_data_request(user_query, full_content, intent_analysis)
        
        elif intent_type == "specific_extraction":
            # User wants specific information extracted
            return await self._process_specific_extraction_request(user_query, full_content, intent_analysis)
        
        else:
            # Default to intelligent processing
            return await self._process_intelligent_default(user_query, full_content, intent_analysis)

    async def _process_raw_content_request(self, user_query: Optional[str], full_content: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request for raw/full content - give them EVERYTHING.
        """
        self._add_reasoning_step("raw_processing", "User wants full raw content - providing everything")
        
        return {
            "selected_content": full_content,
            "confidence": 0.95,
            "relevance": 1.0,
            "completeness": 1.0,
            "metadata": {
                "extraction_type": "full_raw_content",
                "word_count": len(full_content.split()),
                "character_count": len(full_content),
                "user_intent": intent_analysis.get("user_goal", ""),
                "content_format": "raw_text"
            },
            "reasoning": "User requested full page content - provided complete unfiltered content"
        }

    async def _process_structured_data_request(self, user_query: Optional[str], full_content: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request for structured data - AI organizes the content intelligently.
        """
        self._add_reasoning_step("structured_processing", "User wants structured data - AI organizing content")
        
        structuring_prompt = f"""
        STRUCTURE THIS CONTENT BASED ON USER REQUEST

        USER QUERY: "{user_query}"
        USER GOAL: {intent_analysis.get('user_goal', '')}
        
        FULL CONTENT TO STRUCTURE:
        {full_content}

        ORGANIZE this content into a logical structure that best serves the user's goal.
        Consider the content type and user intent.

        Respond with ONLY JSON in this format:
        {{
            "title": "main title or topic",
            "summary": "brief summary of content",
            "main_content": "the primary content organized logically",
            "key_points": ["important", "points", "extracted"],
            "metadata": {{
                "content_type": "detected content type",
                "topics": ["main", "topics"],
                "length_stats": "content length info"
            }},
            "structured_sections": {{
                "section1": "content for section 1",
                "section2": "content for section 2"
            }}
        }}
        """
        
        try:
            structured_response = await self.ai_client.generate_response(structuring_prompt, format="json")
            structured_data = json.loads(structured_response)
            
            return {
                "selected_content": structured_data,
                "confidence": 0.9,
                "relevance": 0.95,
                "completeness": 0.9,
                "metadata": {
                    "extraction_type": "ai_structured_content",
                    "word_count": len(full_content.split()),
                    "user_intent": intent_analysis.get("user_goal", ""),
                    "content_format": "structured_json"
                },
                "reasoning": "AI structured the content based on user intent and content analysis"
            }
        except Exception as e:
            self._add_reasoning_step("structuring_error", f"Structuring failed: {str(e)}")
            return await self._process_raw_content_request(user_query, full_content, intent_analysis)

    async def _process_specific_extraction_request(self, user_query: Optional[str], full_content: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process request for specific information extraction.
        """
        self._add_reasoning_step("specific_processing", "User wants specific extraction - AI focusing on relevant parts")
        
        extraction_prompt = f"""
        EXTRACT SPECIFIC INFORMATION BASED ON USER REQUEST

        USER QUERY: "{user_query}"
        USER GOAL: {intent_analysis.get('user_goal', '')}
        KEY AREAS: {intent_analysis.get('key_content_areas', [])}

        FULL CONTENT TO ANALYZE:
        {full_content}

        Extract ONLY the information that directly answers the user's query.
        Focus on relevance and completeness for their specific need.

        Respond with ONLY JSON:
        {{
            "extracted_info": "the specific information requested",
            "supporting_details": "additional relevant context",
            "confidence_reasoning": "why this extraction is accurate",
            "completeness_assessment": "how complete this answer is"
        }}
        """
        
        try:
            extraction_response = await self.ai_client.generate_response(extraction_prompt, format="json")
            extracted_data = json.loads(extraction_response)
            
            return {
                "selected_content": extracted_data,
                "confidence": 0.85,
                "relevance": 0.9,
                "completeness": 0.8,
                "metadata": {
                    "extraction_type": "specific_ai_extraction",
                    "user_intent": intent_analysis.get("user_goal", ""),
                    "content_format": "targeted_extraction"
                },
                "reasoning": "AI extracted specific information matching user query"
            }
        except Exception as e:
            self._add_reasoning_step("extraction_error", f"Specific extraction failed: {str(e)}")
            return await self._process_raw_content_request(user_query, full_content, intent_analysis)

    async def _process_intelligent_default(self, user_query: Optional[str], full_content: str, intent_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """
        Intelligent default processing when intent is unclear.
        """
        self._add_reasoning_step("intelligent_default", "Using intelligent default processing")
        
        # Provide both raw content AND structured analysis
        return {
            "selected_content": {
                "raw_content": full_content,
                "intelligent_analysis": intent_analysis,
                "ai_assessment": "Content provided in both raw and analyzed formats for maximum utility"
            },
            "confidence": 0.8,
            "relevance": 0.8,
            "completeness": 1.0,
            "metadata": {
                "extraction_type": "intelligent_hybrid",
                "user_intent": intent_analysis.get("user_goal", ""),
                "content_format": "hybrid_raw_and_structured"
            },
            "reasoning": "Provided both raw content and intelligent analysis for maximum user value"
        }
    
    async def _validate_and_enhance_content(self, page: Page, extracted_content: Dict[str, Any], user_query: Optional[str]) -> Dict[str, Any]:
        """
        ITERATIVE MASTER CONTROLLER - Keep going until we get REAL results!
        
        This is where the AI takes full control and doesn't give up until success.
        """
        content = extracted_content.get("selected_content", "")
        original_length = len(content)
        attempt_count = 0
        max_attempts = 20  # We don't give up easily
        
        self._add_reasoning_step("master_controller_start", f"MASTER CONTROLLER ACTIVATED - Starting with {original_length} chars")
        
        # ITERATIVE INTELLIGENCE LOOP - The AI examines its own work and improves
        while attempt_count < max_attempts:
            attempt_count += 1
            self._add_reasoning_step("iteration_start", f"=== ATTEMPT {attempt_count}/{max_attempts} ===")
            
            # AI SELF-EXAMINATION: Look at what we have and decide if it's good enough
            result_analysis = await self._ai_analyze_extraction_quality(content, user_query, attempt_count)
            
            quality_score = result_analysis.get("quality_score", 0.0)
            is_complete = result_analysis.get("is_complete", False)
            missing_elements = result_analysis.get("missing_elements", [])
            improvement_strategy = result_analysis.get("improvement_strategy", "")
            
            self._add_reasoning_step("self_analysis", f"Quality: {quality_score:.2f}, Complete: {is_complete}, Missing: {missing_elements}")
            
            # SUCCESS CHECK: If AI is satisfied, we're done
            if is_complete and quality_score > 0.8 and len(content) > 1000:
                self._add_reasoning_step("success_achieved", f"AI SATISFIED: Quality {quality_score:.2f}, {len(content)} chars")
                break
            
            # ADAPTIVE IMPROVEMENT: AI decides what to try next
            if improvement_strategy and attempt_count < max_attempts - 2:  # Save last 2 attempts for desperate measures
                new_content = await self._execute_ai_improvement_strategy(page, improvement_strategy, content, missing_elements)
                
                if len(new_content) > len(content):
                    content = new_content
                    self._add_reasoning_step("improvement_success", f"Strategy '{improvement_strategy}' improved content: {len(content)} chars")
                    continue
            
            # AGGRESSIVE DISCOVERY: Try different extraction methods
            discovery_methods = [
                ("comprehensive_scan", self._comprehensive_content_extraction),
                ("aggressive_scroll", self._aggressive_content_discovery_with_extraction),
                ("hidden_content", self._discover_and_extract_hidden_content),
                ("dynamic_interaction", self._smart_page_interaction),
                ("desperate_extraction", self._final_desperate_extraction)
            ]
            
            for method_name, method_func in discovery_methods:
                try:
                    if method_name == "aggressive_scroll":
                        await method_func(page)
                        new_content = await self._extract_comprehensive_content(page)
                    else:
                        new_content = await method_func(page)
                    
                    if len(new_content) > len(content) * 1.1:  # At least 10% improvement
                        content = new_content
                        self._add_reasoning_step("discovery_success", f"Method '{method_name}' found more: {len(content)} chars")
                        break
                        
                except Exception as e:
                    self._add_reasoning_step("method_failed", f"Method '{method_name}' failed: {str(e)}")
            
            # AI DECISION: Should we continue?
            if attempt_count > 10:
                continuation_decision = await self._ai_decide_continuation(content, user_query, attempt_count, max_attempts)
                if not continuation_decision.get("should_continue", True):
                    reason = continuation_decision.get("reason", "AI decided to stop")
                    self._add_reasoning_step("ai_stop_decision", f"AI decided to stop at attempt {attempt_count}: {reason}")
                    break
        
        # FINAL AI VALIDATION
        final_analysis = await self._ai_final_validation(content, user_query, attempt_count)
        
        # Update the extracted content with everything we learned
        extracted_content["selected_content"] = content
        
        # Calculate final metrics
        word_count = len(content.split())
        
        # Update metadata with our journey
        metadata = extracted_content.get("metadata", {})
        metadata.update({
            "word_count": word_count,
            "character_count": len(content),
            "extraction_method": "ai_controlled_iterative_adaptive",
            "reasoning_steps": len(self.reasoning_chain),
            "attempts_made": attempt_count,
            "content_growth": len(content) - original_length,
            "final_length": len(content),
            "ai_quality_score": final_analysis.get("final_quality_score", 0.0),
            "ai_completeness": final_analysis.get("is_complete", False),
            "extraction_journey": final_analysis.get("journey_summary", "")
        })
        
        # AI-determined confidence (not fake confidence)
        ai_confidence = final_analysis.get("confidence_score", 0.5)
        extracted_content["confidence"] = ai_confidence
        extracted_content["metadata"] = metadata
        
        self._add_reasoning_step("master_complete", f"MASTER CONTROLLER COMPLETE: {word_count} words, {len(content)} chars, AI confidence: {ai_confidence:.2f}")
        
        return extracted_content
    
    async def _ai_analyze_extraction_quality(self, content: str, user_query: Optional[str], attempt: int) -> Dict[str, Any]:
        """
        AI SELF-EXAMINATION: The AI looks at its own work and decides if it's good enough.
        This is the heart of iterative improvement.
        """
        word_count = len(content.split())
        char_count = len(content)
        
        analysis_prompt = f"""
        ANALYZE YOUR OWN EXTRACTION WORK - BE HONEST AND CRITICAL!
        
        Attempt #{attempt}
        Content Length: {char_count} characters, {word_count} words
        User Query: {user_query or "General extraction"}
        
        CONTENT TO ANALYZE:
        {content[:2000]}{"..." if len(content) > 2000 else ""}
        
        ANALYZE THIS EXTRACTION AND DECIDE:
        1. Quality Score (0.0-1.0): How good is this content?
        2. Is Complete (true/false): Does this feel like complete page content?
        3. Missing Elements: What's likely missing? (navigation, articles, comments, etc.)
        4. Improvement Strategy: What should we try next?
        
        BE CRITICAL - Don't accept mediocre results!
        
        Respond with ONLY a JSON object:
        {{
            "quality_score": 0.0-1.0,
            "is_complete": true/false,
            "missing_elements": ["element1", "element2"],
            "improvement_strategy": "specific strategy to try next",
            "reasoning": "why you gave this score",
            "content_assessment": "what the content actually contains"
        }}
        """
        
        try:
            ai_response = await self.ai_client.generate_response(analysis_prompt, format="json")
            result = json.loads(ai_response)
            
            self._add_reasoning_step("ai_self_analysis", f"AI Quality Assessment: {result.get('quality_score', 0):.2f} - {result.get('reasoning', '')}")
            return result
            
        except Exception as e:
            self._add_reasoning_step("ai_analysis_error", f"AI analysis failed: {str(e)}")
            # Fallback analysis
            return {
                "quality_score": 0.3 if char_count < 1000 else 0.6 if char_count < 3000 else 0.8,
                "is_complete": char_count > 2000 and word_count > 200,
                "missing_elements": ["unknown"],
                "improvement_strategy": "try_scrolling_and_waiting",
                "reasoning": "fallback analysis due to AI error"
            }

    async def _execute_ai_improvement_strategy(self, page: Page, strategy: str, current_content: str, missing_elements: List[str]) -> str:
        """
        Execute the AI's recommended improvement strategy.
        """
        self._add_reasoning_step("executing_strategy", f"AI Strategy: {strategy}")
        
        try:
            if "scroll" in strategy.lower():
                await self._aggressive_content_discovery_with_extraction(page)
                return await self._extract_comprehensive_content(page)
            
            elif "hidden" in strategy.lower() or "expand" in strategy.lower():
                hidden_content = await self._discover_and_extract_hidden_content(page)
                return hidden_content if len(hidden_content) > len(current_content) else current_content
            
            elif "interact" in strategy.lower() or "click" in strategy.lower():
                await self._smart_page_interaction(page)
                return await self._extract_comprehensive_content(page)
            
            elif "wait" in strategy.lower() or "load" in strategy.lower():
                await page.wait_for_timeout(5000)  # Wait for more content
                return await self._extract_comprehensive_content(page)
            
            else:
                # Default comprehensive re-extraction
                return await self._comprehensive_content_extraction(page)
                
        except Exception as e:
            self._add_reasoning_step("strategy_error", f"Strategy '{strategy}' failed: {str(e)}")
            return current_content

    async def _ai_decide_continuation(self, content: str, user_query: Optional[str], attempt: int, max_attempts: int) -> Dict[str, Any]:
        """
        AI decides whether to continue trying or stop.
        """
        decision_prompt = f"""
        SHOULD I CONTINUE TRYING TO EXTRACT MORE CONTENT?
        
        Current Situation:
        - Attempt: {attempt}/{max_attempts}
        - Content: {len(content)} chars, {len(content.split())} words
        - Query: {user_query or "General extraction"}
        
        Content Preview:
        {content[:1000]}{"..." if len(content) > 1000 else ""}
        
        DECIDE: Should I keep trying or is this enough?
        Consider: effort vs. benefit, content quality, user needs
        
        Respond ONLY with JSON:
        {{
            "should_continue": true/false,
            "reason": "why continue or stop",
            "confidence_in_decision": 0.0-1.0
        }}
        """
        
        try:
            ai_response = await self.ai_client.generate_response(decision_prompt, format="json")
            result = json.loads(ai_response)
            self._add_reasoning_step("ai_decision", f"AI Decision: {'Continue' if result.get('should_continue') else 'Stop'} - {result.get('reason', '')}")
            return result
            
        except Exception as e:
            self._add_reasoning_step("decision_error", f"AI decision failed: {str(e)}")
            # Conservative fallback - keep trying if we don't have much content
            return {
                "should_continue": len(content) < 2000,
                "reason": "fallback decision due to AI error",
                "confidence_in_decision": 0.3
            }

    async def _ai_final_validation(self, content: str, user_query: Optional[str], attempts: int) -> Dict[str, Any]:
        """
        Final AI validation of the entire extraction process.
        """
        validation_prompt = f"""
        FINAL VALIDATION OF EXTRACTION PROCESS
        
        Extraction Complete After {attempts} Attempts
        Final Content: {len(content)} chars, {len(content.split())} words
        User Query: {user_query or "General extraction"}
        
        FINAL CONTENT:
        {content[:2000]}{"..." if len(content) > 2000 else ""}
        
        PROVIDE FINAL ASSESSMENT:
        1. Final Quality Score (0.0-1.0)
        2. Is this extraction complete and satisfactory?
        3. Confidence in this result (0.0-1.0)
        4. Summary of the extraction journey
        
        Respond ONLY with JSON:
        {{
            "final_quality_score": 0.0-1.0,
            "is_complete": true/false,
            "confidence_score": 0.0-1.0,
            "journey_summary": "brief summary of extraction process",
            "content_verdict": "final assessment of content quality"
        }}
        """
        
        try:
            ai_response = await self.ai_client.generate_response(validation_prompt, format="json")
            result = json.loads(ai_response)
            self._add_reasoning_step("ai_final_validation", f"Final AI Verdict: {result.get('final_quality_score', 0):.2f} - {result.get('journey_summary', '')}")
            return result
            
        except Exception as e:
            self._add_reasoning_step("validation_error", f"AI validation failed: {str(e)}")
            return {
                "final_quality_score": 0.7 if len(content) > 1000 else 0.4,
                "is_complete": len(content) > 1000,
                "confidence_score": 0.6,
                "journey_summary": f"Completed extraction in {attempts} attempts with {len(content)} characters",
                "content_verdict": "extraction completed with fallback validation"
            }

    async def _aggressive_content_discovery_with_extraction(self, page: Page) -> None:
        """
        Aggressive content discovery through scrolling and interaction.
        """
        try:
            # Progressive scrolling to load dynamic content
            viewport_height = await page.evaluate("() => window.innerHeight")
            current_position = 0
            
            for scroll_step in range(5):  # 5 scroll steps
                # Scroll down
                await page.evaluate(f"window.scrollTo(0, {current_position + viewport_height})")
                current_position += viewport_height
                
                # Wait for content to load
                await page.wait_for_timeout(2000)
                
                # Try to click any "Load More" or "Show More" buttons
                load_more_selectors = [
                    "button:has-text('Load More')",
                    "button:has-text('Show More')",
                    "a:has-text('Load More')",
                    "a:has-text('Show More')",
                    "[class*='load']:has(span):has-text('more')",
                    "[class*='show']:has(span):has-text('more')"
                ]
                
                for selector in load_more_selectors:
                    try:
                        element = await page.query_selector(selector)
                        if element:
                            await element.click()
                            await page.wait_for_timeout(3000)
                            break
                    except:
                        continue
            
            # Scroll back to top
            await page.evaluate("window.scrollTo(0, 0)")
            await page.wait_for_timeout(1000)
            
        except Exception as e:
            self._add_reasoning_step("scroll_error", f"Aggressive scrolling failed: {str(e)}")

    async def _discover_and_extract_hidden_content(self, page: Page) -> str:
        """
        Discover and extract hidden content by expanding elements.
        """
        try:
            content_parts = []
            
            # Try to expand collapsible content
            expandable_selectors = [
                "details",
                "[class*='collapse']",
                "[class*='expand']",
                "[class*='accordion']",
                "[data-toggle='collapse']",
                "button[aria-expanded='false']"
            ]
            
            for selector in expandable_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements[:5]:  # Limit to first 5 to avoid spam
                        await element.click()
                        await page.wait_for_timeout(1000)
                except:
                    continue
            
            # Extract content after expansion
            content = await self._extract_comprehensive_content(page)
            return content
            
        except Exception as e:
            self._add_reasoning_step("hidden_content_error", f"Hidden content discovery failed: {str(e)}")
            return ""

    async def _smart_page_interaction(self, page: Page) -> str:
        """
        Smart interaction with page elements to reveal more content.
        """
        try:
            # Try clicking on tabs, navigation elements
            interactive_selectors = [
                "nav a",
                ".tab",
                ".tabs a",
                "[role='tab']",
                "button[data-tab]",
                ".menu-item"
            ]
            
            for selector in interactive_selectors:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements[:3]:  # Try first 3 elements
                        await element.click()
                        await page.wait_for_timeout(2000)
                        
                        # Check if new content appeared
                        current_content = await self._extract_comprehensive_content(page)
                        if len(current_content) > 500:  # If we got substantial content
                            return current_content
                            
                except:
                    continue
            
            return await self._extract_comprehensive_content(page)
            
        except Exception as e:
            self._add_reasoning_step("interaction_error", f"Smart interaction failed: {str(e)}")
            return ""

    async def _discover_additional_content(self, page: Page, current_content: str) -> str:
        """
        Try alternative methods to discover additional content.
        """
        try:
            # Method 1: Look for pagination
            pagination_selectors = [
                "a:has-text('Next')",
                "button:has-text('Next')",
                ".pagination a",
                "[class*='next']",
                "[class*='more']"
            ]
            
            for selector in pagination_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        await element.click()
                        await page.wait_for_timeout(3000)
                        new_content = await self._extract_comprehensive_content(page)
                        if len(new_content) > len(current_content):
                            return new_content
                except:
                    continue
            
            # Method 2: Try different content containers
            container_selectors = [
                ".container",
                ".wrapper", 
                ".page-content",
                "#main-content",
                ".site-content"
            ]
            
            for selector in container_selectors:
                try:
                    element = await page.query_selector(selector)
                    if element:
                        text = await element.inner_text()
                        if text and len(text) > len(current_content):
                            return text
                except:
                    continue
            
            return current_content
            
        except Exception as e:
            self._add_reasoning_step("additional_discovery_error", f"Additional content discovery failed: {str(e)}")
            return current_content
        """
        Comprehensive content extraction using multiple strategies.
        """
        return await page.evaluate("""
            () => {
                const contentStrategies = [
                    // Strategy 1: Main content areas
                    () => {
                        const selectors = ['main', 'article', '.content', '.post', '.entry', '[role="main"]', '#content', '#main', '.main-content'];
                        for (const selector of selectors) {
                            const element = document.querySelector(selector);
                            if (element) {
                                const text = element.textContent || '';
                                if (text.length > 1000) return text;
                            }
                        }
                        return '';
                    },
                    
                    // Strategy 2: Largest text blocks
                    () => {
                        const allElements = Array.from(document.querySelectorAll('*'));
                        const textBlocks = allElements
                            .filter(el => {
                                const style = window.getComputedStyle(el);
                                return style.display !== 'none' && style.visibility !== 'hidden';
                            })
                            .map(el => ({
                                element: el,
                                text: (el.textContent || '').trim(),
                                length: (el.textContent || '').length
                            }))
                            .filter(item => item.length > 500)
                            .sort((a, b) => b.length - a.length);
                        
                        if (textBlocks.length > 0) {
                            return textBlocks.slice(0, 3).map(item => item.text).join('\\n\\n');
                        }
                        return '';
                    },
                    
                    // Strategy 3: All paragraphs and headings
                    () => {
                        const textElements = document.querySelectorAll('p, h1, h2, h3, h4, h5, h6, li, td, th, span, div');
                        const texts = Array.from(textElements)
                            .map(el => (el.textContent || '').trim())
                            .filter(text => text.length > 50)
                            .slice(0, 100);  // Take first 100 substantial text blocks
                        return texts.join('\\n');
                    },
                    
                    // Strategy 4: Full body text as fallback
                    () => {
                        return document.body.textContent || '';
                    }
                ];
                
                // Try each strategy and return the longest result
                let bestContent = '';
                for (const strategy of contentStrategies) {
                    try {
                        const content = strategy();
                        if (content.length > bestContent.length) {
                            bestContent = content;
                        }
                    } catch (e) {
                        console.warn('Content extraction strategy failed:', e);
                    }
                }
                
                return bestContent.slice(0, 50000);  // Increased limit for comprehensive extraction
            }
        """)
    
    async def _aggressive_content_discovery(self, page: Page):
        """
        Aggressively discover content through scrolling, clicking, and waiting.
        """
        try:
            # Phase 1: Progressive scrolling
            for i in range(self.max_scroll_attempts):
                await page.evaluate(f"window.scrollTo(0, {i * 1000})")
                await asyncio.sleep(0.5)
                
                # Check for "load more" buttons or similar
                load_more_clicked = await page.evaluate("""
                    () => {
                        const loadMoreSelectors = [
                            '.load-more', '.show-more', '.view-more',
                            '[data-action*="load"]', '[data-action*="more"]'
                        ];
                        
                        // Check by text content
                        const buttons = document.querySelectorAll('button, a');
                        for (const btn of buttons) {
                            const text = btn.textContent.toLowerCase();
                            if (text.includes('load more') || text.includes('show more') || 
                                text.includes('read more') || text.includes('view more')) {
                                if (btn.offsetParent !== null) {
                                    btn.click();
                                    return true;
                                }
                            }
                        }
                        
                        for (const selector of loadMoreSelectors) {
                            const element = document.querySelector(selector);
                            if (element && element.offsetParent !== null) {
                                element.click();
                                return true;
                            }
                        }
                        return false;
                    }
                """)
                
                if load_more_clicked:
                    await asyncio.sleep(2)  # Wait for content to load
            
            # Phase 2: Scroll to bottom and wait
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
            await asyncio.sleep(2)
            
            # Phase 3: Look for expandable sections
            await page.evaluate("""
                () => {
                    const expandableSelectors = [
                        '.expandable', '.collapsible', '.accordion',
                        '[data-toggle="collapse"]', '.expand', '.toggle'
                    ];
                    
                    for (const selector of expandableSelectors) {
                        const elements = document.querySelectorAll(selector);
                        elements.forEach(el => {
                            if (el.offsetParent !== null) {
                                el.click();
                            }
                        });
                    }
                }
            """)
            
            await asyncio.sleep(1)
            
        except Exception as e:
            self.logger.warning(f"Aggressive content discovery error: {str(e)}")
    
    async def _extract_comprehensive_content(self, page: Page) -> str:
        """
        Extract comprehensive content from current page state.
        """
        return await page.evaluate("""
            () => {
                // Get all visible text content
                const walker = document.createTreeWalker(
                    document.body,
                    NodeFilter.SHOW_TEXT,
                    {
                        acceptNode: function(node) {
                            const parent = node.parentElement;
                            if (!parent) return NodeFilter.FILTER_REJECT;
                            
                            const style = window.getComputedStyle(parent);
                            if (style.display === 'none' || style.visibility === 'hidden') {
                                return NodeFilter.FILTER_REJECT;
                            }
                            
                            const text = node.textContent.trim();
                            return text.length > 10 ? NodeFilter.FILTER_ACCEPT : NodeFilter.FILTER_REJECT;
                        }
                    },
                    false
                );
                
                const textNodes = [];
                let node;
                while (node = walker.nextNode()) {
                    textNodes.push(node.textContent.trim());
                }
                
                return textNodes.join(' ').slice(0, 100000);  // Very large limit
            }
        """)
    
    async def _discover_additional_content(self, page: Page, current_content: str) -> str:
        """
        Discover additional content using alternative methods.
        """
        try:
            # Method 1: Look for hidden content
            hidden_content = await page.evaluate("""
                () => {
                    const hiddenElements = document.querySelectorAll('[style*="display: none"], .hidden, .collapse');
                    let content = '';
                    
                    hiddenElements.forEach(el => {
                        const text = el.textContent || '';
                        if (text.length > 100) {
                            content += text + '\\n';
                        }
                    });
                    
                    return content;
                }
            """)
            
            # Method 2: Try to reveal hidden content
            await page.evaluate("""
                () => {
                    // Remove display:none styles
                    const hiddenElements = document.querySelectorAll('[style*="display: none"]');
                    hiddenElements.forEach(el => {
                        el.style.display = 'block';
                    });
                    
                    // Show collapsed content
                    const collapsedElements = document.querySelectorAll('.collapse, .collapsed');
                    collapsedElements.forEach(el => {
                        el.classList.remove('collapse', 'collapsed');
                        el.classList.add('show', 'expanded');
                    });
                }
            """)
            
            await asyncio.sleep(1)
            
            # Extract again
            revealed_content = await self._extract_comprehensive_content(page)
            
            # Return the longer content
            if len(revealed_content) > len(current_content):
                return revealed_content + "\n\n" + hidden_content
            else:
                return current_content + "\n\n" + hidden_content if hidden_content else current_content
                
        except Exception as e:
            self.logger.warning(f"Additional content discovery error: {str(e)}")
            return current_content
    
    async def _try_alternative_extraction_methods(self, page: Page):
        """
        Try alternative extraction methods when standard approaches fail.
        """
        try:
            # Method 1: Simulate user interactions
            await page.evaluate("""
                () => {
                    // Trigger common events that might load content
                    ['scroll', 'resize', 'focus', 'mouseover'].forEach(eventType => {
                        window.dispatchEvent(new Event(eventType));
                    });
                    
                    // Click on tabs, buttons, links that might reveal content
                    const interactiveElements = document.querySelectorAll('button, .tab, .nav-item, a[href="#"]');
                    interactiveElements.forEach((el, index) => {
                        if (index < 5) {  // Limit to first 5 elements
                            setTimeout(() => el.click(), index * 500);
                        }
                    });
                }
            """)
            
            await asyncio.sleep(3)
            
            # Method 2: Force page refresh and re-extraction
            await page.reload(wait_until='networkidle')
            await asyncio.sleep(2)
            
        except Exception as e:
            self.logger.warning(f"Alternative extraction methods error: {str(e)}")
    
    async def _comprehensive_content_extraction(self, page: Page) -> str:
        """
        Comprehensive content extraction that gets everything possible.
        """
        try:
            # Get all text content from various sources
            content_parts = []
            
            # Main content areas
            for selector in ["main", "article", "[role='main']", ".content", "#content", ".post", ".entry"]:
                try:
                    elements = await page.query_selector_all(selector)
                    for element in elements:
                        text = await element.inner_text()
                        if text and len(text.strip()) > 50:
                            content_parts.append(text.strip())
                except:
                    continue
            
            # Body content if nothing else found
            if not content_parts:
                try:
                    body_text = await page.evaluate("() => document.body.innerText")
                    if body_text:
                        content_parts.append(body_text)
                except:
                    pass
            
            # Combine all content
            full_content = "\n\n".join(content_parts)
            
            # Clean up the content
            lines = [line.strip() for line in full_content.split('\n') if line.strip()]
            cleaned_content = '\n'.join(lines)
            
            return cleaned_content
            
        except Exception as e:
            self._add_reasoning_step("extraction_error", f"Comprehensive extraction failed: {str(e)}")
            return ""

    async def _final_desperate_extraction(self, page: Page) -> str:
        """
        Last resort: extract absolutely everything we can find.
        """
        return await page.evaluate("""
            () => {
                // Extract EVERYTHING - no limits, no filtering
                const allText = [];
                
                // Get all text nodes
                const walker = document.createTreeWalker(
                    document.documentElement,
                    NodeFilter.SHOW_TEXT,
                    null,
                    false
                );
                
                let node;
                while (node = walker.nextNode()) {
                    const text = node.textContent.trim();
                    if (text.length > 5) {
                        allText.push(text);
                    }
                }
                
                // Also get all innerHTML content
                const allElements = document.querySelectorAll('*');
                allElements.forEach(el => {
                    if (el.textContent && el.textContent.length > 50) {
                        allText.push(el.textContent.trim());
                    }
                });
                
                // Remove duplicates and join
                const uniqueText = [...new Set(allText)];
                return uniqueText.join('\\n').slice(0, 200000);  // Massive limit for desperate extraction
            }
        """)
    
    
    def _fallback_page_analysis(self, page_info: Dict[str, Any], user_query: Optional[str]) -> Dict[str, Any]:
        """
        Provide fallback analysis when AI analysis fails.
        """
        # Heuristic-based analysis
        has_articles = len(page_info.get("articles", [])) > 0
        has_tables = len(page_info.get("tables", [])) > 0
        has_buttons = len(page_info.get("buttons", [])) > 0
        has_forms = len(page_info.get("forms", [])) > 0
        
        if has_articles:
            content_type = "article"
            primary_selectors = ["article", "main", ".content", ".post"]
        elif has_tables:
            content_type = "table"
            primary_selectors = ["table", ".table", ".data-table"]
        elif has_forms:
            content_type = "form"
            primary_selectors = ["form", ".form"]
        else:
            content_type = "mixed"
            primary_selectors = ["main", ".content", "#content", "body"]
        
        return {
            "content_location": {
                "primary_selectors": primary_selectors,
                "fallback_selectors": ["body", "*"],
                "content_type": content_type,
                "extraction_difficulty": "medium"
            },
            "interaction_requirements": {
                "needs_interaction": has_buttons,
                "interaction_types": ["scroll", "click_buttons"] if has_buttons else ["scroll"],
                "specific_elements": [],
                "interaction_sequence": ["Scroll to load content", "Click expand buttons"]
            },
            "technical_challenges": {
                "has_dynamic_content": page_info.get("hasInfiniteScroll", False),
                "requires_javascript": True,
                "has_anti_scraping": False,
                "needs_authentication": False,
                "has_rate_limiting": False,
                "content_behind_paywall": False
            },
            "browser_optimizations": {
                "recommended_wait_time": 2000,
                "needs_custom_headers": False,
                "requires_mobile_viewport": False,
                "disable_images": False,
                "bypass_cloudflare": False
            },
            "extraction_confidence": 0.6,
            "reasoning": "Fallback heuristic analysis",
            "page_info": page_info
        }
    
    def _add_reasoning_step(self, phase: str, details: Any) -> None:
        """Add a step to the reasoning chain for transparency."""
        self.reasoning_chain.append({
            "timestamp": datetime.now().isoformat(),
            "phase": phase,
            "details": details
        })
        self.logger.debug(f"Reasoning step [{phase}]: {details}")
    
    async def health_check(self) -> bool:
        """Check if the adaptive strategy is healthy."""
        try:
            return (
                self.ai_client is not None and
                self.browser_manager is not None and
                await self.ai_client.health_check()
            )
        except Exception:
            return False
