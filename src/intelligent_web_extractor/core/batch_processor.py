"""
Batch Processor Core Component

Provides batch processing capabilities for extracting content from
multiple URLs efficiently with intelligent resource management.
"""

import asyncio
from typing import Dict, List, Optional, Any, Union
from datetime import datetime
import logging

from ..models.extraction_result import ExtractionResult
from ..models.config import ExtractorConfig
from ..core.extractor import AdaptiveContentExtractor


class BatchProcessor:
    """
    High-level batch processor for extracting content from multiple URLs.
    
    Provides efficient batch processing with intelligent resource management,
    progress tracking, and error handling.
    """
    
    def __init__(self, config: Optional[ExtractorConfig] = None):
        """
        Initialize the batch processor.
        
        Args:
            config: Configuration settings
        """
        self.config = config or ExtractorConfig()
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self._total_batches = 0
        self._successful_batches = 0
        self._total_urls = 0
        self._successful_urls = 0
        
        self._initialized = False
    
    async def __aenter__(self):
        """Async context manager entry"""
        await self.initialize()
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.close()
    
    async def initialize(self):
        """Initialize the batch processor"""
        if self._initialized:
            return
        
        self._initialized = True
        self.logger.info("Batch Processor initialized")
    
    async def close(self):
        """Close the batch processor"""
        self._initialized = False
        self.logger.info("Batch Processor closed")
    
    async def process_urls(
        self,
        urls: List[str],
        queries: Optional[List[str]] = None,
        parallel_workers: Optional[int] = None
    ) -> List[ExtractionResult]:
        """
        Process multiple URLs in batch.
        
        Args:
            urls: List of URLs to process
            queries: Optional list of queries (one per URL)
            parallel_workers: Number of parallel workers
            
        Returns:
            List of ExtractionResult objects
        """
        if not self._initialized:
            await self.initialize()
        
        # Set up queries
        if queries is None:
            queries = [None] * len(urls)
        
        # Ensure lists have same length
        if len(queries) != len(urls):
            queries.extend([None] * (len(urls) - len(queries)))
        
        # Set worker count
        if parallel_workers is None:
            parallel_workers = self.config.performance.max_workers
        
        self.logger.info(f"Starting batch processing of {len(urls)} URLs with {parallel_workers} workers")
        
        # Create extractor
        async with AdaptiveContentExtractor(self.config) as extractor:
            # Process URLs in batches
            batch_size = parallel_workers
            results = []
            
            for i in range(0, len(urls), batch_size):
                batch_urls = urls[i:i + batch_size]
                batch_queries = queries[i:i + batch_size]
                
                batch_results = await self._process_batch(extractor, batch_urls, batch_queries)
                results.extend(batch_results)
                
                self.logger.info(f"Completed batch {i//batch_size + 1}/{(len(urls) + batch_size - 1)//batch_size}")
            
            # Update statistics
            self._update_batch_statistics(results)
            
            self.logger.info(f"Batch processing completed: {len([r for r in results if r.success])}/{len(results)} successful")
            
            return results
    
    async def _process_batch(
        self,
        extractor: AdaptiveContentExtractor,
        urls: List[str],
        queries: List[str]
    ) -> List[ExtractionResult]:
        """Process a batch of URLs"""
        tasks = []
        
        for url, query in zip(urls, queries):
            task = extractor.extract_content(url=url, user_query=query)
            tasks.append(task)
        
        # Execute tasks with concurrency limit
        semaphore = asyncio.Semaphore(self.config.performance.max_concurrent_requests)
        
        async def limited_extract(task):
            async with semaphore:
                return await task
        
        results = await asyncio.gather(*[limited_extract(task) for task in tasks], return_exceptions=True)
        
        # Process results
        extraction_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                # Create error result for failed extractions
                error_result = ExtractionResult(
                    url=urls[i],
                    content="",
                    success=False,
                    error_message=str(result),
                    error_type=type(result).__name__,
                    extraction_started=datetime.now(),
                    extraction_completed=datetime.now()
                )
                extraction_results.append(error_result)
            else:
                extraction_results.append(result)
        
        return extraction_results
    
    def _update_batch_statistics(self, results: List[ExtractionResult]):
        """Update batch processing statistics"""
        self._total_batches += 1
        self._total_urls += len(results)
        self._successful_urls += len([r for r in results if r.success])
        
        if len(results) > 0:
            self._successful_batches += 1
    
    def get_batch_statistics(self) -> Dict[str, Any]:
        """Get batch processing statistics"""
        return {
            "total_batches": self._total_batches,
            "successful_batches": self._successful_batches,
            "batch_success_rate": self._successful_batches / max(self._total_batches, 1),
            "total_urls": self._total_urls,
            "successful_urls": self._successful_urls,
            "url_success_rate": self._successful_urls / max(self._total_urls, 1)
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check on batch processor"""
        return {
            "initialized": self._initialized,
            "total_batches": self._total_batches,
            "successful_batches": self._successful_batches,
            "total_urls": self._total_urls,
            "successful_urls": self._successful_urls
        } 