"""
Base classes for pipeline stages.

This module provides abstract base classes for creating pipeline stages
with built-in error handling, metrics, and retry logic.
"""

import time
import logging
from abc import ABC, abstractmethod
from typing import Callable, Dict, Any
from collections import defaultdict

from .context import PipelineContext, PipelineStatus


class PipelineStage(ABC):
    """
    Abstract base class for all pipeline stages.
    
    Each stage processes the context and returns an updated context.
    Stages are composable and can be chained together.
    
    Attributes:
        name: Human-readable name for this stage
        logger: Logger instance for this stage
        metrics: Performance metrics for this stage
    """
    
    def __init__(self, name: str):
        """
        Initialize the pipeline stage.
        
        Args:
            name: Name of this stage
        """
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.metrics = defaultdict(int)
    
    @abstractmethod
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process the context and return updated context.
        
        This method must be implemented by concrete stage classes.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Updated pipeline context
        """
        pass
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute stage with error handling and metrics.
        
        This method wraps the process method with:
        - Error handling and recovery
        - Performance metrics collection
        - Logging
        
        Args:
            context: Current pipeline context
            
        Returns:
            Updated pipeline context
        """
        start_time = time.time()
        stage_name = self.__class__.__name__
        
        try:
            self.logger.debug(f"Stage {stage_name} starting for request {context.request_id}")
            self.metrics["executions"] += 1
            
            # Process the stage
            result = await self.process(context)
            
            # Record timing
            duration = time.time() - start_time
            result = result.add_timing(stage_name, duration)
            
            self.metrics["successes"] += 1
            self.logger.debug(f"Stage {stage_name} completed in {duration:.3f}s")
            
            return result
            
        except Exception as e:
            self.metrics["failures"] += 1
            self.logger.error(f"Stage {stage_name} failed: {e}", exc_info=True)
            
            # Add error to context
            context = context.add_error(stage_name, e)
            
            # Determine if we should retry or fail
            if context.retry_count < context.max_retries:
                return context.with_update(
                    status=PipelineStatus.RETRY,
                    retry_count=context.retry_count + 1
                )
            else:
                return context.with_update(status=PipelineStatus.FAILED)
    
    async def handle_error(self, error: Exception, context: PipelineContext) -> PipelineContext:
        """
        Handle errors in a stage-specific way.
        
        Override this method to provide custom error handling.
        
        Args:
            error: The exception that was raised
            context: Current pipeline context
            
        Returns:
            Updated context with error handling applied
        """
        self.logger.error(f"Error in {self.name}: {error}")
        return context.add_error(self.name, error)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get metrics for this stage.
        
        Returns:
            Dictionary of metrics
        """
        return dict(self.metrics)


class ConditionalStage(PipelineStage):
    """
    A pipeline stage that conditionally executes based on a predicate.
    
    This stage will only run its process method if the condition returns True.
    Otherwise, it will skip processing and mark itself as skipped in metadata.
    """
    
    def __init__(self, name: str, condition: Callable[[PipelineContext], bool]):
        """
        Initialize conditional stage.
        
        Args:
            name: Name of this stage
            condition: Function that determines if stage should run
        """
        super().__init__(name)
        self.condition = condition
    
    async def execute(self, context: PipelineContext) -> PipelineContext:
        """
        Execute only if condition is met.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Updated context, possibly with skipped metadata
        """
        if not self.condition(context):
            self.logger.debug(f"Skipping {self.name} - condition not met")
            self.metrics["skipped"] += 1
            return context.with_update(
                metadata={**context.metadata, f"{self.name}_skipped": True}
            )
        return await super().execute(context)


class ParallelStage(PipelineStage):
    """
    A pipeline stage that runs multiple sub-stages in parallel.
    
    This stage executes multiple stages concurrently and merges their results.
    Useful for operations that can be performed independently.
    """
    
    def __init__(self, name: str, stages: list):
        """
        Initialize parallel stage.
        
        Args:
            name: Name of this stage
            stages: List of stages to run in parallel
        """
        super().__init__(name)
        self.stages = stages
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process multiple stages in parallel.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Merged context from all parallel stages
        """
        import asyncio
        
        # Run all stages in parallel
        results = await asyncio.gather(
            *[stage.execute(context) for stage in self.stages],
            return_exceptions=True
        )
        
        # Merge results (simplified - you might want more sophisticated merging)
        merged_context = context
        for result in results:
            if isinstance(result, PipelineContext):
                # Merge certain fields from each result
                if result.service_results:
                    merged_context = merged_context.with_update(
                        service_results={**merged_context.service_results, **result.service_results}
                    )
                if result.metadata:
                    merged_context = merged_context.with_update(
                        metadata={**merged_context.metadata, **result.metadata}
                    )
        
        return merged_context