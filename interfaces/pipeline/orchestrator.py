"""
Pipeline orchestrator for managing stage execution.

This module provides the central orchestrator that coordinates the execution
of pipeline stages with concurrency control, error recovery, and metrics.
"""

import time
import asyncio
import logging
from typing import List, Dict, Any, Optional
from collections import defaultdict

from .context import PipelineContext, PipelineStatus
from .base import PipelineStage


class PipelineOrchestrator:
    """
    Central orchestrator that manages the execution of pipeline stages.
    
    Handles stage composition, execution flow, error recovery, and metrics.
    Designed for high throughput and concurrent request processing.
    
    Attributes:
        stages: List of pipeline stages to execute
        logger: Logger instance
        semaphore: Concurrency limiter
        metrics: Performance metrics
        active_requests: Currently processing requests
        max_concurrent: Maximum concurrent requests
    """
    
    def __init__(self, max_concurrent: int = 100):
        """
        Initialize the pipeline orchestrator.
        
        Args:
            max_concurrent: Maximum number of concurrent requests
        """
        self.stages: List[PipelineStage] = []
        self.logger = logging.getLogger(f"{__name__}.PipelineOrchestrator")
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.metrics = defaultdict(lambda: defaultdict(int))
        self.active_requests: Dict[str, PipelineContext] = {}
        self.max_concurrent = max_concurrent
    
    def add_stage(self, stage: PipelineStage) -> 'PipelineOrchestrator':
        """
        Add a stage to the pipeline.
        
        Args:
            stage: Stage to add
            
        Returns:
            Self for method chaining
        """
        self.stages.append(stage)
        self.logger.info(f"Added stage: {stage.name}")
        return self
    
    def add_stages(self, *stages: PipelineStage) -> 'PipelineOrchestrator':
        """
        Add multiple stages to the pipeline.
        
        Args:
            *stages: Stages to add
            
        Returns:
            Self for method chaining
        """
        for stage in stages:
            self.add_stage(stage)
        return self
    
    def remove_stage(self, stage_name: str) -> bool:
        """
        Remove a stage by name.
        
        Args:
            stage_name: Name of stage to remove
            
        Returns:
            True if stage was removed
        """
        for i, stage in enumerate(self.stages):
            if stage.name == stage_name:
                self.stages.pop(i)
                self.logger.info(f"Removed stage: {stage_name}")
                return True
        return False
    
    def clear_stages(self):
        """Clear all stages from the pipeline."""
        self.stages.clear()
        self.logger.info("Cleared all pipeline stages")
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process a context through all pipeline stages.
        
        This method executes all stages sequentially with:
        - Retry logic with exponential backoff
        - Error recovery
        - Performance metrics
        - Request tracking
        
        Args:
            context: Initial pipeline context
            
        Returns:
            Final processed context
        """
        async with self.semaphore:
            request_id = context.request_id
            self.active_requests[request_id] = context
            
            try:
                start_time = time.time()
                self.logger.info(f"Processing request {request_id}")
                self.metrics["requests"]["total"] += 1
                
                # Execute stages sequentially
                current_context = context.with_update(status=PipelineStatus.PROCESSING)
                
                for stage in self.stages:
                    # Check if we should continue processing
                    if current_context.status in [PipelineStatus.FAILED, PipelineStatus.COMPLETED]:
                        break
                    
                    # Execute stage with retry logic
                    retry_count = 0
                    while retry_count <= current_context.max_retries:
                        stage_context = await stage.execute(current_context)
                        
                        if stage_context.status == PipelineStatus.RETRY:
                            retry_count += 1
                            self.logger.warning(
                                f"Retrying stage {stage.name} for request {request_id} "
                                f"(attempt {retry_count}/{current_context.max_retries})"
                            )
                            # Exponential backoff with max of 10 seconds
                            await asyncio.sleep(min(2 ** retry_count, 10))
                        else:
                            current_context = stage_context
                            break
                    
                    # Check if stage failed after retries
                    if current_context.status == PipelineStatus.FAILED:
                        self.logger.error(f"Stage {stage.name} failed for request {request_id}")
                        self.metrics["stages"][stage.name]["failures"] += 1
                        break
                    else:
                        self.metrics["stages"][stage.name]["successes"] += 1
                
                # Mark as completed if not failed
                if current_context.status != PipelineStatus.FAILED:
                    current_context = current_context.with_update(status=PipelineStatus.COMPLETED)
                
                # Record metrics
                total_time = time.time() - start_time
                status_key = "completed" if current_context.status == PipelineStatus.COMPLETED else "failed"
                self.metrics["requests"][status_key] += 1
                self.metrics["timings"]["total"] = total_time
                self.metrics["timings"]["average"] = (
                    (self.metrics["timings"].get("average", 0) * 
                     (self.metrics["requests"]["total"] - 1) + total_time) / 
                    self.metrics["requests"]["total"]
                )
                
                self.logger.info(
                    f"Request {request_id} {current_context.status.value} in {total_time:.3f}s"
                )
                
                return current_context
                
            except Exception as e:
                self.logger.error(f"Unexpected error processing request {request_id}: {e}", exc_info=True)
                self.metrics["requests"]["errors"] += 1
                return context.with_update(
                    status=PipelineStatus.FAILED,
                    errors=[{"stage": "orchestrator", "error": str(e)}]
                )
                
            finally:
                # Clean up
                self.active_requests.pop(request_id, None)
    
    async def process_batch(self, contexts: List[PipelineContext]) -> List[PipelineContext]:
        """
        Process multiple contexts in parallel.
        
        Args:
            contexts: List of contexts to process
            
        Returns:
            List of processed contexts
        """
        tasks = [self.process(context) for context in contexts]
        return await asyncio.gather(*tasks, return_exceptions=False)
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get current metrics.
        
        Returns:
            Dictionary containing metrics
        """
        stage_metrics = {}
        for stage in self.stages:
            if hasattr(stage, 'get_metrics'):
                stage_metrics[stage.name] = stage.get_metrics()
        
        return {
            "orchestrator": {
                "active_requests": len(self.active_requests),
                "max_concurrent": self.max_concurrent,
                "total_stages": len(self.stages),
                "stage_names": [stage.name for stage in self.stages],
            },
            "requests": dict(self.metrics["requests"]),
            "timings": dict(self.metrics["timings"]),
            "stages": stage_metrics
        }
    
    def reset_metrics(self):
        """Reset all metrics."""
        self.metrics.clear()
        for stage in self.stages:
            if hasattr(stage, 'metrics'):
                stage.metrics.clear()
        self.logger.info("Metrics reset")
    
    def get_active_requests(self) -> List[str]:
        """
        Get list of active request IDs.
        
        Returns:
            List of request IDs currently being processed
        """
        return list(self.active_requests.keys())
    
    def __repr__(self) -> str:
        """String representation of the orchestrator."""
        return (
            f"PipelineOrchestrator("
            f"stages={len(self.stages)}, "
            f"active={len(self.active_requests)}, "
            f"max_concurrent={self.max_concurrent})"
        )