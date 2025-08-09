"""
Pipeline context module for managing data flow through conversation pipeline.

This module provides the immutable context object that flows through all pipeline
stages, carrying request data, state, and metrics.
"""

import time
import uuid
import traceback
from dataclasses import dataclass, field, replace
from typing import Optional, Dict, Any, List
from enum import Enum


class PipelineStatus(Enum):
    """Status codes for pipeline execution."""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRY = "retry"
    SKIPPED = "skipped"


@dataclass
class PipelineContext:
    """
    Immutable context object that flows through the pipeline.
    
    This context encapsulates all state needed for processing a conversation turn.
    Each pipeline stage creates a new context with updates rather than mutating.
    
    Attributes:
        request_id: Unique identifier for this request
        timestamp: Request creation timestamp
        status: Current pipeline status
        user_input: User's text input
        input_format: Format of input ("text" or "audio")
        raw_audio_data: Raw audio data if input is audio
        conversation_history: Previous conversation messages
        visual_context: Visual context from vision system
        system_prompt: System prompt for LLM
        llm_request_data: Prepared data for LLM API
        llm_response: Response from LLM
        llm_status_code: HTTP status from LLM
        detected_services: Services detected in response
        service_results: Results from service execution
        formatted_response: Final formatted response
        should_speak: Whether to use TTS
        errors: List of errors encountered
        retry_count: Number of retries attempted
        max_retries: Maximum retries allowed
        stage_timings: Timing for each stage
        metadata: Additional metadata
    """
    
    # Core identifiers
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: float = field(default_factory=time.time)
    status: PipelineStatus = PipelineStatus.PENDING
    
    # Input data
    user_input: Optional[str] = None
    input_format: str = "text"  # "text" or "audio"
    raw_audio_data: Optional[Any] = None
    
    # Conversation context
    conversation_history: List[Dict[str, Any]] = field(default_factory=list)
    visual_context: Optional[str] = None
    system_prompt: Optional[str] = None
    
    # LLM interaction
    llm_request_data: Optional[Dict[str, Any]] = None
    llm_response: Optional[str] = None
    llm_status_code: Optional[int] = None
    
    # Service execution
    detected_services: List[str] = field(default_factory=list)
    service_results: Dict[str, Any] = field(default_factory=dict)
    
    # Output formatting
    formatted_response: Optional[str] = None
    should_speak: bool = False
    
    # Error tracking
    errors: List[Dict[str, Any]] = field(default_factory=list)
    retry_count: int = 0
    max_retries: int = 3
    
    # Metrics and metadata
    stage_timings: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def with_update(self, **kwargs) -> 'PipelineContext':
        """
        Create a new context with specified updates (maintains immutability).
        
        Args:
            **kwargs: Fields to update
            
        Returns:
            New PipelineContext with updates applied
        """
        return replace(self, **kwargs)
    
    def add_error(self, stage: str, error: Exception) -> 'PipelineContext':
        """
        Add an error to the context.
        
        Args:
            stage: Name of the stage where error occurred
            error: The exception that was raised
            
        Returns:
            New context with error added
        """
        error_entry = {
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "timestamp": time.time(),
            "traceback": traceback.format_exc()
        }
        new_errors = self.errors + [error_entry]
        return self.with_update(errors=new_errors)
    
    def add_timing(self, stage: str, duration: float) -> 'PipelineContext':
        """
        Record timing for a pipeline stage.
        
        Args:
            stage: Name of the stage
            duration: Time taken in seconds
            
        Returns:
            New context with timing added
        """
        new_timings = {**self.stage_timings, stage: duration}
        return self.with_update(stage_timings=new_timings)
    
    def is_failed(self) -> bool:
        """Check if pipeline has failed."""
        return self.status == PipelineStatus.FAILED
    
    def is_completed(self) -> bool:
        """Check if pipeline has completed successfully."""
        return self.status == PipelineStatus.COMPLETED
    
    def get_total_time(self) -> float:
        """Get total time across all stages."""
        return sum(self.stage_timings.values())