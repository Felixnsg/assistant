"""
Pipeline architecture for conversational AI.

This package provides a modular, scalable pipeline pattern for processing
conversations through distinct stages.
"""

from .context import PipelineContext, PipelineStatus
from .base import PipelineStage, ConditionalStage, ParallelStage

__version__ = "1.0.0"

__all__ = [
    # Context
    "PipelineContext",
    "PipelineStatus",
    
    # Base classes
    "PipelineStage",
    "ConditionalStage", 
    "ParallelStage",
]

# Orchestrator and builder
from .orchestrator import PipelineOrchestrator
from .builder import PipelineBuilder

__all__.extend([
    "PipelineOrchestrator",
    "PipelineBuilder",
])