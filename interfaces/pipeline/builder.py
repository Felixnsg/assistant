"""
Pipeline builder for easy pipeline construction.

This module provides a fluent builder pattern for constructing pipelines
with proper configuration and dependency injection.
"""

import logging
from typing import Optional, Any

from .orchestrator import PipelineOrchestrator
from .base import PipelineStage


class PipelineBuilder:
    """
    Fluent builder for constructing conversation pipelines.
    
    Provides a clean API for assembling pipeline stages with proper
    configuration and dependency injection.
    
    Example:
        pipeline = (
            PipelineBuilder()
            .with_input_handler(speech_module)
            .with_context_builder(memory)
            .with_llm(nlp_instance)
            .with_services(utilities)
            .with_output_handler(config, speech_module)
            .build()
        )
    """
    
    def __init__(self, max_concurrent: int = 10):
        """
        Initialize the pipeline builder.
        
        Args:
            max_concurrent: Maximum concurrent requests
        """
        self.orchestrator = PipelineOrchestrator(max_concurrent=max_concurrent)
        self.logger = logging.getLogger(f"{__name__}.PipelineBuilder")
        self.config = {}
    
    def with_input_handler(self, speech_module: Optional[Any] = None) -> 'PipelineBuilder':
        """
        Add input handler stage.
        
        Args:
            speech_module: Optional speech module for audio input
            
        Returns:
            Self for method chaining
        """
        # Import here to avoid circular dependencies
        from .stages.input import InputHandler
        
        stage = InputHandler(speech_module=speech_module)
        self.orchestrator.add_stage(stage)
        self.logger.debug("Added InputHandler to pipeline")
        return self
    
    def with_context_builder(self, memory_instance: Any) -> 'PipelineBuilder':
        """
        Add context builder stage.
        
        Args:
            memory_instance: Memory manager instance
            
        Returns:
            Self for method chaining
        """
        from .stages.context_builder import ContextBuilder
        
        stage = ContextBuilder(memory_instance=memory_instance)
        self.orchestrator.add_stage(stage)
        self.logger.debug("Added ContextBuilder to pipeline")
        return self
    
    def with_llm(self, nlp_instance: Any) -> 'PipelineBuilder':
        """
        Add LLM handler stage.
        
        Args:
            nlp_instance: NLP/LLM instance
            
        Returns:
            Self for method chaining
        """
        from .stages.llm import LLMHandler
        
        stage = LLMHandler(nlp_instance=nlp_instance)
        self.orchestrator.add_stage(stage)
        self.logger.debug("Added LLMHandler to pipeline")
        return self
    
    def with_services(self, utilities_instance: Optional[Any] = None) -> 'PipelineBuilder':
        """
        Add service dispatcher stage.
        
        Args:
            utilities_instance: Optional utilities instance
            
        Returns:
            Self for method chaining
        """
        from .stages.service import ServiceDispatcher
        
        if utilities_instance:
            stage = ServiceDispatcher(utilities_instance=utilities_instance)
            self.orchestrator.add_stage(stage)
            self.logger.debug("Added ServiceDispatcher to pipeline")
        else:
            self.logger.debug("Skipped ServiceDispatcher (no utilities provided)")
        return self
    
    def with_response_formatter(self, config_instance: Any) -> 'PipelineBuilder':
        """
        Add response formatter stage.
        
        Args:
            config_instance: Configuration instance
            
        Returns:
            Self for method chaining
        """
        from .stages.response import ResponseFormatter
        
        stage = ResponseFormatter(config_instance=config_instance)
        self.orchestrator.add_stage(stage)
        self.logger.debug("Added ResponseFormatter to pipeline")
        return self
    
    def with_memory_manager(self, memory_instance: Any) -> 'PipelineBuilder':
        """
        Add memory manager stage.
        
        Args:
            memory_instance: Memory manager instance
            
        Returns:
            Self for method chaining
        """
        from .stages.memory import MemoryManager
        
        stage = MemoryManager(memory_instance=memory_instance)
        self.orchestrator.add_stage(stage)
        self.logger.debug("Added MemoryManager to pipeline")
        return self
    
    def with_output_handler(self, config_instance: Any, speech_module: Optional[Any] = None) -> 'PipelineBuilder':
        """
        Add output handler stage.
        
        Args:
            config_instance: Configuration instance
            speech_module: Optional speech module for TTS
            
        Returns:
            Self for method chaining
        """
        from .stages.output import OutputHandler
        
        stage = OutputHandler(
            config_instance=config_instance,
            speech_module=speech_module
        )
        self.orchestrator.add_stage(stage)
        self.logger.debug("Added OutputHandler to pipeline")
        return self
    
    def with_custom_stage(self, stage: PipelineStage) -> 'PipelineBuilder':
        """
        Add a custom pipeline stage.
        
        Args:
            stage: Custom stage instance
            
        Returns:
            Self for method chaining
        """
        self.orchestrator.add_stage(stage)
        self.logger.debug(f"Added custom stage: {stage.name}")
        return self
    
    def with_standard_pipeline(
        self,
        memory_instance: Any,
        nlp_instance: Any,
        config_instance: Any,
        utilities_instance: Optional[Any] = None,
        speech_module: Optional[Any] = None
    ) -> 'PipelineBuilder':
        """
        Configure a standard conversation pipeline with all common stages.
        
        Args:
            memory_instance: Memory manager
            nlp_instance: NLP/LLM instance
            config_instance: Configuration
            utilities_instance: Optional utilities
            speech_module: Optional speech module
            
        Returns:
            Self for method chaining
        """
        return (
            self
            .with_input_handler(speech_module)
            .with_context_builder(memory_instance)
            .with_llm(nlp_instance)
            .with_services(utilities_instance)
            .with_response_formatter(config_instance)
            .with_memory_manager(memory_instance)
            .with_output_handler(config_instance, speech_module)
        )
    
    def build(self) -> PipelineOrchestrator:
        """
        Build and return the configured pipeline orchestrator.
        
        Returns:
            Configured PipelineOrchestrator instance
        """
        self.logger.info(
            f"Built pipeline with {len(self.orchestrator.stages)} stages: "
            f"{[s.name for s in self.orchestrator.stages]}"
        )
        return self.orchestrator
    
    def reset(self) -> 'PipelineBuilder':
        """
        Reset the builder to start fresh.
        
        Returns:
            Self for method chaining
        """
        self.orchestrator.clear_stages()
        self.config.clear()
        self.logger.debug("Reset pipeline builder")
        return self