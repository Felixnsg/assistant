"""
Context builder stage for preparing conversation context.

Builds complete context including conversation history and visual context.
"""

import logging
from typing import Any

from ..base import PipelineStage
from ..context import PipelineContext
from .llm import data_prep


class ContextBuilder(PipelineStage):
    """
    Builds conversation context including history and visual context.
    
    This stage is responsible for:
    - Loading conversation history from memory
    - Injecting visual context if available
    - Preparing LLM request data
    """
    
    def __init__(self, memory_instance: Any):
        """
        Initialize the context builder.
        
        Args:
            memory_instance: Memory manager instance
        """
        super().__init__("ContextBuilder")
        self.memory = memory_instance
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Build complete context for LLM.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context with conversation history and LLM request data
        """
        # Skip if no user input
        if not context.user_input:
            self.logger.warning("No user input to build context for")
            return context
        
        # Load conversation history
        conversation_history = []
        if self.memory:
            try:
                conversation_history = self.memory.get_convos()
                self.logger.debug(f"Loaded {len(conversation_history)} conversation turns")
            except Exception as e:
                self.logger.error(f"Error loading conversation history: {e}")
                # Continue without history rather than failing
        
        # Prepare the prompt with visual context if available
        prompt = context.user_input
        if context.visual_context:
            prompt = f"{context.visual_context}\n\n{prompt}"
            self.logger.info(f"Injected visual context: {context.visual_context[:100]}...")
        
        # Prepare LLM request data
        llm_data = data_prep(prompt, conversation_history)
        
        if not llm_data:
            self.logger.error("Failed to prepare LLM request data")
            return context.with_update(
                errors=[{
                    "stage": "ContextBuilder",
                    "message": "Failed to prepare LLM request data"
                }]
            )
        
        return context.with_update(
            conversation_history=conversation_history,
            llm_request_data=llm_data
        )