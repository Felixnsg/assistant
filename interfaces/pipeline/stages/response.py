"""
Response formatter stage for preparing final responses.

Formats responses based on context and services executed.
"""

import logging
from typing import Any

from ..base import PipelineStage
from ..context import PipelineContext


class ResponseFormatter(PipelineStage):
    """
    Formats the final response based on services and context.
    
    This stage is responsible for:
    - Formatting the final response text
    - Adding visual context notes
    - Determining if TTS should be used
    - Applying any response transformations
    """
    
    def __init__(self, config_instance: Any):
        """
        Initialize the response formatter.
        
        Args:
            config_instance: Configuration instance
        """
        super().__init__("ResponseFormatter")
        self.config = config_instance
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Format the final response.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context with formatted response
        """
        # Use LLM response or fallback message
        response = context.llm_response or "I'm sorry, I couldn't generate a response."
        
        # Add visual context note if applicable
        if context.visual_context and "CHECK_VISUAL_CONTEXT" in context.detected_services:
            response += "\n\n[Visual context has been updated and will be included in my next response.]"
            self.logger.debug("Added visual context note to response")
        
        # Determine if we should use TTS
        should_speak = (context.input_format == "audio")
        
        self.logger.debug(f"Formatted response: {response[:100]}...")
        self.logger.debug(f"Should speak: {should_speak}")
        
        return context.with_update(
            formatted_response=response,
            should_speak=should_speak
        )