"""
Memory manager stage for persisting conversations.

Handles saving conversation turns to persistent storage.
"""

import logging
from typing import Any

from ..base import PipelineStage
from ..context import PipelineContext


class MemoryManager(PipelineStage):
    """
    Manages conversation memory persistence.
    
    This stage is responsible for:
    - Saving user input to memory
    - Saving AI responses to memory
    - Handling memory errors gracefully
    """
    
    def __init__(self, memory_instance: Any):
        """
        Initialize the memory manager.
        
        Args:
            memory_instance: Memory manager instance
        """
        super().__init__("MemoryManager")
        self.memory = memory_instance
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Save conversation turn to memory.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context unchanged (memory is a side effect)
        """
        if not self.memory:
            self.logger.debug("No memory instance available")
            return context
        
        if not context.user_input or not context.formatted_response:
            self.logger.debug("Missing user input or response, skipping memory save")
            return context
        
        try:
            # Save user input
            user_saved = self.memory.save_convos("user", context.user_input)
            if not user_saved:
                self.logger.warning("Failed to save user input to memory")
            else:
                self.logger.debug(f"Saved user input: {context.user_input[:50]}...")
            
            # Save AI response
            model_saved = self.memory.save_convos("model", context.formatted_response)
            if not model_saved:
                self.logger.warning("Failed to save AI response to memory")
            else:
                self.logger.debug(f"Saved AI response: {context.formatted_response[:50]}...")
            
            if not user_saved or not model_saved:
                # Add warning but don't fail the pipeline
                return context.with_update(
                    errors=context.errors + [{
                        "stage": "MemoryManager",
                        "message": "Partial memory save failure",
                        "severity": "warning"
                    }]
                )
                
        except Exception as e:
            self.logger.error(f"Error saving to memory: {e}", exc_info=True)
            # Don't fail the pipeline for memory errors
            return context.with_update(
                errors=context.errors + [{
                    "stage": "MemoryManager",
                    "error": str(e),
                    "severity": "warning"
                }]
            )
        
        return context