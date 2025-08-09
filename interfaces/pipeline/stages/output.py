"""
Output handler stage for displaying and speaking responses.

Handles console output and text-to-speech.
"""

import asyncio
import logging
from typing import Any, Optional

from ..base import PipelineStage
from ..context import PipelineContext


class OutputHandler(PipelineStage):
    """
    Handles output to console and TTS.
    
    This stage is responsible for:
    - Printing responses to console
    - Triggering TTS for audio output
    - Handling output errors gracefully
    """
    
    def __init__(self, config_instance: Any, speech_module: Optional[Any] = None):
        """
        Initialize the output handler.
        
        Args:
            config_instance: Configuration instance
            speech_module: Optional speech module for TTS
        """
        super().__init__("OutputHandler")
        self.config = config_instance
        self.speech = speech_module
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Output the response.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context unchanged (output is a side effect)
        """
        if not context.formatted_response:
            self.logger.warning("No formatted response to output")
            return context
        
        # Print response to console
        model_name = getattr(self.config, 'MODEL_NAME', 'Assistant')
        print(f"{model_name}: {context.formatted_response}")
        
        # Handle TTS if needed
        if context.should_speak and self.speech:
            await self._handle_tts(context)
        elif context.should_speak and not self.speech:
            self.logger.warning("TTS requested but speech module not available")
        
        return context
    
    async def _handle_tts(self, context: PipelineContext) -> None:
        """
        Handle text-to-speech output.
        
        Args:
            context: Current pipeline context
        """
        try:
            tts_engine = getattr(self.config, 'DEFAULT_TTS_ENGINE', 'pyttsx3')
            self.logger.info(f"Using TTS engine: {tts_engine}")
            
            success = await asyncio.to_thread(
                self.speech.text_to_speech,
                context.formatted_response,
                engine_choice=tts_engine
            )
            
            if success:
                self.logger.debug("TTS completed successfully")
            else:
                self.logger.error(f"TTS failed using engine: {tts_engine}")
                
        except Exception as e:
            self.logger.error(f"Error during TTS: {e}", exc_info=True)