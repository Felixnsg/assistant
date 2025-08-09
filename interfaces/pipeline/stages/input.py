"""
Input handler stage for processing user input.

Handles both text and audio input methods.
"""

import asyncio
import logging
from typing import Optional, Any

from ..base import PipelineStage
from ..context import PipelineContext, PipelineStatus


class InputHandler(PipelineStage):
    """
    Handles user input from text or audio sources.
    
    This stage is responsible for:
    - Getting text input from console
    - Getting audio input via STT
    - Validating input
    - Setting up initial context
    """
    
    def __init__(self, speech_module: Optional[Any] = None):
        """
        Initialize the input handler.
        
        Args:
            speech_module: Optional speech module for audio input
        """
        super().__init__("InputHandler")
        self.speech = speech_module
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Process user input based on format.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context with user input populated
        """
        # If we already have user input, skip processing
        if context.user_input:
            self.logger.debug(f"User input already present: {context.user_input[:50]}...")
            return context
        
        if context.input_format == "audio":
            return await self._handle_audio_input(context)
        else:
            return await self._handle_text_input(context)
    
    async def _handle_audio_input(self, context: PipelineContext) -> PipelineContext:
        """
        Handle audio input via speech-to-text.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context with transcribed audio as user input
        """
        if not self.speech:
            self.logger.warning("Audio format selected but speech module not available")
            # Fallback to text input
            return await self._handle_text_input(context)
        
        try:
            # Import config here to avoid circular dependency
            import sys
            import os
            sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
            import config
            
            stt_method = getattr(config, 'DEFAULT_STT_METHOD', 'whisper_api')
            self.logger.info(f"Listening via STT method: {stt_method}")
            
            # Get audio input
            audio_text = await asyncio.to_thread(
                self.speech.speech_to_text,
                method=stt_method
            )
            
            if audio_text and audio_text.strip():
                self.logger.info(f"STT recognized: {audio_text[:50]}...")
                return context.with_update(user_input=audio_text.strip())
            else:
                self.logger.warning("No speech detected or STT failed")
                return context.with_update(
                    status=PipelineStatus.FAILED,
                    errors=[{
                        "stage": "InputHandler",
                        "message": "No speech detected"
                    }]
                )
                
        except Exception as e:
            self.logger.error(f"Error during speech recognition: {e}")
            # Fallback to text input
            return await self._handle_text_input(context)
    
    async def _handle_text_input(self, context: PipelineContext) -> PipelineContext:
        """
        Handle text input from console.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context with text input as user input
        """
        try:
            text_input = await asyncio.to_thread(input, "You: ")
            
            if text_input is None:
                return context.with_update(
                    status=PipelineStatus.FAILED,
                    errors=[{
                        "stage": "InputHandler",
                        "message": "No input received"
                    }]
                )
            
            return context.with_update(user_input=text_input.strip())
            
        except EOFError:
            self.logger.info("Input stream closed (EOF)")
            return context.with_update(user_input="exit")
        except Exception as e:
            self.logger.error(f"Error reading text input: {e}")
            return context.with_update(
                status=PipelineStatus.FAILED,
                errors=[{
                    "stage": "InputHandler",
                    "error": str(e)
                }]
            )