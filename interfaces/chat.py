"""
Production-ready pipeline-based conversation manager.

This module provides the main ChatManager class that orchestrates conversations
using a modular pipeline architecture. The pipeline processes conversations through
distinct stages: Input → Context → LLM → Response → Service → Memory → Output.
"""

import sys
import os
import logging
from typing import Optional, Any, Dict

# Path setup
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Core imports
try:
    from core import nlp
    from core import memory 
    import config 
    from services import utilities
except ImportError as e:
    print(f"FATAL: Failed to import core modules: {e}", file=sys.stderr)
    sys.exit(1)

# Pipeline imports
from interfaces.pipeline import PipelineBuilder, PipelineContext

# Logging setup
if not logging.getLogger().hasHandlers():
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s'
    )

logger = logging.getLogger(__name__)


class ChatManager:
    """
    Modern pipeline-based conversation manager.
    
    Uses a modular, scalable pipeline architecture where each conversation
    turn flows through distinct, testable stages. This replaces the monolithic
    design with a clean separation of concerns.
    
    Attributes:
        memory: Memory manager for conversation history
        nlp: LLM API handler
        utilities: Utility services handler
        config: Configuration object
        format: Interaction format (text/audio)
        pipeline: Pipeline orchestrator
        visual_context_cache: Cached visual context for next turn
    """
    
    def __init__(self,
                 memory_instance: memory.Memory,
                 nlp_instance: nlp.LlpCall,
                 config_instance: Any,
                 utilities_instance: utilities.Utilities):
        """
        Initialize the pipeline-based ChatManager.
        
        Args:
            memory_instance: Memory manager for conversation history
            nlp_instance: LLM API handler
            config_instance: Configuration object
            utilities_instance: Utility services handler
        """
        logger.info("Initializing Pipeline-based ChatManager...")
        
        # Store dependencies
        self.memory = memory_instance
        self.nlp = nlp_instance
        self.utilities = utilities_instance
        self.config = config_instance
        
        # Conversation state
        self.format: str = "text"
        self.visual_context_cache: Optional[str] = None
        
        # Import speech module with fallback
        try:
            from interfaces import speech
            self.speech_module = speech
        except ImportError:
            logger.warning("Speech module not available. Audio features disabled.")
            self.speech_module = None
        
        # Build the pipeline using the builder pattern
        self.pipeline = (
            PipelineBuilder()
            .with_standard_pipeline(
                memory_instance=self.memory,
                nlp_instance=self.nlp,
                config_instance=self.config,
                utilities_instance=self.utilities,
                speech_module=self.speech_module
            )
            .build()
        )
        
        # Choose interaction format
        try:
            self.format = self._choose_format()
        except Exception as e:
            logger.error(f"Error during format selection: {e}. Defaulting to 'text'.")
            self.format = "text"
        
        logger.info(f"Pipeline ChatManager initialized with format: {self.format}")
        logger.info(f"Pipeline has {len(self.pipeline.stages)} stages")
    
    def _choose_format(self) -> str:
        """
        Prompts the user to choose interaction format.
        
        Returns:
            Selected format: "text" or "audio"
        """
        while True:
            try:
                format_choice = input("Choose interaction format (text/audio): ").lower().strip()
                if format_choice in ["text", "audio"]:
                    logger.info(f"Interaction format set to: {format_choice}")
                    return format_choice
                else:
                    print("Invalid choice. Please enter 'text' or 'audio'.")
            except EOFError:
                logger.warning("EOF received during format selection. Defaulting to 'text'.")
                return "text"
            except Exception as e:
                logger.error(f"Error during format input: {e}. Defaulting to 'text'.")
                return "text"
    
    async def process_conversation_turn(self, user_input: str) -> Optional[str]:
        """
        Process a single conversation turn through the pipeline.
        
        Args:
            user_input: The user's input text
            
        Returns:
            The AI's response or None if processing failed
        """
        if not user_input or not user_input.strip():
            logger.warning("Received empty user input")
            return None
        
        user_input = user_input.strip()
        print(f"You: {user_input}")
        
        # Handle exit command
        if user_input.lower() == "exit":
            return "exit"
        
        # Create pipeline context
        context = PipelineContext(
            user_input=user_input,
            input_format=self.format,
            visual_context=self.visual_context_cache,
            system_prompt=getattr(self.config, 'SYSTEM_PROMPT', None)
        )
        
        # Process through pipeline
        result_context = await self.pipeline.process(context)
        
        # Handle visual context caching for next turn
        if result_context.visual_context:
            self.visual_context_cache = result_context.visual_context
            logger.info("Visual context cached for next turn")
        else:
            # Clear cache if no new visual context
            self.visual_context_cache = None
        
        # Log any errors that occurred
        if result_context.errors:
            for error in result_context.errors:
                if error.get("severity") == "warning":
                    logger.warning(f"Pipeline warning: {error}")
                else:
                    logger.error(f"Pipeline error: {error}")
        
        # Return the formatted response
        return result_context.formatted_response
    
    async def discussion_turn(self) -> Optional[str]:
        """
        Handle one full discussion turn with the appropriate input method.
        
        This method orchestrates getting input and processing it through the pipeline.
        
        Returns:
            The AI response, "exit" to quit, or None if no valid input
        """
        # Create context with format information
        context = PipelineContext(
            input_format=self.format,
            visual_context=self.visual_context_cache,
            system_prompt=getattr(self.config, 'SYSTEM_PROMPT', None)
        )
        
        # Process through pipeline (InputHandler will get the input)
        result_context = await self.pipeline.process(context)
        
        # Check for exit command
        if result_context.user_input and result_context.user_input.lower() == "exit":
            return "exit"
        
        # Handle visual context caching
        if result_context.visual_context:
            self.visual_context_cache = result_context.visual_context
            logger.info("Visual context cached for next turn")
        else:
            self.visual_context_cache = None
        
        # Log performance metrics
        if result_context.stage_timings:
            total_time = result_context.get_total_time()
            logger.info(f"Turn completed in {total_time:.3f}s")
            logger.debug(f"Stage timings: {result_context.stage_timings}")
        
        # Log any errors
        if result_context.errors:
            for error in result_context.errors:
                if error.get("severity") == "warning":
                    logger.warning(f"Pipeline warning: {error}")
                else:
                    logger.error(f"Pipeline error: {error}")
        
        return result_context.formatted_response
    
    def get_metrics(self) -> Dict[str, Any]:
        """
        Get performance metrics from the pipeline.
        
        Returns:
            Dictionary containing pipeline metrics
        """
        return self.pipeline.get_metrics()
    
    def cleanup(self):
        """Clean up resources."""
        logger.info("Cleaning up ChatManager resources...")
        if hasattr(self.pipeline, 'reset_metrics'):
            self.pipeline.reset_metrics()
        logger.info("ChatManager cleanup complete")