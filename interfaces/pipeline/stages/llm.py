"""
LLM handler stage for language model interactions.

Handles API calls to the language model and response processing.
"""

import time
import asyncio
import logging
from typing import Any, Dict, List, Optional

from ..base import PipelineStage
from ..context import PipelineContext, PipelineStatus


# Module-level logger for data_prep function
logger = logging.getLogger(__name__)


def data_prep(prompt: str, convos: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Prepares the data payload for a Gemini API call.

    Args:
        prompt (str): The latest user input.
        convos (Optional[List[Dict[str, Any]]]): Previous conversation messages.

    Returns:
        Dict[str, Any]: Formatted data for Gemini API.
    """
    if not isinstance(prompt, str):
        logger.error("data_prep: Prompt must be a string.")
        return {}

    try:
        # Import config here to avoid circular dependency
        import sys
        import os
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..")))
        import config
        
        convos_formatted = [c for c in convos] if convos else []
        convos_filtered = [c for c in convos_formatted if c.get("role") in ["user", "model"]]
        convos_filtered.append({"role": "user", "parts": [{"text": prompt}]})
        
        temperature = getattr(config, 'TEMPERATURE', 0.72)
        top_p = getattr(config, 'TOP_P', 0.95)
        top_k = getattr(config, 'TOP_K', 40)
        max_tokens = getattr(config, 'MAX_OUTPUT_TOKENS', 8192)
        safety_settings = getattr(config, 'SAFETY_SETTINGS', [])
        system_instruction = getattr(config, 'SYSTEM_PROMPT', None)

        data: Dict[str, Any] = {
            "contents": convos_filtered,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_tokens,
            },
            "safetySettings": safety_settings
        }
        
        if system_instruction:
            data["system_instruction"] = {"parts": [{"text": system_instruction}]}

        return data

    except Exception as e:
        logger.error(f"Error in data_prep: {e}", exc_info=True)
        return {}


class LLMHandler(PipelineStage):
    """
    Handles LLM API calls and response processing.
    
    This stage is responsible for:
    - Sending requests to the language model
    - Processing responses
    - Handling errors and safety blocks
    - Recording response metrics
    """
    
    def __init__(self, nlp_instance: Any):
        """
        Initialize the LLM handler.
        
        Args:
            nlp_instance: NLP/LLM instance for API calls
        """
        super().__init__("LLMHandler")
        self.nlp = nlp_instance
    
    async def process(self, context: PipelineContext) -> PipelineContext:
        """
        Call LLM and get response.
        
        Args:
            context: Current pipeline context
            
        Returns:
            Context with LLM response
        """
        if not context.llm_request_data:
            self.logger.error("No LLM request data available")
            return context.with_update(
                status=PipelineStatus.FAILED,
                errors=[{
                    "stage": "LLMHandler",
                    "message": "No LLM request data"
                }]
            )
        
        if not self.nlp:
            self.logger.error("NLP instance not available")
            return context.with_update(
                status=PipelineStatus.FAILED,
                errors=[{
                    "stage": "LLMHandler",
                    "message": "NLP instance not configured"
                }]
            )
        
        self.logger.info(f"Sending request to LLM for {context.request_id}")
        start_time = time.time()
        
        try:
            # Make API request
            response = await asyncio.to_thread(
                self.nlp.send_request,
                context.llm_request_data
            )
            
            duration = time.time() - start_time
            self.logger.info(f"LLM response received in {duration:.2f}s")
            
            # Handle response based on type
            if isinstance(response, str):
                if response.startswith("Blocked:"):
                    self.logger.warning(f"LLM response blocked: {response}")
                    return context.with_update(
                        llm_response=f"I cannot provide a response due to safety settings ({response}).",
                        llm_status_code=403
                    )
                
                # Successful response
                self.logger.debug(f"LLM response: {response[:100]}...")
                return context.with_update(
                    llm_response=response,
                    llm_status_code=200
                )
                
            elif isinstance(response, int):
                # Error status code
                self.logger.error(f"LLM request failed with status code: {response}")
                return context.with_update(
                    llm_response=f"Sorry, I encountered an error (code {response}) while processing.",
                    llm_status_code=response,
                    status=PipelineStatus.FAILED
                )
                
            else:
                # Unexpected response type
                self.logger.error(f"Unexpected response type from LLM: {type(response)}")
                return context.with_update(
                    llm_response="Sorry, I received an unexpected response format.",
                    llm_status_code=500,
                    status=PipelineStatus.FAILED
                )
                
        except Exception as e:
            self.logger.error(f"Error calling LLM: {e}", exc_info=True)
            return context.with_update(
                llm_response="Sorry, an error occurred while processing your request.",
                llm_status_code=500,
                status=PipelineStatus.FAILED,
                errors=[{
                    "stage": "LLMHandler",
                    "error": str(e)
                }]
            )