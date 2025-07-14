# File: chat.py
"""
Manages the main conversation flow of the assistant.

Handles user input (text/audio), interacts with the LLM (via nlp module),
manages conversation history (via memory module), triggers utility services,
and handles speech output/input (via speech module).
"""

import requests
import sys
import os
import time
import json
import traceback
import logging 
import asyncio
from typing import Optional, Dict, Any, Union, List

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

# Speech interface imports
try:
    from interfaces import speech
    try:
        from interfaces import streamaudio
    except ImportError:
        print("Warning: Failed to import Streamaudio Module.")
except ImportError:
    print("Warning: 'interfaces.speech' not found. Using mock speech functions.", file=sys.stderr)
    stream_tts_player_instance = None
    STREAM_TTS_PLAYER_AVAILABLE = False
    
    class MockSpeech:
        def text_to_speech(self, text: str, engine_choice: str = 'default') -> bool:
            print(f"TTS (mock, engine={engine_choice}): {text}")
            return True
        def speech_to_text(self, method: str = 'default') -> str:
            print(f"STT (mock, method={method}): Listening...")
            try:
                return input("You (mock audio input): ").lower()
            except EOFError:
                return "exit"
    speech = MockSpeech()

# Video client import
try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IseeYou")))
    from IseeYou import IseeYouClass
except ImportError:
    print("Warning: Cannot import video client. Video features disabled.", file=sys.stderr)

# Logging setup
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')

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


class ChatManager:
    def __init__(self,
                memory_instance: memory.Memory,
                nlp_instance: nlp.LlpCall,
                config_instance: Any,
                utilities_instance: utilities.Utilities,
                ):
        """
        Initializes the ChatManager.
        """
        logger.info("Initializing ChatManager...")
        self.memory = memory_instance
        self.nlp = nlp_instance
        self.utilities = utilities_instance
        self.config = config_instance
        self.current_prompt: str = ""
        self.current_ai_response: str = ""
        self.format: str = "text"
        
        # Visual context injection
        self.pending_visual_context: Optional[str] = None
        
        if not self.utilities:
            logger.warning("Utilities instance is not available. Service calls will be skipped.")

        try:
            self.format = self._choose_format()
        except Exception as e:
            logger.error(f"Error during initial format selection: {e}. Defaulting to 'text'.")
            self.format = "text"

        logger.info(f"ChatManager initialized with format: {self.format}")

    def _choose_format(self) -> str:
        """Prompts the user to choose interaction format."""
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

    async def _call_llm(self, prompt_to_send: str) -> str:
        """Prepares data and makes the API request to get AI response."""
        logger.info(f"Preparing LLM request for prompt: '{str(prompt_to_send[:50])}...'")
        try:
            # Inject any pending visual context
            if self.pending_visual_context:
                prompt_to_send = f"{self.pending_visual_context}\n\n{prompt_to_send}"
                logger.info(f"Injected visual context: {self.pending_visual_context}")
                self.pending_visual_context = None  # Clear after use
                current_convos = self.memory
                llm_data = data_prep(prompt_to_send, current_convos)
                
            
            # Load current conversation history
            current_convos = self.memory.get_convos()

            # Prepare data
            llm_data = data_prep(prompt_to_send, current_convos)
            if not llm_data:
                logger.error("Failed to prepare data for LLM request.")
                return "Sorry, I couldn't prepare your request due to an internal error."

            # Make API request
            logger.info("Sending request to LLM...")
            start_time = time.monotonic()
            response = await asyncio.to_thread(self.nlp.send_request, llm_data)
            duration = time.monotonic() - start_time
            logger.info(f"LLM response received in {duration:.2f} seconds.")

            # Handle response
            if isinstance(response, str):
                if response.startswith("Blocked:"):
                    logger.warning(f"LLM response blocked by safety settings: {response}")
                    return f"I cannot provide a response due to safety settings ({response})."
                self.current_ai_response = response
                logger.debug(f"LLM Raw Response: '{self.current_ai_response[:100]}...'")
            elif isinstance(response, int):
                logger.error(f"LLM request failed with status code: {response}")
                self.current_ai_response = f"Sorry, I encountered an error (code {response}) while trying to get a response."
            else:
                logger.error(f"Received unexpected response type from nlp.send_request: {type(response)}")
                self.current_ai_response = "Sorry, I received an unexpected response format."

            return self.current_ai_response

        except Exception as e:
            logger.error(f"An unexpected error occurred in _call_llm: {e}", exc_info=True)
            self.current_ai_response = "Sorry, an unexpected error occurred while processing your request."
            return self.current_ai_response

    def _save_current_turn(self) -> None:
        """Save the current user prompt and AI response to memory."""
        if not self.current_prompt or not self.current_ai_response:
            logger.warning("Attempted to save conversation turn with empty prompt or response.")
            return
        try:
            user_saved = self.memory.save_convos("user", self.current_prompt)
            model_saved = self.memory.save_convos("model", self.current_ai_response)
            if not user_saved or not model_saved:
                logger.error("Failed to save one or both parts of the conversation turn to memory.")
        except Exception as e:
            logger.error(f"Error saving conversation turn to memory: {e}", exc_info=True)

    def _speak_response(self) -> None:
        """Handle text-to-speech output using the speech interface."""
        if not self.current_ai_response:
            logger.warning("No AI response available to speak.")
            return
        if self.format != "audio": 
            return

        logger.info("Sending response to TTS engine...")
        try:
            tts_engine = getattr(self.config, 'DEFAULT_TTS_ENGINE', 'pyttsx3')
            logger.info(f"Using TTS engine: {tts_engine}")
            success = speech.text_to_speech(self.current_ai_response, engine_choice=tts_engine)

            if not success:
                logger.error(f"Text-to-speech synthesis failed using engine: {tts_engine}")
            else:
                logger.info("TTS call finished.")
        except Exception as e:
            logger.error(f"Error during text-to-speech call: {e}", exc_info=True)

    async def _handle_service_call(self, ai_response: str) -> Optional[Dict[str, Any]]:
        """Checks for and executes utility service triggers."""
        if not self.utilities:
            return None

        if not hasattr(self.utilities, 'dispatch_service'):
            logger.error("Utilities instance is missing the 'dispatch_service' method.")
            return None

        try:
            service_result_payload = await self.utilities.dispatch_service(ai_response)

            if service_result_payload:
                service_name = service_result_payload.get("service")
                result = service_result_payload.get("result")
                logger.info(f"Service '{service_name}' executed")
                logger.info(f"Result: {str(result)[:200]}")
                
                # Handle visual context service specially
                if service_name == "CHECK_VISUAL_CONTEXT" and isinstance(result, dict):
                    context_string = result.get("context_string")
                    if context_string:
                        # Store context for next prompt instead of making another LLM call
                        self.pending_visual_context = context_string
                        logger.info(f"Visual context stored for next prompt: {context_string}")
                
                return service_result_payload

            return None

        except Exception as e:
            logger.error(f"Error during utility service dispatch: {e}", exc_info=True)
            return None

    async def process_conversation_turn(self, user_input: str) -> Optional[str]:
        """
        Processes a single turn of the conversation.
        """
        self.current_prompt = user_input.strip()
        if not self.current_prompt:
            logger.warning("Received empty user input.")
            return None

        print(f"You: {self.current_prompt}")

        if self.current_prompt.lower() == "exit":
            return "exit"

        # Get AI response (may include injected visual context)
        await self._call_llm(self.current_prompt)
        initial_response = self.current_ai_response

        # Handle errors
        if any(err in initial_response for err in [
            "Sorry, I encountered an error",
            "Sorry, I'm having trouble connecting",
            "Sorry, I received an unexpected response format",
            "I cannot provide a response due to safety settings"
        ]):
            print(f"{self.config.MODEL_NAME}: {initial_response}")
            self._save_current_turn()
            return initial_response

        # Save the turn
        self._save_current_turn()

        # Handle service calls
        service_result_payload = await self._handle_service_call(initial_response)
        
        # For visual context, we don't need to change the response
        # The context is stored for the next turn
        final_response = initial_response
        
        # Special handling for visual context service
        if service_result_payload and service_result_payload.get("service") == "CHECK_VISUAL_CONTEXT":
            # Add a note that context will be used in next response
            if self.pending_visual_context:
                final_response += "\n\n[Visual context has been updated and will be included in my next response.]"
        
        # Print the response
        print(f"{self.config.MODEL_NAME}: {final_response}")
        
        # Handle audio output
        if self.format == "audio":
            self.current_ai_response = final_response
            self._speak_response()
        
        return final_response

    async def _get_audio_input(self) -> Optional[str]:
        """Handles audio input using the configured STT method."""
        logger.info("Listening via STT...")
        try:
            stt_method = getattr(self.config, 'DEFAULT_STT_METHOD', 'whisper_api')
            audio_prompt = await asyncio.to_thread(speech.speech_to_text, method=stt_method)

            if audio_prompt is None:
                logger.warning("STT returned None.")
                return None
            elif not audio_prompt.strip():
                logger.info("No speech detected or STT resulted in empty string.")
                return None
            else:
                return audio_prompt
        except Exception as e:
            logger.error(f"Error during speech recognition: {e}", exc_info=True)
            return None

    async def _get_text_input(self) -> Optional[str]:
        """Handles text input from the console."""
        try:
            text_prompt = await asyncio.to_thread(input, "You: ")
            return text_prompt
        except EOFError:
            logger.info("Input stream closed (EOF). Exiting.")
            return "exit"
        except Exception as e:
            logger.error(f"Error reading text input: {e}", exc_info=True)
            return None

    async def discussion_turn(self) -> Optional[str]:
        """
        Handles one full turn of the discussion based on the chosen format.
        """
        user_input: Optional[str] = None

        if self.format == "text":
            user_input = await self._get_text_input()
        elif self.format == "audio":
            user_input = await self._get_audio_input()
        else:
            logger.error(f"Invalid format '{self.format}'. Defaulting to text.")
            self.format = "text"
            user_input = await self._get_text_input()

        # Process the input if received
        if user_input is not None:
            if user_input == "exit":
                return "exit"
            if user_input.strip():
                ai_response = await self.process_conversation_turn(user_input)
                return ai_response
            else:
                return None
        else:
            return None