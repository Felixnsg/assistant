# File: chat.py
# --- REFACTOR: Added file description ---
"""
Manages the main conversation flow of the assistant.

Handles user input (text/audio), interacts with the LLM (via nlp module),
manages conversation history (via memory module), triggers utility services,
and handles speech output/input (via speech module).
"""

import requests # Keep for now, might be needed by mocks or future checks
import sys
import os
import time
import json
import traceback
import logging # --- REFACTOR: Added logging ---
# File: chat.py
# ... (other imports) ...
import asyncio # --- REFACTOR: Added asyncio ---
from typing import Optional, Dict, Any, Union, List # --- REFACTOR: Added typing ---

# --- REFACTOR: Ensure project root is in path for sibling imports ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- REFACTOR: Standard module imports with error handling ---
try:
    from core import nlp # LlpCall class
    from core import memory # Memory class
    import config # config.py
    # from IseeYou import IseeYou # Keep this if needed, already handled below
    from services import utilities
except ImportError as e:
    print(f"FATAL: Failed to import core modules (nlp, memory, config, utilities): {e}", file=sys.stderr)
    sys.exit(1)

# --- REFACTOR: Updated speech imports and mocks ---
try:
    # Use the centralized functions from the refined speech module
    from interfaces import speech # Provides text_to_speech, speech_to_text
    # --- NEW: Import the StreamTTSPlayer instance ---
    try:
        # Assuming the singleton instance is accessible directly
        from interfaces.StreamTTSPlayer import _player_instance as stream_tts_player_instance
        STREAM_TTS_PLAYER_AVAILABLE = True
    except ImportError:
        print("Warning: Failed to import StreamTTSPlayer instance. 'alltalk_tts' wait logic will be skipped.")
        stream_tts_player_instance = None
        STREAM_TTS_PLAYER_AVAILABLE = False

    SPEECH_AVAILABLE = True
    # --- REMOVE old mock wait_until_safe_to_listen ---
    # async def wait_until_safe_to_listen(timeout=60):
    #      await asyncio.sleep(1.0) # Simple fixed delay after TTS finishes
    #      return True

except ImportError:
    print("Warning: 'interfaces.speech' not found. Using mock speech functions.", file=sys.stderr)
    SPEECH_AVAILABLE = False
    stream_tts_player_instance = None # Ensure it's None if speech failed import
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
    # --- REMOVE old mock wait_until_safe_to_listen ---


# --- REFACTOR: Standard utilities import ---
try:
    from services.utilities import Utilities # Utilities class
except ImportError as e:
    print(f"Warning: Failed to import Utilities module: {e}. Service calls will not work.", file=sys.stderr)
    Utilities = None

# --- REFACTOR: Standard IseeYou import ---
try:
    # Corrected path assuming IseeYou.py is in ../IseeYou/
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IseeYou")))
    from IseeYou import IseeYou
except ImportError:
    print("Warning: Cannot import FelixTrackingClient from IseeYou.py. Video features disabled.", file=sys.stderr)
    # Mock class for graceful degradation
    class MockFelixTrackingClient:
        def __init__(self, *args, **kwargs):
            print("Warning: Using MockFelixTrackingClient.")
        async def start_tracking(self, *args, **kwargs) -> bool:
            print("MockFelixTrackingClient start_tracking called.")
            return True
        async def stop_tracking(self, *args, **kwargs) -> bool:
            print("MockFelixTrackingClient stop_tracking called.")
            return True
        def shutdown(self, *args, **kwargs):
             print("MockFelixTrackingClient shutdown called.")
        FelixTrackingClient = MockFelixTrackingClient # type: ignore


# --- REFACTOR: Configure logging ---
# (logging setup remains the same)
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')


# --- REFACTOR: Standalone data_prep function (remains the same) ---
def data_prep(prompt: str, convos: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    # ... (data_prep content remains unchanged) ...
    if not isinstance(prompt, str):
        logging.error("data_prep: Prompt must be a string.")
        return {}

    try:
        # Start with a copy of the existing conversation, or an empty list
        convos_formatted = [c for c in convos] if convos else [] # Ensure deep copy isn't needed here

        # Add the current user prompt
        # Ensure roles are 'user' and 'model' as expected by Gemini API
        # Filter out any potential 'system' roles if they exist in history
        convos_filtered = [c for c in convos_formatted if c.get("role") in ["user", "model"]]

        # Add current prompt
        convos_filtered.append({"role": "user", "parts": [{"text": prompt}]})

        # --- REFACTOR: Ensure config attributes exist or use defaults ---
        temperature = getattr(config, 'TEMPERATURE', 0.72)
        top_p = getattr(config, 'TOP_P', 0.95)
        top_k = getattr(config, 'TOP_K', 40)
        max_tokens = getattr(config, 'MAX_OUTPUT_TOKENS', 8192)
        safety_settings = getattr(config, 'SAFETY_SETTINGS', [])
        system_instruction = getattr(config, 'SYSTEM_PROMPT', None) # Get system prompt

        # --- REFACTOR: Structure for Gemini API (with optional system instruction) ---
        data: Dict[str, Any] = {
            "contents": convos_filtered,
            "generationConfig": {
                "temperature": temperature,
                "topP": top_p,
                "topK": top_k,
                "maxOutputTokens": max_tokens,
                # Add other generation config parameters as needed
            },
            "safetySettings": safety_settings
        }
        # Add system instruction if provided in config
        if system_instruction:
            # For Gemini, system_instruction is typically at the top level
            data["system_instruction"] = {"parts": [{"text": system_instruction}]}

        # logging.debug(f"Prepared LLM request data: {json.dumps(data, indent=2)}") # Very verbose
        return data

    except Exception as e:
        logging.error(f"Error in data_prep: {e}", exc_info=True)
        return {} # Return empty dict on error


class ChatManager:
    # ... (__init__ remains the same) ...
    def __init__(self,
                memory_instance: memory.Memory,
                nlp_instance: nlp.LlpCall,
                config_instance: Any,
                utilities_instance: Optional[utilities.Utilities], # Can be None if Utilities failed import
                isee_client_instance: IseeYou): # Use correct type hint
        """
        Initializes the ChatManager.

        Args:
            memory_instance (memory.Memory): Instance of the Memory class.
            nlp_instance (nlp.LlpCall): Instance of the LlpCall class.
            config_instance (Any): The loaded config module.
            utilities_instance (Optional[Utilities]): Instance of the Utilities class.
            isee_client_instance (FelixTrackingClient): Instance of the FelixTrackingClient.
        """
        logging.info("Initializing ChatManager...")
        self.memory = memory_instance
        self.nlp = nlp_instance
        self.utilities = utilities_instance
        self.config = config_instance
        # --- REFACTOR: Renamed FelixTrackingClient attribute ---
        self.isee_client = isee_client_instance # Store the client instance
        # --- REFACTOR: Initialize attributes ---
        self.current_prompt: str = ""
        self.current_ai_response: str = ""
        self.format: str = "text" # Default, will be set by choose_format

        if not self.utilities:
            logging.warning("Utilities instance is not available. Service calls will be skipped.")

        try:
            # --- REFACTOR: Choose format during initialization ---
            self.format = self._choose_format()
        except Exception as e:
            logging.error(f"Error during initial format selection: {e}. Defaulting to 'text'.")
            self.format = "text"

        logging.info(f"ChatManager initialized with format: {self.format}")


    # ... (_choose_format, _call_llm, _save_current_turn remain the same) ...
    def _choose_format(self) -> str:
        """(Internal) Prompts the user to choose interaction format."""
        while True:
            try:
                # --- REFACTOR: Handle potential EOFError during input ---
                format_choice = input("Choose interaction format (text/audio): ").lower().strip()
                if format_choice in ["text", "audio"]:
                    if format_choice == "audio" and not SPEECH_AVAILABLE:
                         print("Audio format selected, but speech interface is not available. Please check dependencies (PyAudio, etc.)")
                         print("Defaulting to text format.")
                         return "text"
                    logging.info(f"Interaction format set to: {format_choice}")
                    return format_choice
                else:
                    print("Invalid choice. Please enter 'text' or 'audio'.")
            except EOFError:
                 logging.warning("EOF received during format selection. Defaulting to 'text'.")
                 return "text"
            except Exception as e:
                 logging.error(f"Error during format input: {e}. Defaulting to 'text'.")
                 return "text"

    async def _call_llm(self, prompt_to_send: str) -> str:
        """(Internal) Prepares data and makes the API request to get AI response."""
        # --- REFACTOR: Renamed, made async (though LLM call itself is sync here), improved error handling ---
        logging.debug(f"Preparing LLM request for prompt: '{prompt_to_send[:50]}...'")
        try:
            # Load current conversation history
            # --- REFACTOR: Use get_convos() which returns a copy ---
            current_convos = self.memory.get_convos()

            # Prepare data using the standalone function
            llm_data = data_prep(prompt_to_send, current_convos)
            if not llm_data:
                 logging.error("Failed to prepare data for LLM request.")
                 return "Sorry, I couldn't prepare your request due to an internal error."

            # Make API request via the NLP instance
            logging.info("Sending request to LLM...")
            start_time = time.monotonic()
            # --- Run synchronous call in thread ---
            response = await asyncio.to_thread(self.nlp.send_request, llm_data)
            duration = time.monotonic() - start_time
            logging.info(f"LLM response received in {duration:.2f} seconds.")

            # --- REFACTOR: Handle different response types from nlp.send_request ---
            if isinstance(response, str):
                 if response.startswith("Blocked:"):
                      logging.warning(f"LLM response blocked by safety settings: {response}")
                      return f"I cannot provide a response due to safety settings ({response})."
                 self.current_ai_response = response
                 logging.debug(f"LLM Raw Response: '{self.current_ai_response[:100]}...'")
            elif isinstance(response, int): # HTTP status code indicates error
                logging.error(f"LLM request failed with status code: {response}")
                self.current_ai_response = f"Sorry, I encountered an error (code {response}) while trying to get a response."
            else: # Unexpected return type
                 logging.error(f"Received unexpected response type from nlp.send_request: {type(response)}")
                 self.current_ai_response = "Sorry, I received an unexpected response format."

            return self.current_ai_response

        except Exception as e:
             logging.error(f"An unexpected error occurred in _call_llm: {e}", exc_info=True)
             self.current_ai_response = "Sorry, an unexpected error occurred while processing your request."
             return self.current_ai_response

    def _save_current_turn(self) -> None:
        """(Internal) Save the current user prompt and AI response to memory."""
        # --- REFACTOR: Renamed, added checks ---
        if not self.current_prompt or not self.current_ai_response:
             logging.warning("Attempted to save conversation turn with empty prompt or response.")
             return
        try:
            # --- REFACTOR: Check return value of save_convos ---
            user_saved = self.memory.save_convos("user", self.current_prompt)
            model_saved = self.memory.save_convos("model", self.current_ai_response)
            if not user_saved or not model_saved:
                 logging.error("Failed to save one or both parts of the conversation turn to memory.")
            # else: # Debug log
            #      logging.debug("Conversation turn saved to memory.")
        except Exception as e:
            logging.error(f"Error saving conversation turn to memory: {e}", exc_info=True)

    async def _speak_response(self) -> None:
        """(Internal) Handle text-to-speech output using the speech interface."""
        if not self.current_ai_response:
            logging.warning("No AI response available to speak.")
            return
        if not SPEECH_AVAILABLE:
             logging.warning("Speech interface not available, skipping TTS.")
             return
        if self.format != "audio":
             return

        logging.info("Sending response to TTS engine...")
        try:
            # Use the configured TTS engine from config
            tts_engine = getattr(self.config, 'DEFAULT_TTS_ENGINE', 'pyttsx3')
            logging.info(f"Using TTS engine: {tts_engine}")

            # --- Run potentially blocking TTS in a thread ---
            success = await asyncio.to_thread(
                speech.text_to_speech,
                self.current_ai_response,
                engine_choice=tts_engine
            )

            if not success:
                logging.error(f"Text-to-speech synthesis failed using engine: {tts_engine}")
            else:
                logging.info("TTS call finished (playback might continue for some engines).")
        except Exception as e:
            logging.error(f"Error during text-to-speech call: {e}", exc_info=True)


    async def _handle_service_call(self, ai_response: str):
        """(Internal) Checks for and executes utility service triggers."""
        # ... (handle_service_call content remains unchanged) ...
        if not self.utilities:
            # logging.debug("Utilities not available, skipping service check.") # Debug
            return # No utilities instance to dispatch to

        if not hasattr(self.utilities, 'dispatch_service'):
            logging.error("Utilities instance is missing the 'dispatch_service' method.")
            return

        try:
            # --- REFACTOR: Call the async dispatch_service ---
            service_result_payload = await self.utilities.dispatch_service(ai_response)

            if service_result_payload:
                service_name = service_result_payload.get("service")
                result = service_result_payload.get("result")
                logging.info(f"--- Service '{service_name}' executed ---")
                logging.info(f"Result: {str(result)[:200]}") # Log partial result

            # else: # Debug log
                 # logging.debug("No service trigger executed.")

        except Exception as e:
             logging.error(f"Error during utility service dispatch/handling: {e}", exc_info=True)


    async def process_conversation_turn(self, user_input: str) -> Optional[str]:
        """
        Processes a single turn of the conversation (user input -> AI response -> potential service call).
        """
        self.current_prompt = user_input.strip()
        if not self.current_prompt:
            logging.warning("Received empty user input.")
            return None

        print(f"You: {self.current_prompt}")

        if self.current_prompt.lower() == "exit":
            return "exit"

        # 1. Get AI response
        await self._call_llm(self.current_prompt)

        # Handle errors during LLM call
        if "Sorry, I encountered an error" in self.current_ai_response or \
           "Sorry, I'm having trouble connecting" in self.current_ai_response or \
           "Sorry, I received an unexpected response format" in self.current_ai_response or \
           "I cannot provide a response due to safety settings" in self.current_ai_response:
            print(f"{self.config.MODEL_NAME}: {self.current_ai_response}")
            if self.format == "audio":
                await self._speak_response() # Speak the error
                # --- NEW: Add wait after speaking error ---
                await self._wait_after_tts()
            self._save_current_turn()
            return self.current_ai_response

        # 2. Display valid AI response
        print(f"{self.config.MODEL_NAME}: {self.current_ai_response}")

        # 3. Save turn
        self._save_current_turn()

        # 4. Speak the initial AI response
        if self.format == "audio":
            await self._speak_response()
            # --- NEW: Call the dedicated wait function ---
            await self._wait_after_tts()

        # 5. Handle service calls
        await self._handle_service_call(self.current_ai_response)

        # 6. Return response
        return self.current_ai_response

    # --- NEW: Dedicated wait function ---
    async def _wait_after_tts(self, timeout_sec: float = 60.0):
        """Waits appropriately after TTS, especially for streaming engines."""
        tts_engine = getattr(self.config, 'DEFAULT_TTS_ENGINE', 'pyttsx3')
        logging.debug(f"Waiting after TTS engine: {tts_engine}")

        if tts_engine == 'alltalk_tts' and STREAM_TTS_PLAYER_AVAILABLE and stream_tts_player_instance:
            logging.info(f"Waiting for StreamTTSPlayer playback (timeout={timeout_sec}s)...")
            try:
                # Run the player's synchronous wait method in a thread
                success = await asyncio.to_thread(
                    stream_tts_player_instance.wait_until_safe_to_listen,
                    timeout=timeout_sec
                )
                if success:
                    logging.info("StreamTTSPlayer playback finished.")
                else:
                    logging.warning("Timeout or error waiting for StreamTTSPlayer playback.")
            except Exception as e:
                logging.error(f"Error waiting for StreamTTSPlayer: {e}", exc_info=True)
        else:
            # Fallback for other engines (or if StreamTTSPlayer not available)
            # Use a simple sleep, adjust duration as needed. 1.0s might be short for some offline engines.
            wait_duration = 1.5 # Slightly longer default wait
            logging.debug(f"Using generic wait of {wait_duration}s after TTS.")
            await asyncio.sleep(wait_duration)


    async def _get_audio_input(self) -> Optional[str]:
        """(Internal) Handles audio input using the configured STT method."""
        if not SPEECH_AVAILABLE:
            logging.error("Cannot get audio input: Speech interface not available.")
            return None

        # --- Note: The wait is now handled in process_conversation_turn *after* TTS ---
        # logging.info("Waiting for any ongoing TTS to finish before listening...")
        # await wait_until_safe_to_listen() # <= REMOVED FROM HERE

        logging.info("Listening via STT...")
        try:
            stt_method = getattr(self.config, 'DEFAULT_STT_METHOD', 'whisper_api')
            # --- Run potentially blocking STT in a thread ---
            audio_prompt = await asyncio.to_thread(speech.speech_to_text, method=stt_method)

            if audio_prompt is None:
                 logging.warning("STT returned None.")
                 return None
            elif not audio_prompt.strip():
                 logging.info("No speech detected or STT resulted in empty string.")
                 return None
            else:
                 return audio_prompt # Already lowercase from speech.py
        except Exception as e:
            logging.error(f"Error during speech recognition: {e}", exc_info=True)
            return None

    async def _get_text_input(self) -> Optional[str]:
        """(Internal) Handles text input from the console."""
        # ... (_get_text_input remains the same) ...
        try:
            text_prompt = await asyncio.to_thread(input, "You: ") # Run sync input in thread
            return text_prompt
        except EOFError:
             logging.info("Input stream closed (EOF). Exiting.")
             return "exit" # Signal exit if input stream ends
        except Exception as e:
             logging.error(f"Error reading text input: {e}", exc_info=True)
             return None


    async def discussion_turn(self) -> Optional[str]:
        """
        Handles one full turn of the discussion based on the chosen format.
        """
        # ... (discussion_turn logic remains largely the same, calls the updated methods) ...
        user_input: Optional[str] = None

        if self.format == "text":
            user_input = await self._get_text_input()
        elif self.format == "audio":
            # --- Wait happens *after* TTS in process_conversation_turn ---
            # --- Now directly call STT ---
            user_input = await self._get_audio_input()
        else:
            logging.error(f"Invalid format '{self.format}'. Defaulting to text.")
            self.format = "text"
            user_input = await self._get_text_input()

        # Process the input if received
        if user_input is not None:
             if user_input == "exit":
                  return "exit"
             if user_input.strip():
                  # This call now handles the correct waiting internally
                  ai_response = await self.process_conversation_turn(user_input)
                  return ai_response
             else:
                  return None
        else:
             return None

# --- END OF REFINED FILE chat.py ---