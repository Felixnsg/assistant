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
import asyncio # --- REFACTOR: Added asyncio ---
from typing import Optional, Dict, Any, Union, List # --- REFACTOR: Added typing ---
from interfaces import StreamTTSPlayer

# --- REFACTOR: Ensure project root is in path for sibling imports ---
# Assuming 'core', 'interfaces', 'services' are siblings to the directory containing chat.py
# Or adjust based on your actual project structure
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- REFACTOR: Standard module imports with error handling ---
try:
    from core import nlp # LlpCall class
    from core import memory # Memory class
    import config # config.py
    from IseeYou import IseeYou
    from services import utilities
except ImportError as e:
    print(f"FATAL: Failed to import core modules (nlp, memory, config): {e}", file=sys.stderr)
    sys.exit(1)

# --- REFACTOR: Updated speech/streamaudio imports and mocks ---
try:
    # Use the centralized functions from the refined speech module
    from interfaces import speech # Provides text_to_speech, speech_to_text
    # streamaudio might be integrated into speech or handled differently now.
    # Let's assume a basic wait is needed for audio playback before listening.
    # If streamaudio provided more complex stream management, that logic might need review.
    # from interfaces import streamaudio # Comment out if not strictly needed
    SPEECH_AVAILABLE = True
    # Basic mock for waiting if streamaudio not used/available
    async def wait_until_safe_to_listen(timeout=60):
         # Basic estimate: Wait a bit after TTS finishes.
         # A more sophisticated approach would involve checking audio output status.
         # print("(Mock wait for audio)") # Debug
         await asyncio.sleep(1.0) # Simple fixed delay after TTS finishes
         return True

except ImportError:
    print("Warning: 'interfaces.speech' not found. Using mock speech functions.", file=sys.stderr)
    SPEECH_AVAILABLE = False
    class MockSpeech:
        # Use the refined function names
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
    async def wait_until_safe_to_listen(timeout=60): # Keep mock wait
        print("Audio Wait (mock): Assuming audio finished.")
        await asyncio.sleep(0.1)
        return True


# --- REFACTOR: Standard utilities import ---
try:
    from services.utilities import Utilities # Utilities class
except ImportError as e:
    print(f"Warning: Failed to import Utilities module: {e}. Service calls will not work.", file=sys.stderr)
    # Define a mock Utilities if needed, or let it be None and handle checks later
    Utilities = None

# --- REFACTOR: Standard IseeYou import ---
try:
    from IseeYou.IseeYou import FelixTrackingClient
except ImportError:
    print("Warning: Cannot import FelixTrackingClient from IseeYou.py. Video features disabled.", file=sys.stderr)
    # Mock class for graceful degradation
    class MockFelixTrackingClient:
        def __init__(self, *args, **kwargs):
            print("Warning: Using MockFelixTrackingClient.")
        async def start_tracking(self, *args, **kwargs) -> bool: # Needs to be async
            print("MockFelixTrackingClient start_tracking called.")
            return True
        async def stop_tracking(self, *args, **kwargs) -> bool: # Needs to be async
            print("MockFelixTrackingClient stop_tracking called.")
            return True
        def shutdown(self, *args, **kwargs): # Add shutdown if main calls it
             print("MockFelixTrackingClient shutdown called.")
        # Add any other attributes/methods accessed by ChatManager/Utilities
        # tracked_detections = None

    FelixTrackingClient = MockFelixTrackingClient # type: ignore


# --- REFACTOR: Configure logging ---
# Logging setup might be better in main.py, but adding basic config here
if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')


# --- REFACTOR: Standalone data_prep function (mostly unchanged, added logging) ---
def data_prep(prompt: str, convos: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Formats conversation history and current prompt for the LLM API.

    Specifically structured for Google Generative AI (Gemini) API.

    Args:
        prompt (str): The current user prompt.
        convos (Optional[List[Dict[str, Any]]]): The existing conversation history.

    Returns:
        Dict[str, Any]: The data dictionary ready for the LLM API request,
                        or an empty dict if an error occurs.
    """
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
    """
    Orchestrates the conversation flow, LLM interaction, service execution,
    and user interface (text/audio).

    Attributes:
        memory (Memory): Instance for managing conversation history.
        nlp (LlpCall): Instance for interacting with the LLM.
        utilities (Optional[Utilities]): Instance for executing utility services.
        config (Any): Loaded configuration module.
        isee_client (FelixTrackingClient): Instance for controlling video tracking.
        format (str): Interaction format ('text' or 'audio').
        current_prompt (str): The user input currently being processed.
        current_ai_response (str): The LLM response currently being processed.
    """
    # --- REFACTOR: Updated init signature, type hints, logging ---
    def __init__(self,
                memory_instance: memory.Memory,
                nlp_instance: nlp.LlpCall,
                config_instance: Any,
                utilities_instance: Optional[utilities.Utilities], # Can be None if Utilities failed import
                isee_client_instance: IseeYou.FelixTrackingClient,
                StreamTTSPlayer_instance = StreamTTSPlayer.StreamTTSPlayer):
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
        self.StreamTTSPlayer_instance = StreamTTSPlayer_instance
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
            response = self.nlp.send_request(llm_data) # This is currently synchronous
            duration = time.monotonic() - start_time
            logging.info(f"LLM response received in {duration:.2f} seconds.")

            # --- REFACTOR: Handle different response types from nlp.send_request ---
            if isinstance(response, str):
                 # Check for safety blocking message
                 if response.startswith("Blocked:"):
                      logging.warning(f"LLM response blocked by safety settings: {response}")
                      # Return a user-friendly message or the block reason
                      return f"I cannot provide a response due to safety settings ({response})."
                 # Successful response or empty string
                 self.current_ai_response = response
                 logging.debug(f"LLM Raw Response: '{self.current_ai_response[:100]}...'")
            elif isinstance(response, int): # HTTP status code indicates error
                logging.error(f"LLM request failed with status code: {response}")
                self.current_ai_response = f"Sorry, I encountered an error (code {response}) while trying to get a response."
            else: # Unexpected return type
                 logging.error(f"Received unexpected response type from nlp.send_request: {type(response)}")
                 self.current_ai_response = "Sorry, I received an unexpected response format."

            return self.current_ai_response

        # --- REFACTOR: Catch specific exceptions if nlp.send_request raises them ---
        # except (requests.exceptions.RequestException, TimeoutError, ConnectionError) as e:
        #     logging.error(f"Network or connection error during LLM request: {e}", exc_info=True)
        #     self.current_ai_response = "Sorry, I'm having trouble connecting to the AI service."
        #     return self.current_ai_response
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
        # --- REFACTOR: Renamed, made async, use central speech function ---
        if not self.current_ai_response:
            logging.warning("No AI response available to speak.")
            return
        if not SPEECH_AVAILABLE:
             logging.warning("Speech interface not available, skipping TTS.")
             return
        if self.format != "audio":
             # Don't speak if in text mode
             return

        logging.info("Sending response to TTS engine...")
        try:
            # Use the configured TTS engine from config
            tts_engine = getattr(self.config, 'DEFAULT_TTS_ENGINE', 'pyttsx3')
            success = speech.text_to_speech(self.current_ai_response, engine_choice=tts_engine)
            if not success:
                logging.error(f"Text-to-speech synthesis failed using engine: {tts_engine}")
            else:
                logging.info("TTS playback finished.")
        except Exception as e:
            logging.error(f"Error during text-to-speech call: {e}", exc_info=True)


    async def _handle_service_call(self, ai_response: str):
        """(Internal) Checks for and executes utility service triggers."""
        # --- REFACTOR: Renamed, made async ---
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

                # --- REFACTOR: Handle service results (optional follow-up) ---
                # Option 1: Just log and move on (current implementation)

                # Option 2: Generate a follow-up message based on the result
                # This would likely involve another LLM call or predefined messages.
                # Example: If weather service ran:
                # if service_name == utilities.SERVICE_GET_WEATHER and result:
                #     weather_summary = f"Got the weather for {result['location']}. Current condition is {result['condition']}..."
                #     # Maybe send this summary to LLM for a natural language response?
                #     # followup_prompt = f"Here is the weather data you requested: {json.dumps(result)}. Please summarize it for the user."
                #     # followup_response = await self._call_llm(followup_prompt)
                #     # print(f"{self.config.MODEL_NAME}: {followup_response}")
                #     # await self._speak_response() # Speak the summary
                #     # self._save_current_turn() # Save this extra turn?
                # elif service_name == utilities.SERVICE_START_VIDEO:
                #     if result is True:
                #          # Inform user video started (LLM already did, but confirm action)
                #          print(f"{self.config.MODEL_NAME}: (Action confirmed: Video tracking started)")
                #     else:
                #          print(f"{self.config.MODEL_NAME}: (Action failed: Could not start video tracking)")
                #          # Optionally inform user via TTS

                # For now, keep it simple: The LLM's response *before* the trigger
                # is the primary user feedback. The logs confirm execution.

            # else: # Debug log
                 # logging.debug("No service trigger executed.")

        except Exception as e:
             logging.error(f"Error during utility service dispatch/handling: {e}", exc_info=True)


    async def process_conversation_turn(self, user_input: str) -> Optional[str]:
        """
        Processes a single turn of the conversation (user input -> AI response -> potential service call).

        Args:
            user_input (str): The text input from the user.

        Returns:
            Optional[str]: The AI's final response text for this turn, or "exit" if
                           the user entered exit, or None if input was empty/invalid.
        """
        # --- REFACTOR: Renamed from process_conversation, made async ---
        self.current_prompt = user_input.strip()
        if not self.current_prompt:
            logging.warning("Received empty user input.")
            return None # Indicate no valid input processed

        print(f"You: {self.current_prompt}")

        # Check for local exit command *before* sending to LLM
        if self.current_prompt.lower() == "exit":
            return "exit" # Signal to the main loop to terminate

        # 1. Get AI response (updates self.current_ai_response)
        await self._call_llm(self.current_prompt)

        # Check for errors during LLM call
        if "Sorry, I encountered an error" in self.current_ai_response or \
           "Sorry, I'm having trouble connecting" in self.current_ai_response or \
           "Sorry, I received an unexpected response format" in self.current_ai_response or \
           "I cannot provide a response due to safety settings" in self.current_ai_response:
             # Print the error message from LLM call
             print(f"{self.config.MODEL_NAME}: {self.current_ai_response}")
             # Speak the error if in audio mode
             if self.format == "audio":
                  await self._speak_response()
             # Don't save this turn? Or save the error response? Save for now.
             self._save_current_turn()
             return self.current_ai_response # Return the error response

        # 2. Display the valid AI response (before potential service call modifies context)
        print(f"{self.config.MODEL_NAME}: {self.current_ai_response}")

        # 3. Save this turn (User Prompt + Initial AI Response)
        self._save_current_turn()

        # 4. Speak the initial AI response *before* service execution (gives user feedback faster)
        if self.format == "audio":
             await self._speak_response()
             # Wait for speech to likely finish before potential noisy service actions
             await wait_until_safe_to_listen()

        # 5. Check for and handle service calls based on the AI response
        await self._handle_service_call(self.current_ai_response)

        # 6. Return the initial AI response (or modify if follow-up logic was added in _handle_service_call)
        return self.current_ai_response


    async def _get_audio_input(self) -> Optional[str]:
        """(Internal) Handles audio input using the configured STT method."""
        # --- REFACTOR: Renamed, made async ---
        if not SPEECH_AVAILABLE:
             logging.error("Cannot get audio input: Speech interface not available.")
             return None # Or raise error?

        # Wait for any previous TTS to finish before listening
        logging.info("Waiting for any ongoing TTS to finish before listening...")
        await wait_until_safe_to_listen() # Use the async wait function

        logging.info("Listening via STT...")
        try:
            # --- REFACTOR: Use configured STT method ---
            stt_method = getattr(self.config, 'DEFAULT_STT_METHOD', 'whisper_api') # Add DEFAULT_STT_METHOD to config
            audio_prompt = speech.speech_to_text(method=stt_method) # Use central STT function

            if audio_prompt is None: # Handle None return explicitly
                 logging.warning("STT returned None.")
                 return None
            elif not audio_prompt.strip():
                 logging.info("No speech detected or STT resulted in empty string.")
                 return None # Indicate no turn happened, main loop might reprompt
            else:
                 # Return the recognized text (already lowercase from speech.py)
                 return audio_prompt
        except Exception as e:
            logging.error(f"Error during speech recognition: {e}", exc_info=True)
            return None # Indicate failure

    async def _get_text_input(self) -> Optional[str]:
        """(Internal) Handles text input from the console."""
        # --- REFACTOR: Renamed, made async (though input is sync), added error handling ---
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

        Gets user input (text or audio), processes it, gets AI response,
        handles service calls, and provides output (text or audio).

        Returns:
            Optional[str]: The AI response text, "exit" to signal termination,
                           or None if the turn failed (e.g., no input).
        """
        # --- REFACTOR: Renamed from discussion, made async ---
        user_input: Optional[str] = None

        if self.format == "text":
            user_input = await self._get_text_input()
        elif self.format == "audio":
            user_input = await self._get_audio_input()
        else:
            logging.error(f"Invalid format '{self.format}'. Defaulting to text.")
            self.format = "text"
            user_input = await self._get_text_input()

        # Process the input if received
        if user_input is not None:
             # process_conversation_turn handles None/empty input internally too,
             # but checking here prevents unnecessary call
             if user_input == "exit":
                  return "exit"
             if user_input.strip():
                  # This is the main call that orchestrates the turn
                  ai_response = await self.process_conversation_turn(user_input)
                  return ai_response
             else:
                  # Input was empty or only whitespace
                  return None
        else:
             # Getting input failed or resulted in None
             return None

# --- REFACTOR: Removed old sync methods audio_convos, text_convos ---

# --- END OF REFINED FILE chat.py ---