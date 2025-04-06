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
import logging 
import asyncio
from typing import Optional, Dict, Any, Union, List   ##Just for type-hinting 

# Making sure project root is in path for sibling imports because ca m'a cassee les boules cette merde.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Standard module imports with error handling, 
# just for fun and to act fancy in front of other people (I catch error in import I am so better than you)

try:
    from core import nlp
    from core import memory 
    import config 
    from services import utilities
except ImportError as e:
    print(f"FATAL: Failed to import core modules (nlp, memory, config, utilities): {e}", file=sys.stderr)
    sys.exit(1) #exit if the import fails

try:
    from interfaces import speech # Provides text_to_speech, speech_to_text
    try:
        # Assuming the singleton instance is accessible directly
        from interfaces import streamaudio
    except ImportError:
        print("Warning: Failed to import Streamaudio Module. 'alltalk_tts' will be replaced by the other"
        "available engines.")

except ImportError:
    print("Warning: 'interfaces.speech' not found. Using mock speech functions.", file=sys.stderr)
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

## At the time I am commenting this code I don't remenber what I was doing but from what I understand
## If the speech module fails we use a mock/simple version of it directly in chat, I guess...


try:
    from services import utilities     #utilities function
except ImportError as e:
    print(f"Warning: Failed to import Utilities module: {e}. Service calls will not work.", file=sys.stderr)
    Utilities = None

try:
    sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "IseeYou")))
    from IseeYou import IseeYouClass
except ImportError:
    print("Warning: Cannot import FelixTrackingClient from IseeYou.py. Video features disabled.", file=sys.stderr)

##if it fails it fails, no mock class BS my nigga we up. 


if not logging.getLogger().hasHandlers():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - [%(module)s] %(message)s')


# Standalone data_prep function (remains the same) 
def data_prep(prompt: str, convos: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """
    Prepares the data payload for a Gemini API call by formatting the conversation history
    and applying generation configurations.

    Args:
        prompt (str): The latest user input to be added to the conversation.
        convos (Optional[List[Dict[str, Any]]], optional): The list of previous conversation messages,
            each represented as a dictionary with 'role' and 'parts'. Only 'user' and 'model' roles are kept.
            Defaults to None.

    Returns:
        Dict[str, Any]: A dictionary structured to be compatible with the Gemini API,
            containing conversation contents, generation configuration, safety settings,
            and optionally a system instruction.
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
                # we can add other generation config parameters if we need but FLEMME.
            },
            "safetySettings": safety_settings
        }
        # Add system instruction if provided in config
        if system_instruction:
            # For Gemini, system_instruction is typically at the top level
            data["system_instruction"] = {"parts": [{"text": system_instruction}]}

        # logging.debug(f"Prepared LLM request data: {json.dumps(data, indent=2)}") # We just might need more details than ever, we just leave this as a safety net.
        return data

    except Exception as e:
        logging.error(f"Error in data_prep: {e}", exc_info=True)
        return {} # Return empty dict on error


class ChatManager:
    ##The BS class, I hate it with all my bones and muscles
    def __init__(self,
                memory_instance: memory.Memory,
                nlp_instance: nlp.LlpCall,
                config_instance: Any,
                utilities_instance: utilities.Utilities, # Can be None if Utilities failed import, Why I don't remenber but it works so I am not touching it
                ): # Use correct type hint
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
        self.current_prompt: str = ""
        self.current_ai_response: str = ""
        self.format: str = "text"  ## We set it to text as default, because it is my code and I want it to be text.
        if not self.utilities:
            logging.warning("Utilities instance is not available. Service calls will be skipped.") ##This shit was hard to import that is why I have this line

        try:
            self.format = self._choose_format()
        except Exception as e:
            logging.error(f"Error during initial format selection: {e}. Defaulting to 'text'.")
            self.format = "text"

        logging.info(f"ChatManager initialized with format: {self.format}")


    #(_choose_format, _call_llm, _save_current_turn remain the same) ...
    def _choose_format(self) -> str:
        """(Internal) Prompts the user to choose interaction format."""
        while True:  
            try:
                format_choice = input("Choose interaction format (text/audio): ").lower().strip()
                if format_choice in ["text", "audio"]:
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
        logging.info(f"Preparing LLM request for prompt: '{str(prompt_to_send[:50])}...'") ## we send all the prompt here we just wanna print half of it.
        try:
            # Load current conversation history (get convos returns a copy, just for safety reasons)
            current_convos = self.memory.get_convos()

            # Prepare data using the standalone function, dataprep
            llm_data = data_prep(prompt_to_send, current_convos)
            if not llm_data:
                 logging.error("Failed to prepare data for LLM request.")
                 return "Sorry, I couldn't prepare your request due to an internal error." ##Idek why I made it this formal, so future me if you still understand basic python forgive me.

            # Make API request via the NLP instance
            logging.info("Sending request to LLM...")
            start_time = time.monotonic()
            response = await asyncio.to_thread(self.nlp.send_request, llm_data) # uses this as async just in case it takes too much time.
            duration = time.monotonic() - start_time
            logging.info(f"LLM response received in {duration:.2f} seconds.")

            #Handle different response types from nlp.send_request ---
            if isinstance(response, str):
                 if response.startswith("Blocked:"):
                      logging.warning(f"LLM response blocked by safety settings: {response}")
                      return f"I cannot provide a response due to safety settings ({response})."
                 self.current_ai_response = response
                 logging.debug(f"LLM Raw Response: '{self.current_ai_response[:100]}...'")
            elif isinstance(response, int): # HTTP status code indicates error, aahahahah smart move past me.
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
        if not self.current_prompt or not self.current_ai_response:
             logging.warning("Attempted to save conversation turn with empty prompt or response.")
             return
        try:
            user_saved = self.memory.save_convos("user", self.current_prompt)
            model_saved = self.memory.save_convos("model", self.current_ai_response)
            if not user_saved or not model_saved:
                 logging.error("Failed to save one or both parts of the conversation turn to memory.")
            # else: # Debug log
            #      logging.debug("Conversation turn saved to memory.")
        except Exception as e:
            logging.error(f"Error saving conversation turn to memory: {e}", exc_info=True)

    def _speak_response(self) -> None:
        """(Internal) Handle text-to-speech output using the speech interface."""
        if not self.current_ai_response:
            logging.warning("No AI response available to speak.")
            return
        if self.format != "audio": 
             return

        logging.info("Sending response to TTS engine...")
        try:
            # Use the configured TTS engine from config
            tts_engine = getattr(self.config, 'DEFAULT_TTS_ENGINE', 'pyttsx3')
            logging.info(f"Using TTS engine: {tts_engine}")

            # Run potentially blocking TTS in a thread ---
            #success = await asyncio.to_thread(
                #speech.text_to_speech,
                #self.current_ai_response,
                #engine_choice=tts_engine
            #)

            success = speech.text_to_speech(self.current_ai_response, engine_choice = tts_engine)

            if not success:
                logging.error(f"Text-to-speech synthesis failed using engine: {tts_engine}")
            else:
                logging.info("TTS call finished (playback might continue for some engines).")
        except Exception as e:
            logging.error(f"Error during text-to-speech call: {e}", exc_info=True)


    async def _handle_service_call(self, ai_response: str) -> Optional[Dict[str, Any]]:
        """(Internal) Checks for and executes utility service triggers."""
        if not self.utilities:
            return None  # No utilities instance to dispatch to

        if not hasattr(self.utilities, 'dispatch_service'):
            logging.error("Utilities instance is missing the 'dispatch_service' method. So services might be unavailable")
            return None

        try:
            service_result_payload = await self.utilities.dispatch_service(ai_response)

            if service_result_payload:
                service_name = service_result_payload.get("service")
                result = service_result_payload.get("result")
                logging.info(f"--- Service '{service_name}' executed ---")
                logging.info(f"Result: {str(result)[:200]}") # Log partial result
                
                # Return the payload to the caller
                return service_result_payload

            return None  # Return None if no service was executed

        except Exception as e:
            logging.error(f"Error during utility service dispatch/handling: {e}", exc_info=True)
            return None  # Return None on error

    async def process_conversation_turn(self, user_input: str) -> Optional[str]:
        """
        Processes a single turn of the conversation (user input -> AI response -> potential service call).
        """
        self.current_prompt = user_input.strip()
        if not self.current_prompt:
            logging.warning("Received empty user input.")
            return None

        print(f"You: {self.current_prompt}") # Print user input

        if self.current_prompt.lower() == "exit":
            return "exit"

        # 1. Get INITIAL AI response
        await self._call_llm(self.current_prompt)
        initial_response = self.current_ai_response # Store the first response

        # --- Handle initial LLM call errors (as before) ---
        if "Sorry, I encountered an error" in initial_response or \
           "Sorry, I'm having trouble connecting" in initial_response or \
           "Sorry, I received an unexpected response format" in initial_response or \
           "I cannot provide a response due to safety settings" in initial_response:
            print(f"{self.config.MODEL_NAME}: {initial_response}") # Print error message
            # ... (handle audio if needed) ...
            # Save the user prompt and the error response to history
            self._save_current_turn() # Uses self.current_prompt and self.current_ai_response (which is initial_response here)
            return initial_response
        # -------------------------------------------------

        # --- Save the initial turn (user prompt + initial AI response containing trigger) ---
        # Save this BEFORE the service call potentially changes the response for display/memory
        self._save_current_turn() # Uses self.current_prompt and self.current_ai_response
        # ----------------------------------------------------------------------------------

        # --- Handle service calls ---
        service_result_payload = await self._handle_service_call(initial_response)
        # ----------------------------

        # --- Determine the final response to display and potentially add to memory ---
        final_response_to_print = initial_response # Default to the initial response

        if service_result_payload:
            service_name = service_result_payload.get("service")
            service_result = service_result_payload.get("result") # Expect dict for visual context
            
            from services.utilities import SERVICE_CHECK_VISUAL_CONTEXT
            # --- UPDATED BLOCK: Handle the dictionary result cleanly ---
            # Use constants like SERVICE_CHECK_VISUAL_CONTEXT if defined
            if service_name == SERVICE_CHECK_VISUAL_CONTEXT and isinstance(service_result, dict):
                context_prompt = service_result.get("context_prompt")
                context_response = service_result.get("context_response")
                logging.debug(f"--- Debug Check ---")
                logging.debug(f"Received service_name: '{service_name}' (Type: {type(service_name)})")

                logging.debug(f"Is service_name == 'SERVICE_CHECK_VISUAL_CONTEXT'? {service_name == 'SERVICE_CHECK_VISUAL_CONTEXT'}")
        # If using an imported constant, check that too:
        # from services.utilities import SERVICE_CHECK_VISUAL_CONTEXT # (Assuming import at top)
        # logging.debug(f"Is service_name == imported constant? {service_name == SERVICE_CHECK_VISUAL_CONTEXT}")
                logging.debug(f"Received service_result type: {type(service_result)}")
                logging.debug(f"Is service_result a dict? {isinstance(service_result, dict)}")
                if isinstance(service_result, dict):
                    logging.debug(f"service_result keys: {service_result.keys()}")
                    context_response_debug = service_result.get("context_response")
                    logging.debug(f"context_response value from dict: '{str(context_response_debug)[:50]}...'")
                    logging.debug(f"Is context_response truthy? {bool(context_response_debug)}")
                logging.debug(f"--- End Debug Check ---")
                # --- Check if we got a valid response from the context check ---
                if context_response:
                    final_response_to_print = context_response # Use this for display
                    logging.info(f"Using AI response from visual context check: '{context_response[:100]}...'")

                    # --- Save the context interaction to memory ---
                    if context_prompt:
                         logging.info("Saving context prompt to memory (as system)")
                         self.memory.save_convos("system", context_prompt) # Save context prompt
                    logging.info(f"Saving visual context response to memory: '{context_response[:50]}...'")
                    self.memory.save_convos("model", context_response) # Save context response
                    # ---------------------------------------------
                else:
                    # Log if service ran but didn't get a usable response back from LLM
                    logging.warning("CHECK_VISUAL_CONTEXT service ran but LLM returned no valid context_response.")
                    # Keep final_response_to_print as initial_response

            # --- Add elif for other services if they modify the response ---
            # elif service_name == "SERVICE_GET_WEATHER" and isinstance(service_result, dict):
            #    # Format weather data and maybe set final_response_to_print
            #    pass
            # ----------------------------------------------------------

        # --- Print ONLY the final determined response ---
        print(f"{self.config.MODEL_NAME}: {final_response_to_print}") # Only print here
        # ---------------------------------------------

        # --- Handle audio output for the final response (if enabled) ---
        if self.format == "audio":
             self.current_ai_response = final_response_to_print # Update internal state for speaker
             self._speak_response()
             # await self._wait_after_tts() # If needed
        # ----------------------------------------------------------

        # --- Return the final determined response ---
        return final_response_to_print
        # ------------------------------------------
    # --- NEW: Dedicated wait function ---
    
    async def _get_audio_input(self) -> Optional[str]:
        """(Internal) Handles audio input using the configured STT method."""

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