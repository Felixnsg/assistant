# File: main.py
"""
Main entry point for the Smart Assistant application.

Initializes all components (NLP, Memory, Vision Client, Chat Manager, Utilities),
starts the background thread for the vision client, and runs the main asynchronous
interaction loop managed by the ChatManager. Handles graceful shutdown.
"""

import threading
import asyncio
import sys
import os
import logging
from typing import Optional
import config

def setup_logging(log_dir="logs", default_level=logging.INFO):
    """Configures file-based logging for different modules."""
    try:
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory '{log_dir}' ensured.") # Keep this initial print
    except OSError as e:
        print(f"Error creating log directory '{log_dir}': {e}. Logs might not be saved.", file=sys.stderr)
        return # Stop if log dir creation fails

    # Define log format
    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )

    # --- Define loggers and their corresponding files ---
    # Using __name__ derived names (e.g., 'core.nlp', 'interfaces.chat')
    log_config = {
        'core.memory': 'memory.log',
        'core.nlp': 'nlp.log',
        'interfaces.chat': 'chat.log',
        'interfaces.speech': 'speech.log',
        'interfaces.StreamTTSPlayer': 'stream_tts_player.log', # Added StreamTTSPlayer
        'services.utilities': 'utilities.log',
        'IseeYou.IseeYou': 'isee_you_client.log',
        'IseeYou.GPUserver': 'gpu_server.log', # Assuming GPUserver.py is in IseeYou/
        'IseeYou.person_detector': 'person_detector.log', # Assuming person_detector.py is in IseeYou/
        'IseeYou.felix_recognizer': 'felix_recognizer.log', # Assuming felix_recognizer.py is in IseeYou/
        # Add Whisper API server if its logs are desired when run via main (less common)
        # 'Whisper': 'whisper_api.log' # Name depends on how it's run/imported
        # Root logger for main.py itself
        '__main__': 'main.log',
    }

    configured_loggers = []

    for logger_name, log_filename in log_config.items():
        try:
            logger = logging.getLogger(logger_name)
            logger.setLevel(default_level)

            # Create file handler
            file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), encoding='utf-8')
            file_handler.setFormatter(log_formatter)

            # Remove existing handlers to avoid duplication if setup is called again
            for handler in logger.handlers[:]:
                logger.removeHandler(handler)

            # Add the new file handler
            logger.addHandler(file_handler)

            # Crucial: Prevent propagation to root logger ---
            logger.propagate = False
            configured_loggers.append(logger_name)

        except Exception as e:
            print(f"Error configuring logger '{logger_name}': {e}", file=sys.stderr)

    # Optional: Configure the root logger minimally if needed, but keep it quiet
    root_logger = logging.getLogger()
    # Remove default handlers from root logger to silence console output from libraries
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)
    # Set root level high so only critical library errors might appear (or set a NullHandler)
    root_logger.setLevel(logging.ERROR)
    # Alternatively, add a NullHandler to explicitly silence it:
    # root_logger.addHandler(logging.NullHandler())

    print(f"Configured file logging for: {', '.join(configured_loggers)}") # Keep this print

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# --- Call logging setup EARLY ---
# Determine log level from config or default
log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
setup_logging(default_level=log_level) # Use configured level

logger = logging.getLogger(__name__) # Use __name__ for the current module



try:
    print("Attempting to import speech module...")
    from interfaces import speech
    print("Speech module imported successfully!")
    print(f"Available in: {speech.__file__}")
except ImportError as e:
    print(f"Speech module import failed: {e}")

# Ensure root project directory is in path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    from core.memory import Memory
    from core.nlp import LlpCall
    from interfaces.chat import ChatManager # Assuming chat.py is in interfaces/
    from services.utilities import Utilities
    from IseeYou.IseeYou import FelixTrackingClient # Assuming IseeYou.py is in IseeYou/
    import config # Your configuration file (config.py)
except ImportError as e:
     logging.critical(f"FATAL: Failed to import core modules: {e}", exc_info=True)
     logging.critical("Please ensure core/, interfaces/, services/, IseeYou/, and config.py are accessible.")
     sys.exit(1)


# Setup basic logging config. Consider moving to a dedicated logging config function/file.
log_level = getattr(config, 'LOG_LEVEL', 'INFO').upper() # Allow configuring level via config.py
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(levelname)s - [%(threadName)s/%(module)s] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Ensure logs go to stdout


# Keep threading.Event for signaling the separate Felix client thread if needed,
# although calling client.shutdown() should be the primary mechanism.
# Using only client.shutdown() might be sufficient. Let's try that first.
# shutdown_event = asyncio.Event() # For main async loop coordination if needed

async def main():
    """Main asynchronous entry point: Initializes components and runs the application."""
    logging.info("--- Starting Smart Assistant Application ---")

    # --- Component Instances (initialize to None) ---
    nlp_instance: Optional[LlpCall] = None
    memory_instance: Optional[Memory] = None
    isee_client: Optional[FelixTrackingClient] = None
    utilities_instance: Optional[Utilities] = None
    chat_instance: Optional[ChatManager] = None
    client_thread: Optional[threading.Thread] = None

    try:
        # --- Configuration ---
        # These are accessed via the config object directly later

        # --- Initialize Core Components ---
        logging.info("Initializing NLP and Memory...")
        nlp_instance = LlpCall(api_key=config.GEMINI_KEY)
        memory_instance = Memory(system_prompt=config.SYSTEM_PROMPT)
        logging.info("NLP and Memory initialized.")

        # --- Initialize Utilities and ChatManager (pass the client instance) ---
        logging.info("Initializing Utilities and Chat Manager...")
        # Note: Utilities now requires isee_client
        utilities_instance = Utilities(
            config_instance=config,
            nlp_instance=nlp_instance
            # chat_instance=None # Avoid circular dependency if possible
        )

        chat_instance = ChatManager(
            memory_instance=memory_instance,
            nlp_instance=nlp_instance,
            config_instance=config,
            utilities_instance=utilities_instance, # Pass utilities instance
        )

        logging.info("Utilities and Chat Manager initialized.")


        logging.info("\n--- Initialization Complete. Ready for interaction. ---")
        logging.info(f"Assistant Name: {config.MODEL_NAME}")
        logging.info("Enter 'exit' to quit.")

        # --- Main Interaction Loop ---
        first_prompt = True
        while True: # Loop indefinitely until 'exit' or shutdown signal
            if first_prompt:
                try:
                    # Run blocking input in a separate thread to not block the asyncio loop that looks like we removed. (I believe the tracking)
                    choice = await asyncio.to_thread(
                        input, "Options: [Enter] to start, 'delete' memory, 'exit': "
                    )
                    choice = choice.lower().strip()

                    if choice == "delete":
                        if memory_instance.delete_convos():
                            logging.info("Memory Deleted Successfully.")
                            print(f"{config.MODEL_NAME}: Previous conversation history deleted.")
                        else:
                             logging.error("Failed to delete conversation memory.")
                    elif choice == "exit":
                        logging.info("Exit choice received on initial prompt.")
                        break # Exit the main loop
                    # Any other input (including Enter) just continues
                    first_prompt = False # Don't ask again in this session, if the user wanna delete something, he needs to restart the session.

                except EOFError:
                    logging.info("Input stream closed (EOF). Exiting.")
                    break
                except KeyboardInterrupt:
                     logging.info("Ctrl+C detected during initial prompt.")
                     raise # Re-raise to be caught by outer handler
                except Exception as e:
                     logging.error(f"Error during initial prompt: {e}", exc_info=True)
                     # Continue without asking again? Or break? Continue for now.
                     first_prompt = False


            # Let ChatManager handle the conversation turn (now async)
            # discussion_turn returns the AI response string, "exit", or None
            result = await chat_instance.discussion_turn()

            if result == "exit":
                logging.info("Exit command received from user via ChatManager.")
                break # Exit the main loop
            elif result is None:
                 # Handle cases where discussion failed (e.g., no audio input, STT error)
                 logging.warning("Discussion turn returned None (e.g., no input or processing error).")
                 # Optional: Add a small delay or prompt user again?
                 await asyncio.sleep(0.1)


    except KeyboardInterrupt:
        logging.info("\n--- Ctrl+C detected. Initiating shutdown. ---")
    except Exception as e:
         logging.error(f"\n--- An unexpected error occurred in the main async loop: {e} ---", exc_info=True)
    finally:
        logging.info("\n--- Starting Application Cleanup ---")


        # 1. Cleanup Utilities (e.g., close Selenium browser)
        if utilities_instance and hasattr(utilities_instance, 'cleanup'):
            logging.info("Cleaning up Utilities...")
            try:
                 utilities_instance.cleanup()
            except Exception as e:
                 logging.error(f"Error during Utilities cleanup: {e}", exc_info=True)

        logging.info("--- Application Shutdown Complete ---")


# --- Run the Application ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This might be caught inside main() already, but good practice
        logging.info("Application terminated by user (Ctrl+C at top level).")
    except Exception as e:
        # Catch final unexpected errors during asyncio.run() or setup
        logging.critical(f"FATAL: Unhandled exception during application startup or shutdown: {e}", exc_info=True)
        sys.exit(1) # Exit with error code

# --- END OF REFINED FILE main.py ---