# File: main.py
"""
Main entry point for the Smart Assistant application.

Initializes all components (NLP, Memory, Utilities, Chat Manager),
and runs the main asynchronous interaction loop managed by the ChatManager.
Handles graceful shutdown.
"""

import asyncio
import sys
import os
import logging
from typing import Optional
import config # Your configuration file (config.py)

# --- Logging Setup Function (Keep your existing setup_logging here) ---
def setup_logging(log_dir="logs", default_level=logging.INFO):
    """Configures file-based logging for different modules."""
    # --- PASTE YOUR setup_logging FUNCTION CODE HERE ---
    # It should look something like the version from message #11
    # Make sure it handles directory creation and sets up handlers
    # for different modules (core.memory, core.nlp, interfaces.chat, etc.)
    # Ensure logger.propagate = False is set for specific handlers.
    try:
        os.makedirs(log_dir, exist_ok=True)
        print(f"Log directory '{log_dir}' ensured.") # Keep this initial print
    except OSError as e:
        print(f"Error creating log directory '{log_dir}': {e}. Logs might not be saved.", file=sys.stderr)
        return

    log_formatter = logging.Formatter(
        '%(asctime)s - %(levelname)s - [%(name)s] - %(message)s'
    )
    log_config = {
        'core.memory': 'memory.log',
        'core.nlp': 'nlp.log',
        'interfaces.chat': 'chat.log',
        'interfaces.speech': 'speech.log', # Add others as needed
        'services.utilities': 'utilities.log',
        'IseeYou.IseeYouClass': 'isee_you_client.log', # Adjusted name
        'core.cache': 'cache.log', # Added cache log
         # Add vision server components if separate logging desired
        '__main__': 'main.log',
    }
    configured_loggers = []
    for logger_name, log_filename in log_config.items():
        try:
            logger = logging.getLogger(logger_name)
            logger.setLevel(default_level)
            file_handler = logging.FileHandler(os.path.join(log_dir, log_filename), encoding='utf-8')
            file_handler.setFormatter(log_formatter)
            for handler in logger.handlers[:]: logger.removeHandler(handler) # Avoid duplication
            logger.addHandler(file_handler)
            logger.propagate = False # Prevent propagation
            configured_loggers.append(logger_name)
        except Exception as e:
            print(f"Error configuring logger '{logger_name}': {e}", file=sys.stderr)

    root_logger = logging.getLogger()
    for handler in root_logger.handlers[:]: root_logger.removeHandler(handler)
    root_logger.setLevel(logging.ERROR) # Keep root quiet
    # root_logger.addHandler(logging.NullHandler()) # Alternative

    print(f"Configured file logging for: {', '.join(configured_loggers)}") # Keep this print
# --- END OF setup_logging FUNCTION ---


# --- Call logging setup EARLY ---
log_level_str = getattr(config, 'LOG_LEVEL', 'INFO').upper()
log_level = getattr(logging, log_level_str, logging.INFO)
setup_logging(default_level=log_level)

logger = logging.getLogger(__name__) # Logger for main.py itself

# --- Imports ---
# Ensure root project directory is in path (if needed, often not necessary if run from root)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    # Core components
    from core.memory import Memory
    from core.nlp import LlpCall
    # Interfaces
    from interfaces.chat import ChatManager
    from interfaces import speech # Keep speech import attempt if used
    # Services
    from services.utilities import Utilities
    # Vision (only if needed directly in main, usually not)
    # from IseeYou.IseeYouClass import FelixTrackingClient
except ImportError as e:
     logger.critical(f"FATAL: Failed to import core modules: {e}", exc_info=True)
     logger.critical("Please ensure core/, interfaces/, services/, and config.py are accessible.")
     sys.exit(1)
except Exception as e:
    logger.critical(f"FATAL: An unexpected error occurred during imports: {e}", exc_info=True)
    sys.exit(1)


async def main():
    """Main asynchronous entry point: Initializes components and runs the application."""
    logger.info("--- Starting Smart Assistant Application ---")

    # --- Component Instances (initialize to None) ---
    nlp_instance: Optional[LlpCall] = None
    memory_instance: Optional[Memory] = None
    utilities_instance: Optional[Utilities] = None
    chat_instance: Optional[ChatManager] = None

    try:
        # --- Initialize Core Components ---
        logger.info("Initializing NLP and Memory...")
        # Ensure API Key is handled correctly in LlpCall or config
        nlp_instance = LlpCall(api_key=config.GEMINI_KEY)
        # Ensure system prompt is loaded correctly
        system_prompt = getattr(config, 'SYSTEM_PROMPT', 'You are a helpful assistant.')
        memory_instance = Memory(system_prompt=system_prompt)
        logger.info("NLP and Memory initialized.")

        # --- Initialize Utilities and ChatManager ---
        logger.info("Initializing Utilities and Chat Manager...")

        # Step 1: Initialize Utilities (without chat_instance initially)
        utilities_instance = Utilities(
            config_instance=config,
            nlp_instance=nlp_instance
            # chat_instance=None # Don't pass chat instance here
        )

        # Step 2: Initialize ChatManager, giving it the utilities_instance
        chat_instance = ChatManager(
            memory_instance=memory_instance,
            nlp_instance=nlp_instance,
            config_instance=config,
            utilities_instance=utilities_instance,
        )

        # --- Step 3: Inject chat_instance into utilities_instance ---
        # This breaks the initialization cycle dependency
        utilities_instance.chat = chat_instance
        logger.info("ChatManager instance injected into Utilities instance.")
        # ----------------------------------------------------------

        logger.info("Utilities and Chat Manager initialized successfully.")


        logger.info("\n--- Initialization Complete. Ready for interaction. ---")
        assistant_name = getattr(config, 'MODEL_NAME', 'Assistant') # Use getattr for safety
        logger.info(f"Assistant Name: {assistant_name}")
        logger.info("Enter 'exit' to quit.")

        # --- Main Interaction Loop ---
        first_prompt = True
        while True:
            if first_prompt:
                try:
                    # Run blocking input in a separate thread
                    choice = await asyncio.to_thread(
                        input, "Options: [Enter] to start, 'delete' memory, 'exit': "
                    )
                    choice = choice.lower().strip()

                    if choice == "delete":
                        if memory_instance.delete_convos():
                            logger.info("Memory Deleted Successfully.")
                            print(f"{assistant_name}: Previous conversation history deleted.")
                        else:
                             logger.error("Failed to delete conversation memory.")
                             print(f"{assistant_name}: Sorry, I couldn't delete the memory.") # User feedback
                    elif choice == "exit":
                        logger.info("Exit choice received on initial prompt.")
                        break # Exit the main loop
                    # Any other input (including Enter) just continues
                    first_prompt = False

                except EOFError:
                    logger.info("Input stream closed (EOF). Exiting.")
                    break
                except KeyboardInterrupt:
                     logger.info("Ctrl+C detected during initial prompt.")
                     raise # Re-raise to be caught by outer handler
                except Exception as e:
                     logger.error(f"Error during initial prompt: {e}", exc_info=True)
                     first_prompt = False # Avoid getting stuck


            # Let ChatManager handle the conversation turn (async)
            result = await chat_instance.discussion_turn()

            if result == "exit":
                logger.info("Exit command received from user via ChatManager.")
                break # Exit the main loop
            elif result is None and not first_prompt: # Avoid logging warning on initial empty input
                 # Handle cases where discussion failed (e.g., no audio input, STT error)
                 logger.warning("Discussion turn returned None (e.g., no input or processing error).")
                 await asyncio.sleep(0.1)


    except KeyboardInterrupt:
        logger.info("\n--- Ctrl+C detected. Initiating shutdown. ---")
    except Exception as e:
         logger.error(f"\n--- An unexpected error occurred in the main async loop: {e} ---", exc_info=True)
    finally:
        logger.info("\n--- Starting Application Cleanup ---")

        # --- Cleanup Utilities ---
        # 1. Stop Video Service (if running) via Utilities
        if utilities_instance and hasattr(utilities_instance, 'exit') and getattr(utilities_instance, 'running', False):
            logger.info("Stopping Vision Service via Utilities...")
            try:
                 # Ensure the exit method correctly handles stopping tasks/websockets
                 await utilities_instance.exit()
            except Exception as e:
                 logger.error(f"Error during Utilities vision service stop: {e}", exc_info=True)

        # 2. Cleanup other Utilities resources (e.g., Selenium)
        if utilities_instance and hasattr(utilities_instance, 'cleanup'):
            logger.info("Cleaning up Utilities (e.g., Selenium)...")
            try:
                 # Assuming cleanup is synchronous
                 utilities_instance.cleanup()
            except Exception as e:
                 logger.error(f"Error during Utilities cleanup: {e}", exc_info=True)
        # ----------------------

        # Cancel any remaining asyncio tasks managed directly by main (if any)
        # tasks = [t for t in asyncio.all_tasks() if t is not asyncio.current_task()]
        # [task.cancel() for task in tasks]
        # await asyncio.gather(*tasks, return_exceptions=True)


        logging.info("--- Application Shutdown Complete ---")


# --- Run the Application ---
if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        # This might be caught inside main() already, but good practice
        logger.info("Application terminated by user (Ctrl+C at top level).")
    except Exception as e:
        # Catch final unexpected errors during asyncio.run() or setup
        logger.critical(f"FATAL: Unhandled exception during application startup or shutdown: {e}", exc_info=True)
        sys.exit(1) # Exit with error code