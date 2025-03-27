# File: main.py
# --- REFACTOR: Renamed from main_app.py ---
"""
Main entry point for the Smart Assistant application.

Initializes all components (NLP, Memory, Vision Client, Chat Manager, Utilities),
starts the background thread for the vision client, and runs the main asynchronous
interaction loop managed by the ChatManager. Handles graceful shutdown.
"""

import threading
import asyncio
import time
import sys
import os
import traceback
import logging # --- REFACTOR: Added logging ---
from typing import Optional
try:
    print("Attempting to import speech module...")
    from interfaces import speech
    print("Speech module imported successfully!")
    print(f"Available in: {speech.__file__}")
except ImportError as e:
    print(f"Speech module import failed: {e}")

# --- REFACTOR: Standardize imports and path handling ---
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
     # --- REFACTOR: Use logging for critical errors ---
     logging.critical(f"FATAL: Failed to import core modules: {e}", exc_info=True)
     logging.critical("Please ensure core/, interfaces/, services/, IseeYou/, and config.py are accessible.")
     sys.exit(1)


# --- REFACTOR: Configure logging ---
# Setup basic logging config. Consider moving to a dedicated logging config function/file.
log_level = getattr(config, 'LOG_LEVEL', 'INFO').upper() # Allow configuring level via config.py
logging.basicConfig(level=log_level,
                    format='%(asctime)s - %(levelname)s - [%(threadName)s/%(module)s] - %(message)s',
                    handlers=[logging.StreamHandler(sys.stdout)]) # Ensure logs go to stdout


# --- REFACTOR: Shutdown signal using asyncio.Event for the main async loop ---
# Keep threading.Event for signaling the separate Felix client thread if needed,
# although calling client.shutdown() should be the primary mechanism.
# Using only client.shutdown() might be sufficient. Let's try that first.
# shutdown_event = asyncio.Event() # For main async loop coordination if needed

# --- REFACTOR: Simplified Felix client runner ---
def run_felix_client_thread(client_instance: FelixTrackingClient):
    """Target function for the Felix client thread."""
    thread_name = threading.current_thread().name
    logging.info(f"Starting FelixTrackingClient thread '{thread_name}'...")
    try:
        # Create an asyncio loop for this thread
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Use client's video source attribute
        video_source = client_instance.video_source
        logging.info(f"Thread '{thread_name}' starting client.start_tracking() with source: {video_source}")
        
        # Start tracking (non-blocking)
        started = loop.run_until_complete(client_instance.start_tracking(video_source=video_source))
        
        if started:
            # Keep thread running until client signals shutdown
            while client_instance._is_running:
                loop.run_until_complete(asyncio.sleep(1))
        else:
            logging.error(f"Failed to start tracking in thread '{thread_name}'")
            
    except Exception as e:
        logging.error(f"Error in Felix client thread '{thread_name}': {e}", exc_info=True)
    finally:
        logging.info(f"Felix client thread '{thread_name}' finished.")

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

        # --- Initialize FelixTrackingClient ---
        logging.info("Initializing Felix Tracking Client...")
        # server_url is now fetched from config inside FelixTrackingClient init
        isee_client = FelixTrackingClient(server_url=config.FELIX_SERVER_URL)
        # Basic check if tracker component failed internal init
        if isee_client.byte_tracker is None:
             raise RuntimeError("FelixTrackingClient failed internal initialization (ByteTrack). Cannot proceed.")
        logging.info("Felix Tracking Client initialized.")

        # --- Start FelixTrackingClient in Background Thread ---
        logging.info("Starting Felix Tracking Client thread...")
        client_thread = threading.Thread(
            target=run_felix_client_thread,
            args=(isee_client,), # Pass only the client instance
            name="FelixClientThread", # Give the thread a name
            daemon=True # Allows main program to exit even if this thread is running (though we join on shutdown)
        )
        client_thread.start()

        # Give the client a moment to start up (optional, but can help avoid race conditions)
        await asyncio.sleep(2)
        if not client_thread.is_alive():
            logging.warning("Felix client thread did not stay alive after starting. Check client logs.")
            # Decide if this is critical - maybe connection failed immediately?

        # --- Initialize Utilities and ChatManager (pass the client instance) ---
        logging.info("Initializing Utilities and Chat Manager...")
        # Note: Utilities now requires isee_client
        utilities_instance = Utilities(
            config_instance=config,
            isee_client_instance=isee_client,
            nlp_instance=nlp_instance, # Pass nlp if needed by utilities (currently not)
            # chat_instance=None # Avoid circular dependency if possible
        )

        chat_instance = ChatManager(
            memory_instance=memory_instance,
            nlp_instance=nlp_instance,
            config_instance=config,
            utilities_instance=utilities_instance, # Pass utilities instance
            isee_client_instance=isee_client # Pass Felix client instance
        )
        # If Utilities needs ChatManager after init (try to avoid):
        # utilities_instance.chat = chat_instance
        logging.info("Utilities and Chat Manager initialized.")


        logging.info("\n--- Initialization Complete. Ready for interaction. ---")
        logging.info(f"Assistant Name: {config.MODEL_NAME}")
        logging.info("Enter 'exit' to quit.")

        # --- Main Interaction Loop ---
        first_prompt = True
        while True: # Loop indefinitely until 'exit' or shutdown signal
            if first_prompt:
                try:
                    # Run blocking input in a separate thread to not block asyncio loop
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
                    first_prompt = False # Don't ask again in this session

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

        # 1. Signal Felix client to stop its loops gracefully
        if isee_client:
            logging.info("Signaling Felix client thread to shutdown...")
            isee_client.shutdown() # Call the client's shutdown method

        # 2. Cleanup Utilities (e.g., close Selenium browser)
        if utilities_instance and hasattr(utilities_instance, 'cleanup'):
            logging.info("Cleaning up Utilities...")
            try:
                 utilities_instance.cleanup()
            except Exception as e:
                 logging.error(f"Error during Utilities cleanup: {e}", exc_info=True)

        # 3. Wait briefly for the client thread to finish
        if client_thread and client_thread.is_alive():
            logging.info("Waiting for Felix client thread to join...")
            # Run join in a separate thread to avoid blocking asyncio loop if join takes time
            try:
                 await asyncio.to_thread(client_thread.join, timeout=7.0) # Increased timeout slightly
            except TimeoutError:
                 logging.warning("Timeout waiting for Felix client thread to join.")
            except Exception as e:
                 logging.error(f"Error joining Felix client thread: {e}", exc_info=True)

            if client_thread.is_alive():
                 logging.warning("Felix client thread did not exit cleanly after shutdown signal and join timeout.")

        # Add any other cleanup needed (e.g., closing global resources)

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