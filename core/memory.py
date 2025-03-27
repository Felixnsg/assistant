### memory
import json
import sys
import os
# --- REFACTOR: Simplified path appending ---
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
try:
    import config
except ImportError as e:
    print(f"Error importing config module: {e}. Ensure config.py exists and is accessible.")
    sys.exit(1)

DEFAULT_CONVO_FILE = "past_conversations.json"

class Memory:
    """
    Manages the storage and retrieval of conversation history.

    Attributes:
        system_prompt (str): The initial system prompt for the assistant.
        time_prompt (str): A prompt related to time, loaded from config.
        convos (list): A list storing the conversation history in memory.
                       Each item is a dictionary representing a message turn,
                       formatted for the LLM API (e.g., Google Generative AI).
        convo_file (str): The path to the JSON file used for persistence.
    """
    def __init__(self, system_prompt: str = config.SYSTEM_PROMPT, time_prompt: str = config.time_prompt, convo_file: str = DEFAULT_CONVO_FILE):
        """
        Initializes the Memory instance.

        Args:
            system_prompt (str): The system prompt to prepend to conversations.
                                 Defaults to config.SYSTEM_PROMPT.
            time_prompt (str): A prompt related to time. Defaults to config.time_prompt.
            convo_file (str): Path to the conversation history file.
                              Defaults to DEFAULT_CONVO_FILE.
        """
        self.convos = []
        self.convo_file = convo_file
        try:
            self.system_prompt = str(system_prompt)
        except Exception as e:
            print(f"An unexpected error occurred during Memory initialization: {e}")
            self.system_prompt = "You are a helpful assistant."

        self.get_convos() # Load existing conversations on initialization

    def save_convos(self, role: str, prompt: str) -> bool:
        """
        Saves a single conversation turn to the history (in memory and to file).

        Appends the new message to the internal conversation list and then writes
        the entire updated list to the JSON file. Includes the system prompt
        if the conversation history is currently empty.

        Args:
            role (str): The role of the speaker ('user' or 'model').
            prompt (str): The text content of the message.

        Returns:
            bool: True if saving was successful, False otherwise.
        """
        if not isinstance(role, str) or not isinstance(prompt, str):
            print("Error saving conversation: Role and prompt must be strings.")
            return False

        try:
            if not self.convos and role == "user": # Typically user starts
                 if not any(entry.get("role") == "system" for entry in self.convos): 
                    pass 

            self.convos.append({"role": role, "parts": [{"text": prompt}]})

            # Write the entire history to file
            with open(self.convo_file, "w", encoding="utf-8") as f:
                try:
                    json.dump(self.convos, f, indent=2)
                    return True
                except (TypeError, ValueError) as json_error:
                    print(f"Error serializing conversation to JSON in '{self.convo_file}': {json_error}")
                    try:
                        os.remove(self.convo_file)
                    except OSError:
                        print("OSError caught: ", e)
                    return False
                except Exception as e: 
                     print(f"Error writing conversation to file '{self.convo_file}': {e}")
                     return False

        except (FileNotFoundError, PermissionError) as e:
            print(f"Error accessing conversation file '{self.convo_file}': {e} - Check path and permissions.")
            return False
        except OSError as e:
            print(f"OS error during file operation for '{self.convo_file}': {e} - Disk likely full or I/O problem.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred in save_convos: {e}")
            return False

    def get_convos(self) -> list:
        """
        Loads conversation history from the JSON file.

        If the file exists and is valid JSON, it loads the conversations into
        `self.convos`. If the file doesn't exist or is invalid, `self.convos`
        remains empty (or keeps its current state if called after initialization).

        Returns:
            list: The loaded conversation history (list of dictionaries),
                  or an empty list if loading fails or file is empty/doesn't exist.
        """
        if os.path.exists(self.convo_file):
            try:
                with open(self.convo_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    if not content:
                        print(f"Conversation file '{self.convo_file}' is empty.")
                        self.convos = []
                        return self.convos

                    try:
                        # f.seek(0) # Rewind if reading again
                        loaded_convos = json.loads(content) # Use content read
                        if isinstance(loaded_convos, list):
                            self.convos = loaded_convos
                             # print(f"Loaded {len(self.convos)} conversation turns from '{self.convo_file}'.") # Optional debug log
                        else:
                            print(f"Error: Conversation file '{self.convo_file}' does not contain a valid JSON list.")
                            self.convos = [] # Reset if format is wrong
                    except json.JSONDecodeError as json_error:
                        print(f"Error decoding JSON from '{self.convo_file}': {json_error}. File might be corrupted.")
                        self.convos = [] 
            except PermissionError as e:
                print(f"Error reading '{self.convo_file}': Permission denied - {e}")
            except OSError as e:
                print(f"OS error reading '{self.convo_file}': {e}")
            except Exception as e:
                print(f"An unexpected error occurred in get_convos: {e}")
        else:
            print(f"Conversation file '{self.convo_file}' not found. Starting with empty history.")
            self.convos = []

        return list(self.convos)


    def delete_convos(self) -> bool:
        """
        Deletes the conversation history file and clears the in-memory history.

        Returns:
            bool: True if deletion was successful, False otherwise.
        """
        try:
            if os.path.exists(self.convo_file):
                os.remove(self.convo_file)
                print(f"Conversation file '{self.convo_file}' deleted.")
            else:
                print(f"Conversation file '{self.convo_file}' not found, nothing to delete.")
            # Clear in-memory list regardless of file existence
            self.convos = []
            return True
        except (PermissionError, OSError) as e:
            print(f"Error deleting conversation file '{self.convo_file}': {e} - Check permissions or if file is in use.")
            return False
        except Exception as e:
            print(f"An unexpected error occurred in delete_convos: {e}")
            return False