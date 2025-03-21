###memory
import json
import requests
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import config
from services import utilities



class Memory:
    def __init__(self, system_prompt = config.SYSTEM_PROMPT, time_prompt = config.time_prompt):
        self.convos = []
        try:
            self.system_prompt = system_prompt
            self.time_prompt = time_prompt

        except (AttributeError, NameError ) as e:
            print(f"Error :{e} \n Maybe you for to import some modules?")
        except TypeError as e:
            print(f"Error :{e} \n Looks like there is type mismatch, verify your types")
        except Exception as e:
            print(f"An unexpected error occurred: {e}")

        self.get_convos()


    def save_convos(self, role: str, prompt: str) -> bool:
        """Save conversation exchanges between user and assistant to a JSON file.
    
    Appends the new message to the conversation history and saves the 
    entire conversation to 'past_conversations.json'.

    Args:
        role (string): contains user's role
        prompt (string): contains user's prompt"""
        try:
            if not self.convos:
                self.convos.append({"role": role, "parts": [{"text": self.system_prompt}]})   
            
            self.convos.append({"role": role, "parts": [{"text": prompt}]})
            
            with open("past_conversations.json", "w") as f:
                try:
                    json.dump(self.convos, f)
                    return True
                except (TypeError, ValueError) as json_error:
                    print(f"Error serializing conversation to JSON: {json_error}")
                    return False
            
        except (FileNotFoundError, PermissionError) as e:
            print(f"Error: {e} - Check file path and permissions")
            return False
        except OSError as e:
            print(f"Error: {e} - Disk likely full or I/O problem")
            return False
        except Exception as e:
            print(f"Error: {e} - Something went Wrong")
            return False


    def get_convos(self) -> list:
        """This function loads past convos into a json file

        Returns:
            list: return an empty list if past convo is empty, else return the loaded convos
        """

        try:
            
            if os.path.exists("past_conversations.json"):
                with open ("past_conversations.json" , "r") as f:
                    try:
                        self.convos = json.load(f)
                    except (TypeError, ValueError) as json_error:
                        print(f"Error serializing conversation to JSON: {json_error}")
                return self.convos 
            else:
                return []
            
        except PermissionError as e:
            print(f"Error {e}: -Check Permission")
        except OSError as e:
            print(f"Error {e}: Error with the system.")
        
        
    
   


    def delete_convos(self) -> list : 
        """clear the convos file

        Returns:
            list: returns the emptied convos
        """
        try:
            if os.path.exists("past_conversations.json"):
                os.remove("past_conversations.json")
                self.convos = []
                return self.convos
        except (PermissionError, OSError) as e:
            print(f"Error {e}: -Check permisions and System State")





    

