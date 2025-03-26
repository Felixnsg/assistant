from core.memory import Memory
from core.nlp import LlpCall
from interfaces.chat import ChatManager
from services.utilities import Utilities
from IseeYou.IseeYou import FelixTrackingClient  # ✅ NEW
import config

def run():
    """Main entry of our code, initializes all necessary components."""
    nlp_instance = LlpCall()
    memory_instance = Memory(system_prompt=config.SYSTEM_PROMPT)

    isee_instance = FelixTrackingClient(
        
    )

    chat_instance = ChatManager(
        memory_instance=memory_instance,
        nlp_instance=nlp_instance,
        config_instance=config,
        utilities_instances=None,
        IseeYou_instance=isee_instance  
    )

    utilities_instance = Utilities(
        Chat_instance=chat_instance,
        Config_instance=config,
        nlp_instance=nlp_instance
    )
    
    chat_instance.utilities = utilities_instance

    user_input = False
    while True:
        if not user_input:
            choice = input("Type Delete to start over, Pass to ignore: ")     
            if choice.lower() == "delete":
                memory_instance.delete_convos()
                print(f"{config.MODEL_NAME}: Memory Deleted Successfully..........")
            user_input = True
        
        result = chat_instance.discussion()
        
        if result == "exit":
            print("Bye.................")
            break

if __name__ == "__main__":
    run()
