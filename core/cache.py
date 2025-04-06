####file = cache 
import sys
import os
import logging
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from IseeYou.IseeYouClass import FelixTrackingClient
except ImportError as e:
    logging.info("There was an issue importing The vision service: ", e)
except Exception as e:
    logging.info("Some other weird error happened during the import: ", e)


import asyncio
import websockets
import os
from pathlib import Path
import time
import cv2
import asyncio
import time
import logging
from typing import Dict, Any, Optional

try:
    import config # Your configuration file (config.py)
except ImportError as e:
     logging.critical(f"FATAL: Failed to import core modules: {e}", exc_info=True)
     logging.critical("Please ensure core/, interfaces/, services/, IseeYou/, and config.py are accessible.")
     sys.exit(1)

async def main():
    # Create the client instance with proper URL format
    client = FelixTrackingClient(
        server_url="ws://localhost:8080"
    )





class VisualContextCache:
    """
    Stores and manages the most recent visual detection data.
    Provides thread-safe access to Felix detection information.
    """
    
    def __init__(self):
        """
        Initialize the visual context cache with default values and lock.
        """
        # Thread safety lock
        self.flag = None
        self.confidence = 0.0
        self.is_felix = False
        self.lock = asyncio.Lock()
        
        # Detection data structure
        self.detection_data = {
            "is_felix": False,      # Whether Felix is currently detected
            "confidence": 0.0,               # Confidence level (0.0 to 1.0)
            "position": "unknown",           # Position in frame (e.g., "left", "center", "right")
            "timestamp": time.time(),        # When this data was last updated
            "data_available": False          # Whether vision system is providing data
        }
        
        # Logging
        self.logger = logging.getLogger("VisualContextCache")
        self.logger.info("Visual Context Cache initialized")
    
    async def update_from_client(self, client) -> bool:
        """Updates the cache with latest detection data"""
        try:
            # Check if client has detections
            if not hasattr(client, "raw_detections") or not client.raw_detections:
                return False  # No detections available
                
            # Now process detections
            async with self.lock:  # Lock for all data access
                # Process the detection data (use the most confident Felix detection if multiple)
                best_confidence = 0.0
                felix_detected = False
                
                for detection in client.raw_detections:
                    confidence = detection.get('confidence', 0.0)
                    is_felix = detection.get('is_felix', False)
                    
                    # Save the detection with highest confidence
                    if is_felix and confidence > best_confidence:
                        best_confidence = confidence
                        felix_detected = True
                
                # Update the dictionary with the detection results
                self.detection_data.update({
                    "is_felix": felix_detected,
                    "confidence": best_confidence,
                    "timestamp": time.time(),
                    "data_available": True
                })
                
            return True  # Successfully updated
            
        except Exception as e:
            self.logger.error(f"Error updating detection data: {e}")
            return False  # Update failed
        
    
    async def get_detection_info(self) -> Dict[str, Any]:
        """
        Returns the current detection information in a thread-safe manner.
        
        Returns:
            Dict[str, Any]: Dictionary containing current detection data
        """
        # Implement: Return a copy of the detection data under lock
        
        async with self.lock:

            return self.detection_data.copy()
        



    
    async def is_data_stale(self, max_age_seconds: float = 5.0) -> bool:
        """
        Checks if the current detection data is considered stale.
        
        Args:
            max_age_seconds: Maximum age in seconds before data is considered stale
            
        Returns:
            bool: True if data is stale, False otherwise
        """
        # Implement: Compare current time with timestamp to determine staleness
        async with self.lock:
            
            current_time = time.time()

            last_timestamp = self.detection_data['timestamp']

            
            return (current_time -last_timestamp) > max_age_seconds
        
    
    async def start_update_loop(self, client, update_interval: float = 0.5) -> None:
        """
        Starts a background task that periodically updates the cache.
        
        Args:
            client: The FelixTrackingClient instance
            update_interval: How often to update the cache (in seconds)
        """
        # Store the task so we can access it later if needed
        self.update_task = asyncio.create_task(
            self._update_loop(client, update_interval)
        )
        self.logger.info(f"Started visual context update loop (interval: {update_interval}s)")


    async def _update_loop(self, client, update_interval: float):
        """The actual loop that runs in the background"""
        while True:
            try:
                # Update the cache with latest detection data
                await self.update_from_client(client)
                
                # Wait for the next update interval
                await asyncio.sleep(update_interval)
                
            except asyncio.CancelledError:
                # Handle case when the task is cancelled
                self.logger.info("Update loop cancelled")
                break
                
            except Exception as e:
                # Log any errors but don't stop the loop
                self.logger.error(f"Error in visual context update loop: {e}")
                # Wait a bit longer on error to avoid rapid failure loops
                await asyncio.sleep(update_interval * 2)
        
    async def share_info_AI(self, chat_manager: 'ChatManager')-> Optional[Dict[str, str]]:
        """
        Share vision detection information with the AI assistant.
        
        Args:
            chat_manager: An instance of ChatManager to use for LLM communication
        """
        logging.info("Sharing vision context with AI assistant")
        prompt = "[]"
        try:
            # Get the latest detection info with await
            detection_info = await self.get_detection_info()
            
            # Check if data is stale
            if await self.is_data_stale():
                prompt = "[Visual context: Vision system data is stale or unavailable]"
                logging.warning("Using stale detection data for AI communication")
            
            if detection_info and detection_info.get("data_available", False):
                if detection_info.get("is_felix", False):
                    prompt = (
                        f"[Visual context: Felix has been detected with "
                        f"{detection_info['confidence']*100:.1f}% confidence.]"
                    )
                else:
                    prompt = "[Visual context: No Felix detected in camera view.]"
            else:
                prompt = "[Visual context: Vision system is not providing data]"
                        
            # Now send the properly formatted prompt to the LLM
            logging.info(f"Sending vision context to LLM: {prompt}")
            if not chat_manager:
                 logging.error("Cannot share info with AI: chat_manager object is None.")
                 return None
            
            context_response = await chat_manager._call_llm(prompt)
            if context_response and not context_response.startswith("Blocked:"):
                 # Return the dictionary containing both prompt and response
                 return {"context_prompt": prompt, "context_response": context_response}
            else:
                 logging.error(f"LLM call for visual context failed or was blocked: {context_response}")
                 return None
            
            
        except Exception as e:
            logging.error(f"Error sharing vision context with AI: {e}")
            return None