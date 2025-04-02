from IseeYouClass import FelixTrackingClient
import asyncio
import websockets
import os
from pathlib import Path
import time
import cv2

async def main():
    # Create the client instance with proper URL format
    client = FelixTrackingClient(
        server_url="ws://localhost:8080"
    )
    
    # Create an instance of PictureInterface
    picture_interface = PictureInterface()
    
    # Connect to the websocket server
    async with websockets.connect("ws://localhost:8080") as websocket:
        # Start all tasks
        tasks = [
            asyncio.create_task(client.capture_and_send_frames(websocket, 0)),
            asyncio.create_task(client.receive_results(websocket)),
            asyncio.create_task(client.display_loop()),
            asyncio.create_task(picture_interface.process_images(client))
        ]
        
        await asyncio.gather(*tasks)
        
class PictureInterface:
    """Save pictures that the recognizer missed."""
    def __init__(self):
        self.counter = 0# We will use this counter to unify, or make them names be original.
        self.last_save = 0
        Path("Saved_Images").mkdir(exist_ok=True)  # Corrected folder name to match filename

    async def process_images(self, client):
        while True:
            try:
                if hasattr(client, "raw_detections") and client.raw_detections:
                    current_frame = None
                    async with client.frame_lock:
                        if client.current_frame is not None:
                            current_frame = client.current_frame.copy()

                    if current_frame is not None:
                        
                        for detection in client.raw_detections:
                            confidence = detection.get('confidence', 0)
                            is_felix = detection.get('is_felix', False)

                            if confidence < 0.52 or not is_felix:
                                await self.save_images(current_frame, detection)
                
                # Add sleep to prevent CPU overload
                await asyncio.sleep(0.1)
                
            except Exception as e:
                print(f"Exception caught: {e}")
                await asyncio.sleep(1)  # Longer sleep on error

    async def save_images(self, frame, detection):
        try:

            # Use the class counter variable
            current_time = time.time()
            if (current_time - self.last_save >= 3) and self.counter < 1000:
                timestamp = int(time.time())
                self.counter += 1
                filename = f"Saved_Images/detection_{timestamp}_{self.counter}.jpg"
                
                # Create a copy of the frame
                img = frame.copy()
                
                # Save the image
                cv2.imwrite(filename, img)
                print(f"Saved image: {filename}")

                self.last_save = current_time
                
            

        except Exception as e:
            print(f"Error saving image: {e}")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user")
    except Exception as e:
        print(f"Unhandled exception: {e}")