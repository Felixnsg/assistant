# services/invincible.py
import os
import time
import threading
import queue
import json
import sys
import uuid
import tempfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from core import memory
from core import nlp
from interfaces import streamaudio
import config

# Create a queue for Invincible story narration
story_queue = queue.Queue()
is_processing_queue = False
story_thread = None
processing_lock = threading.Lock()  # Add a lock to prevent concurrent processing

# Keep track of recent story responses to prevent duplicates
recent_responses = []
MAX_RECENT_RESPONSES = 5

def start_queue_processor():
    """Start the background thread that processes the narration queue"""
    global story_thread, is_processing_queue
    
    if story_thread is None or not story_thread.is_alive():
        is_processing_queue = True
        story_thread = threading.Thread(target=process_story_queue)
        story_thread.daemon = True
        story_thread.start()
        print("Invincible story narration queue processor started")

def process_story_queue():
    """Process the queue of story segments in a background thread"""
    global is_processing_queue
    
    while is_processing_queue:
        try:
            # Get a story segment from the queue, with a timeout
            # The timeout allows the thread to check is_processing_queue periodically
            story_segment = story_queue.get(timeout=1)
            
            # Skip empty or null content
            if not story_segment or story_segment.strip() == "":
                story_queue.task_done()
                continue
                
            # Acquire lock to ensure only one story is processed at a time
            with processing_lock:
                # Use streamaudio to narrate the story
                # Generate a unique filename based on timestamp to prevent conflicts
                unique_filename = f"response_{uuid.uuid4().hex}.mp3"
                os.environ['RESPONSE_FILENAME'] = unique_filename
                
                # Wait a short time to ensure previous audio processing is complete
                time.sleep(1.5)
                
                # Send to streamaudio
                streamaudio.say(story_segment)
                
                # Wait for audio to complete playing before processing the next item
                # This is important to prevent file access conflicts
                time.sleep(3)
            
            # Mark this task as done
            story_queue.task_done()
            
            # Small delay to prevent CPU hogging
            time.sleep(0.5)
            
        except queue.Empty:
            # Queue is empty, just continue checking
            pass
        except Exception as e:
            print(f"Error processing story queue: {e}")
            # Continue processing despite errors
            time.sleep(1)  # Add delay after errors to avoid rapid error loops
    
    print("Invincible story narration queue processor stopped")

def stop_queue_processor():
    """Stop the queue processor thread"""
    global is_processing_queue
    is_processing_queue = False
    
    # Clear the queue
    while not story_queue.empty():
        try:
            story_queue.get_nowait()
            story_queue.task_done()
        except queue.Empty:
            break

def is_duplicate_response(response):
    """Check if a response is a duplicate of recent responses"""
    global recent_responses
    
    # Create a simplified version for comparison (first 50 chars)
    simplified = response[:50].lower() if response else ""
    
    # Check if this simplified response is in our recent list
    for recent in recent_responses:
        if simplified in recent:
            return True
    
    # Not a duplicate, add to recent responses
    recent_responses.append(simplified)
    
    # Trim the list if it gets too long
    if len(recent_responses) > MAX_RECENT_RESPONSES:
        recent_responses = recent_responses[-MAX_RECENT_RESPONSES:]
    
    return False

def add_to_queue(story_segment):
    """Add a story segment to the narration queue"""
    # Skip empty content
    if not story_segment or story_segment.strip() == "":
        print("Empty story segment - not adding to queue")
        return
    
    # Skip duplicate content
    if is_duplicate_response(story_segment):
        print("Duplicate story segment - not adding to queue")
        return
        
    # Start the queue processor if not already running
    if not is_processing_queue:
        start_queue_processor()
    
    # Add the story segment to the queue
    # Only allow at most 2 items in the queue to prevent buildup
    if story_queue.qsize() < 2:
        story_queue.put(story_segment)
        print(f"Added story segment to queue (queue size: {story_queue.qsize()})")
    else:
        print(f"Queue already contains {story_queue.qsize()} items - not adding more")

def activate_invincible_mode():
    """
    Activate Invincible mode using the existing switch_mode function
    """
    from services import utilities
    
    # Clear any previous responses
    global recent_responses
    recent_responses = []
    
    # Call the existing switch_mode function with the Invincible prompt
    ai_response = utilities.switch_mode(config.switch_mode_prompt)
    
    # Only add valid responses to the queue
    if ai_response and ai_response.strip():
        # Add the response to the narration queue
        add_to_queue(ai_response)
    
    return ai_response

def continue_invincible_story():
    """
    Continue the Invincible story using the existing switch_mode_2 function
    """
    from services import utilities
    
    # Check if we're currently processing - if so, don't add more yet
    if processing_lock.locked():
        print("Still processing previous story segment - please wait")
        return "Still processing previous story segment. Please wait a moment before continuing."
    
    # If queue is already substantial, don't add more yet
    if story_queue.qsize() >= 2:
        print("Queue already has multiple segments - please wait")
        return "Queue already contains multiple story segments. Please wait for them to play before continuing."
    
    # Call the existing switch_mode_2 function with the continue prompt
    ai_response = utilities.switch_mode_2(config.switch_mode_prompt_2)
    
    # Only add valid responses to the queue
    if ai_response and ai_response.strip():
        # Add the response to the narration queue
        add_to_queue(ai_response)
    
    return ai_response

def deactivate_invincible_mode():
    """
    Exit Invincible mode
    """
    # Stop the queue processor
    stop_queue_processor()
    
    # Clear tracking variables
    global recent_responses
    recent_responses = []
    
    return "Exiting Invincible mode. Back to normal assistant mode."

# Note: No automatic initialization here - the queue processor will only start
# when activate_invincible_mode() is called and a valid response is received