import asyncio
import time
import json
import logging
from pathlib import Path
import datetime
from typing import Optional, Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('visual_context_cache')

class VisualContextCache:
    """Pure data store for visual detection data - no video capture or WebSocket handling"""
    
    def __init__(self):
        # Thread safety lock
        self.lock = asyncio.Lock()
        
        # Detection data structure
        self.detection_data = {
            "is_felix": False,
            "confidence": 0.0,
            "timestamp": time.time(),
            "data_available": False,
            "tracker_id": None,
            "raw_detections": []
        }
        
        # Create directory for cache dumps
        self.cache_dir = Path("cache_dumps")
        self.cache_dir.mkdir(exist_ok=True)
        
        # Track when we last wrote to file
        self.last_file_write = 0
        self.file_write_interval = 2.0  # Write to file every 2 seconds
        
        # Additional tracking
        self.update_count = 0
        self.detection_history = []  # Limited history of detections
        
        logger.info("Visual Context Cache initialized as pure data store")
    
    async def update_from_client(self, detections: List[Dict[str, Any]]) -> bool:
        """
        Updates the cache with detection data from the video client.
        This is called by the video client when new detections are available.
        
        Args:
            detections: List of detection dictionaries from the video client
        
        Returns:
            bool: True if update was successful
        """
        try:
            if not detections:
                # No detections means no Felix in view
                async with self.lock:
                    self.detection_data.update({
                        "is_felix": False,
                        "confidence": 0.0,
                        "timestamp": time.time(),
                        "data_available": True,
                        "tracker_id": None,
                        "raw_detections": []
                    })
                return True
                
            async with self.lock:
                # Find best Felix detection
                best_confidence = 0.0
                felix_detected = False
                felix_tracker_id = None
                
                for detection in detections:
                    confidence = detection.get('confidence', 0.0)
                    is_felix = detection.get('is_felix', False)
                    
                    if is_felix and confidence > best_confidence:
                        best_confidence = confidence
                        felix_detected = True
                        felix_tracker_id = detection.get('tracker_id')
                
                # Update detection data
                self.detection_data.update({
                    "is_felix": felix_detected,
                    "confidence": best_confidence,
                    "timestamp": time.time(),
                    "data_available": True,
                    "tracker_id": felix_tracker_id,
                    "raw_detections": detections
                })
                
                # Add to detection history (keep last 10)
                self.update_count += 1
                self.detection_history.append({
                    "update_number": self.update_count,
                    "time": datetime.datetime.now().strftime("%H:%M:%S"),
                    "is_felix": felix_detected,
                    "confidence": best_confidence,
                    "tracker_id": felix_tracker_id,
                    "num_detections": len(detections)
                })
                if len(self.detection_history) > 10:
                    self.detection_history.pop(0)
                
                # Check if it's time to write to file
                current_time = time.time()
                if current_time - self.last_file_write >= self.file_write_interval:
                    await self._write_cache_to_file()
                    self.last_file_write = current_time
                
            return True
            
        except Exception as e:
            logger.error(f"Error updating detection data: {e}")
            return False
    
    async def _write_cache_to_file(self):
        """Write current cache state to a JSON file for debugging"""
        try:
            # Create a comprehensive debug dump
            debug_data = {
                "current_state": self.detection_data.copy(),
                "update_count": self.update_count,
                "detection_history": self.detection_history,
                "dump_time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            
            # Don't include raw_detections in file (too verbose)
            if "raw_detections" in debug_data["current_state"]:
                debug_data["current_state"]["raw_detections"] = f"[{len(debug_data['current_state']['raw_detections'])} detections]"
            
            # Write to latest.json (always overwritten)
            latest_file = self.cache_dir / "latest.json"
            with open(latest_file, 'w') as f:
                json.dump(debug_data, f, indent=2)
                
            return True
            
        except Exception as e:
            logger.error(f"Error writing cache to file: {e}")
            return False
    
    async def get_detection_info(self) -> Dict[str, Any]:
        """Get current detection info thread-safely"""
        async with self.lock:
            # Return a copy without raw_detections for cleaner API
            info = self.detection_data.copy()
            info.pop('raw_detections', None)
            return info
    
    async def get_visual_context_string(self) -> str:
        """
        Get a formatted string describing the current visual context.
        This is what gets injected into prompts.
        """
        detection_info = await self.get_detection_info()
        
        # Check if data is stale (older than 5 seconds)
        age = time.time() - detection_info['timestamp']
        if age > 5.0:
            return "[Visual context: Vision system data is stale or unavailable]"
        
        if not detection_info.get("data_available", False):
            return "[Visual context: Vision system is not providing data]"
        
        if detection_info.get("is_felix", False):
            confidence = detection_info.get('confidence', 0.0)
            tracker_id = detection_info.get('tracker_id')
            
            context = f"[Visual context: Felix detected with {confidence*100:.1f}% confidence"
            if tracker_id is not None:
                context += f" (tracking ID: {tracker_id})"
            context += "]"
            return context
        else:
            return "[Visual context: No Felix detected in camera view]"
    
    async def clear(self):
        """Clear all detection data"""
        async with self.lock:
            self.detection_data = {
                "is_felix": False,
                "confidence": 0.0,
                "timestamp": time.time(),
                "data_available": False,
                "tracker_id": None,
                "raw_detections": []
            }
            self.detection_history = []
            self.update_count = 0
            logger.info("Cache cleared")