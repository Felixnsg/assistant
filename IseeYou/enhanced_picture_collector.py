import asyncio
import cv2
import json
import logging
import time
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np
from dataclasses import dataclass, asdict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('edge_case_collector')

@dataclass
class DetectionMetadata:
    """Metadata for each collected image"""
    timestamp: float
    confidence: float
    is_felix: bool
    tracker_id: Optional[int]
    lighting_condition: str  # 'bright', 'normal', 'dim'
    collection_reason: str  # 'low_confidence', 'borderline', 'false_positive', etc.
    image_quality: float  # Based on blur detection
    face_size: Tuple[int, int]  # Width, height of detection box
    
class EdgeCaseCollector:
    """
    Collects edge cases from the video system for model improvement.
    Integrates with existing FlowControlledClient and VisualContextCache.
    """
    
    def __init__(self, 
                 output_dir: str = "training_data_collection",
                 confidence_thresholds: Dict[str, float] = None):
        """
        Initialize the edge case collector.
        
        Args:
            output_dir: Directory to save collected images
            confidence_thresholds: Dict with threshold values for collection
        """
        # Set up directories
        self.output_dir = Path(output_dir)
        self.felix_dir = self.output_dir / "felix"
        self.notfelix_dir = self.output_dir / "notfelix"
        self.metadata_dir = self.output_dir / "metadata"
        
        # Create directories
        for dir_path in [self.felix_dir, self.notfelix_dir, self.metadata_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
        
        # Collection thresholds
        self.thresholds = confidence_thresholds or {
          'low_confidence_upper': 0.53,     # Collect Felix if < 53% confident it’s Felix
           'borderline_lower':    0.53,      # (optional) start of ’uncertain’ zone
           'borderline_upper':    0.65,      # (optional) end of ’uncertain’ zone
           'high_confidence_lower': 0.65   # Collect Not-Felix if > 65% sure it’s Not-Felix
        
        }

        # Collection settings
        self.min_interval_same_person = 15.0   # Seconds between collecting same tracker_id
        self.max_images_per_session = 1000
        self.balance_ratio = 0.5  # Try to maintain 50/50 felix/notfelix
        
        # Tracking
        self.last_collection_time = {}  # tracker_id -> timestamp
        self.collection_stats = {
            'felix': 0,
            'notfelix': 0,
            'total': 0,
            'reasons': {
                'low_confidence': 0,
                'borderline': 0,
                'false_positive_check': 0,
                'balance_collection': 0
            }
        }
        
        # Session info
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_metadata_file = self.metadata_dir / f"session_{self.session_id}.json"
        
        # Cache reference (will be set by video system)
        self.visual_cache = None
        self.current_frame = None
        self.last_frame_time = 0
        
        logger.info(f"Edge Case Collector initialized. Output: {self.output_dir}")
        logger.info(f"Session ID: {self.session_id}")
    
    def set_visual_cache(self, cache):
        """Set reference to the visual context cache"""
        self.visual_cache = cache
        logger.info("Visual cache reference set")
    
    async def process_detection_update(self, detections: List[Dict], frame: np.ndarray):
        """
        Process detection updates from the video system.
        This is called by the video client when new detections arrive.
        
        Args:
            detections: List of detection dictionaries
            frame: Current frame from video
        """
        if self.collection_stats['total'] >= self.max_images_per_session:
            return
        
        # Update current frame
        self.current_frame = frame
        self.last_frame_time = time.time()
        
        # Process each detection
        for detection in detections:
            should_collect, reason = self._should_collect_detection(detection)
            
            if should_collect:
                await self._collect_image(detection, reason)
    
    def _should_collect_detection(self, detection: Dict) -> Tuple[bool, str]:
        """
        Decide whether to snapshot this detection.

        Returns (True, reason) if we should collect;
        (False, reason) otherwise.
        """
        confidence = detection.get('confidence', 0.0)
        is_felix   = detection.get('is_felix', False)
        tracker_id = detection.get('tracker_id')

        # 0. Throttle rapid repeats of the same person
        last_time = self.last_collection_time.get(tracker_id, 0)
        if time.time() - last_time < self.min_interval_same_person:
            return False, "too_recent"

        # 1. Collect FELIX when model < 0.53 confident it’s Felix
        if is_felix and confidence < self.thresholds['low_confidence_upper']:
            return True, "low_confidence_felix"

        # 2. Collect NOT-FELIX when model ≥ 65% sure it’s not Felix
        nonfelix_conf = 1.0 - confidence
        if (not is_felix) and nonfelix_conf >= self.thresholds['high_confidence_lower']:
            return True, "high_confidence_notfelix"

        # 3. Borderline cases (uncertain zone 0.53–0.65)
        if self.thresholds['borderline_lower'] <= confidence <= self.thresholds['borderline_upper']:
            return True, "borderline"

        # 4. False-positive check: high-confidence Not-Felix samples (10% chance)
        if confidence > self.thresholds['high_confidence_lower'] and not is_felix and np.random.random() < 0.1:
            return True, "false_positive_check"

        # 5. Balance collection – skip some if one class is over-represented
        ratio = self._get_collection_ratio()  # fraction of Felix images so far
        if is_felix and ratio > 0.7:
            if np.random.random() < 0.2:
                return False, "balance_skip"
        elif not is_felix and ratio < 0.3:
            if np.random.random() < 0.2:
                return False, "balance_skip"

        # 6. Random sampling for diversity (5% chance)
        if np.random.random() < 0.05:
            return True, "random_sampling"

        # 7. Otherwise, don’t collect
        return False, "not_selected"

    
    async def _collect_image(self, detection: Dict, reason: str):
        """Save image and metadata"""
        if self.current_frame is None:
            return
        
        try:
            # Extract detection info
            is_felix = detection.get('is_felix', False)
            confidence = detection.get('confidence', 0.0)
            tracker_id = detection.get('tracker_id', -1)
            box = detection.get('box', [0, 0, 100, 100])
            
            # Determine save directory
            save_dir = self.felix_dir if is_felix else self.notfelix_dir
            
            # Generate filename
            timestamp = int(time.time() * 1000)
            filename = f"{self.session_id}_{timestamp}_{tracker_id}_{confidence:.3f}.jpg"
            image_path = save_dir / filename
            
            # Extract face region with padding
            face_img = self._extract_face_region(self.current_frame, box)
            
            # Assess image quality
            quality_score = self._assess_image_quality(face_img)
            
            # Only save if quality is acceptable
            if quality_score > 0.3:  # Adjustable threshold
                # Save image
                cv2.imwrite(str(image_path), face_img)
                
                # Create metadata
                metadata = DetectionMetadata(
                    timestamp=time.time(),
                    confidence=confidence,
                    is_felix=is_felix,
                    tracker_id=tracker_id,
                    lighting_condition=self._assess_lighting(face_img),
                    collection_reason=reason,
                    image_quality=quality_score,
                    face_size=(box[2] - box[0], box[3] - box[1])
                )
                
                # Save metadata
                metadata_path = self.metadata_dir / f"{filename}.json"
                with open(metadata_path, 'w') as f:
                    json.dump(asdict(metadata), f, indent=2)
                
                # Update tracking
                if tracker_id:
                    self.last_collection_time[tracker_id] = time.time()
                
                # Update stats
                self.collection_stats[save_dir.name] += 1
                self.collection_stats['total'] += 1
                self.collection_stats['reasons'][reason] += 1
                
                logger.info(f"Collected: {filename} | Reason: {reason} | "
                           f"Felix: {is_felix} | Conf: {confidence:.3f}")
                
                # Save session stats periodically
                if self.collection_stats['total'] % 10 == 0:
                    self._save_session_metadata()
        
        except Exception as e:
            logger.error(f"Error collecting image: {e}")
    
    def _extract_face_region(self, frame: np.ndarray, box: List[int], 
                            padding: float = 0.2) -> np.ndarray:
        """Extract face region with padding"""
        h, w = frame.shape[:2]
        
        # Calculate padding
        x1, y1, x2, y2 = box
        width = x2 - x1
        height = y2 - y1
        
        pad_x = int(width * padding)
        pad_y = int(height * padding)
        
        # Apply padding with bounds checking
        x1 = max(0, x1 - pad_x)
        y1 = max(0, y1 - pad_y)
        x2 = min(w, x2 + pad_x)
        y2 = min(h, y2 + pad_y)
        
        return frame[y1:y2, x1:x2]
    
    def _assess_image_quality(self, image: np.ndarray) -> float:
        """
        Assess image quality based on blur detection.
        Returns score 0-1 (1 is best quality)
        """
        if image.size == 0:
            return 0.0
        
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Calculate Laplacian variance (blur detection)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        # Normalize to 0-1 range (empirically determined)
        quality_score = min(1.0, laplacian_var / 1000.0)
        
        return quality_score
    
    def _assess_lighting(self, image: np.ndarray) -> str:
        """Assess lighting conditions"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        
        if mean_brightness < 60:
            return "dim"
        elif mean_brightness > 180:
            return "bright"
        else:
            return "normal"
    
    def _get_collection_ratio(self) -> float:
        """Get ratio of felix images to total"""
        total = self.collection_stats['felix'] + self.collection_stats['notfelix']
        if total == 0:
            return 0.5
        return self.collection_stats['felix'] / total
    
    def _save_session_metadata(self):
        """Save session statistics"""
        session_data = {
            'session_id': self.session_id,
            'timestamp': datetime.now().isoformat(),
            'stats': self.collection_stats,
            'thresholds': self.thresholds,
            'collection_ratio': self._get_collection_ratio()
        }
        
        with open(self.session_metadata_file, 'w') as f:
            json.dump(session_data, f, indent=2)
    
    def get_collection_stats(self) -> Dict:
        """Get current collection statistics"""
        return {
            **self.collection_stats,
            'ratio': self._get_collection_ratio(),
            'session_id': self.session_id
        }
    
    async def prepare_for_training(self) -> Dict[str, int]:
        """
        Prepare collected data for training.
        Returns count of images in each category.
        """
        # Final metadata save
        self._save_session_metadata()
        
        # Count images
        felix_count = len(list(self.felix_dir.glob("*.jpg")))
        notfelix_count = len(list(self.notfelix_dir.glob("*.jpg")))
        
        logger.info(f"Collection complete. Felix: {felix_count}, Not Felix: {notfelix_count}")
        
        # Create training metadata file
        training_meta = {
            'prepared_at': datetime.now().isoformat(),
            'felix_count': felix_count,
            'notfelix_count': notfelix_count,
            'total_images': felix_count + notfelix_count,
            'collection_sessions': [self.session_id],
            'ready_for_training': True
        }
        
        training_meta_path = self.output_dir / "training_metadata.json"
        with open(training_meta_path, 'w') as f:
            json.dump(training_meta, f, indent=2)
        
        return {
            'felix': felix_count,
            'notfelix': notfelix_count,
            'total': felix_count + notfelix_count
        }


# Integration function for the video system
async def create_collector_callback(collector: EdgeCaseCollector):
    """
    Creates a callback function for the video client that includes frame data.
    """
    async def callback(detections: List[Dict], frame: np.ndarray = None):
        if frame is not None:
            await collector.process_detection_update(detections, frame)
    
    return callback