import asyncio
import logging
from pathlib import Path
from IseeYou.IseeYouClass import FlowControlledClient
from core.cache import VisualContextCache
from enhanced_picture_collector import EdgeCaseCollector, create_collector_callback

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('collector_integration')

class CollectorVideoClient(FlowControlledClient):
    """
    Extended video client that includes frame data in callbacks.
    """
    
    def __init__(self, server_uri: str, target_fps: int = 30, 
                 cache_callback=None, collector_callback=None):
        super().__init__(server_uri, target_fps, cache_callback)
        self.collector_callback = collector_callback
        
    async def _update_cache_with_detections(self, detections, frame=None):
        """Override to include frame data"""
        # Call parent implementation
        await super()._update_cache_with_detections(detections)
        
        # Also call collector callback with frame
        if self.collector_callback and frame is not None:
            try:
                await self.collector_callback(detections, frame)
            except Exception as e:
                logger.error(f"Error in collector callback: {e}")
    
    async def _receive_results(self, websocket):
        """Override to pass frame to callbacks"""
        while self.running:
            try:
                message = await websocket.recv()
                data = json.loads(message)
                
                if data.get('type') == 'result':
                    frame_id = data.get('frame_id')
                    if frame_id and frame_id in self.results_cache:
                        # Calculate RTT
                        rtt = (time.time() - self.results_cache[frame_id]['sent_at']) * 1000
                        self.rtt_history.append(rtt)
                        if len(self.rtt_history) > 20:
                            self.rtt_history.pop(0)
                        
                        self.pending_frames = max(0, self.pending_frames - 1)
                        self.frames_processed += 1
                        
                        # Get detections and frame
                        detections = data.get('detections', [])
                        frame = self.results_cache[frame_id]['frame']
                        
                        # Update cache and collector with frame
                        await self._update_cache_with_detections(detections, frame)
                        
                        # Store results for display
                        self.latest_results = {
                            'frame': frame,
                            'detections': detections,
                            'process_time': data.get('process_time', 0),
                            'rtt': rtt
                        }
                        
                        # Cleanup
                        if len(self.results_cache) > 30:
                            oldest = min(self.results_cache.keys())
                            del self.results_cache[oldest]
                
                elif data.get('type') == 'ping':
                    await websocket.send(json.dumps({'type': 'pong'}))
                
                elif data.get('type') == 'backpressure':
                    logger.warning("Server backpressure signal received")
                    self.max_pending = max(5, self.max_pending - 1)
                    
            except json.JSONDecodeError:
                logger.error("Failed to decode message")
            except Exception as e:
                if self.running:
                    logger.error(f"Error receiving results: {e}")
                break


async def run_collection_session(duration_minutes: int = 10, 
                               output_dir: str = "training_data_collection"):
    """
    Run a data collection session for the specified duration.
    
    Args:
        duration_minutes: How long to collect data
        output_dir: Where to save collected images
    """
    logger.info(f"Starting collection session for {duration_minutes} minutes")
    
    # Create components
    cache = VisualContextCache()
    collector = EdgeCaseCollector(output_dir=output_dir)
    
    # Set cache reference in collector
    collector.set_visual_cache(cache)
    
    # Create video client with both callbacks
    client = CollectorVideoClient(
        server_uri="ws://localhost:8080",
        target_fps=30,
        cache_callback=cache.update_from_client,
        collector_callback=lambda detections, frame: 
            asyncio.create_task(collector.process_detection_update(detections, frame))
    )
    
    try:
        # Start the client
        client_task = asyncio.create_task(client.start())
        
        # Run for specified duration
        await asyncio.sleep(duration_minutes * 60)
        
        # Stop collection
        logger.info("Collection period ended, stopping...")
        await client.stop()
        
        # Get final stats
        stats = collector.get_collection_stats()
        logger.info(f"Collection stats: {stats}")
        
        # Prepare data for training
        training_counts = await collector.prepare_for_training()
        logger.info(f"Data prepared for training: {training_counts}")
        
        return training_counts
        
    except KeyboardInterrupt:
        logger.info("Collection interrupted by user")
        await client.stop()
    except Exception as e:
        logger.error(f"Error during collection: {e}")
        await client.stop()
        raise


async def run_continuous_collection(output_dir: str = "training_data_collection",
                                  auto_train_threshold: int = 1000):
    """
    Run continuous collection with automatic training triggers.
    
    Args:
        output_dir: Where to save collected images
        auto_train_threshold: Trigger training after this many new images
    """
    logger.info("Starting continuous collection mode")
    
    # Keep track of images collected
    total_collected = 0
    
    while True:
        try:
            # Run collection for 1 hour
            counts = await run_collection_session(
                duration_minutes=60,
                output_dir=output_dir
            )
            
            total_collected += counts['total']
            logger.info(f"Total images collected: {total_collected}")
            
            # Check if we should trigger training
            if total_collected >= auto_train_threshold:
                logger.info("Threshold reached! Ready for fine-tuning.")
                
                # Here you would trigger your fine-tuning script
                # For now, just log it
                logger.info(f"Trigger fine-tuning with {total_collected} new images")
                
                # Reset counter after training
                total_collected = 0
                
                # Optional: Move processed images to archive
                # archive_processed_images(output_dir)
            
            # Short break between sessions
            await asyncio.sleep(60)
            
        except KeyboardInterrupt:
            logger.info("Continuous collection stopped by user")
            break
        except Exception as e:
            logger.error(f"Error in continuous collection: {e}")
            await asyncio.sleep(300)  # Wait 5 minutes before retry


# Utility function to trigger fine-tuning
def trigger_fine_tuning(data_path: str, checkpoint_path: str):
    """
    Trigger the fine-tuning process with collected data.
    This is where you'd call your existing fine-tuning code.
    """
    # Import your fine-tuning function
    from FaceDataset import fine_tune_model
    
    # Run fine-tuning
    model, history = fine_tune_model(
        fine_tune_path=data_path,
        checkpoint_path=checkpoint_path,
        epochs=50
    )
    
    return model, history


# Main execution
async def main():
    """Example usage"""
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "continuous":
        # Run continuous collection
        await run_continuous_collection()
    else:
        # Run single session (default 10 minutes)
        duration = int(sys.argv[1]) if len(sys.argv) > 1 else 10
        await run_collection_session(duration_minutes=duration)


if __name__ == "__main__":
    # Add these imports at the top of CollectorVideoClient
    import json
    import time
    
    asyncio.run(main())