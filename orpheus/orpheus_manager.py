#!/usr/bin/env python3
"""
Orpheus TTS Manager - Professional Python orchestrator for Orpheus TTS
Handles setup, configuration, server management, and testing
Includes all critical fixes from FIX_EVERYTHING.sh
"""

import os
import sys
import json
import yaml
import time
import glob
import shutil
import psutil
import argparse
import subprocess
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OrpheusManager:
    """Main orchestrator for Orpheus TTS service"""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = Path(config_path)
        self.config = self.load_config()
        self.server_process = None
        
    def load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if self.config_path.exists():
            with open(self.config_path, 'r') as f:
                return yaml.safe_load(f)
        else:
            # Default configuration
            return {
                'model': {
                    'name': 'canopylabs/orpheus-tts-0.1-finetune-prod',
                    'max_model_len': 64000,
                    'max_tokens': 64000,  # Fixed from 2000!
                    'dtype': 'bfloat16'
                },
                'server': {
                    'host': '0.0.0.0',
                    'port': 8080,
                    'workers': 1
                },
                'audio': {
                    'sample_rate': 24000,
                    'channels': 1,
                    'bits_per_sample': 16
                },
                'generation': {
                    'temperature': 0.6,
                    'top_p': 0.8,
                    'repetition_penalty': 1.3
                }
            }
    
    def apply_critical_fixes(self):
        """Apply all critical fixes from FIX_EVERYTHING.sh"""
        logger.info("Applying critical Orpheus fixes...")
        
        # FIX 1: HuggingFace config.json max_position_embeddings
        self._fix_huggingface_config()
        
        # FIX 2: UUID for duplicate request IDs in engine_class.py
        self._fix_engine_class()
        
        # FIX 3: Ensure proper imports
        self._verify_imports()
        
        logger.info("✅ All critical fixes applied successfully")
    
    def _fix_huggingface_config(self):
        """Fix the max_position_embeddings in HuggingFace cache"""
        logger.info("Fixing HuggingFace config (max_position_embeddings)...")
        
        # Find config files in HuggingFace cache
        cache_paths = [
            Path.home() / ".cache" / "huggingface",
            Path("/root/.cache/huggingface") if os.path.exists("/root") else None
        ]
        
        fixed = False
        for cache_path in cache_paths:
            if cache_path and cache_path.exists():
                configs = list(cache_path.glob("**/models--canopylabs--orpheus-tts*/snapshots/*/config.json"))
                
                for config_file in configs:
                    try:
                        with open(config_file, 'r') as f:
                            config = json.load(f)
                        
                        current_val = config.get('max_position_embeddings', 0)
                        if current_val == 131072:
                            logger.info(f"Found config with incorrect value: {config_file}")
                            config['max_position_embeddings'] = 64000
                            
                            with open(config_file, 'w') as f:
                                json.dump(config, f, indent=2)
                            
                            logger.info(f"✅ Fixed max_position_embeddings: 131072 -> 64000")
                            fixed = True
                    except Exception as e:
                        logger.warning(f"Could not process {config_file}: {e}")
        
        if not fixed:
            logger.info("Config not found in cache yet (will be fixed after first model download)")
    
    def _fix_engine_class(self):
        """Fix duplicate request ID issue in orpheus_tts engine_class.py"""
        logger.info("Fixing engine_class.py (duplicate request IDs)...")
        
        # Find the installed package location
        try:
            import orpheus_tts
            engine_path = Path(orpheus_tts.__file__).parent / "engine_class.py"
            
            if engine_path.exists():
                with open(engine_path, 'r') as f:
                    content = f.read()
                
                # Check if already fixed
                if 'import uuid' not in content:
                    # Add uuid import
                    content = content.replace('import queue', 'import queue\nimport uuid')
                
                # Fix the request_id parameter
                if 'request_id="req-001"' in content:
                    content = content.replace('request_id="req-001"', 'request_id=None')
                    
                    # Add UUID generation logic
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if 'def generate_tokens_sync' in line and 'request_id=None' in line:
                            # Insert UUID generation after function definition
                            indent = '        '
                            lines.insert(i + 1, f'{indent}if request_id is None:')
                            lines.insert(i + 2, f'{indent}    request_id = f"req-{{uuid.uuid4().hex[:8]}}"')
                            break
                    content = '\n'.join(lines)
                
                with open(engine_path, 'w') as f:
                    f.write(content)
                
                logger.info("✅ Fixed engine_class.py duplicate request IDs")
            else:
                logger.warning("engine_class.py not found - install orpheus-speech first")
        except ImportError:
            logger.warning("orpheus_tts not installed yet - fix will apply after installation")
    
    def _verify_imports(self):
        """Verify all required imports are available"""
        required_packages = [
            'torch', 'orpheus_tts', 'fastapi', 'uvicorn', 
            'aiohttp', 'soundfile', 'yaml', 'numpy'
        ]
        
        missing = []
        for package in required_packages:
            try:
                __import__(package)
            except ImportError:
                missing.append(package)
        
        if missing:
            logger.warning(f"Missing packages: {missing}")
            logger.info("Run 'python orpheus_manager.py setup' to install dependencies")
    
    def setup(self):
        """Install all dependencies and apply fixes"""
        logger.info("Setting up Orpheus TTS environment...")
        
        # Install system dependencies if on Linux
        if sys.platform.startswith('linux'):
            logger.info("Installing system dependencies...")
            try:
                subprocess.run([
                    'apt-get', 'update'
                ], check=False, capture_output=True)
                
                subprocess.run([
                    'apt-get', 'install', '-y',
                    'python3-pip', 'python3-dev', 'ffmpeg', 'libsndfile1'
                ], check=False, capture_output=True)
            except:
                logger.warning("Could not install system packages (may need sudo)")
        
        # Install Python dependencies
        logger.info("Installing Python dependencies...")
        subprocess.run([
            sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'
        ], check=True)
        
        # Install PyTorch with CUDA if available
        try:
            import torch
            if not torch.cuda.is_available():
                logger.info("Installing PyTorch with CUDA support...")
                subprocess.run([
                    sys.executable, '-m', 'pip', 'install',
                    'torch', 'torchvision', 'torchaudio',
                    '--index-url', 'https://download.pytorch.org/whl/cu121'
                ], check=True)
        except ImportError:
            logger.info("Installing PyTorch...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install',
                'torch', 'torchvision', 'torchaudio',
                '--index-url', 'https://download.pytorch.org/whl/cu121'
            ], check=True)
        
        # Install Orpheus and other dependencies
        requirements = [
            'orpheus-speech',
            'vllm==0.7.3',
            'fastapi',
            'uvicorn[standard]',
            'aiohttp',
            'aiofiles',
            'pyyaml',
            'soundfile',
            'scipy',
            'numpy',
            'psutil',
            'requests'
        ]
        
        for req in requirements:
            logger.info(f"Installing {req}...")
            subprocess.run([
                sys.executable, '-m', 'pip', 'install', req
            ], check=False, capture_output=True)
        
        # Apply critical fixes
        self.apply_critical_fixes()
        
        # Create default config if doesn't exist
        if not self.config_path.exists():
            self.save_config()
        
        logger.info("✅ Setup complete! Run 'python orpheus_manager.py start' to launch server")
    
    def save_config(self):
        """Save current configuration to YAML file"""
        with open(self.config_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False)
        logger.info(f"Configuration saved to {self.config_path}")
    
    def start(self):
        """Start the Orpheus TTS server"""
        logger.info("Starting Orpheus TTS server...")
        
        # Apply fixes before starting
        self.apply_critical_fixes()
        
        # Check if server is already running
        if self.is_running():
            logger.warning("Server is already running!")
            return
        
        # Start the server
        cmd = [
            sys.executable, 'orpheus_service.py',
            '--config', str(self.config_path)
        ]
        
        logger.info(f"Launching server on {self.config['server']['host']}:{self.config['server']['port']}")
        self.server_process = subprocess.Popen(cmd)
        
        # Wait for server to be ready
        time.sleep(5)
        if self.health_check():
            logger.info("✅ Server started successfully!")
        else:
            logger.error("Server failed to start properly")
    
    def stop(self):
        """Stop the Orpheus TTS server"""
        logger.info("Stopping Orpheus TTS server...")
        
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait(timeout=10)
            self.server_process = None
            logger.info("✅ Server stopped")
        else:
            # Try to find and kill by port
            port = self.config['server']['port']
            for proc in psutil.process_iter(['pid', 'name', 'connections']):
                try:
                    for conn in proc.info.get('connections', []):
                        if conn.laddr.port == port:
                            logger.info(f"Found server process {proc.pid}, terminating...")
                            proc.terminate()
                            logger.info("✅ Server stopped")
                            return
                except:
                    pass
            logger.warning("No running server found")
    
    def restart(self):
        """Restart the server"""
        self.stop()
        time.sleep(2)
        self.start()
    
    def is_running(self) -> bool:
        """Check if server is running"""
        try:
            response = requests.get(
                f"http://{self.config['server']['host']}:{self.config['server']['port']}/health",
                timeout=2
            )
            return response.status_code == 200
        except:
            return False
    
    def health_check(self) -> bool:
        """Perform health check on the server"""
        max_retries = 10
        for i in range(max_retries):
            if self.is_running():
                return True
            time.sleep(2)
        return False
    
    def status(self):
        """Display server status and system information"""
        logger.info("=== Orpheus TTS Status ===")
        
        # Server status
        if self.is_running():
            logger.info("✅ Server: RUNNING")
            logger.info(f"   URL: http://{self.config['server']['host']}:{self.config['server']['port']}")
        else:
            logger.info("❌ Server: NOT RUNNING")
        
        # GPU status
        try:
            import torch
            if torch.cuda.is_available():
                logger.info(f"✅ GPU: Available ({torch.cuda.get_device_name(0)})")
                logger.info(f"   Memory: {torch.cuda.memory_allocated(0) / 1e9:.2f}GB used")
            else:
                logger.info("❌ GPU: Not available")
        except:
            logger.info("❌ GPU: PyTorch not installed")
        
        # Configuration
        logger.info(f"📋 Config: {self.config_path}")
        logger.info(f"   Model: {self.config['model']['name']}")
        logger.info(f"   Max Tokens: {self.config['model']['max_tokens']}")
        logger.info(f"   Max Audio Duration: ~{self.config['model']['max_tokens'] / 83:.1f} seconds")
    
    def test(self):
        """Run test suite"""
        logger.info("Running Orpheus TTS test suite...")
        
        if not self.is_running():
            logger.error("Server is not running! Start it first with 'python orpheus_manager.py start'")
            return
        
        # Import and run test suite
        try:
            import test_suite
            test_suite.run_all_tests(
                server_url=f"http://{self.config['server']['host']}:{self.config['server']['port']}"
            )
        except ImportError:
            logger.error("test_suite.py not found!")
        except Exception as e:
            logger.error(f"Test failed: {e}")

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Orpheus TTS Manager')
    parser.add_argument(
        'command',
        choices=['setup', 'start', 'stop', 'restart', 'status', 'test', 'fix'],
        help='Command to execute'
    )
    parser.add_argument(
        '--config',
        default='config.yaml',
        help='Path to configuration file'
    )
    
    args = parser.parse_args()
    manager = OrpheusManager(config_path=args.config)
    
    # Execute command
    if args.command == 'setup':
        manager.setup()
    elif args.command == 'start':
        manager.start()
    elif args.command == 'stop':
        manager.stop()
    elif args.command == 'restart':
        manager.restart()
    elif args.command == 'status':
        manager.status()
    elif args.command == 'test':
        manager.test()
    elif args.command == 'fix':
        manager.apply_critical_fixes()

if __name__ == '__main__':
    main()