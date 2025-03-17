# streamaudio.py - Browser-based TTS player with file-based logging
import http.server
import socketserver
import threading
import webbrowser
import json
import time
import os
import urllib.parse
import logging
from datetime import datetime

# Configure logging to file, instead of having the logs appear in the terminal I have them be in file.
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)

log_file = os.path.join(log_dir, "tts_player.log")
logger = logging.getLogger("tts_player")
logger.setLevel(logging.INFO)


# File handler for logging
file_handler = logging.FileHandler(log_file)
file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
logger.addHandler(file_handler)

# Disable propagation to root logger (to avoid console output)
logger.propagate = False


# Configuration
TTS_SERVER = os.getenv("TTS_SERVER", "http://localhost:7851")
DEFAULT_VOICE = "female_01.wav"  # Initial default voice
DEFAULT_LANGUAGE = "en"
WEB_PORT = 8765
QUEUE_FILE = "tts_queue.json"
SETTINGS_FILE = "tts_settings.json"

# Global state
server_thread = None
server_started = False
browser_opened = False
current_voice = DEFAULT_VOICE  # Global variable to store current voice

# Available voices - you can add more based on what's available in your TTS system
AVAILABLE_VOICES = [
    {"id": "female_01.wav", "name": "Female 1"},
    {"id": "female_02.wav", "name": "Female 2"},
    {"id": "female_03.wav", "name": "Female 3"},
    {"id": "female_04.wav", "name": "Female 4"},
    {"id": "female_05.wav", "name": "Female 5"},
    {"id": "male_01.wav", "name": "Male 1"},
    {"id": "male_02.wav", "name": "Male 2"},
    {"id": "male_03.wav", "name": "Male 3"},
    {"id": "Morgan_Freeman CC3.wav", "name": "Morgan Freeman"}
]

def ensure_files():
    """Make sure the queue and settings files exist"""
    if not os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, 'w') as f:
            json.dump({"queue": []}, f)
        logger.info(f"Created queue file: {QUEUE_FILE}")
    
    if not os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({"default_voice": DEFAULT_VOICE}, f)
        logger.info(f"Created settings file: {SETTINGS_FILE}")

def load_settings():
    """Load settings from the settings file"""
    global current_voice
    
    ensure_files()
    
    try:
        with open(SETTINGS_FILE, 'r') as f:
            settings = json.load(f)
            current_voice = settings.get("default_voice", DEFAULT_VOICE)
        logger.info(f"Settings loaded. Current voice: {current_voice}")
    except Exception as e:
        logger.error(f"Error loading settings: {e}")

def save_settings():
    """Save current settings to the settings file"""
    ensure_files()
    
    try:
        with open(SETTINGS_FILE, 'w') as f:
            json.dump({"default_voice": current_voice}, f)
        logger.info(f"Settings saved. Default voice: {current_voice}")
    except Exception as e:
        logger.error(f"Error saving settings: {e}")

def get_available_voices():
    """Return a list of available voices"""
    return AVAILABLE_VOICES

def set_voice(voice_id):
    """
    Set the default voice to use for all TTS
    
    Args:
        voice_id (str): ID of the voice to use (e.g. "female_01.wav")
    
    Returns:
        bool: True if successful, False if voice not found
    """
    global current_voice
    
    # Check if the voice exists
    voice_exists = any(voice["id"] == voice_id for voice in AVAILABLE_VOICES)
    
    if voice_exists:
        current_voice = voice_id
        save_settings()
        logger.info(f"Default voice set to: {voice_id}")
        return True
    else:
        logger.warning(f"Voice '{voice_id}' not found. Using current voice: {current_voice}")
        return False

def start_server():
    """Start the web server if it's not already running"""
    global server_thread, server_started, browser_opened
    
    # Make sure the files exist
    ensure_files()
    
    # Load settings
    load_settings()
    
    if not server_started:
        logger.info("Starting TTS player server...")
        
        # Create a handler for the web server
        class TTSHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                # Serve main player page
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(get_player_html().encode())
                
                # API to get the queue
                elif self.path == '/queue':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    with open(QUEUE_FILE, 'r') as f:
                        self.wfile.write(f.read().encode())
                
                # API to get available voices
                elif self.path == '/voices':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps(AVAILABLE_VOICES).encode())
                
                # API to get current voice
                elif self.path == '/current-voice':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"voice": current_voice}).encode())
                
                # API to get the next item
                elif self.path == '/next':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    with open(QUEUE_FILE, 'r') as f:
                        queue_data = json.load(f)
                    
                    if queue_data["queue"]:
                        next_item = queue_data["queue"][0]
                        queue_data["queue"] = queue_data["queue"][1:]
                        
                        with open(QUEUE_FILE, 'w') as f:
                            json.dump(queue_data, f)
                        
                        self.wfile.write(json.dumps(next_item).encode())
                    else:
                        self.wfile.write(json.dumps({"empty": True}).encode())
                
                # All other paths
                else:
                    return http.server.SimpleHTTPRequestHandler.do_GET(self)
            
            def do_POST(self):
                # API to set current voice
                if self.path == '/set-voice':
                    content_length = int(self.headers['Content-Length'])
                    post_data = self.rfile.read(content_length)
                    voice_data = json.loads(post_data)
                    
                    global current_voice
                    new_voice = voice_data.get("voice")
                    
                    if new_voice:
                        set_voice(new_voice)
                    
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    self.wfile.write(json.dumps({"success": True, "voice": current_voice}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def do_DELETE(self):
                # API to clear the queue
                if self.path == '/queue':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    
                    with open(QUEUE_FILE, 'w') as f:
                        json.dump({"queue": []}, f)
                    
                    self.wfile.write(json.dumps({"success": True}).encode())
                else:
                    self.send_response(404)
                    self.end_headers()
            
            def log_message(self, format, *args):
                # Use our file logger instead of console
                if not any(path in args[1] for path in ['/next', '/queue', '/voices', '/current-voice']):
                    logger.info(f"HTTP: {format % args}")
        
        # Start the server
        try:
            socketserver.TCPServer.allow_reuse_address = True
            httpd = socketserver.TCPServer(("", WEB_PORT), TTSHandler)
            
            server_thread = threading.Thread(target=httpd.serve_forever)
            server_thread.daemon = True
            server_thread.start()
            server_started = True
            
            logger.info(f"TTS player server started on port {WEB_PORT}")
            logger.info(f"Using voice: {current_voice}")
        except Exception as e:
            logger.error(f"Error starting server: {e}")

def say(text, voice=None, language=None):
    """
    Add text to the TTS queue to be spoken
    
    Args:
        text (str): Text to speak
        voice (str): Voice to use (defaults to current global voice)
        language (str): Language code (defaults to DEFAULT_LANGUAGE)
    """
    global browser_opened
    
    # If no text, don't proceed
    if not text or text.strip() == "":
        logger.warning("Empty text provided to say() function. Not opening browser.")
        return False
    
    # Start server if needed
    if not server_started:
        start_server()
        time.sleep(0.5)  # Give the server a moment to start
    
    # Open browser if it hasn't been opened yet and we have actual text to speak
    if not browser_opened and text.strip():
        webbrowser.open(f"http://localhost:{WEB_PORT}")
        browser_opened = True
        logger.info(f"TTS player opened in browser at http://localhost:{WEB_PORT}")
    
    # Set defaults
    voice = voice or current_voice
    language = language or DEFAULT_LANGUAGE
    
    # Add to queue
    with open(QUEUE_FILE, 'r') as f:
        queue_data = json.load(f)
    
    queue_data["queue"].append({
        "text": text,
        "voice": voice,
        "language": language,
        "timestamp": time.time()
    })
    
    with open(QUEUE_FILE, 'w') as f:
        json.dump(queue_data, f)
    
    logger.info(f"Added to TTS queue: {text[:50]}{'...' if len(text) > 50 else ''}")
    return True

def get_player_html():
    """Return the HTML for the browser player"""
    return f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>AI Voice Player</title>
        <style>
            body {{ 
                font-family: Arial, sans-serif; 
                margin: 0; 
                padding: 20px; 
                background: #f0f0f0;
                display: flex;
                justify-content: center;
            }}
            .container {{ 
                max-width: 600px;
                width: 100%;
                padding: 20px; 
                border-radius: 8px; 
                background: white; 
                box-shadow: 0 2px 10px rgba(0,0,0,0.1); 
            }}
            h1 {{ color: #333; margin-top: 0; }}
            .player {{ margin: 20px 0; }}
            .controls {{ margin: 20px 0; display: flex; gap: 10px; flex-wrap: wrap; }}
            button {{ 
                padding: 10px 15px; 
                background: #4CAF50; 
                color: white; 
                border: none; 
                border-radius: 4px; 
                cursor: pointer; 
            }}
            button:hover {{ background: #45a049; }}
            #pauseBtn {{ background: #f44336; }}
            #pauseBtn:hover {{ background: #d32f2f; }}
            .voice-selector {{
                margin: 20px 0;
                padding: 10px;
                background: #f5f5f5;
                border-radius: 4px;
            }}
            select {{
                padding: 8px;
                margin-right: 10px;
                border-radius: 4px;
                border: 1px solid #ddd;
            }}
            .status {{ 
                padding: 10px; 
                background: #e7f3ff; 
                border-radius: 4px; 
                margin-bottom: 20px; 
            }}
            .queue {{ 
                padding: 10px; 
                background: #f9f9f9; 
                border-radius: 4px; 
                margin-top: 20px; 
                max-height: 200px; 
                overflow-y: auto; 
            }}
            .queue-item {{ 
                padding: 8px; 
                margin: 5px 0; 
                background: #eee; 
                border-radius: 4px; 
            }}
            .current {{ 
                background: #e7f3ff; 
                border-left: 4px solid #2196F3; 
                padding-left: 10px; 
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>AI Voice Player</h1>
            
            <div class="status" id="statusBox">
                Click "Play" to start the voice queue. It will automatically play all messages.
            </div>
            
            <div class="voice-selector">
                <label for="voiceSelect"><strong>Voice:</strong></label>
                <select id="voiceSelect"></select>
                <button id="setVoiceBtn">Set as Default</button>
            </div>
            
            <div class="player">
                <audio id="audioPlayer" controls>
                    Your browser does not support the audio element.
                </audio>
            </div>
            
            <div class="controls">
                <button id="playBtn">Play</button>
                <button id="pauseBtn">Pause</button>
                <button id="skipBtn">Skip</button>
                <button id="clearBtn">Clear Queue</button>
                <button id="refreshBtn">Refresh</button>
            </div>
            
            <div>
                <label for="autoplayToggle">
                    <input type="checkbox" id="autoplayToggle" checked> 
                    Autoplay new items
                </label>
            </div>
            
            <div class="queue">
                <h3>Queue</h3>
                <div id="queueItems">
                    <div class="queue-item">No items in queue</div>
                </div>
            </div>
            
            <script>
                // DOM elements
                const audioPlayer = document.getElementById('audioPlayer');
                const statusBox = document.getElementById('statusBox');
                const playBtn = document.getElementById('playBtn');
                const pauseBtn = document.getElementById('pauseBtn');
                const skipBtn = document.getElementById('skipBtn');
                const clearBtn = document.getElementById('clearBtn');
                const refreshBtn = document.getElementById('refreshBtn');
                const autoplayToggle = document.getElementById('autoplayToggle');
                const queueItems = document.getElementById('queueItems');
                const voiceSelect = document.getElementById('voiceSelect');
                const setVoiceBtn = document.getElementById('setVoiceBtn');
                
                // State
                let isPlaying = false;
                let currentItem = null;
                let checking = false;
                let currentVoice = "";
                let availableVoices = [];
                
                // Load voices
                async function loadVoices() {{
                    try {{
                        // Get available voices
                        const voicesResponse = await fetch('/voices');
                        availableVoices = await voicesResponse.json();
                        
                        // Get current voice
                        const currentVoiceResponse = await fetch('/current-voice');
                        const currentVoiceData = await currentVoiceResponse.json();
                        currentVoice = currentVoiceData.voice;
                        
                        // Populate voice selector
                        voiceSelect.innerHTML = '';
                        availableVoices.forEach(voice => {{
                            const option = document.createElement('option');
                            option.value = voice.id;
                            option.text = voice.name;
                            if (voice.id === currentVoice) {{
                                option.selected = true;
                            }}
                            voiceSelect.appendChild(option);
                        }});
                        
                        console.log("Voices loaded. Current voice:", currentVoice);
                    }} catch (error) {{
                        console.error("Error loading voices:", error);
                    }}
                }}
                
                // Set voice
                async function setVoice() {{
                    const newVoice = voiceSelect.value;
                    
                    try {{
                        const response = await fetch('/set-voice', {{
                            method: 'POST',
                            headers: {{
                                'Content-Type': 'application/json'
                            }},
                            body: JSON.stringify({{ voice: newVoice }})
                        }});
                        
                        const data = await response.json();
                        if (data.success) {{
                            currentVoice = data.voice;
                            statusBox.innerHTML = `<strong>Voice set to:</strong> ${{newVoice.replace('.wav', '')}}`;
                            console.log("Voice set to:", currentVoice);
                        }}
                    }} catch (error) {{
                        console.error("Error setting voice:", error);
                        statusBox.innerHTML = `<strong>Error setting voice:</strong> ${{error.message}}`;
                    }}
                }}
                
                // Check for new items every 1 second
                const CHECK_INTERVAL = 1000;
                
                // Play audio
                function playAudio(item) {{
                    currentItem = item;
                    const text = item.text;
                    const voice = item.voice;
                    
                    console.log("Playing audio:", item);
                    
                    // Create TTS URL with GET parameters
                    const params = new URLSearchParams();
                    params.append('text', text);
                    params.append('voice', voice);
                    params.append('language', item.language || 'en');
                    params.append('output_file', `tts_output_${{Date.now()}}.wav`);
                    
                    const url = `{TTS_SERVER}/api/tts-generate-streaming?${{params.toString()}}`;
                    console.log("Generated URL:", url);
                    
                    // Update player
                    try {{
                        audioPlayer.src = url;
                        
                        // Play and catch any errors
                        audioPlayer.play()
                            .then(() => console.log("Play started"))
                            .catch(e => console.error("Play error:", e));
                    }} catch (e) {{
                        console.error("Error setting audio source:", e);
                    }}
                    
                    // Update status
                    statusBox.innerHTML = `
                        <strong>Now playing:</strong><br>
                        Text: ${{text}}<br>
                        Voice: ${{voice.replace('.wav', '')}}
                    `;
                    
                    isPlaying = true;
                    updateQueueDisplay();
                }}
                
                // Check for new items in the queue
                async function checkQueue() {{
                    if (checking) return;
                    checking = true;
                    
                    try {{
                        // Get next item if we're not currently playing
                        if (!isPlaying && autoplayToggle.checked) {{
                            const response = await fetch('/next');
                            const data = await response.json();
                            
                            if (!data.empty) {{
                                playAudio(data);
                            }}
                        }}
                        
                        // Update queue display
                        updateQueueDisplay();
                    }} catch (error) {{
                        console.error("Error checking queue:", error);
                    }} finally {{
                        checking = false;
                    }}
                }}
                
                // Update the queue display
                async function updateQueueDisplay() {{
                    try {{
                        const response = await fetch('/queue');
                        const data = await response.json();
                        
                        // Clear the queue display
                        queueItems.innerHTML = '';
                        
                        // Add current item if playing
                        if (currentItem) {{
                            const currentItemDiv = document.createElement('div');
                            currentItemDiv.className = 'queue-item current';
                            currentItemDiv.innerHTML = `
                                <strong>Now playing:</strong> ${{currentItem.text}}
                                (Voice: ${{currentItem.voice.replace('.wav', '')}})
                            `;
                            queueItems.appendChild(currentItemDiv);
                        }}
                        
                        // Add queue items
                        if (data.queue && data.queue.length > 0) {{
                            data.queue.forEach((item, index) => {{
                                const itemDiv = document.createElement('div');
                                itemDiv.className = 'queue-item';
                                itemDiv.innerHTML = `
                                    <strong>${{index + 1}}:</strong> ${{item.text}}
                                    (Voice: ${{item.voice.replace('.wav', '')}})
                                `;
                                queueItems.appendChild(itemDiv);
                            }});
                        }} else if (!currentItem) {{
                            queueItems.innerHTML = '<div class="queue-item">No items in queue</div>';
                        }}
                    }} catch (error) {{
                        console.error("Error updating queue display:", error);
                    }}
                }}
                
                // Event listeners
                audioPlayer.addEventListener('error', (e) => {{
                    console.error("Audio error event:", e);
                    statusBox.innerHTML = `
                        <strong>Error playing audio:</strong><br>
                        ${{audioPlayer.error ? audioPlayer.error.message : 'Unknown error'}}
                    `;
                    statusBox.style.background = '#ffebee';
                }});
                
                audioPlayer.addEventListener('ended', () => {{
                    console.log("Audio ended");
                    isPlaying = false;
                    currentItem = null;
                    statusBox.textContent = 'Finished playing, checking for more items...';
                    statusBox.style.background = '#e7f3ff';
                    // Check for more items immediately
                    checkQueue();
                }});
                
                playBtn.addEventListener('click', () => {{
                    if (audioPlayer.src) {{
                        console.log("Play button clicked with existing source");
                        audioPlayer.play().catch(e => console.error("Play error:", e));
                        isPlaying = true;
                    }} else {{
                        console.log("Play button clicked, checking queue");
                        checkQueue();
                    }}
                }});
                
                pauseBtn.addEventListener('click', () => {{
                    console.log("Pause button clicked");
                    audioPlayer.pause();
                    isPlaying = false;
                }});
                
                skipBtn.addEventListener('click', () => {{
                    console.log("Skip button clicked");
                    audioPlayer.pause();
                    isPlaying = false;
                    currentItem = null;
                    checkQueue();
                }});
                
                clearBtn.addEventListener('click', async () => {{
                    console.log("Clear button clicked");
                    try {{
                        await fetch('/queue', {{ method: 'DELETE' }});
                        updateQueueDisplay();
                        statusBox.textContent = 'Queue cleared';
                    }} catch (error) {{
                        console.error("Error clearing queue:", error);
                    }}
                }});
                
                refreshBtn.addEventListener('click', () => {{
                    console.log("Refresh button clicked");
                    updateQueueDisplay();
                }});
                
                setVoiceBtn.addEventListener('click', () => {{
                    console.log("Set voice button clicked");
                    setVoice();
                }});
                
                // Load voices on page load
                loadVoices();
                
                // Start checking the queue regularly
                setInterval(checkQueue, CHECK_INTERVAL);
                
                // Initial setup
                console.log("Player initialized");
                updateQueueDisplay();
            </script>
        </div>
    </body>
    </html>
    """

def clear_queue():
    """Clear the TTS queue"""
    ensure_files()
    with open(QUEUE_FILE, 'w') as f:
        json.dump({"queue": []}, f)
    logger.info("TTS queue cleared")

# Create a simple function to show minimal info for the user
def status_message(message):
    """Show a status message to the user without detailed logging"""
    print(message)
    logger.info(message)

# Note: This automatic initialization is removed to prevent browser opening on import
# Instead, the browser will only open when say() is actually called with text