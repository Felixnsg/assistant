# streamaudio.py - Fixed version with proper file existence check

import http.server
import socketserver
import threading
import webbrowser
import urllib.parse
import json
import time
import os
import queue

# Global settings
TTS_SERVER = "http://127.0.0.1:7851"
DEFAULT_VOICE = "female_02.wav"
DEFAULT_LANGUAGE = "en"
WEB_PORT = 8765  # Port for our local web server
QUEUE_FILE = "tts_queue.json"  # File to store the queue

# Queue for TTS requests
tts_queue = queue.Queue()
server_thread = None
server_started = False

def ensure_queue_file_exists():
    """Make sure the queue file exists"""
    if not os.path.exists(QUEUE_FILE):
        with open(QUEUE_FILE, 'w') as f:
            json.dump({"queue": []}, f)

def start_server_if_needed():
    """Start the web server if it's not already running"""
    global server_thread, server_started
    
    # Make sure the queue file exists
    ensure_queue_file_exists()
    
    if not server_started:
        # Create a handler that serves the TTS player and handles API requests
        class TTSServerHandler(http.server.SimpleHTTPRequestHandler):
            def do_GET(self):
                # Serve the main player page
                if self.path == '/':
                    self.send_response(200)
                    self.send_header('Content-type', 'text/html')
                    self.end_headers()
                    self.wfile.write(self.get_player_html().encode())
                
                # API endpoint to get the current queue
                elif self.path == '/queue':
                    self.send_response(200)
                    self.send_header('Content-type', 'application/json')
                    self.end_headers()
                    with open(QUEUE_FILE, 'r') as f:
                        self.wfile.write(f.read().encode())
                
                # API endpoint to get next item in queue
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
                
                # All other paths, serve files from current directory
                else:
                    return http.server.SimpleHTTPRequestHandler.do_GET(self)
            
            def do_DELETE(self):
                # API endpoint to clear the queue
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
            
            def get_player_html(self):
                """Return the HTML for the persistent player"""
                return f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Continuous TTS Player</title>
                    <style>
                        body {{ font-family: Arial, sans-serif; margin: 0; padding: 20px; background: #f0f0f0; }}
                        .container {{ max-width: 700px; margin: 0 auto; padding: 20px; border-radius: 8px; background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                        h1 {{ color: #333; margin-top: 0; }}
                        .player {{ margin: 20px 0; }}
                        .controls {{ margin: 20px 0; display: flex; gap: 10px; }}
                        button {{ padding: 10px 15px; background: #4CAF50; color: white; border: none; border-radius: 4px; cursor: pointer; }}
                        button:hover {{ background: #45a049; }}
                        #pauseBtn {{ background: #f44336; }}
                        #pauseBtn:hover {{ background: #d32f2f; }}
                        .status {{ padding: 10px; background: #e7f3ff; border-radius: 4px; margin-bottom: 20px; }}
                        .queue {{ padding: 10px; background: #f9f9f9; border-radius: 4px; margin-top: 20px; max-height: 200px; overflow-y: auto; }}
                        .queue-item {{ padding: 8px; margin: 5px 0; background: #eee; border-radius: 4px; }}
                        .current {{ background: #e7f3ff; border-left: 4px solid #2196F3; padding-left: 10px; }}
                        .hidden {{ display: none; }}
                    </style>
                </head>
                <body>
                    <div class="container">
                        <h1>Continuous TTS Player</h1>
                        
                        <div class="status" id="statusBox">
                            Waiting for TTS requests...
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
                            
                            // State
                            let isPlaying = false;
                            let currentItem = null;
                            let checking = false;
                            
                            // Check for new items every 1 second
                            const CHECK_INTERVAL = 1000;
                            
                            // Play audio
                            function playAudio(item) {{
                                currentItem = item;
                                const text = item.text;
                                const voice = item.voice;
                                
                                // Create TTS URL
                                const params = new URLSearchParams();
                                params.append('text', text);
                                params.append('voice', voice || '{DEFAULT_VOICE}');
                                params.append('language', item.language || '{DEFAULT_LANGUAGE}');
                                params.append('output_file', `tts_output_${{Date.now()}}.wav`);
                                
                                const url = `{TTS_SERVER}/api/tts-generate-streaming?${{params.toString()}}`;
                                
                                // Update player
                                audioPlayer.src = url;
                                audioPlayer.play().catch(e => console.error('Error playing:', e));
                                
                                // Update status
                                statusBox.innerHTML = `
                                    <strong>Now playing:</strong><br>
                                    Text: ${{text}}<br>
                                    Voice: ${{voice || '{DEFAULT_VOICE}'}}
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
                                    console.error('Error checking queue:', error);
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
                                            (Voice: ${{currentItem.voice || '{DEFAULT_VOICE}'}})
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
                                                (Voice: ${{item.voice || '{DEFAULT_VOICE}'}})
                                            `;
                                            queueItems.appendChild(itemDiv);
                                        }});
                                    }} else if (!currentItem) {{
                                        queueItems.innerHTML = '<div class="queue-item">No items in queue</div>';
                                    }}
                                }} catch (error) {{
                                    console.error('Error updating queue display:', error);
                                }}
                            }}
                            
                            // Event listeners
                            audioPlayer.addEventListener('ended', () => {{
                                isPlaying = false;
                                currentItem = null;
                                statusBox.textContent = 'Finished playing, checking for more items...';
                                // Check for more items immediately
                                checkQueue();
                            }});
                            
                            playBtn.addEventListener('click', () => {{
                                if (audioPlayer.src) {{
                                    audioPlayer.play();
                                    isPlaying = true;
                                }} else {{
                                    checkQueue();
                                }}
                            }});
                            
                            pauseBtn.addEventListener('click', () => {{
                                audioPlayer.pause();
                                isPlaying = false;
                            }});
                            
                            skipBtn.addEventListener('click', async () => {{
                                audioPlayer.pause();
                                isPlaying = false;
                                currentItem = null;
                                checkQueue();
                            }});
                            
                            clearBtn.addEventListener('click', async () => {{
                                try {{
                                    await fetch('/queue', {{ method: 'DELETE' }});
                                    updateQueueDisplay();
                                }} catch (error) {{
                                    console.error('Error clearing queue:', error);
                                }}
                            }});
                            
                            refreshBtn.addEventListener('click', () => {{
                                updateQueueDisplay();
                            }});
                            
                            // Start checking the queue
                            setInterval(checkQueue, CHECK_INTERVAL);
                            
                            // Initial queue check
                            updateQueueDisplay();
                        </script>
                    </div>
                </body>
                </html>
                """
        
        # Start the server
        socketserver.TCPServer.allow_reuse_address = True
        httpd = socketserver.TCPServer(("", WEB_PORT), TTSServerHandler)
        
        # Start server in a separate thread
        server_thread = threading.Thread(target=httpd.serve_forever)
        server_thread.daemon = True  # Daemon threads exit when the main program exits
        server_thread.start()
        server_started = True
        
        # Open the player in the browser
        webbrowser.open(f"http://localhost:{WEB_PORT}")
        
        # Wait a moment for the server to start
        time.sleep(0.5)

def say(text, voice=None, language=None):
    """
    Add text to the TTS queue to be spoken automatically
    
    Args:
        text (str): Text to speak
        voice (str): Voice to use (defaults to DEFAULT_VOICE)
        language (str): Language code (defaults to DEFAULT_LANGUAGE)
    """
    # Set defaults
    voice = voice or DEFAULT_VOICE
    language = language or DEFAULT_LANGUAGE
    
    # Make sure the queue file exists first
    ensure_queue_file_exists()
    
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
    
    # Start server if not already running
    start_server_if_needed()

# Example usage
if __name__ == "__main__":
    print("Starting continuous TTS player...")
    print(f"Open http://localhost:{WEB_PORT} in your browser to see the queue")
    
    # Start the server
    start_server_if_needed()
    
    # Example: Add some items to the queue
    say("This is a continuous TTS player that will keep running in the background.")
    time.sleep(1)
    say("You can add new text to the queue at any time, and it will be spoken automatically.")
    time.sleep(1)
    say("Try running this from different Python scripts to see how it works.")
    
    # Keep the script running
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Exiting...")