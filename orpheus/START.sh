#!/bin/bash
# Quick start script for Orpheus TTS Server
# Run this after cloning the repository

set -e

echo "╔══════════════════════════════════════════╗"
echo "║     ORPHEUS TTS SERVER QUICK START       ║"
echo "╚══════════════════════════════════════════╝"
echo ""

# Check if this is first run
if [ ! -f ".setup_complete" ]; then
    echo "📦 First time setup detected..."
    echo "This will install all dependencies (may take 5-10 minutes)"
    echo ""
    read -p "Continue with setup? (y/n): " -n 1 -r
    echo ""
    
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        # Run setup
        chmod +x setup.sh
        ./setup.sh
        touch .setup_complete
        echo "✅ Setup complete!"
    else
        echo "Setup cancelled. Run './setup.sh' manually when ready."
        exit 1
    fi
fi

# Menu for server selection
echo ""
echo "Select server to start:"
echo "1) Production Server (FastAPI) - Recommended"
echo "2) Async Server (Ultra-low latency <200ms)"
echo "3) Simple Server (Flask development)"
echo "4) Run benchmarks"
echo "5) Start monitor only"
echo "6) Exit"
echo ""
read -p "Enter choice [1-6]: " choice

case $choice in
    1)
        echo "🚀 Starting Production Server..."
        echo "Server will be available at: http://0.0.0.0:8080"
        echo "Press Ctrl+C to stop"
        echo ""
        python3 server.py
        ;;
    2)
        echo "⚡ Starting Async Ultra-Low Latency Server..."
        echo "Server will be available at: http://0.0.0.0:8080"
        echo "Press Ctrl+C to stop"
        echo ""
        python3 server_async.py
        ;;
    3)
        echo "🔧 Starting Development Server..."
        echo "Server will be available at: http://0.0.0.0:8080"
        echo "Press Ctrl+C to stop"
        echo ""
        python3 streaming_server.py
        ;;
    4)
        echo "📊 Running benchmarks..."
        python3 benchmark.py
        ;;
    5)
        echo "📈 Starting monitor..."
        python3 monitor.py --url http://localhost:8080
        ;;
    6)
        echo "Exiting..."
        exit 0
        ;;
    *)
        echo "Invalid choice. Exiting..."
        exit 1
        ;;
esac