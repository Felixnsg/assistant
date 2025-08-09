#!/bin/bash
# vast.ai Deployment Script for Orpheus TTS
# Automates instance rental and deployment

set -e

# Configuration
VAST_API_KEY="${VAST_API_KEY:-your_api_key_here}"
MIN_GPU_RAM=24  # GB (RTX 4090 has 24GB)
MAX_PRICE=0.5   # Max $/hour
DISK_SPACE=50   # GB

echo "============================================"
echo "VAST.AI ORPHEUS TTS DEPLOYMENT"
echo "============================================"

# Install vast CLI if not present
install_vast_cli() {
    if ! command -v vast &> /dev/null; then
        echo "Installing vast.ai CLI..."
        pip install --upgrade vastai
    fi
}

# Search for suitable instances
search_instances() {
    echo "Searching for RTX 4090 instances..."
    
    vast search offers \
        --gpu-name "RTX 4090" \
        --gpu-ram "${MIN_GPU_RAM}-" \
        --disk-space "${DISK_SPACE}-" \
        --max-price "$MAX_PRICE" \
        --cuda-vers "12.1-" \
        --sort-order "price+" \
        --limit 10
}

# Create instance
create_instance() {
    local OFFER_ID=$1
    
    echo "Creating instance from offer $OFFER_ID..."
    
    # Create with PyTorch image
    INSTANCE_ID=$(vast create instance $OFFER_ID \
        --image pytorch/pytorch:2.0.1-cuda12.1-cudnn8-runtime \
        --disk $DISK_SPACE \
        --onstart-cmd "cd /workspace && bash setup.sh" \
        --env PYTHONUNBUFFERED=1 \
        --env CUDA_VISIBLE_DEVICES=0 \
        --ssh \
        --direct)
    
    echo "Instance created: $INSTANCE_ID"
    echo $INSTANCE_ID > .vast_instance_id
}

# Deploy code to instance
deploy_to_instance() {
    local INSTANCE_ID=$1
    
    echo "Waiting for instance to be ready..."
    sleep 30
    
    # Get instance details
    INSTANCE_INFO=$(vast show instance $INSTANCE_ID)
    SSH_HOST=$(echo "$INSTANCE_INFO" | grep -oP 'ssh_host:\s*\K\S+')
    SSH_PORT=$(echo "$INSTANCE_INFO" | grep -oP 'ssh_port:\s*\K\d+')
    
    echo "Instance SSH: root@$SSH_HOST -p $SSH_PORT"
    
    # Copy files to instance
    echo "Deploying code..."
    
    # Create tar archive
    tar -czf orpheus_deploy.tar.gz \
        *.py \
        *.sh \
        *.txt \
        *.yaml \
        *.md \
        Dockerfile \
        docker-compose.yml \
        .env.example
    
    # Copy to instance
    scp -P $SSH_PORT -o StrictHostKeyChecking=no \
        orpheus_deploy.tar.gz \
        root@$SSH_HOST:/workspace/
    
    # Extract and setup
    ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST << 'EOF'
cd /workspace
tar -xzf orpheus_deploy.tar.gz
rm orpheus_deploy.tar.gz
chmod +x setup.sh
./setup.sh
EOF
    
    # Cleanup local archive
    rm orpheus_deploy.tar.gz
    
    echo "Deployment complete!"
}

# Start server on instance
start_server() {
    local INSTANCE_ID=$1
    
    INSTANCE_INFO=$(vast show instance $INSTANCE_ID)
    SSH_HOST=$(echo "$INSTANCE_INFO" | grep -oP 'ssh_host:\s*\K\S+')
    SSH_PORT=$(echo "$INSTANCE_INFO" | grep -oP 'ssh_port:\s*\K\d+')
    
    echo "Starting Orpheus TTS server..."
    
    ssh -p $SSH_PORT -o StrictHostKeyChecking=no root@$SSH_HOST << 'EOF'
cd /workspace
# Start server in tmux for persistence
tmux new-session -d -s orpheus 'python3 server.py'
# Start monitor in another tmux session
tmux new-session -d -s monitor 'python3 monitor.py --url http://localhost:8080'
echo "Server started in tmux session 'orpheus'"
echo "Monitor started in tmux session 'monitor'"
EOF
    
    echo ""
    echo "Server is starting up..."
    echo "SSH: ssh -p $SSH_PORT root@$SSH_HOST"
    echo "API: http://$SSH_HOST:8080"
    echo ""
    echo "To attach to server: tmux attach -t orpheus"
    echo "To attach to monitor: tmux attach -t monitor"
}

# Stop and destroy instance
destroy_instance() {
    if [ -f .vast_instance_id ]; then
        INSTANCE_ID=$(cat .vast_instance_id)
        echo "Destroying instance $INSTANCE_ID..."
        vast destroy instance $INSTANCE_ID
        rm .vast_instance_id
        echo "Instance destroyed"
    else
        echo "No instance ID found"
    fi
}

# Get instance status
instance_status() {
    if [ -f .vast_instance_id ]; then
        INSTANCE_ID=$(cat .vast_instance_id)
        vast show instance $INSTANCE_ID
    else
        echo "No instance ID found"
    fi
}

# Main menu
main_menu() {
    echo ""
    echo "vast.ai Deployment Options:"
    echo "1) Search for instances"
    echo "2) Create and deploy new instance"
    echo "3) Deploy to existing instance"
    echo "4) Start server on instance"
    echo "5) Get instance status"
    echo "6) Destroy instance"
    echo "7) Exit"
    echo ""
    read -p "Select option: " option
    
    case $option in
        1)
            search_instances
            ;;
        2)
            search_instances
            read -p "Enter offer ID to create instance: " OFFER_ID
            create_instance $OFFER_ID
            deploy_to_instance $(cat .vast_instance_id)
            start_server $(cat .vast_instance_id)
            ;;
        3)
            read -p "Enter instance ID: " INSTANCE_ID
            echo $INSTANCE_ID > .vast_instance_id
            deploy_to_instance $INSTANCE_ID
            ;;
        4)
            if [ -f .vast_instance_id ]; then
                start_server $(cat .vast_instance_id)
            else
                echo "No instance ID found"
            fi
            ;;
        5)
            instance_status
            ;;
        6)
            destroy_instance
            ;;
        7)
            exit 0
            ;;
        *)
            echo "Invalid option"
            ;;
    esac
    
    main_menu
}

# Quick deploy function
quick_deploy() {
    echo "Quick deploying Orpheus TTS to vast.ai..."
    
    # Search for best instance
    echo "Finding best RTX 4090 instance..."
    BEST_OFFER=$(vast search offers \
        --gpu-name "RTX 4090" \
        --gpu-ram "${MIN_GPU_RAM}-" \
        --disk-space "${DISK_SPACE}-" \
        --max-price "$MAX_PRICE" \
        --cuda-vers "12.1-" \
        --sort-order "price+" \
        --limit 1 \
        --raw)
    
    if [ -z "$BEST_OFFER" ]; then
        echo "No suitable instances found"
        exit 1
    fi
    
    OFFER_ID=$(echo "$BEST_OFFER" | jq -r '.id')
    PRICE=$(echo "$BEST_OFFER" | jq -r '.dph_total')
    
    echo "Found offer: $OFFER_ID at \$$PRICE/hour"
    read -p "Deploy? (y/n): " confirm
    
    if [ "$confirm" = "y" ]; then
        create_instance $OFFER_ID
        deploy_to_instance $(cat .vast_instance_id)
        start_server $(cat .vast_instance_id)
    fi
}

# Parse arguments
case "${1:-menu}" in
    quick)
        quick_deploy
        ;;
    search)
        search_instances
        ;;
    status)
        instance_status
        ;;
    destroy)
        destroy_instance
        ;;
    menu|*)
        install_vast_cli
        main_menu
        ;;
esac