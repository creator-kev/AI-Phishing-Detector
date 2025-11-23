#!/bin/bash

echo "╔════════════════════════════════════════════╗"
echo "║   AI Phishing Detector - Direct Deploy    ║"
echo "╚════════════════════════════════════════════╝"
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m'

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo -e "${YELLOW}⚠️  Virtual environment not found. Creating...${NC}"
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate

# Install/update dependencies
echo "📦 Installing dependencies..."
pip install -q --upgrade pip
pip install -q -r requirements.txt
pip install -q gunicorn

# Check if model exists
if [ ! -f "models/best_model.pkl" ]; then
    echo -e "${YELLOW}⚠️  Model not found. Training model...${NC}"
    python run_pipeline.py
fi

echo -e "${GREEN}✅ Model found${NC}"

# Create logs directory
mkdir -p logs logs/clients

# Kill any existing process on port 5000
echo "🔍 Checking for existing processes..."
PID=$(lsof -ti:5000)
if [ ! -z "$PID" ]; then
    echo "🛑 Stopping existing process..."
    kill -9 $PID
fi

# Start application with Gunicorn
echo ""
echo "🚀 Starting application..."
nohup gunicorn app.main:app \
    --workers 4 \
    --bind 0.0.0.0:5000 \
    --timeout 120 \
    --access-logfile logs/access.log \
    --error-logfile logs/error.log \
    --log-level info \
    --daemon

# Wait for service to start
echo "⏳ Waiting for service to start..."
sleep 3

# Check if running
if lsof -Pi :5000 -sTCP:LISTEN -t >/dev/null ; then
    echo -e "${GREEN}✅ Service started successfully${NC}"
    
    # Test health endpoint
    HEALTH=$(curl -s http://localhost:5000/health | grep -o "healthy")
    if [ "$HEALTH" == "healthy" ]; then
        echo -e "${GREEN}✅ Health check passed${NC}"
    fi
    
    echo ""
    echo "╔════════════════════════════════════════════╗"
    echo "║          Deployment Successful!            ║"
    echo "╚════════════════════════════════════════════╝"
    echo ""
    echo "🌐 Application: http://localhost:5000"
    echo "📖 API Docs:    http://localhost:5000/apidocs"
    echo "💚 Health:      http://localhost:5000/health"
    echo ""
    echo "Commands:"
    echo "  View logs:    tail -f logs/access.log"
    echo "  Stop:         kill \$(lsof -ti:5000)"
    echo "  Restart:      ./deploy_direct.sh"
    echo ""
    
    # Save PID
    PID=$(lsof -ti:5000)
    echo $PID > .app.pid
    echo "Process ID: $PID (saved to .app.pid)"
else
    echo -e "${RED}❌ Failed to start service${NC}"
    echo "Check logs: tail -f logs/error.log"
    exit 1
fi
