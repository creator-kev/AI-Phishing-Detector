#!/bin/bash

echo "🛑 Stopping AI Phishing Detector..."

# Check if PID file exists
if [ -f ".app.pid" ]; then
    PID=$(cat .app.pid)
    if ps -p $PID > /dev/null; then
        kill $PID
        echo "✅ Process $PID stopped"
    else
        echo "⚠️  Process not running"
    fi
    rm .app.pid
else
    # Kill by port
    PID=$(lsof -ti:5000)
    if [ ! -z "$PID" ]; then
        kill $PID
        echo "✅ Process on port 5000 stopped"
    else
        echo "⚠️  No process running on port 5000"
    fi
fi
