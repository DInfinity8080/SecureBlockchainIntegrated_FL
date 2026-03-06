#!/bin/bash
echo "Stopping all FL processes..."
pkill -f 'python client.py' 2>/dev/null
pkill -f 'python server.py' 2>/dev/null
pkill -f 'ganache' 2>/dev/null

# Kill anything still holding ports 7545 and 9090 (macOS-compatible)
for port in 7545 9090; do
    pid=$(lsof -ti :$port 2>/dev/null)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null
        echo "  Killed process on port $port (PID: $pid)"
    fi
done

echo "All processes stopped."
