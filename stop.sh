#!/bin/bash
echo "Stopping all FL processes..."

# Kill Python FL processes (works on both macOS and Linux)
pkill -9 -f 'client\.py' 2>/dev/null
pkill -9 -f 'server\.py' 2>/dev/null
pkill -9 -f 'ganache' 2>/dev/null

# Kill anything still holding ports 7545 and 9090
for port in 7545 9090; do
    # Try lsof first (macOS + most Linux), fall back to fuser (some Linux)
    pid=$(lsof -ti :$port 2>/dev/null || fuser $port/tcp 2>/dev/null)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null
        echo "  Killed process on port $port (PID: $pid)"
    fi
done

# Final check — force-kill any stragglers
sleep 1
remaining=$(pgrep -f 'client\.py|server\.py' 2>/dev/null)
if [ -n "$remaining" ]; then
    echo "  Force-killing remaining processes: $remaining"
    echo "$remaining" | xargs kill -9 2>/dev/null
fi

echo "All processes stopped."
