#!/bin/bash
echo "Stopping all FL processes..."
pkill -f 'python client.py' 2>/dev/null
pkill -f 'python server.py' 2>/dev/null
pkill -f 'ganache' 2>/dev/null
fuser -k 7545/tcp 2>/dev/null
fuser -k 9090/tcp 2>/dev/null
echo "All processes stopped."
