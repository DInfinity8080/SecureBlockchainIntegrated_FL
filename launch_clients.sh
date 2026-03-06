#!/bin/bash
NUM_CLIENTS=${1:-10}
LOG_DIR="logs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo "  Launching $NUM_CLIENTS FL clients"
echo "  Logs will be saved to $LOG_DIR/"
echo "============================================================"

for ((i=0; i<NUM_CLIENTS; i++)); do
    echo "Starting client $i..."
    python client.py $i $NUM_CLIENTS > "$LOG_DIR/client_${i}.log" 2>&1 &
    sleep 1
done

echo ""
echo "All $NUM_CLIENTS clients launched!"
echo "Monitor with: tail -f $LOG_DIR/client_*.log"
echo "Stop all with: pkill -f 'python client.py'"
echo ""

wait
echo "All clients finished."
