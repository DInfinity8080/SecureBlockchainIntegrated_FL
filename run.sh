#!/bin/bash
NUM_CLIENTS=${1:-10}
NUM_ROUNDS=${2:-5}
DROPOUT_FLAG=""
if [ "$3" == "nodrop" ]; then
    DROPOUT_FLAG="--no-dropout"
fi
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo ""
echo "============================================================"
echo "  Secure Federated Learning — System Launcher"
echo "============================================================"
echo "  Clients:  $NUM_CLIENTS"
echo "  Rounds:   $NUM_ROUNDS"
echo "  Dropout:  $([ -z "$DROPOUT_FLAG" ] && echo 'enabled' || echo 'disabled')"
echo "============================================================"
echo ""

echo "[1/5] Cleaning up old processes..."
fuser -k 7545/tcp 2>/dev/null
fuser -k 9090/tcp 2>/dev/null
pkill -f 'python client.py' 2>/dev/null
pkill -f 'python server.py' 2>/dev/null
pkill -f 'ganache' 2>/dev/null
sleep 2
echo "  Done."

echo ""
echo "[2/5] Starting Ganache (port 7545, $((NUM_CLIENTS + 5)) accounts)..."
ganache --port 7545 --accounts $((NUM_CLIENTS + 5)) > "$LOG_DIR/ganache.log" 2>&1 &
GANACHE_PID=$!
sleep 3
if ! kill -0 $GANACHE_PID 2>/dev/null; then
    echo "  ERROR: Ganache failed to start. Check $LOG_DIR/ganache.log"
    exit 1
fi
echo "  Ganache running (PID: $GANACHE_PID)"

echo ""
echo "[3/5] Deploying smart contracts..."
truffle migrate --reset --network development > "$LOG_DIR/truffle.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  ERROR: Truffle migration failed. Check $LOG_DIR/truffle.log"
    kill $GANACHE_PID
    exit 1
fi
echo "  Contracts deployed."

echo ""
echo "[4/5] Starting FL server ($NUM_CLIENTS clients, $NUM_ROUNDS rounds)..."
python server.py $NUM_CLIENTS $NUM_ROUNDS > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
sleep 5
if ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "  ERROR: Server failed to start. Check $LOG_DIR/server.log"
    kill $GANACHE_PID
    exit 1
fi
echo "  Server running (PID: $SERVER_PID)"

echo ""
echo "[5/5] Launching $NUM_CLIENTS clients..."
for ((i=0; i<NUM_CLIENTS; i++)); do
    python client.py $i $NUM_CLIENTS $DROPOUT_FLAG > "$LOG_DIR/client_${i}.log" 2>&1 &
    echo "  Started client $i"
    sleep 0.5
done

echo ""
echo "============================================================"
echo "  All components launched!"
echo "============================================================"
echo ""
echo "  Monitor server:   tail -f $LOG_DIR/server.log"
echo "  Monitor clients:  tail -f $LOG_DIR/client_*.log"
echo "  Stop everything:  ./stop.sh"
echo "============================================================"
echo ""

wait $SERVER_PID
echo ""
echo "============================================================"
echo "  Server finished. Displaying results..."
echo "============================================================"
echo ""
cat "$LOG_DIR/server.log"

kill $GANACHE_PID 2>/dev/null
pkill -f 'python client.py' 2>/dev/null
echo ""
echo "All processes stopped."
