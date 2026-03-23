#!/bin/bash
# ══════════════════════════════════════════════════════════════
#   Secure Federated Learning — Full System Launcher
# ══════════════════════════════════════════════════════════════
#   Usage:
#     ./run.sh                    # 10 clients, 5 rounds, dropout ON
#     ./run.sh 10 5 nodrop        # 10 clients, 5 rounds, dropout OFF
#     ./run.sh 20 5               # 20 clients, 5 rounds, dropout ON
#     ./run.sh 50 3 nodrop        # 50 clients, 3 rounds, dropout OFF
# ══════════════════════════════════════════════════════════════

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

# ── Step 1: Kill old processes ───────────────────────────────
echo "[1/5] Cleaning up old processes..."
pkill -9 -f 'client\.py' 2>/dev/null
pkill -9 -f 'server\.py' 2>/dev/null
pkill -9 -f 'ganache' 2>/dev/null
# Kill anything still holding ports 7545 and 9090
for port in 7545 9090; do
    pid=$(lsof -ti :$port 2>/dev/null || fuser $port/tcp 2>/dev/null)
    if [ -n "$pid" ]; then
        kill -9 $pid 2>/dev/null
    fi
done
sleep 2
echo "  Done."

# ── Step 2: Start Ganache ────────────────────────────────────
GANACHE_ACCOUNTS=$((NUM_CLIENTS + 5))
echo ""
echo "[2/5] Starting Ganache (port 7545, $GANACHE_ACCOUNTS accounts)..."
ganache --port 7545 --accounts $GANACHE_ACCOUNTS > "$LOG_DIR/ganache.log" 2>&1 &
GANACHE_PID=$!
sleep 3

if ! kill -0 $GANACHE_PID 2>/dev/null; then
    echo "  ERROR: Ganache failed to start. Check $LOG_DIR/ganache.log"
    exit 1
fi
echo "  Ganache running (PID: $GANACHE_PID)"

# ── Step 3: Deploy smart contracts ───────────────────────────
echo ""
echo "[3/5] Deploying smart contracts..."
truffle migrate --reset --network development > "$LOG_DIR/truffle.log" 2>&1
if [ $? -ne 0 ]; then
    echo "  ERROR: Truffle migration failed. Check $LOG_DIR/truffle.log"
    kill $GANACHE_PID
    exit 1
fi
echo "  Contracts deployed."

# ── Step 4: Start FL server ──────────────────────────────────
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

# ── Step 5: Launch clients ───────────────────────────────────
echo ""
echo "[5/5] Launching $NUM_CLIENTS clients..."
for ((i=0; i<NUM_CLIENTS; i++)); do
    python client.py $i $NUM_CLIENTS $DROPOUT_FLAG > "$LOG_DIR/client_${i}.log" 2>&1 &
    echo "  Started client $i"
    sleep 0.5
done

echo ""
echo "============================================================"
echo "  All components launched! Streaming server output..."
echo "  Press Ctrl+C to stop watching (processes continue)"
echo "============================================================"
echo ""

# Stream server log live until server finishes
tail -f "$LOG_DIR/server.log" --pid=$SERVER_PID 2>/dev/null

# Server finished — cleanup
echo ""
echo "============================================================"
echo "  Training complete. Cleaning up..."
echo "============================================================"

kill $GANACHE_PID 2>/dev/null
pkill -f 'python client.py' 2>/dev/null

echo "  All processes stopped."
echo "  Results saved to: results/"
echo ""