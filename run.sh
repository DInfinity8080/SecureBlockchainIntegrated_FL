#!/bin/bash
# ══════════════════════════════════════════════════════════════
#   Secure Federated Learning — Full System Launcher
# ══════════════════════════════════════════════════════════════
#   Usage:
#     ./run.sh                                # 10 clients, 5 rounds, no attacks
#     ./run.sh 10 5 nodrop                    # dropout OFF
#     ./run.sh 10 5 nodrop label_flip 0.2     # 20% malicious (label flip)
#     ./run.sh 10 5 nodrop scaling 0.3        # 30% malicious (scaling)
#     ./run.sh 10 5 nodrop noise_injection 0.1
#
#   Attack types: label_flip | scaling | noise_injection
#   ATTACK_FRAC: fraction of clients that are malicious (0.0–0.5)
# ══════════════════════════════════════════════════════════════

NUM_CLIENTS=${1:-10}
NUM_ROUNDS=${2:-5}
DROPOUT_FLAG=""
if [ "$3" == "nodrop" ]; then
    DROPOUT_FLAG="--no-dropout"
fi

# Attack simulation parameters (positional args 4 and 5)
ATTACK_TYPE=${4:-"none"}       # none | label_flip | scaling | noise_injection
ATTACK_FRAC=${5:-"0.0"}        # fraction of clients to be malicious (e.g. 0.2)

# Data partitioning mode (positional arg 6)
PARTITION_MODE=${6:-"iid"}     # iid | non-iid
PARTITION_FLAG=""
if [ "$PARTITION_MODE" == "non-iid" ]; then
    PARTITION_FLAG="--non-iid"
fi

# Compute number of malicious clients (floor)
NUM_MALICIOUS=$(python3 -c "import math; print(math.floor(${NUM_CLIENTS} * ${ATTACK_FRAC}))")
NUM_HONEST=$(( NUM_CLIENTS - NUM_MALICIOUS ))

LOG_DIR="logs"
mkdir -p "$LOG_DIR"

echo ""
echo "============================================================"
echo "  Secure Federated Learning — System Launcher"
echo "============================================================"
echo "  Clients:  $NUM_CLIENTS ($NUM_HONEST honest, $NUM_MALICIOUS malicious)"
echo "  Rounds:   $NUM_ROUNDS"
echo "  Dropout:  $([ -z "$DROPOUT_FLAG" ] && echo 'enabled' || echo 'disabled')"
echo "  Attacks:  $([ "$ATTACK_TYPE" == "none" ] && echo 'none' || echo "$ATTACK_TYPE (${NUM_MALICIOUS} clients)")"
echo "  Data:     $PARTITION_MODE"
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

GANACHE_READY=0
for _i in $(seq 1 15); do
    sleep 1
    if nc -z 127.0.0.1 7545 2>/dev/null || \
       curl -sf http://127.0.0.1:7545 >/dev/null 2>&1; then
        GANACHE_READY=1; break
    fi
    if ! kill -0 $GANACHE_PID 2>/dev/null; then break; fi
done
if [ "$GANACHE_READY" -eq 0 ] || ! kill -0 $GANACHE_PID 2>/dev/null; then
    echo "  ERROR: Ganache failed to start within 15s. Check $LOG_DIR/ganache.log"
    exit 1
fi
echo "  Ganache ready (PID: $GANACHE_PID)"

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

SERVER_READY=0
for _i in $(seq 1 20); do
    sleep 1
    if nc -z 127.0.0.1 9090 2>/dev/null; then
        SERVER_READY=1; break
    fi
    if ! kill -0 $SERVER_PID 2>/dev/null; then break; fi
done
if [ "$SERVER_READY" -eq 0 ] || ! kill -0 $SERVER_PID 2>/dev/null; then
    echo "  ERROR: Server failed to start within 20s. Check $LOG_DIR/server.log"
    kill $GANACHE_PID
    exit 1
fi
echo "  Server ready (PID: $SERVER_PID)"

# ── Step 5: Launch clients ───────────────────────────────────
echo ""
echo "[5/5] Launching $NUM_CLIENTS clients ($NUM_HONEST honest, $NUM_MALICIOUS malicious)..."

# Honest clients occupy IDs 0 .. NUM_HONEST-1
for ((i=0; i<NUM_HONEST; i++)); do
    python client.py $i $NUM_CLIENTS $DROPOUT_FLAG $PARTITION_FLAG > "$LOG_DIR/client_${i}.log" 2>&1 &
    echo "  Started honest  client $i"
    sleep 0.3
done

# Malicious clients occupy IDs NUM_HONEST .. NUM_CLIENTS-1
if [ "$NUM_MALICIOUS" -gt 0 ] && [ "$ATTACK_TYPE" != "none" ]; then
    for ((i=NUM_HONEST; i<NUM_CLIENTS; i++)); do
        python attack_simulator.py $i $ATTACK_TYPE $NUM_CLIENTS > "$LOG_DIR/client_${i}.log" 2>&1 &
        echo "  Started ATTACK  client $i ($ATTACK_TYPE)"
        sleep 0.3
    done
fi

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