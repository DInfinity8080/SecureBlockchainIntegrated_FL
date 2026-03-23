#!/bin/bash
# ══════════════════════════════════════════════════════════════
#   Batch Experiment Runner — Secure Federated Learning
# ══════════════════════════════════════════════════════════════
#   Runs FL experiments from 10→20 clients for each round
#   configuration: 5, 7, and 10 rounds.
#
#   Total runs: 11 clients × 3 round configs = 33 experiments
#
#   Usage:
#     ./run_experiments.sh          # Run all 33 experiments (dropout ON)
#     ./run_experiments.sh nodrop   # Run all 33 experiments (dropout OFF)
# ══════════════════════════════════════════════════════════════

ROUND_CONFIGS=(5 7 10)
MIN_CLIENTS=10
MAX_CLIENTS=20

# ── Parse nodrop flag ────────────────────────────────────────
DROPOUT_FLAG=""
DROPOUT_LABEL="enabled"
BATCH_SUFFIX=""
if [ "$1" == "nodrop" ]; then
    DROPOUT_FLAG="--no-dropout"
    DROPOUT_LABEL="DISABLED"
    BATCH_SUFFIX="_nodrop"
fi

EXPERIMENT_DIR="experiment_results"
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
BATCH_DIR="${EXPERIMENT_DIR}/batch_${TIMESTAMP}${BATCH_SUFFIX}"

mkdir -p "$BATCH_DIR"

TOTAL_RUNS=$(( (MAX_CLIENTS - MIN_CLIENTS + 1) * ${#ROUND_CONFIGS[@]} ))
CURRENT_RUN=0

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║           BATCH EXPERIMENT RUNNER                       ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Clients range:  ${MIN_CLIENTS} → ${MAX_CLIENTS} (incrementing by 1)       ║"
echo "║  Round configs:  5, 7, 10                               ║"
echo "║  Dropout:        ${DROPOUT_LABEL}                                ║"
echo "║  Total runs:     ${TOTAL_RUNS}                                     ║"
echo "║  Output dir:     ${BATCH_DIR}              ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""

# Track overall timing
BATCH_START=$(date +%s)

COMPLETED=0
FAILED=0

for NUM_ROUNDS in "${ROUND_CONFIGS[@]}"; do
    echo ""
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║  PHASE: ${NUM_ROUNDS} ROUNDS — Clients ${MIN_CLIENTS}→${MAX_CLIENTS}                       ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""

    for ((NUM_CLIENTS=MIN_CLIENTS; NUM_CLIENTS<=MAX_CLIENTS; NUM_CLIENTS++)); do
        CURRENT_RUN=$((CURRENT_RUN + 1))
        RUN_LABEL="run_${NUM_CLIENTS}clients_${NUM_ROUNDS}rounds${BATCH_SUFFIX}"
        RUN_DIR="${BATCH_DIR}/${RUN_LABEL}"
        mkdir -p "$RUN_DIR"

        echo ""
        echo "============================================================"
        echo "  EXPERIMENT ${CURRENT_RUN}/${TOTAL_RUNS}: ${NUM_CLIENTS} clients, ${NUM_ROUNDS} rounds"
        echo "============================================================"
        echo ""

        RUN_START=$(date +%s)

        # ── Step 1: Kill old processes ───────────────────────────────
        echo "[1/6] Cleaning up old processes..."
        pkill -9 -f 'python client.py' 2>/dev/null
        pkill -9 -f 'python server.py' 2>/dev/null
        pkill -9 -f 'ganache' 2>/dev/null
        for port in 7545 9090; do
            pid=$(lsof -ti :$port 2>/dev/null)
            if [ -n "$pid" ]; then
                kill -9 $pid 2>/dev/null
            fi
        done
        sleep 3
        echo "  Done."

        # ── Step 2: Start Ganache ────────────────────────────────────
        GANACHE_ACCOUNTS=$((NUM_CLIENTS + 5))
        echo ""
        echo "[2/6] Starting Ganache (port 7545, $GANACHE_ACCOUNTS accounts)..."
        ganache --port 7545 --accounts $GANACHE_ACCOUNTS > "$RUN_DIR/ganache.log" 2>&1 &
        GANACHE_PID=$!
        sleep 3

        if ! kill -0 $GANACHE_PID 2>/dev/null; then
            echo "  ERROR: Ganache failed to start. Check $RUN_DIR/ganache.log"
            FAILED=$((FAILED + 1))
            continue
        fi
        echo "  Ganache running (PID: $GANACHE_PID)"

        # ── Step 3: Deploy smart contracts ───────────────────────────
        echo ""
        echo "[3/6] Deploying smart contracts..."
        truffle migrate --reset --network development > "$RUN_DIR/truffle.log" 2>&1
        if [ $? -ne 0 ]; then
            echo "  ERROR: Truffle migration failed. Check $RUN_DIR/truffle.log"
            kill $GANACHE_PID 2>/dev/null
            FAILED=$((FAILED + 1))
            continue
        fi
        echo "  Contracts deployed."

        # ── Step 4: Start FL server ──────────────────────────────────
        echo ""
        echo "[4/6] Starting FL server ($NUM_CLIENTS clients, $NUM_ROUNDS rounds)..."
        python server.py $NUM_CLIENTS $NUM_ROUNDS > "$RUN_DIR/server.log" 2>&1 &
        SERVER_PID=$!
        sleep 5

        if ! kill -0 $SERVER_PID 2>/dev/null; then
            echo "  ERROR: Server failed to start. Check $RUN_DIR/server.log"
            kill $GANACHE_PID 2>/dev/null
            FAILED=$((FAILED + 1))
            continue
        fi
        echo "  Server running (PID: $SERVER_PID)"

        # ── Step 5: Launch clients ───────────────────────────────────
        echo ""
        echo "[5/6] Launching $NUM_CLIENTS clients..."
        for ((i=0; i<NUM_CLIENTS; i++)); do
            python client.py $i $NUM_CLIENTS $DROPOUT_FLAG > "$RUN_DIR/client_${i}.log" 2>&1 &
            sleep 0.5
        done
        echo "  All $NUM_CLIENTS clients launched."

        # ── Wait for server to finish (with timeout) ─────────────────
        # Max 5 minutes per experiment to avoid infinite hangs
        MAX_WAIT=300
        echo ""
        echo "  Waiting for training to complete (timeout: ${MAX_WAIT}s)..."
        WAIT_START=$(date +%s)
        while kill -0 $SERVER_PID 2>/dev/null; do
            ELAPSED=$(( $(date +%s) - WAIT_START ))
            if [ $ELAPSED -ge $MAX_WAIT ]; then
                echo "  TIMEOUT: Experiment exceeded ${MAX_WAIT}s — killing..."
                kill -9 $SERVER_PID 2>/dev/null
                pkill -9 -f 'python client.py' 2>/dev/null
                FAILED=$((FAILED + 1))
                break
            fi
            sleep 2
        done

        # ── Step 6: Collect results ──────────────────────────────────
        echo ""
        echo "[6/6] Collecting results..."

        # Copy the results directory into the run-specific folder
        if [ -d "results" ]; then
            cp -r results/* "$RUN_DIR/" 2>/dev/null
            echo "  Results saved to: $RUN_DIR/"
        else
            echo "  WARNING: No results directory found!"
        fi

        # Save the session report (tail of server log)
        tail -80 "$RUN_DIR/server.log" > "$RUN_DIR/session_report.txt" 2>/dev/null

        # Cleanup
        kill -9 $GANACHE_PID 2>/dev/null
        pkill -9 -f 'python client.py' 2>/dev/null
        pkill -9 -f 'python server.py' 2>/dev/null
        sleep 3

        RUN_END=$(date +%s)
        RUN_TIME=$((RUN_END - RUN_START))

        COMPLETED=$((COMPLETED + 1))
        echo ""
        echo "  ✅ ${CURRENT_RUN}/${TOTAL_RUNS} complete: ${NUM_CLIENTS} clients × ${NUM_ROUNDS} rounds (${RUN_TIME}s)"
        echo ""
    done
done

# ── Final cleanup ────────────────────────────────────────────
pkill -f 'ganache' 2>/dev/null
pkill -f 'python client.py' 2>/dev/null
pkill -f 'python server.py' 2>/dev/null

BATCH_END=$(date +%s)
BATCH_TIME=$((BATCH_END - BATCH_START))
BATCH_MINS=$((BATCH_TIME / 60))
BATCH_SECS=$((BATCH_TIME % 60))

echo ""
echo "╔══════════════════════════════════════════════════════════╗"
echo "║              ALL EXPERIMENTS COMPLETE                   ║"
echo "╠══════════════════════════════════════════════════════════╣"
echo "║  Completed: ${COMPLETED}/${TOTAL_RUNS} runs                                ║"
echo "║  Failed:    ${FAILED} runs                                       ║"
echo "║  Total time: ${BATCH_MINS}m ${BATCH_SECS}s                                    ║"
echo "╚══════════════════════════════════════════════════════════╝"
echo ""
echo "  Results directory: ${BATCH_DIR}/"
echo ""
echo "  Structure:"
for NUM_ROUNDS in "${ROUND_CONFIGS[@]}"; do
    echo "    ── ${NUM_ROUNDS} Rounds ──"
    for ((c=MIN_CLIENTS; c<=MAX_CLIENTS; c++)); do
        echo "       run_${c}clients_${NUM_ROUNDS}rounds/"
    done
done
echo ""
echo "  Each folder contains:"
echo "    session_config.json, round_summary.csv,"
echo "    client_round_details.csv, poisoning_detection.csv,"
echo "    reputation_history.csv, communication_metrics.csv,"
echo "    participation_metrics.csv, final_reputations.csv,"
echo "    client_tiers.csv, server.log, session_report.txt"
echo ""
