#!/bin/bash
# run_burst_bench.sh — Benchmark "burst": populate live CSV with a large dataset,
# start the IPS, wait for it to finish processing EVERYTHING, measure.
#
# Differences from run_bench.sh streaming:
#   - Does not use feed_csv.py (CSV is copied in full at the start)
#   - Waits for the IPS to process everything, does NOT run a fixed timer
#   - Results in "real maximum capacity", directly comparable with validate_hailo.py
#
# Usage:
#   bash run_burst_bench.sh
#
# Optional variables:
#   ROOT_DIR       - default: /home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test
#   BURST_CSV      - default: $ROOT_DIR/friday_ddos.csv
#   BATCH          - default: 128
#   POLL           - default: 1 (smaller to detect completion quickly)
#   MAX_WAIT_SEC   - total timeout per variant (default: 600 = 10 min)

set -e

# ---------- CONFIG ----------
ROOT_DIR="${ROOT_DIR:-/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test}"
BURST_CSV="${BURST_CSV:-$ROOT_DIR/friday_ddos.csv}"
LIVE_CSV="$ROOT_DIR/live_sample_bench.csv"

RUN_TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$ROOT_DIR/bench_results/burst_$RUN_TS"

BATCH="${BATCH:-128}"
POLL="${POLL:-1}"
THRESHOLD="${THRESHOLD:-0.5}"   # On friday_ddos.csv, 0.5 gives F1=99%, so it's ok
MAX_WAIT_SEC="${MAX_WAIT_SEC:-600}"

VENV_CPU="${VENV_CPU:-$ROOT_DIR/venv_ids/bin/activate}"
VENV_HAILO="${VENV_HAILO:-$ROOT_DIR/venv_hailo_runtime/bin/activate}"
# ----------------------------

if [[ ! -f "$BURST_CSV" ]]; then
    echo "[ERR] Burst CSV missing: $BURST_CSV"
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "[BURST] Output: $OUT_DIR"
echo "[BURST] Source CSV: $BURST_CSV"
echo "[BURST] Source rows: $(($(wc -l < "$BURST_CSV") - 1))"
echo "[BURST] Batch: $BATCH | Threshold: $THRESHOLD | Poll: ${POLL}s"

# Clean up old iptables rules
if sudo iptables -L INPUT -n 2>/dev/null | grep -q "192.168.10\|172.16"; then
    echo "[BURST] Cleaning up old iptables rules..."
    sudo iptables -L INPUT -n --line-numbers | grep -E "192.168.10|172.16" \
        | awk '{print $1}' | sort -rn | while read num; do
        sudo iptables -D INPUT "$num" 2>/dev/null || true
    done
fi

# Wait for the IPS to finish processing (last_seen == n stable)
wait_for_completion() {
    local LOG_FILE="$1"
    local TIMING_CSV="$2"
    local IPS_PID="$3"

    local LAST_BATCH_COUNT=0
    local STABLE_TICKS=0
    local STABLE_THRESHOLD=4   # 4 ticks of 2 sec = 8 sec without new batch = "done"
    local elapsed=0
    local check_interval=2

    while [[ $elapsed -lt $MAX_WAIT_SEC ]]; do
        sleep "$check_interval"
        elapsed=$((elapsed + check_interval))

        # Is the IPS still alive?
        if ! kill -0 "$IPS_PID" 2>/dev/null; then
            echo "[BURST] IPS stopped (probably an error). Check $LOG_FILE"
            return 1
        fi

        # How many batches does the timing CSV have so far?
        local CURRENT_BATCH_COUNT=0
        if [[ -f "$TIMING_CSV" ]]; then
            CURRENT_BATCH_COUNT=$(($(wc -l < "$TIMING_CSV") - 1))
        fi

        if [[ $CURRENT_BATCH_COUNT -gt $LAST_BATCH_COUNT ]]; then
            echo "[BURST]   batches processed: $CURRENT_BATCH_COUNT (elapsed: ${elapsed}s)"
            LAST_BATCH_COUNT=$CURRENT_BATCH_COUNT
            STABLE_TICKS=0
        else
            STABLE_TICKS=$((STABLE_TICKS + 1))
            if [[ $STABLE_TICKS -ge $STABLE_THRESHOLD ]]; then
                echo "[BURST] Processing appears complete (no-progress timeout)."
                return 0
            fi
        fi
    done

    echo "[BURST] Maximum timeout reached (${MAX_WAIT_SEC}s)."
    return 1
}

# Function: run a variant in burst mode
run_burst () {
    local VARIANT="$1"
    local IPS_SCRIPT="$2"
    local VENV_PATH="$3"
    local EXTRA_ARGS="$4"

    echo ""
    echo "============================================================"
    echo "  BURST: $VARIANT  ($IPS_SCRIPT)"
    echo "============================================================"

    local TIMING_CSV="$OUT_DIR/timing_$VARIANT.csv"
    local RESOURCES_CSV="$OUT_DIR/resources_$VARIANT.csv"
    local LOG_FILE="$OUT_DIR/run_$VARIANT.log"

    # Activate venv
    if [[ -f "$VENV_PATH" ]]; then
        # shellcheck disable=SC1090
        source "$VENV_PATH"
    else
        echo "[ERR] Venv missing: $VENV_PATH"
        return 1
    fi

    # Step 1: start the IPS with an empty live_sample (header only), so it can
    # initialize its ONNX/Hailo session before receiving data
    head -n 1 "$BURST_CSV" > "$LIVE_CSV"
    echo "[BURST] Initial live CSV: header only"

    echo "[BURST] Start IPS ($VARIANT)..."
    BENCH=1 BENCH_OUT="$TIMING_CSV" BENCH_VARIANT="$VARIANT" \
    python3 "$ROOT_DIR/$IPS_SCRIPT" \
        --input "$LIVE_CSV" \
        --threshold "$THRESHOLD" \
        --batch "$BATCH" \
        --poll "$POLL" \
        --dry-run \
        $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
    IPS_PID=$!
    echo "[BURST] IPS PID = $IPS_PID"

    # Step 2: warm-up (ONNX session + Hailo VDevice load)
    echo "[BURST] Warm-up 6s for initialization..."
    sleep 6

    if ! kill -0 "$IPS_PID" 2>/dev/null; then
        echo "[ERR] IPS crashed at startup. See: $LOG_FILE"
        tail -25 "$LOG_FILE"
        return 1
    fi

    # Step 3: start resource sampler
    echo "[BURST] Start resource sampler..."
    python3 "$ROOT_DIR/resource_sampler.py" \
        --pid "$IPS_PID" \
        --variant "$VARIANT" \
        --out "$RESOURCES_CSV" \
        --interval 1 \
        --duration "$MAX_WAIT_SEC" > "$OUT_DIR/sampler_$VARIANT.log" 2>&1 &
    SAMPLER_PID=$!

    # Step 4: BURST! Copy the entire CSV into live_sample.csv in a single atomic operation
    echo "[BURST] Inject FULL CSV ($(wc -l < "$BURST_CSV") lines) into live_sample..."
    local T0
    T0=$(date +%s.%N)
    cp "$BURST_CSV" "$LIVE_CSV.tmp"
    mv "$LIVE_CSV.tmp" "$LIVE_CSV"   # atomic rename
    echo "[BURST] Injected. T0 = $T0"

    # Step 5: wait for completion
    echo "[BURST] Waiting for full processing..."
    wait_for_completion "$LOG_FILE" "$TIMING_CSV" "$IPS_PID"
    local WAIT_RC=$?
    local T1
    T1=$(date +%s.%N)
    local ELAPSED
    ELAPSED=$(echo "$T1 - $T0" | bc)
    echo "[BURST] Total burst time: ${ELAPSED}s"
    echo "$ELAPSED" > "$OUT_DIR/wall_time_$VARIANT.txt"

    # Step 6: orderly shutdown
    echo "[BURST] Stop sampler & IPS..."
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    sleep 1
    kill -TERM "$IPS_PID" 2>/dev/null || true
    sleep 3
    kill -KILL "$IPS_PID" "$SAMPLER_PID" 2>/dev/null || true

    # Small report
    echo ""
    echo "[BURST] $VARIANT done."
    if [[ -f "$TIMING_CSV" ]]; then
        local NB
        NB=$(($(wc -l < "$TIMING_CSV") - 1))
        echo "      timing:    $TIMING_CSV ($NB batches)"
    else
        echo "      [WARN] timing CSV was NOT written"
    fi
    if [[ -f "$RESOURCES_CSV" ]]; then
        local NS
        NS=$(($(wc -l < "$RESOURCES_CSV") - 1))
        echo "      resources: $RESOURCES_CSV ($NS samples)"
    fi
    echo "      wall time: ${ELAPSED}s"

    deactivate 2>/dev/null || true
}

# ---------- Run variants ----------

run_burst "cpu" "ips_realtime_v2.py" "$VENV_CPU" \
    "--model $ROOT_DIR/ids_mlp_binary.onnx \
     --scaler $ROOT_DIR/scaler.joblib \
     --features $ROOT_DIR/feature_names.npy \
     --alert-log $OUT_DIR/alerts_cpu.log \
     --action-log $OUT_DIR/actions_cpu.log"

run_burst "hailo" "ips_hailo.py" "$VENV_HAILO" \
    "--hef $ROOT_DIR/ids_mlp_binary_logits.hef \
     --scaler $ROOT_DIR/scaler_params.npz \
     --features $ROOT_DIR/feature_names.txt \
     --alert-log $OUT_DIR/alerts_hailo.log \
     --action-log $OUT_DIR/actions_hailo.log"

# ---------- Generate report ----------
echo ""
echo "============================================================"
echo "  GENERATE BURST REPORT"
echo "============================================================"

if [[ -f "$VENV_CPU" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_CPU"
fi
python3 "$ROOT_DIR/gen_report.py" --bench-dir "$OUT_DIR" --out "$OUT_DIR/report.md"

# Also add wall-time end-to-end (informational, comparable with validate_hailo.py)
echo ""
echo "============================================================"
echo "  WALL TIME END-TO-END"
echo "============================================================"
for v in cpu hailo; do
    if [[ -f "$OUT_DIR/wall_time_$v.txt" ]]; then
        WT=$(cat "$OUT_DIR/wall_time_$v.txt")
        SRC_ROWS=$(($(wc -l < "$BURST_CSV") - 1))
        # echo "$v: $WT s for $SRC_ROWS rows = ?"
        printf "  %-6s : %.2fs (%d rows)\n" "$v" "$WT" "$SRC_ROWS"
    fi
done

echo ""
echo "[BURST] DONE."
echo "[BURST] Report: $OUT_DIR/report.md"
echo "[BURST] CSVs:   $OUT_DIR/"
