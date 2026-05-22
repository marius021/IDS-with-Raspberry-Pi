#!/usr/bin/env bash
# run_scaling_v2.sh
# Phase E — Scaling benchmark CPU (ONNX) vs Hailo (HEF) with batch sizes 32/128/512.
# Fixes from v1:
#   - All artifact paths are resolved to absolute paths and passed explicitly
#   - IPS process is sanity-checked 5s after start; if dead, the run aborts cleanly
#   - The ips.log is tailed at the end so you immediately see why a run failed
#   - BENCH=1 is set as a safety net in addition to BENCH_OUT
#
# Usage:
#   bash run_scaling_v2.sh [csv_input] [duration_sec]

set -u

CSV_INPUT="${1:-$HOME/Desktop/IDS-with-Raspberry-Pi/ids_test/friday_ddos.csv}"
DURATION="${2:-90}"
BATCH_SIZES=(32 128 512)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$HOME/bench_results/${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/run.log"

# --- Resolve absolute paths to all artifacts ---
ABS_CSV="$(realpath "$CSV_INPUT")"
ABS_ONNX="$(realpath "$SCRIPT_DIR/ids_mlp_binary.onnx" 2>/dev/null || echo '')"
ABS_HEF="$(realpath  "$SCRIPT_DIR/ids_mlp_binary_logits.hef" 2>/dev/null || echo '')"
ABS_SCALER_JOBLIB="$(realpath "$SCRIPT_DIR/scaler.joblib"     2>/dev/null || echo '')"
ABS_SCALER_NPZ="$(realpath    "$SCRIPT_DIR/scaler_params.npz" 2>/dev/null || echo '')"
# Try multiple feature-name conventions
for fname in feature_names.npy feature_names.txt feature_names.json; do
    if [[ -f "$SCRIPT_DIR/$fname" ]]; then
        ABS_FEATS="$(realpath "$SCRIPT_DIR/$fname")"
        break
    fi
done

log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

cleanup() {
    log "Cleanup"
    pkill -f "ips_realtime_v2.py" 2>/dev/null || true
    pkill -f "ips_hailo.py" 2>/dev/null || true
    pkill -f "monitor_resources" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT INT TERM

log "================================================================"
log "Phase E — Scaling Benchmark v2"
log "================================================================"
log "Results dir : $RESULTS_DIR"
log "CSV input   : $ABS_CSV"
log "Duration    : ${DURATION}s per run"
log "Batch sizes : ${BATCH_SIZES[*]}"
log ""
log "Artifacts (absolute paths):"
log "  ONNX            : ${ABS_ONNX:-NOT FOUND}"
log "  HEF             : ${ABS_HEF:-NOT FOUND}"
log "  scaler.joblib   : ${ABS_SCALER_JOBLIB:-NOT FOUND}"
log "  scaler_params   : ${ABS_SCALER_NPZ:-NOT FOUND}"
log "  feature_names   : ${ABS_FEATS:-NOT FOUND}"
log "================================================================"

if [[ ! -f "$ABS_CSV" ]]; then
    log "[ERROR] CSV input not found"; exit 1
fi
if [[ -z "$ABS_FEATS" ]]; then
    log "[ERROR] no feature_names.* file found in $SCRIPT_DIR"; exit 1
fi

# Reuse monitor from v1 (or create if missing)
RESOURCE_MONITOR="$RESULTS_DIR/monitor_resources.py"
cat > "$RESOURCE_MONITOR" << 'PYEOF'
#!/usr/bin/env python3
import sys, time, csv
import psutil

out_path = sys.argv[1]
target_name = sys.argv[2]

def get_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        return None

def find_proc(name):
    for p in psutil.process_iter(['name', 'cmdline']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
            if name in cmd and 'monitor_resources' not in cmd:
                return p
        except Exception:
            continue
    return None

n_cores = psutil.cpu_count(logical=True)
header = ["ts_iso", "ts_epoch", "cpu_total_pct"] + \
         [f"core{i}_pct" for i in range(n_cores)] + \
         ["cpu_proc_pct", "rss_mb", "temp_c", "proc_pid"]
with open(out_path, "w", newline="") as f:
    csv.writer(f).writerow(header)

psutil.cpu_percent(percpu=True)
target = None
while True:
    try:
        time.sleep(1)
        per_core = psutil.cpu_percent(percpu=True)
        total = sum(per_core) / len(per_core)
        if target is None or not target.is_running():
            target = find_proc(target_name)
        proc_pct, rss, pid = 0.0, 0.0, 0
        if target and target.is_running():
            try:
                proc_pct = target.cpu_percent(interval=None)
                rss = target.memory_info().rss / 1e6
                pid = target.pid
            except Exception:
                target = None
        temp = get_temp()
        with open(out_path, "a", newline="") as f:
            csv.writer(f).writerow([
                time.strftime("%Y-%m-%dT%H:%M:%S"), int(time.time()),
                round(total, 1)] + [round(x, 1) for x in per_core] +
                [round(proc_pct, 1), round(rss, 1),
                 round(temp, 1) if temp else "", pid])
    except KeyboardInterrupt:
        break
    except Exception as e:
        sys.stderr.write(f"[monitor] {e}\n")
PYEOF

run_one() {
    local ENGINE="$1"; local BATCH="$2"; local SCRIPT="$3"
    local RUN_DIR="$RESULTS_DIR/${ENGINE}_b${BATCH}"
    mkdir -p "$RUN_DIR"
    log ""
    log "--- Run: engine=$ENGINE batch=$BATCH ---"

    local INPUT_CSV="$RUN_DIR/live_input.csv"
    head -1 "$ABS_CSV" > "$INPUT_CSV"

    local RES_CSV="$RESULTS_DIR/resources_${ENGINE}_b${BATCH}.csv"
    python3 "$RESOURCE_MONITOR" "$RES_CSV" "$SCRIPT" >> "$LOG" 2>&1 &
    local MON_PID=$!
    sleep 2

    # Build IPS command with ABSOLUTE paths
    local TIMING_CSV="$RESULTS_DIR/timing_${ENGINE}_b${BATCH}.csv"
    local IPS_CMD=(python3 "$SCRIPT_DIR/$SCRIPT"
                   --input "$INPUT_CSV"
                   --batch "$BATCH"
                   --poll 1
                   --dry-run
                   --features "$ABS_FEATS")

    if [[ "$ENGINE" == "cpu" ]]; then
        [[ -n "$ABS_ONNX" ]] && IPS_CMD+=(--model "$ABS_ONNX")
        [[ -n "$ABS_SCALER_JOBLIB" ]] && IPS_CMD+=(--scaler "$ABS_SCALER_JOBLIB")
    else
        [[ -n "$ABS_HEF" ]] && IPS_CMD+=(--hef "$ABS_HEF")
        # ips_hailo.py might use scaler_params.npz instead of joblib — try both
        if [[ -n "$ABS_SCALER_NPZ" ]]; then
            IPS_CMD+=(--scaler "$ABS_SCALER_NPZ")
        elif [[ -n "$ABS_SCALER_JOBLIB" ]]; then
            IPS_CMD+=(--scaler "$ABS_SCALER_JOBLIB")
        fi
    fi

    # ALERT/ACTION logs to run dir (so each run is self-contained)
    IPS_CMD+=(--alert-log "$RUN_DIR/alerts.log"
              --action-log "$RUN_DIR/actions.log")

    log "[cmd] ${IPS_CMD[*]}"

    BENCH=1 BENCH_OUT="$TIMING_CSV" BENCH_VARIANT="$ENGINE" \
        "${IPS_CMD[@]}" >> "$RUN_DIR/ips.log" 2>&1 &
    local IPS_PID=$!

    # SANITY CHECK: is IPS still alive after 5 seconds?
    sleep 5
    if ! kill -0 "$IPS_PID" 2>/dev/null; then
        log "[FATAL] IPS died within 5s. Last 20 lines of ips.log:"
        tail -20 "$RUN_DIR/ips.log" | sed 's/^/        /' | tee -a "$LOG"
        kill -TERM "$MON_PID" 2>/dev/null || true
        return 1
    fi
    log "[ok] IPS alive after 5s, PID=$IPS_PID"

    # Feed the input
    log "[feed] appending ${ABS_CSV##*/} ..."
    tail -n +2 "$ABS_CSV" >> "$INPUT_CSV"

    # Run for DURATION
    log "[run] running for ${DURATION}s ..."
    sleep "$DURATION"

    # Stop everything
    kill -TERM "$IPS_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$IPS_PID" 2>/dev/null || true
    kill -TERM "$MON_PID" 2>/dev/null || true
    sleep 1

    # Report
    if [[ -f "$TIMING_CSV" ]]; then
        local n_batches=$(($(wc -l < "$TIMING_CSV") - 1))
        log "[done] $n_batches batches written"
    else
        log "[WARN] no timing CSV. Last 15 lines of ips.log:"
        tail -15 "$RUN_DIR/ips.log" | sed 's/^/        /' | tee -a "$LOG"
    fi
    sleep 3
}

# Run all combinations
for batch in "${BATCH_SIZES[@]}"; do
    if [[ -f "$SCRIPT_DIR/ips_realtime_v2.py" && -n "$ABS_ONNX" ]]; then
        run_one "cpu"   "$batch" "ips_realtime_v2.py"
    fi
    if [[ -f "$SCRIPT_DIR/ips_hailo.py" && -n "$ABS_HEF" ]]; then
        run_one "hailo" "$batch" "ips_hailo.py"
    fi
done

log ""
log "================================================================"
log "All runs completed. Results: $RESULTS_DIR"
log "================================================================"
ls -la "$RESULTS_DIR" | tee -a "$LOG"
