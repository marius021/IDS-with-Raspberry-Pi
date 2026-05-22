#!/usr/bin/env bash
# run_scaling.sh
# Phase E — Scaling benchmark CPU (ONNX) vs Hailo (HEF) for batch sizes 32/128/512.
# Produces bench_results/<timestamp>/{timing_*.csv, resources_*.csv, run.log}
#
# Usage:
#   bash run_scaling.sh [csv_input] [duration_sec]
#
# Defaults:
#   csv_input    = ~/Desktop/IDS-with-Raspberry-Pi/ids_test/friday_ddos.csv
#   duration_sec = 90
#
# Prerequisites:
#   - venv with onnxruntime + hailo_platform active
#   - ips_realtime_v2.py (CPU/ONNX) and ips_hailo.py (NPU) in PWD
#   - bench_timing.py writes timing_cpu.csv / timing_hailo.csv automatically

set -u

# --- Args ---
CSV_INPUT="${1:-$HOME/Desktop/IDS-with-Raspberry-Pi/ids_test/friday_ddos.csv}"
DURATION="${2:-90}"
BATCH_SIZES=(32 128 512)

# --- Paths ---
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TS="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="$HOME/bench_results/${TS}"
mkdir -p "$RESULTS_DIR"
LOG="$RESULTS_DIR/run.log"

# --- Helpers ---
log() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

cleanup() {
    log "Cleanup: killing any leftover IPS / monitor processes"
    pkill -f "ips_realtime_v2.py" 2>/dev/null || true
    pkill -f "ips_hailo.py" 2>/dev/null || true
    pkill -f "monitor_resources" 2>/dev/null || true
    sleep 1
}
trap cleanup EXIT INT TERM

# --- Sanity checks ---
log "================================================================"
log "Phase E — Scaling Benchmark"
log "================================================================"
log "Results dir : $RESULTS_DIR"
log "CSV input   : $CSV_INPUT"
log "Duration    : ${DURATION}s per run"
log "Batch sizes : ${BATCH_SIZES[*]}"
log "================================================================"

if [[ ! -f "$CSV_INPUT" ]]; then
    log "[ERROR] CSV input not found: $CSV_INPUT"
    exit 1
fi
for script in ips_realtime_v2.py ips_hailo.py; do
    if [[ ! -f "$SCRIPT_DIR/$script" ]]; then
        log "[WARN] $script not found in $SCRIPT_DIR — that engine will be skipped"
    fi
done

# Number of rows in source CSV
CSV_ROWS=$(wc -l < "$CSV_INPUT")
log "Source CSV: $CSV_ROWS rows"

# --- Resource monitor (inline Python) ---
RESOURCE_MONITOR="$RESULTS_DIR/monitor_resources.py"
cat > "$RESOURCE_MONITOR" << 'EOF'
#!/usr/bin/env python3
"""Resource monitor: samples CPU% (total + per-core + per-process),
RAM, and temperature once per second. CSV output."""
import sys, time, csv, os
import psutil

if len(sys.argv) < 3:
    print("Usage: monitor_resources.py <output.csv> <target_pname>")
    sys.exit(1)

out_path = sys.argv[1]
target_name = sys.argv[2]

def get_temp():
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except Exception:
        try:
            t = psutil.sensors_temperatures(fahrenheit=False)
            for k, v in t.items():
                if v: return v[0].current
        except Exception:
            pass
    return None

def find_proc(name):
    for p in psutil.process_iter(['name', 'cmdline', 'pid']):
        try:
            cmd = ' '.join(p.info['cmdline'] or [])
            if name in cmd and 'monitor_resources' not in cmd:
                return p
        except Exception:
            continue
    return None

# Header
n_cores = psutil.cpu_count(logical=True)
header = ["ts_iso", "ts_epoch", "cpu_total_pct"]
for i in range(n_cores):
    header.append(f"core{i}_pct")
header += ["cpu_proc_pct", "rss_mb", "temp_c", "proc_pid"]

with open(out_path, "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(header)

# Prime psutil.cpu_percent
psutil.cpu_percent(percpu=True)
target_proc = None
sample_count = 0

while True:
    try:
        time.sleep(1)
        sample_count += 1

        per_core = psutil.cpu_percent(percpu=True)
        total = sum(per_core) / len(per_core)

        if target_proc is None or not target_proc.is_running():
            target_proc = find_proc(target_name)

        proc_pct = 0.0
        rss = 0.0
        pid = 0
        if target_proc and target_proc.is_running():
            try:
                proc_pct = target_proc.cpu_percent(interval=None)
                rss = target_proc.memory_info().rss / 1e6
                pid = target_proc.pid
            except Exception:
                target_proc = None

        temp = get_temp()
        row = [time.strftime("%Y-%m-%dT%H:%M:%S"), int(time.time()),
               round(total, 1)] + [round(x, 1) for x in per_core] + \
              [round(proc_pct, 1), round(rss, 1),
               round(temp, 1) if temp else "", pid]
        with open(out_path, "a", newline="") as f:
            csv.writer(f).writerow(row)
    except KeyboardInterrupt:
        break
    except Exception as e:
        sys.stderr.write(f"[monitor] {e}\n")
EOF
chmod +x "$RESOURCE_MONITOR"

# --- Run one benchmark ---
run_one() {
    local ENGINE="$1"        # cpu | hailo
    local BATCH="$2"
    local SCRIPT="$3"        # ips_realtime_v2.py | ips_hailo.py

    local RUN_DIR="$RESULTS_DIR/${ENGINE}_b${BATCH}"
    mkdir -p "$RUN_DIR"
    log ""
    log "--- Run: engine=$ENGINE batch=$BATCH ---"

    # 1. Make a fresh copy of the CSV the IPS will tail
    local INPUT_CSV="$RUN_DIR/live_input.csv"
    head -1 "$CSV_INPUT" > "$INPUT_CSV"   # header only at start

    # 2. Start resource monitor
    local RES_CSV="$RESULTS_DIR/resources_${ENGINE}_b${BATCH}.csv"
    python3 "$RESOURCE_MONITOR" "$RES_CSV" "$SCRIPT" >> "$LOG" 2>&1 &
    local MON_PID=$!
    log "[monitor] started PID=$MON_PID -> $RES_CSV"
    sleep 2

    # 3. Start IPS in background
    local TIMING_CSV="$RESULTS_DIR/timing_${ENGINE}_b${BATCH}.csv"
    cd "$SCRIPT_DIR"
    BENCH_OUT="$TIMING_CSV" BENCH_VARIANT="$ENGINE" \
        python3 "$SCRIPT" \
            --input "$INPUT_CSV" \
            --batch "$BATCH" \
            --poll 1 \
            --dry-run \
            >> "$RUN_DIR/ips.log" 2>&1 &
    local IPS_PID=$!
    log "[ips] started PID=$IPS_PID engine=$ENGINE batch=$BATCH"
    sleep 3   # let IPS load model

    # 4. Feed CSV gradually — append all rows in chunks (simulates stream)
    log "[feed] appending rows to $INPUT_CSV ..."
    tail -n +2 "$CSV_INPUT" >> "$INPUT_CSV"   # append all data rows

    # 5. Let it run for DURATION seconds
    log "[run] running for ${DURATION}s ..."
    sleep "$DURATION"

    # 6. Stop everything
    kill -TERM "$IPS_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$IPS_PID" 2>/dev/null || true
    kill -TERM "$MON_PID" 2>/dev/null || true
    sleep 1

    # 7. Summary
    if [[ -f "$TIMING_CSV" ]]; then
        local n_batches=$(($(wc -l < "$TIMING_CSV") - 1))
        log "[done] $n_batches batches written to $(basename $TIMING_CSV)"
    else
        log "[WARN] no timing CSV produced for $ENGINE b=$BATCH"
    fi
    if [[ -f "$RES_CSV" ]]; then
        local n_samples=$(($(wc -l < "$RES_CSV") - 1))
        log "[done] $n_samples samples in $(basename $RES_CSV)"
    fi
    sleep 3   # cooldown
}

# --- Main loop ---
for batch in "${BATCH_SIZES[@]}"; do
    if [[ -f "$SCRIPT_DIR/ips_realtime_v2.py" ]]; then
        run_one "cpu"   "$batch" "ips_realtime_v2.py"
    fi
    if [[ -f "$SCRIPT_DIR/ips_hailo.py" ]]; then
        run_one "hailo" "$batch" "ips_hailo.py"
    fi
done

log ""
log "================================================================"
log "All runs completed."
log "Results: $RESULTS_DIR"
log "Next step: python3 gen_report.py --bench-dir $RESULTS_DIR --out scaling_report.md"
log "================================================================"
ls -la "$RESULTS_DIR" | tee -a "$LOG"
