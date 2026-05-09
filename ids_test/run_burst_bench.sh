#!/bin/bash
# run_burst_bench.sh — Benchmark "burst": populează live CSV cu un dataset mare,
# pornește IPS-ul, așteaptă să termine TOT, măsoară.
#
# Diferențe față de run_bench.sh streaming:
#   - Nu folosește feed_csv.py (CSV-ul e copiat integral la început)
#   - Așteaptă ca IPS-ul să proceseze tot, NU rulează un timer fix
#   - Rezultă "capacitate maximă reală", direct comparabilă cu validate_hailo.py
#
# Folosire:
#   bash run_burst_bench.sh
#
# Variabile opționale:
#   ROOT_DIR       - default: /home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test
#   BURST_CSV      - default: $ROOT_DIR/friday_ddos.csv
#   BATCH          - default: 128
#   POLL           - default: 1 (mai mic ca să detecteze rapid finalizarea)
#   MAX_WAIT_SEC   - timeout total per variantă (default: 600 = 10 min)

set -e

# ---------- CONFIG ----------
ROOT_DIR="${ROOT_DIR:-/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test}"
BURST_CSV="${BURST_CSV:-$ROOT_DIR/friday_ddos.csv}"
LIVE_CSV="$ROOT_DIR/live_sample_bench.csv"

RUN_TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$ROOT_DIR/bench_results/burst_$RUN_TS"

BATCH="${BATCH:-128}"
POLL="${POLL:-1}"
THRESHOLD="${THRESHOLD:-0.5}"   # Pe friday_ddos.csv, 0.5 dă F1=99%, deci e ok
MAX_WAIT_SEC="${MAX_WAIT_SEC:-600}"

VENV_CPU="${VENV_CPU:-$ROOT_DIR/venv_ids/bin/activate}"
VENV_HAILO="${VENV_HAILO:-$ROOT_DIR/venv_hailo_runtime/bin/activate}"
# ----------------------------

if [[ ! -f "$BURST_CSV" ]]; then
    echo "[ERR] CSV-ul de burst lipseste: $BURST_CSV"
    exit 1
fi

mkdir -p "$OUT_DIR"
echo "[BURST] Output: $OUT_DIR"
echo "[BURST] Source CSV: $BURST_CSV"
echo "[BURST] Source rows: $(($(wc -l < "$BURST_CSV") - 1))"
echo "[BURST] Batch: $BATCH | Threshold: $THRESHOLD | Poll: ${POLL}s"

# Curatare reguli iptables vechi
if sudo iptables -L INPUT -n 2>/dev/null | grep -q "192.168.10\|172.16"; then
    echo "[BURST] Curat reguli iptables vechi..."
    sudo iptables -L INPUT -n --line-numbers | grep -E "192.168.10|172.16" \
        | awk '{print $1}' | sort -rn | while read num; do
        sudo iptables -D INPUT "$num" 2>/dev/null || true
    done
fi

# Asteapta ca IPS-ul sa termine procesarea (last_seen == n stable)
wait_for_completion() {
    local LOG_FILE="$1"
    local TIMING_CSV="$2"
    local IPS_PID="$3"

    local LAST_BATCH_COUNT=0
    local STABLE_TICKS=0
    local STABLE_THRESHOLD=4   # 4 ticks de 2 sec = 8 sec fara batch nou = "gata"
    local elapsed=0
    local check_interval=2

    while [[ $elapsed -lt $MAX_WAIT_SEC ]]; do
        sleep "$check_interval"
        elapsed=$((elapsed + check_interval))

        # IPS-ul mai e viu?
        if ! kill -0 "$IPS_PID" 2>/dev/null; then
            echo "[BURST] IPS-ul s-a oprit (probabil eroare). Verifica $LOG_FILE"
            return 1
        fi

        # Cate batch-uri are timing CSV pana acum?
        local CURRENT_BATCH_COUNT=0
        if [[ -f "$TIMING_CSV" ]]; then
            CURRENT_BATCH_COUNT=$(($(wc -l < "$TIMING_CSV") - 1))
        fi

        if [[ $CURRENT_BATCH_COUNT -gt $LAST_BATCH_COUNT ]]; then
            echo "[BURST]   batch-uri procesate: $CURRENT_BATCH_COUNT (elapsed: ${elapsed}s)"
            LAST_BATCH_COUNT=$CURRENT_BATCH_COUNT
            STABLE_TICKS=0
        else
            STABLE_TICKS=$((STABLE_TICKS + 1))
            if [[ $STABLE_TICKS -ge $STABLE_THRESHOLD ]]; then
                echo "[BURST] Procesarea pare incheiata (no-progress timeout)."
                return 0
            fi
        fi
    done

    echo "[BURST] Timeout maxim atins (${MAX_WAIT_SEC}s)."
    return 1
}

# Functie: ruleaza o varianta in burst mode
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

    # Activeaza venv-ul
    if [[ -f "$VENV_PATH" ]]; then
        # shellcheck disable=SC1090
        source "$VENV_PATH"
    else
        echo "[ERR] Venv lipseste: $VENV_PATH"
        return 1
    fi

    # Pasul 1: porneste IPS-ul cu un live_sample VID (doar header), ca sa apuce
    # sa-si initializeze sesiunea ONNX/Hailo inainte sa primeasca date
    head -n 1 "$BURST_CSV" > "$LIVE_CSV"
    echo "[BURST] Live CSV initial: doar header"

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

    # Pasul 2: warm-up (ONNX session + Hailo VDevice load)
    echo "[BURST] Warm-up 6s pentru initializare..."
    sleep 6

    if ! kill -0 "$IPS_PID" 2>/dev/null; then
        echo "[ERR] IPS-ul a crapat la pornire. Vezi: $LOG_FILE"
        tail -25 "$LOG_FILE"
        return 1
    fi

    # Pasul 3: porneste resource sampler
    echo "[BURST] Start resource sampler..."
    python3 "$ROOT_DIR/resource_sampler.py" \
        --pid "$IPS_PID" \
        --variant "$VARIANT" \
        --out "$RESOURCES_CSV" \
        --interval 1 \
        --duration "$MAX_WAIT_SEC" > "$OUT_DIR/sampler_$VARIANT.log" 2>&1 &
    SAMPLER_PID=$!

    # Pasul 4: BURST! Copiem CSV-ul intreg in live_sample.csv intr-o singura operatie atomica
    echo "[BURST] Inject TOT CSV-ul ($(wc -l < "$BURST_CSV") linii) in live_sample..."
    local T0
    T0=$(date +%s.%N)
    cp "$BURST_CSV" "$LIVE_CSV.tmp"
    mv "$LIVE_CSV.tmp" "$LIVE_CSV"   # rename atomic
    echo "[BURST] Injectat. T0 = $T0"

    # Pasul 5: asteapta sa termine
    echo "[BURST] Astept procesare completa..."
    wait_for_completion "$LOG_FILE" "$TIMING_CSV" "$IPS_PID"
    local WAIT_RC=$?
    local T1
    T1=$(date +%s.%N)
    local ELAPSED
    ELAPSED=$(echo "$T1 - $T0" | bc)
    echo "[BURST] Timp total burst: ${ELAPSED}s"
    echo "$ELAPSED" > "$OUT_DIR/wall_time_$VARIANT.txt"

    # Pasul 6: stop ordonat
    echo "[BURST] Stop sampler & IPS..."
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    sleep 1
    kill -TERM "$IPS_PID" 2>/dev/null || true
    sleep 3
    kill -KILL "$IPS_PID" "$SAMPLER_PID" 2>/dev/null || true

    # Mic raport
    echo ""
    echo "[BURST] $VARIANT terminat."
    if [[ -f "$TIMING_CSV" ]]; then
        local NB
        NB=$(($(wc -l < "$TIMING_CSV") - 1))
        echo "      timing:    $TIMING_CSV ($NB batch-uri)"
    else
        echo "      [WARN] timing CSV NU s-a scris"
    fi
    if [[ -f "$RESOURCES_CSV" ]]; then
        local NS
        NS=$(($(wc -l < "$RESOURCES_CSV") - 1))
        echo "      resources: $RESOURCES_CSV ($NS sample-uri)"
    fi
    echo "      wall time: ${ELAPSED}s"

    deactivate 2>/dev/null || true
}

# ---------- Ruleaza variantele ----------

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

# ---------- Genereaza raportul ----------
echo ""
echo "============================================================"
echo "  GENERATE BURST REPORT"
echo "============================================================"

if [[ -f "$VENV_CPU" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_CPU"
fi
python3 "$ROOT_DIR/gen_report.py" --bench-dir "$OUT_DIR" --out "$OUT_DIR/report.md"

# Adauga si wall-time end-to-end (informativ, comparabil cu validate_hailo.py)
echo ""
echo "============================================================"
echo "  WALL TIME END-TO-END"
echo "============================================================"
for v in cpu hailo; do
    if [[ -f "$OUT_DIR/wall_time_$v.txt" ]]; then
        WT=$(cat "$OUT_DIR/wall_time_$v.txt")
        SRC_ROWS=$(($(wc -l < "$BURST_CSV") - 1))
        # echo "$v: $WT s pentru $SRC_ROWS randuri = ?"
        printf "  %-6s : %.2fs (%d randuri)\n" "$v" "$WT" "$SRC_ROWS"
    fi
done

echo ""
echo "[BURST] DONE."
echo "[BURST] Raport: $OUT_DIR/report.md"
echo "[BURST] CSVs:   $OUT_DIR/"
