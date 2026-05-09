#!/bin/bash
# run_bench.sh — Orchestrator benchmark CPU vs Hailo (v2)
#
# Modificări v2:
#   - Argumente feed_csv.py corecte: --chunk + --delay + --loop
#   - Verificare sample_big.csv exists, fallback la sample_labeled.csv
#   - Trunchiere live_sample.csv corectă cu doar header
#   - Verificare iptables pre-curățare (avertisment dacă sunt reguli)
#
# Folosire:
#   bash run_bench.sh
#
# Variabile de mediu (toate opționale):
#   ROOT_DIR          - default: /home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test
#   SOURCE_CSV        - default: $ROOT_DIR/sample_big.csv (fallback sample_labeled.csv)
#   RUN_DURATION      - sec per variantă (default: 90)
#   FEED_CHUNK        - rânduri per tick (default: 50)
#   FEED_DELAY        - sec între tick-uri (default: 1.0 — dă ~50 rows/sec)
#   THRESHOLD         - default: 0.01
#   BATCH             - default: 32
#   POLL              - default: 2

set -e

# ---------- CONFIG ----------
ROOT_DIR="${ROOT_DIR:-/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test}"

# Caută o sursă CSV mai mare; fallback la sample_labeled.csv
if [[ -f "$ROOT_DIR/sample_big.csv" ]]; then
    SOURCE_CSV="${SOURCE_CSV:-$ROOT_DIR/sample_big.csv}"
elif [[ -f "$ROOT_DIR/sample_labeled.csv" ]]; then
    SOURCE_CSV="${SOURCE_CSV:-$ROOT_DIR/sample_labeled.csv}"
else
    echo "[ERR] Nu gasesc nici sample_big.csv, nici sample_labeled.csv in $ROOT_DIR"
    exit 1
fi

LIVE_CSV="$ROOT_DIR/live_sample_bench.csv"
RUN_TS=$(date +%Y%m%d_%H%M%S)
OUT_DIR="$ROOT_DIR/bench_results/$RUN_TS"
RUN_DURATION="${RUN_DURATION:-90}"

# Feed: 50 randuri / 1 sec = ~50 rows/sec sustained
FEED_CHUNK="${FEED_CHUNK:-50}"
FEED_DELAY="${FEED_DELAY:-1.0}"

THRESHOLD="${THRESHOLD:-0.01}"
BATCH="${BATCH:-32}"
POLL="${POLL:-2}"
VENV_CPU="${VENV_CPU:-$ROOT_DIR/venv_ids/bin/activate}"
VENV_HAILO="${VENV_HAILO:-$ROOT_DIR/venv_hailo_runtime/bin/activate}"
# ----------------------------

mkdir -p "$OUT_DIR"
echo "[RUN] Output: $OUT_DIR"
echo "[RUN] Source: $SOURCE_CSV"
echo "[RUN] Duration per variant: ${RUN_DURATION}s"
echo "[RUN] Feed: $FEED_CHUNK rows / ${FEED_DELAY}s tick"

# Avertisment daca exista reguli iptables active pe IP-uri de test
if sudo iptables -L INPUT -n 2>/dev/null | grep -q "192.168.10"; then
    echo "[WARN] Reguli iptables active pe 192.168.10.* — poate afecta benchmark-ul."
    echo "[WARN] Curata cu: sudo iptables -D INPUT -s 192.168.10.X -j DROP"
fi

# Pregateste live_sample.csv vid (doar header)
head -n 1 "$SOURCE_CSV" > "$LIVE_CSV"

# Functie: ruleaza o varianta
run_variant () {
    local VARIANT="$1"
    local IPS_SCRIPT="$2"
    local VENV_PATH="$3"
    local EXTRA_ARGS="$4"

    echo ""
    echo "============================================================"
    echo "  BENCHMARK: $VARIANT  ($IPS_SCRIPT)"
    echo "============================================================"

    # Reseteaza live_sample (doar header)
    head -n 1 "$SOURCE_CSV" > "$LIVE_CSV"

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

    # Porneste IPS-ul cu BENCH activ
    echo "[RUN] Start IPS ($VARIANT)..."
    BENCH=1 BENCH_OUT="$TIMING_CSV" BENCH_VARIANT="$VARIANT" \
    python3 "$ROOT_DIR/$IPS_SCRIPT" \
        --input "$LIVE_CSV" \
        --threshold "$THRESHOLD" \
        --batch "$BATCH" \
        --poll "$POLL" \
        --dry-run \
        $EXTRA_ARGS > "$LOG_FILE" 2>&1 &
    IPS_PID=$!
    echo "[RUN] IPS PID = $IPS_PID"

    sleep 4   # warm-up: ONNX session + Hailo VDevice load dureaza

    # Verifica ca IPS-ul e inca viu
    if ! kill -0 "$IPS_PID" 2>/dev/null; then
        echo "[ERR] IPS-ul a crapat la pornire. Vezi: $LOG_FILE"
        tail -20 "$LOG_FILE"
        return 1
    fi

    # Porneste resource sampler (paralel)
    echo "[RUN] Start resource sampler..."
    python3 "$ROOT_DIR/resource_sampler.py" \
        --pid "$IPS_PID" \
        --variant "$VARIANT" \
        --out "$RESOURCES_CSV" \
        --interval 1 \
        --duration "$RUN_DURATION" > "$OUT_DIR/sampler_$VARIANT.log" 2>&1 &
    SAMPLER_PID=$!

    # Porneste feed-ul in paralel — varsa SOURCE_CSV in LIVE_CSV
    echo "[RUN] Start feed: chunk=$FEED_CHUNK delay=$FEED_DELAY"
    python3 "$ROOT_DIR/feed_csv.py" \
        --src "$SOURCE_CSV" \
        --dst "$LIVE_CSV" \
        --chunk "$FEED_CHUNK" \
        --delay "$FEED_DELAY" \
        --loop > "$OUT_DIR/feed_$VARIANT.log" 2>&1 &
    FEED_PID=$!

    # Verificare dupa 3 sec ca feed-ul ruleaza
    sleep 3
    if ! kill -0 "$FEED_PID" 2>/dev/null; then
        echo "[ERR] Feed-ul a crapat. Vezi: $OUT_DIR/feed_$VARIANT.log"
        tail -10 "$OUT_DIR/feed_$VARIANT.log"
        kill -TERM "$IPS_PID" "$SAMPLER_PID" 2>/dev/null || true
        return 1
    fi

    # Asteapta durata totala
    echo "[RUN] Rulez timp de ${RUN_DURATION}s..."
    sleep "$RUN_DURATION"

    # Stop ordonat
    echo "[RUN] Stop feed -> sampler -> IPS..."
    kill -TERM "$FEED_PID" 2>/dev/null || true
    sleep 1
    kill -TERM "$SAMPLER_PID" 2>/dev/null || true
    sleep 1
    kill -TERM "$IPS_PID" 2>/dev/null || true
    sleep 3
    kill -KILL "$IPS_PID" 2>/dev/null || true
    kill -KILL "$FEED_PID" "$SAMPLER_PID" 2>/dev/null || true

    # Mic raport pe consola
    echo ""
    echo "[RUN] $VARIANT terminat."
    if [[ -f "$TIMING_CSV" ]]; then
        local NB
        NB=$(($(wc -l < "$TIMING_CSV") - 1))
        echo "      timing:    $TIMING_CSV ($NB batch-uri)"
    else
        echo "      [WARN] timing CSV NU s-a scris — verifica $LOG_FILE"
    fi
    if [[ -f "$RESOURCES_CSV" ]]; then
        local NS
        NS=$(($(wc -l < "$RESOURCES_CSV") - 1))
        echo "      resources: $RESOURCES_CSV ($NS sample-uri)"
    fi

    deactivate 2>/dev/null || true
}

# ---------- Ruleaza variantele ----------

# 1. CPU (ONNX)
run_variant "cpu" "ips_realtime_v2.py" "$VENV_CPU" \
    "--model $ROOT_DIR/ids_mlp_binary.onnx \
     --scaler $ROOT_DIR/scaler.joblib \
     --features $ROOT_DIR/feature_names.npy \
     --alert-log $OUT_DIR/alerts_cpu.log \
     --action-log $OUT_DIR/actions_cpu.log"

# 2. Hailo
run_variant "hailo" "ips_hailo.py" "$VENV_HAILO" \
    "--hef $ROOT_DIR/ids_mlp_binary_logits.hef \
     --scaler $ROOT_DIR/scaler_params.npz \
     --features $ROOT_DIR/feature_names.txt \
     --alert-log $OUT_DIR/alerts_hailo.log \
     --action-log $OUT_DIR/actions_hailo.log"

# ---------- Genereaza raportul ----------
echo ""
echo "============================================================"
echo "  GENERATE REPORT"
echo "============================================================"

if [[ -f "$VENV_CPU" ]]; then
    # shellcheck disable=SC1090
    source "$VENV_CPU"
fi
python3 "$ROOT_DIR/gen_report.py" --bench-dir "$OUT_DIR" --out "$OUT_DIR/report.md"

echo ""
echo "[RUN] DONE."
echo "[RUN] Raport: $OUT_DIR/report.md"
echo "[RUN] CSVs:   $OUT_DIR/"