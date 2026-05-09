#!/bin/bash
# run_burst_scaling.sh — Burst benchmark cu mai multe batch sizes.
#
# Rulează run_burst_bench.sh cu BATCH=32, 128, 512 (sau ce e in BATCH_SIZES),
# apoi agregă rezultatele într-un raport unic de scaling.
#
# Folosire:
#   bash run_burst_scaling.sh
#
# Variabile opționale:
#   BATCH_SIZES   - default: "32 128 512"
#   BURST_CSV     - default: $ROOT_DIR/friday_ddos.csv
#   MAX_WAIT_SEC  - timeout per rulare (default: 600)

set -e

ROOT_DIR="${ROOT_DIR:-/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test}"
BATCH_SIZES="${BATCH_SIZES:-32 128 512}"
BURST_CSV="${BURST_CSV:-$ROOT_DIR/friday_ddos.csv}"
MAX_WAIT_SEC="${MAX_WAIT_SEC:-600}"

SCALING_TS=$(date +%Y%m%d_%H%M%S)
SCALING_DIR="$ROOT_DIR/bench_results/burst_scaling_$SCALING_TS"
mkdir -p "$SCALING_DIR"

echo "=================================================="
echo "  BURST SCALING BENCHMARK"
echo "=================================================="
echo "  Batch sizes:  $BATCH_SIZES"
echo "  Source CSV:   $BURST_CSV"
echo "  Source rows:  $(($(wc -l < "$BURST_CSV") - 1))"
echo "  Output:       $SCALING_DIR"
echo ""

for BATCH_VAL in $BATCH_SIZES; do
    echo ""
    echo "##################################################"
    echo "##  BURST cu BATCH = $BATCH_VAL"
    echo "##################################################"

    BATCH="$BATCH_VAL" \
    BURST_CSV="$BURST_CSV" \
    MAX_WAIT_SEC="$MAX_WAIT_SEC" \
        bash "$ROOT_DIR/run_burst_bench.sh" || {
        echo "[ERR] Rulare cu BATCH=$BATCH_VAL a esuat. Continui."
        continue
    }

    # Mut rezultatele in directorul de scaling
    LATEST_DIR=$(ls -1dt "$ROOT_DIR/bench_results/burst_"*/ 2>/dev/null \
                 | grep -v "burst_scaling_" | head -1 | sed 's:/$::')
    if [[ -z "$LATEST_DIR" ]]; then
        echo "[WARN] Nu am gasit directorul de output pentru BATCH=$BATCH_VAL"
        continue
    fi

    echo "[SCALE] Mut rezultate in $SCALING_DIR/batch_${BATCH_VAL}/"
    mv "$LATEST_DIR" "$SCALING_DIR/batch_${BATCH_VAL}"
done

# Genereaza raportul de scaling
echo ""
echo "=================================================="
echo "  GENERARE RAPORT SCALING"
echo "=================================================="

if [[ -f "$ROOT_DIR/venv_ids/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$ROOT_DIR/venv_ids/bin/activate"
fi

python3 "$ROOT_DIR/gen_scaling_report.py" \
    --scaling-dir "$SCALING_DIR" \
    --out "$SCALING_DIR/scaling_report.md"

# Adauga wall time la raport
echo ""
echo "=================================================="
echo "  WALL TIME END-TO-END (sumar)"
echo "=================================================="

WALL_TIME_TABLE="$SCALING_DIR/wall_time_summary.md"
{
    echo "# Wall Time End-to-End per BATCH"
    echo ""
    echo "Sursa: $BURST_CSV"
    echo "Numar randuri: $(($(wc -l < "$BURST_CSV") - 1))"
    echo ""
    echo "| Batch | CPU (sec) | Hailo (sec) | CPU rows/s | Hailo rows/s |"
    echo "|---|---:|---:|---:|---:|"
    SRC_ROWS=$(($(wc -l < "$BURST_CSV") - 1))
    for d in "$SCALING_DIR"/batch_*/; do
        if [[ -d "$d" ]]; then
            BATCH_VAL=$(basename "$d" | sed 's/batch_//')
            CPU_WT=$(cat "$d/wall_time_cpu.txt" 2>/dev/null || echo "—")
            HAILO_WT=$(cat "$d/wall_time_hailo.txt" 2>/dev/null || echo "—")
            CPU_RPS="—"
            HAILO_RPS="—"
            if [[ "$CPU_WT" != "—" ]]; then
                CPU_RPS=$(echo "scale=1; $SRC_ROWS / $CPU_WT" | bc)
            fi
            if [[ "$HAILO_WT" != "—" ]]; then
                HAILO_RPS=$(echo "scale=1; $SRC_ROWS / $HAILO_WT" | bc)
            fi
            CPU_WT_FMT=$(printf "%.2f" "$CPU_WT" 2>/dev/null || echo "$CPU_WT")
            HAILO_WT_FMT=$(printf "%.2f" "$HAILO_WT" 2>/dev/null || echo "$HAILO_WT")
            echo "| $BATCH_VAL | $CPU_WT_FMT | $HAILO_WT_FMT | $CPU_RPS | $HAILO_RPS |"
        fi
    done
} > "$WALL_TIME_TABLE"

cat "$WALL_TIME_TABLE"

echo ""
echo "[DONE] Raport scaling:  $SCALING_DIR/scaling_report.md"
echo "[DONE] Raport wall-time: $WALL_TIME_TABLE"
