#!/bin/bash
# run_scaling.sh — Rulează benchmark cu mai multe batch sizes și agregă rezultatele.
#
# Pentru fiecare BATCH din BATCH_SIZES, apelează run_bench.sh cu durata redusă (60s).
# La final, agregă timing CSVs în tabelul de scaling.
#
# Folosire:
#   bash run_scaling.sh
#
# Variabile opționale:
#   BATCH_SIZES   - lista de batch-uri (default: "32 128 256")
#   RUN_DURATION  - sec per rulare (default: 60)

set -e

ROOT_DIR="${ROOT_DIR:-/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test}"
BATCH_SIZES="${BATCH_SIZES:-32 128 256}"
RUN_DURATION="${RUN_DURATION:-60}"

SCALING_TS=$(date +%Y%m%d_%H%M%S)
SCALING_DIR="$ROOT_DIR/bench_results/scaling_$SCALING_TS"
mkdir -p "$SCALING_DIR"

echo "=================================================="
echo "  SCALING BENCHMARK"
echo "=================================================="
echo "  Batch sizes: $BATCH_SIZES"
echo "  Duration per batch: ${RUN_DURATION}s"
echo "  Output: $SCALING_DIR"
echo ""

# Ruleaza pentru fiecare batch size
for BATCH_VAL in $BATCH_SIZES; do
    echo ""
    echo "##################################################"
    echo "##  BATCH = $BATCH_VAL"
    echo "##################################################"

    # Marker fisier pentru a sti unde e rezultatul curent
    BEFORE=$(ls -1 "$ROOT_DIR/bench_results/" 2>/dev/null | wc -l)

    # Apeleaza run_bench.sh cu BATCH si RUN_DURATION setate
    BATCH="$BATCH_VAL" RUN_DURATION="$RUN_DURATION" \
        bash "$ROOT_DIR/run_bench.sh" || {
        echo "[ERR] Rulare cu BATCH=$BATCH_VAL a esuat. Continui cu urmatorul."
        continue
    }

    # Gaseste directorul nou creat
    LATEST_DIR=$(ls -1dt "$ROOT_DIR/bench_results/"*/ 2>/dev/null | head -1 | sed 's:/$::')
    if [[ -z "$LATEST_DIR" ]]; then
        echo "[WARN] Nu am gasit directorul de output pentru BATCH=$BATCH_VAL"
        continue
    fi

    # Mut rezultatele in directorul de scaling, cu prefix de batch
    echo "[SCALE] Mut rezultate in $SCALING_DIR/batch_${BATCH_VAL}/"
    mv "$LATEST_DIR" "$SCALING_DIR/batch_${BATCH_VAL}"
done

# Genereaza tabelul de scaling
echo ""
echo "=================================================="
echo "  GENERARE TABEL SCALING"
echo "=================================================="

if [[ -f "$ROOT_DIR/venv_ids/bin/activate" ]]; then
    # shellcheck disable=SC1090
    source "$ROOT_DIR/venv_ids/bin/activate"
fi

python3 "$ROOT_DIR/gen_scaling_report.py" \
    --scaling-dir "$SCALING_DIR" \
    --out "$SCALING_DIR/scaling_report.md"

echo ""
echo "[DONE] Raport scaling: $SCALING_DIR/scaling_report.md"
