# IDS/IPS with Raspberry Pi + Hailo

A real-time intrusion detection and prevention system built around a binary MLP model trained on CICIDS-style network flow data. The model runs on a Raspberry Pi, optionally accelerated by a Hailo-8 AI chip.

**Detection backend:** ONNX Runtime (CPU) or Hailo HEF (NPU)
**Prevention action:** `iptables DROP` on the source IP of detected attacks
**Input:** a growing CSV of network flow features (e.g. produced by CICFlowMeter)

---

## Repository layout

```
IDS-with-Raspberry-Pi/
├── ids_inference.py            Offline batch inference (CSV → CSV with predictions)
├── ids_test/
│   ├── ids_realtime.py         Real-time IDS only (alerts, no blocking)
│   ├── ips_realtime_v1.py      Real-time IPS — ONNX, basic
│   ├── ips_realtime_v2.py      Real-time IPS — ONNX, with seen-IP dedup + bench instrumentation
│   ├── ips_hailo.py            Real-time IPS — Hailo HEF, no scikit-learn + bench instrumentation
│   ├── feed_csv.py             Simulate a live stream by appending to a growing CSV
│   ├── export_scaler.py        Convert scaler.joblib → scaler_params.npz (for Hailo runtime)
│   ├── feature_names_generate.py  Convert feature_names.npy → feature_names.txt
│   ├── make_logits_onnx.py     Rewrite ONNX output to expose raw logits
│   ├── bench_timing.py         Per-stage timing module (opt-in via BENCH=1 env var)
│   ├── resource_sampler.py     Parallel CPU/RAM/temperature sampler (uses psutil)
│   ├── gen_report.py           Generate Markdown benchmark report from timing/resource CSVs
│   ├── gen_scaling_report.py   Generate scaling report across multiple batch sizes
│   ├── run_bench.sh            Streaming benchmark orchestrator: CPU vs Hailo, fixed duration
│   ├── run_burst_bench.sh      Burst benchmark: inject full CSV at once, wait for completion
│   ├── run_scaling.sh          Run run_bench.sh across multiple batch sizes
│   ├── run_burst_scaling.sh    Run run_burst_bench.sh across multiple batch sizes
│   ├── ids_mlp_binary.onnx     Binary model (sigmoid output)
│   ├── ids_mlp_binary_logits.hef  Hailo HEF model (logits output, sigmoid applied in Python)
│   ├── scaler.joblib           StandardScaler for ONNX runtime
│   ├── scaler_params.npz       Scaler parameters for Hailo runtime (no sklearn needed)
│   ├── feature_names.npy / .txt  Ordered list of expected input features
│   ├── whitelist.txt           IPs that are never blocked
│   ├── live_sample_bench.csv   Dedicated live CSV used exclusively by benchmark scripts
│   ├── bench_results/          Output directory for all benchmark runs (timestamped subdirs)
│   └── sample.csv              Small example of expected CSV format
├── Debug Scripts/
│   ├── csv-preprocessing.py    Full training pipeline (CICIDS CSV → ONNX + artifacts)
│   ├── check_artifacts.py      Validate that model, scaler, features, and CSV are aligned
│   ├── eval_quick.py           Print classification report from a predictions CSV
│   └── multiclass_test.py      Inspect sample CSV columns and types
├── Dell Files/
│   ├── Dockerfile              Container image for training / development
│   ├── docker-compose.yaml     Training and inference services
│   ├── requirements.txt        Full dependency set (PyTorch, scikit-learn, ONNX)
│   ├── tasks.json              VS Code Docker run tasks
│   └── launch.json             VS Code Docker debug configuration
├── Raspi Files/
│   └── ids_realtime.py         Lean IDS for Raspberry Pi (ONNX, alert-only)
├── requirements.txt            Minimal runtime dependencies (no torch, no sklearn)
├── classes.npy                 Class names for the multiclass model
└── label_encoder.joblib        Label encoder used during training
```

---

## Hardware

| Component | Notes |
|-----------|-------|
| Raspberry Pi 4 or 5 | tested and confirmed working |
| Hailo-8 M.2 / HAT | optional — enables NPU acceleration via `ips_hailo.py` |
| Network tap / port mirror | feed live traffic to CICFlowMeter running on the Pi |

---

## Dependencies

### Raspberry Pi (inference only, no Hailo)

```bash
pip install -r requirements.txt
```

Pinned versions: `numpy 2.4`, `pandas 3.0`, `onnxruntime 1.24`, `joblib 1.5`.

### Raspberry Pi (with Hailo)

Install HailoRT Python bindings from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) for your HailoRT version, then:

```bash
pip install -r requirements.txt
```

`ips_hailo.py` has no scikit-learn dependency — it uses `scaler_params.npz` directly.

### Dell / training machine

```bash
pip install -r "Dell Files/requirements.txt"
```

Includes PyTorch 2.3, scikit-learn 1.7, ONNX 1.x.

---

## Workflow

### 1 — Train a model (Dell)

```bash
python "Debug Scripts/csv-preprocessing.py" --data /path/to/cicids_csvs/
```

This produces in the working directory:
- `ids_mlp_binary.onnx` — binary classifier
- `ids_mlp_multiclass.onnx` — attack-type classifier
- `scaler.joblib`, `feature_names.npy`, `classes.npy`, `label_encoder.joblib`

### 2 — Export artifacts for the Raspberry Pi

```bash
# Export scaler for the Hailo runtime (no sklearn on Pi)
python ids_test/export_scaler.py --scaler scaler.joblib --out ids_test/scaler_params.npz

# Convert feature list to plain text (safer across numpy versions)
python ids_test/feature_names_generate.py
```

### 3 — Copy artifacts to the Pi

Transfer these files to `~/ids/` on the Raspberry Pi:

```
ids_mlp_binary.onnx           (CPU / ONNX runtime)
ids_mlp_binary_logits.hef     (Hailo runtime)
scaler.joblib                  (CPU / ONNX runtime)
scaler_params.npz              (Hailo runtime)
feature_names.npy or .txt
whitelist.txt
```

### 4 — Validate alignment

```bash
python "Debug Scripts/check_artifacts.py" \
    --model ids_test/ids_mlp_binary.onnx \
    --scaler ids_test/scaler.joblib \
    --features ids_test/feature_names.npy \
    --sample ids_test/sample.csv
```

---

## Running the system

### Offline batch inference

Useful for validating a CSV of captured traffic against the model.

```bash
python ids_inference.py \
    --input traffic.csv \
    --model ids_test/ids_mlp_binary.onnx \
    --scaler ids_test/scaler.joblib \
    --features ids_test/feature_names.npy \
    --mode binary \
    --out predictions.csv
```

For the multiclass model:

```bash
python ids_inference.py \
    --input traffic.csv \
    --model ids_mlp_multiclass.onnx \
    --scaler scaler.joblib \
    --features feature_names.npy \
    --classes classes.npy \
    --mode multi \
    --out predictions.csv
```

Output columns added: `pred_attack_proba`, `pred_label` (binary) or `pred_class_idx`, `pred_class` (multi).

---

### Real-time IDS — alerts only, no blocking (ONNX)

```bash
python ids_test/ids_realtime.py \
    --input ~/ids/live_sample.csv \
    --model ~/ids/ids_mlp_binary.onnx \
    --scaler ~/ids/scaler.joblib \
    --features ~/ids/feature_names.npy \
    --alert-log ~/ids/alerts.log \
    --poll 3 \
    --threshold 0.5
```

Monitors `live_sample.csv`. Every 3 s it reads new rows, runs inference, and appends attack events to `alerts.log` as JSON lines:

```json
{"ts": 1744300012, "prob": 0.971, "row_index": 142}
```

---

### Real-time IPS — ONNX + iptables blocking

```bash
# Safe test — prints what would be blocked without touching iptables
python ids_test/ips_realtime_v2.py \
    --input ~/ids/live_sample.csv \
    --model ~/ids/ids_mlp_binary.onnx \
    --scaler ~/ids/scaler.joblib \
    --features ~/ids/feature_names.npy \
    --whitelist ~/ids/whitelist.txt \
    --dry-run

# Live mode — actually blocks IPs
python ids_test/ips_realtime_v2.py \
    --input ~/ids/live_sample.csv \
    --model ~/ids/ids_mlp_binary.onnx \
    --scaler ~/ids/scaler.joblib \
    --features ~/ids/feature_names.npy \
    --whitelist ~/ids/whitelist.txt \
    --threshold 0.5 \
    --poll 3
```

Two logs are written:
- `alerts.log` — every detected attack row
- `actions.log` — every IP blocking decision (including `skip` reasons)

`ips_realtime_v2.py` deduplicates: once an IP is seen in a session it is not re-checked or re-logged. Use `ips_realtime_v1.py` if you prefer all hits logged even for repeat IPs.

---

### Real-time IPS — Hailo NPU

Requires Hailo HEF model and HailoRT Python bindings.

```bash
# Safe test
python ids_test/ips_hailo.py \
    --input ~/ids/live_sample.csv \
    --hef ~/ids/ids_mlp_binary_logits.hef \
    --scaler ~/ids/scaler_params.npz \
    --features ~/ids/feature_names.txt \
    --whitelist ~/ids/whitelist.txt \
    --threshold 0.5 \
    --dry-run

# Live mode
python ids_test/ips_hailo.py \
    --input ~/ids/live_sample.csv \
    --hef ~/ids/ids_mlp_binary_logits.hef \
    --scaler ~/ids/scaler_params.npz \
    --features ~/ids/feature_names.txt \
    --whitelist ~/ids/whitelist.txt \
    --threshold 0.5
```

The HEF model outputs raw logits; `ips_hailo.py` applies sigmoid internally. The feature file is auto-discovered in the input CSV's directory if `--features` is omitted (searches for `.txt`, `.json`, `.csv`, `.npy`).

**Note:** A lower threshold (e.g. `--threshold 0.005`) was used during initial Hailo testing to account for quantization effects. Tune this against a known-good traffic sample.

---

### Simulating a live stream (testing)

```bash
python ids_test/feed_csv.py \
    --source ids_test/sample.csv \
    --out ~/ids/live_sample.csv \
    --chunk 50 \
    --interval 2
```

Appends 50 rows from `sample.csv` every 2 seconds, mimicking a live flow export.

---

## CLI reference

### `ids_inference.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Input CSV with network flow features |
| `--model` | required | ONNX model path |
| `--scaler` | `scaler.joblib` | Saved scaler |
| `--mode` | required | `binary` or `multi` |
| `--out` | `predictions.csv` | Output CSV path |
| `--features` | `feature_names.npy` | Ordered feature list from training |
| `--classes` | `classes.npy` | Class names (multi mode only) |
| `--threshold` | `0.5` | Attack probability threshold (binary only) |

### `ips_realtime_v2.py` / `ids_realtime.py`

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `~/ids/live_sample.csv` | Growing CSV to monitor |
| `--model` | `~/ids/ids_mlp_binary.onnx` | ONNX model |
| `--scaler` | `~/ids/scaler.joblib` | Scaler |
| `--features` | `~/ids/feature_names.npy` | Feature list |
| `--alert-log` | `~/ids/alerts.log` | Alert output path |
| `--action-log` | `~/ids/actions.log` | IPS action log path |
| `--whitelist` | _(none)_ | Path to whitelist file |
| `--poll` | `3` | Polling interval in seconds |
| `--batch` | `1024` | Inference batch size |
| `--threshold` | `0.5` | Attack probability threshold |
| `--dry-run` | off | Log what would be blocked, don't touch iptables |
| `--debug` | off | Verbose output |

### `ips_hailo.py`

Same as above, with these replacements:

| Argument | Default | Description |
|----------|---------|-------------|
| `--hef` | `~/Desktop/.../ids_mlp_binary.hef` | Hailo HEF model path |
| `--scaler` | `…/scaler_params.npz` | NPZ scaler params (no sklearn) |
| `--features` | _(auto-discover)_ | Feature file; auto-found if omitted |

---

## Benchmarking

The benchmarking subsystem measures end-to-end IPS latency and resource usage for CPU (ONNX) vs Hailo (HEF). All timing is **opt-in** — normal IPS operation is unchanged when the `BENCH` env var is not set.

### Architecture overview

```
run_bench.sh / run_burst_bench.sh
  ├── starts ips_realtime_v2.py  (BENCH=1)  → timing_cpu.csv
  ├── starts ips_hailo.py        (BENCH=1)  → timing_hailo.csv
  ├── starts resource_sampler.py (parallel) → resources_cpu.csv / resources_hailo.csv
  └── calls gen_report.py at the end        → report.md
```

Each batch written by an IPS loop contributes one row to the timing CSV, with columns:
`timestamp, variant, batch_idx, n_rows, t_preprocess_ms, t_inference_ms, t_postprocess_ms, t_log_ms, t_total_ms`

---

### Streaming benchmark — sustained throughput (run_bench.sh)

Runs both CPU and Hailo variants for a fixed duration while `feed_csv.py` continuously appends rows to `live_sample_bench.csv`.

```bash
# Default run (90 s per variant, batch=32, ~50 rows/s feed)
bash ids_test/run_bench.sh

# Override any parameter via env vars
RUN_DURATION=120 BATCH=128 FEED_CHUNK=100 FEED_DELAY=1.0 \
    bash ids_test/run_bench.sh
```

Results land in `ids_test/bench_results/<YYYYMMDD_HHMMSS>/`:
- `timing_cpu.csv`, `timing_hailo.csv`
- `resources_cpu.csv`, `resources_hailo.csv`
- `report.md` — auto-generated Markdown with latency / throughput / resource tables

Environment variables for `run_bench.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_DIR` | `ids_test/` | Path to the ids_test directory |
| `SOURCE_CSV` | `sample_big.csv` → `sample_labeled.csv` | CSV to feed from |
| `RUN_DURATION` | `90` | Seconds per variant |
| `FEED_CHUNK` | `50` | Rows appended per tick |
| `FEED_DELAY` | `1.0` | Seconds between feed ticks |
| `THRESHOLD` | `0.01` | Attack detection threshold |
| `BATCH` | `32` | Inference batch size |
| `POLL` | `2` | IPS polling interval (seconds) |
| `VENV_CPU` | `venv_ids/bin/activate` | Python venv for ONNX variant |
| `VENV_HAILO` | `venv_hailo_runtime/bin/activate` | Python venv for Hailo variant |

---

### Burst benchmark — max throughput on a static dataset (run_burst_bench.sh)

Copies a full CSV into `live_sample_bench.csv` atomically, then waits for the IPS to drain it completely. Measures wall-clock time end-to-end — directly comparable with offline `ids_inference.py` runs.

Requires a larger labeled CSV (e.g. `friday_ddos.csv`). The script auto-detects completion by monitoring the timing CSV row count.

```bash
# Default run (batch=128, threshold=0.5, max wait=600 s)
bash ids_test/run_burst_bench.sh

# Override
BURST_CSV=ids_test/friday_ddos.csv BATCH=256 \
    bash ids_test/run_burst_bench.sh
```

Results land in `ids_test/bench_results/burst_<YYYYMMDD_HHMMSS>/` and include wall-time per variant in `wall_time_cpu.txt` / `wall_time_hailo.txt`.

Environment variables for `run_burst_bench.sh`:

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_DIR` | `ids_test/` | Path to the ids_test directory |
| `BURST_CSV` | `friday_ddos.csv` | Full dataset to burst-inject |
| `BATCH` | `128` | Inference batch size |
| `POLL` | `1` | IPS polling interval (seconds) |
| `THRESHOLD` | `0.5` | Attack detection threshold |
| `MAX_WAIT_SEC` | `600` | Per-variant timeout (seconds) |

---

### Scaling benchmark — sweep across batch sizes (run_scaling.sh / run_burst_scaling.sh)

Runs the streaming or burst benchmark for each value in `BATCH_SIZES` and aggregates all results into a single scaling report.

```bash
# Streaming scaling (batch sizes: 32, 128, 256)
BATCH_SIZES="32 128 256" RUN_DURATION=60 bash ids_test/run_scaling.sh

# Burst scaling (batch sizes: 32, 128, 512)
BATCH_SIZES="32 128 512" bash ids_test/run_burst_scaling.sh
```

Results land in `bench_results/scaling_<ts>/batch_<N>/` or `bench_results/burst_scaling_<ts>/batch_<N>/`.
The final `scaling_report.md` contains 8 tables (throughput, latency, p95, inference, preprocess, CPU, RAM — one column per batch size for each variant).

---

### Run bench_timing manually (single variant)

```bash
# CPU variant — timing active
BENCH=1 BENCH_OUT=ids_test/timing_cpu.csv BENCH_VARIANT=cpu \
    python ids_test/ips_realtime_v2.py \
        --input ids_test/live_sample_bench.csv \
        --model ids_test/ids_mlp_binary.onnx \
        --scaler ids_test/scaler.joblib \
        --features ids_test/feature_names.npy \
        --whitelist ids_test/whitelist.txt \
        --threshold 0.01 --batch 32 --dry-run

# Hailo variant — timing active
BENCH=1 BENCH_OUT=ids_test/timing_hailo.csv BENCH_VARIANT=hailo \
    python ids_test/ips_hailo.py \
        --input ids_test/live_sample_bench.csv \
        --hef ids_test/ids_mlp_binary_logits.hef \
        --scaler ids_test/scaler_params.npz \
        --features ids_test/feature_names.txt \
        --whitelist ids_test/whitelist.txt \
        --threshold 0.01 --batch 32 --dry-run
```

---

### Resource sampler (standalone)

Attach to a running IPS process to sample CPU, RAM, and CPU temperature at 1-second intervals.

```bash
# Attach by PID
python ids_test/resource_sampler.py \
    --pid <IPS_PID> \
    --variant cpu \
    --out ids_test/resources_cpu.csv \
    --interval 1 \
    --duration 120

# Attach by process name (auto-discovers PID)
python ids_test/resource_sampler.py \
    --proc-name ips_realtime_v2.py \
    --variant cpu \
    --out ids_test/resources_cpu.csv
```

Requires `psutil` (`pip install psutil`). Stop with Ctrl-C or via `--duration`.

Output columns: `timestamp, variant, cpu_total_pct, cpu_proc_pct, rss_mb, temp_c, core0_pct … coreN_pct`

---

### Generate reports manually

```bash
# Standard report from a single benchmark run directory
python ids_test/gen_report.py \
    --bench-dir ids_test/bench_results/20260510_120000 \
    --out report.md

# Scaling report from a scaling benchmark directory
python ids_test/gen_scaling_report.py \
    --scaling-dir ids_test/bench_results/scaling_20260510_120000 \
    --out scaling_report.md
```

`gen_report.py` produces four tables: per-stage latency (ms/batch), end-to-end throughput (rows/s, p50/p95/p99), system resource utilization, and per-core CPU distribution.

`gen_scaling_report.py` produces eight tables (S1–S8) covering throughput, batch latency, per-inference latency, inference-only time, preprocess time, p95 latency, CPU%, and RAM, with the winner bolded per row.

---

## Whitelist

`ids_test/whitelist.txt` — one IP per line, lines starting with `#` are comments.

```
# gateway
192.168.1.1
# monitoring host
192.168.1.140
```

The defaults `127.0.0.1`, `0.0.0.0`, and `::1` are always whitelisted in code regardless of the file.

---

## Input CSV format

The CSV must contain the same numeric feature columns that were present during training. Column names are normalized to lowercase with leading/trailing whitespace stripped, so `Flow Duration`, `flow duration`, and `flow_duration` all resolve to the same feature.

An expected feature list is stored in `ids_test/feature_names.txt` (80 features). The source IP column is detected automatically from any of: `src ip`, `src_ip`, `source ip`, `source_ip`, `ip src`, `ip_src`.

Non-numeric columns and extra numeric columns are ignored. Missing required columns cause the script to exit with a clear error listing the missing names.

---

## Hailo model conversion (reference)

The HEF files already in the repository were produced from `ids_mlp_binary_logits.onnx` using the Hailo Model Zoo toolchain. Key steps:

```bash
# 1. Parse ONNX → HAR
hailo parse onnx ids_mlp_binary_logits.onnx --net-name ids_mlp_binary_logits

# 2. Optimize / quantize
hailo optimize ids_mlp_binary_logits.har --use-random-calib-set --output ids_mlp_binary_logits_quant.har

# 3. Compile → HEF
hailo compile ids_mlp_binary_logits_quant.har --output ids_mlp_binary_logits.hef
```

The logits variant is used because Hailo quantization degrades sigmoid output; applying sigmoid post-inference in Python on the raw logits gives better results.

---

## Progress log

| Date | Milestone |
|------|-----------|
| 28.09.2025 | IDS flow integrated into Docker project; Hailo infer flow pending fix |
| 14.10.2025 | Offline inference working on Raspberry Pi; `ids_realtime.py` initial version |
| 02.04.2026 | IDS and IPS working on CPU/ONNX without Hailo |
| 10.04.2026 | Hailo pipeline runs end-to-end; HEF loading, preprocessing, and batch inference confirmed; `iptables` dry-run verified |
| 10.05.2026 | Benchmarking subsystem added: `bench_timing.py` (opt-in per-stage timer), `resource_sampler.py` (CPU/RAM/temp), `gen_report.py`, `gen_scaling_report.py`; both `ips_realtime_v2.py` and `ips_hailo.py` instrumented; four orchestration shell scripts added (`run_bench.sh`, `run_burst_bench.sh`, `run_scaling.sh`, `run_burst_scaling.sh`) for streaming, burst, and scaling benchmarks |

### Next steps

- Run full streaming and burst scaling benchmarks on Pi 5 with `friday_ddos.csv` to get CPU vs Hailo comparison numbers
- Tune threshold per variant based on benchmark F1 results
- Document actual benchmark results (throughput, latency tables) once collected
- Consider keeping Hailo context open between batches to reduce per-batch activation overhead (~10–50 ms)
