<h1 align="center">IDS/IPS with Raspberry Pi + Hailo-8</h1>

<p align="center">
  <em>A real-time, ML-based intrusion detection & prevention system that runs entirely on the edge —<br>
  from packet capture to automatic kernel-level blocking of attackers.</em>
</p>

<p align="center">
  <img alt="Platform" src="https://img.shields.io/badge/platform-Raspberry%20Pi%205-c51a4a">
  <img alt="Accelerator" src="https://img.shields.io/badge/accelerator-Hailo--8-00b4a0">
  <img alt="Model" src="https://img.shields.io/badge/model-MLP%20(CICIDS2017)-4b8bbe">
  <img alt="F1" src="https://img.shields.io/badge/F1-99.20%25-success">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-lightgrey">
</p>

---

A binary MLP classifier trained on CICIDS2017 network-flow data runs on a Raspberry Pi 5, optionally accelerated by a Hailo-8 AI chip. The system closes the full loop: **capture → flow extraction → inference → block**.

|  |  |
|--|--|
| **Detection backend** | ONNX Runtime (CPU) or Hailo HEF (NPU) |
| **Prevention action** | `iptables DROP` on the source IP of detected attacks |
| **Input** | a CSV of network-flow features, produced on the Pi by `pcap_to_cicids.py` (scapy-based, no external flow meter required) |

---

## Table of contents

- [Results at a glance](#results-at-a-glance)
- [How it works](#how-it-works)
- [Quick start](#quick-start)
- [Repository layout](#repository-layout)
- [Hardware](#hardware)
- [Dependencies](#dependencies)
- [Full workflow](#full-workflow)
- [Running the system](#running-the-system)
- [PCAP processing](#pcap-processing--primary-flow-extraction-path)
- [Legacy CICFlowMeter converters](#legacy--cicflowmeter-converters)
- [Blocking demo & end-to-end attack test](#blocking-demo--end-to-end-attack-test)
- [Hailo utilities](#hailo-utilities)
- [CLI reference](#cli-reference)
- [Benchmarking](#benchmarking)
- [Input CSV format](#input-csv-format)
- [Progress log](#progress-log)
- [Future work](#future-work)

---

## Results at a glance

The system was validated end-to-end on a Raspberry Pi 5 + Hailo-8.

| Metric | Value |
|--------|-------|
| **F1 score** — Hailo-8, INT8 (CICIDS2017 Friday DDoS) | **99.20%** |
| F1 score — ONNX, FP32 baseline | 99.84% |
| Quantization F1 degradation | −0.64 pp |
| **CPU usage reduction** (Hailo vs ONNX, batch 32) | **−71.8%** |
| **Max temperature reduction** (Hailo vs ONNX, batch 32) | **−10.5 °C** |
| Hailo raw HW throughput (`hailortcli benchmark`) | 56,568 inferences/s |
| End-to-end attack test (real SYN flood) | detected @ prob 1.000, attacker IP blocked |

> **The takeaway:** on a small MLP (~10k parameters) the Hailo-8 does **not** win on raw throughput — a modern CPU keeps up easily. Its value is that it moves inference almost entirely **off** the CPU, freeing it for capture, flow extraction, logging, and iptables, while running markedly cooler. For a system meant to run unattended 24/7, that resource and thermal headroom is the decisive advantage.

---

## How it works

```
                    ┌─────────────────────── Raspberry Pi 5 ───────────────────────┐
   Network          │                                                              │
   traffic  ─────►  │  tcpdump ──► capture.pcap ──► pcap_to_cicids.py ──► flows.csv │
   (eth0)           │                                   (scapy)              │      │
                    │                                                        ▼      │
                    │                                              ┌──────────────┐ │
                    │                                              │  ips_hailo   │ │
                    │                                              │  (HEF / NPU) │ │
                    │                                              │   or ONNX    │ │
                    │                                              └──────┬───────┘ │
                    │                                  prob > 0.5 & not   │         │
                    │                                  whitelisted ?      ▼         │
                    │                                          iptables -A INPUT    │
                    │                                          -s <ip> -j DROP      │
                    └──────────────────────────────────────────────────────────────┘

   Off-device (development workstation): training, ONNX export, Hailo HEF compilation.
```

---

## Quick start

Already have the artifacts on a Pi with Hailo? Detect & block in three commands:

```bash
# 1. Capture some traffic
sudo tcpdump -i eth0 -w capture.pcap

# 2. Turn the capture into CICIDS2017 flows
python3 ids_test/pcap_to_cicids.py capture.pcap flows.csv

# 3. Run the IPS (drop --dry-run to actually block)
python3 ids_test/ips_hailo.py \
    --input flows.csv \
    --hef ids_test/ids_mlp_binary_logits.hef \
    --scaler ids_test/scaler_params.npz \
    --features ids_test/feature_names.txt \
    --whitelist ids_test/whitelist.txt \
    --dry-run
```

---

## Repository layout

```
IDS-with-Raspberry-Pi/
├── ids_inference.py            Offline batch inference (CSV → CSV with predictions)
├── requirements.txt            Minimal runtime dependencies (no torch, no sklearn)
├── classes.npy                 Class names for the multiclass model
├── label_encoder.joblib        Label encoder used during training
│
├── ids_test/
│   ├── ids_realtime.py         Real-time IDS only (alerts, no blocking)
│   ├── ips_realtime_v2.py      Real-time IPS — ONNX, seen-IP dedup + bench instrumentation
│   ├── ips_hailo.py            Real-time IPS — Hailo HEF, no scikit-learn + bench instrumentation
│   ├── pcap_to_cicids.py       PCAP → CICIDS2017 CSV via scapy (primary flow extractor)
│   ├── feed_csv.py             Simulate a live stream by appending to a growing CSV
│   │
│   ├── validate_hailo.py       End-to-end HEF validation: F1 / threshold sweep / ONNX comparison
│   ├── check_hailo_artifacts.py  Pre-flight check: HEF loads + I/O shapes + synthetic run
│   ├── recompile_hailo.py      Recompile ONNX → HEF with runtime-matched calibration data
│   │
│   ├── bench_timing.py         Per-stage timing module (opt-in via BENCH=1 env var)
│   ├── resource_sampler.py     Parallel CPU/RAM/temperature sampler (uses psutil)
│   ├── gen_report.py           Markdown benchmark report from timing/resource CSVs
│   ├── gen_report_v2.py        Extended report: 8 tables + 6 PNG figures
│   ├── gen_scaling_report.py   Scaling report across multiple batch sizes
│   ├── run_bench.sh            Streaming benchmark: CPU vs Hailo, fixed duration
│   ├── run_burst_bench.sh      Burst benchmark: inject full CSV at once
│   ├── run_scaling.sh          run_bench.sh across multiple batch sizes
│   ├── run_scaling_v2.sh       Scaling benchmark with absolute paths + health checks
│   ├── run_burst_scaling.sh    run_burst_bench.sh across multiple batch sizes
│   ├── demo_phase_F1.sh        IPS reaction demo: blocking response to a controlled attack
│   ├── bench_results/          Output dir for benchmark runs (timestamped subdirs)
│   │
│   ├── ids_mlp_binary.onnx         Binary model (sigmoid output, CPU path)
│   ├── ids_mlp_binary_logits.onnx  Binary model (logits output, source for the HEF)
│   ├── ids_mlp_binary_logits.hef   Hailo HEF model (logits; sigmoid applied in Python)
│   ├── scaler.joblib               StandardScaler for the ONNX runtime
│   ├── scaler_params.npz           Scaler mean/std for the Hailo runtime (no sklearn)
│   ├── feature_names.npy / .txt    Ordered list of the 80 input features
│   ├── whitelist.txt               IPs that are never blocked
│   └── sample.csv                  Small example of the expected CSV format
│
├── Debug Scripts/
│   ├── csv-preprocessing.py    Full training pipeline (CICIDS CSV → ONNX + artifacts)
│   ├── make_sample_labeled.py  Build a balanced BENIGN/ATTACK CSV from CICIDS2017 files
│   ├── check_artifacts.py      Validate that model, scaler, features, and CSV align
│   └── eval_quick.py           Print a classification report from a predictions CSV
│
└── legacy/
    ├── cicfm_to_cicids.py      CICFlowMeter Python fork CSV → CICIDS2017 (time-unit fix)
    └── cicflow_to_cicids_v2.py CICFlowMeter Java v4 CSV → CICIDS2017 (verified renames)
```

> Large/local files (`friday_ddos.csv`, `sample_labeled.csv`, `sample_big.csv`, `live_sample*.csv`), the CICIDS2017 dataset directory, virtual environments, and runtime logs are git-ignored. See `.gitignore`.

---

## Hardware

| Component | Notes |
|-----------|-------|
| **Raspberry Pi 5** | primary platform — tested and confirmed working |
| **Hailo-8** (M.2 / AI HAT) | optional — enables NPU acceleration via `ips_hailo.py` |
| Active cooler | recommended for sustained benchmarking on the Pi 5 |
| Ethernet (wired) | traffic is captured on the wired interface with `tcpdump`; Wi-Fi is disabled on the Pi to reduce capture noise |

A second machine (the **development workstation**, with an NVIDIA GPU satisfying Hailo's CUDA requirements) handles training, HEF compilation, and — in the validation experiments — acts as a controlled attack traffic source. It is **not needed at runtime**.

---

## Dependencies

<details>
<summary><strong>Raspberry Pi — inference only (no Hailo)</strong></summary>

```bash
pip install -r requirements.txt
```

> Pin the runtime dependencies (numpy, pandas, onnxruntime, joblib) to versions known to work on your Raspberry Pi OS image, and verify `requirements.txt` matches what is actually installed.
</details>

<details>
<summary><strong>Raspberry Pi — with Hailo</strong></summary>

Install HailoRT and its Python bindings from the [Hailo Developer Zone](https://hailo.ai/developer-zone/) for your HailoRT version, then:

```bash
pip install -r requirements.txt
pip install scapy            # required by pcap_to_cicids.py
```

`ips_hailo.py` has **no scikit-learn dependency** — it uses `scaler_params.npz` directly. The only scikit-learn touchpoint on the Pi is `scaler.joblib` for the ONNX path; if used, it must match the version the scaler was created with (`scikit-learn==1.3.2`).
</details>

<details>
<summary><strong>Development / training machine</strong></summary>

Training (`Debug Scripts/csv-preprocessing.py`) requires PyTorch, scikit-learn, ONNX, pandas, and NumPy. Keep the training-side scikit-learn aligned with the version required to deserialize `scaler.joblib` on the Pi (`1.3.2`), to avoid version-mismatch warnings or subtle scaler differences.
</details>

---

## Full workflow

### 1 — Build a labeled sample (workstation)

`make_sample_labeled.py` builds a balanced BENIGN/ATTACK CSV from the raw CICIDS2017 files, later consumed by `validate_hailo.py`.

```bash
python "Debug Scripts/make_sample_labeled.py" \
    --data-dir "/path/to/CICIDS2017/MachineLearningCVE" \
    --out ids_test/sample_labeled.csv \
    --per-file 300
```

### 2 — Train a model (workstation)

```bash
python "Debug Scripts/csv-preprocessing.py" --data /path/to/cicids_csvs/
```

Produces: `ids_mlp_binary.onnx`, `ids_mlp_multiclass.onnx`, `scaler.joblib`, `feature_names.npy`, `classes.npy`, `label_encoder.joblib`.

### 3 — Compile for Hailo (workstation)

Use `recompile_hailo.py` to produce the HEF with **runtime-matched calibration data** (see [Hailo utilities](#hailo-utilities)). Preferred over a manual parse/optimize/compile with a random calibration set — calibration quality directly affects quantization accuracy.

### 4 — Required artifacts on the Pi

```
ids_mlp_binary.onnx           (CPU / ONNX runtime)
ids_mlp_binary_logits.hef     (Hailo runtime)
scaler.joblib                 (CPU / ONNX runtime)
scaler_params.npz             (Hailo runtime — scaler mean/std as a plain NPZ)
feature_names.npy / .txt      (ordered list of 80 feature names)
whitelist.txt
```

`scaler_params.npz` holds the StandardScaler's `mean` and `scale` arrays so the Hailo path can normalize inputs without importing scikit-learn. `feature_names.txt` is a plain-text copy of `feature_names.npy`, more robust across numpy versions.

### 5 — Validate alignment

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

```bash
python ids_inference.py \
    --input traffic.csv \
    --model ids_test/ids_mlp_binary.onnx \
    --scaler ids_test/scaler.joblib \
    --features ids_test/feature_names.npy \
    --mode binary \
    --out predictions.csv
```

Multiclass: swap in `ids_mlp_multiclass.onnx`, add `--classes classes.npy --mode multi`.
Output columns added: `pred_attack_proba`, `pred_label` (binary) or `pred_class_idx`, `pred_class` (multi).

### Real-time IDS — alerts only (ONNX)

```bash
python ids_test/ids_realtime.py \
    --input ~/ids/live_sample.csv \
    --model ~/ids/ids_mlp_binary.onnx \
    --scaler ~/ids/scaler.joblib \
    --features ~/ids/feature_names.npy \
    --alert-log ~/ids/alerts.log \
    --poll 3 --threshold 0.5
```

Monitors a growing CSV and appends attack events to `alerts.log` as JSON lines:

```json
{"ts": 1744300012, "prob": 0.971, "row_index": 142}
```

### Real-time IPS — ONNX + iptables blocking

```bash
# Safe test (no iptables changes)
python ids_test/ips_realtime_v2.py \
    --input ~/ids/live_sample.csv --model ~/ids/ids_mlp_binary.onnx \
    --scaler ~/ids/scaler.joblib --features ~/ids/feature_names.npy \
    --whitelist ~/ids/whitelist.txt --dry-run

# Live mode (blocks IPs)
python ids_test/ips_realtime_v2.py \
    --input ~/ids/live_sample.csv --model ~/ids/ids_mlp_binary.onnx \
    --scaler ~/ids/scaler.joblib --features ~/ids/feature_names.npy \
    --whitelist ~/ids/whitelist.txt --threshold 0.5 --poll 3
```

Two logs: `alerts.log` (every detected attack row) and `actions.log` (every blocking decision, including `skip` reasons). `ips_realtime_v2.py` deduplicates — once an IP is seen in a session it is not re-checked or re-logged.

### Real-time IPS — Hailo NPU

```bash
# Safe test
python ids_test/ips_hailo.py \
    --input ~/ids/live_sample.csv --hef ~/ids/ids_mlp_binary_logits.hef \
    --scaler ~/ids/scaler_params.npz --features ~/ids/feature_names.txt \
    --whitelist ~/ids/whitelist.txt --threshold 0.5 --dry-run

# Live mode
python ids_test/ips_hailo.py \
    --input ~/ids/live_sample.csv --hef ~/ids/ids_mlp_binary_logits.hef \
    --scaler ~/ids/scaler_params.npz --features ~/ids/feature_names.txt \
    --whitelist ~/ids/whitelist.txt --threshold 0.5
```

The HEF outputs raw logits; `ips_hailo.py` applies sigmoid internally. The feature file is auto-discovered in the input CSV's directory if `--features` is omitted.

### Simulating a live stream (testing)

```bash
python ids_test/feed_csv.py \
    --source ids_test/sample.csv --out ~/ids/live_sample.csv \
    --chunk 50 --interval 2
```

---

## PCAP processing — primary flow extraction path

`pcap_to_cicids.py` reads a raw PCAP (from `tcpdump` or Wireshark) and computes all 80 CICIDS2017 features per flow — **no external flow meter needed**. This is the path used in the end-to-end attack experiments.

```
• scapy (pure Python) packet parsing — fully portable on ARM
• groups packets into bidirectional flows by 5-tuple
• computes the 80 CICIDS2017 statistical features directly
• output CSV is directly compatible with ips_hailo.py / ips_realtime_v2.py
• column names match CICIDS2017 directly — no rename/mapping step
```

```bash
sudo tcpdump -i eth0 -w capture.pcap
python3 ids_test/pcap_to_cicids.py capture.pcap flows.csv
```

Requires `scapy` (`pip install scapy`).

> **⚡ Attack-traffic tip:** when generating a SYN flood with `hping3`, **fix the source port** (`-s <port> --keep`). Without it, each packet gets its own ephemeral port and the capture explodes into thousands of one-packet flows whose statistical signature doesn't match the trained DDoS pattern — leading to non-detection. A fixed source port consolidates packets into a few coherent flows the model recognizes correctly.

---

## Legacy — CICFlowMeter converters

> **Deprecated in favor of `pcap_to_cicids.py`.** Kept in `legacy/` for reference and for users who already have CICFlowMeter output. CICFlowMeter (Java) has native dependencies that are awkward on the Pi's ARM architecture — the reason the pure-Python extractor was written.

```bash
# CICFlowMeter Java v4 → CICIDS2017
python legacy/cicflow_to_cicids_v2.py capture_Flow.csv out.csv --features ids_test/feature_names.npy

# CICFlowMeter Python fork → CICIDS2017 (also converts seconds → microseconds)
python legacy/cicfm_to_cicids.py capture_Flow.csv out.csv
```

`0 features missing` in the final check means the output is fully model-compatible.

---

## Blocking demo & end-to-end attack test

### Phase F — IPS reaction to a controlled attack

`demo_phase_F1.sh` backs up iptables, starts `ips_hailo.py` live, injects a single DDoS flow from a chosen IP, and verifies the DROP rule was added.

```bash
bash ids_test/demo_phase_F1.sh 10.99.99.42   # IP must be outside your LAN
```

The script refuses to run if the target IP is inside your local network.

### End-to-end attack (capture → detect → block)

```bash
# 1. On the Pi: listener + capture
nc -l -p 9999 > /dev/null &
sudo tcpdump -i eth0 -w attack.pcap "host <ATTACKER_IP> and host <PI_IP>" &

# 2. From the attack machine: SYN flood with a FIXED source port
sudo hping3 -i u100 -c 5000 -s 31337 --keep -S -p 9999 <PI_IP>

# 3. On the Pi: extract flows and run the IPS for real
python3 ids_test/pcap_to_cicids.py attack.pcap attack_flows.csv
cp attack_flows.csv live_sample.csv
python3 -u ids_test/ips_hailo.py \
    --input live_sample.csv --hef ids_test/ids_mlp_binary_logits.hef \
    --scaler ids_test/scaler_params.npz --features ids_test/feature_names.npy \
    --whitelist ids_test/whitelist.txt --batch 100 --poll 1 --debug

# 4. Confirm the kernel is dropping the attacker's packets
sudo iptables -L INPUT -v -n --line-numbers
```

> **Reference run:** 14,862 captured packets → 2 consolidated flows → both classified as attack at probability **1.000** → attacker IP blocked, **250 packets / 30,105 bytes** dropped by the kernel during the IPS window.

---

## Hailo utilities

<details>
<summary><strong>Pre-flight artifact check</strong></summary>

```bash
python ids_test/check_hailo_artifacts.py \
    --hef ids_test/ids_mlp_binary_logits.hef \
    --scaler ids_test/scaler_params.npz \
    --features ids_test/feature_names.txt
```

Five checks: file existence, scaler key shapes, feature count, HEF I/O shapes, and a synthetic all-zeros inference pass.
</details>

<details>
<summary><strong>Model validation on labeled data</strong></summary>

```bash
python ids_test/validate_hailo.py \
    --csv ids_test/sample_labeled.csv \
    --hef ids_test/ids_mlp_binary_logits.hef \
    --scaler ids_test/scaler_params.npz \
    --features ids_test/feature_names.txt \
    --sweep --onnx ids_test/ids_mlp_binary.onnx \
    --out validation_report.json --batch 512
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--csv` | required | Labeled CSV (must have a `Label` column) |
| `--hef` | required | Hailo HEF model file |
| `--scaler` | required | `scaler_params.npz` |
| `--features` | required | Feature names file |
| `--threshold` | `0.5` | Classification threshold |
| `--sweep` | off | Sweep thresholds to find best F1 |
| `--onnx` | _(none)_ | ONNX model for side-by-side comparison |
| `--batch` | `512` | Hailo inference batch size |
| `--out` | _(none)_ | Save results as JSON |
</details>

<details>
<summary><strong>Recompile HEF (workstation)</strong></summary>

If Hailo inference gives wrong results (e.g. all logits collapsed to a near-constant), the HEF was quantized with mismatched calibration data. `recompile_hailo.py` regenerates it using StandardScaler-normalized float32 inputs that exactly match runtime, with `opt_level=2` (bias correction + adaround).

```bash
source hailo_env/bin/activate
cd ids_test
python3 recompile_hailo.py
python3 recompile_hailo.py --calib-csv sample_labeled.csv --n-calib 512
```
</details>

---

## CLI reference

<details>
<summary><strong><code>ids_inference.py</code></strong></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | required | Input CSV with network flow features |
| `--model` | required | ONNX model path |
| `--scaler` | `scaler.joblib` | Saved scaler |
| `--mode` | required | `binary` or `multi` |
| `--out` | `predictions.csv` | Output CSV path |
| `--features` | `feature_names.npy` | Ordered feature list |
| `--classes` | `classes.npy` | Class names (multi mode only) |
| `--threshold` | `0.5` | Attack probability threshold (binary) |
</details>

<details>
<summary><strong><code>ips_realtime_v2.py</code> / <code>ids_realtime.py</code></strong></summary>

| Argument | Default | Description |
|----------|---------|-------------|
| `--input` | `~/ids/live_sample.csv` | Growing CSV to monitor |
| `--model` | `~/ids/ids_mlp_binary.onnx` | ONNX model |
| `--scaler` | `~/ids/scaler.joblib` | Scaler |
| `--features` | `~/ids/feature_names.npy` | Feature list |
| `--alert-log` | `~/ids/alerts.log` | Alert output path |
| `--action-log` | `~/ids/actions.log` | IPS action log path |
| `--whitelist` | _(none)_ | Whitelist file |
| `--poll` | `3` | Polling interval (s) |
| `--batch` | `1024` | Inference batch size |
| `--threshold` | `0.5` | Attack probability threshold |
| `--dry-run` | off | Log only, don't touch iptables |
| `--debug` | off | Verbose output |
</details>

<details>
<summary><strong><code>ips_hailo.py</code></strong></summary>

Same as above, with:

| Argument | Default | Description |
|----------|---------|-------------|
| `--hef` | `.../ids_mlp_binary_logits.hef` | Hailo HEF model path |
| `--scaler` | `.../scaler_params.npz` | NPZ scaler params (no sklearn) |
| `--features` | _(auto-discover)_ | Feature file; auto-found if omitted |
</details>

---

## Benchmarking

All timing is **opt-in** — normal IPS operation is unchanged when the `BENCH` env var is not set.

```
run_bench.sh / run_burst_bench.sh
  ├── starts ips_realtime_v2.py  (BENCH=1)  → timing_cpu.csv
  ├── starts ips_hailo.py        (BENCH=1)  → timing_hailo.csv
  ├── starts resource_sampler.py (parallel) → resources_cpu.csv / resources_hailo.csv
  └── calls gen_report.py at the end        → report.md
```

### Headline numbers (Pi 5 + Hailo-8, 90 s per config, steady state)

| Backend | Batch | Throughput (rows/s) | Avg proc CPU (%) | Max temp (°C) |
|---------|------:|--------------------:|-----------------:|--------------:|
| CPU (ONNX)    | 32  | 1,516 | 342.4 | 67.8 |
| **Hailo (HEF)** | 32  | 1,414 | **96.5** | **57.3** |
| CPU (ONNX)    | 128 | 4,416 | 210.1 | 67.2 |
| **Hailo (HEF)** | 128 | 3,992 | **96.2** | **57.9** |
| CPU (ONNX)    | 512 | 8,170 | 112.4 | 61.1 |
| **Hailo (HEF)** | 512 | 7,110 | **90.0** | **57.9** |

CPU is slightly faster in raw throughput (~7–13%), but Hailo holds CPU usage to roughly one core and runs ~10 °C cooler at the smallest batch.

<details>
<summary><strong>Streaming benchmark — <code>run_bench.sh</code></strong></summary>

```bash
bash ids_test/run_bench.sh
RUN_DURATION=120 BATCH=128 FEED_CHUNK=100 FEED_DELAY=1.0 bash ids_test/run_bench.sh
```

Results → `ids_test/bench_results/<ts>/` (`timing_*.csv`, `resources_*.csv`, `report.md`).

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_DIR` | `ids_test/` | Path to the ids_test directory |
| `SOURCE_CSV` | `sample_big.csv` → `sample_labeled.csv` | CSV to feed from |
| `RUN_DURATION` | `90` | Seconds per variant |
| `FEED_CHUNK` | `50` | Rows appended per tick |
| `FEED_DELAY` | `1.0` | Seconds between feed ticks |
| `THRESHOLD` | `0.01` | Detection threshold |
| `BATCH` | `32` | Inference batch size |
| `POLL` | `2` | IPS polling interval (s) |
| `VENV_CPU` | `venv_ids/bin/activate` | venv for ONNX variant |
| `VENV_HAILO` | `venv_hailo_runtime/bin/activate` | venv for Hailo variant |
</details>

<details>
<summary><strong>Burst benchmark — <code>run_burst_bench.sh</code></strong></summary>

Copies a full CSV into `live_sample_bench.csv` atomically, then waits for the IPS to drain it. Measures wall-clock time end-to-end.

```bash
bash ids_test/run_burst_bench.sh
BURST_CSV=ids_test/friday_ddos.csv BATCH=256 bash ids_test/run_burst_bench.sh
```

Results → `ids_test/bench_results/burst_<ts>/`, with `wall_time_cpu.txt` / `wall_time_hailo.txt`.

| Variable | Default | Description |
|----------|---------|-------------|
| `ROOT_DIR` | `ids_test/` | Path to the ids_test directory |
| `BURST_CSV` | `friday_ddos.csv` | Full dataset to burst-inject |
| `BATCH` | `128` | Inference batch size |
| `POLL` | `1` | IPS polling interval (s) |
| `THRESHOLD` | `0.5` | Detection threshold |
| `MAX_WAIT_SEC` | `600` | Per-variant timeout (s) |
</details>

<details>
<summary><strong>Scaling benchmark — <code>run_scaling*.sh</code></strong></summary>

```bash
BATCH_SIZES="32 128 256" RUN_DURATION=60 bash ids_test/run_scaling.sh
BATCH_SIZES="32 128 512" bash ids_test/run_burst_scaling.sh
bash ids_test/run_scaling_v2.sh ids_test/friday_ddos.csv 90
```

`run_scaling_v2.sh` adds absolute artifact paths and a health check 5 s after each process starts — it aborts cleanly and tails `ips.log` if a variant fails to start.
</details>

<details>
<summary><strong>Resource sampler & report generators</strong></summary>

```bash
python ids_test/resource_sampler.py --proc-name ips_realtime_v2.py \
    --variant cpu --out ids_test/resources_cpu.csv --interval 1 --duration 120

python ids_test/gen_report.py        --bench-dir ids_test/bench_results/<ts> --out report.md
python ids_test/gen_report_v2.py     --bench-dir ids_test/bench_results/<ts> --out report.md
python ids_test/gen_scaling_report.py --scaling-dir ids_test/bench_results/scaling_<ts> --out scaling_report.md
```

`gen_report.py` → per-stage latency, throughput (p50/p95/p99), system resources, per-core CPU.
`gen_report_v2.py` → same + PNG figures (throughput, latency dist, per-stage breakdown, CPU/temp over time, efficiency scatter).
`gen_scaling_report.py` → throughput, batch/per-inference/inference-only/preprocess/p95 latency, CPU%, RAM, winner bolded per row.
</details>

---

## Input CSV format

The CSV must contain the same numeric feature columns present during training. Column names are normalized to lowercase with surrounding whitespace stripped, so `Flow Duration`, `flow duration`, and `flow_duration` all resolve to the same feature.

The expected list lives in `ids_test/feature_names.txt` (80 features). The source-IP column is auto-detected from any of: `src ip`, `src_ip`, `source ip`, `source_ip`, `ip src`, `ip_src`. Non-numeric and extra numeric columns are ignored; missing required columns cause a clear error listing the missing names.

**Whitelist** — `ids_test/whitelist.txt`, one IP per line, `#` for comments. `127.0.0.1`, `0.0.0.0`, and `::1` are always whitelisted in code.

---

## A note on the Hailo HEF

The HEF was produced from the **logits** ONNX model via the Hailo Dataflow Compiler. The recommended path is `recompile_hailo.py` (runtime-matched calibration + `opt_level=2`) rather than a random calibration set — calibration quality has a direct, measurable effect on quantization accuracy. The logits variant is used because quantizing through the final sigmoid degrades output; applying sigmoid in Python on the raw logits gives better results. The model loads as `HAILO8`, single context, input `1×1×80` UINT8, output `NC(1)`.

---

## Progress log

| Date | Milestone |
|------|-----------|
| 28.09.2025 | IDS flow integrated into the project; Hailo inference flow pending fix |
| 14.10.2025 | Offline inference working on Raspberry Pi; `ids_realtime.py` initial version |
| 02.04.2026 | IDS and IPS working on CPU/ONNX without Hailo |
| 10.04.2026 | Hailo pipeline runs end-to-end; HEF loading, preprocessing, batch inference confirmed; `iptables` dry-run verified |
| 10.05.2026 | Benchmarking subsystem added (`bench_timing.py`, `resource_sampler.py`, `gen_report.py`, `gen_scaling_report.py`); both IPS variants instrumented; orchestration scripts added |
| 22.05.2026 | **Phase D** — full HEF validation on CICIDS2017 Friday DDoS (225,745 flows): Hailo INT8 F1 = 99.20%, ONNX FP32 F1 = 99.84%, quantization gap −0.64 pp, Hailo/ONNX agreement 99.2% |
| 22.05.2026 | **Phase E** — streaming/scaling benchmarks (batch 32/128/512): Hailo cuts process CPU by up to 71.8% and max temp by 10.5 °C vs ONNX at batch 32; raw HW throughput 56,568 inf/s |
| 23.05.2026 | **Phase F** — IPS reaction demo with real iptables blocking; port-level granularity verified (blocked port dropped while SSH/ICMP kept working) |
| 24.05.2026 | **Phase G** — full end-to-end attack test: real SYN flood (fixed source port) → 14,862 packets → 2 consolidated flows → both detected @ prob 1.000 → attacker IP blocked, 250 packets / 30,105 bytes dropped by the kernel |

---

## Future work

- **Continuous streaming** — move from batch to packet-by-packet processing to cut capture-to-decision latency
- **Multi-class detection** — extend the binary classifier to DDoS / port scan / brute force / … using the CICIDS2017 attack labels
- **In-line bridge mode** — drop malicious traffic before it reaches the target, not just at the destination host
- **Larger models** — exploit the Hailo headroom (56k inf/s vs ~1.4–7k used) for recurrent or transformer-based flow analysis
- **Persistent Hailo context** — keep the context open between batches to trim per-batch activation overhead

---

<p align="center"><sub>Built as an engineering thesis project — Raspberry Pi 5 · Hailo-8 · CICIDS2017</sub></p>
