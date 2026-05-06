-- work in progress project --

# IDS with Raspberry Pi

This repository currently contains the training, inference, IDS, IPS, and Hailo integration work completed so far for the project.

## Progress log

### 28.09.2025
- IDS flow integrated into the Docker project for the Hailo module.
- Known issue: the `infer` flow still needs to be fixed.
- Planned split into a Dell branch and a Raspberry Pi branch is not done yet.

### 14.10.2025
- Inference ran successfully on Raspberry Pi.
- Predictions were written to `/home/maurice/ids/preds.csv`.
- Runtime work in progress around `ids_realtime.py`:
  - monitors a growing CSV input
  - loads `feature_names.npy`, `scaler.joblib`, and the ONNX model
  - cleans `inf` / `NaN` / outlier values
  - aligns columns exactly with the training feature order
  - runs inference in batches
  - writes an alert log

### 02.04.2026
- IDS and IPS work without Hailo.
- Work still in progress:
  - Hailo implementation and testing
  - simulation on larger CSV files

### 10.04.2026
- Raspberry Pi + Hailo pipeline runs without errors.
- HEF loading, preprocessing, and batch inference are working.
- Observed Hailo input shape: `(1, 1, 80)`.
- Observed output stream: `ids_mlp_binary_logits/fc1`.
- Attacks were detected with `--threshold 0.005`.
- IPS `iptables` blocking works in `--dry-run` mode.

## Implemented so far

### 1. Training and artifact generation
- A full CSV preprocessing and training pipeline exists in `Debug Scripts/csv-preprocessing.py`.
- The pipeline loads CICIDS-style flow CSV files with robust encoding fallback and automatic label-column detection.
- Headers are normalized, timestamps are detected or synthesized, numeric columns are cleaned, and rows are sorted in time order.
- A binary IDS model is trained for `BENIGN` vs `ATTACK`.
- A multiclass IDS model is also trained for attack-type classification.
- Training uses PyTorch MLP models and exports ONNX models for deployment.
- The training/export flow produces artifacts such as:
  - `ids_mlp_binary.onnx`
  - `ids_mlp_multiclass.onnx`
  - `scaler.joblib`
  - `feature_names.npy`
  - `classes.npy`
  - `label_encoder.joblib`
- Artifacts currently checked into this repository include:
  - `ids_test/ids_mlp_binary.onnx`
  - `ids_test/scaler.joblib`
  - `ids_test/feature_names.npy`
  - `classes.npy`
  - `label_encoder.joblib`

### 2. Offline inference
- `ids_inference.py` performs offline inference from CSV input to CSV output.
- It supports both binary and multiclass ONNX models.
- It robustly reads CSV files with multiple encoding fallbacks.
- It normalizes column names and checks feature compatibility against the saved training feature list.
- It cleans invalid numeric values, clips extreme outliers, fills missing values, applies the saved scaler, and runs ONNX Runtime inference.
- For binary inference it writes `pred_attack_proba` and `pred_label`.
- For multiclass inference it writes `pred_class_idx` and `pred_class`.

### 3. Real-time IDS runtime on Raspberry Pi / CPU
- A real-time IDS runtime is implemented in:
  - `Raspi Files/ids_realtime.py`
  - `ids_test/ids_realtime.py`
- These scripts monitor a growing CSV file and process only the newly appended rows.
- They apply the same feature alignment and preprocessing used during training.
- They run ONNX inference in batches and append alerts as JSON lines in a log file.
- The test/runtime variant adds configurable threshold, alert log path, debug mode, and handling for recreated or truncated input files.

### 4. Real-time IPS runtime on CPU / ONNX
- A CPU-based IPS flow is implemented in:
  - `ids_test/ips_realtime_v1.py`
  - `ids_test/ips_realtime_v2.py`
- These scripts extend the IDS flow by taking response actions after attack detection.
- They detect the source IP column automatically from common column-name variants.
- They support whitelisting through `whitelist.txt` or another text file passed from CLI.
- They write both alert logs and IPS action logs.
- They support `--dry-run` for safe testing without actually applying firewall rules.
- They can block malicious IPs with `iptables`.
- `ips_realtime_v2.py` also avoids repeated blocking/logging for the same IP during the same run by keeping a seen cache.

### 5. Hailo acceleration path
- A Hailo-based runtime is implemented in `ids_test/ips_hailo.py`.
- This runtime removes the dependency on scikit-learn at deployment time by using exported scaler parameters from `scaler_params.npz`.
- It loads feature names from multiple supported formats:
  - `.txt`
  - `.json`
  - `.csv`
  - `.npy`
- It configures the Hailo device, prepares float32 input tensors, runs batch inference, and applies the same alert/action flow as the CPU IPS version.
- It supports:
  - growing CSV monitoring
  - alert and action logs
  - whitelist loading
  - `--dry-run`
  - debug diagnostics for stream names, stream shapes, and raw outputs
- Hailo conversion/build artifacts already present in the repository include:
  - `ids_mlp_binary_logits.onnx`
  - `ids_mlp_binary.hef`
  - `ids_mlp_binary_logits.hef`
  - `ids_mlp_binary.har`
  - `ids_mlp_binary_compiled.har`
  - `ids_mlp_binary_quant.har`
  - `ids_mlp_binary_prob.har`
  - `ids_mlp_binary_prob_quant.har`
  - `ids_mlp_binary_logits.har`
  - `ids_mlp_binary_logits_compiled.har`
  - `ids_mlp_binary_logits_quant.har`
  - `opt_script.alls`

### 6. Helper and validation scripts
- `ids_test/export_scaler.py` exports the scaler parameters from `scaler.joblib` into `scaler_params.npz` for the Hailo runtime.
- `ids_test/feature_names_generate.py` converts `feature_names.npy` into `feature_names.txt`.
- `ids_test/make_logits_onnx.py` rewrites the ONNX output so the logits tensor becomes the official exported output.
- `ids_test/feed_csv.py` simulates a live stream by appending chunks from a larger CSV into a growing CSV file.
- `Debug Scripts/check_artifacts.py` validates that the model, scaler, feature names, and sample CSV are aligned and runnable together.
- `Debug Scripts/eval_quick.py` prints a quick classification report from prediction output.
- `Debug Scripts/multiclass_test.py` is used to inspect sample CSV contents and column types.

### 7. Environment and deployment tooling
- `requirements.txt` contains the runtime dependencies for ONNX-based inference.
- `Dell Files/requirements.txt` contains the broader dependency set used for model training and development.
- `Dell Files/Dockerfile` provides a Python container image for the project.
- `Dell Files/docker-compose.yaml` defines separate training and inference services.
- `Dell Files/tasks.json` and `Dell Files/launch.json` provide VS Code Docker run/debug configurations.

## Current status summary

- IDS on CPU / ONNX is implemented.
- IPS on CPU / ONNX is implemented.
- Raspberry Pi inference has already been run successfully.
- Hailo runtime code, conversion helpers, and generated Hailo artifacts are already in the repository.
- End-to-end Hailo validation and larger-scale simulation still appear to be the main remaining steps.
