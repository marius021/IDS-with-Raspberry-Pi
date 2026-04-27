-- work in progress project --

===== UPDATE 28.09.2025 =====
* IDS flow integrated in docker project for Hailo Module ( CHECKED )
* BUG FOR "infer" -> needs to be fixed (????)
* TO DO: Branch for Dell and Branch for Raspberry Pi (NOT DONE YET)

===== UPDATE 14.10.2025 =====
* INFERENCE RUNNED ON RPI - SUCCESS! - PREDS WERE WROTE IN /home/maurice/ids/preds.csv ( to test in the future )
* WORK IN PROGRESS:
  **Runtime script – ids_realtime.py
      -Monitors a folder (or a single CSV that keeps growing).
      -Loads feature_names.npy, scaler.joblib, ONNX model.
      -Cleans up numbers (inf/NaN/outliers), aligns columns exactly as in training.
      -Runs inference in small batches, writes an alert log.

===== UPDATE 2.04.2026 ====
* IPS and IDS WORKS, WITHOUT HAILO
*WORK IN PROGRESS:
  - Hailo implementation and test
  - Simulate on larger CSV
 
===== UPDATE 10.04.2026 =====
What works
* Raspberry + Hailo pipeline runs without errors
* HEF loads, preprocessing is correct, inference processes all batches
* Input shape Hailo: (1, 1, 80), output: ids_mlp_binary_logits/fc1
* Attacks detected at --threshold 0.005
* IPS with iptables blocking working in --dry-run
