# Benchmark IPS: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/burst_20260509_234540`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

---

### Tabel 1 — Latență per stadiu (ms / batch, valori medii)

| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |
|---|---:|---:|---:|
| read | 0.000 | 0.000 | — |
| preprocess | 18.215 | 21.362 | 1.17 |
| inference | 0.489 | 7.770 | 15.88 |
| postprocess | 0.015 | 0.025 | 1.66 |
| log | 9.206 | 8.959 | 0.97 |
| **TOTAL** | **27.925** | **38.116** | **1.36** |

### Tabel 2 — Throughput end-to-end

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| Total rânduri procesate | 225745 | 225745 |
| Total batch-uri | 1764 | 1764 |
| Throughput (rows/s) | 4582.7 | 3357.5 |
| Latență medie per rând (ms) | 0.0038 | 0.0607 |
| Latență medie per batch (ms) | 27.925 | 38.116 |
| Latență p50 per batch (ms) | 29.824 | 37.720 |
| Latență p95 per batch (ms) | 41.256 | 52.724 |
| Latență p99 per batch (ms) | 49.106 | 61.924 |

### Tabel 3 — Utilizare resurse (steady state)

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| CPU total avg (%) | 75.0 | 37.1 |
| CPU total peak (%) | 98.5 | 50.3 |
| CPU proces avg (%) | 264.1 | 105.4 |
| CPU proces peak (%) | 355.5 | 119.5 |
| RAM (RSS) avg (MB) | 650.3 | 417.5 |
| RAM (RSS) peak (MB) | 1049.8 | 912.2 |
| Temperatura avg (°C) | 64.5 | 59.3 |
| Temperatura peak (°C) | 69.2 | 62.0 |

### Tabel 4 — Distribuție %CPU per core (avg)

| Core | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| core0_pct | 68.0 | 37.5 |
| core1_pct | 73.5 | 39.0 |
| core2_pct | 69.2 | 33.2 |
| core3_pct | 89.1 | 38.9 |

---

## Note interpretare

- **Latență per stadiu**: include overhead-ul activării contextului Hailo per batch 
  (~10-50 ms tipic). Pentru un IPS real-time asta e dezirabil; pentru throughput pur 
  s-ar putea reduce dacă păstrezi contextul activ între batch-uri.
- **CPU proces**: pe varianta CPU(ONNX), procesul saturează 1 core (~100% pe un core 
  înseamnă ~25% pe Pi cu 4 cores). Pe Hailo, calculul e descărcat → CPU stă în repaus.
- **RAM**: similar pe ambele variante (modelul e mic), majoritatea consumului e Python 
  + pandas + onnxruntime/HailoRT.