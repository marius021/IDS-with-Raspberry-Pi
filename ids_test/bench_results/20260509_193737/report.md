# Benchmark IPS: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/20260509_193737`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

---

### Tabel 1 — Latență per stadiu (ms / batch, valori medii)

| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |
|---|---:|---:|---:|
| read | 0.000 | 0.000 | — |
| preprocess | 16.553 | 19.043 | 1.15 |
| inference | 0.247 | 3.025 | 12.23 |
| postprocess | 0.010 | 0.019 | 1.97 |
| log | 0.797 | 0.702 | 0.88 |
| **TOTAL** | **17.607** | **22.789** | **1.29** |

### Tabel 2 — Throughput end-to-end

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| Total rânduri procesate | 4500 | 4550 |
| Total batch-uri | 178 | 178 |
| Throughput (rows/s) | 1435.8 | 1121.7 |
| Latență medie per rând (ms) | 0.0098 | 0.1183 |
| Latență medie per batch (ms) | 17.607 | 22.789 |
| Latență p50 per batch (ms) | 15.226 | 20.281 |
| Latență p95 per batch (ms) | 25.322 | 32.409 |
| Latență p99 per batch (ms) | 46.823 | 39.948 |

### Tabel 3 — Utilizare resurse (steady state)

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| CPU total avg (%) | 5.5 | 3.6 |
| CPU total peak (%) | 28.8 | 27.7 |
| CPU proces avg (%) | 14.3 | 5.0 |
| CPU proces peak (%) | 40.9 | 15.9 |
| RAM (RSS) avg (MB) | 154.6 | 80.7 |
| RAM (RSS) peak (MB) | 157.2 | 85.2 |
| Temperatura avg (°C) | 47.5 | 47.5 |
| Temperatura peak (°C) | 49.4 | 48.8 |

### Tabel 4 — Distribuție %CPU per core (avg)

| Core | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| core0_pct | 4.9 | 3.0 |
| core1_pct | 5.7 | 2.1 |
| core2_pct | 5.7 | 2.8 |
| core3_pct | 5.5 | 6.4 |

---

## Note interpretare

- **Latență per stadiu**: include overhead-ul activării contextului Hailo per batch 
  (~10-50 ms tipic). Pentru un IPS real-time asta e dezirabil; pentru throughput pur 
  s-ar putea reduce dacă păstrezi contextul activ între batch-uri.
- **CPU proces**: pe varianta CPU(ONNX), procesul saturează 1 core (~100% pe un core 
  înseamnă ~25% pe Pi cu 4 cores). Pe Hailo, calculul e descărcat → CPU stă în repaus.
- **RAM**: similar pe ambele variante (modelul e mic), majoritatea consumului e Python 
  + pandas + onnxruntime/HailoRT.