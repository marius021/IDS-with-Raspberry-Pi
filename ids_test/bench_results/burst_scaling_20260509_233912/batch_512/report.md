# Benchmark IPS: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/burst_20260509_234829`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

---

### Tabel 1 — Latență per stadiu (ms / batch, valori medii)

| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |
|---|---:|---:|---:|
| read | 0.000 | 0.000 | — |
| preprocess | 20.840 | 25.790 | 1.24 |
| inference | 1.280 | 21.725 | 16.98 |
| postprocess | 0.025 | 0.026 | 1.07 |
| log | 34.859 | 32.936 | 0.94 |
| **TOTAL** | **57.003** | **80.478** | **1.41** |

### Tabel 2 — Throughput end-to-end

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| Total rânduri procesate | 225745 | 225745 |
| Total batch-uri | 441 | 441 |
| Throughput (rows/s) | 8980.1 | 6360.7 |
| Latență medie per rând (ms) | 0.0025 | 0.0424 |
| Latență medie per batch (ms) | 57.003 | 80.478 |
| Latență p50 per batch (ms) | 73.572 | 89.794 |
| Latență p95 per batch (ms) | 91.135 | 121.490 |
| Latență p99 per batch (ms) | 115.292 | 138.373 |

### Tabel 3 — Utilizare resurse (steady state)

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| CPU total avg (%) | 51.4 | 36.4 |
| CPU total peak (%) | 92.5 | 49.2 |
| CPU proces avg (%) | 170.9 | 109.7 |
| CPU proces peak (%) | 320.7 | 142.0 |
| RAM (RSS) avg (MB) | 688.8 | 436.4 |
| RAM (RSS) peak (MB) | 1048.6 | 916.9 |
| Temperatura avg (°C) | 60.8 | 58.9 |
| Temperatura peak (°C) | 65.3 | 60.9 |

### Tabel 4 — Distribuție %CPU per core (avg)

| Core | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| core0_pct | 39.3 | 36.0 |
| core1_pct | 38.3 | 24.4 |
| core2_pct | 43.8 | 39.2 |
| core3_pct | 83.9 | 46.5 |

---

## Note interpretare

- **Latență per stadiu**: include overhead-ul activării contextului Hailo per batch 
  (~10-50 ms tipic). Pentru un IPS real-time asta e dezirabil; pentru throughput pur 
  s-ar putea reduce dacă păstrezi contextul activ între batch-uri.
- **CPU proces**: pe varianta CPU(ONNX), procesul saturează 1 core (~100% pe un core 
  înseamnă ~25% pe Pi cu 4 cores). Pe Hailo, calculul e descărcat → CPU stă în repaus.
- **RAM**: similar pe ambele variante (modelul e mic), majoritatea consumului e Python 
  + pandas + onnxruntime/HailoRT.