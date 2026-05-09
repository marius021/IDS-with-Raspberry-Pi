# Benchmark IPS: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/20260509_222812`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

---

### Tabel 1 — Latență per stadiu (ms / batch, valori medii)

| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |
|---|---:|---:|---:|
| read | 0.000 | 0.000 | — |
| preprocess | 19.287 | 23.380 | 1.21 |
| inference | 3.724 | 7.017 | 1.88 |
| postprocess | 0.015 | 0.021 | 1.33 |
| log | 8.235 | 2.246 | 0.27 |
| **TOTAL** | **31.261** | **32.664** | **1.04** |

### Tabel 2 — Throughput end-to-end

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| Total rânduri procesate | 3050 | 3130 |
| Total batch-uri | 31 | 31 |
| Throughput (rows/s) | 3147.3 | 3091.1 |
| Latență medie per rând (ms) | 0.0379 | 0.0695 |
| Latență medie per batch (ms) | 31.261 | 32.664 |
| Latență p50 per batch (ms) | 23.225 | 34.437 |
| Latență p95 per batch (ms) | 99.341 | 45.862 |
| Latență p99 per batch (ms) | 150.723 | 56.790 |

### Tabel 3 — Utilizare resurse (steady state)

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| CPU total avg (%) | 8.8 | 3.2 |
| CPU total peak (%) | 50.1 | 19.3 |
| CPU proces avg (%) | 6.0 | 2.3 |
| CPU proces peak (%) | 16.9 | 7.0 |
| RAM (RSS) avg (MB) | 141.7 | 80.5 |
| RAM (RSS) peak (MB) | 148.7 | 84.2 |
| Temperatura avg (°C) | 48.2 | 46.9 |
| Temperatura peak (°C) | 51.0 | 48.8 |

### Tabel 4 — Distribuție %CPU per core (avg)

| Core | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| core0_pct | 8.6 | 2.1 |
| core1_pct | 8.4 | 3.8 |
| core2_pct | 8.5 | 2.9 |
| core3_pct | 9.7 | 4.0 |

---

## Note interpretare

- **Latență per stadiu**: include overhead-ul activării contextului Hailo per batch 
  (~10-50 ms tipic). Pentru un IPS real-time asta e dezirabil; pentru throughput pur 
  s-ar putea reduce dacă păstrezi contextul activ între batch-uri.
- **CPU proces**: pe varianta CPU(ONNX), procesul saturează 1 core (~100% pe un core 
  înseamnă ~25% pe Pi cu 4 cores). Pe Hailo, calculul e descărcat → CPU stă în repaus.
- **RAM**: similar pe ambele variante (modelul e mic), majoritatea consumului e Python 
  + pandas + onnxruntime/HailoRT.