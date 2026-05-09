# Benchmark IPS: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/burst_20260509_233913`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

---

### Tabel 1 — Latență per stadiu (ms / batch, valori medii)

| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |
|---|---:|---:|---:|
| read | 0.000 | 0.000 | — |
| preprocess | 17.613 | 19.219 | 1.09 |
| inference | 0.272 | 3.675 | 13.52 |
| postprocess | 0.013 | 0.022 | 1.73 |
| log | 2.762 | 2.612 | 0.95 |
| **TOTAL** | **20.660** | **25.528** | **1.24** |

### Tabel 2 — Throughput end-to-end

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| Total rânduri procesate | 225745 | 225745 |
| Total batch-uri | 7055 | 7055 |
| Throughput (rows/s) | 1548.8 | 1253.4 |
| Latență medie per rând (ms) | 0.0085 | 0.1148 |
| Latență medie per batch (ms) | 20.660 | 25.528 |
| Latență p50 per batch (ms) | 19.028 | 24.160 |
| Latență p95 per batch (ms) | 29.772 | 33.444 |
| Latență p99 per batch (ms) | 43.613 | 40.373 |

### Tabel 3 — Utilizare resurse (steady state)

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| CPU total avg (%) | 88.1 | 35.5 |
| CPU total peak (%) | 99.8 | 67.9 |
| CPU proces avg (%) | 312.6 | 101.1 |
| CPU proces peak (%) | 375.3 | 105.7 |
| RAM (RSS) avg (MB) | 620.1 | 398.1 |
| RAM (RSS) peak (MB) | 999.7 | 937.2 |
| Temperatura avg (°C) | 64.3 | 58.4 |
| Temperatura peak (°C) | 70.3 | 62.0 |

### Tabel 4 — Distribuție %CPU per core (avg)

| Core | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| core0_pct | 87.3 | 17.9 |
| core1_pct | 85.4 | 44.4 |
| core2_pct | 87.4 | 42.8 |
| core3_pct | 92.4 | 36.5 |

---

## Note interpretare

- **Latență per stadiu**: include overhead-ul activării contextului Hailo per batch 
  (~10-50 ms tipic). Pentru un IPS real-time asta e dezirabil; pentru throughput pur 
  s-ar putea reduce dacă păstrezi contextul activ între batch-uri.
- **CPU proces**: pe varianta CPU(ONNX), procesul saturează 1 core (~100% pe un core 
  înseamnă ~25% pe Pi cu 4 cores). Pe Hailo, calculul e descărcat → CPU stă în repaus.
- **RAM**: similar pe ambele variante (modelul e mic), majoritatea consumului e Python 
  + pandas + onnxruntime/HailoRT.