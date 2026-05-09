# Benchmark IPS: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/20260509_222547`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

---

### Tabel 1 — Latență per stadiu (ms / batch, valori medii)

| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |
|---|---:|---:|---:|
| read | 0.000 | 0.000 | — |
| preprocess | 30.736 | 25.399 | 0.83 |
| inference | 16.396 | 7.107 | 0.43 |
| postprocess | 0.016 | 0.026 | 1.56 |
| log | 7.058 | 4.156 | 0.59 |
| **TOTAL** | **54.206** | **36.688** | **0.68** |

### Tabel 2 — Throughput end-to-end

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| Total rânduri procesate | 3000 | 3130 |
| Total batch-uri | 32 | 32 |
| Throughput (rows/s) | 1729.5 | 2666.1 |
| Latență medie per rând (ms) | 0.1749 | 0.0727 |
| Latență medie per batch (ms) | 54.206 | 36.688 |
| Latență p50 per batch (ms) | 24.307 | 34.211 |
| Latență p95 per batch (ms) | 158.023 | 64.971 |
| Latență p99 per batch (ms) | 548.935 | 101.513 |

### Tabel 3 — Utilizare resurse (steady state)

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| CPU total avg (%) | 7.2 | 8.5 |
| CPU total peak (%) | 26.1 | 36.6 |
| CPU proces avg (%) | 6.4 | 2.4 |
| CPU proces peak (%) | 19.9 | 9.0 |
| RAM (RSS) avg (MB) | 127.0 | 79.5 |
| RAM (RSS) peak (MB) | 133.4 | 82.2 |
| Temperatura avg (°C) | 47.6 | 48.0 |
| Temperatura peak (°C) | 49.9 | 50.5 |

### Tabel 4 — Distribuție %CPU per core (avg)

| Core | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| core0_pct | 6.9 | 6.8 |
| core1_pct | 6.6 | 10.6 |
| core2_pct | 8.0 | 7.7 |
| core3_pct | 7.4 | 8.6 |

---

## Note interpretare

- **Latență per stadiu**: include overhead-ul activării contextului Hailo per batch 
  (~10-50 ms tipic). Pentru un IPS real-time asta e dezirabil; pentru throughput pur 
  s-ar putea reduce dacă păstrezi contextul activ între batch-uri.
- **CPU proces**: pe varianta CPU(ONNX), procesul saturează 1 core (~100% pe un core 
  înseamnă ~25% pe Pi cu 4 cores). Pe Hailo, calculul e descărcat → CPU stă în repaus.
- **RAM**: similar pe ambele variante (modelul e mic), majoritatea consumului e Python 
  + pandas + onnxruntime/HailoRT.