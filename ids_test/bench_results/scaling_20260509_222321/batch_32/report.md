# Benchmark IPS: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/20260509_222321`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

---

### Tabel 1 — Latență per stadiu (ms / batch, valori medii)

| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |
|---|---:|---:|---:|
| read | 0.000 | 0.000 | — |
| preprocess | 24.308 | 19.670 | 0.81 |
| inference | 4.865 | 3.197 | 0.66 |
| postprocess | 0.012 | 0.020 | 1.66 |
| log | 1.903 | 1.099 | 0.58 |
| **TOTAL** | **31.088** | **23.986** | **0.77** |

### Tabel 2 — Throughput end-to-end

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| Total rânduri procesate | 2850 | 3100 |
| Total batch-uri | 112 | 122 |
| Throughput (rows/s) | 818.5 | 1059.4 |
| Latență medie per rând (ms) | 0.1912 | 0.1258 |
| Latență medie per batch (ms) | 31.088 | 23.986 |
| Latență p50 per batch (ms) | 16.627 | 21.063 |
| Latență p95 per batch (ms) | 60.207 | 33.612 |
| Latență p99 per batch (ms) | 153.602 | 45.428 |

### Tabel 3 — Utilizare resurse (steady state)

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| CPU total avg (%) | 8.7 | 5.9 |
| CPU total peak (%) | 40.9 | 17.8 |
| CPU proces avg (%) | 13.7 | 5.0 |
| CPU proces peak (%) | 45.8 | 13.0 |
| RAM (RSS) avg (MB) | 123.2 | 80.6 |
| RAM (RSS) peak (MB) | 128.6 | 82.4 |
| Temperatura avg (°C) | 48.0 | 47.3 |
| Temperatura peak (°C) | 51.0 | 49.4 |

### Tabel 4 — Distribuție %CPU per core (avg)

| Core | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| core0_pct | 8.3 | 7.8 |
| core1_pct | 8.7 | 5.6 |
| core2_pct | 8.8 | 5.9 |
| core3_pct | 9.0 | 4.3 |

---

## Note interpretare

- **Latență per stadiu**: include overhead-ul activării contextului Hailo per batch 
  (~10-50 ms tipic). Pentru un IPS real-time asta e dezirabil; pentru throughput pur 
  s-ar putea reduce dacă păstrezi contextul activ între batch-uri.
- **CPU proces**: pe varianta CPU(ONNX), procesul saturează 1 core (~100% pe un core 
  înseamnă ~25% pe Pi cu 4 cores). Pe Hailo, calculul e descărcat → CPU stă în repaus.
- **RAM**: similar pe ambele variante (modelul e mic), majoritatea consumului e Python 
  + pandas + onnxruntime/HailoRT.