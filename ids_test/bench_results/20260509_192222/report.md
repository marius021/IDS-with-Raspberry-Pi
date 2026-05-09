# Benchmark IPS: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/20260509_192222`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

---

### Tabel 1 — Latență per stadiu (ms / batch, valori medii)

| Stage | CPU (ONNX) | Hailo (HEF) | Hailo / CPU |
|---|---:|---:|---:|
| read | — | — | — |
| preprocess | — | — | — |
| inference | — | — | — |
| postprocess | — | — | — |
| log | — | — | — |
| **TOTAL** | **—** | **—** | **—** |

### Tabel 2 — Throughput end-to-end

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| Total rânduri procesate | — | — |
| Total batch-uri | — | — |
| Throughput (rows/s) | — | — |
| Latență medie per rând (ms) | — | — |
| Latență medie per batch (ms) | — | — |
| Latență p50 per batch (ms) | — | — |
| Latență p95 per batch (ms) | — | — |
| Latență p99 per batch (ms) | — | — |

### Tabel 3 — Utilizare resurse (steady state)

| Metric | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| CPU total avg (%) | 1.4 | 1.5 |
| CPU total peak (%) | 16.6 | 19.0 |
| CPU proces avg (%) | 0.4 | 0.7 |
| CPU proces peak (%) | 1.0 | 2.0 |
| RAM (RSS) avg (MB) | 155.3 | 75.8 |
| RAM (RSS) peak (MB) | 155.3 | 75.8 |
| Temperatura avg (°C) | 44.9 | 44.7 |
| Temperatura peak (°C) | 47.2 | 46.6 |

### Tabel 4 — Distribuție %CPU per core (avg)

| Core | CPU (ONNX) | Hailo (HEF) |
|---|---:|---:|
| core0_pct | 2.0 | 1.3 |
| core1_pct | 1.5 | 1.3 |
| core2_pct | 0.9 | 1.8 |
| core3_pct | 1.4 | 1.5 |

---

## Note interpretare

- **Latență per stadiu**: include overhead-ul activării contextului Hailo per batch 
  (~10-50 ms tipic). Pentru un IPS real-time asta e dezirabil; pentru throughput pur 
  s-ar putea reduce dacă păstrezi contextul activ între batch-uri.
- **CPU proces**: pe varianta CPU(ONNX), procesul saturează 1 core (~100% pe un core 
  înseamnă ~25% pe Pi cu 4 cores). Pe Hailo, calculul e descărcat → CPU stă în repaus.
- **RAM**: similar pe ambele variante (modelul e mic), majoritatea consumului e Python 
  + pandas + onnxruntime/HailoRT.