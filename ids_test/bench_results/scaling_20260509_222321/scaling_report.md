# Scaling Benchmark: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/scaling_20260509_222321`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

Batch sizes testate: 32, 128, 256

**Bold** = câștigătorul la metrica respectivă.

---

### Tabel S1 — Throughput (rows/sec, mai mult = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 818.5 | **1059.4** |
| 128 | 1729.5 | **2666.1** |
| 256 | **3147.3** | 3091.1 |

### Tabel S2 — Latență medie per batch (ms, mai puțin = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 31.09 | **23.99** |
| 128 | 54.21 | **36.69** |
| 256 | **31.26** | 32.66 |

### Tabel S3 — Latență per inferență (ms/rând, mai puțin = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 0.1912 | **0.1258** |
| 128 | 0.1749 | **0.0727** |
| 256 | **0.0379** | 0.0695 |

### Tabel S4 — Timp inferență pură per batch (ms, fără preprocess/log)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 4.865 | **3.197** |
| 128 | 16.396 | **7.107** |
| 256 | **3.724** | 7.017 |

### Tabel S5 — Preprocess per batch (ms — domină total!)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 24.308 | **19.670** |
| 128 | 30.736 | **25.399** |
| 256 | **19.287** | 23.380 |

### Tabel S6 — Latență p95 per batch (ms, predictibilitate)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 60.21 | **33.61** |
| 128 | 158.02 | **64.97** |
| 256 | 99.34 | **45.86** |

### Tabel S7 — CPU mediu (%, mai puțin = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 13.7 | **5.0** |
| 128 | 6.4 | **2.4** |
| 256 | 6.0 | **2.3** |

### Tabel S8 — RAM (MB, mai puțin = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 123.2 | **80.6** |
| 128 | 127.0 | **79.5** |
| 256 | 141.7 | **80.5** |

---

## Cum se citesc tabelele

- **Throughput (S1)**: numărul de rânduri / secundă procesate end-to-end. 
  Include preprocesare + inferență + log. 
  Pentru un IPS, reprezintă debitul maxim de flow-uri pe care îl poate susține.
- **Latență per batch (S2)**: cât durează un batch de la `pd.read_csv` slice 
  până la log scris. Include overhead-ul Hailo (~3 ms / batch context activate).
- **Inferență pură (S4)**: doar timpul pentru `runner.infer()` sau `sess.run()`. 
  La batch mare, costul fix Hailo se amortizează — vezi cum coloana CPU se 
  apropie sau e depășită de Hailo.
- **Preprocess (S5)**: pandas + scaler. **Bottleneck-ul** la batch mic. 
  Indiferent ce backend ML folosești, asta domină timpul total.
- **p95 (S6)**: 95% din batch-uri se termină sub această valoare. 
  Important pentru SLA: arată coada distribuției, nu doar media.
- **CPU & RAM (S7-S8)**: utilizarea procesului. 
  La Hailo, CPU-ul e mai puțin folosit pentru că calculul migrează pe NPU.

## Concluzie scaling

Pentru un MLP binary de dimensiune mică, **Hailo nu câștigă la throughput** 
decât eventual la batch-uri mari. Câștigul real e la **utilizare resurse** 
(CPU, RAM) și **predictibilitate** (p95/p99 mai mici). Pentru un sistem IPS 
unde ML e una din mai multe componente (captură, parsare flow, decizie, 
actuator iptables), valoarea Hailo este eliberarea CPU-ului pentru restul 
pipeline-ului, nu accelerarea ML-ului în sine.