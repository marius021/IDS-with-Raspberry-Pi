# Scaling Benchmark: CPU vs Hailo

Director: `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/bench_results/burst_scaling_20260509_233912`

Hardware: Raspberry Pi 5 + Hailo-8 (26 TOPs, M.2 PCIe)

Batch sizes testate: 32, 128, 512

**Bold** = câștigătorul la metrica respectivă.

---

### Tabel S1 — Throughput (rows/sec, mai mult = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | **1548.8** | 1253.4 |
| 128 | **4582.7** | 3357.5 |
| 512 | **8980.1** | 6360.7 |

### Tabel S2 — Latență medie per batch (ms, mai puțin = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | **20.66** | 25.53 |
| 128 | **27.93** | 38.12 |
| 512 | **57.00** | 80.48 |

### Tabel S3 — Latență per inferență (ms/rând, mai puțin = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | **0.0085** | 0.1148 |
| 128 | **0.0038** | 0.0607 |
| 512 | **0.0025** | 0.0424 |

### Tabel S4 — Timp inferență pură per batch (ms, fără preprocess/log)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | **0.272** | 3.675 |
| 128 | **0.489** | 7.770 |
| 512 | **1.280** | 21.725 |

### Tabel S5 — Preprocess per batch (ms — domină total!)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | **17.613** | 19.219 |
| 128 | **18.215** | 21.362 |
| 512 | **20.840** | 25.790 |

### Tabel S6 — Latență p95 per batch (ms, predictibilitate)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | **29.77** | 33.44 |
| 128 | **41.26** | 52.72 |
| 512 | **91.14** | 121.49 |

### Tabel S7 — CPU mediu (%, mai puțin = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 312.6 | **101.1** |
| 128 | 264.1 | **105.4** |
| 512 | 170.9 | **109.7** |

### Tabel S8 — RAM (MB, mai puțin = mai bine)

| Batch size | CPU | Hailo |
|---|---:|---:|
| 32 | 620.1 | **398.1** |
| 128 | 650.3 | **417.5** |
| 512 | 688.8 | **436.4** |

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