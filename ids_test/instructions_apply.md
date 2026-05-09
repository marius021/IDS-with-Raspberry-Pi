# Aplicare instrumentare timing — ips_realtime_v2.py & ips_hailo.py

Toată instrumentarea e opt-in: dacă **NU** setezi `BENCH=1` sau `BENCH_OUT`, scripturile rulează identic ca înainte, fără overhead.

## 1. Pune `bench_timing.py` lângă scripturi

Copiază `bench_timing.py` în `/home/maurice/Desktop/IDS-with-Raspberry-Pi/ids_test/`.

## 2. Patch pentru `ips_realtime_v2.py`

### a) În top-level (după importurile existente, ~linia 12)

Adaugă:

```python
from bench_timing import StageTimer, maybe_writer
```

### b) Înainte de `while True:` din `main()` (~linia 215)

Adaugă:

```python
    bench = maybe_writer(default_path="timing_cpu.csv", default_variant="cpu")
    timer = StageTimer()
    batch_idx = 0
```

### c) Înlocuiește bucla interioară `for start in range(0, len(df_new), args.batch):` (~linia 230)

Înlocuiește **întreg blocul** `for start in range(...)` cu:

```python
                    for start in range(0, len(df_new), args.batch):
                        chunk = df_new.iloc[start:start + args.batch].copy()

                        if args.debug:
                            print(f"[DEBUG] procesez chunk {start}:{start + len(chunk)}")

                        timer.reset()

                        timer.start("preprocess")
                        Xs = build_feature_matrix(chunk, scaler, feats_p)
                        timer.stop("preprocess")

                        timer.start("inference")
                        prob, pred = run_batch(sess, input_name, Xs, args.threshold)
                        timer.stop("inference")

                        timer.start("postprocess")
                        # sigmoid + threshold sunt deja în run_batch; aici doar marcăm
                        n_attacks = int(pred.sum())
                        timer.stop("postprocess")

                        if args.debug:
                            print(
                                f"[DEBUG] batch rows={len(chunk)} | "
                                f"max_prob={float(prob.max()):.6f} | attacks={n_attacks}"
                            )

                        timer.start("log")
                        append_alerts(chunk, prob, pred, alert_log)
                        append_actions(chunk, prob, pred, src_ip_col, action_log,
                                       whitelist, args.dry_run, seen_cache, args.debug)
                        timer.stop("log")

                        if bench:
                            bench.write_batch(batch_idx, len(chunk), timer)
                            batch_idx += 1
```

### d) Pentru timing-ul `read` (citirea CSV)

Adaugă cronometrarea în jurul `pd.read_csv` din loop-ul principal. Localizează linia `df_all = pd.read_csv(input_csv, low_memory=False)` și înconjoar-o:

```python
                # Doar dacă vrei să măsori și citirea CSV (opțional)
                # timer la nivel de poll, nu de batch — măsurători separate
```

(De fapt, citirea CSV se face o dată per poll, nu per batch — deci o lăsăm ne-instrumentată sau o adăugăm la primul batch din poll. Pentru simplitate, las read_ms = 0 în CSV și menționăm separat citirea în raport.)

## 3. Patch pentru `ips_hailo.py`

Identic cu pașii (a), (b), (c) de mai sus, doar cu trei diferențe:

- La (b), folosește `default_variant="hailo"`:
  ```python
  bench = maybe_writer(default_path="timing_hailo.csv", default_variant="hailo")
  ```

- La (c), `build_feature_matrix` ia 3 argumente în varianta Hailo:
  ```python
  Xs = build_feature_matrix(chunk, scaler, wanted_features)
  ```

- Inferența folosește `run_batch_hailo`:
  ```python
  timer.start("inference")
  prob, pred = run_batch_hailo(runner, Xs, args.threshold)
  timer.stop("inference")
  ```

## 4. Verificare rapidă (înainte de benchmark mare)

```bash
# Rulare normală (timing inactiv) — verifică că nu strică nimic
python ips_realtime_v2.py --input live_sample.csv ... --debug

# Rulare cu timing activ
BENCH=1 BENCH_OUT=timing_cpu.csv BENCH_VARIANT=cpu python ips_realtime_v2.py ...
# Ar trebui să apară mesajul: [BENCH] Timing activat. Scriu în: timing_cpu.csv (variant=cpu)
# După câteva batch-uri, verifică:
head -5 timing_cpu.csv
```

CSV-ul ar trebui să arate cam așa:

```
timestamp,variant,batch_idx,n_rows,t_read_ms,t_preprocess_ms,t_inference_ms,t_postprocess_ms,t_log_ms,t_total_ms
1715268000.123,cpu,0,32,0.0,4.5,1.2,0.1,0.8,6.6
1715268000.130,cpu,1,32,0.0,4.6,1.1,0.1,0.7,6.5
```
