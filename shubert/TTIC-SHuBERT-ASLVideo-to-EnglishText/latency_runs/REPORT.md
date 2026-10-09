# Post-cut latency distribution — measured 2026-10-08

Measurement only. No model, weight, threshold, dtype, device or pipeline setting was
changed; `git` is clean at `6e97617` and no tracked file was modified. Every number below
is measured. Nothing is estimated, extrapolated or smoothed. Where a quantity could not be
measured it is left blank and said so.

**Session wall clock: 20:57:01 → 21:41:23 PDT (44 min).** I projected 2–2.5 h; the overlap
passes finished far faster than projected because most of their utterances were refused by
the shipping backlog guard (see §Anomalies 1).

---

## 1. Headline

| | sequential | overlapped |
|---|---|---|
| utterances attempted | 120 | 120 |
| **measurements produced** | **91** | **35** |
| declined by the shipping backlog guard | 29 | 82 |
| failed (CUDA OOM) | 0 | 3 |
| post-cut latency median | **9.33 s** | **56.78 s** |
| mean | 10.23 s | 56.83 s |
| p90 / p95 | 14.05 / 14.50 s | 91.32 / 94.34 s |
| min / max | 5.11 / 19.40 s | 9.52 / 97.61 s |
| fraction ≤ 12 s | 71.4 % | 5.7 % |
| fraction ≤ 24 s | **100 %** | **17.1 %** |
| fraction ≤ 30 s | 100 % | 17.1 % |

Four findings that bear on the paper, each detailed below:

1. **In sequential operation the ByT5 decode dominates, not the perception drain** —
   54.5 % vs 40.6 % of post-cut latency. The project log says the drain dominates.
2. **Decode costs 77.8 ms/step on the live path, not 31.3 ms** (R² 0.94, n=126).
3. **At live pace the shipping configuration refuses roughly 2 of every 3 utterances**, and
   the latency of those it accepts is set by queue depth, not by utterance duration
   (r = −0.008).
4. **The three CUDA OOMs happened with >1 GB available and a 2 MB largest free block.**
   This is direct evidence for the fragmentation hypothesis the project log opened on
   2026-09-20 and had never been able to test.

---

## 2. Definitions used

- **Post-cut latency** = `done_ts − cut_ts`: from the instant the utterance stops being fed
  (replay ends) to the instant the translated string exists. Includes queue wait.
- **Utterance duration** = wall-clock length of the replay (`capture_seconds`), plus
  `retained_frames_after_trim` as the frame count. `duration_s` is the ffprobe container
  duration; the two agree to ~0.03 s.

**Three places the probe defines things differently, as requested:**

1. `live_worker_probe.py`'s own `seconds` starts when the **worker dequeues**, not at the
   cut, so in overlap mode it silently excludes queue wait. Both are logged here:
   `post_cut_latency_s` (your definition) and `worker_s` (the probe's). In overlap the two
   differ by a median of 47.3 s, so this distinction is not cosmetic.
2. **No trim is applied.** Head/tail trimming is driven by live motion scores in v5's
   camera loop, which a replay does not produce, and OpenASL clips carry no dead air.
   `retained_frames_after_trim` therefore equals the retained count. **This measurement
   says nothing about what the trim costs or saves.** On live manual clips the log reports
   the trim removing ~23 % of frames, which would reduce these latencies.
3. **The probe does not enforce v5's retained-frame budget** (`_capture_budget()` and
   `_retain_frame()` live in the camera loop, which the probe does not run). My harness
   calls the real functions, so it is *stricter* than `live_worker_probe.py` and matches
   the shipping app. Prior overlap findings in the log were measured without that guard.

---

## 3. Environment

Full capture in `latency_runs/env.txt`, taken at 20:57:01 immediately before the first
pass. Summary:

- NVIDIA Jetson Orin Nano Developer Kit, L4T R36.4.7, kernel 5.15.148-tegra, aarch64, 6 cores
- **Power mode 0 = 15 W (the maximum on this board).** `jetson_clocks` **not** applied —
  clocks left dynamic, by your decision; `jetson_clocks --show` needs root and is recorded
  as unavailable. Under load CPU sat at 1510 MHz (median across all six passes), so the
  governor was at maximum during the work regardless.
- python 3.10.12, torch 2.11.0 (CUDA available), cv2 4.11.0, numpy 1.24.3,
  transformers 4.30.2, mediapipe 0.10.13
- Shipping settings confirmed effective: beams 4, `BYT5_MAX_LENGTH` 768, bf16 on cuda from
  `checkpoint-11625-bf16`, `FRAME_STRIDE` 2, `PERCEPTION_WORKERS` 2, `PERCEPTION_CHUNK` 30,
  `MAX_LIVE_STREAMS` 2, `MAX_RETAINED_FRAMES` 450, `MIN_AVAILABLE_MB` 350,
  `RECORD_MODE` manual, num_hands 2, DINOv2 fp16 batch 32, `GPU_SERIALIZE` on,
  `PYTORCH_NO_CUDA_MEMORY_CACHING` 1. No env override was set for any run.
- git `6e97617`, working tree clean.
- **Ambient load:** `gnome-software`, `tracker-miner-fs-3` and `update-notifier` were closed
  before the first pass (logged in `run_all.log`). `snapd` and `packagekitd` are root-owned
  and were left running — they are the only non-trivial processes that remained besides the
  1 Hz power-fault witness (`crash_witness.py`, which must stay up) and the desktop shell
  needed for the preview window.
- **One environmental difference from earlier probe runs:** the background camera loop ran
  at **15.0 fps** in all six passes, where the 2026-08-24 probes recorded 29.7–29.8 fps.
  The camera is the same ConferenceCam; the likely cause is UVC frame-rate reduction under
  low room light, which I did not verify. It is constant across both conditions so it does
  not bias the comparison between them, but it means the background CPU load here is lighter
  than in a 30 fps session, so these latencies may be slightly optimistic relative to one.

---

## 4. Clip set and commands

40 clips from the 200-clip OpenASL set, 8 per duration bucket, **selected by ffprobe
duration only** — blind to latency and to translation quality. Seed `20261008`. Pool sizes
were 41/41/30/21/16, so no bucket was short. Written to `latency_runs/clip_set.csv`.

```
python3 latency_runs/select_clips.py                     # -> clip_set.csv
./latency_runs/capture_env.sh                            # -> env.txt
./latency_runs/run_all.sh                                # the six passes + analysis
```

`run_all.sh` runs, for i in 1..3, with seeds 1001/1002/1003:

```
tegrastats --interval 1000 > latency_runs/tegrastats_<cond>_pass<i>.log &
shubert_venv/bin/python3 latency_runs/latency_probe.py \
    --condition <sequential|overlap> --pass-index <i> --seed <seed> \
    --out latency_runs/<cond>.csv --limit 0
```

then `parse_tegrastats.py` and `analysis.py`. Conditions **alternate** (seq p1, ovl p1,
seq p2, …) so session drift is shared between them rather than loaded onto whichever ran
last. One process per pass, each with its own model load, because the probe's own docstring
warns a run leaves the box several hundred MB dirtier than it found it.

Instrumentation is `latency_runs/instrument.py`: timing wrappers on `features.test`,
`model.generate`, and forward hooks on `model.encoder` and
`model.encoder.adapter.signhubert_adapter`. No pipeline file was edited. The 3-pass
determinism check (§7) independently confirms outputs were unaffected.

---

## 5. Sequential condition (n = 91)

### By duration bucket

| bucket | n | mean | median | p90 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| 2–4 s | 21 | 7.65 | 7.44 | 11.75 | 11.87 | 5.11 | 11.88 |
| 4–6 s | 19 | 8.01 | 7.96 | 9.23 | 9.64 | 6.43 | 9.92 |
| 6–9 s | 16 | 10.77 | 10.12 | 13.71 | 13.98 | 9.03 | 14.05 |
| 9–12 s | 17 | 11.89 | 12.62 | 13.89 | 14.07 | 7.83 | 14.39 |
| 12–15 s | 18 | 13.56 | 13.04 | 18.32 | 19.33 | 8.79 | 19.40 |

All-set percentiles: median 9.33, p90 14.05, p95 14.50, p99 19.33, min 5.11, max 19.40.
**p99 at n=91 sits between the two largest samples — it is effectively the max and carries
no information the max does not.**

### Latency vs duration

```
post_cut_latency = 0.554 * duration_s + 5.993     R^2 0.498   Pearson r 0.706
post_cut_latency = 0.0455 * retained_frames + 5.420   R^2 0.607   Pearson r 0.779
```

Frame count predicts better than wall-clock duration, which is expected: perception and
encoder input are both per-frame, and stride-2 frame count is not an exact multiple of
duration across clips of differing source frame rates.

### Stage breakdown (share computed per utterance, then averaged)

| stage | mean share | median share | mean s | median s |
|---|---|---|---|---|
| queue wait | 0.0 % | 0.0 % | 0.00 | 0.00 |
| drain (cut → perception+DINOv2 done) | 40.6 % | 38.5 % | 3.85 | 3.93 |
| SHuBERT encoder | 0.9 % | 0.8 % | 0.09 | 0.08 |
| ByT5 encoder blocks | 2.6 % | 2.4 % | 0.26 | 0.21 |
| **ByT5 decode** | **54.5 %** | **56.0 %** | **5.87** | 5.66 |
| ByT5 stage total | 58.2 % | 59.8 % | 6.25 | 6.03 |
| unattributed | 1.2 % | 1.3 % | 0.13 | 0.13 |

### Five worst utterances

| clip | pass | dur | frames | bytes | steps | latency | wait | drain | shubert | enc | decode | dominated by |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| jsOSzFyx8RA_03 | 3 | 12.71 | 191 | 153 | 155 | 19.40 | 0.0 | 5.3 | 0.1 | 0.4 | 13.4 | decode |
| jsOSzFyx8RA_03 | 1 | 12.71 | 191 | 153 | 155 | 19.32 | 0.0 | 5.2 | 0.1 | 0.4 | 13.3 | decode |
| oP2WdYlaflE_03 | 1 | 13.93 | 209 | 148 | 150 | 17.89 | 0.0 | 5.3 | 0.1 | 0.5 | 11.7 | decode |
| oP2WdYlaflE_03 | 2 | 13.93 | 209 | 148 | 150 | 16.89 | 0.0 | 4.2 | 0.1 | 0.4 | 11.8 | decode |
| uZwKNtHx9FE_01 | 1 | 14.47 | 174 | 125 | 127 | 14.60 | 0.0 | 2.6 | 0.3 | 0.7 | 10.8 | decode |

All five are decode-dominated, and the worst two are the longest *output*, not the longest
input. The split is measured per stage, so this is attribution, not inference.

---

## 6. Overlapped condition (n = 35)

**The headline is the acceptance rate, not the latency.** With a 1.5 s inter-utterance gap
and v5's shipping backlog guard in force, 82 of 120 utterances were refused outright and 3
more died of CUDA OOM: **35 of 120 utterances (29 %) produced a translation.**

Latency: median 56.78, mean 56.83, p90 91.32, p95 94.34, min 9.52, max 97.61 s.
Only 17.1 % landed at or below 24 s.

### Latency is set by queue depth, not by utterance duration

```
post_cut_latency vs duration:  slope -0.053   R^2 0.000   Pearson r -0.008
```

There is **no** relationship with duration. The relationship is with how many utterances
were already in flight at the cut:

| queue depth at cut | n | mean | median | min | max |
|---|---|---|---|---|---|
| 1 | 3 | 13.09 | 13.01 | 10.28 | 15.99 |
| 2 | 4 | 22.52 | 18.76 | 9.52 | 43.03 |
| 3 | 8 | 44.70 | 42.12 | 34.43 | 55.22 |
| 4 | 6 | 58.67 | 58.54 | 43.77 | 75.73 |
| 5 | 7 | 80.59 | 80.23 | 64.48 | 96.34 |
| 6 | 4 | 81.53 | 81.17 | 75.26 | 88.54 |
| 7 | 3 | 86.64 | 82.62 | 79.69 | 97.61 |

At depth 1 the overlapped numbers (median 13.01 s) agree with the sequential distribution.
Everything above that is queueing.

### Stage breakdown

| stage | mean share | median share | mean s | n |
|---|---|---|---|---|
| **queue wait** | **60.9 %** | 68.2 % | 39.48 | 35 |
| drain | 11.2 % | 4.2 % | 2.30 | 18 |
| SHuBERT | 0.4 % | 0.1 % | 0.12 | 35 |
| ByT5 encoder | 1.0 % | 0.5 % | 0.29 | 35 |
| ByT5 decode | 16.2 % | 9.9 % | 6.15 | 35 |
| unattributed | 15.4 % | 1.1 % | 9.52 | 35 |

`drain` has n=18 not 35 because **17 of the 35 were deferred** — over the
`MAX_LIVE_STREAMS=2` cap, perception never started during capture, so there is no drain to
report and it is left blank rather than inferred. For those 17 the perception, crop and
DINOv2 work runs *inside* `process_frames`, a path whose internals I did not instrument;
that is what the large `unattributed` figure is (up to 32.5 s on one utterance). It is
named, not attributed to a stage I did not measure.

---

## 7. Sanity checks (Step 5)

### Determinism — the streamed path is exact; the deferred path is not

- **Sequential, 3 passes: 36 clips were measured in more than one pass. 0 differ.**
  Byte-identical output across passes. This also confirms the instrumentation changed
  nothing.
- **Overlap vs sequential: 28 clips measured in both. 9 differ.**

**All 9 differing clips were DEFERRED in the overlap run and streamed in the sequential
run. Every clip that was streamed in both conditions (13 of 13) was byte-identical.** Five
of the 14 deferred clips matched anyway, so deferral changes the text on 9 of 14 (64 %).

Examples:

| clip | streamed | deferred |
|---|---|---|
| f75G_hZMSHs_04 | "The company will likely start their wedding this weekend." | "The tourist wedding is scheduled for this weekend." |
| ZgvLTS2zJdo_01 | "The first thing we want to do is to open and ensure that some students have access to sign language…" | "The first is to open and provide resources for deaf and hard of hearing people." |
| TXyNxVRRPt8_01 | "We brought them to the Deaf Community Forum to bring their concerns to South Carolina." | "We bring them to the deaf community forum to bring their concerns to South Carolina." |

**Established:** the deferred fallback produces a different translation from the streamed
path on most clips. **Hypothesised, not established:** the cause is that the streamed path
runs two perception workers over contiguous 30-frame chunks — re-detecting at each chunk
boundary, a cost `streaming_perception.py` documents — while the deferred path runs one
detector over every frame in order via `video_holistic`, giving MediaPipe different
temporal tracking state. Confirming that needs a `STREAM_PERCEPTION=0` run on the same
clips, which is a config change I did not make. The log's own chunk sweep (raw BLEU 18.66
at chunk 10 vs 19.54 at chunk 30) is consistent with chunking moving outputs.

### Output size

| | sequential | overlap |
|---|---|---|
| output bytes, mean | 75.5 | 74.7 |
| output bytes, max | 153 | 148 |
| **at or over the 768-byte cap** | **0** | **0** |
| decode steps, mean / max | 77.5 / 155 | 76.7 / 150 |

No output came near the cap. The longest was 153 bytes, 20 % of it.

### Memory and thermal (tegrastats, all six passes — `latency_runs/thermal.txt`)

| | range across the six passes |
|---|---|
| peak RAM used | 7199–7238 MB of 7620 |
| min available (probe sampler) | 221–270 MB |
| peak swap used | 456–956 MB |
| **lfb largest free block, minimum** | **1 MB in every pass** |
| Tj max | 55.6–56.1 °C |
| CPU core MHz, median / max | 1510 / 1510 |
| VDD_IN max | 10237–10498 mW (budget 15 W) |

No thermal throttling signature: Tj peaked at 56.1 °C, median CPU clock was the 1510 MHz
maximum in every pass, and the median Tj *fell* slightly from pass 1 to pass 3 (54.4 →
54.1 °C). tegrastats exposes no throttle flag, so this is the clock-and-temperature record,
not a throttle measurement.

---

## 8. Perception real-time factor (B)

Measured directly, per utterance, from `perception_frames_at_cut / capture_seconds` —
frames actually through MediaPipe when the cut arrived, divided by the replay's length.

| | sequential (n=91) | overlap, streamed only (n=18) |
|---|---|---|
| sustained perception fps during capture, mean | 11.75 | 12.00 |
| median | 12.32 | 11.62 |
| min / max | 4.50 / 14.38 | 7.72 / 13.97 |

**Derivation of the "required" factor from measured quantities.** With `FRAME_STRIDE=2` on
a 30 fps source, retained frames arrive at **15 fps**. Perception finishes inside the
utterance — zero drain — exactly when it sustains 15 fps. So the required speed-up is
`15 / sustained_fps`:

- **mean 1.35×, median 1.22×, min 1.04×, max 3.33×.**
- Against the raw 30 fps camera rate instead: mean 2.70×.
- Aggregate perception CPU seconds per second of utterance: mean 1.64 (this is summed
  across `PERCEPTION_WORKERS=2`, so ~0.82 per worker).

**Neither 3.5× nor 6× reproduces, and I cannot reproduce the 6× derivation.** Perception is
within ~1.3× of keeping up, not 6× short. On the measured data, closing the perception gap
entirely would remove a mean of 3.85 s from a mean 10.23 s sequential latency and leave the
5.87 s decode untouched.

---

## 9. Drift across passes (C) and cold start (D)

**Per-pass median latency:**

| | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| sequential | 9.96 s (n=40) | 9.03 s (n=19) | 9.30 s (n=32) |
| overlap | 60.30 s (n=11) | 48.73 s (n=10) | 71.75 s (n=14) |

Sequential pass 3 is **not** slower than pass 1 (9.30 vs 9.96 s); with no thermal drift and
clocks at maximum throughout, there is no throttling effect to report. The sequential `n`
falls in passes 2 and 3 because of the memory declines (§Anomalies 2) — so those medians are
over different, smaller subsets and are not a clean pass-to-pass comparison. The overlap
medians move with how many utterances were accepted and at what queue depth, not with time.

**Cold start.** Model warmup: **18.3, 18.4, 18.7, 18.7, 23.1, 23.3 s** across the six
passes (mean 20.1 s). It is reported separately and never counted as latency.

First utterance after warmup, per pass, against the steady-state median:

| condition | pass | clip | duration | first-utterance latency |
|---|---|---|---|---|
| sequential | 1 | uZwKNtHx9FE_01 | 14.47 s | 14.60 s |
| sequential | 2 | sP6B4dlSOoc_04 | 5.57 s | 9.13 s |
| sequential | 3 | mNF1kg9azQw_03 | 11.39 s | 10.94 s |
| overlap | 1 | uZwKNtHx9FE_01 | 14.47 s | 15.99 s |
| overlap | 2 | sP6B4dlSOoc_04 | 5.57 s | 10.28 s |
| overlap | 3 | mNF1kg9azQw_03 | 11.39 s | 13.01 s |

Sequential steady state excluding those: n=88, median 9.30 s. The first utterances are not
outliers once their duration is accounted for — each sits inside its own duration bucket's
range. **No measurable first-utterance penalty beyond the warmup itself.**

---

## 10. Decode-length dependence (F)

Both conditions pooled (decode time is a property of the stage, not the condition), n=126:

```
byt5_decode_s vs output_bytes :  77.81 ms/byte   intercept  0.096 s   R^2 0.940   r 0.969
byt5_decode_s vs decode_steps :  77.81 ms/step   intercept -0.060 s   R^2 0.940   r 0.969
naive decode_s / decode_steps :  mean 76.93 ms/step   median 73.22   min 68.08   max 114.00
```

ByT5 is byte-level, so steps ≈ bytes + specials and the two fits coincide. The relationship
is tight (R² 0.94) and the intercept is ~0: **decode time is almost entirely proportional to
output length.** Measured per-step cost is **77.8 ms**, against the project log's 31.3 ms —
see §12.

---

## 11. Anomalies

### 1. At live pace the shipping configuration refuses most utterances

82 of 120 overlapped utterances were declined by v5's own `_capture_budget()`: 66 for
backlog (460–629 retained frames against the 450 limit) and 16 for free memory
(242–322 MB against the 350 MB floor). Declines begin at **order index 8–9 in every pass** —
i.e. after about eight utterances at a 1.5 s gap, the system stops accepting new ones.

This is shipping behaviour, not a harness artifact: on the live path the signer sees
`BUSY - backlog full` and the sentence is dropped. It is the honest answer to "what happens
when a signer does not wait", and it is a better number for the paper than the latency
distribution alone.

### 2. Available memory degrades monotonically within a single process

Each pass is a fresh process and starts with 1.9–2.7 GB available. Within the pass it falls
until the 350 MB floor refuses clips:

| | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| sequential: available at first cut → last cut | 1914 → 835 MB | 2739 → 317 MB | 2138 → 292 MB |
| sequential: first decline at order index | — (none) | 19 | 32 |
| overlap: first decline at order index | 8 | 8 | 9 |

Sequential pass 1 completed all 40 utterances; passes 2 and 3 declined 21 and 8. Peak swap
reached 956 MB. **The mechanism is not identified by this measurement** — candidates are the
CUDA driver not returning freed blocks under `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, MediaPipe
native buffers, or genuine Python-side retention, and separating them was not in scope.

Consequence worth stating in the paper: a long live session will progressively stop
accepting utterances and needs a process restart. The longest real signing session in the
project log is ~9 clips, which is below where this starts to bite.

### 3. The three CUDA OOMs were fragmentation, with >1 GB free

All three failures (`YR7BS6l_95E_05`, `ZgvLTS2zJdo_01`, `sP6B4dlSOoc_05`, overlap pass 2)
landed at 21:23:08. The 1 Hz power-fault witness recorded, across the four samples spanning
that instant:

```
21:23:05  lfb_4mb=0  lfb_max_mb=2  memavail_mb=1092  load=4.00
21:23:06  lfb_4mb=0  lfb_max_mb=2  memavail_mb=1082  load=4.00
21:23:07  lfb_4mb=0  lfb_max_mb=2  memavail_mb=1083  load=4.00
21:23:08  lfb_4mb=7  lfb_max_mb=4  memavail_mb=1027  load=3.92   <- clips released
```

**Zero free 4 MB blocks and a 2 MB largest free block, with 1.03–1.09 GB MemAvailable.**
A CUDA allocation needs contiguous NvMap carveout; there was over a gigabyte free and none
of it contiguous enough. `MIN_AVAILABLE_MB=350` did not fire and could not have — it was
measuring the wrong quantity.

This is the test the 2026-09-20 log entry set up and labelled "NOT YET EXERCISED": low lfb
with healthy memavail. Three independent OOMs show exactly that signature. Alignment is
±1 s (witness samples at 1 Hz against my `done_ts`), but the lfb=0 / lfb_max=2 MB condition
held for at least three consecutive samples before the failures, so the attribution does
not depend on sub-second alignment.

### 4. The deferred path changes the translation

See §7. Nine of 28 clips measured in both conditions produced different text, and all nine
were deferred. Not a latency anomaly, but it is a correctness finding about a fallback path
that exists for memory safety and whose output equivalence appears never to have been
checked.

**Nothing was re-run to make any of these go away.** Every failure, decline and divergence
is in the CSVs with its reason.

---

## 12. Disagreements with the project log

| claim in `01_PROJECT_COMPLETE_REFERENCE.md` | measured here |
|---|---|
| "live post-cut latency **12–24 s**" (§1.3, §11.4, Phase 11) | Sequential: median **9.33 s**, max **19.40 s**; 100 % ≤ 24 s but only 71 % ≤ 12 s. The envelope holds for single utterances but the median sits *below* it. Overlapped: median **56.78 s**, max **97.61 s** — far outside it. The figure describes neither case well. |
| "the drain dominates" (§1.3) | Sequential: decode **54.5 %** vs drain **40.6 %**. Decode dominates. The claim is right only for the overlapped case, where queue wait dominates both (60.9 %). |
| "**31.3 ms** per decode step" (Phase 9) | **77.8 ms/step** (fit, R² 0.94, n=126); naive mean 76.9, min 68.1, max 114.0. **2.5× higher — and now explained, see §15.** The 31.3 ms is a number measured with `PYTORCH_NO_CUDA_MEMORY_CACHING` *unset*, quoted as if it described the shipping configuration, which sets it to 1. |
| "perception needs ~**6×** realtime and gets ~**3.5×**" (Phase 11) | Sustained **11.75 fps** against the **15 fps** that `FRAME_STRIDE=2` requires for a zero drain → **1.35× mean shortfall** (median 1.22×). Against raw 30 fps: 2.70×. **Neither figure reproduces, and I cannot reproduce the 6× derivation from any measured quantity.** |
| "**2.53×** from two perception workers on the live path" (Phase 7 / Step 28) | **Not checked.** Testing it means running `PERCEPTION_WORKERS=1`, a config change I did not make. |
| `MIN_AVAILABLE_MB=350` as the memory guard (§8.2) | Did not fire on any of the three OOMs, which occurred at >1 GB available. It is denominated in a gauge that does not bind. |

The 12–24 s and 31.3 ms entries are the two I would change before submitting. The 6×/3.5×
sentence I would remove unless its derivation can be produced, because I could not
reconstruct it from measurement.

---

---

## 15. Confirming tests (run 2026-10-08/09 at your request)

Two questions this report left open, now answered. Driver: `latency_runs/run_confirm.sh`.
Both needed a config change, which is why they were not in the first pass.

### 15.1 The 77.8 ms/step decode — explained, and one correction to §12

`PERCEPTION_WORKERS=1` would NOT have isolated this: in sequential mode `stream.finish()`
joins the perception and embed threads before ByT5 starts, so perception is idle during
decode either way. The isolating test is the project's own committed microbenchmark.

| cell | 2026-08-24 (`nocache_ab.py`) | 2026-10-08 |
|---|---|---|
| flag unset | 38.84 / 38.45 | **38.92 min / 40.01 median** ms/step |
| flag=0 | — | 37.02 |
| **flag=1 — what ships** | 99.74 / 98.24 | **60.02** |

The flag-unset cell reproduces August almost exactly, which validates the comparison.
**The 31.3 ms in Phase 9 is a flag-unset number quoted as if it were the shipping one.**

`decode_bound.py` at flag=1, beams 4, T=100, varying forced steps:

```
 steps    total   ms/step
    16    1.98s    124.0
    32    2.97s     92.8
    64    6.04s     94.4
   128    8.58s     67.0
```

**Correction to §12.** This report first attributed a residual 1.30x to "live context" by
comparing `nocache_ab`'s min-of-5 (60.02) against the live mean (77.8). That was wrong: the
live 77.8 ms/step sits *inside* the standalone shipping-flag range above, and per-step cost
falls as the step count amortises a fixed per-`generate` overhead. **The whole 2.5x gap is the
flag plus step-count amortisation. There is no live-context penalty.** Note also that the
flag's cost has fallen from 2.56x to ~1.54x since August, so §13.7's rejection of removing it
rests on a number that has moved — the NVML assert is the durable reason.

### 15.2 The output divergence — it is the WORKER COUNT, not the deferred path

Three cells, all sequential at queue depth 0 so queueing and memory pressure cannot confound
the text comparison, 40 clips each. With `PERCEPTION_WORKERS=1`, `add_frame`'s
`(index // PERCEPTION_CHUNK) % 1` is always 0, so one detector sees every frame in order and
there are no chunk boundaries — the same thing `video_holistic` does on the deferred path.

| comparison | shared | differ |
|---|---|---|
| `PERCEPTION_WORKERS=1` (streamed) vs `MAX_LIVE_STREAMS=0` (non-streamed) | 40 | **0** |
| baseline 2 workers vs `PERCEPTION_WORKERS=1` | 40 | **26** |
| baseline 2 workers vs non-streamed | 40 | **26 — the same 26 clips** |
| baseline 2 workers vs control, both 2 workers | 27 | **0** |

**The streamed/deferred distinction is irrelevant; the worker count is everything.** One
detector in order gives byte-identical text through either code path. Splitting frames across
two workers in 30-frame chunks forces a MediaPipe re-detection at each boundary and changes
the translation on **26 of 40 clips (65 %)**. The shipping default is 2 workers.

This is not a quality claim. The committed chunk sweep already measured that axis on 200
clips: raw BLEU **19.54** at chunk 30 / 2 workers against a **1-worker control at 19.51**.
**So 65 % of clips change text while corpus BLEU moves 0.03** — the cleanest in-house
demonstration of the *Beyond BLEU* (arXiv 2609.03734) instrument limit the 2026-09-20 scan
imported, and it belongs in threats-to-validity beside that citation.

One documentation defect falls out of it: `streaming_perception.py`'s docstring calls the
change "a pure scheduling change" whose "landmarks are identical, not approximated". **That is
true only at `PERCEPTION_WORKERS=1`.** What its argument actually establishes is that the
reorder buffer preserves frame ORDER into the embed stage, which it does; it does not preserve
per-frame landmarks, because each worker's detector sees a different subsequence.

### 15.3 The 2.53x two-worker claim does not reproduce

| | measured | post-cut median | drain median | sustained perception fps |
|---|---|---|---|---|
| 1 worker | 40 | **14.38 s** | 6.63 | 7.74 |
| 2 workers (control, run last) | 27 | **9.33 s** | 3.76 | 11.89 |
| 2 workers (baseline, seq pass 1) | 40 | **9.96 s** | 3.84 | 12.80 |

Two workers buy **1.44–1.54x** on post-cut latency, **1.73–1.76x** on the drain alone and
**1.54–1.65x** on sustained perception fps. The last reproduces the notes' *standalone* 1.66x;
the documented **live 2.53x** (Step 28, 4 own-footage clips) does not reproduce on 40 OpenASL
clips. The control declined 13 of 40 to the memory degradation of §11.2, so its n is 27, but
its median matches baseline pass 1 — latency itself did not drift.

**The non-streamed path costs ~3.3x:** all-deferred post-cut median **32.97 s** (mean 30.35,
max 56.81) against 9.96 s. That is what a clip pays when it arrives over the stream cap.

### 15.4 An unplanned finding: the pipeline cannot cold-start offline

The first `defer` attempt died during warmup:

```
torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)
  -> urllib.error.HTTPError: HTTP Error 504: Gateway Time-out     (then SIGABRT)
```

`dinov2_features.py:70` fetches the model definition from GitHub at every startup, and
`torch.hub._get_cache_or_reload` resolves the branch over HTTPS **before** consulting the
cache — so the cached repo at `~/.cache/torch/hub/facebookresearch_dinov2_main` (verified
present at the time of failure) does not help, and neither do local weights with
`pretrained=False`, because it is the architecture that comes over the wire. A single
transient GitHub 504 takes down the live path at launch, and it aborts rather than raising
cleanly. For a contribution about edge deployment this is worth fixing (`source='local'`) or
stating. The run was relaunched and completed; it had produced no measurement before dying, so
nothing was discarded.

## 13. What I could not measure, and why

- **Live-camera utterances** — dropped at your instruction.
- **The trim's effect** on latency or frame count: not reproducible on replayed clips (§2.3).
- **A throttle flag**: tegrastats does not expose one. Reported the clock, temperature and
  power record instead.
- **`byt5_encoder_s` / `shubert_s` GPU tails**: the hooks time CPU-side wall at module
  boundaries with no mid-`generate` `cuda.synchronize()`, because inserting one would itself
  change execution timing. For decode this is the right instrument (the log measured it as
  ~100 % CPU dispatch with a 0.01 ms GPU tail); for the SHuBERT/encoder split it may
  under-attribute GPU work landing after the boundary. Both are small (0.09 s and 0.26 s
  mean) so this does not affect any conclusion here.
- **Per-stage internals of the deferred path**: perception, crop and DINOv2 run inside
  `process_frames` there, which I did not instrument. Reported as `unattributed` rather
  than split by guess (§6).
- **The cause of the memory degradation** (§11.2): characterised, not diagnosed. Still open.
- ~~The cause of the deferred-path divergence~~ — **now measured, see §15.2: it is the worker
  count, not the deferred path.**

## 14. Files

```
latency_runs/clip_set.csv              the 40 clips, with measured durations
latency_runs/env.txt                   environment of record, 20:57:01
latency_runs/sequential.csv            120 rows, 38 columns
latency_runs/overlap.csv               120 rows, 38 columns
latency_runs/analysis.txt              full analysis output
latency_runs/latency_vs_duration.png   scatter + fits, both conditions
latency_runs/thermal.txt               per-pass tegrastats summary
latency_runs/tegrastats_*.log          raw 1 Hz tegrastats, 6 files
latency_runs/run_*_pass*.log           per-pass stdout
latency_runs/run_all.log               driver log
latency_runs/{select_clips,latency_probe,instrument,analysis,parse_tegrastats}.py
latency_runs/{capture_env,run_all}.sh
```
