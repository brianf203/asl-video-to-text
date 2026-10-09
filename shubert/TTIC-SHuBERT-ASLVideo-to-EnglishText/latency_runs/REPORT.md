# Post-cut latency on an edge SHuBERT pipeline — complete measurement report

**Measured 2026-10-08 / 2026-10-09 on an NVIDIA Jetson Orin Nano 8 GB.**
**Self-contained handoff document.** Everything needed to update the paper is in this file,
including the full per-utterance data in Appendix A and the raw CSVs in Appendix B. No other
file is required.

**What this is.** A commissioned measurement of per-utterance post-cut latency for the paper,
which had been carrying "12–24 s post-cut" with no distribution behind it. It grew into a
correction of five numbers the project's own documentation had been repeating, plus three
findings that were not being looked for.

**Provenance.** Measurement only. No model, weight, threshold, dtype, device or pipeline
setting was changed for any run reported here. The repository was clean at commit `6e97617`
throughout the measurement; the harness and results were committed afterwards as `952aff1`,
and a documentation-only docstring correction as `e3b358e` (proved documentation-only by
comparing the Python AST with docstrings stripped against the previous commit). Every number
below is measured. Nothing is estimated, extrapolated, smoothed, or carried over from earlier
sessions unless explicitly labelled as such.

**Instrument caveat that applies everywhere.** Latency was measured by replaying recorded
OpenASL clips through the real live worker at true wall-clock camera rate, with the camera
open and the preview window up — the configuration `live_worker_probe.py` exists to provide.
It is *not* a live signing session. The two differences that matter are stated in §3 and
repeated wherever they affect a number: **no head/tail trim is applied**, and the background
camera loop ran at 15 fps rather than 30.


## Contents

  - [1. For the paper: what to change, and what to say instead](#1-for-the-paper-what-to-change-and-what-to-say-instead)
  - [2. Definitions used](#2-definitions-used)
  - [3. Environment](#3-environment)
  - [4. Method](#4-method)
  - [5. Results — sequential condition](#5-results-—-sequential-condition)
  - [6. Results — overlapped condition](#6-results-—-overlapped-condition)
  - [7. Where the time goes — per-stage breakdown (request A)](#7-where-the-time-goes-—-per-stage-breakdown-request-a)
  - [8. Perception real-time factor (request B)](#8-perception-real-time-factor-request-b)
  - [9. Thermal state and drift across passes (request C)](#9-thermal-state-and-drift-across-passes-request-c)
  - [10. Cold start and first-utterance cost (request D)](#10-cold-start-and-first-utterance-cost-request-d)
  - [11. Live-camera utterances (request E) — not performed](#11-live-camera-utterances-request-e-—-not-performed)
  - [12. Decode-length dependence (request F)](#12-decode-length-dependence-request-f)
  - [13. Determinism, and the finding it uncovered (request: Step 5)](#13-determinism-and-the-finding-it-uncovered-request-step-5)
  - [14. Confirming tests (run 2026-10-08/09)](#14-confirming-tests-run-2026-10-0809)
  - [15. Anomalies — every failure, decline and divergence](#15-anomalies-—-every-failure-decline-and-divergence)
  - [16. Disagreements with the project log (request G)](#16-disagreements-with-the-project-log-request-g)
  - [17. What could not be measured, and why](#17-what-could-not-be-measured-and-why)
  - [18. Everything else done in this session, in order](#18-everything-else-done-in-this-session-in-order)
  - [19. Open items this report leaves behind](#19-open-items-this-report-leaves-behind)
  - [20. Artifacts](#20-artifacts)
- [Appendix A — every utterance, readable](#appendix-a-—-every-utterance-readable)
  - [A.1  Sequential — 120 attempted utterances](#a1-sequential-—-120-attempted-utterances)
  - [A.2  Overlapped — 120 attempted utterances](#a2-overlapped-—-120-attempted-utterances)
  - [A.3  Confirming cell: `PERCEPTION_WORKERS=1`, sequential, 40 utterances](#a3-confirming-cell-perceptionworkers1-sequential-40-utterances)
  - [A.4  Confirming cell: `MAX_LIVE_STREAMS=0` (all deferred), sequential, 40 utterances](#a4-confirming-cell-maxlivestreams0-all-deferred-sequential-40-utterances)
  - [A.5  Confirming cell: shipping config control, run last, 40 utterances](#a5-confirming-cell-shipping-config-control-run-last-40-utterances)
  - [A.6  Translations, sequential condition (pass 1)](#a6-translations-sequential-condition-pass-1)
- [Appendix B — raw CSVs, verbatim](#appendix-b-—-raw-csvs-verbatim)
  - [B.1  `sequential.csv`](#b1-sequentialcsv)
  - [B.2  `overlap.csv`](#b2-overlapcsv)
  - [B.3  `confirm_w1.csv` — `PERCEPTION_WORKERS=1`](#b3-confirmw1csv-—-perceptionworkers1)
  - [B.4  `confirm_defer.csv` — `MAX_LIVE_STREAMS=0`](#b4-confirmdefercsv-—-maxlivestreams0)
  - [B.5  `confirm_control.csv` — shipping control](#b5-confirmcontrolcsv-—-shipping-control)
  - [B.6  `clip_set.csv` — the utterance set](#b6-clipsetcsv-—-the-utterance-set)
- [Appendix C — environment of record, verbatim](#appendix-c-—-environment-of-record-verbatim)
- [Appendix D — per-pass thermal and memory summary, verbatim](#appendix-d-—-per-pass-thermal-and-memory-summary-verbatim)
- [Appendix E — full statistical output, verbatim](#appendix-e-—-full-statistical-output-verbatim)

---

## 1. For the paper: what to change, and what to say instead

Five numbers in the project documentation are superseded. Each is corrected inline in the
relevant section below; this table is the index.

| the paper / docs currently say | measured here | section |
|---|---|---|
| "12–24 s post-cut latency" | **median 9.33 s, p95 14.50 s, max 19.40 s** (n = 91, sequential); **median 56.78 s** when utterances overlap, where **82 of 120 were refused outright** | §5, §6 |
| "the drain dominates" | **ByT5 decode dominates sequentially: 54.5 % vs the drain's 40.6 %.** Under overlap, queue wait dominates both at 60.9 % | §7 |
| "31.3 ms per decode step" | A flag-**unset** figure. Shipping sets `PYTORCH_NO_CUDA_MEMORY_CACHING=1` and measures **60.02 ms/step standalone, 77.81 ms/step live** | §12.1, §14 |
| "perception needs ~6× realtime and gets ~3.5×" | **Retired — neither figure reproduces and the derivation could not be reconstructed.** Perception sustains **11.75 fps** against the **15 fps** stride-2 requires for a zero drain: **1.35× short** | §8 |
| "parallel perception is 2.53× on the live path" | **1.44–1.54×** on post-cut latency over 40 clips (1.73–1.76× on the drain). The *standalone* 1.66× does reproduce | §12.3 |

Three findings that are new rather than corrections:

1. **The long-unexplained CUDA OOM is memory fragmentation.** Three OOMs were caught against
   1 Hz telemetry with **zero free 4 MB blocks, a 2 MB largest free block, and 1.03–1.09 GB
   MemAvailable**. The shipping guard `MIN_AVAILABLE_MB=350` watches a quantity that does not
   bind and could not have fired. (§11.3)
2. **`PERCEPTION_WORKERS=2`, the shipping default, is not output-preserving.** It changes the
   translation on **26 of 40 clips (65 %)** relative to one worker, while corpus BLEU moves
   **0.03**. This is the cleanest available demonstration that corpus BLEU cannot resolve
   clip-level rewrites, and it belongs in threats-to-validity. (§12.2)
3. **The pipeline cannot cold-start without internet**, despite all weights being local.
   (§12.4)

### 1.1 Suggested replacement sentences

Drop-in text, with the caveats attached where they are load-bearing:

> Post-cut latency on the shipping configuration has a median of 9.3 s and a 95th percentile
> of 14.5 s (n = 91 utterances, 2–15 s in length, replayed through the live worker). Latency
> grows with utterance length as 0.554 s per second of signing (R² = 0.50), and more tightly
> with retained frame count (R² = 0.61).

> Measured per stage, ByT5 decoding accounts for 54.5 % of post-cut latency and the
> perception drain for 40.6 %. Decoding time is proportional to output length at 77.8 ms per
> decode step (R² = 0.94, n = 126).

> When utterances are submitted back to back without waiting for the previous translation,
> as a signer would, latency is no longer a function of utterance length (r = −0.008) but of
> queue depth, rising from a 13.0 s median at depth 1 to 82.6 s at depth 7. Under these
> conditions the system's admission control refused 82 of 120 utterances.

> Perception sustains 11.75 fps against the 15 fps that stride-2 sampling requires for the
> backlog to clear during capture — a shortfall of 1.35×. Eliminating the drain entirely
> would remove 3.85 s of a 10.23 s mean and leave 5.87 s of decoding untouched.

**Do not write** "real-time", "no network dependency", or "the landmarks are identical" without
the one-worker qualifier. **Do not quote** a single latency figure without saying whether
utterances were sequential or overlapped — the two medians differ by 6×.

---

## 2. Definitions used

- **Post-cut latency** = `done_ts − cut_ts`. From the instant the utterance stops being fed
  (the replay ends, equivalent to the signer's cut) to the instant the translated string
  exists in the worker. **Includes queue wait.**
- **Utterance duration** = `capture_seconds`, the wall-clock length of the replay.
  `duration_s` is the ffprobe container duration; the two agree to ~0.03 s.
- **Retained frames** = frames kept after `FRAME_STRIDE=2`, i.e. what the model actually sees.
- **Drain** (`capture_end_to_perception_done_s`) = wall time inside `stream.finish()`: from the
  cut until the last frame's MediaPipe landmarks *and* DINOv2 features are complete.
- **Queue depth at cut** = utterances in flight, including this one, when the cut happened.
- **Declined** = the shipping admission guard `_capture_budget()` refused to start the
  utterance. On the live path the signer sees `BUSY - backlog full`. A decline is a result,
  not a failure, and is logged as its own row.
- **Deferred** = the utterance was recorded but exceeded `MAX_LIVE_STREAMS=2`, so perception
  never started during capture and all of it runs after the cut.

### 2.1 Three places `live_worker_probe.py` defines things differently

1. **Its `seconds` starts when the worker dequeues, not at the cut**, so under overlap it
   silently excludes queue wait. Both are reported here: `post_cut_latency_s` (the definition
   above) and `worker_s` (the probe's). Under overlap they differ by a **median of 47.3 s**,
   so the distinction is not cosmetic. Any latency figure taken from that probe's JSON under
   `--overlap` is a service time, not a latency.
2. **No trim is applied on a replay.** Head/tail trimming is driven by live motion scores in
   the camera loop, which a replay does not produce, and OpenASL clips carry no dead air.
   `retained_frames_after_trim` therefore equals the retained count. **This report says
   nothing about what the trim costs or saves.** On live manual clips the project log reports
   the trim removing ~23 % of frames, which would *reduce* these latencies.
3. **The probe does not enforce the shipping admission guard.** `_capture_budget()` and
   `_retain_frame()` live in the camera loop, which the probe does not run. The harness used
   here calls the real functions, so it is *stricter* than `live_worker_probe.py` and matches
   the shipping application. **Every overlap result in the project log predating 2026-10-08
   was measured on a configuration more permissive than what ships.**

---

## 3. Environment

Captured at 20:57:01 PDT on 2026-10-08, immediately before the first pass, by
`latency_runs/capture_env.sh` (re-runnable; full output in `latency_runs/env.txt`).

| | |
|---|---|
| Board | NVIDIA Jetson Orin Nano Developer Kit, 8 GB unified CPU/GPU memory (7620 MB usable), 6 cores |
| OS | L4T R36.4.7 (JetPack 6.2), kernel 5.15.148-tegra, aarch64 |
| Power mode | **mode 0 = 15 W, the maximum on this board** (`/var/lib/nvpmodel/status` = `pmode:0000`) |
| Clocks | **`jetson_clocks` NOT applied** — left dynamic. Under load the CPU sat at 1510 MHz (median across all six passes), i.e. the governor was at maximum during the work regardless. `jetson_clocks --show` requires root and is recorded as unavailable |
| Python / torch | 3.10.12 / **2.11.0** (CUDA available) |
| Other libs | cv2 4.11.0, numpy 1.24.3, transformers 4.30.2, mediapipe 0.10.13 |
| Disk | 96 GB free of 233 GB |
| git | `6e97617`, working tree clean |

**Pipeline settings in force** (all code defaults; no environment variable was overridden for
any measurement run):

```
beams 4                    BYT5_MAX_LENGTH 768        BYT5_DEVICE cuda
BYT5_DTYPE bfloat16        checkpoint checkpoint-11625-bf16
FRAME_STRIDE 2             PERCEPTION_WORKERS 2       PERCEPTION_CHUNK 30
MAX_LIVE_STREAMS 2         MAX_RETAINED_FRAMES 450    MIN_AVAILABLE_MB 350
RECORD_MODE manual         MANUAL_TRIM 1              MAX_CLIP_SECONDS 15
MEDIAPIPE_NUM_HANDS 2      MEDIAPIPE_VIDEO_MODE 1     USE_ONNX_PERCEPTION 0
DINOv2 float16, batch 32   GPU_SERIALIZE 1            PYTORCH_NO_CUDA_MEMORY_CACHING 1
```

**Ambient load.** `gnome-software`, `tracker-miner-fs-3` and `update-notifier` were closed
before the first pass. `snapd` and `packagekitd` are root-owned and were left running. The
only other non-trivial processes were the 1 Hz power-fault witness (`crash_witness.py`, which
must stay up — it is the telemetry that produced §11.3) and the desktop shell needed for the
preview window.

**One environmental difference from earlier probe runs, stated because it affects
comparability.** The background camera loop ran at **15.0 fps** in all six passes, where the
2026-08-24 probes recorded 29.7–29.8 fps. Same ConferenceCam. The likely cause is UVC
frame-rate reduction under low room light, which was **not verified**. It is constant across
both conditions so it does not bias the sequential-vs-overlapped comparison, but the
background CPU load was lighter than in a 30 fps session, so **these latencies may be slightly
optimistic relative to one**.

---

## 4. Method

### 4.1 Clip set (reproducible)

40 clips drawn from the project's existing 200-clip OpenASL evaluation set, **8 per duration
bucket, selected by ffprobe container duration only** — blind to latency and to translation
quality. Fixed seed `20261008`; the selector is `latency_runs/select_clips.py` and the result
is `latency_runs/clip_set.csv`. Available pool sizes were 41 / 41 / 30 / 21 / 16, so no bucket
was short. Total replay time 315.6 s per pass.

| bucket | n | durations (s) |
|---|---|---|
| 2–4 s | 8 | 2.04, 2.40, 2.77, 2.88, 3.04, 3.42, 3.42, 3.60 |
| 4–6 s | 8 | 4.30, 4.43, 4.50, 4.70, 4.70, 4.94, 5.00, 5.57 |
| 6–9 s | 8 | 6.27, 7.00, 7.77, 7.78, 7.91, 8.01, 8.14, 8.40 |
| 9–12 s | 8 | 9.01, 9.10, 9.38, 10.30, 10.40, 10.98, 11.39, 11.60 |
| 12–15 s | 8 | 12.47, 12.71, 13.50, 13.93, 14.15, 14.47, 14.50, 14.72 |

### 4.2 Conditions

- **Sequential** — one utterance at a time; the feeder waits for each translation. Queue depth
  is always 1.
- **Overlapped** — utterances submitted 1.5 s apart without waiting, as a signer would, **with
  the shipping admission guard in force**. This last point is a deliberate design decision:
  feeding 40 clips back to back with no guard would buffer multiple gigabytes of frames and
  OOM by construction, measuring a condition the live application cannot produce (it refuses
  to start a clip past 450 retained frames). The guard's refusals are logged and reported as a
  headline result rather than engineered away.

3 passes per condition = **240 attempted utterances**. Clip order randomised per pass with
seeds **1001 / 1002 / 1003**, the same seeds in both conditions. One process per pass, each
with its own model load, because `live_worker_probe.py`'s own docstring warns that a run
leaves the board several hundred MB dirtier than it found it. **Conditions alternated** (seq
p1, ovl p1, seq p2, ovl p2, seq p3, ovl p3) so that session drift — heat, memory — is shared
between the two conditions instead of landing entirely on whichever ran last.

### 4.3 Harness and instrumentation

Built on `live_worker_probe.py`'s own components — `feed_clip()`, `CameraLoop`, `Sampler`, the
real pipeline config — so the memory profile and the true-wall-clock replay pacing are
identical to the probe the project already trusts. **`live_worker_probe.py` itself was not
modified.** New code lives entirely in `latency_runs/`:

| file | role |
|---|---|
| `select_clips.py` | the duration-only clip draw |
| `capture_env.sh` | environment of record |
| `latency_probe.py` | one pass of one condition; 38-column CSV, one `fsync`'d row per utterance |
| `instrument.py` | the per-stage timing split (see below) |
| `run_all.sh` | the six passes, tegrastats per pass, then analysis |
| `run_confirm.sh` | the confirming tests of §12 |
| `analysis.py`, `parse_tegrastats.py` | statistics, figure, thermal summary |

**Per-utterance rows are `fsync`'d as they complete.** This board loses power silently — 11
times since 2026-08-19 — so a cut costs at most the one utterance in flight.

**The stage split is obtained from forward hooks and function wrappers, not pipeline edits.**
`generate()` calls `self.encoder(...)` explicitly before the decoder loop, and the SHuBERT
model sits at `model.encoder.adapter.signhubert_adapter`, so timing hooks on those two module
boundaries plus a wrapper around `model.generate` yield SHuBERT time, ByT5 encoder time, decode
time, and the exact decode step count. **Honest limit:** these time CPU-side wall at module
boundaries with no mid-`generate` `cuda.synchronize()`, because inserting one would itself
change execution timing. For decode this is the right instrument — the project measured the
step as ~100 % CPU dispatch with a 0.01 ms GPU tail — but for the SHuBERT/encoder split it may
under-attribute GPU work landing after the boundary. Both terms are small (0.09 s and 0.26 s
mean) so no conclusion here rests on them. The 3-pass determinism check (§10) independently
confirms the instrumentation changed no output.

### 4.4 Commands run

```bash
python3 latency_runs/select_clips.py          # -> clip_set.csv
./latency_runs/capture_env.sh                 # -> env.txt
./latency_runs/run_all.sh                     # the six passes + analysis + thermal
./latency_runs/run_confirm.sh                 # the confirming tests of §12
```

`run_all.sh` per pass, for i in 1..3 and seeds 1001/1002/1003:

```bash
tegrastats --interval 1000 > latency_runs/tegrastats_<cond>_pass<i>.log &
shubert_venv/bin/python3 latency_runs/latency_probe.py \
    --condition <sequential|overlap> --pass-index <i> --seed <seed> \
    --out latency_runs/<cond>.csv --limit 0
```

Total wall clock for the six passes: **20:57:01 → 21:41:23, 44 minutes.** (A 2–2.5 h
projection was given beforehand; the overlapped passes finished far faster than projected
because most of their utterances were refused — §11.1.)

---

## 5. Results — sequential condition

**120 utterances attempted, 91 measured, 29 declined by the admission guard, 0 failed.**

### 5.1 Post-cut latency, seconds

| n | mean | median | p90 | p95 | min | max |
|---|---|---|---|---|---|---|
| 91 | 10.23 | **9.33** | 14.05 | 14.50 | 5.11 | 19.40 |

A p99 is not quoted: at n = 91 it sits between the two largest samples and carries no
information the max does not.

### 5.2 By utterance-duration bucket

| bucket | n | mean | median | p90 | p95 | min | max |
|---|---|---|---|---|---|---|---|
| 2–4 s | 21 | 7.65 | 7.44 | 11.75 | 11.87 | 5.11 | 11.88 |
| 4–6 s | 19 | 8.01 | 7.96 | 9.23 | 9.64 | 6.43 | 9.92 |
| 6–9 s | 16 | 10.77 | 10.12 | 13.71 | 13.98 | 9.03 | 14.05 |
| 9–12 s | 17 | 11.89 | 12.62 | 13.89 | 14.07 | 7.83 | 14.39 |
| 12–15 s | 18 | 13.56 | 13.04 | 18.32 | 19.33 | 8.79 | 19.40 |

### 5.3 Latency vs utterance length

    post_cut_latency = 0.554 * duration_s   + 5.993      R^2 0.498   Pearson r 0.706
    post_cut_latency = 0.0455 * retained_frames + 5.420   R^2 0.607   Pearson r 0.779

Frame count predicts better than wall-clock duration, as expected: perception cost and
encoder input length are both per-frame, and stride-2 frame count is not an exact multiple
of duration across clips of differing source frame rates. **Neither explains more than
61 % of the variance** — the remainder is output length, which drives decode (§9).

### 5.4 Fraction at or below a threshold

| threshold | count | fraction |
|---|---|---|
| ≤ 10 s | 51/91 | 56.0 % |
| ≤ 12 s | 65/91 | 71.4 % |
| ≤ 15 s | 87/91 | 95.6 % |
| ≤ 24 s | 91/91 | 100.0 % |
| ≤ 30 s | 91/91 | 100.0 % |

---

## 6. Results — overlapped condition

**120 utterances attempted, 35 measured, 82 declined, 3 failed with CUDA OOM.**
17 of the 35 measured were *deferred* (over the
`MAX_LIVE_STREAMS=2` cap, so perception never started during capture).

**The headline here is the acceptance rate, not the latency.** With a 1.5 s inter-utterance
gap, the shipping configuration produced a translation for **35 of 120 utterances (29 %)**.

| n | mean | median | p90 | p95 | min | max |
|---|---|---|---|---|---|---|
| 35 | 56.83 | **56.78** | 91.32 | 94.34 | 9.52 | 97.61 |

| threshold | count | fraction |
|---|---|---|
| ≤ 12 s | 2/35 | 5.7 % |
| ≤ 24 s | 6/35 | 17.1 % |
| ≤ 30 s | 6/35 | 17.1 % |
| ≤ 60 s | 18/35 | 51.4 % |
| ≤ 90 s | 31/35 | 88.6 % |

### 6.1 Latency is set by queue depth, not by utterance length

    post_cut_latency vs duration:  slope -0.053   R^2 0.000   Pearson r -0.008

**There is no relationship with utterance length at all.** The relationship is with how many
utterances were already in flight at the cut:

| queue depth at cut | n | mean | median | min | max |
|---|---|---|---|---|---|
| 1 | 3 | 13.09 | 13.01 | 10.28 | 15.99 |
| 2 | 4 | 22.52 | 18.76 | 9.52 | 43.03 |
| 3 | 8 | 44.70 | 42.12 | 34.43 | 55.22 |
| 4 | 6 | 58.67 | 58.54 | 43.77 | 75.73 |
| 5 | 7 | 80.59 | 80.23 | 64.48 | 96.34 |
| 6 | 4 | 81.53 | 81.17 | 75.26 | 88.54 |
| 7 | 3 | 86.64 | 82.62 | 79.69 | 97.61 |

At depth 1 the overlapped numbers agree with the sequential distribution (median 13.01 s).
Everything above depth 1 is waiting, not working. **This is the table to put in the paper**
if it claims anything about conversational pace.

---

## 7. Where the time goes — per-stage breakdown (request A)

Shares are computed per utterance and then averaged, so they are not sensitive to a few
long utterances dominating a pooled sum.

### 7.1 Sequential

| stage | mean share | median share | mean s | median s | n |
|---|---|---|---|---|---|
| queue wait | 0.0 % | 0.0 % | 0.00 | 0.00 | 91 |
| drain (perception + DINOv2) | 40.6 % | 38.5 % | 3.85 | 3.93 | 91 |
| SHuBERT encoder | 0.9 % | 0.8 % | 0.09 | 0.08 | 91 |
| ByT5 encoder blocks | 2.6 % | 2.4 % | 0.26 | 0.21 | 91 |
| **ByT5 decode** | 54.5 % | 56.0 % | 5.87 | 5.66 | 91 |
| ByT5 stage total | 58.2 % | 59.8 % | 6.25 | 6.03 | 91 |
| unattributed | 1.2 % | 1.3 % | 0.13 | 0.13 | 91 |

### 7.2 Overlapped

| stage | mean share | median share | mean s | median s | n |
|---|---|---|---|---|---|
| queue wait | 60.9 % | 68.2 % | 39.48 | 47.29 | 35 |
| drain (perception + DINOv2) | 11.2 % | 4.2 % | 2.30 | 1.88 | 18 |
| SHuBERT encoder | 0.4 % | 0.1 % | 0.12 | 0.09 | 35 |
| ByT5 encoder blocks | 1.0 % | 0.5 % | 0.29 | 0.22 | 35 |
| **ByT5 decode** | 16.2 % | 9.9 % | 6.15 | 6.27 | 35 |
| ByT5 stage total | 17.9 % | 11.4 % | 6.66 | 6.72 | 35 |
| unattributed | 15.4 % | 1.1 % | 9.52 | 0.27 | 35 |

**Sequential: ByT5 decode is the dominant term at 54.5 %, against the perception drain's
40.6 %.** The project documentation said the drain dominates; that came from a single live
clip and the systematic measurement reverses it. SHuBERT itself is negligible at 0.9 %.

**Overlapped: queue wait dominates both at 60.9 %.** Two notes on reading that table:

* `drain` has n = 18 rather than 35 because **17 measured utterances were deferred** — over
  the stream cap, perception never started during capture, so there is no drain to report
  and the cell is left blank rather than inferred.
* For those 17, perception, cropping and DINOv2 all run *inside* `process_frames`, a path
  whose internals were not instrumented. That is what the large `unattributed` figure is
  (up to 32.5 s on one utterance). It is named, not attributed to a stage it was not
  measured in.

### 7.3 The five worst utterances in each condition

**Sequential**

| clip | pass | dur s | frames | bytes | steps | **latency s** | wait | drain | shubert | enc | decode | other | dominated by |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `jsOSzFyx8RA_03` | 3 | 12.71 | 191 | 153 | 155 | **19.40** | 0.0 | 5.3 | 0.1 | 0.4 | 13.4 | 0.2 | decode |
| `jsOSzFyx8RA_03` | 1 | 12.71 | 191 | 153 | 155 | **19.32** | 0.0 | 5.2 | 0.1 | 0.4 | 13.3 | 0.2 | decode |
| `oP2WdYlaflE_03` | 1 | 13.93 | 209 | 148 | 150 | **17.89** | 0.0 | 5.3 | 0.1 | 0.5 | 11.7 | 0.2 | decode |
| `oP2WdYlaflE_03` | 2 | 13.93 | 209 | 148 | 150 | **16.89** | 0.0 | 4.2 | 0.1 | 0.4 | 11.8 | 0.2 | decode |
| `uZwKNtHx9FE_01` | 1 | 14.47 | 174 | 125 | 127 | **14.60** | 0.0 | 2.6 | 0.3 | 0.7 | 10.8 | 0.1 | decode |

**Overlapped**

| clip | pass | dur s | frames | bytes | steps | **latency s** | wait | drain | shubert | enc | decode | other | dominated by |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| `LT0-h4P_xBY_04` | 3 | 7.77 | 117 | 96 | 98 | **97.61** | 64.5 | — | 0.1 | 0.2 | 7.9 | 24.9 | queue wait *(deferred)* |
| `mNF1kg9azQw_05` | 1 | 10.30 | 124 | 87 | 89 | **96.34** | 63.6 | — | 0.1 | 0.2 | 6.4 | 26.0 | queue wait *(deferred)* |
| `MJijm9kbdHA_01` | 1 | 14.50 | 174 | 100 | 102 | **93.48** | 52.2 | — | 0.1 | 0.4 | 8.2 | 32.5 | queue wait *(deferred)* |
| `WMoeSSvCyTU_03` | 1 | 7.91 | 119 | 60 | 62 | **93.17** | 86.7 | 1.6 | 0.1 | 0.2 | 4.4 | 0.2 | queue wait |
| `J-0KHhPS_m4_03` | 3 | 5.00 | 75 | 58 | 60 | **88.54** | 68.5 | — | 0.1 | 0.2 | 4.2 | 15.5 | queue wait *(deferred)* |

All five worst sequential utterances are decode-dominated, and the two worst are the longest
**output**, not the longest input. All five worst overlapped utterances are queue-wait
dominated. The split is measured per stage, so this is attribution rather than inference.

---

## 8. Perception real-time factor (request B)

Measured directly per utterance as `perception_frames_at_cut / capture_seconds` — frames
actually through MediaPipe when the cut arrived, over the replay's length.

| | sequential (n = 91) | overlapped, streamed only (n = 18) |
|---|---|---|
| sustained perception fps, mean | **11.75** | 12.00 |
| median | 12.32 | 11.62 |
| min / max | 4.50 / 14.38 | 7.72 / 13.97 |
| shortfall vs 15 fps, mean | **1.35×** | 1.27× |
| shortfall vs 15 fps, median | 1.22× | 1.29× |
| same against raw 30 fps, mean | 2.70× | 2.54× |
| aggregate perception CPU s per s of utterance | 1.64 (≈0.82 per worker) | — |

### 8.1 Deriving the "required" factor from measured quantities

The project documentation says *"perception needs ~6× realtime and gets ~3.5×"* with no
derivation recorded. Here is the derivation that can be built from measurement:

1. The camera runs at 30 fps and `FRAME_STRIDE=2`, so **retained frames arrive at 15 fps**.
2. Perception finishes inside the utterance — a zero drain — exactly when it sustains 15 fps.
3. Measured sustained rate: **11.75 fps** mean.
4. Required speed-up = 15 / 11.75 = **1.35× mean, 1.22× median**.

**Neither 6× nor 3.5× reproduces under any definition that could be constructed, and the 6×
derivation could not be reconstructed.** Against the raw 30 fps camera rate the figure is
2.70×, which is the closest any definition comes to 3.5×, and it is the wrong
denominator — the model only ever sees every second frame.

**What this does to the paper's framing.** The remaining gap is still structural, but the
mechanism changes: closing the perception gap entirely would remove the mean drain
(3.85 s) from the mean post-cut
latency (10.23 s) and leave the mean decode
(5.87 s) untouched. Decode is ~100 % CPU dispatch
(project measurement, 2026-08-24: 31.29 ms CPU enqueue vs 31.30 ms total wall, GPU tail
0.01 ms), so **the binding constraint is single-core dispatch throughput, not detector
speed.** That is a different argument from the one the paper currently makes.

---

## 9. Thermal state and drift across passes (request C)

1 Hz `tegrastats` was logged for every pass (`latency_runs/tegrastats_*.log`, committed).

| pass log | samples | RAM used max MB | lfb largest-free min | swap max MB | CPU MHz min/med/max | Tj max °C | Tj median °C | VDD_IN max mW |
|---|---|---|---|---|---|---|---|---|
| `control` | 339 | 7215 | **1 MB** | 644 | 729/1510/1510 | 55.9 | 54.5 | 10141 |
| `defer` | 86 | 3012 | **488 MB** | 559 | 729/1510/1510 | 52.4 | 51.8 | 6071 |
| `overlap_pass1` | 170 | 7220 | **1 MB** | 456 | 729/1510/1510 | 56.1 | 53.8 | 10419 |
| `overlap_pass2` | 163 | 7238 | **1 MB** | 824 | 729/1510/1510 | 55.8 | 53.8 | 10498 |
| `overlap_pass3` | 205 | 7210 | **1 MB** | 608 | 729/1510/1510 | 55.7 | 53.8 | 10435 |
| `sequential_pass1` | 462 | 7231 | **1 MB** | 956 | 729/1510/1510 | 56.1 | 54.4 | 10300 |
| `sequential_pass2` | 272 | 7211 | **1 MB** | 475 | 729/1510/1510 | 55.8 | 54.2 | 10237 |
| `sequential_pass3` | 419 | 7199 | **1 MB** | 675 | 729/1510/1510 | 55.6 | 54.1 | 10260 |
| `w1` | 762 | 7077 | **1 MB** | 579 | 729/1510/1510 | 55.0 | 54.0 | 9547 |

**No thermal throttling signature.** Tj peaked at 56.1 °C against a board that throttles far
higher; the median CPU clock was the 1510 MHz maximum in every pass; peak draw was 10.5 W of
the 15 W budget. `tegrastats` exposes no throttle flag, so this is the clock-and-temperature
record rather than a throttle measurement — but a clock pinned at maximum and a median Tj
that *falls* from pass 1 to pass 3 leaves no room for a throttling story.

**Note `lfb` bottomed at 1 MB in every pass.** That is the Jetson contiguous-carveout gauge,
and it is the subject of §11.3.

### 9.1 Per-pass median latency (the drift check you asked for)

| condition | pass 1 | pass 2 | pass 3 |
|---|---|---|---|
| sequential | 9.96 s (n=40) | 9.03 s (n=19) | 9.30 s (n=32) |
| overlapped | 60.30 s (n=11) | 48.73 s (n=10) | 71.75 s (n=14) |

**Sequential pass 3 is not slower than pass 1** (9.30 s vs 9.96 s), so there is no drift
effect to report. Two honest qualifications: the sequential n falls in passes 2 and 3
because of the memory degradation of §11.2, so those medians are over smaller and different
subsets; and the overlapped medians move with how many utterances were accepted and at what
queue depth, not with elapsed time.

---

## 10. Cold start and first-utterance cost (request D)

**Model warmup** (`SHuBERTProcessor.warmup()`: both DINOv2 embedders plus the ByT5
checkpoint): **18.3, 18.4, 18.7, 18.8, 23.1, 23.3 s** across the six passes. Reported separately and never
counted as latency.

The first utterance of each pass is kept and flagged rather than discarded, because its cost
was requested separately. (The brief said to discard the warmup; that is read as the
model-load time, not as the first utterance.)

| condition | pass | clip | duration | first-utterance latency | that bucket's range |
|---|---|---|---|---|---|
| sequential | 1 | `uZwKNtHx9FE_01` | 14.47 s | **14.60 s** | 8.79–19.40 s |
| sequential | 2 | `sP6B4dlSOoc_04` | 5.57 s | **9.13 s** | 6.43–9.92 s |
| sequential | 3 | `mNF1kg9azQw_03` | 11.39 s | **10.94 s** | 7.83–14.39 s |
| overlapped | 1 | `uZwKNtHx9FE_01` | 14.47 s | **15.99 s** | 15.99–93.48 s |
| overlapped | 2 | `sP6B4dlSOoc_04` | 5.57 s | **10.28 s** | 10.28–88.54 s |
| overlapped | 3 | `mNF1kg9azQw_03` | 11.39 s | **13.01 s** | 13.01–96.34 s |

Sequential steady state excluding those first utterances: n = 88, median
9.30 s — against 9.33 s including them.
**Every first utterance falls inside its own duration bucket's range, so there is no
measurable first-utterance penalty beyond the warmup itself.**

---

## 11. Live-camera utterances (request E) — not performed

Ten live push-to-record utterances were requested and then **withdrawn by the requester**
mid-session. They were not measured and nothing in this report is live signing. Had they
been run they would also have been the only data here exercising the head/tail trim (§2.1).

---

## 12. Decode-length dependence (request F)

Both conditions pooled, since decode cost is a property of the stage rather than the
condition (n = 126):

    byt5_decode_s vs output_bytes :  77.81 ms/byte   intercept +0.096 s   R^2 0.940   r 0.969
    byt5_decode_s vs decode_steps :  77.81 ms/step   intercept -0.060 s   R^2 0.940   r 0.969
    naive decode_s / decode_steps :  mean 76.93 ms/step   median 73.22   min 68.08   max 114.00

ByT5 is byte-level, so steps ≈ bytes + specials and the two fits coincide. **The relationship
is tight (R² 0.94) and the intercept is ~0: decode time is almost purely proportional to
output length.** Validity range is the measured one, 47–153 bytes.

### 12.1 Against the project's 31.3 ms/step figure

The documentation's **31.3 ms/step** is 2.5× below this. §14.1 shows why, and it is not a
discrepancy in this measurement: 31.3 ms was measured with the CUDA caching-allocator
workaround *disabled*, and the shipping configuration enables it.

### 12.2 Output size

| | sequential | overlapped |
|---|---|---|
| output bytes, mean | 75.5 | 74.7 |
| decode steps, mean | 77.5 | 76.7 |
| output bytes, max | 153 | 148 |
| decode steps, max | 155 | 150 |
| **at or over the 768-byte cap** | **0** | **0** |

**No output came near the `BYT5_MAX_LENGTH=768` cap.** The longest was 153 bytes, 20 % of it.
The cap is therefore not shaping any latency reported here.

---

## 13. Determinism, and the finding it uncovered (request: Step 5)

### 13.1 The pipeline is deterministic at a fixed configuration

**Sequential, 3 passes: 36 clips were measured in more than one pass, and
0 differ. Byte-identical output.** This also confirms that the forward-hook
instrumentation of §4.3 changed nothing.

A separate control run at the shipping configuration, executed last in the session, agrees
with pass 1 on **27 of 27** shared clips.

### 13.2 But overlapped output differed — and the cause was not what it looked like

Comparing sequential against overlapped: **28 clips measured in both, 9
differ — and all 9 of the differing clips were *deferred* in the overlapped run.**
Every clip that was streamed in both conditions was byte-identical.

The obvious reading is "the deferred fallback path corrupts output". **That reading is
wrong**, and a three-cell test (§14.2) showed the cause is the number of perception workers.

---

## 14. Confirming tests (run 2026-10-08/09)

Two follow-up questions, each requiring a configuration change that the measurement brief
forbade, so both were run only after explicit approval. Driver: `latency_runs/run_confirm.sh`.

### 14.1 Why decode measures 77.8 ms/step against the documented 31.3 ms

`PERCEPTION_WORKERS=1` would **not** have isolated this: in sequential mode `stream.finish()`
joins the perception and embed threads before ByT5 starts, so perception is idle during
decode either way. The isolating test is the project's own committed microbenchmark,
`benchmarks/nocache_ab.py`, run in each cell of the allocator flag.

| cell | measured 2026-08-24 | measured 2026-10-08 |
|---|---|---|
| flag unset | 38.84 / 38.45 ms/step | **38.92 min / 40.01 median** |
| `=0` | — | 37.02 |
| **`=1` — what ships** | 99.74 / 98.24 | **60.02** |

The flag-unset cell reproduces the August figure almost exactly, which validates the
comparison. **So the documented 31.3 ms is a flag-unset number quoted as if it described the
shipping configuration, which sets `PYTORCH_NO_CUDA_MEMORY_CACHING=1`.**

`benchmarks/decode_bound.py` at the shipping flag, beams 4, T = 100, varying forced steps:

```
 steps    total   ms/step
    16    1.98s    124.0
    32    2.97s     92.8
    64    6.04s     94.4
   128    8.58s     67.0
```

Per-step cost falls as the step count amortises a fixed per-`generate` overhead, and **the
live 77.8 ms/step sits inside this standalone shipping-flag range.** An earlier draft of this
report attributed a residual 1.30× to "live CPU contention" by comparing `nocache_ab`'s
min-of-5 (60.02) against the live mean (77.8); **that was wrong and is retracted.** The whole
2.5× gap against the documentation is the flag plus step-count amortisation, with no
contention term needed.

Secondary finding: **the flag's own cost has fallen from 2.56× to ~1.54× since August**
(38.92 → 60.02). The project's §13.7 rejection of removing the flag was argued partly on the
2.56×, so that argument has weakened — the NVML assert it also guards is the durable reason.

### 14.2 The output divergence is the WORKER COUNT, not the deferred path

Three cells, all sequential at queue depth 0 so neither queueing nor memory pressure can
confound the text comparison, 40 clips each. The key observation enabling the test: with
`PERCEPTION_WORKERS=1`, `add_frame`'s `(index // PERCEPTION_CHUNK) % 1` is always 0, so one
detector sees every frame in order and there are **no chunk boundaries** — exactly what
`video_holistic` does on the non-streamed path. `MAX_LIVE_STREAMS=0` forces every clip onto
that non-streamed path.

| comparison | shared clips | differ |
|---|---|---|
| `PERCEPTION_WORKERS=1` (streamed) vs `MAX_LIVE_STREAMS=0` (non-streamed) | 40 | **0** |
| baseline, 2 workers vs `PERCEPTION_WORKERS=1` | 40 | **26** |
| baseline, 2 workers vs non-streamed | 40 | **26** |
| baseline, 2 workers vs control, also 2 workers | 27 | **0** |

**One detector seeing every frame in order gives byte-identical output through either code
path (0 of 40 differ). The streamed/deferred distinction is irrelevant.** What moves
the text is splitting frames across two workers in 30-frame chunks, which forces a MediaPipe
re-detection at each boundary: **26 of 40 clips (65 %) differ**, and the non-streamed
path differs from the 2-worker default on exactly the same 26 clips.
The deferred path differed in §13.2 only because it is implicitly single-worker.

**The shipping default is 2 workers.** So the shipping configuration's translations differ
from single-worker perception on about two thirds of clips.

**This is not a quality claim in either direction.** The project's committed 200-clip chunk
sweep already measured that axis: raw BLEU **19.54** at chunk 30 / 2 workers against a
**1-worker control at 19.51**. So **65 % of clips change text while corpus BLEU moves 0.03**.
For the paper that is the single most useful sentence in this report: it is a direct,
in-house demonstration of the instrument limit that *Beyond BLEU* (arXiv 2609.03734)
establishes in general, and it belongs in threats-to-validity next to that citation.

A documentation defect follows from it, now fixed in the code (commit `e3b358e`):
`streaming_perception.py` described the change as "a pure scheduling change" whose
"landmarks are identical, not approximated". **That is true only at one worker.** What its
argument actually establishes is that the reorder buffer preserves frame *order* into the
embed stage, which it does; it does not preserve per-frame landmarks, because each worker's
detector sees a different subsequence.

#### The 26 clips, with both translations

`one_worker` and `non_streamed` were byte-identical on all 26, so only one is shown.

| clip | dur s | frames | 2 workers (SHIPPING) | 1 worker |
|---|---|---|---|---|
| `ZgvLTS2zJdo_02` | 14.72 | 177 | I want to share a little bit of an opportunity with you. | I want to share a little bit of a story with you about one more time. |
| `uZwKNtHx9FE_01` | 14.47 | 174 | Most of the outbreaks are that children who do not have a vaccine who live in an ultra-orthodox Jewish community in New York. | Most of the outbreaks are that children who do not have a vaccine who live in strict ultra-orthodox Jewish communities in New York. |
| `oP2WdYlaflE_03` | 13.93 | 209 | And a limited number of Deaf and hard of hearing students can be recommended in primary reports such as the Department of Health and Human Services. | The study also limits the number of community members to primary reports such as the Department of Health and People and Human Services. |
| `I976oI3s1jQ_02` | 13.50 | 162 | She said she was still working for two days, but she had a high fever and chills and decided to stay home recovering. | He said he was still working for two days, but on a high fever and chills, he decided to stay home and recover. |
| `jsOSzFyx8RA_03` | 12.71 | 191 | Or the show isn't a lot about the experience, but it's about who you are as a person and how you through surveillance through life, the photography, too. | Or the show isn't a lot about the experience, but it's about who you are as a person and how you through savingstories throughout life, photography, as a Deaf person. |
| `uZwKNtHx9FE_07` | 12.47 | 150 | They said the words were added without delay, were immediately introduced into the legislation to make sure that the work began. | They said the words were added without delays, were immediately introduced into the legislation to make sure the work began. |
| `mNF1kg9azQw_03` | 11.39 | 137 | Jennifer shared her experience with a woman who was currently in the photography of Floyd. | Next we will talk about the next step, the next step, the next step, the next step. |
| `sP6B4dlSOoc_05` | 10.98 | 165 | On Wednesday President Trump signed an executive order about trade reform and high kidney care. | On Wednesday President Trump signed an executive order about trade reform and life kidney care. |
| `5iRIeHAK9b0_02` | 10.40 | 156 | On Friday two men in California warned of child abuse in court for confession today (Friday). | A California male couple called for child abuse Friday in a court called for police to plead guilty to child abuse. |
| `mNF1kg9azQw_05` | 10.30 | 124 | I'm wondering how to send it to another person who sent it to me. | I wonder how do you recommend people respond to someone who has been vlogging about it? |
| `oP2WdYlaflE_02` | 9.10 | 137 | The expected percentage of a person from the Institute for Health is experimenting at the University of Washington. | The expected percentage of a person from the Institute for Health is falling from the University of Washington. |
| `wH3lJm0Typ0_03` | 9.01 | 135 | At the end of the game, there were also fans who cheered their fans and critics who criticized them. | At the end of the game, there were also people who cheered their fans with cheers and critics canceling them. |
| `f75G_hZMSHs_02` | 7.78 | 117 | The similar conversation between the police officers and his hometown includes the former police officer and his hometown. | The similar conversations between the police officers and his hometown are including the previous officers and his hometown. |
| `LT0-h4P_xBY_04` | 7.77 | 117 | Of course, I want to make sure that the police officers and the police officers are shared with the people and the clients. | Of course, I want to clearly warn you that I just shared with the people who are self-esteeming. |
| `I976oI3s1jQ_01` | 7.00 | 84 | My strategy on the chill was a little bit patient and I thought it would be perfect. | I feel like it's a chill and I've been very patient with it and I think it's possible. |
| `TXyNxVRRPt8_01` | 6.27 | 94 | We brought them to the Deaf Community Forum to bring their concerns to South Carolina. | We bring them to the deaf community forum to bring their concerns to South Carolina. |
| `J-0KHhPS_m4_03` | 5.00 | 75 | Here's a quick question: How can I help? | Here's a question from the next question: How did it stop? |
| `5iRIeHAK9b0_03` | 4.70 | 71 | Children ranging in age from 3 to 29 years old. | Children between the ages of three and twenty-nine are kids. |
| `TXyNxVRRPt8_03` | 4.50 | 68 | Most people don't know about it. | Most people don't know about the same NAD programs. |
| `k5Gxbifw8s8_05` | 4.43 | 67 | And more young people wear rabbits. | And more young people use rabbit. |
| `iMJ9CjBX7eo_02` | 4.30 | 65 | Thank you for joining us at the Community Forum and we will host it together. | Thank you for joining us at the Community Forum and I will be hosting it together. |
| `fNT8a6e1gx8_08` | 3.60 | 54 | It's just that we have problems with expressing ourselves. | It's just about our problems of expressing ourselves. |
| `ZgvLTS2zJdo_01` | 3.42 | 41 | The first thing we want to do is to open and ensure that some students have access to sign language interpreters. | The first is to open and provide resources for deaf and hard of hearing people. |
| `rSiciLyYOyI_07` | 2.88 | 35 | That was amazing. | It was an amazing and never experienced that. |
| `fNT8a6e1gx8_06` | 2.77 | 42 | Tend to ask your students to graduate. | I feel like again, the students have been asking me to go to school for their time. |
| `f75G_hZMSHs_04` | 2.40 | 36 | The company will likely start their wedding this weekend. | The tourist wedding is scheduled for this weekend. |

### 14.3 The documented 2.53× for two perception workers does not reproduce

The `PERCEPTION_WORKERS=1` cell doubles as the first independent check of the project's
"live probe 19.7 → 7.8 s/clip = 2.53×" (2026-08-10, 4 own-footage clips).

| configuration | n | post-cut median | drain median | sustained perception fps |
|---|---|---|---|---|
| 1 worker | 40 | **14.38 s** | 6.63 s | 7.74 |
| 2 workers — control, run last | 27 | **9.33 s** | 3.76 s | 11.89 |
| 2 workers — baseline, seq pass 1 | 40 | **9.96 s** | 3.84 s | 12.80 |

Two workers buy **1.44–1.54× on post-cut latency**, **1.73–1.76× on the drain alone**, and
**1.54–1.65× on sustained perception fps**. The last reproduces the project's *standalone*
1.66× (1 worker 6.7 s, 2 workers 4.0 s) almost exactly. **So the component figure survives
and the live 2.53× does not** — the reverse of this project's usual direction of error, where
component wins evaporate on the live path. The control declined 13 of 40 utterances to the
memory degradation of §15.2, so its n is 27; its median still matches pass 1, so latency
itself did not drift.

**Also measured: the non-streamed path costs ~3.3×.** With `MAX_LIVE_STREAMS=0`, every clip
deferred: post-cut median **32.97 s** (mean 30.35, max 56.81) against the baseline's 9.96 s. That is what
an utterance pays when it arrives over the stream cap, and it is why the overlapped deferred
utterances in §6 look so bad.

### 14.4 An unplanned finding: the pipeline cannot cold-start offline

The first `MAX_LIVE_STREAMS=0` attempt died during model warmup:

```
torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg', pretrained=False)
  -> urllib.error.HTTPError: HTTP Error 504: Gateway Time-out
FATAL: exception not rethrown                      (SIGABRT, exit 134)
```

`dinov2_features.py:70` fetches the model *definition* from GitHub at every startup, and
`torch.hub._get_cache_or_reload` resolves the branch over HTTPS **before** it consults the
cache. So the cached repo at `~/.cache/torch/hub/facebookresearch_dinov2_main` does not help
— verified present at the moment of failure — and neither do local weights with
`pretrained=False`, because it is the architecture that comes over the wire. **A single
transient GitHub outage takes down the live path at launch, and it aborts rather than raising
cleanly.** For a contribution about edge deployment this deserves either a fix
(`source='local'` against the cached repo) or an explicit statement. It is **not fixed**.
The run was relaunched and completed; it had produced no measurement before dying, so no
result was discarded to make this go away.

---

## 15. Anomalies — every failure, decline and divergence

**Nothing was re-run to make any of these disappear.** The one re-run in the session
(§14.4) died in warmup before producing a measurement.

### 15.1 At live pace the shipping guard refuses two utterances in three

82 of 120 overlapped utterances were declined by the shipping `_capture_budget()`:
**66 for backlog** (460–629 retained frames against the 450 limit) and
**16 for free memory** (242–322 MB against the 350 MB floor).

| condition | pass | first decline at order index | declines in pass |
|---|---|---|---|
| sequential | 1 | — (none) | 0 |
| sequential | 2 | 19 | 21 |
| sequential | 3 | 32 | 8 |
| overlapped | 1 | 8 | 29 |
| overlapped | 2 | 8 | 27 |
| overlapped | 3 | 9 | 26 |

**Declines begin after about eight utterances at a 1.5 s gap, in every overlapped pass.**
This is shipping behaviour, not a harness artifact: on the live path the signer sees
`BUSY - backlog full` and the sentence is dropped. It is the honest answer to "what happens
when a signer does not wait", and for the paper it is a stronger number than the latency
distribution alone.

### 15.2 Available memory degrades monotonically within a single process

Each pass is a fresh process. Each starts with 1.9–2.7 GB available and falls until the
350 MB floor refuses utterances.

| condition | pass | available at first cut | at last cut | min seen | declines |
|---|---|---|---|---|---|
| sequential | 1 | 1914 MB | 835 MB | 283 MB | 0 |
| sequential | 2 | 2739 MB | 317 MB | 279 MB | 21 |
| sequential | 3 | 2138 MB | 292 MB | 292 MB | 8 |
| overlapped | 1 | 1958 MB | 363 MB | 330 MB | 29 |
| overlapped | 2 | 2744 MB | 322 MB | 242 MB | 27 |
| overlapped | 3 | 2340 MB | 526 MB | 363 MB | 26 |

Sequential pass 1 completed all 40; passes 2 and 3 declined 21 and 8. Peak swap reached
956 MB. **The mechanism was not diagnosed** — candidates are the CUDA driver not returning
freed blocks under `PYTORCH_NO_CUDA_MEMORY_CACHING=1`, MediaPipe native buffers, or
Python-side retention — and separating them was outside the brief.

**Consequence worth stating in the paper: a long live session progressively stops accepting
utterances and needs a process restart.** The longest real signing session in the project's
records is ~9 clips, which is why this had never been observed.

### 15.3 The three CUDA OOMs were memory FRAGMENTATION, with over 1 GB free

All 3 failures (`YR7BS6l_95E_05`, `ZgvLTS2zJdo_01`, `sP6B4dlSOoc_05`, overlapped pass 2) landed
at 21:23:08. The independent 1 Hz power-fault witness recorded, across the samples spanning
that instant:

```
21:23:05  lfb_4mb=0  lfb_max_mb=2  memavail_mb=1092  load=4.00
21:23:06  lfb_4mb=0  lfb_max_mb=2  memavail_mb=1082  load=4.00
21:23:07  lfb_4mb=0  lfb_max_mb=2  memavail_mb=1083  load=4.00
21:23:08  lfb_4mb=7  lfb_max_mb=4  memavail_mb=1027  load=3.92   <- failed clips released
```

**Zero free 4 MB blocks, a 2 MB largest free block, and 1.03–1.09 GB MemAvailable.** A CUDA
allocation on Jetson needs contiguous NvMap carveout, so fragmentation gates it, not free
megabytes. Three consequences:

* **`MIN_AVAILABLE_MB=350` cannot prevent these OOMs.** It did not fire and could not have,
  at over a gigabyte free.
* **Every memory-headroom figure in the project's records is denominated in MemAvailable**,
  including the "447–702 MB" that the overlap path was said to run near. Those numbers are
  real but they are not the quantity that binds.
* A principled guard would read `lfb_max_mb` from `/proc/buddyinfo`. **Not obviously safe to
  add**: refusing utterances on fragmentation could refuse constantly, and the project has a
  recorded precedent — a 1100 MB floor stopped all recording and never resumed.

This test was set up on 2026-09-20 and labelled "NOT YET EXERCISED, and it is the point of
the whole exercise" — check whether lfb is low while memavail is healthy. **Three
independent OOMs show exactly that signature.** Alignment is ±1 s (1 Hz witness against the
recorded failure time), but the lfb = 0 condition held for at least three consecutive
samples beforehand, so the attribution does not depend on sub-second alignment. The earlier
caveat that a killed TensorRT build might have caused the fragmentation no longer applies:
no TensorRT was anywhere near this session.

### 15.4 Full list of failures and declines

| condition | pass | clip | kind | reason | queue depth |
|---|---|---|---|---|---|
| overlapped | 2 | `YR7BS6l_95E_05` | **CUDA OOM** | AcceleratorError: CUDA error: out of memory Search for `cuda | 3 |
| overlapped | 2 | `ZgvLTS2zJdo_01` | **CUDA OOM** | AcceleratorError: CUDA error: out of memory Search for `cuda | 2 |
| overlapped | 2 | `sP6B4dlSOoc_05` | **CUDA OOM** | AcceleratorError: CUDA error: out of memory Search for `cuda | 3 |
| sequential | 2 | `oP2WdYlaflE_02` | declined | 316MB free | 0 |
| sequential | 2 | `LT0-h4P_xBY_06` | declined | 316MB free | 0 |
| sequential | 2 | `uZwKNtHx9FE_01` | declined | 316MB free | 0 |
| sequential | 2 | `fNT8a6e1gx8_08` | declined | 316MB free | 0 |
| sequential | 2 | `f75G_hZMSHs_04` | declined | 316MB free | 0 |
| sequential | 2 | `TXyNxVRRPt8_01` | declined | 316MB free | 0 |
| sequential | 2 | `wPJ_InXF_I8_06` | declined | 316MB free | 0 |
| sequential | 2 | `z3jAMn3xpoM_02` | declined | 316MB free | 0 |
| sequential | 2 | `fNT8a6e1gx8_06` | declined | 316MB free | 0 |
| sequential | 2 | `f75G_hZMSHs_02` | declined | 317MB free | 0 |
| sequential | 2 | `LT0-h4P_xBY_04` | declined | 317MB free | 0 |
| sequential | 2 | `WMoeSSvCyTU_03` | declined | 317MB free | 0 |
| sequential | 2 | `I976oI3s1jQ_02` | declined | 317MB free | 0 |
| sequential | 2 | `mNF1kg9azQw_03` | declined | 317MB free | 0 |
| sequential | 2 | `I976oI3s1jQ_01` | declined | 317MB free | 0 |
| sequential | 2 | `iMJ9CjBX7eo_02` | declined | 317MB free | 0 |
| sequential | 2 | `k5Gxbifw8s8_05` | declined | 317MB free | 0 |
| sequential | 2 | `J-0KHhPS_m4_03` | declined | 317MB free | 0 |
| sequential | 2 | `mNF1kg9azQw_07` | declined | 317MB free | 0 |
| sequential | 2 | `MJijm9kbdHA_01` | declined | 317MB free | 0 |
| sequential | 2 | `jsOSzFyx8RA_03` | declined | 317MB free | 0 |
| sequential | 3 | `oP2WdYlaflE_02` | declined | 292MB free | 0 |
| sequential | 3 | `MJijm9kbdHA_01` | declined | 292MB free | 0 |
| sequential | 3 | `5iRIeHAK9b0_02` | declined | 292MB free | 0 |
| sequential | 3 | `sP6B4dlSOoc_05` | declined | 292MB free | 0 |
| sequential | 3 | `J-0KHhPS_m4_03` | declined | 292MB free | 0 |
| sequential | 3 | `LT0-h4P_xBY_06` | declined | 292MB free | 0 |
| sequential | 3 | `oP2WdYlaflE_03` | declined | 292MB free | 0 |
| sequential | 3 | `YR7BS6l_95E_05` | declined | 292MB free | 0 |
| overlapped | 1 | `wPJ_InXF_I8_06` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `I976oI3s1jQ_01` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `jsOSzFyx8RA_03` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `fNT8a6e1gx8_06` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `J-0KHhPS_m4_03` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `FND2JfzngVY_06` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `I976oI3s1jQ_02` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `ZgvLTS2zJdo_02` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `YR7BS6l_95E_05` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `NBpYolVo5WQ_06` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `TXyNxVRRPt8_01` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `5iRIeHAK9b0_02` | declined | backlog 608 frames | 5 |
| overlapped | 1 | `sP6B4dlSOoc_04` | declined | backlog 484 frames | 4 |
| overlapped | 1 | `z3jAMn3xpoM_02` | declined | backlog 484 frames | 4 |
| overlapped | 1 | `uZwKNtHx9FE_07` | declined | backlog 484 frames | 4 |
| overlapped | 1 | `D9HR4q0vbwE_05` | declined | backlog 484 frames | 4 |
| overlapped | 1 | `f75G_hZMSHs_04` | declined | backlog 484 frames | 4 |
| overlapped | 1 | `f75G_hZMSHs_02` | declined | backlog 484 frames | 4 |
| overlapped | 1 | `ZgvLTS2zJdo_01` | declined | backlog 484 frames | 4 |
| overlapped | 1 | `LT0-h4P_xBY_04` | declined | backlog 484 frames | 4 |
| overlapped | 1 | `oP2WdYlaflE_02` | declined | backlog 629 frames | 4 |
| overlapped | 1 | `LT0-h4P_xBY_06` | declined | backlog 629 frames | 4 |
| overlapped | 1 | `mNF1kg9azQw_07` | declined | backlog 629 frames | 4 |
| overlapped | 1 | `TXyNxVRRPt8_03` | declined | backlog 629 frames | 4 |
| overlapped | 1 | `wH3lJm0Typ0_03` | declined | backlog 629 frames | 4 |
| overlapped | 1 | `oP2WdYlaflE_03` | declined | backlog 629 frames | 4 |
| overlapped | 1 | `I976oI3s1jQ_05` | declined | backlog 629 frames | 4 |
| overlapped | 1 | `k5Gxbifw8s8_03` | declined | backlog 629 frames | 4 |
| overlapped | 1 | `rSiciLyYOyI_07` | declined | backlog 629 frames | 4 |
| overlapped | 2 | `uZwKNtHx9FE_07` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `wPJ_InXF_I8_02` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `5iRIeHAK9b0_03` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `TXyNxVRRPt8_03` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `k5Gxbifw8s8_03` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `FND2JfzngVY_06` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `wH3lJm0Typ0_03` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `I976oI3s1jQ_05` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `NBpYolVo5WQ_06` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `ZgvLTS2zJdo_02` | declined | backlog 489 frames | 3 |
| overlapped | 2 | `D9HR4q0vbwE_05` | declined | 275MB free | 2 |
| overlapped | 2 | `oP2WdYlaflE_02` | declined | 273MB free | 2 |
| overlapped | 2 | `uZwKNtHx9FE_01` | declined | backlog 491 frames | 3 |
| overlapped | 2 | `wPJ_InXF_I8_06` | declined | 244MB free | 4 |
| overlapped | 2 | `z3jAMn3xpoM_02` | declined | 244MB free | 4 |
| overlapped | 2 | `fNT8a6e1gx8_06` | declined | 244MB free | 4 |
| overlapped | 2 | `f75G_hZMSHs_02` | declined | 243MB free | 4 |
| overlapped | 2 | `LT0-h4P_xBY_04` | declined | 243MB free | 4 |
| overlapped | 2 | `WMoeSSvCyTU_03` | declined | 243MB free | 4 |
| overlapped | 2 | `I976oI3s1jQ_02` | declined | 243MB free | 4 |
| overlapped | 2 | `I976oI3s1jQ_01` | declined | 242MB free | 5 |
| overlapped | 2 | `iMJ9CjBX7eo_02` | declined | 275MB free | 5 |
| overlapped | 2 | `k5Gxbifw8s8_05` | declined | 274MB free | 5 |
| overlapped | 2 | `J-0KHhPS_m4_03` | declined | 296MB free | 5 |
| overlapped | 2 | `mNF1kg9azQw_07` | declined | 322MB free | 5 |
| overlapped | 2 | `MJijm9kbdHA_01` | declined | 322MB free | 5 |
| overlapped | 2 | `jsOSzFyx8RA_03` | declined | 322MB free | 5 |
| overlapped | 3 | `uZwKNtHx9FE_01` | declined | backlog 460 frames | 6 |
| overlapped | 3 | `D9HR4q0vbwE_05` | declined | backlog 460 frames | 6 |
| overlapped | 3 | `k5Gxbifw8s8_03` | declined | backlog 460 frames | 6 |
| overlapped | 3 | `iMJ9CjBX7eo_02` | declined | backlog 460 frames | 6 |
| overlapped | 3 | `WMoeSSvCyTU_03` | declined | backlog 472 frames | 7 |
| overlapped | 3 | `fNT8a6e1gx8_08` | declined | backlog 472 frames | 7 |
| overlapped | 3 | `ZgvLTS2zJdo_02` | declined | backlog 472 frames | 7 |
| overlapped | 3 | `wH3lJm0Typ0_03` | declined | backlog 472 frames | 7 |
| overlapped | 3 | `jsOSzFyx8RA_03` | declined | backlog 552 frames | 7 |
| overlapped | 3 | `k5Gxbifw8s8_05` | declined | backlog 552 frames | 7 |
| overlapped | 3 | `I976oI3s1jQ_02` | declined | backlog 552 frames | 7 |
| overlapped | 3 | `mNF1kg9azQw_07` | declined | backlog 552 frames | 7 |
| overlapped | 3 | `wPJ_InXF_I8_02` | declined | backlog 552 frames | 7 |
| overlapped | 3 | `TXyNxVRRPt8_01` | declined | backlog 552 frames | 7 |
| overlapped | 3 | `sP6B4dlSOoc_04` | declined | backlog 552 frames | 7 |
| overlapped | 3 | `fNT8a6e1gx8_06` | declined | backlog 478 frames | 6 |
| overlapped | 3 | `TXyNxVRRPt8_03` | declined | backlog 478 frames | 6 |
| overlapped | 3 | `z3jAMn3xpoM_02` | declined | backlog 478 frames | 6 |
| overlapped | 3 | `f75G_hZMSHs_04` | declined | backlog 478 frames | 6 |
| overlapped | 3 | `oP2WdYlaflE_02` | declined | backlog 478 frames | 6 |
| overlapped | 3 | `MJijm9kbdHA_01` | declined | backlog 478 frames | 6 |
| overlapped | 3 | `5iRIeHAK9b0_02` | declined | backlog 478 frames | 6 |
| overlapped | 3 | `sP6B4dlSOoc_05` | declined | backlog 478 frames | 6 |
| overlapped | 3 | `LT0-h4P_xBY_06` | declined | backlog 512 frames | 6 |
| overlapped | 3 | `oP2WdYlaflE_03` | declined | backlog 512 frames | 6 |
| overlapped | 3 | `YR7BS6l_95E_05` | declined | backlog 512 frames | 6 |

---

## 16. Disagreements with the project log (request G)

Every number measured here that contradicts `docs/01_PROJECT_COMPLETE_REFERENCE.md`, with both
values. All six have been corrected in that document, in `02_QA_BANK.md`, in `03_PAPER_KIT.md`
and in the LaTeX draft (which still builds: 14 pages, zero undefined citations or references,
zero overfull boxes).

| claim in the project log | measured here | verdict |
|---|---|---|
| "live post-cut latency **12–24 s**" (§1.3, §11.4, Phase 11) | Sequential **median 9.33 s, max 19.40 s**; 100 % ≤ 24 s but only 71 % ≤ 12 s. Overlapped **median 56.78 s, max 97.61 s** | **Describes neither case.** The sequential median sits *below* the range; the overlapped distribution sits far above it |
| "**the drain dominates**" (§1.3) | Sequential: decode **54.5 %**, drain **40.6 %** | **Reversed.** True only of the overlapped case, where queue wait dominates both at 60.9 % |
| "**31.3 ms** per decode step" (Phase 9) | **77.81 ms/step** live (R² 0.94, n = 126); **60.02 ms/step** standalone at the shipping flag | **Explained, not contradicted.** 31.3 ms was measured with the allocator flag unset; shipping sets it to 1 |
| "perception needs ~**6×** realtime and gets ~**3.5×**" (Phase 11, §16.3) | Sustained **11.75 fps** against the **15 fps** required: **1.35× short** | **Retired.** Neither figure reproduces under any constructible definition, and the 6× derivation could not be reconstructed |
| "parallel perception is **2.53×** on the live path" (Phase 7, Step 28) | **1.44–1.54×** on post-cut latency, 1.73–1.76× on the drain, 1.54–1.65× on perception fps | **Does not reproduce.** The *standalone* 1.66× does |
| "removing the caching flag costs **2.56×**" (§13.7) | **1.54×** (38.92 → 60.02 ms/step) | **Moved.** The rejection stands on the NVML assert instead |
| `MIN_AVAILABLE_MB=350` as the memory guard (§8.2) | Did not fire on any of 3 OOMs, which occurred at >1 GB available | **Wrong gauge.** Fragmentation binds, not free megabytes |
| "the landmarks are identical, not approximated" (§6.2, `streaming_perception.py`) | True at 1 worker (0 of 40 differ); **false at the shipping 2 workers** (26 of 40 differ) | **Scoped.** Corrected in code as commit `e3b358e` |
| "runs fully on-device with no network dependency" (resume bullet, `02_QA_BANK.md`) | `torch.hub` fetches the DINOv2 definition from GitHub at every launch | **False at startup.** Inference is local; startup is not |

Two items that were checked and **stand unchanged**, recorded so they are not re-litigated:

- **Decode is ~100 % CPU dispatch** with a 0.01 ms GPU tail. Only the 31.30 ms figure attached
  to it needed context; the dispatch-bound finding itself is untouched and is now load-bearing
  for the paper's structural argument (§8.1).
- **Output is deterministic at a fixed configuration.** 36 clips across 3 passes, zero
  differences, plus a control run agreeing on 27 of 27 (§13.1).

---

## 17. What could not be measured, and why

| requested | status |
|---|---|
| 10 live-camera utterances (request E) | **Withdrawn by the requester mid-session.** Not measured |
| The head/tail trim's contribution | **Not measurable on a replay** — it is driven by live motion scores in the camera loop, and OpenASL clips carry no dead air (§2.1). On live manual clips the project log reports it removing ~23 % of frames, which would reduce these latencies |
| A thermal throttle flag | **`tegrastats` exposes none.** The full clock, temperature and power record is given instead (§9) |
| `shubert_s` / `byt5_encoder_s` GPU tails | Hooks time CPU-side wall at module boundaries with no mid-`generate` `cuda.synchronize()`, since inserting one would itself change timing. Right instrument for decode; may under-attribute GPU work for the encoder split. Both terms are ≤ 0.26 s mean, so nothing here rests on them |
| Per-stage internals of the deferred path | Perception, cropping and DINOv2 run inside `process_frames` there, which was not instrumented. Reported as `unattributed` rather than split by guess (§7.2) |
| The cause of the within-process memory degradation | **Characterised (§15.2), not diagnosed.** Still open |
| Whether CPU contention affects decode | **Not isolated.** §14.1 shows no contention term is needed to explain the numbers, but no experiment was run that would detect a small one |

---

## 18. Everything else done in this session, in order

Context for whoever picks this up: the measurement was the main task, but the session began
with a state assessment and ended with a documentation pass, and both produced things that
matter to the paper.

**1. State assessment at session start.** The board had just booted after being off since
2026-09-20. Four things surfaced that were not in the project log:

- **An 11th power cut occurred 2026-09-20 18:31:11**, after 8 h 38 m of uptime — the longest
  witness-measured uptime before a cut. Telemetry at the last sample: 5.08 W, Tj 49.3 °C,
  over-current counters flat, `lfb 30x4MB`, 4765 MB available. Idle and healthy on every
  gauge, consistent with the existing power-delivery-fault diagnosis. **It is unrecorded
  whether this was the fault or a deliberate power-off**, and it should be resolved before the
  cut count is quoted.
- **The missing-RTC artifact is worse than documented.** Because the clock is restored
  pre-NTP, today's first 26 witness rows carried timestamps from 2026-09-20, and
  `boot_ledger.csv`'s last row — dated `2026-09-20T18:28:40` — is in fact *this* boot. One
  witness CSV can therefore hold rows that appear to be from different days.
  `crash_report.py` handles it correctly (it reads `uptime_s`), but the raw files mislead.
- **The camera had no `/dev/video*` node.** The ConferenceCam was enumerated on USB but
  `uvcvideo` was not loaded. Re-plugging the device made udev load it; no root was needed.
  Without it `live_worker_probe.py` exits at startup by design, since holding the camera and
  preview window on shared memory is the entire reason that probe exists separately from the
  evaluation harness.
- `transcripts/` still does not exist, so the transcript autosave shipped 2026-08-26 has still
  never run in a live session. The replay harness does not exercise it either.

**2. Design decisions taken before measuring**, both referred to the requester:

- **The overlapped condition was shaped to use the shipping admission guard** rather than
  feeding 40 clips with no limit, which would have OOM'd by construction and measured a
  condition the live application cannot produce (§4.2).
- **Clocks were left dynamic** rather than pinned with `jetson_clocks`. That is the state the
  board boots into and what live sessions have actually run under; pinning also raises
  sustained draw on a board with an open power-delivery fault. Under load the governor sat at
  maximum anyway (§9).

**3. The measurement**, §4–§13. 44 minutes of runtime, 240 attempted utterances.

**4. The confirming tests**, §14. Two configuration changes, each approved first.

**5. A documentation pass across the project's doc set** (all of it gitignored and local):

- `PROJECT_CONTEXT.md` — a new dated session block in the project's house style with nine
  `RESULT` subsections and a `WHAT TO DO NEXT` list; the previous entry-point block demoted;
  the file header rewritten.
- `01_PROJECT_COMPLETE_REFERENCE.md` — 11 edits: a new "Corrections of 2026-10-09" section
  after the headline table, the headline table itself, §6.2, §11.4, §13.7, §16.2 (where the
  "unexplained CUDA OOM" item became the fragmentation diagnosis), §16.3, Phase 11, Step 28,
  and a new §2.2 gotcha for the offline cold-start.
- `02_QA_BANK.md` — 20 edits, including a "numbers that changed, relearn these five" table at
  the top, since these answers are meant to be spoken from memory. Rewrote A5, A13, A14, B5,
  B9, C6, C11, D5 (whose title was literally *"Your latency is 12–24 seconds"*), the
  numbers-to-memorise table and all three resume-bullet variants.
- `03_PAPER_KIT.md` — 12 edits across the claim inventory: seven claims added to "fully
  supported", the 2.53× moved out of it, "perception needs 6× realtime" retired outright, and
  three new entries under "must not claim".
- `paper/asl_edge_slt.tex` — 7 edits: abstract, contributions, waterfall caption, the
  streaming-equivalence paragraph, parallel perception, and a rewritten *"The remaining gap is
  structural"* paragraph that now runs through decode rather than perception. **Rebuilt
  successfully: 14 pages, 0 undefined citations or references, 0 overfull boxes.**
- **Found while doing it: the paper kit's inline copy of the LaTeX is stale and will not
  compile.** It still contains the 13 `\num{}` calls that the 2026-09-20 compile fix stripped
  from the canonical file, including `\num{byte-identical}`, so it fails with `Missing $
  inserted`. The kit claims it is "reproduced in full below so this document stands alone",
  which is not true. A warning was added; the two copies are otherwise not in sync and
  `paper/asl_edge_slt.tex` is the only build target.

**6. Code changes — documentation only, and the only code touched all session.** Commit
`e3b358e` scoped the landmark-equivalence claim in `streaming_perception.py` (module
docstring, the `PERCEPTION_WORKERS` comment block, and `_emit_in_order`'s docstring) and the
identical sentence in `auto_segment_v5.py`'s `STREAM_PERCEPTION` comment. Verified
documentation-only by comparing the Python AST with docstrings stripped against the previous
commit — identical for both files.

**7. Commits, both pushed to `origin/main`.** Identity verified as `brianf203` before pushing.

```
952aff1  Add the per-utterance latency harness and its 240-utterance results   (41 files, +15566)
e3b358e  Scope the landmark-equivalence claim to one perception worker          (2 files, +46/-7)
```

`docs/` and `PROJECT_CONTEXT.md` remain gitignored, so the corrected documentation set and the
session block are local only.

---

## 19. Open items this report leaves behind

1. **The power-delivery fault** remains the project's #1 blocker and is physical. Resolve
   whether the 2026-09-20 18:31 shutdown was the fault or deliberate.
2. **The within-process memory degradation (§15.2) is undiagnosed** and has a user-visible
   consequence: long sessions stop accepting utterances.
3. **Whether `PERCEPTION_WORKERS=2` should stay the default (§14.2)** is a judgement call, not
   a bug: a 1.5× latency win against output that no single-worker run reproduces, at a measured
   corpus-BLEU cost of 0.03.
4. **Whether to fix the offline cold-start (§14.4)** or state it as a limitation.
5. **Whether a fragmentation-aware admission guard is safe to add (§15.3).**
6. The paper kit's inline LaTeX copy should be regenerated from the canonical file.

---

## 20. Artifacts

All paths relative to `shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText/`. Everything listed is
committed as of `952aff1`.

```
latency_runs/REPORT.md                  this document
latency_runs/clip_set.csv               the 40 clips with measured durations
latency_runs/env.txt                    environment of record, 20:57:01
latency_runs/sequential.csv             120 rows x 38 columns  (Appendix B.1)
latency_runs/overlap.csv                120 rows x 38 columns  (Appendix B.2)
latency_runs/confirm_w1.csv             PERCEPTION_WORKERS=1, 40 rows
latency_runs/confirm_defer.csv          MAX_LIVE_STREAMS=0, 40 rows
latency_runs/confirm_control.csv        shipping config control, 40 rows
latency_runs/analysis.txt               full statistical output
latency_runs/latency_vs_duration.png    scatter + fits, both conditions
latency_runs/thermal.txt                per-pass tegrastats summary
latency_runs/tegrastats_*.log           raw 1 Hz telemetry, 8 files
latency_runs/run_*.log                  per-pass stdout, incl. the OOM and 504 tracebacks
latency_runs/{select_clips,latency_probe,instrument,analysis,parse_tegrastats}.py
latency_runs/{capture_env,run_all,run_confirm}.sh
```

Power-fault telemetry referenced in §15.3 lives outside the repository at
`~/oc_load_test/witness/witness_<bootid>.csv`, written at 1 Hz by `~/oc_load_test/crash_witness.py`.
---

# Appendix A — every utterance, readable

One row per attempted utterance, in the order it was fed. Blank cells are quantities that
do not exist for that row (a declined utterance was never recorded; a deferred one has no
drain). Times in seconds. `qd` = queue depth at cut, `avail` = MemAvailable at the cut in MB.

## A.1  Sequential — 120 attempted utterances

| pass | # | clip | dur | frames | **latency** | wait | drain | shub | enc | decode | bytes | steps | qd | avail | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | `uZwKNtHx9FE_01` | 14.47 | 174 | **14.60** | 0.00 | 2.65 | 0.29 | 0.68 | 10.80 | 125 | 127 | 1 | 1914 | ok, 1st after warmup |
| 1 | 1 | `5iRIeHAK9b0_03` | 4.70 | 71 | **7.53** | 0.00 | 3.76 | 0.09 | 0.20 | 3.39 | 47 | 49 | 1 | 1517 | ok |
| 1 | 2 | `fNT8a6e1gx8_08` | 3.60 | 54 | **7.83** | 0.00 | 3.32 | 0.05 | 0.14 | 4.20 | 58 | 60 | 1 | 1424 | ok |
| 1 | 3 | `sP6B4dlSOoc_05` | 10.98 | 165 | **12.95** | 0.00 | 4.86 | 0.10 | 0.40 | 7.36 | 95 | 97 | 1 | 1176 | ok |
| 1 | 4 | `mNF1kg9azQw_03` | 11.39 | 137 | **10.19** | 0.00 | 3.16 | 0.08 | 0.32 | 6.43 | 90 | 92 | 1 | 1021 | ok |
| 1 | 5 | `iMJ9CjBX7eo_02` | 4.30 | 65 | **9.92** | 0.01 | 3.87 | 0.07 | 0.18 | 5.70 | 77 | 79 | 1 | 1312 | ok |
| 1 | 6 | `k5Gxbifw8s8_05` | 4.43 | 67 | **6.43** | 0.00 | 3.44 | 0.08 | 0.20 | 2.61 | 35 | 37 | 1 | 937 | ok |
| 1 | 7 | `MJijm9kbdHA_01` | 14.50 | 174 | **10.94** | 0.00 | 2.05 | 0.08 | 0.40 | 8.14 | 100 | 102 | 1 | 695 | ok |
| 1 | 8 | `wPJ_InXF_I8_06` | 4.94 | 74 | **8.75** | 0.00 | 4.84 | 0.06 | 0.20 | 3.54 | 50 | 52 | 1 | 1012 | ok |
| 1 | 9 | `I976oI3s1jQ_01` | 7.00 | 84 | **9.21** | 0.00 | 2.66 | 0.07 | 0.19 | 6.12 | 84 | 86 | 1 | 550 | ok |
| 1 | 10 | `jsOSzFyx8RA_03` | 12.71 | 191 | **19.32** | 0.01 | 5.17 | 0.10 | 0.43 | 13.31 | 153 | 155 | 1 | 457 | ok |
| 1 | 11 | `fNT8a6e1gx8_06` | 2.77 | 42 | **7.44** | 0.00 | 4.35 | 0.07 | 0.17 | 2.82 | 38 | 40 | 1 | 831 | ok |
| 1 | 12 | `J-0KHhPS_m4_03` | 5.00 | 75 | **7.12** | 0.00 | 3.43 | 0.08 | 0.20 | 3.29 | 40 | 42 | 1 | 504 | ok |
| 1 | 13 | `FND2JfzngVY_06` | 8.01 | 120 | **9.72** | 0.00 | 4.71 | 0.08 | 0.22 | 4.55 | 53 | 55 | 1 | 484 | ok |
| 1 | 14 | `I976oI3s1jQ_02` | 13.50 | 162 | **11.58** | 0.00 | 2.15 | 0.09 | 0.36 | 8.76 | 117 | 119 | 1 | 664 | ok |
| 1 | 15 | `ZgvLTS2zJdo_02` | 14.72 | 177 | **8.79** | 0.00 | 2.33 | 0.11 | 0.40 | 5.66 | 56 | 58 | 1 | 454 | ok |
| 1 | 16 | `YR7BS6l_95E_05` | 11.60 | 174 | **13.99** | 0.00 | 5.25 | 0.09 | 0.40 | 8.01 | 97 | 99 | 1 | 403 | ok |
| 1 | 17 | `NBpYolVo5WQ_06` | 3.04 | 37 | **8.01** | 0.00 | 4.70 | 0.09 | 0.16 | 3.03 | 41 | 43 | 1 | 515 | ok |
| 1 | 18 | `TXyNxVRRPt8_01` | 6.27 | 94 | **10.59** | 0.00 | 4.01 | 0.06 | 0.19 | 6.16 | 86 | 88 | 1 | 448 | ok |
| 1 | 19 | `5iRIeHAK9b0_02` | 10.40 | 156 | **12.61** | 0.00 | 4.35 | 0.09 | 0.36 | 7.62 | 93 | 95 | 1 | 633 | ok |
| 1 | 20 | `mNF1kg9azQw_05` | 10.30 | 124 | **7.86** | 0.00 | 2.72 | 0.08 | 0.22 | 4.61 | 65 | 67 | 1 | 436 | ok |
| 1 | 21 | `WMoeSSvCyTU_03` | 7.91 | 119 | **10.00** | 0.00 | 5.23 | 0.09 | 0.21 | 4.33 | 60 | 62 | 1 | 406 | ok |
| 1 | 22 | `sP6B4dlSOoc_04` | 5.57 | 84 | **7.96** | 0.01 | 3.49 | 0.06 | 0.18 | 4.07 | 44 | 46 | 1 | 332 | ok |
| 1 | 23 | `z3jAMn3xpoM_02` | 8.14 | 122 | **11.03** | 0.00 | 4.50 | 0.07 | 0.23 | 6.08 | 81 | 83 | 1 | 341 | ok |
| 1 | 24 | `uZwKNtHx9FE_07` | 12.47 | 150 | **12.72** | 0.00 | 2.74 | 0.09 | 0.37 | 9.29 | 128 | 130 | 1 | 376 | ok |
| 1 | 25 | `D9HR4q0vbwE_05` | 2.04 | 25 | **5.23** | 0.00 | 3.52 | 0.08 | 0.14 | 1.45 | 18 | 20 | 1 | 449 | ok |
| 1 | 26 | `f75G_hZMSHs_04` | 2.40 | 36 | **8.65** | 0.00 | 3.81 | 0.07 | 0.18 | 4.55 | 57 | 59 | 1 | 443 | ok |
| 1 | 27 | `f75G_hZMSHs_02` | 7.78 | 117 | **13.96** | 0.00 | 4.04 | 0.08 | 0.22 | 9.45 | 122 | 124 | 1 | 424 | ok |
| 1 | 28 | `ZgvLTS2zJdo_01` | 3.42 | 41 | **11.75** | 0.00 | 3.51 | 0.08 | 0.17 | 7.94 | 113 | 115 | 1 | 455 | ok |
| 1 | 29 | `LT0-h4P_xBY_04` | 7.77 | 117 | **13.47** | 0.00 | 4.04 | 0.09 | 0.23 | 8.91 | 123 | 125 | 1 | 293 | ok |
| 1 | 30 | `wPJ_InXF_I8_02` | 14.15 | 212 | **14.28** | 0.00 | 5.49 | 0.09 | 0.48 | 7.93 | 104 | 106 | 1 | 343 | ok |
| 1 | 31 | `oP2WdYlaflE_02` | 9.10 | 137 | **14.39** | 0.00 | 5.24 | 0.09 | 0.32 | 8.53 | 115 | 117 | 1 | 358 | ok |
| 1 | 32 | `LT0-h4P_xBY_06` | 8.40 | 126 | **9.27** | 0.00 | 4.19 | 0.07 | 0.23 | 4.63 | 51 | 53 | 1 | 283 | ok |
| 1 | 33 | `mNF1kg9azQw_07` | 9.38 | 113 | **11.27** | 0.00 | 2.08 | 0.08 | 0.20 | 8.63 | 120 | 122 | 1 | 448 | ok |
| 1 | 34 | `TXyNxVRRPt8_03` | 4.50 | 68 | **6.98** | 0.00 | 3.64 | 0.07 | 0.20 | 2.95 | 32 | 34 | 1 | 401 | ok |
| 1 | 35 | `wH3lJm0Typ0_03` | 9.01 | 135 | **13.60** | 0.00 | 4.77 | 0.09 | 0.32 | 8.22 | 100 | 102 | 1 | 672 | ok |
| 1 | 36 | `oP2WdYlaflE_03` | 13.93 | 209 | **17.89** | 0.00 | 5.31 | 0.10 | 0.46 | 11.74 | 148 | 150 | 1 | 380 | ok |
| 1 | 37 | `I976oI3s1jQ_05` | 3.42 | 41 | **7.39** | 0.00 | 3.34 | 0.06 | 0.15 | 3.74 | 49 | 51 | 1 | 674 | ok |
| 1 | 38 | `k5Gxbifw8s8_03` | 4.70 | 71 | **8.27** | 0.01 | 3.99 | 0.06 | 0.18 | 3.86 | 48 | 50 | 1 | 702 | ok |
| 1 | 39 | `rSiciLyYOyI_07` | 2.88 | 35 | **5.32** | 0.00 | 3.69 | 0.08 | 0.17 | 1.33 | 17 | 19 | 1 | 835 | ok |
| 2 | 0 | `sP6B4dlSOoc_04` | 5.57 | 84 | **9.13** | 0.00 | 3.97 | 0.27 | 0.41 | 4.38 | 44 | 46 | 1 | 2739 | ok, 1st after warmup |
| 2 | 1 | `rSiciLyYOyI_07` | 2.88 | 35 | **5.94** | 0.00 | 4.35 | 0.08 | 0.16 | 1.32 | 17 | 19 | 1 | 2009 | ok |
| 2 | 2 | `YR7BS6l_95E_05` | 11.60 | 174 | **13.80** | 0.00 | 4.94 | 0.09 | 0.41 | 8.11 | 97 | 99 | 1 | 1573 | ok |
| 2 | 3 | `ZgvLTS2zJdo_01` | 3.42 | 41 | **11.88** | 0.00 | 3.64 | 0.08 | 0.15 | 7.93 | 113 | 115 | 1 | 1665 | ok |
| 2 | 4 | `sP6B4dlSOoc_05` | 10.98 | 165 | **12.83** | 0.00 | 4.75 | 0.09 | 0.42 | 7.33 | 95 | 97 | 1 | 1298 | ok |
| 2 | 5 | `mNF1kg9azQw_05` | 10.30 | 124 | **8.19** | 0.00 | 3.15 | 0.08 | 0.21 | 4.59 | 65 | 67 | 1 | 1176 | ok |
| 2 | 6 | `oP2WdYlaflE_03` | 13.93 | 209 | **16.89** | 0.00 | 4.20 | 0.11 | 0.43 | 11.84 | 148 | 150 | 1 | 1076 | ok |
| 2 | 7 | `5iRIeHAK9b0_02` | 10.40 | 156 | **12.62** | 0.00 | 4.48 | 0.09 | 0.36 | 7.51 | 93 | 95 | 1 | 799 | ok |
| 2 | 8 | `uZwKNtHx9FE_07` | 12.47 | 150 | **12.81** | 0.00 | 2.79 | 0.09 | 0.35 | 9.35 | 128 | 130 | 1 | 521 | ok |
| 2 | 9 | `wPJ_InXF_I8_02` | 14.15 | 212 | **14.37** | 0.00 | 5.54 | 0.10 | 0.47 | 7.95 | 104 | 106 | 1 | 413 | ok |
| 2 | 10 | `5iRIeHAK9b0_03` | 4.70 | 71 | **7.67** | 0.00 | 3.98 | 0.07 | 0.19 | 3.34 | 47 | 49 | 1 | 976 | ok |
| 2 | 11 | `TXyNxVRRPt8_03` | 4.50 | 68 | **7.31** | 0.00 | 4.03 | 0.07 | 0.20 | 2.95 | 32 | 34 | 1 | 828 | ok |
| 2 | 12 | `k5Gxbifw8s8_03` | 4.70 | 71 | **8.34** | 0.00 | 4.06 | 0.07 | 0.18 | 3.88 | 48 | 50 | 1 | 716 | ok |
| 2 | 13 | `FND2JfzngVY_06` | 8.01 | 120 | **9.03** | 0.00 | 4.04 | 0.07 | 0.21 | 4.54 | 53 | 55 | 1 | 450 | ok |
| 2 | 14 | `wH3lJm0Typ0_03` | 9.01 | 135 | **13.52** | 0.00 | 4.75 | 0.09 | 0.33 | 8.17 | 100 | 102 | 1 | 601 | ok |
| 2 | 15 | `I976oI3s1jQ_05` | 3.42 | 41 | **7.27** | 0.00 | 3.24 | 0.07 | 0.16 | 3.76 | 49 | 51 | 1 | 279 | ok |
| 2 | 16 | `NBpYolVo5WQ_06` | 3.04 | 37 | **7.58** | 0.00 | 4.28 | 0.08 | 0.18 | 3.00 | 41 | 43 | 1 | 462 | ok |
| 2 | 17 | `ZgvLTS2zJdo_02` | 14.72 | 177 | **8.80** | 0.00 | 2.42 | 0.10 | 0.41 | 5.62 | 56 | 58 | 1 | 375 | ok |
| 2 | 18 | `D9HR4q0vbwE_05` | 2.04 | 25 | **5.23** | 0.00 | 3.52 | 0.07 | 0.16 | 1.45 | 18 | 20 | 1 | 300 | ok |
| 2 | 19 | `oP2WdYlaflE_02` | 9.10 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 20 | `LT0-h4P_xBY_06` | 8.40 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 21 | `uZwKNtHx9FE_01` | 14.47 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 22 | `fNT8a6e1gx8_08` | 3.60 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 23 | `f75G_hZMSHs_04` | 2.40 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 24 | `TXyNxVRRPt8_01` | 6.27 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 25 | `wPJ_InXF_I8_06` | 4.94 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 26 | `z3jAMn3xpoM_02` | 8.14 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 27 | `fNT8a6e1gx8_06` | 2.77 |  | **** |  |  |  |  |  |  |  | 0 | 316 | DECLINED — 316MB free |
| 2 | 28 | `f75G_hZMSHs_02` | 7.78 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 29 | `LT0-h4P_xBY_04` | 7.77 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 30 | `WMoeSSvCyTU_03` | 7.91 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 31 | `I976oI3s1jQ_02` | 13.50 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 32 | `mNF1kg9azQw_03` | 11.39 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 33 | `I976oI3s1jQ_01` | 7.00 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 34 | `iMJ9CjBX7eo_02` | 4.30 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 35 | `k5Gxbifw8s8_05` | 4.43 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 36 | `J-0KHhPS_m4_03` | 5.00 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 37 | `mNF1kg9azQw_07` | 9.38 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 38 | `MJijm9kbdHA_01` | 14.50 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 2 | 39 | `jsOSzFyx8RA_03` | 12.71 |  | **** |  |  |  |  |  |  |  | 0 | 317 | DECLINED — 317MB free |
| 3 | 0 | `mNF1kg9azQw_03` | 11.39 | 137 | **10.94** | 0.00 | 3.21 | 0.35 | 0.54 | 6.62 | 90 | 92 | 1 | 2138 | ok, 1st after warmup |
| 3 | 1 | `I976oI3s1jQ_01` | 7.00 | 84 | **9.03** | 0.00 | 2.50 | 0.08 | 0.19 | 6.09 | 84 | 86 | 1 | 1679 | ok |
| 3 | 2 | `5iRIeHAK9b0_03` | 4.70 | 71 | **7.59** | 0.00 | 3.87 | 0.07 | 0.18 | 3.36 | 47 | 49 | 1 | 1855 | ok |
| 3 | 3 | `uZwKNtHx9FE_07` | 12.47 | 150 | **12.77** | 0.00 | 2.75 | 0.09 | 0.35 | 9.39 | 128 | 130 | 1 | 1405 | ok |
| 3 | 4 | `f75G_hZMSHs_02` | 7.78 | 117 | **14.05** | 0.00 | 4.11 | 0.08 | 0.22 | 9.48 | 122 | 124 | 1 | 1132 | ok |
| 3 | 5 | `NBpYolVo5WQ_06` | 3.04 | 37 | **7.67** | 0.00 | 4.36 | 0.08 | 0.16 | 3.02 | 41 | 43 | 1 | 1448 | ok |
| 3 | 6 | `wPJ_InXF_I8_06` | 4.94 | 74 | **8.68** | 0.00 | 4.67 | 0.07 | 0.20 | 3.61 | 50 | 52 | 1 | 1334 | ok |
| 3 | 7 | `I976oI3s1jQ_05` | 3.42 | 41 | **7.28** | 0.00 | 3.24 | 0.06 | 0.14 | 3.74 | 49 | 51 | 1 | 1115 | ok |
| 3 | 8 | `ZgvLTS2zJdo_01` | 3.42 | 41 | **11.87** | 0.00 | 3.62 | 0.08 | 0.18 | 7.93 | 113 | 115 | 1 | 1250 | ok |
| 3 | 9 | `uZwKNtHx9FE_01` | 14.47 | 174 | **13.28** | 0.00 | 2.24 | 0.09 | 0.41 | 10.21 | 125 | 127 | 1 | 945 | ok |
| 3 | 10 | `D9HR4q0vbwE_05` | 2.04 | 25 | **5.11** | 0.00 | 3.44 | 0.07 | 0.14 | 1.44 | 18 | 20 | 1 | 520 | ok |
| 3 | 11 | `k5Gxbifw8s8_03` | 4.70 | 71 | **8.46** | 0.00 | 4.18 | 0.08 | 0.20 | 3.88 | 48 | 50 | 1 | 1012 | ok |
| 3 | 12 | `iMJ9CjBX7eo_02` | 4.30 | 65 | **9.61** | 0.00 | 3.49 | 0.08 | 0.20 | 5.72 | 77 | 79 | 1 | 549 | ok |
| 3 | 13 | `mNF1kg9azQw_05` | 10.30 | 124 | **7.83** | 0.00 | 2.71 | 0.07 | 0.21 | 4.67 | 65 | 67 | 1 | 683 | ok |
| 3 | 14 | `rSiciLyYOyI_07` | 2.88 | 35 | **5.49** | 0.00 | 3.87 | 0.08 | 0.16 | 1.36 | 17 | 19 | 1 | 784 | ok |
| 3 | 15 | `FND2JfzngVY_06` | 8.01 | 120 | **9.57** | 0.00 | 4.47 | 0.08 | 0.22 | 4.58 | 53 | 55 | 1 | 926 | ok |
| 3 | 16 | `WMoeSSvCyTU_03` | 7.91 | 119 | **9.28** | 0.00 | 4.55 | 0.07 | 0.22 | 4.30 | 60 | 62 | 1 | 582 | ok |
| 3 | 17 | `fNT8a6e1gx8_08` | 3.60 | 54 | **7.75** | 0.00 | 3.32 | 0.06 | 0.17 | 4.13 | 58 | 60 | 1 | 913 | ok |
| 3 | 18 | `ZgvLTS2zJdo_02` | 14.72 | 177 | **9.33** | 0.00 | 2.87 | 0.11 | 0.41 | 5.68 | 56 | 58 | 1 | 338 | ok |
| 3 | 19 | `wH3lJm0Typ0_03` | 9.01 | 135 | **13.82** | 0.00 | 5.00 | 0.09 | 0.33 | 8.22 | 100 | 102 | 1 | 300 | ok |
| 3 | 20 | `LT0-h4P_xBY_04` | 7.77 | 117 | **13.27** | 0.00 | 4.11 | 0.08 | 0.21 | 8.72 | 123 | 125 | 1 | 411 | ok |
| 3 | 21 | `jsOSzFyx8RA_03` | 12.71 | 191 | **19.40** | 0.00 | 5.28 | 0.10 | 0.41 | 13.39 | 153 | 155 | 1 | 314 | ok |
| 3 | 22 | `k5Gxbifw8s8_05` | 4.43 | 67 | **6.73** | 0.00 | 3.77 | 0.07 | 0.19 | 2.59 | 35 | 37 | 1 | 360 | ok |
| 3 | 23 | `I976oI3s1jQ_02` | 13.50 | 162 | **11.90** | 0.00 | 2.42 | 0.08 | 0.38 | 8.71 | 117 | 119 | 1 | 320 | ok |
| 3 | 24 | `mNF1kg9azQw_07` | 9.38 | 113 | **11.67** | 0.00 | 2.65 | 0.08 | 0.20 | 8.59 | 120 | 122 | 1 | 333 | ok |
| 3 | 25 | `wPJ_InXF_I8_02` | 14.15 | 212 | **14.36** | 0.01 | 5.64 | 0.09 | 0.48 | 7.89 | 104 | 106 | 1 | 522 | ok |
| 3 | 26 | `TXyNxVRRPt8_01` | 6.27 | 94 | **10.23** | 0.00 | 3.74 | 0.07 | 0.19 | 6.06 | 86 | 88 | 1 | 333 | ok |
| 3 | 27 | `sP6B4dlSOoc_04` | 5.57 | 84 | **8.17** | 0.00 | 3.72 | 0.07 | 0.18 | 4.08 | 44 | 46 | 1 | 757 | ok |
| 3 | 28 | `fNT8a6e1gx8_06` | 2.77 | 42 | **7.27** | 0.00 | 4.25 | 0.07 | 0.18 | 2.74 | 38 | 40 | 1 | 998 | ok |
| 3 | 29 | `TXyNxVRRPt8_03` | 4.50 | 68 | **7.55** | 0.00 | 4.19 | 0.07 | 0.20 | 2.99 | 32 | 34 | 1 | 781 | ok |
| 3 | 30 | `z3jAMn3xpoM_02` | 8.14 | 122 | **10.59** | 0.00 | 4.06 | 0.08 | 0.22 | 6.09 | 81 | 83 | 1 | 688 | ok |
| 3 | 31 | `f75G_hZMSHs_04` | 2.40 | 36 | **8.76** | 0.00 | 3.93 | 0.07 | 0.17 | 4.54 | 57 | 59 | 1 | 381 | ok |
| 3 | 32 | `oP2WdYlaflE_02` | 9.10 |  | **** |  |  |  |  |  |  |  | 0 | 292 | DECLINED — 292MB free |
| 3 | 33 | `MJijm9kbdHA_01` | 14.50 |  | **** |  |  |  |  |  |  |  | 0 | 292 | DECLINED — 292MB free |
| 3 | 34 | `5iRIeHAK9b0_02` | 10.40 |  | **** |  |  |  |  |  |  |  | 0 | 292 | DECLINED — 292MB free |
| 3 | 35 | `sP6B4dlSOoc_05` | 10.98 |  | **** |  |  |  |  |  |  |  | 0 | 292 | DECLINED — 292MB free |
| 3 | 36 | `J-0KHhPS_m4_03` | 5.00 |  | **** |  |  |  |  |  |  |  | 0 | 292 | DECLINED — 292MB free |
| 3 | 37 | `LT0-h4P_xBY_06` | 8.40 |  | **** |  |  |  |  |  |  |  | 0 | 292 | DECLINED — 292MB free |
| 3 | 38 | `oP2WdYlaflE_03` | 13.93 |  | **** |  |  |  |  |  |  |  | 0 | 292 | DECLINED — 292MB free |
| 3 | 39 | `YR7BS6l_95E_05` | 11.60 |  | **** |  |  |  |  |  |  |  | 0 | 292 | DECLINED — 292MB free |

## A.2  Overlapped — 120 attempted utterances

| pass | # | clip | dur | frames | **latency** | wait | drain | shub | enc | decode | bytes | steps | qd | avail | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | `uZwKNtHx9FE_01` | 14.47 | 174 | **15.99** | 0.00 | 2.67 | 0.35 | 0.86 | 11.89 | 125 | 127 | 1 | 1958 | ok, 1st after warmup |
| 1 | 1 | `5iRIeHAK9b0_03` | 4.70 | 71 | **17.85** | 9.64 | 4.41 | 0.08 | 0.20 | 3.44 | 47 | 49 | 2 | 1537 | ok |
| 1 | 2 | `fNT8a6e1gx8_08` | 3.60 | 54 | **34.43** | 12.80 |  | 0.15 | 0.23 | 6.27 | 53 | 55 | 3 | 1533 | deferred |
| 1 | 3 | `sP6B4dlSOoc_05` | 10.98 | 165 | **60.30** | 21.97 |  | 0.10 | 0.35 | 7.02 | 95 | 97 | 4 | 1530 | deferred |
| 1 | 4 | `mNF1kg9azQw_03` | 11.39 | 137 | **55.22** | 47.29 | 0.77 | 0.08 | 0.32 | 6.58 | 90 | 92 | 3 | 884 | ok |
| 1 | 5 | `iMJ9CjBX7eo_02` | 4.30 | 65 | **56.78** | 49.31 | 0.17 | 0.07 | 0.18 | 6.94 | 77 | 79 | 4 | 1031 | ok |
| 1 | 6 | `k5Gxbifw8s8_05` | 4.43 | 67 | **68.20** | 50.91 |  | 0.07 | 0.19 | 3.65 | 33 | 35 | 5 | 650 | deferred |
| 1 | 7 | `MJijm9kbdHA_01` | 14.50 | 174 | **93.48** | 52.23 |  | 0.11 | 0.44 | 8.15 | 100 | 102 | 5 | 841 | deferred |
| 1 | 8 | `wPJ_InXF_I8_06` | 4.94 |  | **** |  |  |  |  |  |  |  | 5 | 853 | DECLINED — backlog 608 frames |
| 1 | 9 | `I976oI3s1jQ_01` | 7.00 |  | **** |  |  |  |  |  |  |  | 5 | 721 | DECLINED — backlog 608 frames |
| 1 | 10 | `jsOSzFyx8RA_03` | 12.71 |  | **** |  |  |  |  |  |  |  | 5 | 545 | DECLINED — backlog 608 frames |
| 1 | 11 | `fNT8a6e1gx8_06` | 2.77 |  | **** |  |  |  |  |  |  |  | 5 | 581 | DECLINED — backlog 608 frames |
| 1 | 12 | `J-0KHhPS_m4_03` | 5.00 |  | **** |  |  |  |  |  |  |  | 5 | 581 | DECLINED — backlog 608 frames |
| 1 | 13 | `FND2JfzngVY_06` | 8.01 |  | **** |  |  |  |  |  |  |  | 5 | 581 | DECLINED — backlog 608 frames |
| 1 | 14 | `I976oI3s1jQ_02` | 13.50 |  | **** |  |  |  |  |  |  |  | 5 | 583 | DECLINED — backlog 608 frames |
| 1 | 15 | `ZgvLTS2zJdo_02` | 14.72 |  | **** |  |  |  |  |  |  |  | 5 | 692 | DECLINED — backlog 608 frames |
| 1 | 16 | `YR7BS6l_95E_05` | 11.60 |  | **** |  |  |  |  |  |  |  | 5 | 692 | DECLINED — backlog 608 frames |
| 1 | 17 | `NBpYolVo5WQ_06` | 3.04 |  | **** |  |  |  |  |  |  |  | 5 | 692 | DECLINED — backlog 608 frames |
| 1 | 18 | `TXyNxVRRPt8_01` | 6.27 |  | **** |  |  |  |  |  |  |  | 5 | 692 | DECLINED — backlog 608 frames |
| 1 | 19 | `5iRIeHAK9b0_02` | 10.40 |  | **** |  |  |  |  |  |  |  | 5 | 692 | DECLINED — backlog 608 frames |
| 1 | 20 | `mNF1kg9azQw_05` | 10.30 | 124 | **96.34** | 63.62 |  | 0.09 | 0.23 | 6.37 | 87 | 89 | 5 | 790 | deferred |
| 1 | 21 | `WMoeSSvCyTU_03` | 7.91 | 119 | **93.17** | 86.70 | 1.62 | 0.09 | 0.20 | 4.36 | 60 | 62 | 5 | 427 | ok |
| 1 | 22 | `sP6B4dlSOoc_04` | 5.57 |  | **** |  |  |  |  |  |  |  | 4 | 348 | DECLINED — backlog 484 frames |
| 1 | 23 | `z3jAMn3xpoM_02` | 8.14 |  | **** |  |  |  |  |  |  |  | 4 | 347 | DECLINED — backlog 484 frames |
| 1 | 24 | `uZwKNtHx9FE_07` | 12.47 |  | **** |  |  |  |  |  |  |  | 4 | 331 | DECLINED — backlog 484 frames |
| 1 | 25 | `D9HR4q0vbwE_05` | 2.04 |  | **** |  |  |  |  |  |  |  | 4 | 330 | DECLINED — backlog 484 frames |
| 1 | 26 | `f75G_hZMSHs_04` | 2.40 |  | **** |  |  |  |  |  |  |  | 4 | 330 | DECLINED — backlog 484 frames |
| 1 | 27 | `f75G_hZMSHs_02` | 7.78 |  | **** |  |  |  |  |  |  |  | 4 | 372 | DECLINED — backlog 484 frames |
| 1 | 28 | `ZgvLTS2zJdo_01` | 3.42 |  | **** |  |  |  |  |  |  |  | 4 | 372 | DECLINED — backlog 484 frames |
| 1 | 29 | `LT0-h4P_xBY_04` | 7.77 |  | **** |  |  |  |  |  |  |  | 4 | 372 | DECLINED — backlog 484 frames |
| 1 | 30 | `wPJ_InXF_I8_02` | 14.15 | 212 | **75.73** | 65.36 | 1.50 | 0.09 | 0.50 | 7.98 | 104 | 106 | 4 | 404 | ok |
| 1 | 31 | `oP2WdYlaflE_02` | 9.10 |  | **** |  |  |  |  |  |  |  | 4 | 404 | DECLINED — backlog 629 frames |
| 1 | 32 | `LT0-h4P_xBY_06` | 8.40 |  | **** |  |  |  |  |  |  |  | 4 | 335 | DECLINED — backlog 629 frames |
| 1 | 33 | `mNF1kg9azQw_07` | 9.38 |  | **** |  |  |  |  |  |  |  | 4 | 356 | DECLINED — backlog 629 frames |
| 1 | 34 | `TXyNxVRRPt8_03` | 4.50 |  | **** |  |  |  |  |  |  |  | 4 | 337 | DECLINED — backlog 629 frames |
| 1 | 35 | `wH3lJm0Typ0_03` | 9.01 |  | **** |  |  |  |  |  |  |  | 4 | 335 | DECLINED — backlog 629 frames |
| 1 | 36 | `oP2WdYlaflE_03` | 13.93 |  | **** |  |  |  |  |  |  |  | 4 | 335 | DECLINED — backlog 629 frames |
| 1 | 37 | `I976oI3s1jQ_05` | 3.42 |  | **** |  |  |  |  |  |  |  | 4 | 383 | DECLINED — backlog 629 frames |
| 1 | 38 | `k5Gxbifw8s8_03` | 4.70 |  | **** |  |  |  |  |  |  |  | 4 | 363 | DECLINED — backlog 629 frames |
| 1 | 39 | `rSiciLyYOyI_07` | 2.88 |  | **** |  |  |  |  |  |  |  | 4 | 363 | DECLINED — backlog 629 frames |
| 2 | 0 | `sP6B4dlSOoc_04` | 5.57 | 84 | **10.28** | 0.00 | 4.91 | 0.31 | 0.48 | 4.46 | 44 | 46 | 1 | 2744 | ok, 1st after warmup |
| 2 | 1 | `rSiciLyYOyI_07` | 2.88 | 35 | **9.52** | 5.78 | 2.11 | 0.09 | 0.16 | 1.34 | 17 | 19 | 2 | 2038 | ok |
| 2 | 2 | `YR7BS6l_95E_05` | 11.60 | 174 | **20.29** | 0.00 |  |  |  |  |  |  | 3 | 1911 | **CUDA OOM** |
| 2 | 3 | `ZgvLTS2zJdo_01` | 3.42 | 41 | **15.24** | 15.24 |  |  |  |  |  |  | 2 | 1420 | **CUDA OOM** |
| 2 | 4 | `sP6B4dlSOoc_05` | 10.98 | 165 | **2.65** | 2.64 |  |  |  |  |  |  | 3 | 1086 | **CUDA OOM** |
| 2 | 5 | `mNF1kg9azQw_05` | 10.30 | 124 | **43.77** | 0.00 |  | 0.15 | 0.24 | 6.45 | 87 | 89 | 4 | 1266 | deferred |
| 2 | 6 | `oP2WdYlaflE_03` | 13.93 | 209 | **43.03** | 28.23 | 2.16 | 0.10 | 0.47 | 11.81 | 148 | 150 | 2 | 531 | ok |
| 2 | 7 | `5iRIeHAK9b0_02` | 10.40 | 156 | **42.31** | 30.97 | 2.13 | 0.18 | 0.53 | 8.30 | 93 | 95 | 3 | 617 | ok |
| 2 | 8 | `uZwKNtHx9FE_07` | 12.47 |  | **** |  |  |  |  |  |  |  | 3 | 506 | DECLINED — backlog 489 frames |
| 2 | 9 | `wPJ_InXF_I8_02` | 14.15 |  | **** |  |  |  |  |  |  |  | 3 | 279 | DECLINED — backlog 489 frames |
| 2 | 10 | `5iRIeHAK9b0_03` | 4.70 |  | **** |  |  |  |  |  |  |  | 3 | 277 | DECLINED — backlog 489 frames |
| 2 | 11 | `TXyNxVRRPt8_03` | 4.50 |  | **** |  |  |  |  |  |  |  | 3 | 271 | DECLINED — backlog 489 frames |
| 2 | 12 | `k5Gxbifw8s8_03` | 4.70 |  | **** |  |  |  |  |  |  |  | 3 | 271 | DECLINED — backlog 489 frames |
| 2 | 13 | `FND2JfzngVY_06` | 8.01 |  | **** |  |  |  |  |  |  |  | 3 | 308 | DECLINED — backlog 489 frames |
| 2 | 14 | `wH3lJm0Typ0_03` | 9.01 |  | **** |  |  |  |  |  |  |  | 3 | 287 | DECLINED — backlog 489 frames |
| 2 | 15 | `I976oI3s1jQ_05` | 3.42 |  | **** |  |  |  |  |  |  |  | 3 | 287 | DECLINED — backlog 489 frames |
| 2 | 16 | `NBpYolVo5WQ_06` | 3.04 |  | **** |  |  |  |  |  |  |  | 3 | 287 | DECLINED — backlog 489 frames |
| 2 | 17 | `ZgvLTS2zJdo_02` | 14.72 |  | **** |  |  |  |  |  |  |  | 3 | 287 | DECLINED — backlog 489 frames |
| 2 | 18 | `D9HR4q0vbwE_05` | 2.04 |  | **** |  |  |  |  |  |  |  | 2 | 275 | DECLINED — 275MB free |
| 2 | 19 | `oP2WdYlaflE_02` | 9.10 |  | **** |  |  |  |  |  |  |  | 2 | 273 | DECLINED — 273MB free |
| 2 | 20 | `LT0-h4P_xBY_06` | 8.40 | 126 | **54.04** | 14.38 |  | 0.13 | 0.23 | 5.22 | 51 | 53 | 3 | 784 | deferred |
| 2 | 21 | `uZwKNtHx9FE_01` | 14.47 |  | **** |  |  |  |  |  |  |  | 3 | 784 | DECLINED — backlog 491 frames |
| 2 | 22 | `fNT8a6e1gx8_08` | 3.60 | 54 | **53.70** | 47.28 | 1.95 | 0.06 | 0.15 | 4.16 | 58 | 60 | 3 | 544 | ok |
| 2 | 23 | `f75G_hZMSHs_04` | 2.40 | 36 | **61.10** | 49.84 |  | 0.06 | 0.14 | 3.62 | 50 | 52 | 4 | 543 | deferred |
| 2 | 24 | `TXyNxVRRPt8_01` | 6.27 | 94 | **80.23** | 53.35 |  | 0.09 | 0.20 | 7.43 | 84 | 86 | 5 | 244 | deferred |
| 2 | 25 | `wPJ_InXF_I8_06` | 4.94 |  | **** |  |  |  |  |  |  |  | 4 | 244 | DECLINED — 244MB free |
| 2 | 26 | `z3jAMn3xpoM_02` | 8.14 |  | **** |  |  |  |  |  |  |  | 4 | 244 | DECLINED — 244MB free |
| 2 | 27 | `fNT8a6e1gx8_06` | 2.77 |  | **** |  |  |  |  |  |  |  | 4 | 244 | DECLINED — 244MB free |
| 2 | 28 | `f75G_hZMSHs_02` | 7.78 |  | **** |  |  |  |  |  |  |  | 4 | 243 | DECLINED — 243MB free |
| 2 | 29 | `LT0-h4P_xBY_04` | 7.77 |  | **** |  |  |  |  |  |  |  | 4 | 243 | DECLINED — 243MB free |
| 2 | 30 | `WMoeSSvCyTU_03` | 7.91 |  | **** |  |  |  |  |  |  |  | 4 | 243 | DECLINED — 243MB free |
| 2 | 31 | `I976oI3s1jQ_02` | 13.50 |  | **** |  |  |  |  |  |  |  | 4 | 243 | DECLINED — 243MB free |
| 2 | 32 | `mNF1kg9azQw_03` | 11.39 | 137 | **64.48** | 56.56 | 0.82 | 0.10 | 0.33 | 6.41 | 90 | 92 | 5 | 257 | ok |
| 2 | 33 | `I976oI3s1jQ_01` | 7.00 |  | **** |  |  |  |  |  |  |  | 5 | 242 | DECLINED — 242MB free |
| 2 | 34 | `iMJ9CjBX7eo_02` | 4.30 |  | **** |  |  |  |  |  |  |  | 5 | 275 | DECLINED — 275MB free |
| 2 | 35 | `k5Gxbifw8s8_05` | 4.43 |  | **** |  |  |  |  |  |  |  | 5 | 274 | DECLINED — 274MB free |
| 2 | 36 | `J-0KHhPS_m4_03` | 5.00 |  | **** |  |  |  |  |  |  |  | 5 | 296 | DECLINED — 296MB free |
| 2 | 37 | `mNF1kg9azQw_07` | 9.38 |  | **** |  |  |  |  |  |  |  | 5 | 322 | DECLINED — 322MB free |
| 2 | 38 | `MJijm9kbdHA_01` | 14.50 |  | **** |  |  |  |  |  |  |  | 5 | 322 | DECLINED — 322MB free |
| 2 | 39 | `jsOSzFyx8RA_03` | 12.71 |  | **** |  |  |  |  |  |  |  | 5 | 322 | DECLINED — 322MB free |
| 3 | 0 | `mNF1kg9azQw_03` | 11.39 | 137 | **13.01** | 0.00 | 3.84 | 0.51 | 0.84 | 7.66 | 90 | 92 | 1 | 2340 | ok, 1st after warmup |
| 3 | 1 | `I976oI3s1jQ_01` | 7.00 | 84 | **19.66** | 4.41 | 5.51 | 0.09 | 0.35 | 9.17 | 84 | 86 | 2 | 1798 | ok |
| 3 | 2 | `5iRIeHAK9b0_03` | 4.70 | 71 | **39.18** | 13.52 |  | 0.07 | 0.18 | 4.30 | 60 | 62 | 3 | 1616 | deferred |
| 3 | 3 | `uZwKNtHx9FE_07` | 12.47 | 150 | **36.75** | 25.08 | 1.51 | 0.09 | 0.34 | 9.55 | 128 | 130 | 3 | 1237 | ok |
| 3 | 4 | `f75G_hZMSHs_02` | 7.78 | 117 | **41.93** | 27.36 | 1.80 | 0.10 | 0.23 | 12.25 | 122 | 124 | 3 | 897 | ok |
| 3 | 5 | `NBpYolVo5WQ_06` | 3.04 | 37 | **54.32** | 37.46 |  | 0.11 | 0.16 | 2.98 | 41 | 43 | 4 | 588 | deferred |
| 3 | 6 | `wPJ_InXF_I8_06` | 4.94 | 74 | **68.23** | 47.90 |  | 0.07 | 0.21 | 3.57 | 50 | 52 | 5 | 522 | deferred |
| 3 | 7 | `I976oI3s1jQ_05` | 3.42 | 41 | **76.02** | 63.34 |  | 0.06 | 0.14 | 3.58 | 49 | 51 | 6 | 555 | deferred |
| 3 | 8 | `ZgvLTS2zJdo_01` | 3.42 | 41 | **86.32** | 71.13 |  | 0.06 | 0.14 | 5.71 | 79 | 81 | 6 | 618 | deferred |
| 3 | 9 | `uZwKNtHx9FE_01` | 14.47 |  | **** |  |  |  |  |  |  |  | 6 | 618 | DECLINED — backlog 460 frames |
| 3 | 10 | `D9HR4q0vbwE_05` | 2.04 |  | **** |  |  |  |  |  |  |  | 6 | 618 | DECLINED — backlog 460 frames |
| 3 | 11 | `k5Gxbifw8s8_03` | 4.70 |  | **** |  |  |  |  |  |  |  | 6 | 618 | DECLINED — backlog 460 frames |
| 3 | 12 | `iMJ9CjBX7eo_02` | 4.30 |  | **** |  |  |  |  |  |  |  | 6 | 618 | DECLINED — backlog 460 frames |
| 3 | 13 | `mNF1kg9azQw_05` | 10.30 | 124 | **75.26** | 68.38 | 1.78 | 0.07 | 0.20 | 4.67 | 65 | 67 | 6 | 1486 | ok |
| 3 | 14 | `rSiciLyYOyI_07` | 2.88 | 35 | **82.62** | 70.94 |  | 0.07 | 0.16 | 3.22 | 45 | 47 | 7 | 1168 | deferred |
| 3 | 15 | `FND2JfzngVY_06` | 8.01 | 120 | **79.69** | 72.97 | 1.69 | 0.08 | 0.21 | 4.61 | 53 | 55 | 7 | 490 | ok |
| 3 | 16 | `WMoeSSvCyTU_03` | 7.91 |  | **** |  |  |  |  |  |  |  | 7 | 542 | DECLINED — backlog 472 frames |
| 3 | 17 | `fNT8a6e1gx8_08` | 3.60 |  | **** |  |  |  |  |  |  |  | 7 | 378 | DECLINED — backlog 472 frames |
| 3 | 18 | `ZgvLTS2zJdo_02` | 14.72 |  | **** |  |  |  |  |  |  |  | 7 | 378 | DECLINED — backlog 472 frames |
| 3 | 19 | `wH3lJm0Typ0_03` | 9.01 |  | **** |  |  |  |  |  |  |  | 7 | 378 | DECLINED — backlog 472 frames |
| 3 | 20 | `LT0-h4P_xBY_04` | 7.77 | 117 | **97.61** | 64.45 |  | 0.09 | 0.22 | 7.93 | 96 | 98 | 7 | 615 | deferred |
| 3 | 21 | `jsOSzFyx8RA_03` | 12.71 |  | **** |  |  |  |  |  |  |  | 7 | 614 | DECLINED — backlog 552 frames |
| 3 | 22 | `k5Gxbifw8s8_05` | 4.43 |  | **** |  |  |  |  |  |  |  | 7 | 613 | DECLINED — backlog 552 frames |
| 3 | 23 | `I976oI3s1jQ_02` | 13.50 |  | **** |  |  |  |  |  |  |  | 7 | 463 | DECLINED — backlog 552 frames |
| 3 | 24 | `mNF1kg9azQw_07` | 9.38 |  | **** |  |  |  |  |  |  |  | 7 | 379 | DECLINED — backlog 552 frames |
| 3 | 25 | `wPJ_InXF_I8_02` | 14.15 |  | **** |  |  |  |  |  |  |  | 7 | 420 | DECLINED — backlog 552 frames |
| 3 | 26 | `TXyNxVRRPt8_01` | 6.27 |  | **** |  |  |  |  |  |  |  | 7 | 421 | DECLINED — backlog 552 frames |
| 3 | 27 | `sP6B4dlSOoc_04` | 5.57 |  | **** |  |  |  |  |  |  |  | 7 | 420 | DECLINED — backlog 552 frames |
| 3 | 28 | `fNT8a6e1gx8_06` | 2.77 |  | **** |  |  |  |  |  |  |  | 6 | 462 | DECLINED — backlog 478 frames |
| 3 | 29 | `TXyNxVRRPt8_03` | 4.50 |  | **** |  |  |  |  |  |  |  | 6 | 451 | DECLINED — backlog 478 frames |
| 3 | 30 | `z3jAMn3xpoM_02` | 8.14 |  | **** |  |  |  |  |  |  |  | 6 | 608 | DECLINED — backlog 478 frames |
| 3 | 31 | `f75G_hZMSHs_04` | 2.40 |  | **** |  |  |  |  |  |  |  | 6 | 603 | DECLINED — backlog 478 frames |
| 3 | 32 | `oP2WdYlaflE_02` | 9.10 |  | **** |  |  |  |  |  |  |  | 6 | 542 | DECLINED — backlog 478 frames |
| 3 | 33 | `MJijm9kbdHA_01` | 14.50 |  | **** |  |  |  |  |  |  |  | 6 | 410 | DECLINED — backlog 478 frames |
| 3 | 34 | `5iRIeHAK9b0_02` | 10.40 |  | **** |  |  |  |  |  |  |  | 6 | 363 | DECLINED — backlog 478 frames |
| 3 | 35 | `sP6B4dlSOoc_05` | 10.98 |  | **** |  |  |  |  |  |  |  | 6 | 363 | DECLINED — backlog 478 frames |
| 3 | 36 | `J-0KHhPS_m4_03` | 5.00 | 75 | **88.54** | 68.54 |  | 0.07 | 0.19 | 4.25 | 58 | 60 | 6 | 736 | deferred |
| 3 | 37 | `LT0-h4P_xBY_06` | 8.40 |  | **** |  |  |  |  |  |  |  | 6 | 750 | DECLINED — backlog 512 frames |
| 3 | 38 | `oP2WdYlaflE_03` | 13.93 |  | **** |  |  |  |  |  |  |  | 6 | 611 | DECLINED — backlog 512 frames |
| 3 | 39 | `YR7BS6l_95E_05` | 11.60 |  | **** |  |  |  |  |  |  |  | 6 | 526 | DECLINED — backlog 512 frames |

## A.3  Confirming cell: `PERCEPTION_WORKERS=1`, sequential, 40 utterances

| pass | # | clip | dur | frames | **latency** | wait | drain | shub | enc | decode | bytes | steps | qd | avail | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | `uZwKNtHx9FE_01` | 14.47 | 174 | **19.43** | 0.00 | 8.01 | 0.26 | 0.65 | 10.38 | 131 | 133 | 1 | 2450 | ok, 1st after warmup |
| 1 | 1 | `5iRIeHAK9b0_03` | 4.70 | 71 | **10.25** | 0.00 | 5.59 | 0.08 | 0.17 | 4.29 | 60 | 62 | 1 | 2305 | ok |
| 1 | 2 | `fNT8a6e1gx8_08` | 3.60 | 54 | **9.53** | 0.00 | 4.38 | 0.06 | 0.14 | 4.87 | 53 | 55 | 1 | 2237 | ok |
| 1 | 3 | `sP6B4dlSOoc_05` | 10.98 | 165 | **18.38** | 0.00 | 10.58 | 0.10 | 0.40 | 7.08 | 95 | 97 | 1 | 1937 | ok |
| 1 | 4 | `mNF1kg9azQw_03` | 11.39 | 137 | **16.68** | 0.00 | 9.86 | 0.09 | 0.32 | 6.20 | 83 | 85 | 1 | 1788 | ok |
| 1 | 5 | `iMJ9CjBX7eo_02` | 4.30 | 65 | **12.77** | 0.00 | 6.64 | 0.08 | 0.18 | 5.75 | 82 | 84 | 1 | 1934 | ok |
| 1 | 6 | `k5Gxbifw8s8_05` | 4.43 | 67 | **9.23** | 0.00 | 5.19 | 0.09 | 0.21 | 3.66 | 33 | 35 | 1 | 1907 | ok |
| 1 | 7 | `MJijm9kbdHA_01` | 14.50 | 174 | **14.16** | 0.00 | 5.46 | 0.09 | 0.41 | 7.93 | 100 | 102 | 1 | 1590 | ok |
| 1 | 8 | `wPJ_InXF_I8_06` | 4.94 | 74 | **11.82** | 0.00 | 7.82 | 0.07 | 0.20 | 3.62 | 50 | 52 | 1 | 1755 | ok |
| 1 | 9 | `I976oI3s1jQ_01` | 7.00 | 84 | **11.63** | 0.00 | 5.21 | 0.07 | 0.19 | 6.03 | 86 | 88 | 1 | 1434 | ok |
| 1 | 10 | `jsOSzFyx8RA_03` | 12.71 | 191 | **25.39** | 0.00 | 11.28 | 0.10 | 0.42 | 13.36 | 166 | 168 | 1 | 1412 | ok |
| 1 | 11 | `fNT8a6e1gx8_06` | 2.77 | 42 | **11.06** | 0.00 | 4.91 | 0.07 | 0.17 | 5.83 | 83 | 85 | 1 | 1657 | ok |
| 1 | 12 | `J-0KHhPS_m4_03` | 5.00 | 75 | **10.62** | 0.00 | 6.02 | 0.07 | 0.17 | 4.23 | 58 | 60 | 1 | 1645 | ok |
| 1 | 13 | `FND2JfzngVY_06` | 8.01 | 120 | **12.59** | 0.00 | 7.61 | 0.09 | 0.24 | 4.47 | 53 | 55 | 1 | 1369 | ok |
| 1 | 14 | `I976oI3s1jQ_02` | 13.50 | 162 | **15.40** | 0.00 | 6.31 | 0.10 | 0.36 | 8.40 | 111 | 113 | 1 | 1063 | ok |
| 1 | 15 | `ZgvLTS2zJdo_02` | 14.72 | 177 | **14.86** | 0.00 | 7.57 | 0.11 | 0.42 | 6.54 | 69 | 71 | 1 | 1051 | ok |
| 1 | 16 | `YR7BS6l_95E_05` | 11.60 | 174 | **18.12** | 0.00 | 9.68 | 0.09 | 0.42 | 7.64 | 97 | 99 | 1 | 1056 | ok |
| 1 | 17 | `NBpYolVo5WQ_06` | 3.04 | 37 | **7.98** | 0.00 | 4.73 | 0.07 | 0.16 | 2.99 | 41 | 43 | 1 | 1168 | ok |
| 1 | 18 | `TXyNxVRRPt8_01` | 6.27 | 94 | **14.39** | 0.00 | 6.50 | 0.09 | 0.19 | 7.46 | 84 | 86 | 1 | 1077 | ok |
| 1 | 19 | `5iRIeHAK9b0_02` | 10.40 | 156 | **17.36** | 0.00 | 8.43 | 0.08 | 0.36 | 8.21 | 115 | 117 | 1 | 1173 | ok |
| 1 | 20 | `mNF1kg9azQw_05` | 10.30 | 124 | **14.72** | 0.00 | 7.99 | 0.09 | 0.24 | 6.27 | 87 | 89 | 1 | 642 | ok |
| 1 | 21 | `WMoeSSvCyTU_03` | 7.91 | 119 | **14.37** | 0.00 | 9.72 | 0.08 | 0.21 | 4.25 | 60 | 62 | 1 | 728 | ok |
| 1 | 22 | `sP6B4dlSOoc_04` | 5.57 | 84 | **10.22** | 0.00 | 5.24 | 0.07 | 0.18 | 4.58 | 44 | 46 | 1 | 898 | ok |
| 1 | 23 | `z3jAMn3xpoM_02` | 8.14 | 122 | **14.75** | 0.00 | 8.41 | 0.09 | 0.22 | 5.89 | 81 | 83 | 1 | 532 | ok |
| 1 | 24 | `uZwKNtHx9FE_07` | 12.47 | 150 | **17.71** | 0.00 | 8.17 | 0.08 | 0.36 | 8.92 | 124 | 126 | 1 | 1116 | ok |
| 1 | 25 | `D9HR4q0vbwE_05` | 2.04 | 25 | **5.86** | 0.00 | 4.10 | 0.09 | 0.17 | 1.48 | 18 | 20 | 1 | 656 | ok |
| 1 | 26 | `f75G_hZMSHs_04` | 2.40 | 36 | **8.89** | 0.00 | 4.96 | 0.07 | 0.16 | 3.65 | 50 | 52 | 1 | 643 | ok |
| 1 | 27 | `f75G_hZMSHs_02` | 7.78 | 117 | **15.45** | 0.00 | 6.10 | 0.10 | 0.23 | 8.83 | 124 | 126 | 1 | 1225 | ok |
| 1 | 28 | `ZgvLTS2zJdo_01` | 3.42 | 41 | **10.79** | 0.00 | 4.82 | 0.09 | 0.17 | 5.67 | 79 | 81 | 1 | 695 | ok |
| 1 | 29 | `LT0-h4P_xBY_04` | 7.77 | 117 | **17.83** | 0.00 | 9.52 | 0.08 | 0.22 | 7.84 | 96 | 98 | 1 | 601 | ok |
| 1 | 30 | `wPJ_InXF_I8_02` | 14.15 | 212 | **21.54** | 0.00 | 12.80 | 0.10 | 0.49 | 7.89 | 104 | 106 | 1 | 812 | ok |
| 1 | 31 | `oP2WdYlaflE_02` | 9.10 | 137 | **20.60** | 0.00 | 11.11 | 0.08 | 0.34 | 8.81 | 111 | 113 | 1 | 1105 | ok |
| 1 | 32 | `LT0-h4P_xBY_06` | 8.40 | 126 | **17.60** | 0.00 | 12.05 | 0.09 | 0.23 | 5.11 | 51 | 53 | 1 | 626 | ok |
| 1 | 33 | `mNF1kg9azQw_07` | 9.38 | 113 | **15.28** | 0.00 | 6.25 | 0.07 | 0.20 | 8.57 | 120 | 122 | 1 | 651 | ok |
| 1 | 34 | `TXyNxVRRPt8_03` | 4.50 | 68 | **10.93** | 0.00 | 6.63 | 0.08 | 0.21 | 3.92 | 51 | 53 | 1 | 639 | ok |
| 1 | 35 | `wH3lJm0Typ0_03` | 9.01 | 135 | **18.91** | 0.00 | 9.09 | 0.10 | 0.33 | 9.26 | 109 | 111 | 1 | 1180 | ok |
| 1 | 36 | `oP2WdYlaflE_03` | 13.93 | 209 | **25.95** | 0.00 | 14.40 | 0.11 | 0.46 | 10.71 | 136 | 138 | 1 | 497 | ok |
| 1 | 37 | `I976oI3s1jQ_05` | 3.42 | 41 | **9.34** | 0.00 | 5.44 | 0.07 | 0.16 | 3.63 | 49 | 51 | 1 | 568 | ok |
| 1 | 38 | `k5Gxbifw8s8_03` | 4.70 | 71 | **9.46** | 0.00 | 5.22 | 0.08 | 0.20 | 3.87 | 48 | 50 | 1 | 642 | ok |
| 1 | 39 | `rSiciLyYOyI_07` | 2.88 | 35 | **8.34** | 0.00 | 4.92 | 0.08 | 0.15 | 3.16 | 45 | 47 | 1 | 733 | ok |

## A.4  Confirming cell: `MAX_LIVE_STREAMS=0` (all deferred), sequential, 40 utterances

| pass | # | clip | dur | frames | **latency** | wait | drain | shub | enc | decode | bytes | steps | qd | avail | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | `uZwKNtHx9FE_01` | 14.47 | 174 | **48.23** | 0.00 |  | 0.36 | 0.90 | 12.17 | 131 | 133 | 1 | 3750 | deferred, 1st after warmup |
| 1 | 1 | `5iRIeHAK9b0_03` | 4.70 | 71 | **18.25** | 0.00 |  | 0.06 | 0.18 | 4.21 | 60 | 62 | 1 | 2284 | deferred |
| 1 | 2 | `fNT8a6e1gx8_08` | 3.60 | 54 | **15.96** | 0.00 |  | 0.07 | 0.17 | 4.84 | 53 | 55 | 1 | 1858 | deferred |
| 1 | 3 | `sP6B4dlSOoc_05` | 10.98 | 165 | **41.01** | 0.00 |  | 0.10 | 0.35 | 6.91 | 95 | 97 | 1 | 1899 | deferred |
| 1 | 4 | `mNF1kg9azQw_03` | 11.39 | 137 | **38.06** | 0.00 |  | 0.09 | 0.33 | 6.30 | 83 | 85 | 1 | 1361 | deferred |
| 1 | 5 | `iMJ9CjBX7eo_02` | 4.30 | 65 | **20.03** | 0.00 |  | 0.07 | 0.18 | 5.83 | 82 | 84 | 1 | 1339 | deferred |
| 1 | 6 | `k5Gxbifw8s8_05` | 4.43 | 67 | **16.90** | 0.00 |  | 0.08 | 0.18 | 3.70 | 33 | 35 | 1 | 1588 | deferred |
| 1 | 7 | `MJijm9kbdHA_01` | 14.50 | 174 | **41.48** | 0.00 |  | 0.09 | 0.39 | 8.04 | 100 | 102 | 1 | 1611 | deferred |
| 1 | 8 | `wPJ_InXF_I8_06` | 4.94 | 74 | **20.79** | 0.00 |  | 0.06 | 0.19 | 3.58 | 50 | 52 | 1 | 1173 | deferred |
| 1 | 9 | `I976oI3s1jQ_01` | 7.00 | 84 | **23.78** | 0.00 |  | 0.07 | 0.20 | 6.17 | 86 | 88 | 1 | 1456 | deferred |
| 1 | 10 | `jsOSzFyx8RA_03` | 12.71 | 191 | **51.81** | 0.00 |  | 0.12 | 0.43 | 13.29 | 166 | 168 | 1 | 1040 | deferred |
| 1 | 11 | `fNT8a6e1gx8_06` | 2.77 | 42 | **15.87** | 0.00 |  | 0.05 | 0.14 | 5.81 | 83 | 85 | 1 | 1074 | deferred |
| 1 | 12 | `J-0KHhPS_m4_03` | 5.00 | 75 | **19.86** | 0.01 |  | 0.08 | 0.20 | 4.24 | 58 | 60 | 1 | 1005 | deferred |
| 1 | 13 | `FND2JfzngVY_06` | 8.01 | 120 | **30.29** | 0.00 |  | 0.08 | 0.21 | 4.55 | 53 | 55 | 1 | 1382 | deferred |
| 1 | 14 | `I976oI3s1jQ_02` | 13.50 | 162 | **41.96** | 0.00 |  | 0.09 | 0.35 | 8.73 | 111 | 113 | 1 | 977 | deferred |
| 1 | 15 | `ZgvLTS2zJdo_02` | 14.72 | 177 | **44.52** | 0.00 |  | 0.10 | 0.41 | 7.20 | 69 | 71 | 1 | 908 | deferred |
| 1 | 16 | `YR7BS6l_95E_05` | 11.60 | 174 | **46.19** | 0.00 |  | 0.10 | 0.41 | 8.15 | 97 | 99 | 1 | 1010 | deferred |
| 1 | 17 | `NBpYolVo5WQ_06` | 3.04 | 37 | **12.66** | 0.01 |  | 0.05 | 0.14 | 2.99 | 41 | 43 | 1 | 958 | deferred |
| 1 | 18 | `TXyNxVRRPt8_01` | 6.27 | 94 | **28.73** | 0.01 |  | 0.07 | 0.17 | 7.46 | 84 | 86 | 1 | 888 | deferred |
| 1 | 19 | `5iRIeHAK9b0_02` | 10.40 | 156 | **39.24** | 0.00 |  | 0.08 | 0.36 | 8.62 | 115 | 117 | 1 | 857 | deferred |
| 1 | 20 | `mNF1kg9azQw_05` | 10.30 | 124 | **35.60** | 0.00 |  | 0.08 | 0.20 | 6.38 | 87 | 89 | 1 | 1067 | deferred |
| 1 | 21 | `WMoeSSvCyTU_03` | 7.91 | 119 | **33.17** | 0.00 |  | 0.07 | 0.20 | 4.26 | 60 | 62 | 1 | 1079 | deferred |
| 1 | 22 | `sP6B4dlSOoc_04` | 5.57 | 84 | **23.14** | 0.00 |  | 0.07 | 0.18 | 4.60 | 44 | 46 | 1 | 1004 | deferred |
| 1 | 23 | `z3jAMn3xpoM_02` | 8.14 | 122 | **32.76** | 0.00 |  | 0.07 | 0.21 | 5.90 | 81 | 83 | 1 | 1074 | deferred |
| 1 | 24 | `uZwKNtHx9FE_07` | 12.47 | 150 | **41.55** | 0.00 |  | 0.08 | 0.35 | 9.36 | 124 | 126 | 1 | 862 | deferred |
| 1 | 25 | `D9HR4q0vbwE_05` | 2.04 | 25 | **7.62** | 0.01 |  | 0.07 | 0.14 | 1.44 | 18 | 20 | 1 | 1096 | deferred |
| 1 | 26 | `f75G_hZMSHs_04` | 2.40 | 36 | **12.19** | 0.00 |  | 0.05 | 0.14 | 3.70 | 50 | 52 | 1 | 1055 | deferred |
| 1 | 27 | `f75G_hZMSHs_02` | 7.78 | 117 | **34.84** | 0.00 |  | 0.07 | 0.20 | 8.89 | 124 | 126 | 1 | 1009 | deferred |
| 1 | 28 | `ZgvLTS2zJdo_01` | 3.42 | 41 | **16.26** | 0.00 |  | 0.06 | 0.14 | 5.71 | 79 | 81 | 1 | 1068 | deferred |
| 1 | 29 | `LT0-h4P_xBY_04` | 7.77 | 117 | **35.80** | 0.00 |  | 0.08 | 0.20 | 7.91 | 96 | 98 | 1 | 1002 | deferred |
| 1 | 30 | `wPJ_InXF_I8_02` | 14.15 | 212 | **52.61** | 0.00 |  | 0.11 | 0.53 | 8.39 | 104 | 106 | 1 | 866 | deferred |
| 1 | 31 | `oP2WdYlaflE_02` | 9.10 | 137 | **41.40** | 0.00 |  | 0.08 | 0.34 | 8.87 | 111 | 113 | 1 | 901 | deferred |
| 1 | 32 | `LT0-h4P_xBY_06` | 8.40 | 126 | **35.39** | 0.00 |  | 0.07 | 0.21 | 5.13 | 51 | 53 | 1 | 918 | deferred |
| 1 | 33 | `mNF1kg9azQw_07` | 9.38 | 113 | **33.53** | 0.00 |  | 0.07 | 0.19 | 8.50 | 120 | 122 | 1 | 872 | deferred |
| 1 | 34 | `TXyNxVRRPt8_03` | 4.50 | 68 | **19.49** | 0.00 |  | 0.06 | 0.18 | 3.94 | 51 | 53 | 1 | 925 | deferred |
| 1 | 35 | `wH3lJm0Typ0_03` | 9.01 | 135 | **40.01** | 0.00 |  | 0.07 | 0.33 | 9.15 | 109 | 111 | 1 | 860 | deferred |
| 1 | 36 | `oP2WdYlaflE_03` | 13.93 | 209 | **56.81** | 0.00 |  | 0.11 | 0.45 | 10.74 | 136 | 138 | 1 | 875 | deferred |
| 1 | 37 | `I976oI3s1jQ_05` | 3.42 | 41 | **13.73** | 0.01 |  | 0.05 | 0.14 | 3.65 | 49 | 51 | 1 | 1048 | deferred |
| 1 | 38 | `k5Gxbifw8s8_03` | 4.70 | 71 | **20.22** | 0.00 |  | 0.07 | 0.18 | 3.89 | 48 | 50 | 1 | 918 | deferred |
| 1 | 39 | `rSiciLyYOyI_07` | 2.88 | 35 | **12.44** | 0.01 |  | 0.06 | 0.14 | 3.16 | 45 | 47 | 1 | 904 | deferred |

## A.5  Confirming cell: shipping config control, run last, 40 utterances

| pass | # | clip | dur | frames | **latency** | wait | drain | shub | enc | decode | bytes | steps | qd | avail | status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0 | `uZwKNtHx9FE_01` | 14.47 | 174 | **14.66** | 0.00 | 2.92 | 0.30 | 0.65 | 10.60 | 125 | 127 | 1 | 1864 | ok, 1st after warmup |
| 1 | 1 | `5iRIeHAK9b0_03` | 4.70 | 71 | **7.65** | 0.00 | 3.92 | 0.08 | 0.20 | 3.37 | 47 | 49 | 1 | 1893 | ok |
| 1 | 2 | `fNT8a6e1gx8_08` | 3.60 | 54 | **7.87** | 0.00 | 3.40 | 0.08 | 0.18 | 4.12 | 58 | 60 | 1 | 1754 | ok |
| 1 | 3 | `sP6B4dlSOoc_05` | 10.98 | 165 | **12.76** | 0.01 | 4.67 | 0.09 | 0.38 | 7.34 | 95 | 97 | 1 | 1127 | ok |
| 1 | 4 | `mNF1kg9azQw_03` | 11.39 | 137 | **9.94** | 0.00 | 2.90 | 0.09 | 0.32 | 6.39 | 90 | 92 | 1 | 1069 | ok |
| 1 | 5 | `iMJ9CjBX7eo_02` | 4.30 | 65 | **9.84** | 0.00 | 3.76 | 0.09 | 0.21 | 5.70 | 77 | 79 | 1 | 1646 | ok |
| 1 | 6 | `k5Gxbifw8s8_05` | 4.43 | 67 | **6.27** | 0.00 | 3.30 | 0.07 | 0.19 | 2.59 | 35 | 37 | 1 | 1045 | ok |
| 1 | 7 | `MJijm9kbdHA_01` | 14.50 | 174 | **10.85** | 0.00 | 2.11 | 0.09 | 0.41 | 8.03 | 100 | 102 | 1 | 1034 | ok |
| 1 | 8 | `wPJ_InXF_I8_06` | 4.94 | 74 | **8.96** | 0.00 | 5.11 | 0.06 | 0.19 | 3.51 | 50 | 52 | 1 | 1157 | ok |
| 1 | 9 | `I976oI3s1jQ_01` | 7.00 | 84 | **9.17** | 0.00 | 2.62 | 0.07 | 0.20 | 6.12 | 84 | 86 | 1 | 803 | ok |
| 1 | 10 | `jsOSzFyx8RA_03` | 12.71 | 191 | **19.14** | 0.00 | 5.11 | 0.09 | 0.42 | 13.26 | 153 | 155 | 1 | 442 | ok |
| 1 | 11 | `fNT8a6e1gx8_06` | 2.77 | 42 | **7.38** | 0.00 | 4.32 | 0.06 | 0.16 | 2.75 | 38 | 40 | 1 | 1155 | ok |
| 1 | 12 | `J-0KHhPS_m4_03` | 5.00 | 75 | **7.14** | 0.00 | 3.51 | 0.08 | 0.20 | 3.23 | 40 | 42 | 1 | 954 | ok |
| 1 | 13 | `FND2JfzngVY_06` | 8.01 | 120 | **9.33** | 0.00 | 4.35 | 0.08 | 0.23 | 4.53 | 53 | 55 | 1 | 1019 | ok |
| 1 | 14 | `I976oI3s1jQ_02` | 13.50 | 162 | **11.79** | 0.00 | 2.31 | 0.10 | 0.38 | 8.72 | 117 | 119 | 1 | 427 | ok |
| 1 | 15 | `ZgvLTS2zJdo_02` | 14.72 | 177 | **9.12** | 0.00 | 2.68 | 0.09 | 0.39 | 5.73 | 56 | 58 | 1 | 309 | ok |
| 1 | 16 | `YR7BS6l_95E_05` | 11.60 | 174 | **13.83** | 0.00 | 5.05 | 0.09 | 0.41 | 8.04 | 97 | 99 | 1 | 351 | ok |
| 1 | 17 | `NBpYolVo5WQ_06` | 3.04 | 37 | **7.30** | 0.00 | 4.04 | 0.08 | 0.18 | 2.96 | 41 | 43 | 1 | 902 | ok |
| 1 | 18 | `TXyNxVRRPt8_01` | 6.27 | 94 | **10.63** | 0.00 | 4.09 | 0.07 | 0.19 | 6.09 | 86 | 88 | 1 | 245 | ok |
| 1 | 19 | `5iRIeHAK9b0_02` | 10.40 | 156 | **12.63** | 0.00 | 4.48 | 0.08 | 0.36 | 7.53 | 93 | 95 | 1 | 268 | ok |
| 1 | 20 | `mNF1kg9azQw_05` | 10.30 | 124 | **7.72** | 0.00 | 2.51 | 0.08 | 0.22 | 4.68 | 65 | 67 | 1 | 526 | ok |
| 1 | 21 | `WMoeSSvCyTU_03` | 7.91 | 119 | **9.67** | 0.00 | 4.89 | 0.09 | 0.23 | 4.26 | 60 | 62 | 1 | 242 | ok |
| 1 | 22 | `sP6B4dlSOoc_04` | 5.57 | 84 | **8.00** | 0.00 | 3.54 | 0.07 | 0.18 | 4.07 | 44 | 46 | 1 | 280 | ok |
| 1 | 23 | `z3jAMn3xpoM_02` | 8.14 | 122 | **10.82** | 0.00 | 4.36 | 0.08 | 0.22 | 6.00 | 81 | 83 | 1 | 302 | ok |
| 1 | 24 | `uZwKNtHx9FE_07` | 12.47 | 150 | **12.95** | 0.00 | 3.04 | 0.10 | 0.36 | 9.22 | 128 | 130 | 1 | 262 | ok |
| 1 | 25 | `D9HR4q0vbwE_05` | 2.04 | 25 | **5.27** | 0.00 | 3.57 | 0.06 | 0.14 | 1.46 | 18 | 20 | 1 | 267 | ok |
| 1 | 26 | `f75G_hZMSHs_04` | 2.40 | 36 | **9.15** | 0.00 | 4.37 | 0.07 | 0.18 | 4.49 | 57 | 59 | 1 | 233 | ok |
| 1 | 27 | `f75G_hZMSHs_02` | 7.78 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 28 | `ZgvLTS2zJdo_01` | 3.42 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 29 | `LT0-h4P_xBY_04` | 7.77 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 30 | `wPJ_InXF_I8_02` | 14.15 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 31 | `oP2WdYlaflE_02` | 9.10 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 32 | `LT0-h4P_xBY_06` | 8.40 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 33 | `mNF1kg9azQw_07` | 9.38 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 34 | `TXyNxVRRPt8_03` | 4.50 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 35 | `wH3lJm0Typ0_03` | 9.01 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 36 | `oP2WdYlaflE_03` | 13.93 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 37 | `I976oI3s1jQ_05` | 3.42 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 38 | `k5Gxbifw8s8_03` | 4.70 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |
| 1 | 39 | `rSiciLyYOyI_07` | 2.88 |  | **** |  |  |  |  |  |  |  | 0 | 281 | DECLINED — 281MB free |

## A.6  Translations, sequential condition (pass 1)

The decoded text for each clip, for anyone checking quality alongside latency. Identical
across all three sequential passes (§13.1).

| clip | dur | bytes | translation |
|---|---|---|---|
| `D9HR4q0vbwE_05` | 2.04 | 18 | How are you doing? |
| `f75G_hZMSHs_04` | 2.40 | 57 | The company will likely start their wedding this weekend. |
| `fNT8a6e1gx8_06` | 2.77 | 38 | Tend to ask your students to graduate. |
| `rSiciLyYOyI_07` | 2.88 | 17 | That was amazing. |
| `NBpYolVo5WQ_06` | 3.04 | 41 | See you tomorrow and stay with the light! |
| `I976oI3s1jQ_05` | 3.42 | 49 | I want to make sure that everyone understands me. |
| `ZgvLTS2zJdo_01` | 3.42 | 113 | The first thing we want to do is to open and ensure that some students have access to sign language interpreters. |
| `fNT8a6e1gx8_08` | 3.60 | 58 | It's just that we have problems with expressing ourselves. |
| `iMJ9CjBX7eo_02` | 4.30 | 77 | Thank you for joining us at the Community Forum and we will host it together. |
| `k5Gxbifw8s8_05` | 4.43 | 35 | And more young people wear rabbits. |
| `TXyNxVRRPt8_03` | 4.50 | 32 | Most people don't know about it. |
| `5iRIeHAK9b0_03` | 4.70 | 47 | Children ranging in age from 3 to 29 years old. |
| `k5Gxbifw8s8_03` | 4.70 | 48 | This job opportunity for Deaf people is growing. |
| `wPJ_InXF_I8_06` | 4.94 | 50 | the students storytelling comprehension abilities. |
| `J-0KHhPS_m4_03` | 5.00 | 40 | Here's a quick question: How can I help? |
| `sP6B4dlSOoc_04` | 5.57 | 44 | Sometimes simple sidewalks can be dangerous. |
| `TXyNxVRRPt8_01` | 6.27 | 86 | We brought them to the Deaf Community Forum to bring their concerns to South Carolina. |
| `I976oI3s1jQ_01` | 7.00 | 84 | My strategy on the chill was a little bit patient and I thought it would be perfect. |
| `LT0-h4P_xBY_04` | 7.77 | 123 | Of course, I want to make sure that the police officers and the police officers are shared with the people and the clients. |
| `f75G_hZMSHs_02` | 7.78 | 122 | The similar conversation between the police officers and his hometown includes the former police officer and his hometown. |
| `WMoeSSvCyTU_03` | 7.91 | 60 | The drug hydrogen is still going on and it is controversial. |
| `FND2JfzngVY_06` | 8.01 | 53 | The daylight savings will be back on March 8 at 2 AM. |
| `z3jAMn3xpoM_02` | 8.14 | 81 | One of them filmed on Facebook LIVE showing officers lying on the floor on blood. |
| `LT0-h4P_xBY_06` | 8.40 | 51 | Do you have a race on the owner of the Ioniar Knox? |
| `wH3lJm0Typ0_03` | 9.01 | 100 | At the end of the game, there were also fans who cheered their fans and critics who criticized them. |
| `oP2WdYlaflE_02` | 9.10 | 115 | The expected percentage of a person from the Institute for Health is experimenting at the University of Washington. |
| `mNF1kg9azQw_07` | 9.38 | 120 | If you are interested in food or money, there is an article by MTV that lists different ways you can support your needs. |
| `mNF1kg9azQw_05` | 10.30 | 65 | I'm wondering how to send it to another person who sent it to me. |
| `5iRIeHAK9b0_02` | 10.40 | 93 | On Friday two men in California warned of child abuse in court for confession today (Friday). |
| `sP6B4dlSOoc_05` | 10.98 | 95 | On Wednesday President Trump signed an executive order about trade reform and high kidney care. |
| `mNF1kg9azQw_03` | 11.39 | 90 | Jennifer shared her experience with a woman who was currently in the photography of Floyd. |
| `YR7BS6l_95E_05` | 11.60 | 97 | The helicopter travels to the area and flies to a shelter to help provide supplies with supplies. |
| `uZwKNtHx9FE_07` | 12.47 | 128 | They said the words were added without delay, were immediately introduced into the legislation to make sure that the work began. |
| `jsOSzFyx8RA_03` | 12.71 | 153 | Or the show isn't a lot about the experience, but it's about who you are as a person and how you through surveillance through life, the photography, too. |
| `I976oI3s1jQ_02` | 13.50 | 117 | She said she was still working for two days, but she had a high fever and chills and decided to stay home recovering. |
| `oP2WdYlaflE_03` | 13.93 | 148 | And a limited number of Deaf and hard of hearing students can be recommended in primary reports such as the Department of Health and Human Services. |
| `wPJ_InXF_I8_02` | 14.15 | 104 | AP also said Barack Obama has declared to endorse him, but Biden said he asked Obama to not endorse him. |
| `uZwKNtHx9FE_01` | 14.47 | 125 | Most of the outbreaks are that children who do not have a vaccine who live in an ultra-orthodox Jewish community in New York. |
| `MJijm9kbdHA_01` | 14.50 | 100 | Biden said Bernie had a vote multiple times against a brady bill that required stricter gun control. |
| `ZgvLTS2zJdo_02` | 14.72 | 56 | I want to share a little bit of an opportunity with you. |

---

# Appendix B — raw CSVs, verbatim

The complete machine-readable data, exactly as written by the harness, so this document is
sufficient on its own: copy a block into a `.csv` file and it will load. 38 columns.

**Column meanings.** `condition`, `pass`, `order_index` (position in that pass's randomised
order), `clip_id`, `duration_s` (ffprobe), `capture_seconds` (replay wall clock),
`raw_frames`, `retained_frames_after_trim` (after FRAME_STRIDE=2; no trim is applied on a
replay — see §2.1), `post_cut_latency_s` (**the headline quantity**: cut → text available),
`worker_s` (dequeue → text, i.e. what `live_worker_probe.py` would have called `seconds`),
`queue_wait_s`, `capture_end_to_perception_done_s` (the drain), `shubert_s`,
`byt5_encoder_s`, `byt5_decode_s`, `byt5_stage_s`, `generate_s`, `other_s` (unattributed),
`decode_steps`, `output_bytes`, `output_text`, `oom_or_error`, `error`, `declined`,
`decline_reason`, `deferred`, `queue_depth_at_cut`, `perception_frames_at_cut` (frames
through MediaPipe at the moment of the cut — the input to §8), `perception_busy_seconds_cpu`
(**aggregate across both workers**, not wall time), `embed_busy_seconds`, `embedded_frames`,
`min_available_mb_during_utterance`, `min_available_mb_during_run`, `avail_mb_at_cut`,
`first_after_warmup`, `warmup_seconds`, `cut_ts`, `done_ts` (both Unix epoch seconds).

Note: some `error` fields contain embedded newlines (CUDA OOM messages are multi-line), so
these must be parsed with a real CSV reader, not split on newlines.

## B.1  `sequential.csv`

```csv
condition,pass,order_index,clip_id,duration_s,capture_seconds,raw_frames,retained_frames_after_trim,post_cut_latency_s,worker_s,queue_wait_s,capture_end_to_perception_done_s,shubert_s,byt5_encoder_s,byt5_decode_s,byt5_stage_s,generate_s,other_s,decode_steps,output_bytes,output_text,oom_or_error,error,declined,decline_reason,deferred,queue_depth_at_cut,perception_frames_at_cut,perception_busy_seconds_cpu,embed_busy_seconds,embedded_frames,min_available_mb_during_utterance,min_available_mb_during_run,avail_mb_at_cut,first_after_warmup,warmup_seconds,cut_ts,done_ts
sequential,1,0,uZwKNtHx9FE_01,14.473,14.43,347,174,14.6,14.6,0.0,2.65,0.29,0.68,10.8,11.81,11.77,0.14,127,125,Most of the outbreaks are that children who do not have a vaccine who live in an ultra-orthodox Jewish community in New York.,False,,False,,False,1,169,20.21,13.28,174,1791,1791,1914,True,18.4,1791518287.4787068,1791518302.08
sequential,1,1,5iRIeHAK9b0_03,4.700,4.68,141,71,7.53,7.52,0.0,3.76,0.09,0.2,3.39,3.68,3.67,0.08,49,47,Children ranging in age from 3 to 29 years old.,False,,False,,False,1,63,7.71,4.73,71,1374,1374,1517,False,18.4,1791518306.9406984,1791518314.469
sequential,1,2,fNT8a6e1gx8_08,3.604,3.57,108,54,7.83,7.83,0.0,3.32,0.05,0.14,4.2,4.41,4.4,0.09,60,58,It's just that we have problems with expressing ourselves.,False,,False,,False,1,44,6.0,3.46,54,1180,1180,1424,False,18.4,1791518318.2239885,1791518326.053
sequential,1,3,sP6B4dlSOoc_05,10.978,10.95,329,165,12.95,12.95,0.0,4.86,0.1,0.4,7.36,7.89,7.86,0.2,97,95,On Wednesday President Trump signed an executive order about trade reform and high kidney care.,False,,False,,False,1,153,19.04,11.93,165,1026,1026,1176,False,18.4,1791518337.1860757,1791518350.133
sequential,1,4,mNF1kg9azQw_03,11.387,11.35,273,137,10.19,10.19,0.0,3.16,0.08,0.32,6.43,6.86,6.83,0.16,92,90,Jennifer shared her experience with a woman who was currently in the photography of Floyd.,False,,False,,False,1,130,18.52,10.34,137,964,949,1021,False,18.4,1791518361.6558225,1791518371.847
sequential,1,5,iMJ9CjBX7eo_02,4.305,4.27,129,65,9.92,9.91,0.01,3.87,0.07,0.18,5.7,5.98,5.95,0.06,79,77,Thank you for joining us at the Community Forum and we will host it together.,False,,False,,False,1,49,7.95,4.3,65,797,797,1312,False,18.4,1791518376.3177454,1791518386.236
sequential,1,6,k5Gxbifw8s8_05,4.434,4.4,133,67,6.43,6.43,0.0,3.44,0.08,0.2,2.61,2.9,2.89,0.09,37,35,And more young people wear rabbits.,False,,False,,False,1,56,7.0,4.51,67,807,757,937,False,18.4,1791518390.8278174,1791518397.257
sequential,1,7,MJijm9kbdHA_01,14.500,14.46,348,174,10.94,10.93,0.0,2.05,0.08,0.4,8.14,8.65,8.62,0.23,102,100,Biden said Bernie had a vote multiple times against a brady bill that required stricter gun control.,False,,False,,False,1,169,18.15,11.91,174,690,669,695,False,18.4,1791518411.891051,1791518422.827
sequential,1,8,wPJ_InXF_I8_06,4.939,4.91,148,74,8.75,8.75,0.0,4.84,0.06,0.2,3.54,3.82,3.8,0.1,52,50,the students storytelling comprehension abilities.,False,,False,,False,1,50,10.67,5.02,74,591,591,1012,False,18.4,1791518427.9327917,1791518436.685
sequential,1,9,I976oI3s1jQ_01,7.000,6.96,168,84,9.21,9.21,0.0,2.66,0.07,0.19,6.12,6.41,6.38,0.14,86,84,My strategy on the chill was a little bit patient and I thought it would be perfect.,False,,False,,False,1,78,9.44,5.91,84,547,547,550,False,18.4,1791518443.8381007,1791518453.046
sequential,1,10,jsOSzFyx8RA_03,12.713,12.68,381,191,19.32,19.31,0.01,5.17,0.1,0.43,13.31,13.89,13.84,0.25,155,153,"Or the show isn't a lot about the experience, but it's about who you are as a person and how you through surveillance through life, the photography, too.",False,,False,,False,1,181,21.37,14.43,191,456,456,457,False,18.4,1791518465.915946,1791518485.234
sequential,1,11,fNT8a6e1gx8_06,2.770,2.74,83,42,7.44,7.44,0.0,4.35,0.07,0.17,2.82,3.07,3.06,0.02,40,38,Tend to ask your students to graduate.,False,,False,,False,1,23,5.44,2.86,42,606,456,831,False,18.4,1791518488.1718833,1791518495.609
sequential,1,12,J-0KHhPS_m4_03,5.000,4.97,150,75,7.12,7.12,0.0,3.43,0.08,0.2,3.29,3.58,3.56,0.11,42,40,Here's a quick question: How can I help?,False,,False,,False,1,64,8.17,5.04,75,444,444,504,False,18.4,1791518500.7750344,1791518507.899
sequential,1,13,FND2JfzngVY_06,8.008,7.98,240,120,9.72,9.72,0.0,4.71,0.08,0.22,4.55,4.87,4.85,0.14,55,53,The daylight savings will be back on March 8 at 2 AM.,False,,False,,False,1,106,13.49,9.01,120,382,355,484,False,18.4,1791518516.0753226,1791518525.795
sequential,1,14,I976oI3s1jQ_02,13.500,13.46,324,162,11.58,11.58,0.0,2.15,0.09,0.36,8.76,9.25,9.21,0.19,119,117,"She said she was still working for two days, but she had a high fever and chills and decided to stay home recovering.",False,,False,,False,1,159,16.88,11.11,162,435,355,664,False,18.4,1791518539.442908,1791518551.026
sequential,1,15,ZgvLTS2zJdo_02,14.724,14.69,353,177,8.79,8.78,0.0,2.33,0.11,0.4,5.66,6.18,6.16,0.27,58,56,I want to share a little bit of an opportunity with you.,False,,False,,False,1,166,19.17,11.4,177,449,355,454,False,18.4,1791518565.9738011,1791518574.76
sequential,1,16,YR7BS6l_95E_05,11.600,11.58,348,174,13.99,13.99,0.0,5.25,0.09,0.4,8.01,8.53,8.51,0.21,99,97,The helicopter travels to the area and flies to a shelter to help provide supplies with supplies.,False,,False,,False,1,164,19.89,12.95,174,366,355,403,False,18.4,1791518586.5334346,1791518600.527
sequential,1,17,NBpYolVo5WQ_06,3.042,3.0,73,37,8.01,8.0,0.0,4.7,0.09,0.16,3.03,3.29,3.27,0.02,43,41,See you tomorrow and stay with the light!,False,,False,,False,1,21,5.84,2.62,37,467,355,515,False,18.4,1791518603.7296398,1791518611.735
sequential,1,18,TXyNxVRRPt8_01,6.273,6.24,188,94,10.59,10.59,0.0,4.01,0.06,0.19,6.16,6.44,6.41,0.14,88,86,We brought them to the Deaf Community Forum to bring their concerns to South Carolina.,False,,False,,False,1,82,10.16,6.59,94,303,303,448,False,18.4,1791518618.1564791,1791518628.744
sequential,1,19,5iRIeHAK9b0_02,10.400,10.37,312,156,12.61,12.61,0.0,4.35,0.09,0.36,7.62,8.1,8.07,0.16,95,93,On Friday two men in California warned of child abuse in court for confession today (Friday).,False,,False,,False,1,146,16.06,11.25,156,302,302,633,False,18.4,1791518639.2999742,1791518651.91
sequential,1,20,mNF1kg9azQw_05,10.302,10.27,247,124,7.86,7.86,0.0,2.72,0.08,0.22,4.61,4.93,4.91,0.21,67,65,I'm wondering how to send it to another person who sent it to me.,False,,False,,False,1,117,14.8,8.59,124,430,302,436,False,18.4,1791518662.3720975,1791518670.231
sequential,1,21,WMoeSSvCyTU_03,7.908,7.88,237,119,10.0,10.0,0.0,5.23,0.09,0.21,4.33,4.65,4.63,0.12,62,60,The drug hydrogen is still going on and it is controversial.,False,,False,,False,1,103,15.26,9.11,119,289,284,406,False,18.4,1791518678.293532,1791518688.293
sequential,1,22,sP6B4dlSOoc_04,5.573,5.54,167,84,7.96,7.95,0.01,3.49,0.06,0.18,4.07,4.33,4.31,0.14,46,44,Sometimes simple sidewalks can be dangerous.,False,,False,,False,1,73,8.95,5.54,84,331,284,332,False,18.4,1791518694.0236075,1791518701.981
sequential,1,23,z3jAMn3xpoM_02,8.142,8.34,244,122,11.03,11.03,0.0,4.5,0.07,0.23,6.08,6.42,6.39,0.12,83,81,One of them filmed on Facebook LIVE showing officers lying on the floor on blood.,False,,False,,False,1,109,13.26,9.29,122,297,284,341,False,18.4,1791518710.5217896,1791518721.552
sequential,1,24,uZwKNtHx9FE_07,12.471,12.44,299,150,12.72,12.72,0.0,2.74,0.09,0.37,9.29,9.79,9.75,0.19,130,128,"They said the words were added without delay, were immediately introduced into the legislation to make sure that the work began.",False,,False,,False,1,139,17.04,10.29,150,373,284,376,False,18.4,1791518734.1740382,1791518746.895
sequential,1,25,D9HR4q0vbwE_05,2.044,2.0,49,25,5.23,5.23,0.0,3.52,0.08,0.14,1.45,1.68,1.67,0.03,20,18,How are you doing?,False,,False,,False,1,9,3.73,1.68,25,351,284,449,False,18.4,1791518749.0375526,1791518754.27
sequential,1,26,f75G_hZMSHs_04,2.403,2.37,72,36,8.65,8.65,0.0,3.81,0.07,0.18,4.55,4.82,4.8,0.02,59,57,The company will likely start their wedding this weekend.,False,,False,,False,1,21,4.47,2.44,36,307,284,443,False,18.4,1791518756.831966,1791518765.482
sequential,1,27,f75G_hZMSHs_02,7.775,7.74,233,117,13.96,13.96,0.0,4.04,0.08,0.22,9.45,9.79,9.75,0.13,124,122,The similar conversation between the police officers and his hometown includes the former police officer and his hometown.,False,,False,,False,1,106,12.4,8.43,117,328,284,424,False,18.4,1791518773.4181113,1791518787.383
sequential,1,28,ZgvLTS2zJdo_01,3.421,3.38,82,41,11.75,11.75,0.0,3.51,0.08,0.17,7.94,8.22,8.18,0.02,115,113,The first thing we want to do is to open and ensure that some students have access to sign language interpreters.,False,,False,,False,1,32,5.39,2.88,41,295,284,455,False,18.4,1791518790.987431,1791518802.733
sequential,1,29,LT0-h4P_xBY_04,7.767,7.74,233,117,13.47,13.47,0.0,4.04,0.09,0.23,8.91,9.27,9.24,0.16,125,123,"Of course, I want to make sure that the police officers and the police officers are shared with the people and the clients.",False,,False,,False,1,103,13.79,7.76,117,325,284,293,False,18.4,1791518810.6735687,1791518824.14
sequential,1,30,wPJ_InXF_I8_02,14.148,14.12,424,212,14.28,14.27,0.0,5.49,0.09,0.48,7.93,8.54,8.51,0.25,106,104,"AP also said Barack Obama has declared to endorse him, but Biden said he asked Obama to not endorse him.",False,,False,,False,1,202,23.23,15.65,212,318,284,343,False,18.4,1791518838.4420357,1791518852.718
sequential,1,31,oP2WdYlaflE_02,9.100,9.07,273,137,14.39,14.39,0.0,5.24,0.09,0.32,8.53,8.97,8.93,0.18,117,115,The expected percentage of a person from the Institute for Health is experimenting at the University of Washington.,False,,False,,False,1,124,16.17,10.19,137,351,284,358,False,18.4,1791518861.97001,1791518876.359
sequential,1,32,LT0-h4P_xBY_06,8.400,8.37,252,126,9.27,9.27,0.0,4.19,0.07,0.23,4.63,4.94,4.92,0.14,53,51,Do you have a race on the owner of the Ioniar Knox?,False,,False,,False,1,111,15.13,8.29,126,283,283,283,False,18.4,1791518884.9156823,1791518894.188
sequential,1,33,mNF1kg9azQw_07,9.385,9.35,225,113,11.27,11.26,0.0,2.08,0.08,0.2,8.63,8.95,8.91,0.24,122,120,"If you are interested in food or money, there is an article by MTV that lists different ways you can support your needs.",False,,False,,False,1,108,12.01,7.84,113,443,280,448,False,18.4,1791518903.7224536,1791518914.988
sequential,1,34,TXyNxVRRPt8_03,4.505,4.47,135,68,6.98,6.97,0.0,3.64,0.07,0.2,2.95,3.23,3.22,0.1,34,32,Most people don't know about it.,False,,False,,False,1,58,7.62,4.36,68,322,280,401,False,18.4,1791518919.6539488,1791518926.629
sequential,1,35,wH3lJm0Typ0_03,9.009,8.98,270,135,13.6,13.6,0.0,4.77,0.09,0.32,8.22,8.66,8.63,0.16,102,100,"At the end of the game, there were also fans who cheered their fans and critics who criticized them.",False,,False,,False,1,125,15.57,9.94,135,325,280,672,False,18.4,1791518935.7864761,1791518949.384
sequential,1,36,oP2WdYlaflE_03,13.934,13.91,418,209,17.89,17.88,0.0,5.31,0.1,0.46,11.74,12.35,12.3,0.23,150,148,And a limited number of Deaf and hard of hearing students can be recommended in primary reports such as the Department of Health and Human Services.,False,,False,,False,1,199,24.73,15.58,209,378,259,380,False,18.4,1791518963.5150888,1791518981.401
sequential,1,37,I976oI3s1jQ_05,3.417,3.38,82,41,7.39,7.38,0.0,3.34,0.06,0.15,3.74,3.96,3.94,0.09,51,49,I want to make sure that everyone understands me.,False,,False,,False,1,32,5.03,2.84,41,428,259,674,False,18.4,1791518984.9610364,1791518992.348
sequential,1,38,k5Gxbifw8s8_03,4.700,4.67,141,71,8.27,8.26,0.01,3.99,0.06,0.18,3.86,4.11,4.1,0.15,50,48,This job opportunity for Deaf people is growing.,False,,False,,False,1,62,7.71,5.14,71,302,259,702,False,18.4,1791518997.2341585,1791519005.5
sequential,1,39,rSiciLyYOyI_07,2.878,2.84,69,35,5.32,5.32,0.0,3.69,0.08,0.17,1.33,1.59,1.58,0.04,19,17,That was amazing.,False,,False,,False,1,23,4.77,2.37,35,689,259,835,False,18.4,1791519008.4695404,1791519013.789
sequential,2,0,sP6B4dlSOoc_04,5.573,5.54,167,84,9.13,9.13,0.0,3.97,0.27,0.41,4.38,5.08,5.06,0.08,46,44,Sometimes simple sidewalks can be dangerous.,False,,False,,False,1,74,8.77,6.05,84,2025,2025,2739,True,18.7,1791519347.5616744,1791519356.689
sequential,2,1,rSiciLyYOyI_07,2.878,2.84,69,35,5.94,5.94,0.0,4.35,0.08,0.16,1.32,1.57,1.56,0.02,19,17,That was amazing.,False,,False,,False,1,19,5.4,2.45,35,1736,1736,2009,False,18.7,1791519359.7133188,1791519365.65
sequential,2,2,YR7BS6l_95E_05,11.600,11.57,348,174,13.8,13.8,0.0,4.94,0.09,0.41,8.11,8.65,8.62,0.21,99,97,The helicopter travels to the area and flies to a shelter to help provide supplies with supplies.,False,,False,,False,1,164,19.17,12.97,174,1470,1470,1573,False,18.7,1791519377.399292,1791519391.195
sequential,2,3,ZgvLTS2zJdo_01,3.421,3.38,82,41,11.88,11.88,0.0,3.64,0.08,0.15,7.93,8.19,8.16,0.05,115,113,The first thing we want to do is to open and ensure that some students have access to sign language interpreters.,False,,False,,False,1,30,5.52,2.93,41,1424,1424,1665,False,18.7,1791519394.7556405,1791519406.635
sequential,2,4,sP6B4dlSOoc_05,10.978,10.95,329,165,12.83,12.83,0.0,4.75,0.09,0.42,7.33,7.87,7.83,0.21,97,95,On Wednesday President Trump signed an executive order about trade reform and high kidney care.,False,,False,,False,1,154,18.17,12.05,165,1161,1161,1298,False,18.7,1791519417.7689931,1791519430.597
sequential,2,5,mNF1kg9azQw_05,10.302,10.28,247,124,8.19,8.19,0.0,3.15,0.08,0.21,4.59,4.9,4.88,0.13,67,65,I'm wondering how to send it to another person who sent it to me.,False,,False,,False,1,118,15.03,9.08,124,931,931,1176,False,18.7,1791519441.046938,1791519449.234
sequential,2,6,oP2WdYlaflE_03,13.934,13.9,418,209,16.89,16.89,0.0,4.2,0.11,0.43,11.84,12.43,12.38,0.25,150,148,And a limited number of Deaf and hard of hearing students can be recommended in primary reports such as the Department of Health and Human Services.,False,,False,,False,1,198,24.01,14.57,209,728,728,1076,False,18.7,1791519463.3480275,1791519480.235
sequential,2,7,5iRIeHAK9b0_02,10.400,10.37,312,156,12.62,12.61,0.0,4.48,0.09,0.36,7.51,7.98,7.95,0.16,95,93,On Friday two men in California warned of child abuse in court for confession today (Friday).,False,,False,,False,1,146,16.67,11.04,156,563,563,799,False,18.7,1791519490.7886333,1791519503.405
sequential,2,8,uZwKNtHx9FE_07,12.471,12.43,299,150,12.81,12.8,0.0,2.79,0.09,0.35,9.35,9.83,9.79,0.19,130,128,"They said the words were added without delay, were immediately introduced into the legislation to make sure that the work began.",False,,False,,False,1,138,17.56,10.77,150,507,507,521,False,18.7,1791519516.0312858,1791519528.838
sequential,2,9,wPJ_InXF_I8_02,14.148,14.12,424,212,14.37,14.37,0.0,5.54,0.1,0.47,7.95,8.55,8.52,0.29,106,104,"AP also said Barack Obama has declared to endorse him, but Biden said he asked Obama to not endorse him.",False,,False,,False,1,203,23.5,15.73,212,393,393,413,False,18.7,1791519543.1610613,1791519557.535
sequential,2,10,5iRIeHAK9b0_03,4.700,4.67,141,71,7.67,7.67,0.0,3.98,0.07,0.19,3.34,3.61,3.6,0.08,49,47,Children ranging in age from 3 to 29 years old.,False,,False,,False,1,62,7.57,5.07,71,649,393,976,False,18.7,1791519562.3964956,1791519570.07
sequential,2,11,TXyNxVRRPt8_03,4.505,4.48,135,68,7.31,7.31,0.0,4.03,0.07,0.2,2.95,3.23,3.21,0.05,34,32,Most people don't know about it.,False,,False,,False,1,58,7.58,4.68,68,487,393,828,False,18.7,1791519574.732703,1791519582.041
sequential,2,12,k5Gxbifw8s8_03,4.700,4.68,141,71,8.34,8.34,0.0,4.06,0.07,0.18,3.88,4.15,4.14,0.14,50,48,This job opportunity for Deaf people is growing.,False,,False,,False,1,56,8.0,5.26,71,465,393,716,False,18.7,1791519586.8955996,1791519595.241
sequential,2,13,FND2JfzngVY_06,8.008,7.98,240,120,9.03,9.03,0.0,4.04,0.07,0.21,4.54,4.84,4.83,0.15,55,53,The daylight savings will be back on March 8 at 2 AM.,False,,False,,False,1,110,13.17,8.38,120,288,288,450,False,18.7,1791519603.3991973,1791519612.428
sequential,2,14,wH3lJm0Typ0_03,9.009,8.98,270,135,13.52,13.52,0.0,4.75,0.09,0.33,8.17,8.62,8.59,0.14,102,100,"At the end of the game, there were also fans who cheered their fans and critics who criticized them.",False,,False,,False,1,125,16.04,10.11,135,371,288,601,False,18.7,1791519621.58046,1791519635.097
sequential,2,15,I976oI3s1jQ_05,3.417,3.38,82,41,7.27,7.27,0.0,3.24,0.07,0.16,3.76,4.01,4.0,0.02,51,49,I want to make sure that everyone understands me.,False,,False,,False,1,32,5.07,2.67,41,279,279,279,False,18.7,1791519638.651701,1791519645.924
sequential,2,16,NBpYolVo5WQ_06,3.042,3.0,73,37,7.58,7.58,0.0,4.28,0.08,0.18,3.0,3.28,3.26,0.02,43,41,See you tomorrow and stay with the light!,False,,False,,False,1,21,5.45,2.64,37,356,279,462,False,18.7,1791519649.1173923,1791519656.699
sequential,2,17,ZgvLTS2zJdo_02,14.724,14.68,353,177,8.8,8.8,0.0,2.42,0.1,0.41,5.62,6.15,6.13,0.24,58,56,I want to share a little bit of an opportunity with you.,False,,False,,False,1,167,19.04,11.51,177,371,270,375,False,18.7,1791519671.575197,1791519680.379
sequential,2,18,D9HR4q0vbwE_05,2.044,2.0,49,25,5.23,5.23,0.0,3.52,0.07,0.16,1.45,1.7,1.68,0.01,20,18,How are you doing?,False,,False,,False,1,11,3.6,1.9,25,274,270,300,False,18.7,1791519682.5526886,1791519687.78
sequential,2,19,oP2WdYlaflE_02,9.100,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,20,LT0-h4P_xBY_06,8.400,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,21,uZwKNtHx9FE_01,14.473,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,22,fNT8a6e1gx8_08,3.604,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,23,f75G_hZMSHs_04,2.403,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,24,TXyNxVRRPt8_01,6.273,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,25,wPJ_InXF_I8_06,4.939,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,26,z3jAMn3xpoM_02,8.142,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,27,fNT8a6e1gx8_06,2.770,,,,,,,,,,,,,,,,,False,,True,316MB free,,0,,,,,,,316,,18.7,,
sequential,2,28,f75G_hZMSHs_02,7.775,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,29,LT0-h4P_xBY_04,7.767,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,30,WMoeSSvCyTU_03,7.908,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,31,I976oI3s1jQ_02,13.500,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,32,mNF1kg9azQw_03,11.387,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,33,I976oI3s1jQ_01,7.000,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,34,iMJ9CjBX7eo_02,4.305,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,35,k5Gxbifw8s8_05,4.434,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,36,J-0KHhPS_m4_03,5.000,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,37,mNF1kg9azQw_07,9.385,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,38,MJijm9kbdHA_01,14.500,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,2,39,jsOSzFyx8RA_03,12.713,,,,,,,,,,,,,,,,,False,,True,317MB free,,0,,,,,,,317,,18.7,,
sequential,3,0,mNF1kg9azQw_03,11.387,11.35,273,137,10.94,10.94,0.0,3.21,0.35,0.54,6.62,7.54,7.51,0.19,92,90,Jennifer shared her experience with a woman who was currently in the photography of Floyd.,False,,False,,False,1,127,18.11,10.44,137,1917,1917,2138,True,18.3,1791520023.5620415,1791520034.503
sequential,3,1,I976oI3s1jQ_01,7.000,6.96,168,84,9.03,9.03,0.0,2.5,0.08,0.19,6.09,6.39,6.36,0.15,86,84,My strategy on the chill was a little bit patient and I thought it would be perfect.,False,,False,,False,1,78,9.46,5.66,84,1504,1504,1679,False,18.3,1791520041.6482823,1791520050.682
sequential,3,2,5iRIeHAK9b0_03,4.700,4.67,141,71,7.59,7.59,0.0,3.87,0.07,0.18,3.36,3.63,3.62,0.09,49,47,Children ranging in age from 3 to 29 years old.,False,,False,,False,1,63,7.59,4.94,71,1438,1438,1855,False,18.3,1791520055.5343192,1791520063.126
sequential,3,3,uZwKNtHx9FE_07,12.471,12.43,299,150,12.77,12.77,0.0,2.75,0.09,0.35,9.39,9.87,9.83,0.16,130,128,"They said the words were added without delay, were immediately introduced into the legislation to make sure that the work began.",False,,False,,False,1,139,17.33,10.39,150,1148,1148,1405,False,18.3,1791520075.7561755,1791520088.53
sequential,3,4,f75G_hZMSHs_02,7.775,7.75,233,117,14.05,14.05,0.0,4.11,0.08,0.22,9.48,9.81,9.77,0.13,124,122,The similar conversation between the police officers and his hometown includes the former police officer and his hometown.,False,,False,,False,1,106,12.8,8.61,117,1154,1134,1132,False,18.3,1791520096.4541066,1791520110.503
sequential,3,5,NBpYolVo5WQ_06,3.042,3.0,73,37,7.67,7.67,0.0,4.36,0.08,0.16,3.02,3.27,3.26,0.04,43,41,See you tomorrow and stay with the light!,False,,False,,False,1,21,5.42,2.73,37,1246,1134,1448,False,18.3,1791520113.6929822,1791520121.362
sequential,3,6,wPJ_InXF_I8_06,4.939,4.91,148,74,8.68,8.68,0.0,4.67,0.07,0.2,3.61,3.9,3.88,0.11,52,50,the students storytelling comprehension abilities.,False,,False,,False,1,56,9.89,5.4,74,1207,1134,1334,False,18.3,1791520126.4459307,1791520135.131
sequential,3,7,I976oI3s1jQ_05,3.417,3.38,82,41,7.28,7.27,0.0,3.24,0.06,0.14,3.74,3.96,3.95,0.07,51,49,I want to make sure that everyone understands me.,False,,False,,False,1,34,4.84,2.91,41,827,827,1115,False,18.3,1791520138.7100313,1791520145.985
sequential,3,8,ZgvLTS2zJdo_01,3.421,3.38,82,41,11.87,11.87,0.0,3.62,0.08,0.18,7.93,8.23,8.18,0.02,115,113,The first thing we want to do is to open and ensure that some students have access to sign language interpreters.,False,,False,,False,1,30,5.6,2.81,41,972,827,1250,False,18.3,1791520149.533168,1791520161.405
sequential,3,9,uZwKNtHx9FE_01,14.473,14.43,347,174,13.28,13.28,0.0,2.24,0.09,0.41,10.21,10.75,10.71,0.29,127,125,Most of the outbreaks are that children who do not have a vaccine who live in an ultra-orthodox Jewish community in New York.,False,,False,,False,1,169,19.71,12.12,174,654,625,945,False,18.3,1791520176.01061,1791520189.293
sequential,3,10,D9HR4q0vbwE_05,2.044,2.0,49,25,5.11,5.11,0.0,3.44,0.07,0.14,1.44,1.66,1.65,0.01,20,18,How are you doing?,False,,False,,False,1,11,3.58,1.79,25,520,520,520,False,18.3,1791520191.4537826,1791520196.563
sequential,3,11,k5Gxbifw8s8_03,4.700,4.67,141,71,8.46,8.46,0.0,4.18,0.08,0.2,3.88,4.17,4.15,0.1,50,48,This job opportunity for Deaf people is growing.,False,,False,,False,1,62,7.92,5.04,71,685,520,1012,False,18.3,1791520201.4194837,1791520209.876
sequential,3,12,iMJ9CjBX7eo_02,4.305,4.29,129,65,9.61,9.61,0.0,3.49,0.08,0.2,5.72,6.03,6.0,0.1,79,77,Thank you for joining us at the Community Forum and we will host it together.,False,,False,,False,1,50,7.77,4.07,65,418,418,549,False,18.3,1791520214.3362064,1791520223.95
sequential,3,13,mNF1kg9azQw_05,10.302,10.28,247,124,7.83,7.83,0.0,2.71,0.07,0.21,4.67,4.97,4.95,0.15,67,65,I'm wondering how to send it to another person who sent it to me.,False,,False,,False,1,119,14.77,8.91,124,436,418,683,False,18.3,1791520234.4007087,1791520242.229
sequential,3,14,rSiciLyYOyI_07,2.878,2.84,69,35,5.49,5.49,0.0,3.87,0.08,0.16,1.36,1.6,1.59,0.02,19,17,That was amazing.,False,,False,,False,1,23,5.0,2.4,35,777,415,784,False,18.3,1791520245.2720358,1791520250.766
sequential,3,15,FND2JfzngVY_06,8.008,7.98,240,120,9.57,9.57,0.0,4.47,0.08,0.22,4.58,4.9,4.88,0.2,55,53,The daylight savings will be back on March 8 at 2 AM.,False,,False,,False,1,105,13.85,8.8,120,420,415,926,False,18.3,1791520258.9334097,1791520268.501
sequential,3,16,WMoeSSvCyTU_03,7.908,7.88,237,119,9.28,9.28,0.0,4.55,0.07,0.22,4.3,4.61,4.59,0.13,62,60,The drug hydrogen is still going on and it is controversial.,False,,False,,False,1,106,14.5,8.34,119,312,312,582,False,18.3,1791520276.6082964,1791520285.889
sequential,3,17,fNT8a6e1gx8_08,3.604,3.57,108,54,7.75,7.74,0.0,3.32,0.06,0.17,4.13,4.38,4.36,0.06,60,58,It's just that we have problems with expressing ourselves.,False,,False,,False,1,44,5.84,3.55,54,636,312,913,False,18.3,1791520289.6343598,1791520297.381
sequential,3,18,ZgvLTS2zJdo_02,14.724,14.68,353,177,9.33,9.33,0.0,2.87,0.11,0.41,5.68,6.21,6.19,0.25,58,56,I want to share a little bit of an opportunity with you.,False,,False,,False,1,164,19.17,11.37,177,314,312,338,False,18.3,1791520312.2807167,1791520321.612
sequential,3,19,wH3lJm0Typ0_03,9.009,8.98,270,135,13.82,13.82,0.0,5.0,0.09,0.33,8.22,8.67,8.64,0.15,102,100,"At the end of the game, there were also fans who cheered their fans and critics who criticized them.",False,,False,,False,1,124,16.17,9.91,135,248,239,300,False,18.3,1791520330.7732165,1791520344.592
sequential,3,20,LT0-h4P_xBY_04,7.767,7.74,233,117,13.27,13.27,0.0,4.11,0.08,0.21,8.72,9.05,9.01,0.12,125,123,"Of course, I want to make sure that the police officers and the police officers are shared with the people and the clients.",False,,False,,False,1,103,13.63,7.82,117,242,239,411,False,18.3,1791520352.5380557,1791520365.812
sequential,3,21,jsOSzFyx8RA_03,12.713,12.68,381,191,19.4,19.4,0.0,5.28,0.1,0.41,13.39,13.95,13.9,0.18,155,153,"Or the show isn't a lot about the experience, but it's about who you are as a person and how you through surveillance through life, the photography, too.",False,,False,,False,1,181,21.14,14.55,191,268,234,314,False,18.3,1791520378.6897457,1791520398.095
sequential,3,22,k5Gxbifw8s8_05,4.434,4.4,133,67,6.73,6.72,0.0,3.77,0.07,0.19,2.59,2.86,2.85,0.09,37,35,And more young people wear rabbits.,False,,False,,False,1,57,6.92,4.81,67,260,234,360,False,18.3,1791520402.6793349,1791520409.407
sequential,3,23,I976oI3s1jQ_02,13.500,13.46,324,162,11.9,11.9,0.0,2.42,0.08,0.38,8.71,9.21,9.17,0.27,119,117,"She said she was still working for two days, but she had a high fever and chills and decided to stay home recovering.",False,,False,,False,1,158,16.99,10.93,162,320,234,320,False,18.3,1791520423.0458028,1791520434.949
sequential,3,24,mNF1kg9azQw_07,9.385,9.35,225,113,11.67,11.67,0.0,2.65,0.08,0.2,8.59,8.89,8.86,0.13,122,120,"If you are interested in food or money, there is an article by MTV that lists different ways you can support your needs.",False,,False,,False,1,107,12.52,8.43,113,250,234,333,False,18.3,1791520444.4828846,1791520456.151
sequential,3,25,wPJ_InXF_I8_02,14.148,14.12,424,212,14.36,14.36,0.01,5.64,0.09,0.48,7.89,8.49,8.46,0.23,106,104,"AP also said Barack Obama has declared to endorse him, but Biden said he asked Obama to not endorse him.",False,,False,,False,1,202,22.84,15.84,212,297,234,522,False,18.3,1791520470.4473674,1791520484.811
sequential,3,26,TXyNxVRRPt8_01,6.273,6.24,188,94,10.23,10.23,0.0,3.74,0.07,0.19,6.06,6.34,6.31,0.16,88,86,We brought them to the Deaf Community Forum to bring their concerns to South Carolina.,False,,False,,False,1,81,10.28,6.29,94,249,234,333,False,18.3,1791520491.2170384,1791520501.451
sequential,3,27,sP6B4dlSOoc_04,5.573,5.54,167,84,8.17,8.17,0.0,3.72,0.07,0.18,4.08,4.35,4.33,0.1,46,44,Sometimes simple sidewalks can be dangerous.,False,,False,,False,1,73,8.79,5.92,84,403,234,757,False,18.3,1791520507.1700938,1791520515.344
sequential,3,28,fNT8a6e1gx8_06,2.770,2.74,83,42,7.27,7.27,0.0,4.25,0.07,0.18,2.74,3.0,2.99,0.02,40,38,Tend to ask your students to graduate.,False,,False,,False,1,24,5.28,2.9,42,817,234,998,False,18.3,1791520518.2761793,1791520525.55
sequential,3,29,TXyNxVRRPt8_03,4.505,4.48,135,68,7.55,7.55,0.0,4.19,0.07,0.2,2.99,3.27,3.26,0.09,34,32,Most people don't know about it.,False,,False,,False,1,51,7.83,4.97,68,399,234,781,False,18.3,1791520530.1644568,1791520537.712
sequential,3,30,z3jAMn3xpoM_02,8.142,8.11,244,122,10.59,10.59,0.0,4.06,0.08,0.22,6.09,6.41,6.39,0.12,83,81,One of them filmed on Facebook LIVE showing officers lying on the floor on blood.,False,,False,,False,1,111,12.77,8.57,122,397,234,688,False,18.3,1791520546.0248237,1791520556.619
sequential,3,31,f75G_hZMSHs_04,2.403,2.37,72,36,8.76,8.76,0.0,3.93,0.07,0.17,4.54,4.81,4.79,0.02,59,57,The company will likely start their wedding this weekend.,False,,False,,False,1,21,4.47,2.58,36,264,234,381,False,18.3,1791520559.213785,1791520567.977
sequential,3,32,oP2WdYlaflE_02,9.100,,,,,,,,,,,,,,,,,False,,True,292MB free,,0,,,,,,,292,,18.3,,
sequential,3,33,MJijm9kbdHA_01,14.500,,,,,,,,,,,,,,,,,False,,True,292MB free,,0,,,,,,,292,,18.3,,
sequential,3,34,5iRIeHAK9b0_02,10.400,,,,,,,,,,,,,,,,,False,,True,292MB free,,0,,,,,,,292,,18.3,,
sequential,3,35,sP6B4dlSOoc_05,10.978,,,,,,,,,,,,,,,,,False,,True,292MB free,,0,,,,,,,292,,18.3,,
sequential,3,36,J-0KHhPS_m4_03,5.000,,,,,,,,,,,,,,,,,False,,True,292MB free,,0,,,,,,,292,,18.3,,
sequential,3,37,LT0-h4P_xBY_06,8.400,,,,,,,,,,,,,,,,,False,,True,292MB free,,0,,,,,,,292,,18.3,,
sequential,3,38,oP2WdYlaflE_03,13.934,,,,,,,,,,,,,,,,,False,,True,292MB free,,0,,,,,,,292,,18.3,,
sequential,3,39,YR7BS6l_95E_05,11.600,,,,,,,,,,,,,,,,,False,,True,292MB free,,0,,,,,,,292,,18.3,,
```

## B.2  `overlap.csv`

```csv
condition,pass,order_index,clip_id,duration_s,capture_seconds,raw_frames,retained_frames_after_trim,post_cut_latency_s,worker_s,queue_wait_s,capture_end_to_perception_done_s,shubert_s,byt5_encoder_s,byt5_decode_s,byt5_stage_s,generate_s,other_s,decode_steps,output_bytes,output_text,oom_or_error,error,declined,decline_reason,deferred,queue_depth_at_cut,perception_frames_at_cut,perception_busy_seconds_cpu,embed_busy_seconds,embedded_frames,min_available_mb_during_utterance,min_available_mb_during_run,avail_mb_at_cut,first_after_warmup,warmup_seconds,cut_ts,done_ts
overlap,1,0,uZwKNtHx9FE_01,14.473,14.43,347,174,15.99,15.99,0.0,2.67,0.35,0.86,11.89,13.16,13.11,0.16,127,125,Most of the outbreaks are that children who do not have a vaccine who live in an ultra-orthodox Jewish community in New York.,False,,False,,False,1,169,19.91,13.24,174,1505,1505,1958,True,18.8,1791519081.5567055,1791519097.543
overlap,1,1,5iRIeHAK9b0_03,4.700,4.67,141,71,17.85,8.21,9.64,4.41,0.08,0.2,3.44,3.74,3.72,0.06,49,47,Children ranging in age from 3 to 29 years old.,False,,False,,False,2,62,7.38,15.4,71,1232,1232,1537,False,18.8,1791519087.9403975,1791519105.786
overlap,1,2,fNT8a6e1gx8_08,3.604,3.57,108,54,34.43,21.63,12.8,,0.15,0.23,6.27,7.44,6.65,14.19,55,53,It's just about our problems of expressing ourselves.,False,,False,,True,3,,,,,871,871,1533,False,18.8,1791519093.02522,1791519127.455
overlap,1,8,wPJ_InXF_I8_06,4.939,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,853,,18.8,,
overlap,1,9,I976oI3s1jQ_01,7.000,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,721,,18.8,,
overlap,1,10,jsOSzFyx8RA_03,12.713,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,545,,18.8,,
overlap,1,11,fNT8a6e1gx8_06,2.770,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,581,,18.8,,
overlap,1,12,J-0KHhPS_m4_03,5.000,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,581,,18.8,,
overlap,1,13,FND2JfzngVY_06,8.008,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,581,,18.8,,
overlap,1,14,I976oI3s1jQ_02,13.500,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,583,,18.8,,
overlap,1,15,ZgvLTS2zJdo_02,14.724,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,692,,18.8,,
overlap,1,16,YR7BS6l_95E_05,11.600,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,692,,18.8,,
overlap,1,17,NBpYolVo5WQ_06,3.042,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,692,,18.8,,
overlap,1,18,TXyNxVRRPt8_01,6.273,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,692,,18.8,,
overlap,1,19,5iRIeHAK9b0_02,10.400,,,,,,,,,,,,,,,,,False,,True,backlog 608 frames,,5,,,,,,,692,,18.8,,
overlap,1,3,sP6B4dlSOoc_05,10.978,10.95,329,165,60.3,38.33,21.97,,0.1,0.35,7.02,7.5,7.47,30.82,97,95,On Wednesday President Trump signed an executive order about trade reform and life kidney care.,False,,False,,True,4,,,,,545,545,1530,False,18.8,1791519105.4889882,1791519165.786
overlap,1,4,mNF1kg9azQw_03,11.387,11.35,273,137,55.22,7.93,47.29,0.77,0.08,0.32,6.58,7.0,6.97,0.16,92,90,Jennifer shared her experience with a woman who was currently in the photography of Floyd.,False,,False,,False,3,130,18.18,21.38,137,545,545,884,False,18.8,1791519118.5031722,1791519173.723
overlap,1,5,iMJ9CjBX7eo_02,4.305,4.28,129,65,56.78,7.48,49.31,0.17,0.07,0.18,6.94,7.24,7.19,0.07,79,77,Thank you for joining us at the Community Forum and we will host it together.,False,,False,,False,4,49,7.66,10.41,65,545,545,1031,False,18.8,1791519124.4475105,1791519181.232
overlap,1,22,sP6B4dlSOoc_04,5.573,,,,,,,,,,,,,,,,,False,,True,backlog 484 frames,,4,,,,,,,348,,18.8,,
overlap,1,23,z3jAMn3xpoM_02,8.142,,,,,,,,,,,,,,,,,False,,True,backlog 484 frames,,4,,,,,,,347,,18.8,,
overlap,1,24,uZwKNtHx9FE_07,12.471,,,,,,,,,,,,,,,,,False,,True,backlog 484 frames,,4,,,,,,,331,,18.8,,
overlap,1,25,D9HR4q0vbwE_05,2.044,,,,,,,,,,,,,,,,,False,,True,backlog 484 frames,,4,,,,,,,330,,18.8,,
overlap,1,26,f75G_hZMSHs_04,2.403,,,,,,,,,,,,,,,,,False,,True,backlog 484 frames,,4,,,,,,,330,,18.8,,
overlap,1,27,f75G_hZMSHs_02,7.775,,,,,,,,,,,,,,,,,False,,True,backlog 484 frames,,4,,,,,,,372,,18.8,,
overlap,1,28,ZgvLTS2zJdo_01,3.421,,,,,,,,,,,,,,,,,False,,True,backlog 484 frames,,4,,,,,,,372,,18.8,,
overlap,1,29,LT0-h4P_xBY_04,7.767,,,,,,,,,,,,,,,,,False,,True,backlog 484 frames,,4,,,,,,,372,,18.8,,
overlap,1,6,k5Gxbifw8s8_05,4.434,4.4,133,67,68.2,17.29,50.91,,0.07,0.19,3.65,3.93,3.91,13.36,35,33,And more young people use rabbit.,False,,False,,True,5,,,,,330,330,650,False,18.8,1791519130.3755708,1791519198.574
overlap,1,31,oP2WdYlaflE_02,9.100,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,404,,18.8,,
overlap,1,32,LT0-h4P_xBY_06,8.400,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,335,,18.8,,
overlap,1,33,mNF1kg9azQw_07,9.385,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,356,,18.8,,
overlap,1,34,TXyNxVRRPt8_03,4.505,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,337,,18.8,,
overlap,1,35,wH3lJm0Typ0_03,9.009,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,335,,18.8,,
overlap,1,36,oP2WdYlaflE_03,13.934,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,335,,18.8,,
overlap,1,37,I976oI3s1jQ_05,3.417,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,383,,18.8,,
overlap,1,38,k5Gxbifw8s8_03,4.700,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,363,,18.8,,
overlap,1,39,rSiciLyYOyI_07,2.878,,,,,,,,,,,,,,,,,False,,True,backlog 629 frames,,4,,,,,,,363,,18.8,,
overlap,1,7,MJijm9kbdHA_01,14.500,14.46,348,174,93.48,41.25,52.23,,0.11,0.44,8.15,8.73,8.69,32.52,102,100,Biden said Bernie had a vote multiple times against a brady bill that required stricter gun control.,False,,False,,True,5,,,,,254,254,841,False,18.8,1791519146.3533084,1791519239.833
overlap,1,20,mNF1kg9azQw_05,10.302,10.26,247,124,96.34,32.72,63.62,,0.09,0.23,6.37,6.72,6.69,26.0,89,87,I wonder how do you recommend people respond to someone who has been vlogging about it?,False,,False,,True,5,,,,,254,254,790,False,18.8,1791519176.214564,1791519272.558
overlap,1,21,WMoeSSvCyTU_03,7.908,7.98,237,119,93.17,6.47,86.7,1.62,0.09,0.2,4.36,4.67,4.64,0.19,62,60,The drug hydrogen is still going on and it is controversial.,False,,False,,False,5,92,16.5,11.14,119,254,254,427,False,18.8,1791519185.8613286,1791519279.036
overlap,1,30,wPJ_InXF_I8_02,14.148,14.12,424,212,75.73,10.37,65.36,1.5,0.09,0.5,7.98,8.6,8.57,0.27,106,104,"AP also said Barack Obama has declared to endorse him, but Biden said he asked Obama to not endorse him.",False,,False,,False,4,196,27.12,21.41,212,254,254,404,False,18.8,1791519213.7289283,1791519289.457
overlap,2,0,sP6B4dlSOoc_04,5.573,5.54,167,84,10.28,10.28,0.0,4.91,0.31,0.48,4.46,5.28,5.25,0.09,46,44,Sometimes simple sidewalks can be dangerous.,False,,False,,False,1,73,8.71,7.14,84,1712,1712,2744,True,23.3,1791519750.406984,1791519760.685
overlap,2,1,rSiciLyYOyI_07,2.878,2.85,69,35,9.52,3.74,5.78,2.11,0.09,0.16,1.34,1.6,1.58,0.03,19,17,That was amazing.,False,,False,,False,2,22,4.9,6.56,35,1678,1678,2038,False,23.3,1791519754.9337134,1791519764.456
overlap,2,2,YR7BS6l_95E_05,11.600,11.57,348,174,20.29,20.29,0.0,,,,,,,20.29,,,,True,"AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
",False,,True,3,,,,,1068,1068,1911,False,23.3,1791519768.0167289,1791519788.311
overlap,2,3,ZgvLTS2zJdo_01,3.421,3.38,82,41,15.24,0.0,15.24,,,,,,,0.0,,,,True,"AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
",False,,False,2,34,5.11,0.0,0,1068,1068,1420,False,23.3,1791519773.0780156,1791519788.316
overlap,2,4,sP6B4dlSOoc_05,10.978,10.95,329,165,2.65,0.0,2.64,,,,,,,0.01,,,,True,"AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/cuda/cuda-runtime-api/group__CUDART__TYPES.html for more information.
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.
",False,,False,3,150,20.24,0.0,0,1068,1063,1086,False,23.3,1791519785.6974635,1791519788.343
overlap,2,8,uZwKNtHx9FE_07,12.471,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,506,,23.3,,
overlap,2,9,wPJ_InXF_I8_02,14.148,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,279,,23.3,,
overlap,2,10,5iRIeHAK9b0_03,4.700,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,277,,23.3,,
overlap,2,11,TXyNxVRRPt8_03,4.505,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,271,,23.3,,
overlap,2,12,k5Gxbifw8s8_03,4.700,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,271,,23.3,,
overlap,2,13,FND2JfzngVY_06,8.008,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,308,,23.3,,
overlap,2,14,wH3lJm0Typ0_03,9.009,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,287,,23.3,,
overlap,2,15,I976oI3s1jQ_05,3.417,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,287,,23.3,,
overlap,2,16,NBpYolVo5WQ_06,3.042,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,287,,23.3,,
overlap,2,17,ZgvLTS2zJdo_02,14.724,,,,,,,,,,,,,,,,,False,,True,backlog 489 frames,,3,,,,,,,287,,23.3,,
overlap,2,5,mNF1kg9azQw_05,10.302,10.26,247,124,43.77,43.77,0.0,,0.15,0.24,6.45,7.42,6.84,36.35,89,87,I wonder how do you recommend people respond to someone who has been vlogging about it?,False,,False,,True,4,,,,,270,270,1266,False,23.3,1791519797.4740534,1791519841.245
overlap,2,18,D9HR4q0vbwE_05,2.044,,,,,,,,,,,,,,,,,False,,True,275MB free,,2,,,,,,,275,,23.3,,
overlap,2,19,oP2WdYlaflE_02,9.100,,,,,,,,,,,,,,,,,False,,True,273MB free,,2,,,,,,,273,,23.3,,
overlap,2,21,uZwKNtHx9FE_01,14.473,,,,,,,,,,,,,,,,,False,,True,backlog 491 frames,,3,,,,,,,784,,23.3,,
overlap,2,6,oP2WdYlaflE_03,13.934,13.9,418,209,43.03,14.81,28.23,2.16,0.1,0.47,11.81,12.43,12.38,0.22,150,148,And a limited number of Deaf and hard of hearing students can be recommended in primary reports such as the Department of Health and Human Services.,False,,False,,False,2,185,29.05,28.11,209,270,270,531,False,23.3,1791519813.0240262,1791519856.058
overlap,2,7,5iRIeHAK9b0_02,10.400,10.38,312,156,42.31,11.33,30.97,2.13,0.18,0.53,8.3,9.05,9.01,0.15,95,93,On Friday two men in California warned of child abuse in court for confession today (Friday).,False,,False,,False,3,145,16.51,26.74,156,270,270,617,False,23.3,1791519825.1127138,1791519867.418
overlap,2,25,wPJ_InXF_I8_06,4.939,,,,,,,,,,,,,,,,,False,,True,244MB free,,4,,,,,,,244,,23.3,,
overlap,2,26,z3jAMn3xpoM_02,8.142,,,,,,,,,,,,,,,,,False,,True,244MB free,,4,,,,,,,244,,23.3,,
overlap,2,27,fNT8a6e1gx8_06,2.770,,,,,,,,,,,,,,,,,False,,True,244MB free,,4,,,,,,,244,,23.3,,
overlap,2,28,f75G_hZMSHs_02,7.775,,,,,,,,,,,,,,,,,False,,True,243MB free,,4,,,,,,,243,,23.3,,
overlap,2,29,LT0-h4P_xBY_04,7.767,,,,,,,,,,,,,,,,,False,,True,243MB free,,4,,,,,,,243,,23.3,,
overlap,2,30,WMoeSSvCyTU_03,7.908,,,,,,,,,,,,,,,,,False,,True,243MB free,,4,,,,,,,243,,23.3,,
overlap,2,31,I976oI3s1jQ_02,13.500,,,,,,,,,,,,,,,,,False,,True,243MB free,,4,,,,,,,243,,23.3,,
overlap,2,33,I976oI3s1jQ_01,7.000,,,,,,,,,,,,,,,,,False,,True,242MB free,,5,,,,,,,242,,23.3,,
overlap,2,34,iMJ9CjBX7eo_02,4.305,,,,,,,,,,,,,,,,,False,,True,275MB free,,5,,,,,,,275,,23.3,,
overlap,2,35,k5Gxbifw8s8_05,4.434,,,,,,,,,,,,,,,,,False,,True,274MB free,,5,,,,,,,274,,23.3,,
overlap,2,36,J-0KHhPS_m4_03,5.000,,,,,,,,,,,,,,,,,False,,True,296MB free,,5,,,,,,,296,,23.3,,
overlap,2,37,mNF1kg9azQw_07,9.385,,,,,,,,,,,,,,,,,False,,True,322MB free,,5,,,,,,,322,,23.3,,
overlap,2,38,MJijm9kbdHA_01,14.500,,,,,,,,,,,,,,,,,False,,True,322MB free,,5,,,,,,,322,,23.3,,
overlap,2,39,jsOSzFyx8RA_03,12.713,,,,,,,,,,,,,,,,,False,,True,322MB free,,5,,,,,,,322,,23.3,,
overlap,2,20,LT0-h4P_xBY_06,8.400,8.37,252,126,54.04,39.66,14.38,,0.13,0.23,5.22,6.21,5.58,33.46,53,51,Do you have a race on the owner of the Ioniar Knox?,False,,False,,True,3,,,,,221,221,784,False,23.3,1791519853.0815902,1791519907.126
overlap,2,22,fNT8a6e1gx8_08,3.604,3.57,108,54,53.7,6.42,47.28,1.95,0.06,0.15,4.16,4.39,4.37,0.08,60,58,It's just that we have problems with expressing ourselves.,False,,False,,False,3,44,5.81,11.83,54,221,221,544,False,23.3,1791519859.84837,1791519913.548
overlap,2,23,f75G_hZMSHs_04,2.403,2.37,72,36,61.1,11.26,49.84,,0.06,0.14,3.62,3.84,3.82,7.43,52,50,The tourist wedding is scheduled for this weekend.,False,,False,,True,4,,,,,221,221,543,False,23.3,1791519863.737231,1791519924.839
overlap,2,24,TXyNxVRRPt8_01,6.273,6.24,188,94,80.23,26.88,53.35,,0.09,0.2,7.43,7.75,7.72,19.13,86,84,We bring them to the deaf community forum to bring their concerns to South Carolina.,False,,False,,True,5,,,,,221,221,244,False,23.3,1791519871.4926193,1791519951.719
overlap,2,32,mNF1kg9azQw_03,11.387,11.35,273,137,64.48,7.93,56.56,0.82,0.1,0.33,6.41,6.87,6.84,0.23,92,90,Jennifer shared her experience with a woman who was currently in the photography of Floyd.,False,,False,,False,5,130,18.54,20.46,137,237,221,257,False,23.3,1791519895.1673596,1791519959.65
overlap,3,0,mNF1kg9azQw_03,11.387,11.35,273,137,13.01,13.01,0.0,3.84,0.51,0.84,7.66,9.04,9.01,0.14,92,90,Jennifer shared her experience with a woman who was currently in the photography of Floyd.,False,,False,,False,1,128,18.48,10.96,137,1774,1774,2340,True,23.1,1791520636.3205132,1791520649.335
overlap,3,1,I976oI3s1jQ_01,7.000,6.96,168,84,19.66,15.24,4.41,5.51,0.09,0.35,9.17,9.64,9.61,0.09,86,84,My strategy on the chill was a little bit patient and I thought it would be perfect.,False,,False,,False,2,77,9.3,13.37,84,1322,1322,1798,False,23.1,1791520644.9531367,1791520664.61
overlap,3,2,5iRIeHAK9b0_03,4.700,4.67,141,71,39.18,25.66,13.52,,0.07,0.18,4.3,4.57,4.55,21.09,62,60,Children between the ages of three and twenty-nine are kids.,False,,False,,True,3,,,,,522,522,1616,False,23.1,1791520651.1403277,1791520690.316
overlap,3,9,uZwKNtHx9FE_01,14.473,,,,,,,,,,,,,,,,,False,,True,backlog 460 frames,,6,,,,,,,618,,23.1,,
overlap,3,10,D9HR4q0vbwE_05,2.044,,,,,,,,,,,,,,,,,False,,True,backlog 460 frames,,6,,,,,,,618,,23.1,,
overlap,3,11,k5Gxbifw8s8_03,4.700,,,,,,,,,,,,,,,,,False,,True,backlog 460 frames,,6,,,,,,,618,,23.1,,
overlap,3,12,iMJ9CjBX7eo_02,4.305,,,,,,,,,,,,,,,,,False,,True,backlog 460 frames,,6,,,,,,,618,,23.1,,
overlap,3,3,uZwKNtHx9FE_07,12.471,12.43,299,150,36.75,11.67,25.08,1.51,0.09,0.34,9.55,10.01,9.97,0.15,130,128,"They said the words were added without delay, were immediately introduced into the legislation to make sure that the work began.",False,,False,,False,3,138,17.34,26.56,150,513,513,1237,False,23.1,1791520665.2421558,1791520701.991
overlap,3,4,f75G_hZMSHs_02,7.775,7.74,233,117,41.93,14.57,27.36,1.8,0.1,0.23,12.25,12.62,12.58,0.15,124,122,The similar conversation between the police officers and his hometown includes the former police officer and his hometown.,False,,False,,False,3,95,15.13,16.15,117,513,513,897,False,23.1,1791520674.66129,1791520716.596
overlap,3,16,WMoeSSvCyTU_03,7.908,,,,,,,,,,,,,,,,,False,,True,backlog 472 frames,,7,,,,,,,542,,23.1,,
overlap,3,17,fNT8a6e1gx8_08,3.604,,,,,,,,,,,,,,,,,False,,True,backlog 472 frames,,7,,,,,,,378,,23.1,,
overlap,3,18,ZgvLTS2zJdo_02,14.724,,,,,,,,,,,,,,,,,False,,True,backlog 472 frames,,7,,,,,,,378,,23.1,,
overlap,3,19,wH3lJm0Typ0_03,9.009,,,,,,,,,,,,,,,,,False,,True,backlog 472 frames,,7,,,,,,,378,,23.1,,
overlap,3,5,NBpYolVo5WQ_06,3.042,3.0,73,37,54.32,16.86,37.46,,0.11,0.16,2.98,3.8,3.26,13.06,43,41,See you tomorrow and stay with the light!,False,,False,,True,4,,,,,378,378,588,False,23.1,1791520679.1747189,1791520733.495
overlap,3,21,jsOSzFyx8RA_03,12.713,,,,,,,,,,,,,,,,,False,,True,backlog 552 frames,,7,,,,,,,614,,23.1,,
overlap,3,22,k5Gxbifw8s8_05,4.434,,,,,,,,,,,,,,,,,False,,True,backlog 552 frames,,7,,,,,,,613,,23.1,,
overlap,3,23,I976oI3s1jQ_02,13.500,,,,,,,,,,,,,,,,,False,,True,backlog 552 frames,,7,,,,,,,463,,23.1,,
overlap,3,24,mNF1kg9azQw_07,9.385,,,,,,,,,,,,,,,,,False,,True,backlog 552 frames,,7,,,,,,,379,,23.1,,
overlap,3,25,wPJ_InXF_I8_02,14.148,,,,,,,,,,,,,,,,,False,,True,backlog 552 frames,,7,,,,,,,420,,23.1,,
overlap,3,26,TXyNxVRRPt8_01,6.273,,,,,,,,,,,,,,,,,False,,True,backlog 552 frames,,7,,,,,,,421,,23.1,,
overlap,3,27,sP6B4dlSOoc_04,5.573,,,,,,,,,,,,,,,,,False,,True,backlog 552 frames,,7,,,,,,,420,,23.1,,
overlap,3,6,wPJ_InXF_I8_06,4.939,4.91,148,74,68.23,20.32,47.9,,0.07,0.21,3.57,3.87,3.85,16.46,52,50,the students storytelling comprehension abilities.,False,,False,,True,5,,,,,243,243,522,False,23.1,1791520685.5964522,1791520753.825
overlap,3,28,fNT8a6e1gx8_06,2.770,,,,,,,,,,,,,,,,,False,,True,backlog 478 frames,,6,,,,,,,462,,23.1,,
overlap,3,29,TXyNxVRRPt8_03,4.505,,,,,,,,,,,,,,,,,False,,True,backlog 478 frames,,6,,,,,,,451,,23.1,,
overlap,3,30,z3jAMn3xpoM_02,8.142,,,,,,,,,,,,,,,,,False,,True,backlog 478 frames,,6,,,,,,,608,,23.1,,
overlap,3,31,f75G_hZMSHs_04,2.403,,,,,,,,,,,,,,,,,False,,True,backlog 478 frames,,6,,,,,,,603,,23.1,,
overlap,3,32,oP2WdYlaflE_02,9.100,,,,,,,,,,,,,,,,,False,,True,backlog 478 frames,,6,,,,,,,542,,23.1,,
overlap,3,33,MJijm9kbdHA_01,14.500,,,,,,,,,,,,,,,,,False,,True,backlog 478 frames,,6,,,,,,,410,,23.1,,
overlap,3,34,5iRIeHAK9b0_02,10.400,,,,,,,,,,,,,,,,,False,,True,backlog 478 frames,,6,,,,,,,363,,23.1,,
overlap,3,35,sP6B4dlSOoc_05,10.978,,,,,,,,,,,,,,,,,False,,True,backlog 478 frames,,6,,,,,,,363,,23.1,,
overlap,3,7,I976oI3s1jQ_05,3.417,3.38,82,41,76.02,12.68,63.34,,0.06,0.14,3.58,3.8,3.78,8.89,51,49,I want to make sure that everyone understands me.,False,,False,,True,6,,,,,243,243,555,False,23.1,1791520690.486291,1791520766.511
overlap,3,37,LT0-h4P_xBY_06,8.400,,,,,,,,,,,,,,,,,False,,True,backlog 512 frames,,6,,,,,,,750,,23.1,,
overlap,3,38,oP2WdYlaflE_03,13.934,,,,,,,,,,,,,,,,,False,,True,backlog 512 frames,,6,,,,,,,611,,23.1,,
overlap,3,39,YR7BS6l_95E_05,11.600,,,,,,,,,,,,,,,,,False,,True,backlog 512 frames,,6,,,,,,,526,,23.1,,
overlap,3,8,ZgvLTS2zJdo_01,3.421,3.38,82,41,86.32,15.19,71.13,,0.06,0.14,5.71,5.94,5.92,9.25,81,79,The first is to open and provide resources for deaf and hard of hearing people.,False,,False,,True,6,,,,,243,243,618,False,23.1,1791520695.3814273,1791520781.701
overlap,3,13,mNF1kg9azQw_05,10.302,10.27,247,124,75.26,6.88,68.38,1.78,0.07,0.2,4.67,4.96,4.94,0.13,67,65,I'm wondering how to send it to another person who sent it to me.,False,,False,,False,6,118,14.36,21.06,124,243,243,1486,False,23.1,1791520713.3285844,1791520788.589
overlap,3,14,rSiciLyYOyI_07,2.878,2.84,69,35,82.62,11.69,70.94,,0.07,0.16,3.22,3.46,3.45,8.22,47,45,It was an amazing and never experienced that.,False,,False,,True,7,,,,,243,243,1168,False,23.1,1791520717.6876452,1791520800.312
overlap,3,15,FND2JfzngVY_06,8.008,7.98,240,120,79.69,6.73,72.97,1.69,0.08,0.21,4.61,4.91,4.9,0.12,55,53,The daylight savings will be back on March 8 at 2 AM.,False,,False,,False,7,108,14.31,14.46,120,243,243,490,False,23.1,1791520727.3530123,1791520807.046
overlap,3,20,LT0-h4P_xBY_04,7.767,7.74,233,117,97.61,33.16,64.45,,0.09,0.22,7.93,8.28,8.25,24.88,98,96,"Of course, I want to clearly warn you that I just shared with the people who are self-esteeming.",False,,False,,True,7,,,,,226,226,615,False,23.1,1791520742.629816,1791520840.24
overlap,3,36,J-0KHhPS_m4_03,5.000,4.97,150,75,88.54,20.0,68.54,,0.07,0.19,4.25,4.54,4.51,15.46,60,58,Here's a question from the next question: How did it stop?,False,,False,,True,6,,,,,226,226,736,False,23.1,1791520771.7143593,1791520860.258
```

## B.3  `confirm_w1.csv` — `PERCEPTION_WORKERS=1`

```csv
condition,pass,order_index,clip_id,duration_s,capture_seconds,raw_frames,retained_frames_after_trim,post_cut_latency_s,worker_s,queue_wait_s,capture_end_to_perception_done_s,shubert_s,byt5_encoder_s,byt5_decode_s,byt5_stage_s,generate_s,other_s,decode_steps,output_bytes,output_text,oom_or_error,error,declined,decline_reason,deferred,queue_depth_at_cut,perception_frames_at_cut,perception_busy_seconds_cpu,embed_busy_seconds,embedded_frames,min_available_mb_during_utterance,min_available_mb_during_run,avail_mb_at_cut,first_after_warmup,warmup_seconds,cut_ts,done_ts
sequential,1,0,uZwKNtHx9FE_01,14.473,14.43,347,174,19.43,19.43,0.0,8.01,0.26,0.65,10.38,11.33,11.29,0.09,133,131,Most of the outbreaks are that children who do not have a vaccine who live in strict ultra-orthodox Jewish communities in New York.,False,,False,,False,1,112,20.75,12.35,174,2304,2304,2450,True,28.4,1791527105.9275932,1791527125.359
sequential,1,1,5iRIeHAK9b0_03,4.700,4.67,141,71,10.25,10.25,0.0,5.59,0.08,0.17,4.29,4.55,4.53,0.1,62,60,Children between the ages of three and twenty-nine are kids.,False,,False,,False,1,36,8.01,5.29,71,2067,2067,2305,False,28.4,1791527130.1206372,1791527140.372
sequential,1,2,fNT8a6e1gx8_08,3.604,3.57,108,54,9.53,9.53,0.0,4.38,0.06,0.14,4.87,5.09,5.07,0.06,55,53,It's just about our problems of expressing ourselves.,False,,False,,False,1,28,6.4,3.58,54,2003,2003,2237,False,28.4,1791527144.0410728,1791527153.571
sequential,1,3,sP6B4dlSOoc_05,10.978,10.95,329,165,18.38,18.38,0.0,10.58,0.1,0.4,7.08,7.61,7.58,0.19,97,95,On Wednesday President Trump signed an executive order about trade reform and life kidney care.,False,,False,,False,1,90,19.43,11.12,165,1903,1903,1937,False,28.4,1791527164.6214333,1791527183.005
sequential,1,4,mNF1kg9azQw_03,11.387,11.35,273,137,16.68,16.68,0.0,9.86,0.09,0.32,6.2,6.63,6.61,0.19,85,83,"Next we will talk about the next step, the next step, the next step, the next step.",False,,False,,False,1,75,19.78,9.54,137,1771,1771,1788,False,28.4,1791527194.467904,1791527211.144
sequential,1,5,iMJ9CjBX7eo_02,4.305,4.27,129,65,12.77,12.77,0.0,6.64,0.08,0.18,5.75,6.03,6.0,0.09,84,82,Thank you for joining us at the Community Forum and I will be hosting it together.,False,,False,,False,1,30,8.73,4.41,65,1632,1632,1934,False,28.4,1791527215.5131748,1791527228.28
sequential,1,6,k5Gxbifw8s8_05,4.434,4.4,133,67,9.23,9.23,0.0,5.19,0.09,0.21,3.66,3.97,3.96,0.07,35,33,And more young people use rabbit.,False,,False,,False,1,38,7.48,4.52,67,1548,1548,1907,False,28.4,1791527232.7897134,1791527242.018
sequential,1,7,MJijm9kbdHA_01,14.500,14.46,348,174,14.16,14.16,0.0,5.46,0.09,0.41,7.93,8.46,8.43,0.23,102,100,Biden said Bernie had a vote multiple times against a brady bill that required stricter gun control.,False,,False,,False,1,136,18.04,12.32,174,1502,1502,1590,False,28.4,1791527256.5762193,1791527270.734
sequential,1,8,wPJ_InXF_I8_06,4.939,4.91,148,74,11.82,11.81,0.0,7.82,0.07,0.2,3.62,3.91,3.89,0.09,52,50,the students storytelling comprehension abilities.,False,,False,,False,1,30,11.21,4.93,74,1423,1423,1755,False,28.4,1791527275.7468746,1791527287.562
sequential,1,9,I976oI3s1jQ_01,7.000,6.99,168,84,11.63,11.63,0.0,5.21,0.07,0.19,6.03,6.32,6.29,0.11,88,86,I feel like it's a chill and I've been very patient with it and I think it's possible.,False,,False,,False,1,53,10.69,6.16,84,1394,1394,1434,False,28.4,1791527294.656405,1791527306.291
sequential,1,10,jsOSzFyx8RA_03,12.713,12.68,381,191,25.39,25.39,0.0,11.28,0.1,0.42,13.36,13.92,13.88,0.19,168,166,"Or the show isn't a lot about the experience, but it's about who you are as a person and how you through savingstories throughout life, photography, as a Deaf person.",False,,False,,False,1,112,21.85,12.88,191,1282,1278,1412,False,28.4,1791527319.053848,1791527344.444
sequential,1,11,fNT8a6e1gx8_06,2.770,2.74,83,42,11.06,11.06,0.0,4.91,0.07,0.17,5.83,6.09,6.07,0.05,85,83,"I feel like again, the students have been asking me to go to school for their time.",False,,False,,False,1,15,5.82,2.83,42,1451,1278,1657,False,28.4,1791527347.2775507,1791527358.337
sequential,1,12,J-0KHhPS_m4_03,5.000,4.97,150,75,10.62,10.62,0.0,6.02,0.07,0.17,4.23,4.49,4.47,0.11,60,58,Here's a question from the next question: How did it stop?,False,,False,,False,1,38,9.33,4.97,75,1189,1189,1645,False,28.4,1791527363.364892,1791527373.987
sequential,1,13,FND2JfzngVY_06,8.008,7.98,240,120,12.59,12.59,0.0,7.61,0.09,0.24,4.47,4.81,4.79,0.17,55,53,The daylight savings will be back on March 8 at 2 AM.,False,,False,,False,1,63,13.89,8.05,120,1098,1098,1369,False,28.4,1791527382.1547446,1791527394.743
sequential,1,14,I976oI3s1jQ_02,13.500,13.46,324,162,15.4,15.4,0.0,6.31,0.1,0.36,8.4,8.9,8.86,0.19,113,111,"He said he was still working for two days, but on a high fever and chills, he decided to stay home and recover.",False,,False,,False,1,122,17.62,10.93,162,1029,1029,1063,False,28.4,1791527408.296249,1791527423.697
sequential,1,15,ZgvLTS2zJdo_02,14.724,14.68,353,177,14.86,14.86,0.0,7.57,0.11,0.42,6.54,7.09,7.07,0.2,71,69,I want to share a little bit of a story with you about one more time.,False,,False,,False,1,127,20.95,11.53,177,1002,939,1051,False,28.4,1791527438.4813092,1791527453.345
sequential,1,16,YR7BS6l_95E_05,11.600,11.57,348,174,18.12,18.12,0.0,9.68,0.09,0.42,7.64,8.18,8.15,0.25,99,97,The helicopter travels to the area and flies to a shelter to help provide supplies with supplies.,False,,False,,False,1,103,19.46,12.33,174,836,836,1056,False,28.4,1791527465.0104678,1791527483.13
sequential,1,17,NBpYolVo5WQ_06,3.042,3.0,73,37,7.98,7.98,0.0,4.73,0.07,0.16,2.99,3.23,3.22,0.02,43,41,See you tomorrow and stay with the light!,False,,False,,False,1,17,5.66,2.64,37,924,836,1168,False,28.4,1791527486.2517784,1791527494.236
sequential,1,18,TXyNxVRRPt8_01,6.273,6.24,188,94,14.39,14.38,0.0,6.5,0.09,0.19,7.46,7.76,7.74,0.12,86,84,We bring them to the deaf community forum to bring their concerns to South Carolina.,False,,False,,False,1,54,10.71,6.33,94,709,709,1077,False,28.4,1791527500.5674741,1791527514.954
sequential,1,19,5iRIeHAK9b0_02,10.400,10.37,312,156,17.36,17.36,0.0,8.43,0.08,0.36,8.21,8.69,8.66,0.25,117,115,A California male couple called for child abuse Friday in a court called for police to plead guilty to child abuse.,False,,False,,False,1,92,16.81,10.98,156,726,709,1173,False,28.4,1791527525.37228,1791527542.737
sequential,1,20,mNF1kg9azQw_05,10.302,10.26,247,124,14.72,14.72,0.0,7.99,0.09,0.24,6.27,6.63,6.6,0.1,89,87,I wonder how do you recommend people respond to someone who has been vlogging about it?,False,,False,,False,1,78,16.24,8.44,124,639,639,642,False,28.4,1791527553.0911193,1791527567.809
sequential,1,21,WMoeSSvCyTU_03,7.908,7.88,237,119,14.37,14.36,0.0,9.72,0.08,0.21,4.25,4.56,4.54,0.08,62,60,The drug hydrogen is still going on and it is controversial.,False,,False,,False,1,52,15.91,7.96,119,658,639,728,False,28.4,1791527575.802843,1791527590.168
sequential,1,22,sP6B4dlSOoc_04,5.573,5.54,167,84,10.22,10.21,0.0,5.24,0.07,0.18,4.58,4.84,4.83,0.14,46,44,Sometimes simple sidewalks can be dangerous.,False,,False,,False,1,47,9.2,5.66,84,740,639,898,False,28.4,1791527595.7949057,1791527606.01
sequential,1,23,z3jAMn3xpoM_02,8.142,8.11,244,122,14.75,14.75,0.0,8.41,0.09,0.22,5.89,6.23,6.2,0.11,83,81,One of them filmed on Facebook LIVE showing officers lying on the floor on blood.,False,,False,,False,1,65,13.77,11.03,122,513,513,532,False,28.4,1791527614.240999,1791527628.992
sequential,1,24,uZwKNtHx9FE_07,12.471,12.43,299,150,17.71,17.71,0.0,8.17,0.08,0.36,8.92,9.41,9.37,0.13,126,124,"They said the words were added without delays, were immediately introduced into the legislation to make sure the work began.",False,,False,,False,1,99,18.69,10.6,150,699,513,1116,False,28.4,1791527641.5278184,1791527659.237
sequential,1,25,D9HR4q0vbwE_05,2.044,2.0,49,25,5.86,5.86,0.0,4.1,0.09,0.17,1.48,1.75,1.74,0.02,20,18,How are you doing?,False,,False,,False,1,10,3.73,2.32,25,599,513,656,False,28.4,1791527661.3401418,1791527667.204
sequential,1,26,f75G_hZMSHs_04,2.403,2.37,72,36,8.89,8.89,0.0,4.96,0.07,0.16,3.65,3.91,3.89,0.02,52,50,The tourist wedding is scheduled for this weekend.,False,,False,,False,1,18,4.72,3.24,36,488,488,643,False,28.4,1791527669.6677802,1791527678.558
sequential,1,27,f75G_hZMSHs_02,7.775,7.74,233,117,15.45,15.45,0.0,6.1,0.1,0.23,8.83,9.19,9.15,0.16,126,124,The similar conversations between the police officers and his hometown are including the previous officers and his hometown.,False,,False,,False,1,73,12.38,7.87,117,830,488,1225,False,28.4,1791527686.387122,1791527701.839
sequential,1,28,ZgvLTS2zJdo_01,3.421,3.38,82,41,10.79,10.79,0.0,4.82,0.09,0.17,5.67,5.95,5.93,0.02,81,79,The first is to open and provide resources for deaf and hard of hearing people.,False,,False,,False,1,22,5.98,3.39,41,549,488,695,False,28.4,1791527705.287577,1791527716.08
sequential,1,29,LT0-h4P_xBY_04,7.767,7.74,233,117,17.83,17.83,0.0,9.52,0.08,0.22,7.84,8.17,8.14,0.14,98,96,"Of course, I want to clearly warn you that I just shared with the people who are self-esteeming.",False,,False,,False,1,57,15.45,9.71,117,527,488,601,False,28.4,1791527723.9200797,1791527741.747
sequential,1,30,wPJ_InXF_I8_02,14.148,14.12,424,212,21.54,21.53,0.0,12.8,0.1,0.49,7.89,8.51,8.48,0.22,106,104,"AP also said Barack Obama has declared to endorse him, but Biden said he asked Obama to not endorse him.",False,,False,,False,1,117,24.15,16.42,212,500,488,812,False,28.4,1791527755.9463162,1791527777.482
sequential,1,31,oP2WdYlaflE_02,9.100,9.07,273,137,20.6,20.59,0.0,11.11,0.08,0.34,8.81,9.27,9.24,0.22,113,111,The expected percentage of a person from the Institute for Health is falling from the University of Washington.,False,,False,,False,1,64,18.1,9.66,137,601,488,1105,False,28.4,1791527786.6611853,1791527807.257
sequential,1,32,LT0-h4P_xBY_06,8.400,8.37,252,126,17.6,17.6,0.0,12.05,0.09,0.23,5.11,5.44,5.42,0.11,53,51,Do you have a race on the owner of the Ioniar Knox?,False,,False,,False,1,55,17.62,11.65,126,562,488,626,False,28.4,1791527815.7314458,1791527833.331
sequential,1,33,mNF1kg9azQw_07,9.385,9.35,225,113,15.28,15.28,0.0,6.25,0.07,0.2,8.57,8.88,8.84,0.16,122,120,"If you are interested in food or money, there is an article by MTV that lists different ways you can support your needs.",False,,False,,False,1,83,12.94,11.08,113,640,488,651,False,28.4,1791527842.783013,1791527858.068
sequential,1,34,TXyNxVRRPt8_03,4.505,4.48,135,68,10.93,10.93,0.0,6.63,0.08,0.21,3.92,4.23,4.2,0.07,53,51,Most people don't know about the same NAD programs.,False,,False,,False,1,32,8.34,6.22,68,539,488,639,False,28.4,1791527862.64376,1791527873.574
sequential,1,35,wH3lJm0Typ0_03,9.009,8.98,270,135,18.91,18.91,0.0,9.09,0.1,0.33,9.26,9.72,9.69,0.09,111,109,"At the end of the game, there were also people who cheered their fans with cheers and critics canceling them.",False,,False,,False,1,73,16.12,9.39,135,872,488,1180,False,28.4,1791527882.6487424,1791527901.557
sequential,1,36,oP2WdYlaflE_03,13.934,13.9,418,209,25.95,25.95,0.0,14.4,0.11,0.46,10.71,11.32,11.28,0.23,138,136,The study also limits the number of community members to primary reports such as the Department of Health and People and Human Services.,False,,False,,False,1,104,25.83,18.59,209,485,485,497,False,28.4,1791527915.5571601,1791527941.505
sequential,1,37,I976oI3s1jQ_05,3.417,3.38,82,41,9.34,9.34,0.0,5.44,0.07,0.16,3.63,3.88,3.86,0.02,51,49,I want to make sure that everyone understands me.,False,,False,,False,1,21,5.79,3.87,41,471,471,568,False,28.4,1791527944.971636,1791527954.309
sequential,1,38,k5Gxbifw8s8_03,4.700,4.67,141,71,9.46,9.46,0.0,5.22,0.08,0.2,3.87,4.16,4.15,0.07,50,48,This job opportunity for Deaf people is growing.,False,,False,,False,1,41,7.86,4.98,71,573,471,642,False,28.4,1791527959.0676837,1791527968.524
sequential,1,39,rSiciLyYOyI_07,2.878,2.85,69,35,8.34,8.34,0.0,4.92,0.08,0.15,3.16,3.4,3.39,0.01,47,45,It was an amazing and never experienced that.,False,,False,,False,1,19,5.64,2.57,35,545,471,733,False,28.4,1791527971.4665895,1791527979.806
```

## B.4  `confirm_defer.csv` — `MAX_LIVE_STREAMS=0`

```csv
condition,pass,order_index,clip_id,duration_s,capture_seconds,raw_frames,retained_frames_after_trim,post_cut_latency_s,worker_s,queue_wait_s,capture_end_to_perception_done_s,shubert_s,byt5_encoder_s,byt5_decode_s,byt5_stage_s,generate_s,other_s,decode_steps,output_bytes,output_text,oom_or_error,error,declined,decline_reason,deferred,queue_depth_at_cut,perception_frames_at_cut,perception_busy_seconds_cpu,embed_busy_seconds,embedded_frames,min_available_mb_during_utterance,min_available_mb_during_run,avail_mb_at_cut,first_after_warmup,warmup_seconds,cut_ts,done_ts
sequential,1,0,uZwKNtHx9FE_01,14.473,14.43,347,174,48.23,48.23,0.0,,0.36,0.9,12.17,13.47,13.42,34.76,133,131,Most of the outbreaks are that children who do not have a vaccine who live in strict ultra-orthodox Jewish communities in New York.,False,,False,,True,1,,,,,1957,1957,3750,True,18.4,1791528651.2513044,1791528699.481
sequential,1,1,5iRIeHAK9b0_03,4.700,4.67,141,71,18.25,18.25,0.0,,0.06,0.18,4.21,4.47,4.45,13.78,62,60,Children between the ages of three and twenty-nine are kids.,False,,False,,True,1,,,,,1861,1861,2284,False,18.4,1791528704.163323,1791528722.41
sequential,1,2,fNT8a6e1gx8_08,3.604,3.57,108,54,15.96,15.96,0.0,,0.07,0.17,4.84,5.1,5.08,10.86,55,53,It's just about our problems of expressing ourselves.,False,,False,,True,1,,,,,1489,1489,1858,False,18.4,1791528726.0004938,1791528741.96
sequential,1,3,sP6B4dlSOoc_05,10.978,10.95,329,165,41.01,41.01,0.0,,0.1,0.35,6.91,7.4,7.36,33.62,97,95,On Wednesday President Trump signed an executive order about trade reform and life kidney care.,False,,False,,True,1,,,,,1329,1329,1899,False,18.4,1791528752.922062,1791528793.936
sequential,1,4,mNF1kg9azQw_03,11.387,11.35,273,137,38.06,38.06,0.0,,0.09,0.33,6.3,6.76,6.73,31.31,85,83,"Next we will talk about the next step, the next step, the next step, the next step.",False,,False,,True,1,,,,,1260,1260,1361,False,18.4,1791528805.3043082,1791528843.369
sequential,1,5,iMJ9CjBX7eo_02,4.305,4.27,129,65,20.03,20.02,0.0,,0.07,0.18,5.83,6.11,6.08,13.92,84,82,Thank you for joining us at the Community Forum and I will be hosting it together.,False,,False,,True,1,,,,,1260,1260,1339,False,18.4,1791528847.6572394,1791528867.683
sequential,1,6,k5Gxbifw8s8_05,4.434,4.4,133,67,16.9,16.9,0.0,,0.08,0.18,3.7,3.97,3.96,12.93,35,33,And more young people use rabbit.,False,,False,,True,1,,,,,1241,1241,1588,False,18.4,1791528872.1095116,1791528889.012
sequential,1,7,MJijm9kbdHA_01,14.500,14.46,348,174,41.48,41.47,0.0,,0.09,0.39,8.04,8.55,8.52,32.92,102,100,Biden said Bernie had a vote multiple times against a brady bill that required stricter gun control.,False,,False,,True,1,,,,,1131,1131,1611,False,18.4,1791528903.4885347,1791528944.964
sequential,1,8,wPJ_InXF_I8_06,4.939,4.91,148,74,20.79,20.79,0.0,,0.06,0.19,3.58,3.84,3.83,16.95,52,50,the students storytelling comprehension abilities.,False,,False,,True,1,,,,,1095,1095,1173,False,18.4,1791528949.89505,1791528970.689
sequential,1,9,I976oI3s1jQ_01,7.000,6.96,168,84,23.78,23.78,0.0,,0.07,0.2,6.17,6.48,6.45,17.3,88,86,I feel like it's a chill and I've been very patient with it and I think it's possible.,False,,False,,True,1,,,,,978,978,1456,False,18.4,1791528977.6635573,1791529001.441
sequential,1,10,jsOSzFyx8RA_03,12.713,12.68,381,191,51.81,51.8,0.0,,0.12,0.43,13.29,13.89,13.84,37.92,168,166,"Or the show isn't a lot about the experience, but it's about who you are as a person and how you through savingstories throughout life, photography, as a Deaf person.",False,,False,,True,1,,,,,930,930,1040,False,18.4,1791529014.142058,1791529065.947
sequential,1,11,fNT8a6e1gx8_06,2.770,2.74,83,42,15.87,15.86,0.0,,0.05,0.14,5.81,6.03,6.0,9.84,85,83,"I feel like again, the students have been asking me to go to school for their time.",False,,False,,True,1,,,,,965,930,1074,False,18.4,1791529068.7102427,1791529084.576
sequential,1,12,J-0KHhPS_m4_03,5.000,4.97,150,75,19.86,19.85,0.01,,0.08,0.2,4.24,4.56,4.53,15.29,60,58,Here's a question from the next question: How did it stop?,False,,False,,True,1,,,,,1373,930,1005,False,18.4,1791529089.5608683,1791529109.42
sequential,1,13,FND2JfzngVY_06,8.008,7.98,240,120,30.29,30.29,0.0,,0.08,0.21,4.55,4.85,4.83,25.43,55,53,The daylight savings will be back on March 8 at 2 AM.,False,,False,,True,1,,,,,866,866,1382,False,18.4,1791529117.4208791,1791529147.707
sequential,1,14,I976oI3s1jQ_02,13.500,13.46,324,162,41.96,41.96,0.0,,0.09,0.35,8.73,9.21,9.17,32.75,113,111,"He said he was still working for two days, but on a high fever and chills, he decided to stay home and recover.",False,,False,,True,1,,,,,839,839,977,False,18.4,1791529161.1818867,1791529203.139
sequential,1,15,ZgvLTS2zJdo_02,14.724,14.68,353,177,44.52,44.52,0.0,,0.1,0.41,7.2,7.73,7.71,36.8,71,69,I want to share a little bit of a story with you about one more time.,False,,False,,True,1,,,,,968,839,908,False,18.4,1791529217.8495758,1791529262.374
sequential,1,16,YR7BS6l_95E_05,11.600,11.57,348,174,46.19,46.18,0.0,,0.1,0.41,8.15,8.7,8.67,37.49,99,97,The helicopter travels to the area and flies to a shelter to help provide supplies with supplies.,False,,False,,True,1,,,,,793,793,1010,False,18.4,1791529273.9687188,1791529320.154
sequential,1,17,NBpYolVo5WQ_06,3.042,3.0,73,37,12.66,12.66,0.01,,0.05,0.14,2.99,3.2,3.18,9.46,43,41,See you tomorrow and stay with the light!,False,,False,,True,1,,,,,853,793,958,False,18.4,1791529323.1788712,1791529335.843
sequential,1,18,TXyNxVRRPt8_01,6.273,6.24,188,94,28.73,28.72,0.01,,0.07,0.17,7.46,7.74,7.71,20.97,86,84,We bring them to the deaf community forum to bring their concerns to South Carolina.,False,,False,,True,1,,,,,785,785,888,False,18.4,1791529342.0993574,1791529370.825
sequential,1,19,5iRIeHAK9b0_02,10.400,10.37,312,156,39.24,39.24,0.0,,0.08,0.36,8.62,9.1,9.06,30.14,117,115,A California male couple called for child abuse Friday in a court called for police to plead guilty to child abuse.,False,,False,,True,1,,,,,919,785,857,False,18.4,1791529381.2135406,1791529420.454
sequential,1,20,mNF1kg9azQw_05,10.302,10.26,247,124,35.6,35.6,0.0,,0.08,0.2,6.38,6.69,6.66,28.91,89,87,I wonder how do you recommend people respond to someone who has been vlogging about it?,False,,False,,True,1,,,,,953,785,1067,False,18.4,1791529430.736104,1791529466.336
sequential,1,21,WMoeSSvCyTU_03,7.908,7.88,237,119,33.17,33.17,0.0,,0.07,0.2,4.26,4.54,4.52,28.63,62,60,The drug hydrogen is still going on and it is controversial.,False,,False,,True,1,,,,,886,785,1079,False,18.4,1791529474.2354798,1791529507.405
sequential,1,22,sP6B4dlSOoc_04,5.573,5.54,167,84,23.14,23.13,0.0,,0.07,0.18,4.6,4.88,4.85,18.26,46,44,Sometimes simple sidewalks can be dangerous.,False,,False,,True,1,,,,,943,785,1004,False,18.4,1791529512.9624119,1791529536.102
sequential,1,23,z3jAMn3xpoM_02,8.142,8.11,244,122,32.76,32.76,0.0,,0.07,0.21,5.9,6.21,6.19,26.55,83,81,One of them filmed on Facebook LIVE showing officers lying on the floor on blood.,False,,False,,True,1,,,,,764,764,1074,False,18.4,1791529544.2305875,1791529576.992
sequential,1,24,uZwKNtHx9FE_07,12.471,12.43,299,150,41.55,41.55,0.0,,0.08,0.35,9.36,9.83,9.79,31.72,126,124,"They said the words were added without delays, were immediately introduced into the legislation to make sure the work began.",False,,False,,True,1,,,,,984,764,862,False,18.4,1791529589.4390287,1791529630.988
sequential,1,25,D9HR4q0vbwE_05,2.044,2.0,49,25,7.62,7.61,0.01,,0.07,0.14,1.44,1.66,1.65,5.95,20,18,How are you doing?,False,,False,,True,1,,,,,1010,764,1096,False,18.4,1791529633.0100062,1791529640.628
sequential,1,26,f75G_hZMSHs_04,2.403,2.37,72,36,12.19,12.19,0.0,,0.05,0.14,3.7,3.9,3.89,8.29,52,50,The tourist wedding is scheduled for this weekend.,False,,False,,True,1,,,,,960,764,1055,False,18.4,1791529643.0198586,1791529655.212
sequential,1,27,f75G_hZMSHs_02,7.775,7.74,233,117,34.84,34.84,0.0,,0.07,0.2,8.89,9.2,9.17,25.64,126,124,The similar conversations between the police officers and his hometown are including the previous officers and his hometown.,False,,False,,True,1,,,,,922,764,1009,False,18.4,1791529662.9722016,1791529697.813
sequential,1,28,ZgvLTS2zJdo_01,3.421,3.38,82,41,16.26,16.26,0.0,,0.06,0.14,5.71,5.93,5.9,10.33,81,79,The first is to open and provide resources for deaf and hard of hearing people.,False,,False,,True,1,,,,,946,764,1068,False,18.4,1791529701.218122,1791529717.475
sequential,1,29,LT0-h4P_xBY_04,7.767,7.74,233,117,35.8,35.79,0.0,,0.08,0.2,7.91,8.22,8.19,27.57,98,96,"Of course, I want to clearly warn you that I just shared with the people who are self-esteeming.",False,,False,,True,1,,,,,824,764,1002,False,18.4,1791529725.225409,1791529761.021
sequential,1,30,wPJ_InXF_I8_02,14.148,14.12,424,212,52.61,52.61,0.0,,0.11,0.53,8.39,9.06,9.03,43.55,106,104,"AP also said Barack Obama has declared to endorse him, but Biden said he asked Obama to not endorse him.",False,,False,,True,1,,,,,718,718,866,False,18.4,1791529775.15151,1791529827.766
sequential,1,31,oP2WdYlaflE_02,9.100,9.07,273,137,41.4,41.4,0.0,,0.08,0.34,8.87,9.32,9.29,32.08,113,111,The expected percentage of a person from the Institute for Health is falling from the University of Washington.,False,,False,,True,1,,,,,754,718,901,False,18.4,1791529836.851124,1791529878.252
sequential,1,32,LT0-h4P_xBY_06,8.400,8.37,252,126,35.39,35.39,0.0,,0.07,0.21,5.13,5.43,5.41,29.96,53,51,Do you have a race on the owner of the Ioniar Knox?,False,,False,,True,1,,,,,706,706,918,False,18.4,1791529886.6375601,1791529922.03
sequential,1,33,mNF1kg9azQw_07,9.385,9.34,225,113,33.53,33.53,0.0,,0.07,0.19,8.5,8.8,8.77,24.73,122,120,"If you are interested in food or money, there is an article by MTV that lists different ways you can support your needs.",False,,False,,True,1,,,,,777,706,872,False,18.4,1791529931.3932736,1791529964.923
sequential,1,34,TXyNxVRRPt8_03,4.505,4.47,135,68,19.49,19.49,0.0,,0.06,0.18,3.94,4.2,4.18,15.3,53,51,Most people don't know about the same NAD programs.,False,,False,,True,1,,,,,765,706,925,False,18.4,1791529969.4164321,1791529988.91
sequential,1,35,wH3lJm0Typ0_03,9.009,8.98,270,135,40.01,40.01,0.0,,0.07,0.33,9.15,9.6,9.56,30.41,111,109,"At the end of the game, there were also people who cheered their fans with cheers and critics canceling them.",False,,False,,True,1,,,,,761,706,860,False,18.4,1791529997.908809,1791530037.92
sequential,1,36,oP2WdYlaflE_03,13.934,13.9,418,209,56.81,56.81,0.0,,0.11,0.45,10.74,11.35,11.3,45.47,138,136,The study also limits the number of community members to primary reports such as the Department of Health and People and Human Services.,False,,False,,True,1,,,,,756,706,875,False,18.4,1791530051.8407729,1791530108.654
sequential,1,37,I976oI3s1jQ_05,3.417,3.38,82,41,13.73,13.72,0.01,,0.05,0.14,3.65,3.85,3.84,9.87,51,49,I want to make sure that everyone understands me.,False,,False,,True,1,,,,,871,706,1048,False,18.4,1791530112.049695,1791530125.785
sequential,1,38,k5Gxbifw8s8_03,4.700,4.67,141,71,20.22,20.22,0.0,,0.07,0.18,3.89,4.15,4.13,16.07,50,48,This job opportunity for Deaf people is growing.,False,,False,,True,1,,,,,770,706,918,False,18.4,1791530130.4779286,1791530150.694
sequential,1,39,rSiciLyYOyI_07,2.878,2.84,69,35,12.44,12.43,0.01,,0.06,0.14,3.16,3.37,3.36,9.06,47,45,It was an amazing and never experienced that.,False,,False,,True,1,,,,,854,706,904,False,18.4,1791530153.5498502,1791530165.994
```

## B.5  `confirm_control.csv` — shipping control

```csv
condition,pass,order_index,clip_id,duration_s,capture_seconds,raw_frames,retained_frames_after_trim,post_cut_latency_s,worker_s,queue_wait_s,capture_end_to_perception_done_s,shubert_s,byt5_encoder_s,byt5_decode_s,byt5_stage_s,generate_s,other_s,decode_steps,output_bytes,output_text,oom_or_error,error,declined,decline_reason,deferred,queue_depth_at_cut,perception_frames_at_cut,perception_busy_seconds_cpu,embed_busy_seconds,embedded_frames,min_available_mb_during_utterance,min_available_mb_during_run,avail_mb_at_cut,first_after_warmup,warmup_seconds,cut_ts,done_ts
sequential,1,0,uZwKNtHx9FE_01,14.473,14.43,347,174,14.66,14.65,0.0,2.92,0.3,0.65,10.6,11.6,11.56,0.14,127,125,Most of the outbreaks are that children who do not have a vaccine who live in an ultra-orthodox Jewish community in New York.,False,,False,,False,1,169,19.97,13.54,174,1812,1812,1864,True,18.7,1791528046.6069763,1791528061.262
sequential,1,1,5iRIeHAK9b0_03,4.700,4.68,141,71,7.65,7.65,0.0,3.92,0.08,0.2,3.37,3.67,3.65,0.06,49,47,Children ranging in age from 3 to 29 years old.,False,,False,,False,1,63,7.64,4.96,71,1545,1545,1893,False,18.7,1791528066.1165938,1791528073.768
sequential,1,2,fNT8a6e1gx8_08,3.604,3.57,108,54,7.87,7.87,0.0,3.4,0.08,0.18,4.12,4.4,4.38,0.07,60,58,It's just that we have problems with expressing ourselves.,False,,False,,False,1,45,5.84,3.56,54,1486,1486,1754,False,18.7,1791528077.5325787,1791528085.4
sequential,1,3,sP6B4dlSOoc_05,10.978,10.96,329,165,12.76,12.75,0.01,4.67,0.09,0.38,7.34,7.84,7.81,0.23,97,95,On Wednesday President Trump signed an executive order about trade reform and high kidney care.,False,,False,,False,1,154,19.02,11.79,165,1129,1129,1127,False,18.7,1791528096.5359285,1791528109.295
sequential,1,4,mNF1kg9azQw_03,11.387,11.35,273,137,9.94,9.94,0.0,2.9,0.09,0.32,6.39,6.83,6.8,0.22,92,90,Jennifer shared her experience with a woman who was currently in the photography of Floyd.,False,,False,,False,1,130,18.32,10.12,137,1065,1065,1069,False,18.7,1791528120.81917,1791528130.763
sequential,1,5,iMJ9CjBX7eo_02,4.305,4.27,129,65,9.84,9.84,0.0,3.76,0.09,0.21,5.7,6.03,6.0,0.05,79,77,Thank you for joining us at the Community Forum and we will host it together.,False,,False,,False,1,50,7.91,4.21,65,1254,1065,1646,False,18.7,1791528135.225455,1791528145.062
sequential,1,6,k5Gxbifw8s8_05,4.434,4.41,133,67,6.27,6.27,0.0,3.3,0.07,0.19,2.59,2.86,2.85,0.1,37,35,And more young people wear rabbits.,False,,False,,False,1,57,7.02,4.3,67,924,924,1045,False,18.7,1791528149.6620965,1791528155.93
sequential,1,7,MJijm9kbdHA_01,14.500,14.46,348,174,10.85,10.84,0.0,2.11,0.09,0.41,8.03,8.56,8.53,0.18,102,100,Biden said Bernie had a vote multiple times against a brady bill that required stricter gun control.,False,,False,,False,1,170,18.03,12.05,174,911,843,1034,False,18.7,1791528170.5669641,1791528181.413
sequential,1,8,wPJ_InXF_I8_06,4.939,4.91,148,74,8.96,8.96,0.0,5.11,0.06,0.19,3.51,3.78,3.76,0.08,52,50,the students storytelling comprehension abilities.,False,,False,,False,1,52,10.5,5.77,74,703,703,1157,False,18.7,1791528186.511556,1791528195.475
sequential,1,9,I976oI3s1jQ_01,7.000,6.96,168,84,9.17,9.17,0.0,2.62,0.07,0.2,6.12,6.43,6.4,0.12,86,84,My strategy on the chill was a little bit patient and I thought it would be perfect.,False,,False,,False,1,77,9.93,5.68,84,581,581,803,False,18.7,1791528202.6291413,1791528211.802
sequential,1,10,jsOSzFyx8RA_03,12.713,12.68,381,191,19.14,19.14,0.0,5.11,0.09,0.42,13.26,13.81,13.77,0.21,155,153,"Or the show isn't a lot about the experience, but it's about who you are as a person and how you through surveillance through life, the photography, too.",False,,False,,False,1,181,21.65,14.27,191,441,441,442,False,18.7,1791528224.686615,1791528243.826
sequential,1,11,fNT8a6e1gx8_06,2.770,2.74,83,42,7.38,7.37,0.0,4.32,0.06,0.16,2.75,2.99,2.97,0.06,40,38,Tend to ask your students to graduate.,False,,False,,False,1,22,5.47,2.8,42,969,441,1155,False,18.7,1791528246.7523923,1791528254.13
sequential,1,12,J-0KHhPS_m4_03,5.000,4.97,150,75,7.14,7.14,0.0,3.51,0.08,0.2,3.23,3.52,3.5,0.11,42,40,Here's a quick question: How can I help?,False,,False,,False,1,65,8.01,5.24,75,707,441,954,False,18.7,1791528259.2842672,1791528266.424
sequential,1,13,FND2JfzngVY_06,8.008,7.98,240,120,9.33,9.33,0.0,4.35,0.08,0.23,4.53,4.86,4.84,0.12,55,53,The daylight savings will be back on March 8 at 2 AM.,False,,False,,False,1,110,13.4,8.73,120,616,368,1019,False,18.7,1791528274.6108432,1791528283.94
sequential,1,14,I976oI3s1jQ_02,13.500,13.46,324,162,11.79,11.78,0.0,2.31,0.1,0.38,8.72,9.24,9.2,0.24,119,117,"She said she was still working for two days, but she had a high fever and chills and decided to stay home recovering.",False,,False,,False,1,160,16.33,11.15,162,425,368,427,False,18.7,1791528297.5978477,1791528309.384
sequential,1,15,ZgvLTS2zJdo_02,14.724,14.68,353,177,9.12,9.12,0.0,2.68,0.09,0.39,5.73,6.24,6.21,0.21,58,56,I want to share a little bit of an opportunity with you.,False,,False,,False,1,165,19.09,11.47,177,304,286,309,False,18.7,1791528324.2585223,1791528333.382
sequential,1,16,YR7BS6l_95E_05,11.600,11.57,348,174,13.83,13.83,0.0,5.05,0.09,0.41,8.04,8.58,8.55,0.2,99,97,The helicopter travels to the area and flies to a shelter to help provide supplies with supplies.,False,,False,,False,1,165,19.6,12.74,174,343,275,351,False,18.7,1791528345.153491,1791528358.988
sequential,1,17,NBpYolVo5WQ_06,3.042,3.0,73,37,7.3,7.3,0.0,4.04,0.08,0.18,2.96,3.23,3.22,0.03,43,41,See you tomorrow and stay with the light!,False,,False,,False,1,22,5.22,2.64,37,670,275,902,False,18.7,1791528362.199782,1791528369.503
sequential,1,18,TXyNxVRRPt8_01,6.273,6.24,188,94,10.63,10.63,0.0,4.09,0.07,0.19,6.09,6.38,6.35,0.16,88,86,We brought them to the Deaf Community Forum to bring their concerns to South Carolina.,False,,False,,False,1,82,10.49,6.39,94,241,241,245,False,18.7,1791528375.920386,1791528386.551
sequential,1,19,5iRIeHAK9b0_02,10.400,10.37,312,156,12.63,12.63,0.0,4.48,0.08,0.36,7.53,8.01,7.98,0.14,95,93,On Friday two men in California warned of child abuse in court for confession today (Friday).,False,,False,,False,1,140,16.47,11.4,156,261,241,268,False,18.7,1791528397.0928154,1791528409.722
sequential,1,20,mNF1kg9azQw_05,10.302,10.27,247,124,7.72,7.72,0.0,2.51,0.08,0.22,4.68,5.0,4.98,0.2,67,65,I'm wondering how to send it to another person who sent it to me.,False,,False,,False,1,116,14.69,8.44,124,324,214,526,False,18.7,1791528420.1804788,1791528427.897
sequential,1,21,WMoeSSvCyTU_03,7.908,7.88,237,119,9.67,9.67,0.0,4.89,0.09,0.23,4.26,4.59,4.57,0.19,62,60,The drug hydrogen is still going on and it is controversial.,False,,False,,False,1,103,15.26,8.76,119,231,214,242,False,18.7,1791528435.9738135,1791528445.641
sequential,1,22,sP6B4dlSOoc_04,5.573,5.54,167,84,8.0,8.0,0.0,3.54,0.07,0.18,4.07,4.34,4.32,0.13,46,44,Sometimes simple sidewalks can be dangerous.,False,,False,,False,1,73,8.88,5.71,84,280,214,280,False,18.7,1791528451.375251,1791528459.378
sequential,1,23,z3jAMn3xpoM_02,8.142,8.12,244,122,10.82,10.82,0.0,4.36,0.08,0.22,6.0,6.32,6.3,0.14,83,81,One of them filmed on Facebook LIVE showing officers lying on the floor on blood.,False,,False,,False,1,107,13.77,8.9,122,233,214,302,False,18.7,1791528467.7116659,1791528478.535
sequential,1,24,uZwKNtHx9FE_07,12.471,12.44,299,150,12.95,12.95,0.0,3.04,0.1,0.36,9.22,9.72,9.68,0.19,130,128,"They said the words were added without delay, were immediately introduced into the legislation to make sure that the work began.",False,,False,,False,1,138,17.52,10.63,150,254,214,262,False,18.7,1791528491.1496189,1791528504.097
sequential,1,25,D9HR4q0vbwE_05,2.044,2.0,49,25,5.27,5.27,0.0,3.57,0.06,0.14,1.46,1.67,1.66,0.03,20,18,How are you doing?,False,,False,,False,1,11,3.65,1.89,25,267,214,267,False,18.7,1791528506.3213599,1791528511.591
sequential,1,26,f75G_hZMSHs_04,2.403,2.37,72,36,9.15,9.15,0.0,4.37,0.07,0.18,4.49,4.76,4.74,0.02,59,57,The company will likely start their wedding this weekend.,False,,False,,False,1,17,4.89,2.58,36,233,214,233,False,18.7,1791528514.190541,1791528523.341
sequential,1,27,f75G_hZMSHs_02,7.775,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,28,ZgvLTS2zJdo_01,3.421,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,29,LT0-h4P_xBY_04,7.767,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,30,wPJ_InXF_I8_02,14.148,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,31,oP2WdYlaflE_02,9.100,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,32,LT0-h4P_xBY_06,8.400,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,33,mNF1kg9azQw_07,9.385,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,34,TXyNxVRRPt8_03,4.505,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,35,wH3lJm0Typ0_03,9.009,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,36,oP2WdYlaflE_03,13.934,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,37,I976oI3s1jQ_05,3.417,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,38,k5Gxbifw8s8_03,4.700,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
sequential,1,39,rSiciLyYOyI_07,2.878,,,,,,,,,,,,,,,,,False,,True,281MB free,,0,,,,,,,281,,18.7,,
```

## B.6  `clip_set.csv` — the utterance set

```csv
bucket_s,clip_id,duration_s,bucket_pool_size
2-4,D9HR4q0vbwE_05,2.044,41
2-4,f75G_hZMSHs_04,2.403,41
2-4,fNT8a6e1gx8_06,2.770,41
2-4,rSiciLyYOyI_07,2.878,41
2-4,NBpYolVo5WQ_06,3.042,41
2-4,I976oI3s1jQ_05,3.417,41
2-4,ZgvLTS2zJdo_01,3.421,41
2-4,fNT8a6e1gx8_08,3.604,41
4-6,iMJ9CjBX7eo_02,4.305,41
4-6,k5Gxbifw8s8_05,4.434,41
4-6,TXyNxVRRPt8_03,4.505,41
4-6,5iRIeHAK9b0_03,4.700,41
4-6,k5Gxbifw8s8_03,4.700,41
4-6,wPJ_InXF_I8_06,4.939,41
4-6,J-0KHhPS_m4_03,5.000,41
4-6,sP6B4dlSOoc_04,5.573,41
6-9,TXyNxVRRPt8_01,6.273,30
6-9,I976oI3s1jQ_01,7.000,30
6-9,LT0-h4P_xBY_04,7.767,30
6-9,f75G_hZMSHs_02,7.775,30
6-9,WMoeSSvCyTU_03,7.908,30
6-9,FND2JfzngVY_06,8.008,30
6-9,z3jAMn3xpoM_02,8.142,30
6-9,LT0-h4P_xBY_06,8.400,30
9-12,wH3lJm0Typ0_03,9.009,21
9-12,oP2WdYlaflE_02,9.100,21
9-12,mNF1kg9azQw_07,9.385,21
9-12,mNF1kg9azQw_05,10.302,21
9-12,5iRIeHAK9b0_02,10.400,21
9-12,sP6B4dlSOoc_05,10.978,21
9-12,mNF1kg9azQw_03,11.387,21
9-12,YR7BS6l_95E_05,11.600,21
12-15,uZwKNtHx9FE_07,12.471,16
12-15,jsOSzFyx8RA_03,12.713,16
12-15,I976oI3s1jQ_02,13.500,16
12-15,oP2WdYlaflE_03,13.934,16
12-15,wPJ_InXF_I8_02,14.148,16
12-15,uZwKNtHx9FE_01,14.473,16
12-15,MJijm9kbdHA_01,14.500,16
12-15,ZgvLTS2zJdo_02,14.724,16
```

---

# Appendix C — environment of record, verbatim

```
=== captured 2026-10-08T20:57:01-07:00 ===

--- uname -a ---
Linux jetson 5.15.148-tegra #1 SMP PREEMPT Thu Sep 18 15:08:33 PDT 2025 aarch64 aarch64 aarch64 GNU/Linux

--- L4T / JetPack (/etc/nv_tegra_release) ---
# R36 (release), REVISION: 4.7, GCID: 42132812, BOARD: generic, EABI: aarch64, DATE: Thu Sep 18 22:54:44 UTC 2025
# KERNEL_VARIANT: oot
TARGET_USERSPACE_LIB_DIR=nvidia
TARGET_USERSPACE_LIB_DIR_PATH=usr/lib/aarch64-linux-gnu/nvidia

--- nvpmodel -q ---
NV Power Mode: 15W
0
(/var/lib/nvpmodel/status: pmode:0000)

--- jetson_clocks --show ---
SOC family:tegra234  Machine:NVIDIA Jetson Orin Nano Developer Kit
Error: Run this script(/usr/bin/jetson_clocks) as a root user
UNAVAILABLE: jetson_clocks requires root; this agent shell has no tty for sudo.

--- CPU/GPU clock state from sysfs (jetson_clocks substitute) ---
/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq = 729600
/sys/devices/system/cpu/cpu1/cpufreq/scaling_cur_freq = 1036800
/sys/devices/system/cpu/cpu2/cpufreq/scaling_cur_freq = 1510400
/sys/devices/system/cpu/cpu3/cpufreq/scaling_cur_freq = 1510400
/sys/devices/system/cpu/cpu4/cpufreq/scaling_cur_freq = 729600
/sys/devices/system/cpu/cpu5/cpufreq/scaling_cur_freq = 729600
cpu governor = schedutil
gpu cur/max  = 306000000/624750000

--- tegrastats, 5 s idle @1Hz ---
10-08-2026 20:57:02 RAM 3096/7620MB (lfb 68x4MB) SWAP 27/5858MB (cached 0MB) CPU [9%@1510,9%@1510,2%@1510,19%@1510,4%@729,8%@729] GR3D_FREQ 10% cpu@51.062C soc2@49.843C soc0@48.312C gpu@50.718C tj@51.062C soc1@50C VDD_IN 5382mW/5382mW VDD_CPU_GPU_CV 716mW/716mW VDD_SOC 1555mW/1555mW
10-08-2026 20:57:03 RAM 3097/7620MB (lfb 68x4MB) SWAP 27/5858MB (cached 0MB) CPU [6%@1510,13%@1510,3%@1510,12%@1510,6%@729,11%@729] GR3D_FREQ 10% cpu@51.093C soc2@49.781C soc0@48.312C gpu@50.718C tj@51.093C soc1@49.781C VDD_IN 5382mW/5382mW VDD_CPU_GPU_CV 716mW/716mW VDD_SOC 1555mW/1555mW
10-08-2026 20:57:04 RAM 3097/7620MB (lfb 68x4MB) SWAP 27/5858MB (cached 0MB) CPU [4%@729,18%@729,1%@729,1%@729,4%@729,3%@729] GR3D_FREQ 13% cpu@51.25C soc2@49.781C soc0@48.093C gpu@50.718C tj@51.25C soc1@49.812C VDD_IN 5271mW/5345mW VDD_CPU_GPU_CV 677mW/703mW VDD_SOC 1557mW/1556mW
10-08-2026 20:57:05 RAM 3102/7620MB (lfb 68x4MB) SWAP 27/5858MB (cached 0MB) CPU [11%@1510,23%@1510,10%@1510,24%@1510,5%@729,28%@729] GR3D_FREQ 34% cpu@51.156C soc2@49.781C soc0@48.218C gpu@50.718C tj@51.156C soc1@49.843C VDD_IN 5652mW/5422mW VDD_CPU_GPU_CV 914mW/756mW VDD_SOC 1592mW/1565mW
10-08-2026 20:57:06 RAM 3103/7620MB (lfb 68x4MB) SWAP 27/5858MB (cached 0MB) CPU [6%@729,19%@729,5%@729,7%@729,8%@729,2%@729] GR3D_FREQ 0% cpu@51C soc2@49.75C soc0@48.25C gpu@50.5C tj@51C soc1@49.937C VDD_IN 5351mW/5408mW VDD_CPU_GPU_CV 717mW/748mW VDD_SOC 1555mW/1563mW

--- free -m ---
               total        used        free      shared  buff/cache   available
Mem:            7619        2810        3392           5        1416        4570
Swap:           5857          26        5831

--- nproc ---
6

--- df -h / ---
Filesystem      Size  Used Avail Use% Mounted on
/dev/nvme0n1p1  233G  125G   97G  57% /

--- python / torch ---
python 3.10.12
torch 2.11.0 cuda_available True
cv2 4.11.0 | numpy 1.24.3 | transformers 4.30.2

--- mediapipe ---
mediapipe 0.10.13

--- git ---
6e97617eb426cf8fce73c504730bea59d909fd49
?? shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText/latency_runs/
(PROJECT_CONTEXT.md and docs/ are gitignored; latency_runs/ is untracked)

--- effective pipeline settings (code defaults unless an env var overrides) ---
  env BYT5_MAX_LENGTH = <unset>
  env BYT5_NUM_BEAMS = <unset>
  env BYT5_DEVICE = <unset>
  env BYT5_DTYPE = <unset>
  env BYT5_CKPT = <unset>
  env FRAME_STRIDE = <unset>
  env PERCEPTION_WORKERS = <unset>
  env PERCEPTION_CHUNK = <unset>
  env MAX_LIVE_STREAMS = <unset>
  env MAX_RETAINED_FRAMES = <unset>
  env MIN_AVAILABLE_MB = <unset>
  env RECORD_MODE = <unset>
  env MANUAL_TRIM = <unset>
  env MAX_CLIP_SECONDS = <unset>
  env DINOV2_BATCH_SIZE = <unset>
  env GPU_SERIALIZE = <unset>
  env MEDIAPIPE_NUM_HANDS = <unset>
  env MEDIAPIPE_VIDEO_MODE = <unset>
  env USE_ONNX_PERCEPTION = <unset>
  env PYTORCH_NO_CUDA_MEMORY_CACHING = <unset>
  env TRANSCRIPT = <unset>
Wav2Vec2Config is imported from: fairseq.models.wav2vec.wav2vec2
Full path: /home/sllu/asl-video-to-text/shubert/TTIC-SHuBERT-ASLVideo-to-EnglishText/shubert_venv/lib/python3.10/site-packages/fairseq/models/wav2vec/wav2vec2.py
  effective MAX_LIVE_STREAMS      = 2
  effective MAX_RETAINED_FRAMES   = 450
  effective MIN_AVAILABLE_MB      = 350
  effective MANUAL_RECORD         = True (RECORD_MODE=manual -> True)
  effective MAX_CLIP_SECONDS      = 15.0
  effective STREAM_PERCEPTION     = True
  effective STREAM_DINOV2         = True
  effective PERCEPTION_WORKERS    = 2
  effective PERCEPTION_CHUNK      = 30
  effective FRAME_STRIDE          = 2
  effective MEDIAPIPE num_hands   = 2 presence 0.5 tracking 0.5 video_mode True
  effective DINOv2 dtypes         = torch.float16 torch.float16 batch 32
  ByT5 checkpoint (v5 config)     = /home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models/checkpoint-11625-bf16
  ByT5 defaults                   = beams 4 max_length 768 device cuda dtype bfloat16

--- camera ---
crw-rw----+ 1 root video 81, 0 Oct  8 20:53 /dev/video0
crw-rw----+ 1 root video 81, 1 Oct  8 20:53 /dev/video1
Bus 001 Device 008: ID 046d:084b Logitech, Inc. ConferenceCam Connect Video
Bus 001 Device 007: ID 046d:084c Logitech, Inc. ConferenceCam Connect
Bus 001 Device 005: ID 046d:084e Logitech, Inc. ConferenceCam Connect
uvcvideo: loaded

--- other load on the box (ambient state) ---
uptime:  20:57:25 up 36 min,  1 user,  load average: 1.35, 1.17, 0.83
%CPU %MEM   RSS     ELAPSED COMMAND
10.3  5.3 414068      35:24 claude
 7.6  0.5 46236       36:00 /usr/libexec/gnome-terminal-server
 6.3  2.2 179020      36:11 /usr/lib/xorg/Xorg vt2 -displayfd 3 -auth /run/user/1000/gdm/Xauthority -nolisten tcp -background none -noreset -keeptty -novtswitch -verbose 3
 4.3  4.8 381456      36:09 /usr/bin/gnome-shell
 1.6  1.1 88136       36:22 /usr/libexec/packagekitd
 0.8  0.0     0       36:28 [sugov:0]
 0.7  0.0     0       36:35 [nvmap-bz]
 0.7  0.0     0       36:28 [nvgpu_channel_p]
 0.4  0.0     0       36:35 [irq/199-gk20a_s]
 0.4  0.0  4620       36:35 /lib/systemd/systemd-udevd
 0.4  0.5 41632       36:28 /usr/lib/snapd/snapd

witness (power-fault telemetry) is expected to be running at 1 Hz:
   1184       36:28 /bin/sh -c sleep 20 && /usr/bin/python3 /home/sllu/oc_load_test/crash_witness.py >> /home/sllu/oc_load_test/witness/witness.log 2>&1
   2766       36:08 /usr/bin/python3 /home/sllu/oc_load_test/crash_witness.py
```

---

# Appendix D — per-pass thermal and memory summary, verbatim

```
tegrastats_overlap_pass1.log  samples=170
   RAM used MB      : max 7220  median 6744.0  (MemTotal 7620)
   free MB (7620-)  : min available-by-this-gauge 400
   lfb largest-free : min 1MB  median 23.5MB   <- Jetson contiguous-carveout gauge
   swap used MB     : max 456
   CPU core MHz     : min 729  median 1510.0  max 1510
   Tj C             : max 56.1  median 53.8
   VDD_IN mW        : max 10419  median 6745.0
   NOTE: tegrastats has no throttle flag. A CPU clock below the pass max may be the governor idling between utterances, not throttling.

tegrastats_overlap_pass2.log  samples=163
   RAM used MB      : max 7238  median 6463  (MemTotal 7620)
   free MB (7620-)  : min available-by-this-gauge 382
   lfb largest-free : min 1MB  median 14MB   <- Jetson contiguous-carveout gauge
   swap used MB     : max 824
   CPU core MHz     : min 729  median 1510.0  max 1510
   Tj C             : max 55.8  median 53.8
   VDD_IN mW        : max 10498  median 6785
   NOTE: tegrastats has no throttle flag. A CPU clock below the pass max may be the governor idling between utterances, not throttling.

tegrastats_overlap_pass3.log  samples=205
   RAM used MB      : max 7210  median 6621  (MemTotal 7620)
   free MB (7620-)  : min available-by-this-gauge 410
   lfb largest-free : min 1MB  median 8MB   <- Jetson contiguous-carveout gauge
   swap used MB     : max 608
   CPU core MHz     : min 729  median 1510.0  max 1510
   Tj C             : max 55.7  median 53.8
   VDD_IN mW        : max 10435  median 6745
   NOTE: tegrastats has no throttle flag. A CPU clock below the pass max may be the governor idling between utterances, not throttling.

tegrastats_sequential_pass1.log  samples=462
   RAM used MB      : max 7231  median 6902.5  (MemTotal 7620)
   free MB (7620-)  : min available-by-this-gauge 389
   lfb largest-free : min 1MB  median 4.0MB   <- Jetson contiguous-carveout gauge
   swap used MB     : max 956
   CPU core MHz     : min 729  median 1510.0  max 1510
   Tj C             : max 56.1  median 54.4
   VDD_IN mW        : max 10300  median 6745.0
   NOTE: tegrastats has no throttle flag. A CPU clock below the pass max may be the governor idling between utterances, not throttling.

tegrastats_sequential_pass2.log  samples=272
   RAM used MB      : max 7211  median 6367.5  (MemTotal 7620)
   free MB (7620-)  : min available-by-this-gauge 409
   lfb largest-free : min 1MB  median 10.0MB   <- Jetson contiguous-carveout gauge
   swap used MB     : max 475
   CPU core MHz     : min 729  median 1510.0  max 1510
   Tj C             : max 55.8  median 54.2
   VDD_IN mW        : max 10237  median 6785.0
   NOTE: tegrastats has no throttle flag. A CPU clock below the pass max may be the governor idling between utterances, not throttling.

tegrastats_sequential_pass3.log  samples=419
   RAM used MB      : max 7199  median 6697  (MemTotal 7620)
   free MB (7620-)  : min available-by-this-gauge 421
   lfb largest-free : min 1MB  median 6MB   <- Jetson contiguous-carveout gauge
   swap used MB     : max 675
   CPU core MHz     : min 729  median 1510.0  max 1510
   Tj C             : max 55.6  median 54.1
   VDD_IN mW        : max 10260  median 6745
   NOTE: tegrastats has no throttle flag. A CPU clock below the pass max may be the governor idling between utterances, not throttling.
```

---

# Appendix E — full statistical output, verbatim

Produced by `latency_runs/analysis.py`. Overlaps with the body of this report; included so
that every figure quoted above can be traced to the tool that produced it.

```
ANALYSIS -- measured values only, nothing smoothed or extrapolated.
sequential.csv rows: 120   overlap.csv rows: 120

==============================================================================
CONDITION: sequential
==============================================================================
  utterances attempted      : 120
  measurements (text out)   : 91
  failures / drops          : 0
  declined by backlog guard : 29
      DECLINED oP2WdYlaflE_02 pass2: 316MB free (queue depth 0)
      DECLINED LT0-h4P_xBY_06 pass2: 316MB free (queue depth 0)
      DECLINED uZwKNtHx9FE_01 pass2: 316MB free (queue depth 0)
      DECLINED fNT8a6e1gx8_08 pass2: 316MB free (queue depth 0)
      DECLINED f75G_hZMSHs_04 pass2: 316MB free (queue depth 0)
      DECLINED TXyNxVRRPt8_01 pass2: 316MB free (queue depth 0)
      DECLINED wPJ_InXF_I8_06 pass2: 316MB free (queue depth 0)
      DECLINED z3jAMn3xpoM_02 pass2: 316MB free (queue depth 0)
      DECLINED fNT8a6e1gx8_06 pass2: 316MB free (queue depth 0)
      DECLINED f75G_hZMSHs_02 pass2: 317MB free (queue depth 0)
      DECLINED LT0-h4P_xBY_04 pass2: 317MB free (queue depth 0)
      DECLINED WMoeSSvCyTU_03 pass2: 317MB free (queue depth 0)
      DECLINED I976oI3s1jQ_02 pass2: 317MB free (queue depth 0)
      DECLINED mNF1kg9azQw_03 pass2: 317MB free (queue depth 0)
      DECLINED I976oI3s1jQ_01 pass2: 317MB free (queue depth 0)
      DECLINED iMJ9CjBX7eo_02 pass2: 317MB free (queue depth 0)
      DECLINED k5Gxbifw8s8_05 pass2: 317MB free (queue depth 0)
      DECLINED J-0KHhPS_m4_03 pass2: 317MB free (queue depth 0)
      DECLINED mNF1kg9azQw_07 pass2: 317MB free (queue depth 0)
      DECLINED MJijm9kbdHA_01 pass2: 317MB free (queue depth 0)
      DECLINED jsOSzFyx8RA_03 pass2: 317MB free (queue depth 0)
      DECLINED oP2WdYlaflE_02 pass3: 292MB free (queue depth 0)
      DECLINED MJijm9kbdHA_01 pass3: 292MB free (queue depth 0)
      DECLINED 5iRIeHAK9b0_02 pass3: 292MB free (queue depth 0)
      DECLINED sP6B4dlSOoc_05 pass3: 292MB free (queue depth 0)
      DECLINED J-0KHhPS_m4_03 pass3: 292MB free (queue depth 0)
      DECLINED LT0-h4P_xBY_06 pass3: 292MB free (queue depth 0)
      DECLINED oP2WdYlaflE_03 pass3: 292MB free (queue depth 0)
      DECLINED YR7BS6l_95E_05 pass3: 292MB free (queue depth 0)
  deferred (over stream cap): 0

-- post-cut latency, seconds (p99 at this n is effectively the max) --
  all: n=91  mean 10.23  median 9.33  p90 14.05  p95 14.50  p99 19.33  min 5.11  max 19.40

-- post-cut latency by utterance-duration bucket (s) --
   2-4 s: n=21  mean 7.65  median 7.44  p90 11.75  p95 11.87  p99 11.88  min 5.11  max 11.88
   4-6 s: n=19  mean 8.01  median 7.96  p90 9.23  p95 9.64  p99 9.86  min 6.43  max 9.92
   6-9 s: n=16  mean 10.77  median 10.12  p90 13.71  p95 13.98  p99 14.04  min 9.03  max 14.05
   9-12s: n=17  mean 11.89  median 12.62  p90 13.89  p95 14.07  p99 14.33  min 7.83  max 14.39
  12-15s: n=18  mean 13.56  median 13.04  p90 18.32  p95 19.33  p99 19.39  min 8.79  max 19.40

-- latency vs utterance duration: latency = 0.554 * duration + 5.993   R^2 0.498   Pearson r 0.706
-- latency vs retained frames : latency = 0.0455 * frames + 5.420   R^2 0.607   Pearson r 0.779

-- fraction of measurements at or below a threshold --
  <= 12s :  65/91 = 71.4%
  <= 24s :  91/91 = 100.0%
  <= 30s :  91/91 = 100.0%

-- stage shares of post-cut latency (share computed per utterance) --
  queue wait          mean   0.0%  median   0.0%   (mean  0.00s, median  0.00s, n=91)
  drain               mean  40.6%  median  38.5%   (mean  3.85s, median  3.93s, n=91)
  SHuBERT             mean   0.9%  median   0.8%   (mean  0.09s, median  0.08s, n=91)
  ByT5 encoder        mean   2.6%  median   2.4%   (mean  0.26s, median  0.21s, n=91)
  ByT5 decode         mean  54.5%  median  56.0%   (mean  5.87s, median  5.66s, n=91)
  ByT5 stage (total)  mean  58.2%  median  59.8%   (mean  6.25s, median  6.03s, n=91)
  unattributed        mean   1.2%  median   1.3%   (mean  0.13s, median  0.13s, n=91)

-- 5 worst utterances by post-cut latency --
  jsOSzFyx8RA_03       pass3 dur 12.71s frames 191 bytes 153 steps 155 lat  19.40s | wait 0.0 drain 5.3 shubert 0.1 enc 0.4 decode 13.4 other 0.2 | dominated by decode
  jsOSzFyx8RA_03       pass1 dur 12.71s frames 191 bytes 153 steps 155 lat  19.32s | wait 0.0 drain 5.2 shubert 0.1 enc 0.4 decode 13.3 other 0.2 | dominated by decode
  oP2WdYlaflE_03       pass1 dur 13.93s frames 209 bytes 148 steps 150 lat  17.89s | wait 0.0 drain 5.3 shubert 0.1 enc 0.5 decode 11.7 other 0.2 | dominated by decode
  oP2WdYlaflE_03       pass2 dur 13.93s frames 209 bytes 148 steps 150 lat  16.89s | wait 0.0 drain 4.2 shubert 0.1 enc 0.4 decode 11.8 other 0.2 | dominated by decode
  uZwKNtHx9FE_01       pass1 dur 14.47s frames 174 bytes 125 steps 127 lat  14.60s | wait 0.0 drain 2.6 shubert 0.3 enc 0.7 decode 10.8 other 0.1 | dominated by decode

-- per-pass median latency (drift / throttling check) --
  pass 1: n=40 median 9.96s mean 10.47s
  pass 2: n=19 median 9.03s mean 10.17s
  pass 3: n=32 median 9.30s mean 9.98s

-- cold start (D) --
  pass 1 first utterance after warmup: uZwKNtHx9FE_01 dur 14.47s latency 14.60s (warmup itself 18.4s)
  pass 2 first utterance after warmup: sP6B4dlSOoc_04 dur 5.57s latency 9.13s (warmup itself 18.7s)
  pass 3 first utterance after warmup: mNF1kg9azQw_03 dur 11.39s latency 10.94s (warmup itself 18.3s)
  steady state (excluding those): n=88 median 9.30s

-- output size --
  output bytes: mean 75.5  max 153  at/over the 768-byte cap: 0
  decode steps: mean 77.5  max 155  (ByT5 is byte-level, so steps ~ bytes + specials)

-- perception real-time factor (B), per utterance --
  sustained perception fps DURING capture: mean 11.75  median 12.32  min 4.50  max 14.38  (n=91)
  shortfall vs the 15 fps needed for a zero drain: mean 1.35x  median 1.22x  min 1.04x  max 3.33x
  same figure against the raw 30 fps camera rate: mean 2.70x
  aggregate perception CPU seconds / utterance duration: mean 1.64  median 1.65  min 1.25  max 2.17   [AGGREGATE across PERCEPTION_WORKERS=2, so divide by 2 for per-worker: mean 0.82]

==============================================================================
CONDITION: overlap
==============================================================================
  utterances attempted      : 120
  measurements (text out)   : 35
  failures / drops          : 3
  declined by backlog guard : 82
      FAIL YR7BS6l_95E_05 pass2: AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/
      FAIL ZgvLTS2zJdo_01 pass2: AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/
      FAIL sP6B4dlSOoc_05 pass2: AcceleratorError: CUDA error: out of memory
Search for `cudaErrorMemoryAllocation' in https://docs.nvidia.com/
      DECLINED wPJ_InXF_I8_06 pass1: backlog 608 frames (queue depth 5)
      DECLINED I976oI3s1jQ_01 pass1: backlog 608 frames (queue depth 5)
      DECLINED jsOSzFyx8RA_03 pass1: backlog 608 frames (queue depth 5)
      DECLINED fNT8a6e1gx8_06 pass1: backlog 608 frames (queue depth 5)
      DECLINED J-0KHhPS_m4_03 pass1: backlog 608 frames (queue depth 5)
      DECLINED FND2JfzngVY_06 pass1: backlog 608 frames (queue depth 5)
      DECLINED I976oI3s1jQ_02 pass1: backlog 608 frames (queue depth 5)
      DECLINED ZgvLTS2zJdo_02 pass1: backlog 608 frames (queue depth 5)
      DECLINED YR7BS6l_95E_05 pass1: backlog 608 frames (queue depth 5)
      DECLINED NBpYolVo5WQ_06 pass1: backlog 608 frames (queue depth 5)
      DECLINED TXyNxVRRPt8_01 pass1: backlog 608 frames (queue depth 5)
      DECLINED 5iRIeHAK9b0_02 pass1: backlog 608 frames (queue depth 5)
      DECLINED sP6B4dlSOoc_04 pass1: backlog 484 frames (queue depth 4)
      DECLINED z3jAMn3xpoM_02 pass1: backlog 484 frames (queue depth 4)
      DECLINED uZwKNtHx9FE_07 pass1: backlog 484 frames (queue depth 4)
      DECLINED D9HR4q0vbwE_05 pass1: backlog 484 frames (queue depth 4)
      DECLINED f75G_hZMSHs_04 pass1: backlog 484 frames (queue depth 4)
      DECLINED f75G_hZMSHs_02 pass1: backlog 484 frames (queue depth 4)
      DECLINED ZgvLTS2zJdo_01 pass1: backlog 484 frames (queue depth 4)
      DECLINED LT0-h4P_xBY_04 pass1: backlog 484 frames (queue depth 4)
      DECLINED oP2WdYlaflE_02 pass1: backlog 629 frames (queue depth 4)
      DECLINED LT0-h4P_xBY_06 pass1: backlog 629 frames (queue depth 4)
      DECLINED mNF1kg9azQw_07 pass1: backlog 629 frames (queue depth 4)
      DECLINED TXyNxVRRPt8_03 pass1: backlog 629 frames (queue depth 4)
      DECLINED wH3lJm0Typ0_03 pass1: backlog 629 frames (queue depth 4)
      DECLINED oP2WdYlaflE_03 pass1: backlog 629 frames (queue depth 4)
      DECLINED I976oI3s1jQ_05 pass1: backlog 629 frames (queue depth 4)
      DECLINED k5Gxbifw8s8_03 pass1: backlog 629 frames (queue depth 4)
      DECLINED rSiciLyYOyI_07 pass1: backlog 629 frames (queue depth 4)
      DECLINED uZwKNtHx9FE_07 pass2: backlog 489 frames (queue depth 3)
      DECLINED wPJ_InXF_I8_02 pass2: backlog 489 frames (queue depth 3)
      DECLINED 5iRIeHAK9b0_03 pass2: backlog 489 frames (queue depth 3)
      DECLINED TXyNxVRRPt8_03 pass2: backlog 489 frames (queue depth 3)
      DECLINED k5Gxbifw8s8_03 pass2: backlog 489 frames (queue depth 3)
      DECLINED FND2JfzngVY_06 pass2: backlog 489 frames (queue depth 3)
      DECLINED wH3lJm0Typ0_03 pass2: backlog 489 frames (queue depth 3)
      DECLINED I976oI3s1jQ_05 pass2: backlog 489 frames (queue depth 3)
      DECLINED NBpYolVo5WQ_06 pass2: backlog 489 frames (queue depth 3)
      DECLINED ZgvLTS2zJdo_02 pass2: backlog 489 frames (queue depth 3)
      DECLINED D9HR4q0vbwE_05 pass2: 275MB free (queue depth 2)
      DECLINED oP2WdYlaflE_02 pass2: 273MB free (queue depth 2)
      DECLINED uZwKNtHx9FE_01 pass2: backlog 491 frames (queue depth 3)
      DECLINED wPJ_InXF_I8_06 pass2: 244MB free (queue depth 4)
      DECLINED z3jAMn3xpoM_02 pass2: 244MB free (queue depth 4)
      DECLINED fNT8a6e1gx8_06 pass2: 244MB free (queue depth 4)
      DECLINED f75G_hZMSHs_02 pass2: 243MB free (queue depth 4)
      DECLINED LT0-h4P_xBY_04 pass2: 243MB free (queue depth 4)
      DECLINED WMoeSSvCyTU_03 pass2: 243MB free (queue depth 4)
      DECLINED I976oI3s1jQ_02 pass2: 243MB free (queue depth 4)
      DECLINED I976oI3s1jQ_01 pass2: 242MB free (queue depth 5)
      DECLINED iMJ9CjBX7eo_02 pass2: 275MB free (queue depth 5)
      DECLINED k5Gxbifw8s8_05 pass2: 274MB free (queue depth 5)
      DECLINED J-0KHhPS_m4_03 pass2: 296MB free (queue depth 5)
      DECLINED mNF1kg9azQw_07 pass2: 322MB free (queue depth 5)
      DECLINED MJijm9kbdHA_01 pass2: 322MB free (queue depth 5)
      DECLINED jsOSzFyx8RA_03 pass2: 322MB free (queue depth 5)
      DECLINED uZwKNtHx9FE_01 pass3: backlog 460 frames (queue depth 6)
      DECLINED D9HR4q0vbwE_05 pass3: backlog 460 frames (queue depth 6)
      DECLINED k5Gxbifw8s8_03 pass3: backlog 460 frames (queue depth 6)
      DECLINED iMJ9CjBX7eo_02 pass3: backlog 460 frames (queue depth 6)
      DECLINED WMoeSSvCyTU_03 pass3: backlog 472 frames (queue depth 7)
      DECLINED fNT8a6e1gx8_08 pass3: backlog 472 frames (queue depth 7)
      DECLINED ZgvLTS2zJdo_02 pass3: backlog 472 frames (queue depth 7)
      DECLINED wH3lJm0Typ0_03 pass3: backlog 472 frames (queue depth 7)
      DECLINED jsOSzFyx8RA_03 pass3: backlog 552 frames (queue depth 7)
      DECLINED k5Gxbifw8s8_05 pass3: backlog 552 frames (queue depth 7)
      DECLINED I976oI3s1jQ_02 pass3: backlog 552 frames (queue depth 7)
      DECLINED mNF1kg9azQw_07 pass3: backlog 552 frames (queue depth 7)
      DECLINED wPJ_InXF_I8_02 pass3: backlog 552 frames (queue depth 7)
      DECLINED TXyNxVRRPt8_01 pass3: backlog 552 frames (queue depth 7)
      DECLINED sP6B4dlSOoc_04 pass3: backlog 552 frames (queue depth 7)
      DECLINED fNT8a6e1gx8_06 pass3: backlog 478 frames (queue depth 6)
      DECLINED TXyNxVRRPt8_03 pass3: backlog 478 frames (queue depth 6)
      DECLINED z3jAMn3xpoM_02 pass3: backlog 478 frames (queue depth 6)
      DECLINED f75G_hZMSHs_04 pass3: backlog 478 frames (queue depth 6)
      DECLINED oP2WdYlaflE_02 pass3: backlog 478 frames (queue depth 6)
      DECLINED MJijm9kbdHA_01 pass3: backlog 478 frames (queue depth 6)
      DECLINED 5iRIeHAK9b0_02 pass3: backlog 478 frames (queue depth 6)
      DECLINED sP6B4dlSOoc_05 pass3: backlog 478 frames (queue depth 6)
      DECLINED LT0-h4P_xBY_06 pass3: backlog 512 frames (queue depth 6)
      DECLINED oP2WdYlaflE_03 pass3: backlog 512 frames (queue depth 6)
      DECLINED YR7BS6l_95E_05 pass3: backlog 512 frames (queue depth 6)
  deferred (over stream cap): 17

-- post-cut latency, seconds (p99 at this n is effectively the max) --
  all: n=35  mean 56.83  median 56.78  p90 91.32  p95 94.34  p99 97.18  min 9.52  max 97.61

-- post-cut latency by utterance-duration bucket (s) --
   2-4 s: n=8  mean 57.25  median 57.71  p90 83.73  p95 85.02  p99 86.06  min 9.52  max 86.32
   4-6 s: n=7  mean 49.87  median 56.78  p90 76.35  p95 82.45  p99 87.32  min 10.28  max 88.54
   6-9 s: n=7  mean 66.62  median 79.69  p90 94.95  p95 96.28  p99 97.34  min 19.66  max 97.61
   9-12s: n=8  mean 56.34  median 57.76  p90 81.58  p95 88.96  p99 94.86  min 13.01  max 96.34
  12-15s: n=5  mean 53.00  median 43.03  p90 86.38  p95 89.93  p99 92.77  min 15.99  max 93.48

-- latency vs utterance duration: latency = -0.053 * duration + 57.241   R^2 0.000   Pearson r -0.008
-- latency vs retained frames : latency = 0.0101 * frames + 55.780   R^2 0.000   Pearson r 0.019

-- fraction of measurements at or below a threshold --
  <= 12s :   2/35 = 5.7%
  <= 24s :   6/35 = 17.1%
  <= 30s :   6/35 = 17.1%

-- stage shares of post-cut latency (share computed per utterance) --
  queue wait          mean  60.9%  median  68.2%   (mean 39.48s, median 47.29s, n=35)
  drain               mean  11.2%  median   4.2%   (mean  2.30s, median  1.88s, n=18)
  SHuBERT             mean   0.4%  median   0.1%   (mean  0.12s, median  0.09s, n=35)
  ByT5 encoder        mean   1.0%  median   0.5%   (mean  0.29s, median  0.22s, n=35)
  ByT5 decode         mean  16.2%  median   9.9%   (mean  6.15s, median  6.27s, n=35)
  ByT5 stage (total)  mean  17.9%  median  11.4%   (mean  6.66s, median  6.72s, n=35)
  unattributed        mean  15.4%  median   1.1%   (mean  9.52s, median  0.27s, n=35)

-- latency vs queue depth at cut --
  depth 1: n=3  mean 13.09  median 13.01  p90 15.39  p95 15.69  p99 15.93  min 10.28  max 15.99
  depth 2: n=4  mean 22.52  median 18.76  p90 36.02  p95 39.52  p99 42.33  min 9.52  max 43.03
  depth 3: n=8  mean 44.70  median 42.12  p90 54.39  p95 54.81  p99 55.14  min 34.43  max 55.22
  depth 4: n=6  mean 58.67  median 58.54  p90 68.42  p95 72.07  p99 75.00  min 43.77  max 75.73
  depth 5: n=7  mean 80.59  median 80.23  p90 94.62  p95 95.48  p99 96.17  min 64.48  max 96.34
  depth 6: n=4  mean 81.53  median 81.17  p90 87.87  p95 88.21  p99 88.47  min 75.26  max 88.54
  depth 7: n=3  mean 86.64  median 82.62  p90 94.61  p95 96.11  p99 97.31  min 79.69  max 97.61

-- 5 worst utterances by post-cut latency --
  LT0-h4P_xBY_04       pass3 dur  7.77s frames 117 bytes 96 steps 98 lat  97.61s | wait 64.5 drain -- shubert 0.1 enc 0.2 decode 7.9 other 24.9 | dominated by queue wait  [DEFERRED: perception not started during capture]
  mNF1kg9azQw_05       pass1 dur 10.30s frames 124 bytes 87 steps 89 lat  96.34s | wait 63.6 drain -- shubert 0.1 enc 0.2 decode 6.4 other 26.0 | dominated by queue wait  [DEFERRED: perception not started during capture]
  MJijm9kbdHA_01       pass1 dur 14.50s frames 174 bytes 100 steps 102 lat  93.48s | wait 52.2 drain -- shubert 0.1 enc 0.4 decode 8.2 other 32.5 | dominated by queue wait  [DEFERRED: perception not started during capture]
  WMoeSSvCyTU_03       pass1 dur  7.91s frames 119 bytes 60 steps 62 lat  93.17s | wait 86.7 drain 1.6 shubert 0.1 enc 0.2 decode 4.4 other 0.2 | dominated by queue wait
  J-0KHhPS_m4_03       pass3 dur  5.00s frames  75 bytes 58 steps 60 lat  88.54s | wait 68.5 drain -- shubert 0.1 enc 0.2 decode 4.2 other 15.5 | dominated by queue wait  [DEFERRED: perception not started during capture]

-- per-pass median latency (drift / throttling check) --
  pass 1: n=11 median 60.30s mean 60.68s
  pass 2: n=10 median 48.73s mean 46.25s
  pass 3: n=14 median 71.75s mean 61.37s

-- cold start (D) --
  pass 1 first utterance after warmup: uZwKNtHx9FE_01 dur 14.47s latency 15.99s (warmup itself 18.8s)
  pass 2 first utterance after warmup: sP6B4dlSOoc_04 dur 5.57s latency 10.28s (warmup itself 23.3s)
  pass 3 first utterance after warmup: mNF1kg9azQw_03 dur 11.39s latency 13.01s (warmup itself 23.1s)
  steady state (excluding those): n=32 median 60.70s

-- output size --
  output bytes: mean 74.7  max 148  at/over the 768-byte cap: 0
  decode steps: mean 76.7  max 150  (ByT5 is byte-level, so steps ~ bytes + specials)

-- perception real-time factor (B), per utterance --
  sustained perception fps DURING capture: mean 12.00  median 11.62  min 7.72  max 13.97  (n=18)
  shortfall vs the 15 fps needed for a zero drain: mean 1.27x  median 1.29x  min 1.07x  max 1.94x
  same figure against the raw 30 fps camera rate: mean 2.54x
  aggregate perception CPU seconds / utterance duration: mean 1.67  median 1.63  min 1.34  max 2.09   [AGGREGATE across PERCEPTION_WORKERS=2, so divide by 2 for per-worker: mean 0.84]

==============================================================================
DECODE-LENGTH DEPENDENCE (F)
==============================================================================
  decode_s vs output_bytes : slope 77.81 ms/byte  intercept 0.096s  R^2 0.940  Pearson r 0.969  (n=126)
  decode_s vs decode_steps : slope 77.81 ms/step  intercept -0.060s  R^2 0.940  Pearson r 0.969
  naive decode_s/steps     : mean 76.93 ms/step  median 73.22  min 68.08  max 114.00

==============================================================================
DETERMINISM (STEP 5)
==============================================================================
  sequential: 36 clips measured in more than one pass; 0 differ across passes
  overlap vs sequential: 28 clips in both; 9 differ
    DIFFERS J-0KHhPS_m4_03:
       sequential: Here's a quick question: How can I help?
       overlap   : Here's a question from the next question: How did it stop?
    DIFFERS LT0-h4P_xBY_04:
       sequential: Of course, I want to make sure that the police officers and the police officers are shared with the 
       overlap   : Of course, I want to clearly warn you that I just shared with the people who are self-esteeming.
    DIFFERS TXyNxVRRPt8_01:
       sequential: We brought them to the Deaf Community Forum to bring their concerns to South Carolina.
       overlap   : We bring them to the deaf community forum to bring their concerns to South Carolina.
    DIFFERS ZgvLTS2zJdo_01:
       sequential: The first thing we want to do is to open and ensure that some students have access to sign language 
       overlap   : The first is to open and provide resources for deaf and hard of hearing people.
    DIFFERS f75G_hZMSHs_04:
       sequential: The company will likely start their wedding this weekend.
       overlap   : The tourist wedding is scheduled for this weekend.
    DIFFERS fNT8a6e1gx8_08:
       sequential: It's just that we have problems with expressing ourselves.
       overlap   : It's just about our problems of expressing ourselves.
    DIFFERS k5Gxbifw8s8_05:
       sequential: And more young people wear rabbits.
       overlap   : And more young people use rabbit.
    DIFFERS mNF1kg9azQw_05:
       sequential: I'm wondering how to send it to another person who sent it to me.
       overlap   : I wonder how do you recommend people respond to someone who has been vlogging about it?
    DIFFERS sP6B4dlSOoc_05:
       sequential: On Wednesday President Trump signed an executive order about trade reform and high kidney care.
       overlap   : On Wednesday President Trump signed an executive order about trade reform and life kidney care.
```
