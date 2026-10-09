"""Per-utterance post-cut latency on the live worker. ONE pass of ONE condition.

Built on live_worker_probe.py's own pieces -- feed_clip(), CameraLoop, Sampler, the real
v5 config -- so the memory profile and the true-wall-clock replay pacing are identical to
the probe the project already trusts. live_worker_probe.py itself is NOT modified.

WHAT THIS ADDS over that probe, all of it measurement:
  * post-cut latency to the USER'S definition: cut (replay ends) -> text available.
    The probe's own `seconds` starts when the WORKER DEQUEUES, so in overlap mode it
    silently excludes queue wait. Both are logged here (post_cut_latency_s, worker_s).
  * per-stage split via latency_runs/instrument.py (forward hooks, no pipeline edits).
  * v5's SHIPPING BACKLOG GUARD. The probe never calls _capture_budget() or
    _retain_frame(), so its overlap mode is more permissive than the live app: past
    MAX_RETAINED_FRAMES/MIN_AVAILABLE_MB v5 REFUSES to start a clip. Reproduced here, and
    a refusal is logged as a row with declined=True -- it is a result, not an error.
  * one fsync'd CSV row per utterance, so a power cut costs at most the utterance in
    flight. This board cuts power without warning (11 times since 2026-08-19).

NO trim is applied: head/tail trimming is driven by live motion scores in v5's camera
loop, which a replay does not produce. retained_frames_after_trim therefore equals the
retained frame count, and this run says nothing about what the trim costs or saves.
"""
import argparse
import csv
import os
import random
import sys
import threading
import time

os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2

import auto_segment_v5 as v5
import live_worker_probe as probe
from features import SHuBERTProcessor
from streaming_perception import stride_from_env
from latency_runs import instrument

FIELDS = [
    "condition", "pass", "order_index", "clip_id", "duration_s", "capture_seconds",
    "raw_frames", "retained_frames_after_trim", "post_cut_latency_s", "worker_s",
    "queue_wait_s", "capture_end_to_perception_done_s", "shubert_s", "byt5_encoder_s",
    "byt5_decode_s", "byt5_stage_s", "generate_s", "other_s", "decode_steps",
    "output_bytes", "output_text", "oom_or_error", "error", "declined", "decline_reason",
    "deferred", "queue_depth_at_cut", "perception_frames_at_cut",
    "perception_busy_seconds_cpu", "embed_busy_seconds", "embedded_frames",
    "min_available_mb_during_utterance", "min_available_mb_during_run",
    "avail_mb_at_cut", "first_after_warmup", "warmup_seconds", "cut_ts", "done_ts",
]


class MemTrace(threading.Thread):
    """Timestamped MemAvailable samples, so each utterance can be given its own minimum."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.samples = []
        self._lock = threading.Lock()

    def run(self):
        while not self.stop_event.is_set():
            try:
                m = probe.meminfo()
            except Exception:
                m = None
            if m:
                with self._lock:
                    self.samples.append((time.time(), m["available"], m["used"]))
            self.stop_event.wait(self.interval)

    def min_available(self, t0=None, t1=None):
        with self._lock:
            vals = [a for (t, a, _u) in self.samples
                    if (t0 is None or t >= t0) and (t1 is None or t <= t1)]
        return min(vals) if vals else None

    def peak_used(self):
        with self._lock:
            return max((u for (_t, _a, u) in self.samples), default=None)


def load_clip_set(path):
    with open(path) as fh:
        return [(r["clip_id"], float(r["duration_s"])) for r in csv.DictReader(fh)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--condition", choices=["sequential", "overlap"], required=True)
    ap.add_argument("--pass-index", type=int, required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--clip-set", default="latency_runs/clip_set.csv")
    ap.add_argument("--eval-dir", default="eval_set_openasl")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gap", type=float, default=1.5,
                    help="overlap only: seconds between utterances, as a signer pauses")
    ap.add_argument("--limit", type=int, default=0, help="smoke test: first N clips only")
    args = ap.parse_args()

    overlap = args.condition == "overlap"
    clips = load_clip_set(args.clip_set)
    order = list(range(len(clips)))
    random.Random(args.seed).shuffle(order)
    if args.limit:
        order = order[:args.limit]
    paths = {cid: os.path.join(args.eval_dir, "clips", f"{cid}.mp4") for cid, _ in clips}
    for cid, _ in clips:
        if not os.path.exists(paths[cid]):
            raise SystemExit(f"missing clip: {paths[cid]}")

    mem = MemTrace()
    mem.start()

    cap = cv2.VideoCapture(probe.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cap.isOpened():
        raise SystemExit(f"could not open camera at index {probe.CAMERA_INDEX} -- the "
                         f"probe holds it open deliberately; see REPORT.md")
    camera = probe.CameraLoop(cap)
    camera.start()

    processor = SHuBERTProcessor(v5.config)
    t0 = time.time()
    processor.warmup()
    warmup = time.time() - t0
    print(f"[lat] warmup {warmup:.1f}s", flush=True)
    instrument.install(v5.config)

    stride = stride_from_env()
    embed_config = v5.config if v5.STREAM_DINOV2 else None

    new_file = not os.path.exists(args.out)
    out_fh = open(args.out, "a", newline="")
    writer = csv.DictWriter(out_fh, fieldnames=FIELDS, extrasaction="ignore")
    if new_file:
        writer.writeheader()
    rows_lock = threading.Lock()

    def emit(row):
        """One fsync'd row. A power cut must not take the whole pass with it."""
        with rows_lock:
            writer.writerow(row)
            out_fh.flush()
            os.fsync(out_fh.fileno())

    in_flight = [0]
    flight_lock = threading.Lock()
    import queue as _queue
    work = _queue.Queue()
    first_done = [False]

    def worker():
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            cid, stream, buffered, kept, row = item
            camera.status = f"translating {cid}"
            instrument.reset()
            dequeue_ts = time.time()
            row["queue_wait_s"] = round(dequeue_ts - row["cut_ts"], 2)
            text, failed, err = "", False, ""
            drain = None
            try:
                if stream is not None:
                    t_drain = time.time()
                    frames, landmarks, embeddings = stream.finish(kept)
                    drain = time.time() - t_drain
                    text = processor.process_frames(
                        frames, landmarks=landmarks,
                        mediapipe_seconds=stream.busy_seconds,
                        embeddings=embeddings,
                        embed_seconds=stream.embed_busy_seconds)
                else:
                    # Deferred: perception never started, so all of it lands here. Not a
                    # separable "drain" -- left blank rather than inferred.
                    text = processor.process_frames(buffered)
            except Exception as e:
                failed = True
                err = f"{type(e).__name__}: {e}"
                print(f"[lat] {cid} FAILED: {err}", flush=True)
            done_ts = time.time()
            st = instrument.current()
            if stream is not None:
                row["perception_busy_seconds_cpu"] = round(stream.busy_seconds, 2)
                row["embed_busy_seconds"] = round(stream.embed_busy_seconds, 2)
                row["embedded_frames"] = stream.embedded_frames
                stream.close()
                v5._release_stream_slot()
            v5._release_frames(kept)
            with flight_lock:
                in_flight[0] -= 1
            post_cut = done_ts - row["cut_ts"]
            row.update({
                "post_cut_latency_s": round(post_cut, 2),
                "worker_s": round(done_ts - dequeue_ts, 2),
                "capture_end_to_perception_done_s": None if drain is None else round(drain, 2),
                "shubert_s": _r(st.get("shubert_s")),
                "byt5_encoder_s": _r(st.get("byt5_encoder_s")),
                "byt5_decode_s": _r(st.get("byt5_decode_s")),
                "byt5_stage_s": _r(st.get("byt5_stage_s")),
                "generate_s": _r(st.get("generate_s")),
                "decode_steps": st.get("decode_steps"),
                "output_text": text.replace("\n", " ") if text else "",
                "output_bytes": len(text.encode("utf-8")) if text and not failed else None,
                "oom_or_error": failed,
                "error": err,
                "done_ts": round(done_ts, 3),
                "min_available_mb_during_utterance": mem.min_available(row["cut_ts"], done_ts),
                "min_available_mb_during_run": mem.min_available(),
                "first_after_warmup": not first_done[0],
            })
            # drain + ByT5 + pose + bookkeeping should account for the worker's time; the
            # remainder is reported rather than attributed to a stage it was not measured in.
            known = (row["queue_wait_s"] or 0) + (drain or 0) + (st.get("byt5_stage_s") or 0)
            row["other_s"] = round(max(0.0, post_cut - known), 2)
            first_done[0] = True
            emit(row)
            print(f"[lat] {cid}: post-cut {post_cut:.1f}s "
                  f"(wait {row['queue_wait_s']:.1f} drain {drain if drain is None else round(drain,1)} "
                  f"byt5 {st.get('byt5_stage_s', 0):.1f}) -> {text[:50]}", flush=True)
            work.task_done()

    def _r(v, n=2):
        return None if v is None else round(v, n)

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    for order_index, idx in enumerate(order):
        cid, duration = clips[idx]
        base = {"condition": args.condition, "pass": args.pass_index,
                "order_index": order_index, "clip_id": cid,
                "duration_s": f"{duration:.3f}", "warmup_seconds": round(warmup, 1)}
        # v5 checks the budget BEFORE it starts recording and refuses past it. A refused
        # utterance is what a signer actually experiences, so it is logged, not skipped.
        ok, reason = v5._capture_budget()
        if not ok:
            with flight_lock:
                depth = in_flight[0]
            base.update({"declined": True, "decline_reason": reason,
                         "queue_depth_at_cut": depth, "oom_or_error": False,
                         "avail_mb_at_cut": v5._available_mb()})
            emit(base)
            print(f"[lat] {cid} DECLINED — {reason}", flush=True)
            if overlap:
                time.sleep(args.gap)
            else:
                work.join()
            continue

        camera.status = f"feeding {cid}"
        with flight_lock:
            in_flight[0] += 1
            depth = in_flight[0]
        stream, buffered, kept, raw, capture_seconds = probe.feed_clip(
            paths[cid], stride, embed_config, camera)
        cut_ts = time.time()
        # Account the retained frames against v5's budget, as the camera loop does.
        for _ in range(kept):
            v5._retain_frame()
        base.update({
            "capture_seconds": round(capture_seconds, 2), "raw_frames": raw,
            "retained_frames_after_trim": kept, "queue_depth_at_cut": depth,
            "deferred": stream is None, "declined": False, "decline_reason": "",
            "perception_frames_at_cut": None if stream is None else stream.processed_frames,
            "avail_mb_at_cut": v5._available_mb(), "cut_ts": round(cut_ts, 3),
        })
        base["cut_ts"] = cut_ts
        work.put((cid, stream, buffered, kept, base))
        if overlap:
            time.sleep(args.gap)
        else:
            work.join()

    work.put(None)
    work.join()
    worker_thread.join(timeout=30)

    camera.stop_event.set()
    camera.join(timeout=5)
    fps = camera.fps
    cap.release()
    cv2.destroyAllWindows()
    mem.stop_event.set()
    out_fh.close()
    print(f"[lat] done: camera {fps:.1f}fps  peak_used {mem.peak_used()}MB  "
          f"min_available {mem.min_available()}MB  warmup {warmup:.1f}s", flush=True)


if __name__ == "__main__":
    main()
