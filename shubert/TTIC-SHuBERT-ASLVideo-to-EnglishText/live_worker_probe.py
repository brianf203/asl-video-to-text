"""Measure the live worker unattended, by replaying recorded clips into it.

`run_eval.py` scores translation quality in a clean process. That is the wrong shape for
answering "is this config safe on the live path?", because the live path also holds camera
buffers and a preview window, and the Jetson's 8GB is shared between CPU and GPU -- a
config that fits in the eval can still push the live worker into swap or CUDA OOM.

This reproduces the live memory profile exactly (camera open at 640x480 MJPG, imshow
window up, persistent in-process worker thread, StreamingPerception per clip) but feeds
pre-recorded eval clips into the queue instead of needing someone to stand in front of the
camera and sign. So a config can be A/B'd repeatably and unattended.

Frames are fed at TRUE wall-clock camera rate. That is the whole point: streaming
perception is credited with the overlap a real camera can supply and no more, so feeding
as fast as the disk allows would invent overlap that could never happen live.

    python3 live_worker_probe.py --beams 4 --tag beam4
    python3 live_worker_probe.py --overlap --beams 4 --tag beam4_overlap

--overlap is the mode that matters for memory. By default the probe waits for each
translation before feeding the next clip, so exactly one StreamingPerception is ever alive.
A real signer does not wait: they keep signing while the worker is still busy, so several
streams coexist, each holding a MediaPipe detector, its retained frames and its own DINOv2
GPU work. That is what OOM'd the first live session (5 of 9 clips lost) after four
sequential probes had all passed clean -- the sequential probes were measuring a case live
signing never produces.

Compare runs only from matched start states -- a run leaves the box several hundred MB
dirtier than it found it, so run 1 vs run 2 is not a controlled comparison. Re-run the
control after the variant and compare that pair.
"""
import argparse
import json
import os
import queue
import threading
import time

os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import cv2

import auto_segment_v5 as v5
from features import SHuBERTProcessor
from streaming_perception import StreamingPerception, stride_from_env

CAMERA_INDEX = v5.CAMERA_INDEX
DEFAULT_CLIPS = ["004", "005", "006", "003"]


def meminfo():
    """MemTotal-MemAvailable is the metric that binds here.

    The Jetson has unified memory, so GPU allocations already show up in /proc/meminfo --
    and torch.cuda.max_memory_allocated() reads 0 anyway because
    PYTORCH_NO_CUDA_MEMORY_CACHING=1 bypasses the allocator those stats come from.
    """
    vals = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, v = line.partition(":")
            vals[k] = int(v.split()[0]) // 1024  # MB
    return {
        "used": vals["MemTotal"] - vals["MemAvailable"],
        "available": vals["MemAvailable"],
        "swap_used": vals["SwapTotal"] - vals["SwapFree"],
    }


class Sampler(threading.Thread):
    """Poll /proc/meminfo in the background for peak/min figures."""

    def __init__(self, interval=0.5):
        super().__init__(daemon=True)
        self.interval = interval
        self.stop_event = threading.Event()
        self.peak_used = 0
        self.peak_swap = 0
        self.min_available = 10 ** 9

    def run(self):
        while not self.stop_event.is_set():
            m = meminfo()
            self.peak_used = max(self.peak_used, m["used"])
            self.peak_swap = max(self.peak_swap, m["swap_used"])
            self.min_available = min(self.min_available, m["available"])
            self.stop_event.wait(self.interval)


class CameraLoop(threading.Thread):
    """Hold the camera open and pump the preview, exactly as v5's main loop does.

    This contributes nothing to the translation; it exists so the measurement includes the
    camera's buffers, the colour conversions and the imshow window, which is the entire
    reason for not just running the eval harness.
    """

    def __init__(self, cap):
        super().__init__(daemon=True)
        self.cap = cap
        self.stop_event = threading.Event()
        self.frames = 0
        self.started = None
        self.status = "probe running"

    def run(self):
        self.started = time.time()
        while not self.stop_event.is_set():
            ok, frame = self.cap.read()
            if not ok:
                break
            self.frames += 1
            cv2.putText(frame, self.status[:70], (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.imshow("live worker probe", frame)
            cv2.waitKey(1)

    @property
    def fps(self):
        if not self.started or not self.frames:
            return 0.0
        return self.frames / (time.time() - self.started)


def feed_clip(path, stride, embed_config, camera):
    """Replay one clip at its own frame rate, mirroring v5's RECORDING branch.

    Takes a stream slot exactly as v5 does, so the MAX_LIVE_STREAMS cap is exercised here
    too -- otherwise the probe would bypass the very thing it is meant to validate. Over
    the cap it buffers stride-2 RGB frames instead, which is v5's deferred path.

    Returns (stream, frames, kept, raw, capture_seconds); exactly one of stream/frames is
    not None.
    """
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    interval = 1.0 / fps

    stream, frames = None, None
    if v5._take_stream_slot():
        stream = StreamingPerception(
            v5.config['mediapipe_face_model_path'],
            v5.config['mediapipe_hands_model_path'],
            embed_config=embed_config,
        )
    else:
        frames = []

    raw = 0
    kept = 0
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Pace to the source frame rate so the worker gets exactly the overlap a live
        # camera would have given it -- no more.
        due = t0 + raw * interval
        delay = due - time.time()
        if delay > 0:
            time.sleep(delay)
        if raw % stride == 0:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if stream is not None:
                stream.add_frame(rgb)
            else:
                frames.append(rgb)
            kept += 1
        raw += 1
        camera.status = f"feeding {os.path.basename(path)} {raw} frames"
    cap.release()
    return stream, frames, kept, raw, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--beams", type=int, default=None,
                    help="BYT5_NUM_BEAMS for this run (default: leave env/default alone)")
    ap.add_argument("--clips", default=",".join(DEFAULT_CLIPS),
                    help="comma-separated eval_set clip ids, replayed in order")
    ap.add_argument("--tag", default="", help="label for the output json")
    ap.add_argument("--eval-dir", default="eval_set")
    ap.add_argument("--overlap", action="store_true",
                    help="keep feeding clips without waiting for translations, as a real "
                         "signer does — this is what puts several streams on the GPU at once")
    ap.add_argument("--gap", type=float, default=1.5,
                    help="seconds between clips in --overlap mode (default 1.5, matching "
                         "STILL_DURATION_SECONDS)")
    args = ap.parse_args()

    if args.beams is not None:
        os.environ["BYT5_NUM_BEAMS"] = str(args.beams)

    clip_ids = [c.strip() for c in args.clips.split(",") if c.strip()]
    paths = [os.path.join(args.eval_dir, "clips", f"{cid}.mp4") for cid in clip_ids]
    for p in paths:
        if not os.path.exists(p):
            raise SystemExit(f"missing clip: {p}")

    start_mem = meminfo()
    sampler = Sampler()
    sampler.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cap.isOpened():
        raise SystemExit(f"could not open camera at index {CAMERA_INDEX}")
    camera = CameraLoop(cap)
    camera.start()

    processor = SHuBERTProcessor(v5.config)
    t0 = time.time()
    processor.warmup()
    warmup = time.time() - t0
    print(f"[probe] models loaded in {warmup:.1f}s")

    stride = stride_from_env()
    embed_config = v5.config if v5.STREAM_DINOV2 else None
    results = []
    results_lock = threading.Lock()
    work = queue.Queue()
    # Clips in flight, NOT StreamingPerception count -- past MAX_LIVE_STREAMS a clip is in
    # flight without a stream. Reported separately from the deferred count below so the two
    # are not confused: 4 in flight with a cap of 2 means the cap did its job.
    live_streams = [0]
    max_live = [0]

    def worker():
        """Same shape as v5's translation_worker: drain the backlog, then run the rest."""
        while True:
            item = work.get()
            if item is None:
                work.task_done()
                return
            cid, stream, buffered, kept, record = item
            camera.status = f"translating {cid}"
            t_clip = time.time()
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
                    record["drain_seconds"] = round(drain, 1)
                    record["mediapipe_seconds"] = round(stream.busy_seconds, 1)
                    record["embed_seconds"] = round(stream.embed_busy_seconds, 1)
                else:
                    # Deferred clip: perception never started, so it all happens here.
                    drain = 0.0
                    text = processor.process_frames(buffered)
                    record["deferred"] = True
                record["seconds"] = round(time.time() - t_clip, 1)
                record["text"] = text
                record["failed"] = False
                print(f"[probe] {cid}: {record['seconds']}s (drain {drain:.1f}s"
                      f"{', deferred' if stream is None else ''}) -> {text}")
            except Exception as e:
                record["seconds"] = round(time.time() - t_clip, 1)
                record["failed"] = True
                record["text"] = f"{type(e).__name__}: {e}"
                print(f"[probe] {cid} FAILED: {type(e).__name__}: {e}")
            finally:
                if stream is not None:
                    stream.close()
                    v5._release_stream_slot()
                live_streams[0] -= 1
                record["mem_used_after"] = meminfo()["used"]
                with results_lock:
                    results.append(record)
                work.task_done()

    worker_thread = threading.Thread(target=worker, daemon=True)
    worker_thread.start()

    for cid, path in zip(clip_ids, paths):
        camera.status = f"feeding {cid}"
        # The stream goes live the moment the clip starts "recording", exactly as v5 does
        # it -- so in overlap mode its detector and DINOv2 thread coexist with whatever the
        # worker is still chewing on.
        live_streams[0] += 1
        max_live[0] = max(max_live[0], live_streams[0])
        stream, buffered, kept, raw, capture_seconds = feed_clip(
            path, stride, embed_config, camera)
        record = {"id": cid, "kept_frames": kept, "raw_frames": raw,
                  "capture_seconds": round(capture_seconds, 1),
                  "live_streams_at_queue": live_streams[0],
                  "deferred": stream is None}
        work.put((cid, stream, buffered, kept, record))
        if args.overlap:
            # A signer pauses between sentences but does not wait for the translation.
            time.sleep(args.gap)
        else:
            work.join()

    work.put(None)
    work.join()
    worker_thread.join(timeout=10)
    results.sort(key=lambda r: clip_ids.index(r["id"]))

    camera.stop_event.set()
    camera.join(timeout=5)
    fps = camera.fps
    cap.release()
    cv2.destroyAllWindows()
    sampler.stop_event.set()

    ok = [r for r in results if not r["failed"]]
    out = {
        "tag": args.tag,
        "timestamp": time.strftime("%Y%m%d-%H%M%S"),
        "byt5_num_beams": os.environ.get("BYT5_NUM_BEAMS", "4"),
        "byt5_device": os.environ.get("BYT5_DEVICE", "cuda"),
        "frame_stride": stride,
        "stream_perception": v5.STREAM_PERCEPTION,
        "stream_dinov2": v5.STREAM_DINOV2,
        "overlap": args.overlap,
        "gap_seconds": args.gap if args.overlap else None,
        "max_clips_in_flight": max_live[0],
        "max_live_streams": v5.MAX_LIVE_STREAMS,
        "clips_deferred": sum(1 for r in results if r.get("deferred")),
        "warmup_seconds": round(warmup, 1),
        "mem_used_at_start": start_mem["used"],
        "peak_mem_used": sampler.peak_used,
        "peak_swap_used": sampler.peak_swap,
        "min_available": sampler.min_available,
        "camera_fps": round(fps, 1),
        "clips_failed": sum(1 for r in results if r["failed"]),
        "mean_seconds_per_clip": round(sum(r["seconds"] for r in ok) / len(ok), 1) if ok else None,
        "results": results,
    }
    name = f"probe_{args.tag or 'run'}_{out['timestamp']}.json"
    with open(name, "w") as f:
        json.dump(out, f, indent=2)

    print("\n" + "=" * 68)
    print(f"PROBE {args.tag}  beams={out['byt5_num_beams']}  overlap={args.overlap}  "
          f"max clips in flight={max_live[0]} (stream cap {v5.MAX_LIVE_STREAMS}, "
          f"{out['clips_deferred']} deferred)")
    print("=" * 68)
    print(f"  warmup            : {out['warmup_seconds']}s")
    print(f"  mean s/clip       : {out['mean_seconds_per_clip']}")
    print(f"  mem used at start : {out['mem_used_at_start']}MB")
    print(f"  peak mem used     : {out['peak_mem_used']}MB")
    print(f"  peak swap used    : {out['peak_swap_used']}MB")
    print(f"  min available     : {out['min_available']}MB")
    print(f"  camera fps        : {out['camera_fps']}")
    print(f"  clips failed      : {out['clips_failed']}")
    for r in results:
        print(f"    [{r['id']}] {r['seconds']}s  {r['text'][:60]}")
    print(f"\nwrote {name}")


if __name__ == "__main__":
    main()
