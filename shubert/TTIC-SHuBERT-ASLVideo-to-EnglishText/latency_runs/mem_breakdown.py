"""Where does the pipeline's memory go? Stage-by-stage snapshots of ONE live-worker process.

Answers "is quantization worth it for memory" before anything is quantized: how much of the
process is model weights, versus the CUDA context, the allocator's cache, activations,
retained frames, MediaPipe and the Python/torch libraries.

Same pieces as latency_probe.py (v5 config, CameraLoop with the preview window, feed_clip at
wall-clock pace, the real SHuBERTProcessor), sequential only. Measurement only: no pipeline
code is modified.

Gauges per snapshot:
  MemAvailable / MemFree / swap        system-wide, /proc/meminfo
  rss / anon / file, pss               this process, /proc/self/{status,smaps_rollup}
  nvmap_mb                             this process's /dev/nvmap mappings in smaps (if any)
  alloc / reserved                     torch.cuda allocator counters
  lfb_4mb                              free blocks >= 4 MB, /proc/buddyinfo (contiguity)
On Jetson the GPU allocates from the same DRAM. A calibration step (a 64 MB tensor)
records which of these gauges a CUDA allocation actually moves, so the deltas can be read.

Usage (from TTIC-SHuBERT-ASLVideo-to-EnglishText/):
  shubert_venv/bin/python3 latency_runs/mem_breakdown.py --out latency_runs/mem/breakdown.json --clips 10
"""
import argparse
import json
import os
import random
import re
import subprocess
import sys
import threading
import time

os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "0")  # match auto_segment_v5

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

T_START = time.time()


def _kb(text, key):
    m = re.search(rf"^{key}:\s+(\d+)", text, re.M)
    return int(m.group(1)) // 1024 if m else None


def lfb_4mb():
    """Free blocks of >= 4 MB across zones (order >= 10 at 4 KB pages)."""
    n = 0
    with open("/proc/buddyinfo") as fh:
        for line in fh:
            counts = [int(x) for x in line.split()[4:]]
            n += sum(c * (2 ** (o - 10)) for o, c in enumerate(counts) if o >= 10)
    return n


def nvmap_mb():
    total = 0
    cur = None
    try:
        with open("/proc/self/smaps") as fh:
            for line in fh:
                if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                    cur = "nvmap" in line
                elif cur and line.startswith("Rss:"):
                    total += int(line.split()[1])
    except OSError:
        return None
    return total // 1024


def snap(tag, torch_mod=None, extra=None):
    mi = open("/proc/meminfo").read()
    st = open("/proc/self/status").read()
    sm = open("/proc/self/smaps_rollup").read()
    s = {
        "tag": tag, "t": round(time.time() - T_START, 1),
        "mem_available": _kb(mi, "MemAvailable"), "mem_free": _kb(mi, "MemFree"),
        "cached": _kb(mi, "Cached"),
        "swap_used": _kb(mi, "SwapTotal") - _kb(mi, "SwapFree"),
        "rss": _kb(st, "VmRSS"), "anon": _kb(st, "RssAnon"), "file": _kb(st, "RssFile"),
        "pss": _kb(sm, "Pss"), "nvmap_mb": nvmap_mb(), "lfb_4mb": lfb_4mb(),
        "alloc": None, "reserved": None, "max_alloc": None,
    }
    if torch_mod is not None and torch_mod.cuda.is_initialized():
        s["alloc"] = torch_mod.cuda.memory_allocated() >> 20
        s["reserved"] = torch_mod.cuda.memory_reserved() >> 20
        s["max_alloc"] = torch_mod.cuda.max_memory_allocated() >> 20
    if extra:
        s.update(extra)
    print(f"[mem] {tag:28s} avail {s['mem_available']:5d} free {s['mem_free']:5d} "
          f"swap {s['swap_used']:5d} | rss {s['rss']:5d} anon {s['anon']:5d} "
          f"file {s['file']:4d} nvmap {s['nvmap_mb']} | alloc {s['alloc']} rsv {s['reserved']} "
          f"maxalloc {s['max_alloc']} | lfb {s['lfb_4mb']}", flush=True)
    return s


class PeakSampler(threading.Thread):
    """Peak RSS / anon and min MemAvailable between mark() calls, at 0.2 s."""

    def __init__(self):
        super().__init__(daemon=True)
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.mark()

    def mark(self):
        with self.lock:
            self.peak_rss = self.peak_anon = 0
            self.min_avail = 10 ** 9
            self.min_lfb = 10 ** 9

    def read(self):
        with self.lock:
            return {"peak_rss": self.peak_rss, "peak_anon": self.peak_anon,
                    "min_avail": self.min_avail, "min_lfb": self.min_lfb}

    def run(self):
        while not self.stop_event.is_set():
            try:
                st = open("/proc/self/status").read()
                mi = open("/proc/meminfo").read()
                lf = lfb_4mb()
                with self.lock:
                    self.peak_rss = max(self.peak_rss, _kb(st, "VmRSS"))
                    self.peak_anon = max(self.peak_anon, _kb(st, "RssAnon"))
                    self.min_avail = min(self.min_avail, _kb(mi, "MemAvailable"))
                    self.min_lfb = min(self.min_lfb, lf)
            except Exception:
                pass
            self.stop_event.wait(0.2)


def param_mb(module):
    seen, n = set(), 0
    for p in module.parameters():
        if p.data_ptr() in seen:
            continue
        seen.add(p.data_ptr())
        n += p.numel() * p.element_size()
    for b in module.buffers():
        n += b.numel() * b.element_size()
    return round(n / 2 ** 20, 1)


def smaps_groups(top=25):
    """This process's RSS grouped by mapping: shared libraries by file, [heap], [stack],
    device mappings by node, and anonymous mappings bucketed by size (glibc malloc arenas
    are 64 MB-aligned anon regions; CUDA/NvMap allocations are anon on this board)."""
    groups = {}
    cur = None
    with open("/proc/self/smaps") as fh:
        for line in fh:
            if re.match(r"^[0-9a-f]+-[0-9a-f]+ ", line):
                parts = line.split()
                lo, hi = (int(x, 16) for x in parts[0].split("-"))
                name = parts[5] if len(parts) > 5 else ""
                if not name:
                    size_mb = (hi - lo) / 2 ** 20
                    bucket = ("anon <1MB" if size_mb < 1 else "anon 1-32MB" if size_mb < 32
                              else "anon 32-64MB" if size_mb <= 64 else "anon >64MB")
                    cur = bucket
                elif name.startswith("/") and ".so" in name:
                    base = os.path.basename(name)
                    cur = "lib:" + base.split(".so")[0]
                else:
                    cur = name if name.startswith("[") or name.startswith("/dev") else "file:" + os.path.basename(name)
            elif line.startswith("Rss:") and cur is not None:
                groups[cur] = groups.get(cur, 0) + int(line.split()[1])
    rows = sorted(((k, v // 1024) for k, v in groups.items()), key=lambda kv: -kv[1])
    return [{"mapping": k, "rss_mb": v} for k, v in rows[:top] if v > 0]


def other_processes():
    """Top RSS consumers outside this process (RSS overcounts shared pages)."""
    out = subprocess.run(["ps", "-eo", "pid,rss,comm", "--sort=-rss"],
                         capture_output=True, text=True).stdout.splitlines()[1:16]
    rows = []
    for line in out:
        pid, rss, comm = line.split(None, 2)
        if int(pid) != os.getpid():
            rows.append({"pid": int(pid), "rss_mb": int(rss) // 1024, "comm": comm})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--clips", type=int, default=10)
    ap.add_argument("--seed", type=int, default=1001)
    ap.add_argument("--clip-set", default="latency_runs/clip_set.csv")
    ap.add_argument("--eval-dir", default="eval_set_openasl")
    args = ap.parse_args()
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)

    result = {"env": {"PYTORCH_NO_CUDA_MEMORY_CACHING": os.environ["PYTORCH_NO_CUDA_MEMORY_CACHING"]},
              "others_at_start": other_processes(), "snapshots": [], "clips": []}
    S = result["snapshots"]
    S.append(snap("start (python only)"))

    import torch
    S.append(snap("import torch", torch))
    import cv2
    import auto_segment_v5 as v5
    import live_worker_probe as probe
    from features import SHuBERTProcessor
    from streaming_perception import stride_from_env
    import inference
    import dinov2_features
    S.append(snap("import pipeline modules", torch))

    torch.zeros(1, device="cuda")
    torch.cuda.synchronize()
    S.append(snap("CUDA context", torch))

    # Calibration: which gauges does a 64 MB CUDA allocation move?
    before = snap("calib before", torch)
    t = torch.empty(64 * 2 ** 20, dtype=torch.uint8, device="cuda")
    t.fill_(1)
    torch.cuda.synchronize()
    after = snap("calib +64MB cuda tensor", torch)
    del t
    torch.cuda.empty_cache()
    result["calibration_64mb"] = {k: (after[k] - before[k]) for k in
                                  ("mem_available", "mem_free", "rss", "anon", "file", "pss", "nvmap_mb")
                                  if after[k] is not None and before[k] is not None}
    print(f"[mem] calibration deltas for +64MB: {result['calibration_64mb']}", flush=True)

    cap = cv2.VideoCapture(probe.CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    if not cap.isOpened():
        raise SystemExit("could not open camera")
    camera = probe.CameraLoop(cap)
    camera.start()
    time.sleep(3)
    S.append(snap("camera + preview window", torch))

    processor = SHuBERTProcessor(v5.config)
    cfg = v5.config
    from dinov2_features import HANDS_DTYPE, FACE_DTYPE
    dinov2_features.preload_embedder(cfg['dino_hands_model_path'], dtype=HANDS_DTYPE)
    S.append(snap("DINOv2 hands loaded", torch))
    dinov2_features.preload_embedder(cfg['dino_face_model_path'], dtype=FACE_DTYPE)
    S.append(snap("DINOv2 face loaded", torch))
    inference.preload_model(cfg['slt_model_checkpoint'], cfg['slt_tokenizer_checkpoint'], cfg['temp_dir'])
    import gc
    gc.collect()
    S.append(snap("ByT5+SHuBERT loaded", torch))
    processor.warmup()  # no-op for the loads above; kept so the path matches latency_probe

    # Weight bytes by component.
    weights = {}
    for key, emb in dinov2_features._embedder_cache.items():
        weights[f"dinov2:{os.path.basename(str(key[0]))}:{key[3]}"] = param_mb(emb.model)
    for key, (model, _tok, _dev) in inference._model_cache.items():
        weights["byt5_total"] = param_mb(model)
        enc = model.get_encoder()
        sh = getattr(getattr(enc, "adapter", None), "signhubert", None)
        if sh is not None:
            weights["shubert (encoder.adapter.signhubert)"] = param_mb(sh)
        weights["byt5_encoder_incl_adapter"] = param_mb(enc)
        weights["byt5_decoder"] = param_mb(model.get_decoder())
        dtypes = {}
        for p in model.parameters():
            dtypes[str(p.dtype)] = dtypes.get(str(p.dtype), 0) + p.numel()
        weights["byt5_param_dtypes"] = dtypes
    result["weights_mb"] = weights
    print(f"[mem] weights: {json.dumps(weights)}", flush=True)

    stride = stride_from_env()
    embed_config = v5.config if v5.STREAM_DINOV2 else None
    import csv
    with open(args.clip_set) as fh:
        clips = [(r["clip_id"], float(r["duration_s"])) for r in csv.DictReader(fh)]
    order = list(range(len(clips)))
    random.Random(args.seed).shuffle(order)
    order = order[:args.clips]

    sampler = PeakSampler()
    sampler.start()
    S.append(snap("ready (before clip 1)", torch))
    result["smaps_ready"] = smaps_groups()

    for i, idx in enumerate(order):
        cid, dur = clips[idx]
        path = os.path.join(args.eval_dir, "clips", f"{cid}.mp4")
        row = {"i": i, "clip_id": cid, "duration_s": dur}
        torch.cuda.reset_peak_memory_stats()
        sampler.mark()
        stream, buffered, kept, raw, cap_s = probe.feed_clip(path, stride, embed_config, camera)
        row["kept_frames"] = kept
        row["frames_mb_est"] = round(kept * 640 * 480 * 3 / 2 ** 20, 1)
        row["capture"] = {**sampler.read(), "torch_peak_alloc": torch.cuda.max_memory_allocated() >> 20}
        row["at_cut"] = snap(f"clip {i} at cut ({kept} fr)", torch)
        if i == 0:
            result["smaps_at_first_cut"] = smaps_groups()

        torch.cuda.reset_peak_memory_stats()
        sampler.mark()
        if stream is not None:
            frames, landmarks, embeddings = stream.finish(kept)
        row["drain"] = {**sampler.read(), "torch_peak_alloc": torch.cuda.max_memory_allocated() >> 20}

        torch.cuda.reset_peak_memory_stats()
        sampler.mark()
        if stream is not None:
            text = processor.process_frames(frames, landmarks=landmarks,
                                            mediapipe_seconds=stream.busy_seconds,
                                            embeddings=embeddings,
                                            embed_seconds=stream.embed_busy_seconds)
            stream.close()
            v5._release_stream_slot()
        else:
            text = processor.process_frames(buffered)
        row["translate"] = {**sampler.read(), "torch_peak_alloc": torch.cuda.max_memory_allocated() >> 20}
        row["text"] = text
        del stream, buffered
        frames = landmarks = embeddings = None
        gc.collect()
        row["after"] = snap(f"clip {i} done", torch)
        result["clips"].append(row)
        print(f"[mem] clip {i} {cid}: capture peak_rss {row['capture']['peak_rss']} "
              f"torch_peak {row['capture']['torch_peak_alloc']} | drain torch_peak "
              f"{row['drain']['torch_peak_alloc']} | translate torch_peak "
              f"{row['translate']['torch_peak_alloc']} peak_rss {row['translate']['peak_rss']} "
              f"min_avail {row['translate']['min_avail']} -> {text[:40]}", flush=True)
        with open(args.out, "w") as fh:
            json.dump(result, fh, indent=1)

    sampler.stop_event.set()
    camera.stop_event.set()
    camera.join(timeout=5)
    cap.release()
    cv2.destroyAllWindows()
    S.append(snap("end", torch))
    result["smaps_end"] = smaps_groups()
    # Leaked, or freed-but-retained by glibc? malloc_trim(0) returns free arena/heap pages
    # to the OS; whatever RSS it releases was never live data.
    import ctypes
    released = ctypes.CDLL("libc.so.6").malloc_trim(0)
    time.sleep(1)
    S.append(snap(f"after malloc_trim(0) (ret {released})", torch))
    result["smaps_after_trim"] = smaps_groups()
    result["env"]["MALLOC_ARENA_MAX"] = os.environ.get("MALLOC_ARENA_MAX")
    result["others_at_end"] = other_processes()
    with open(args.out, "w") as fh:
        json.dump(result, fh, indent=1)
    print(f"[mem] wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
