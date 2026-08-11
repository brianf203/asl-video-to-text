"""One GPU consumer at a time, process-wide.

The Jetson has 8GB shared between CPU and GPU, and the live path can have several GPU
consumers alive at once: each recording clip runs its own DINOv2 embedding stage (see
streaming_perception.py) while the worker is running ByT5 on a previous clip. Nothing
bounded that, and it is what lost 5 of 9 clips to CUDA OOM in the first real signing
session -- reproduced unattended afterwards with `live_worker_probe.py --overlap`, which
failed 2-4 of 6 clips at six concurrent streams while the same clips fed sequentially
passed clean four times.

Serialising costs almost nothing here. DINOv2's speedup came from hiding behind CPU-bound
MediaPipe, not from overlapping with ByT5, so holding the GPU one job at a time preserves
the pipelining that matters while making the peak predictable.

An RLock rather than a Lock: re-entering from the same thread should be harmless rather
than a deadlock, since a future caller may reasonably wrap a larger region.

Set GPU_SERIALIZE=0 to disable (for measuring what the lock costs).
"""
import os
import threading
from contextlib import nullcontext

_lock = threading.RLock()

ENABLED = os.environ.get("GPU_SERIALIZE", "1") not in ("0", "false", "False")


def gpu_serial():
    """Context manager held for the duration of one GPU job.

    Keep the region as tight as the actual device work: CPU-side preprocessing (cropping,
    tokenising) should stay outside so it still overlaps with another thread's GPU work.
    """
    return _lock if ENABLED else nullcontext()
