"""Per-core and per-THREAD CPU trace around one latency_probe.py pass. Measurement only.

Answers "does the pipeline use the Jetson's six cores?" at a finer grain than tegrastats or
mpstat can: those show per-core load, but not WHICH thread of the pipeline is on it. This runs
latency_probe.main() in-process and samples, every INTERVAL seconds:

  * /proc/stat            -> per-core busy % (what `mpstat -P ALL` prints)
  * /proc/self/task/*/stat -> CPU seconds per thread, labelled with the Python thread name
                              where one exists (perception-0, embed, ...) and the kernel
                              comm otherwise (MediaPipe/XNNPACK/torch native pools)

Usage (same arguments as latency_probe.py, plus CPU_TRACE for the output):
    CPU_TRACE=latency_runs/cpu_seq.csv shubert_venv/bin/python3 latency_runs/cpu_profile.py \
        --condition sequential --pass-index 1 --seed 1001 --out latency_runs/cpu_seq_lat.csv

No pipeline code is touched; latency_probe.py is imported unmodified.
"""
import csv
import os
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

INTERVAL = float(os.environ.get("CPU_INTERVAL", "0.5"))
TICK = os.sysconf("SC_CLK_TCK")


def read_cores():
    out = {}
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu") and line[3].isdigit():
                f = line.split()
                vals = list(map(int, f[1:]))
                idle = vals[3] + vals[4]          # idle + iowait
                out[f[0]] = (sum(vals), idle)
    return out


def read_threads():
    out = {}
    for tid in os.listdir("/proc/self/task"):
        try:
            with open(f"/proc/self/task/{tid}/stat") as fh:
                s = fh.read()
        except OSError:
            continue
        comm = s[s.index("(") + 1:s.rindex(")")]
        f = s[s.rindex(")") + 2:].split()
        # fields after comm: state=0 ... utime=11 stime=12, processor=36
        out[int(tid)] = (comm, int(f[11]) + int(f[12]), int(f[36]))
    return out


class Sampler(threading.Thread):
    def __init__(self, path):
        super().__init__(daemon=True, name="cpu-sampler")
        self.path = path
        self.stop_event = threading.Event()

    def run(self):
        fh = open(self.path, "w", newline="")
        w = csv.writer(fh)
        w.writerow(["ts", "kind", "key", "label", "value", "cpu"])
        prev_c, prev_t = read_cores(), read_threads()
        while not self.stop_event.wait(INTERVAL):
            ts = round(time.time(), 3)
            cores, threads = read_cores(), read_threads()
            for k, (tot, idle) in cores.items():
                dt = tot - prev_c[k][0]
                di = idle - prev_c[k][1]
                w.writerow([ts, "core", k, "", round(100.0 * (dt - di) / dt, 1) if dt else 0, ""])
            names = {t.native_id: t.name for t in threading.enumerate()}
            for tid, (comm, ticks, cpu) in threads.items():
                d = ticks - prev_t.get(tid, (comm, ticks, cpu))[1]
                if d:
                    w.writerow([ts, "thread", tid, names.get(tid, f"[{comm}]"),
                                round(d / TICK, 3), cpu])
            fh.flush()
            prev_c, prev_t = cores, threads
        fh.close()


if __name__ == "__main__":
    from latency_runs import latency_probe
    s = Sampler(os.environ.get("CPU_TRACE", "latency_runs/cpu_trace.csv"))
    s.start()
    try:
        latency_probe.main()
    finally:
        s.stop_event.set()
        s.join(timeout=5)
