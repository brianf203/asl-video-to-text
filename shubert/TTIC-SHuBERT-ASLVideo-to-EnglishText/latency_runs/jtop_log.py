"""Log jtop (jetson-stats) samples to CSV until killed. Run with the system python3, not the venv."""
import csv, signal, sys
from jtop import jtop

out, interval = sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 0.5
signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
with jtop(interval=interval) as j, open(out, "w", newline="") as f:
    w = None
    while j.ok():
        row = dict(j.stats)
        row["lfb"] = j.memory["RAM"]["lfb"]
        row["oc3_events"] = int(open("/sys/class/hwmon/hwmon3/oc3_event_cnt").read())
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row), extrasaction="ignore")
            w.writeheader()
        w.writerow(row)
        f.flush()
