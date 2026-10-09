"""Summarise the tegrastats logs taken during each pass (C).

tegrastats exposes no throttle FLAG, so this reports the clock and temperature record and
says plainly what it does and does not prove: a clock below the pass maximum may be the
governor idling between utterances rather than thermal throttling. The max Tj and the
clock minimum under load are the two numbers that matter.
"""
import glob
import os
import re
import statistics as st

RAM = re.compile(r"RAM (\d+)/(\d+)MB \(lfb (\d+)x(\d+)MB\)")
SWAP = re.compile(r"SWAP (\d+)/(\d+)MB")
CPU = re.compile(r"CPU \[([^\]]+)\]")
GR3D = re.compile(r"GR3D_FREQ (\d+)%")
TJ = re.compile(r"tj@([\d.]+)C")
VDD = re.compile(r"VDD_IN (\d+)mW")


def main():
    for path in sorted(glob.glob("latency_runs/tegrastats_*.log")):
        ram, lfb, clocks, tj, vdd, swap = [], [], [], [], [], []
        n = 0
        for line in open(path, errors="replace"):
            m = RAM.search(line)
            if not m:
                continue
            n += 1
            ram.append(int(m.group(1)))
            lfb.append(int(m.group(3)) * int(m.group(4)))
            s = SWAP.search(line)
            if s:
                swap.append(int(s.group(1)))
            c = CPU.search(line)
            if c:
                for core in c.group(1).split(","):
                    if "@" in core:
                        clocks.append(int(core.split("@")[1]))
            t = TJ.search(line)
            if t:
                tj.append(float(t.group(1)))
            v = VDD.search(line)
            if v:
                vdd.append(int(v.group(1)))
        if not n:
            print(f"{os.path.basename(path)}: no samples")
            continue
        print(f"{os.path.basename(path)}  samples={n}")
        print(f"   RAM used MB      : max {max(ram)}  median {st.median(ram)}  "
              f"(MemTotal 7620)")
        print(f"   free MB (7620-)  : min available-by-this-gauge {7620 - max(ram)}")
        print(f"   lfb largest-free : min {min(lfb)}MB  median {st.median(lfb)}MB"
              f"   <- Jetson contiguous-carveout gauge")
        if swap:
            print(f"   swap used MB     : max {max(swap)}")
        if clocks:
            print(f"   CPU core MHz     : min {min(clocks)}  median "
                  f"{st.median(clocks)}  max {max(clocks)}")
        if tj:
            print(f"   Tj C             : max {max(tj):.1f}  median {st.median(tj):.1f}")
        if vdd:
            print(f"   VDD_IN mW        : max {max(vdd)}  median {st.median(vdd)}")
        print("   NOTE: tegrastats has no throttle flag. A CPU clock below the pass max "
              "may be the governor idling between utterances, not throttling.")
        print()


if __name__ == "__main__":
    main()
