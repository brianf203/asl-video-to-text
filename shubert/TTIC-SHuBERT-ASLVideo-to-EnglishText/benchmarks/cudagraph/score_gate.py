#!/usr/bin/env python3
"""Phase 1 scoring: turn cudagraph_decode.py's paired eager/graph jsonl into two run_eval-style
results JSONs (for paired_bootstrap.py) and print BLEU/chrF, differ count and timing.

Usage (SHuBERT dir): shubert_venv/bin/python3 benchmarks/cudagraph/score_gate.py \
    benchmarks/cudagraph/results/phase1_200.jsonl
"""
import json
import os
import statistics as st
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
os.chdir(ROOT)
import run_eval  # noqa: E402

src = sys.argv[1]
recs = [json.loads(l) for l in open(src)]
man = {}
for l in open(os.path.join(run_eval.EVAL_DIR, "manifest.jsonl")):
    it = json.loads(l)
    man[it["id"]] = it
items = [man[r["id"]] for r in recs]
for cond in ("eager", "graph"):
    hyps = [r[f"{cond}_text"].strip() for r in recs]
    res = run_eval.score(items, hyps)
    secs = [r[f"{cond}_s"] for r in recs]
    out = {"condition": cond, "bleu": res["bleu"], "chrf": res["chrf"],
           "mean_seconds_per_clip": sum(secs) / len(secs),
           "outputs": [{"id": it["id"], "reference": it["reference"], "hypothesis": h}
                       for it, h in zip(items, hyps)]}
    path = src.replace(".jsonl", f"_{cond}.json")
    json.dump(out, open(path, "w"), indent=1)
    print(f"{cond:5s} BLEU {res['bleu']:.2f} (raw {res['bleu_raw']:.2f}) chrF {res['chrf']:.2f}  generate total {sum(secs):.1f}s "
          f"median {st.median(secs):.2f}s  -> {path}")
print(f"ids identical {sum(r['graph_same'] for r in recs)}/{len(recs)}; "
      f"ms/step median eager {st.median(r['eager_ms_step'] for r in recs):.2f} "
      f"graph {st.median(r['graph_ms_step'] for r in recs):.2f}")
