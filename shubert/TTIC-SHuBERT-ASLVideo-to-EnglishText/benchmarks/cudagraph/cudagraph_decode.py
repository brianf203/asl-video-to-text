#!/usr/bin/env python3
"""Phase 0: CUDA-graph decode step vs eager, on real saved ByT5 inputs. Measurement only.

For each clip (benchmarks/quant/features/<id>.npz, the 200-clip replay set) runs the SHIPPING
model (bf16 decoder, int8 encoder default, beams 4, max_length 768) three ways through HF's own
generate():
  eager     unmodified model
  static    static_decoder math WITHOUT the graph (isolates numerics from capture)
  graph     static_decoder with CUDA-graph replay
and records generated ids, decode steps, and generate() seconds (cuda-synced).

Usage (SHuBERT dir):
  PYTORCH_NO_CUDA_MEMORY_CACHING=0 shubert_venv/bin/python3 benchmarks/cudagraph/cudagraph_decode.py \
      --limit 20 --out benchmarks/cudagraph/results/phase0.jsonl
"""
import argparse
import glob
import json
import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import inference  # noqa: E402
import run_eval  # noqa: E402
import static_decoder  # noqa: E402

FEATS = os.path.join(ROOT, "benchmarks", "quant", "features")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=20)
    ap.add_argument("--out", required=True)
    ap.add_argument("--skip-static", action="store_true", help="skip the no-graph control")
    args = ap.parse_args()

    cfg = run_eval.config
    model, tok, dev = inference._get_cached_model(cfg["slt_model_checkpoint"],
                                                  cfg["slt_tokenizer_checkpoint"], cfg["temp_dir"])
    dt = next(model.parameters()).dtype
    beams = max(1, int(os.environ.get("BYT5_NUM_BEAMS", "4")))
    max_len = max(64, int(os.environ.get("BYT5_MAX_LENGTH", "768")))
    gen_kw = dict(max_length=max_len, num_beams=beams, no_repeat_ngram_size=0, early_stopping=True,
                  pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id)

    paths = sorted(glob.glob(f"{FEATS}/*.npz"))[:args.limit]
    print(f"{len(paths)} clips, beams {beams}, max_length {max_len}, dtype {dt}", flush=True)

    def tensors(z):
        return {k: torch.tensor(z[s], dtype=dt).unsqueeze(0).to(dev) for k, s in
                (("face_features", "face"), ("left_hand_features", "left"),
                 ("right_hand_features", "right"), ("pose_features", "body"))}

    def run(inp):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            ids = model.generate(**inp, **gen_kw)
        torch.cuda.synchronize()
        return ids[0].tolist(), time.time() - t0

    def enc_time(inp):
        torch.cuda.synchronize()
        t0 = time.time()
        with torch.no_grad():
            model.encoder(**inp, return_dict=True)
        torch.cuda.synchronize()
        return time.time() - t0

    # warm up eager on the first clip (cuBLAS handles, allocator)
    run(tensors(np.load(paths[0])))
    base_alloc = torch.cuda.memory_allocated()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    recs = []
    with open(args.out, "w") as fh:
        sd, uninstall = None, None
        for i, p in enumerate(paths):
            cid = os.path.splitext(os.path.basename(p))[0]
            inp = tensors(np.load(p))
            te = enc_time(inp)
            ids_e, s_e = run(inp)
            rec = {"id": cid, "enc_len": None, "steps": len(ids_e) - 1, "enc_s": round(te, 4),
                   "eager_s": round(s_e, 4)}
            if not args.skip_static:
                sd, uninstall = static_decoder.install(model, beams, max_len, use_graph=False, sd=sd)
                ids_s, s_s = run(inp)
                uninstall()
                rec.update(static_s=round(s_s, 4), static_same=ids_s == ids_e)
            sd, uninstall = static_decoder.install(model, beams, max_len, use_graph=True, sd=sd)
            # capture (first time a bucket is seen) happens inside prepare(); keep it out of timing
            n_before = len(sd.graphs)
            with torch.no_grad():
                enc = model.encoder(**inp, return_dict=True)[0]
                sd.prepare(enc)
                sd.capture_all(sd.cur)
            rec["captured_now"] = len(sd.graphs) > n_before
            ids_g, s_g = run(inp)
            uninstall()
            rec.update(graph_s=round(s_g, 4), graph_same=ids_g == ids_e, enc_len=sd.cur,
                       graph_text=tok.decode(ids_g, skip_special_tokens=True),
                       eager_text=tok.decode(ids_e, skip_special_tokens=True))
            n = max(rec["steps"], 1)
            rec["eager_ms_step"] = round(1000 * (s_e - te) / n, 2)
            rec["graph_ms_step"] = round(1000 * (s_g - te) / n, 2)
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            recs.append(rec)
            print(f"{i:3d} {cid:18s} steps {rec['steps']:3d} eager {rec['eager_ms_step']:6.2f} "
                  f"graph {rec['graph_ms_step']:6.2f} ms/step  same(static/graph) "
                  f"{rec.get('static_same')}/{rec['graph_same']}", flush=True)
    import statistics as st
    print(f"\nmedian ms/step eager {st.median(r['eager_ms_step'] for r in recs):.2f}  graph "
          f"{st.median(r['graph_ms_step'] for r in recs):.2f}; generate s eager "
          f"{sum(r['eager_s'] for r in recs):.1f} graph {sum(r['graph_s'] for r in recs):.1f}; "
          f"ids identical {sum(r['graph_same'] for r in recs)}/{len(recs)}"
          + ("" if args.skip_static else f" (static-no-graph {sum(r['static_same'] for r in recs)}/{len(recs)})"))
    print(f"extra CUDA alloc after graphs: {(torch.cuda.memory_allocated() - base_alloc) / 2**20:.1f} MB, "
          f"graphs captured: {len(sd.graphs) if sd else 0} (buckets {sorted(sd.graphs) if sd else []})")


if __name__ == "__main__":
    main()
