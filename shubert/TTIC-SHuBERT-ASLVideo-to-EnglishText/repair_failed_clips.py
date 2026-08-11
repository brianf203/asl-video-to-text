#!/usr/bin/env python3
"""Re-run the clips that FAILED in an eval and patch their hypotheses back in.

A clip that raised (CUDA OOM is the one that happens here, under memory pressure)
is recorded with an empty hypothesis. An empty hypothesis is not a quality datapoint
-- it drags corpus BLEU down for a reason that has nothing to do with the config under
test, so a run containing one is not comparable against runs that had none. This
re-runs only those clips, in a fresh process with the box in whatever state it is now,
and writes the hypotheses into the results JSON.

Then rescore, which recomputes every metric from the patched outputs:
    python3 run_eval.py --rescore eval_set_openasl/results_<tag>_<stamp>.json

The env vars for the config MUST match the run being repaired -- the script checks the
config recorded in the results file and refuses if they differ, since a clip decoded
under different settings is exactly the kind of silent mixing the eval's resume guard
exists to prevent.

    EVAL_DIR=eval_set_openasl PERCEPTION_CHUNK=20 python3 repair_failed_clips.py \
        eval_set_openasl/results_stream_w2_c20_<stamp>.json
"""

import json
import os
import sys
import time

import run_eval
from run_eval import EVAL_DIR, config, load_manifest, score_streaming, trim_to_motion


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    results_path = sys.argv[1]
    with open(results_path) as f:
        data = json.load(f)

    failed = [o["id"] for o in data["outputs"] if not o["hypothesis"].strip()]
    if not failed:
        print("no empty hypotheses -- nothing to repair")
        return
    print(f"{len(failed)} clip(s) to re-run: {', '.join(failed)}")

    import dinov2_features
    from features import SHuBERTProcessor

    # Refuse to repair with settings that differ from the run being patched.
    want = {
        "frame_stride": os.environ.get("FRAME_STRIDE", "2"),
        "use_onnx_perception": os.environ.get("USE_ONNX_PERCEPTION", "0"),
        "mediapipe_video_mode": os.environ.get("MEDIAPIPE_VIDEO_MODE", "1"),
        "perception_workers": os.environ.get("PERCEPTION_WORKERS", "2"),
        "perception_chunk": os.environ.get("PERCEPTION_CHUNK", "10"),
        "dinov2_hands_dtype": str(dinov2_features.HANDS_DTYPE),
        "dinov2_face_dtype": str(dinov2_features.FACE_DTYPE),
        "byt5_dtype": os.environ.get("BYT5_DTYPE", "bfloat16"),
        "byt5_device": os.environ.get("BYT5_DEVICE", "cuda"),
        "byt5_num_beams": os.environ.get("BYT5_NUM_BEAMS", "4"),
    }
    differs = {k: (data.get(k), v) for k, v in want.items()
               if k in data and str(data[k]) != str(v)}
    if differs:
        sys.exit(f"config mismatch (file -> env): {differs}\n"
                 "Set the env vars to match the run, or the repaired clip would be "
                 "decoded under different settings than the other 199.")
    if not data.get("streaming"):
        sys.exit("this repairs --streaming runs only")

    items = {it["id"]: it for it in load_manifest()}
    processor = SHuBERTProcessor(config)
    t0 = time.time()
    processor.warmup()
    print(f"warmup {time.time() - t0:.1f}s")

    os.makedirs(config['temp_dir'], exist_ok=True)
    fixed = {}
    for cid in failed:
        it = items[cid]
        path = os.path.join(EVAL_DIR, it["file"])
        trimmed = None
        if not data.get("no_trim"):
            trimmed = os.path.join(config['temp_dir'], f"repair_{cid}.mp4")
            kept, total = trim_to_motion(path, trimmed)
            if kept and kept < total:
                path = trimmed
            else:
                trimmed = None
        t0 = time.time()
        try:
            hyp = score_streaming(processor, path)
        except Exception as e:
            print(f"[{cid}] FAILED AGAIN {type(e).__name__}: {e}")
            continue
        finally:
            if trimmed and os.path.exists(trimmed):
                os.remove(trimmed)
        print(f"[{cid}] {time.time() - t0:.1f}s  {hyp}")
        fixed[cid] = hyp

    if not fixed:
        sys.exit("nothing repaired -- results file left untouched")

    for o in data["outputs"]:
        if o["id"] in fixed:
            o["hypothesis"] = fixed[o["id"]]
    # Record the patch in the file itself so the run is never mistaken for a clean one.
    data.setdefault("repaired_clips", []).extend(sorted(fixed))
    with open(results_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"\npatched {len(fixed)} clip(s) into {results_path}")
    print(f"now run:  python3 run_eval.py --rescore {results_path}")


if __name__ == "__main__":
    main()
