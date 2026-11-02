#!/usr/bin/env python3
"""Run the 200-clip streaming eval (run_eval.py --streaming) and save every clip's ByT5 inputs.

Quantization only touches the ByT5/SHuBERT stage, so once the four feature arrays each clip
hands to `inference.test()` are on disk, every quantization variant can be gated by decoding
those arrays alone (benchmarks/quant/quant_gate.py) instead of re-running perception.

This is run_eval.main() unchanged, with two wrappers:
  * run_eval.score_streaming -> records which clip is in flight
  * features.test            -> saves (face, left, right, body) to features/<id>.npz

Usage (from the SHuBERT dir):
  EVAL_DIR=eval_set_openasl DINOV2_DTYPE=float16 BYT5_DEVICE=cuda \
    shubert_venv/bin/python3 benchmarks/quant/capture_features.py --tag quant_capture --streaming
"""
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np  # noqa: E402

import run_eval  # noqa: E402
import features  # noqa: E402

OUT = os.path.join(HERE, "features")
os.makedirs(OUT, exist_ok=True)
_current = {"id": None}

_orig_stream = run_eval.score_streaming
_orig_test = features.test


def _clip_id(path):
    base = os.path.splitext(os.path.basename(path))[0]
    return re.sub(r"^eval_", "", base)


def score_streaming(processor, path):
    _current["id"] = _clip_id(path)
    try:
        return _orig_stream(processor, path)
    finally:
        _current["id"] = None


def test(face, left, right, body, *a, **kw):
    cid = _current["id"]
    if cid is None:
        raise RuntimeError("features.test called outside a clip")
    # tmp + fsync + rename: run_eval fsyncs its partial line, so without this a power cut left
    # clips marked done with a 0-byte npz (10 of 200 on 2026-10-31/11-01).
    dst = os.path.join(OUT, f"{cid}.npz")
    with open(dst + ".tmp", "wb") as fh:
        np.savez(fh, face=face, left=left, right=right, body=body)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(dst + ".tmp", dst)
    dfd = os.open(OUT, os.O_RDONLY)
    os.fsync(dfd)
    os.close(dfd)
    return _orig_test(face, left, right, body, *a, **kw)


run_eval.score_streaming = score_streaming
features.test = test

if __name__ == "__main__":
    run_eval.main()
