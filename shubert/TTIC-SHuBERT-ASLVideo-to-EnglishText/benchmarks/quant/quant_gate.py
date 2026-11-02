#!/usr/bin/env python3
"""Quality/memory gate for torchao weight-only quantization of ByT5+SHuBERT.

Replays the per-clip ByT5 inputs saved by capture_features.py (features/<id>.npz) through
inference.test() with the model quantized in place, so every variant decodes the exact same
features and perception is never re-run.

Variants (quantization touches nn.Linear only; lm_head is tied to the decoder embedding and
is left alone):
  bf16      no quantization -- must reproduce the capture run's text exactly (replay check)
  dec_int4  decoder Linears int4 weight-only, group 128, tile_packed_to_4d (aten tinygemm)
  enc_int8  encoder Linears (incl. the SHuBERT adapter) int8 weight-only, per-row
  both      dec_int4 + enc_int8
  bf16_slowmha  no quantization, fairseq MHA fast path off -- the control for enc_int8/both

enc_int8/both need fairseq's MultiheadAttention fast path OFF: it hands q/k/v_proj.weight
straight to F.multi_head_attention_forward, which calls aten.t on them, and torchao's Int8Tensor
does not implement it. skip_embed_dim_check=True (which otherwise only skips an assert) routes
attention through the q/k/v_proj Linear modules instead. That path is not guaranteed bitwise
equal to the fused one, so compare enc_int8/both against bf16_slowmha, not against bf16.

Output: results/<variant>.jsonl, one fsync'd line per clip (a power cut loses at most one
clip; re-running resumes), then a summary JSON with BLEU/chrF, the number of clips whose text
differs from the bf16 capture run, generate() timing, and model memory.

Usage (from the SHuBERT dir, torchao on PYTHONPATH):
  EVAL_DIR=eval_set_openasl PYTHONPATH=<dir containing torchao> BYT5_DEVICE=cuda \
    shubert_venv/bin/python3 benchmarks/quant/quant_gate.py --variant dec_int4
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
os.chdir(ROOT)

import numpy as np  # noqa: E402
import torch  # noqa: E402

import inference  # noqa: E402
import run_eval  # noqa: E402

FEATS = os.path.join(HERE, "features")
RESULTS = os.path.join(HERE, "results")
VARIANTS = ("bf16", "dec_int4", "enc_int8", "both", "bf16_slowmha")
SLOW_MHA = ("enc_int8", "both", "bf16_slowmha")


def _is_linear(m):
    return isinstance(m, torch.nn.Linear)


def _dec_filter(m, fqn):
    return _is_linear(m) and fqn.startswith("decoder.")


def _enc_filter(m, fqn):
    return _is_linear(m) and not fqn.startswith("decoder.") and fqn != "lm_head"


def _param_bytes(model):
    """Bytes held by parameters, counting torchao tensor subclasses by their inner tensors."""
    total = 0
    for p in model.parameters():
        t = p.data if isinstance(p, torch.nn.Parameter) else p
        names = getattr(t, "tensor_data_names", None)
        if names:
            for n in names:
                total += getattr(t, n).untyped_storage().nbytes()
            for n in getattr(t, "optional_tensor_data_names", None) or []:
                x = getattr(t, n, None)
                if x is not None:
                    total += x.untyped_storage().nbytes()
        else:
            total += t.numel() * t.element_size()
    return total


def quantize(model, variant, info):
    from torchao.quantization import (Int4WeightOnlyConfig, Int8WeightOnlyConfig,
                                      quantize_)
    info["param_mb_before"] = round(_param_bytes(model) / 2**20, 1)
    info["cuda_alloc_mb_before"] = round(torch.cuda.memory_allocated() / 2**20, 1)
    n = {"dec": 0, "enc": 0}
    for fqn, m in model.named_modules():
        if _dec_filter(m, fqn):
            n["dec"] += 1
        elif _enc_filter(m, fqn):
            n["enc"] += 1
    info["linears"] = n
    t0 = time.time()
    if variant in ("dec_int4", "both"):
        quantize_(model, Int4WeightOnlyConfig(group_size=128,
                                              int4_packing_format="tile_packed_to_4d"),
                  filter_fn=_dec_filter)
    if variant in ("enc_int8", "both"):
        quantize_(model, Int8WeightOnlyConfig(), filter_fn=_enc_filter)
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    # Without this the summary's peak is the quantize transient, not inference (11-01 run:
    # dec_int4 and both both reported 1381.6 MB, the int4 packing step).
    info["peak_quantize_mb"] = round(torch.cuda.max_memory_allocated() / 2**20, 1)
    torch.cuda.reset_peak_memory_stats()
    info["quantize_s"] = round(time.time() - t0, 1)
    info["param_mb_after"] = round(_param_bytes(model) / 2**20, 1)
    info["cuda_alloc_mb_after"] = round(torch.cuda.memory_allocated() / 2**20, 1)
    print(f"[quant] {variant}: params {info['param_mb_before']} -> {info['param_mb_after']} MB, "
          f"cuda alloc {info['cuda_alloc_mb_before']} -> {info['cuda_alloc_mb_after']} MB, "
          f"linears {n}, {info['quantize_s']}s", flush=True)


def baseline_hyps(path):
    with open(path) as f:
        return {o["id"]: o["hypothesis"] for o in json.load(f)["outputs"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=VARIANTS, required=True)
    ap.add_argument("--baseline", default=None,
                    help="capture run results JSON (default: newest results_quant_capture_*)")
    ap.add_argument("--limit", type=int, default=0, help="first N clips only (smoke test)")
    args = ap.parse_args()

    eval_dir = run_eval.EVAL_DIR
    items = [json.loads(l) for l in open(os.path.join(eval_dir, "manifest.jsonl"))]
    have = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{FEATS}/*.npz")}
    missing = [it["id"] for it in items if it["id"] not in have]
    items = [it for it in items if it["id"] in have]
    if args.limit:
        items = items[:args.limit]
    print(f"{len(items)} clips with features ({len(missing)} missing)", flush=True)

    base_path = args.baseline or max(glob.glob(os.path.join(eval_dir,
                                                            "results_quant_capture_*.json")))
    base = baseline_hyps(base_path)

    # Quantize once, right after the cached model is first built.
    info = {"variant": args.variant, "baseline": os.path.basename(base_path)}
    orig_get = inference._get_cached_model

    def get_cached(*a, **kw):
        fresh = not inference._model_cache
        out = orig_get(*a, **kw)
        if fresh and args.variant in SLOW_MHA:
            from fairseq.modules.multihead_attention import MultiheadAttention
            mods = [m for m in out[0].modules() if isinstance(m, MultiheadAttention)]
            for m in mods:
                m.skip_embed_dim_check = True
            info["slow_mha_modules"] = len(mods)
        if fresh and args.variant not in ("bf16", "bf16_slowmha"):
            quantize(out[0], args.variant, info)
        elif fresh:
            info["param_mb_after"] = round(_param_bytes(out[0]) / 2**20, 1)
            info["cuda_alloc_mb_after"] = round(torch.cuda.memory_allocated() / 2**20, 1)
        return out
    inference._get_cached_model = get_cached

    # Time generate() alone (test() also prints it; this keeps it machine-readable).
    gen_times = {}
    orig_gen = inference.generate_text_from_features

    def gen(*a, **kw):
        torch.cuda.synchronize()
        t0 = time.time()
        out = orig_gen(*a, **kw)
        torch.cuda.synchronize()
        gen_times["last"] = time.time() - t0
        return out
    inference.generate_text_from_features = gen

    os.makedirs(RESULTS, exist_ok=True)
    out_path = os.path.join(RESULTS, f"{args.variant}{'_lim' + str(args.limit) if args.limit else ''}.jsonl")
    done = {}
    if os.path.exists(out_path):
        for l in open(out_path):
            try:
                r = json.loads(l)
                done[r["id"]] = r
            except ValueError:
                pass  # torn last line from a power cut
    cfg = run_eval.config
    with open(out_path, "a") as fh:
        for i, it in enumerate(items):
            if it["id"] in done:
                continue
            z = np.load(os.path.join(FEATS, f"{it['id']}.npz"))
            hyp = inference.test(z["face"], z["left"], z["right"], z["body"],
                                 cfg["slt_model_config"], cfg["slt_model_checkpoint"],
                                 cfg["slt_tokenizer_checkpoint"], cfg["temp_dir"])
            rec = {"id": it["id"], "hypothesis": hyp,
                   "generate_s": round(gen_times["last"], 3),
                   "same_as_bf16": hyp == base.get(it["id"])}
            fh.write(json.dumps(rec) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
            done[it["id"]] = rec
            print(f"[{i + 1}/{len(items)}] {it['id']} {rec['generate_s']:.2f}s "
                  f"{'same' if rec['same_as_bf16'] else 'DIFF'}", flush=True)

    hyps = [done[it["id"]]["hypothesis"] for it in items]
    res = run_eval.score(items, hyps)
    base_res = run_eval.score(items, [base[it["id"]] for it in items])
    gts = [done[it["id"]]["generate_s"] for it in items]
    first_clip_loaded = "param_mb_after" in info
    summary = {
        **info,
        "n": len(items),
        "differ_vs_bf16": sum(not done[it["id"]]["same_as_bf16"] for it in items),
        "bleu": round(res["bleu"], 2), "chrf": round(res["chrf"], 2),
        "bleu_bf16_same_clips": round(base_res["bleu"], 2),
        "chrf_bf16_same_clips": round(base_res["chrf"], 2),
        "generate_s_median": round(float(np.median(gts)), 3),
        "generate_s_mean": round(float(np.mean(gts)), 3),
        "generate_s_total": round(float(np.sum(gts)), 1),
        "peak_cuda_alloc_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1)
        if first_clip_loaded else None,
    }
    with open(out_path.replace(".jsonl", "_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
