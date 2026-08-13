"""Score the pipeline against the personal eval set.

Run this BEFORE any QLoRA work to get a baseline, and again afterwards with the adapter
loaded. Without a baseline there is no way to tell whether fine-tuning helped, and the
whole point of the eval set is to answer one question: is the weakness fingerspelling
specifically, or general translation quality?

    python3 run_eval.py                     # score current pipeline
    python3 run_eval.py --tag baseline      # label the results file
    python3 run_eval.py --fresh             # discard a partial run, start over
    python3 run_eval.py --compare a.json b.json

Reports corpus BLEU and chrF overall and per category, plus a proper-noun recall check
that is the actual metric of interest for the fingerspelling items -- BLEU barely moves
when one name in a sentence is wrong, but that one name is the whole failure.

A full 50-clip run takes over half an hour, so each clip's hypothesis is appended to a
partial file the moment it is produced and re-running resumes from there. Losing the
machine at clip 49 used to cost the entire run.
"""
import argparse
import json
import os
import re
import time

import cv2
import numpy as np
import sacrebleu

HERE = os.path.dirname(os.path.abspath(__file__))
# EVAL_DIR lets the same harness score a second set -- e.g. an OpenASL benchmark subset
# of native signing with published references, alongside our own-footage set.
EVAL_DIR = os.environ.get("EVAL_DIR", os.path.join(HERE, "eval_set"))
MANIFEST = os.path.join(EVAL_DIR, "manifest.jsonl")

MODELS_BASE = "/home/sllu/.cache/huggingface/hub/models--ShesterG--SHuBERT/snapshots/578a0233e770c8ce4dc75d859b91fdea7c34f5aa/models"

config = {
    'yolov8_model_path': os.path.join(MODELS_BASE, 'yolov8n.pt'),
    'dino_face_model_path': os.path.join(MODELS_BASE, 'dinov2face.pth'),
    'dino_hands_model_path': os.path.join(MODELS_BASE, 'dinov2hand.pth'),
    'mediapipe_face_model_path': os.path.join(MODELS_BASE, 'face_landmarker_v2_with_blendshapes.task'),
    'mediapipe_hands_model_path': os.path.join(MODELS_BASE, 'hand_landmarker.task'),
    'shubert_model_path': os.path.join(MODELS_BASE, 'checkpoint_836_400000.pt'),
    'slt_model_config': os.path.join(MODELS_BASE, 'byt5_base', 'config.json'),
    'slt_model_checkpoint': os.path.join(MODELS_BASE, 'checkpoint-11625'),
    'slt_tokenizer_checkpoint': os.path.join(MODELS_BASE, 'byt5_base'),
    'temp_dir': 'temp',
}

# Words that are capitalised mid-sentence are the proper nouns the signer fingerspelled.
# Sentence-initial capitals are excluded since they carry no fingerspelling signal.
_STOP_INITIAL = re.compile(r"^[A-Z][a-z]*$")


def proper_nouns(sentence):
    tokens = re.findall(r"[A-Za-z']+", sentence)
    return {t.lower() for i, t in enumerate(tokens)
            if i > 0 and t[0].isupper() and not _STOP_INITIAL.match(t) is None}


# --- number normalisation ---------------------------------------------------------
# The model writes numerals ("15", "10th", "$5") where the references spell them out
# ("fifteen", "tenth", "five dollars"). Raw BLEU counts those as errors even when the
# number is right, which is why the numbers category first scored 5.05 -- an artefact,
# not a measurement. Canonicalise both sides to digits before scoring.
_UNITS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
    "seven": 7, "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12,
    "thirteen": 13, "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
}
_TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
         "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90}
_ORDINALS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6,
    "seventh": 7, "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11,
    "twelfth": 12, "thirteenth": 13, "fourteenth": 14, "fifteenth": 15,
    "twentieth": 20, "thirtieth": 30,
}


def normalize_numbers(text):
    text = re.sub(r"\$\s*(\d+)", r"\1 dollars", text)
    tokens = re.findall(r"[A-Za-z']+|\d+|[^\sA-Za-z\d']", text.lower())

    out = []
    for tok in tokens:
        if tok in _UNITS:
            out.append(str(_UNITS[tok]))
        elif tok in _TENS:
            out.append(str(_TENS[tok]))
        elif tok in _ORDINALS:
            out.append(str(_ORDINALS[tok]))
        elif re.fullmatch(r"\d+(st|nd|rd|th)", tok):
            out.append(re.sub(r"(st|nd|rd|th)$", "", tok))
        else:
            out.append(tok)

    # "forty five" -> 45, after both halves became digits; and drop the orphaned
    # ordinal suffix left behind when the tokenizer splits "10th" into "10" + "th".
    merged = []
    i = 0
    while i < len(out):
        if (i + 1 < len(out) and out[i].isdigit() and out[i + 1].isdigit()
                and int(out[i]) in _TENS.values() and 1 <= int(out[i + 1]) <= 9):
            merged.append(str(int(out[i]) + int(out[i + 1])))
            i += 2
        elif (out[i] in ("st", "nd", "rd", "th")
              and merged and merged[-1].isdigit()):
            i += 1
        else:
            merged.append(out[i])
            i += 1
    return " ".join(merged)


def char_similarity(a, b):
    """1.0 - normalised Levenshtein distance. 'blian' vs 'brian' -> 0.8."""
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return 1.0 - prev[-1] / max(len(a), len(b))


NEAR_HIT = 0.6  # 'blian'/'brian' = 0.80, 'columbs'/'columbus' = 0.88, 'osd'/'osu' = 0.67


# Recorded clips are bracketed by however long it took to reach for the spacebar --
# often 5-7s of stillness around a 4s sign. That matters twice over: latency is ~0.22s
# per captured frame, and dead air is what made the model hallucinate during live
# testing. It also makes the eval unfaithful, because the live path feeds process_video
# clips that auto_segment_v5 has already trimmed. So trim here with the same
# frame-differencing logic and thresholds the live segmenter uses.
MOTION_FLOOR = 0.5          # absolute noise floor for mean frame difference
TRIM_PAD_SECONDS = 0.25     # matches auto_segment_v5.TAIL_PAD_SECONDS
MIN_KEEP_FRACTION = 0.40    # below this, distrust the trim and keep the whole clip


def trim_to_motion(src, dst, fps=30.0):
    """Write src to dst keeping only the moving span. Returns (kept, total).

    The threshold is derived per clip rather than fixed. A fixed value (1.5, borrowed
    from the live segmenter) cut clip 005 from 387 frames to 18 -- 0.6s, far too short to
    contain the sign -- because that clip's signing motion simply sat below it. Lighting
    and signing energy vary enough between clips that an absolute cutoff is unsafe.

    Two guards, because silently discarding most of a sign is much worse than leaving
    some dead air in: the threshold sits a quarter of the way up the clip's own motion
    range, and if the result keeps less than MIN_KEEP_FRACTION the trim is abandoned
    entirely rather than trusted.
    """
    cap = cv2.VideoCapture(src)
    frames = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        return 0, 0

    prev = None
    scores = []
    for frame in frames:
        gray = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)
        scores.append(0.0 if prev is None
                      else float(np.mean(cv2.absdiff(prev, gray))))
        prev = gray

    quiet, loud = np.percentile(scores, 20), np.percentile(scores, 90)
    threshold = max(MOTION_FLOOR, quiet + 0.25 * (loud - quiet))

    idxs = [i for i, s in enumerate(scores) if s > threshold]
    if not idxs:
        return len(frames), len(frames)  # no motion detected: keep everything

    pad = int(TRIM_PAD_SECONDS * fps)
    lo = max(0, idxs[0] - pad)
    hi = min(len(frames), idxs[-1] + pad + 1)
    if (hi - lo) < MIN_KEEP_FRACTION * len(frames):
        return len(frames), len(frames)  # implausible span: don't trust it
    kept = frames[lo:hi]

    h, w = kept[0].shape[:2]
    writer = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for f in kept:
        writer.write(f)
    writer.release()
    return len(kept), len(frames)


def load_manifest():
    if not os.path.exists(MANIFEST):
        raise SystemExit(
            f"no eval set found at {MANIFEST}\n"
            f"record one first:  python3 record_eval_clips.py")
    items = []
    with open(MANIFEST) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


# --- crash-resumable partial results ----------------------------------------------
# Clips are appended here as they finish so an interrupted run can pick up where it
# stopped. The file is deleted once the final results JSON is written, so its presence
# means "a run died partway".


def partial_path(tag):
    # Keyed by tag so two configurations run under different tags cannot resume into
    # each other. Dotfile to keep it out of the results listing.
    return os.path.join(EVAL_DIR, f".partial_{tag or 'default'}.jsonl")


def append_partial(fh, record):
    """Append one JSONL record and force it to disk.

    fsync, not just flush: the failure being guarded against is the machine losing
    power mid-run, and a flushed-but-unsynced line is exactly what a power cut eats.
    """
    fh.write(json.dumps(record) + "\n")
    fh.flush()
    os.fsync(fh.fileno())


def load_partial(path, run_cfg):
    """Read completed clips from an interrupted run. Returns {clip_id: record}.

    Refuses to resume across a settings change. Silently mixing, say, fp16 and fp32
    clips into one BLEU score would produce a number that describes no configuration
    at all -- worse than losing the run, because it looks valid.
    """
    done = {}
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                # A power cut can truncate the line being written. By construction it
                # is the last one, so drop it and recompute that clip.
                print(f"  {os.path.basename(path)}:{lineno} truncated, ignoring")
                continue
            if "_config" in rec:
                # Compare the UNION of keys. Iterating only the saved record hides any knob
                # added since it was written: a partial from before `crop_jitter_mode`
                # existed would resume under CROP_JITTER_MODE=static and merge two jitter
                # modes into one score -- precisely what this guard exists to prevent.
                saved = rec["_config"]
                MISSING = "<not recorded>"
                differs = {k: (saved.get(k, MISSING), run_cfg.get(k, MISSING))
                           for k in saved.keys() | run_cfg.keys()
                           if saved.get(k, MISSING) != run_cfg.get(k, MISSING)}
                if differs:
                    raise SystemExit(
                        f"{path}\nis from a run with different settings, so resuming "
                        f"it would mix configurations in one score:\n"
                        + "".join(f"  {k}: saved {s!r}, now {n!r}\n"
                                  for k, (s, n) in differs.items())
                        + "pass --fresh to discard it, or use a different --tag.")
                continue
            done[rec["id"]] = rec
    return done


def score(items, hyps):
    refs = [it["reference"] for it in items]
    n_refs = [normalize_numbers(r) for r in refs]
    n_hyps = [normalize_numbers(h) for h in hyps]
    out = {
        "n": len(items),
        "bleu_raw": sacrebleu.corpus_bleu(hyps, [refs]).score,
        "bleu": sacrebleu.corpus_bleu(n_hyps, [n_refs]).score,
        "chrf": sacrebleu.corpus_chrf(n_hyps, [n_refs]).score,
        "by_category": {},
    }
    cats = sorted({it["category"] for it in items})
    for cat in cats:
        idxs = [i for i, it in enumerate(items) if it["category"] == cat]
        out["by_category"][cat] = {
            "n": len(idxs),
            "bleu_raw": sacrebleu.corpus_bleu([hyps[i] for i in idxs],
                                              [[refs[i] for i in idxs]]).score,
            "bleu": sacrebleu.corpus_bleu([n_hyps[i] for i in idxs],
                                          [[n_refs[i] for i in idxs]]).score,
            "chrf": sacrebleu.corpus_chrf([n_hyps[i] for i in idxs],
                                          [[n_refs[i] for i in idxs]]).score,
        }

    # Proper nouns are scored by character similarity, not exact match. Fingerspelling
    # fails by degrees -- "Blian" for "Brian" is one character out, which exact matching
    # scores identically to a total miss and so hides the signal that matters most.
    exact = near = total = 0
    sims = []
    detail = []
    for it, hyp in zip(items, hyps):
        want = proper_nouns(it["reference"])
        if not want:
            continue
        got = re.findall(r"[A-Za-z']+", hyp.lower())
        for w in want:
            total += 1
            best, best_tok = 0.0, ""
            for g in got:
                s = char_similarity(w, g)
                if s > best:
                    best, best_tok = s, g
            sims.append(best)
            if best == 1.0:
                exact += 1
            elif best >= NEAR_HIT:
                near += 1
                detail.append((it["id"], w, best_tok, round(best, 2), hyp))
            else:
                detail.append((it["id"], w, best_tok or "-", round(best, 2), hyp))
    out["proper_noun_total"] = total
    out["proper_noun_recall"] = (exact / total * 100) if total else None
    out["proper_noun_near_recall"] = ((exact + near) / total * 100) if total else None
    out["proper_noun_mean_similarity"] = (sum(sims) / len(sims) * 100) if sims else None
    out["proper_noun_detail"] = detail
    return out


def report(res, tag=""):
    print("\n" + "=" * 72)
    print(f"EVAL RESULTS {tag}".rstrip())
    print("=" * 72)
    print(f"  clips: {res['n']}")
    print(f"  BLEU : {res['bleu']:.2f}   (raw, before number normalisation: "
          f"{res.get('bleu_raw', float('nan')):.2f})")
    print(f"  chrF : {res['chrf']:.2f}")
    print("\n  by category:")
    for cat, c in sorted(res["by_category"].items()):
        raw = c.get("bleu_raw")
        raw_s = f" (raw {raw:5.2f})" if raw is not None else ""
        print(f"    {cat:12s} n={c['n']:3d}  BLEU {c['bleu']:6.2f}{raw_s}  "
              f"chrF {c['chrf']:6.2f}")
    if res.get("proper_noun_recall") is not None:
        print(f"\n  proper nouns ({res['proper_noun_total']} names):")
        print(f"    exact          : {res['proper_noun_recall']:.1f}%")
        print(f"    near (>={NEAR_HIT:.1f} sim): {res['proper_noun_near_recall']:.1f}%")
        print(f"    mean similarity: {res['proper_noun_mean_similarity']:.1f}%")
        if res.get("proper_noun_detail"):
            print("    non-exact:")
            for cid, want, got, sim, _hyp in res["proper_noun_detail"][:14]:
                mark = "~" if sim >= NEAR_HIT else " "
                print(f"      {mark} [{cid}] {want:12s} -> {got:14s} sim {sim:.2f}")
    print("=" * 72)


def score_streaming(processor, path):
    """Translate a clip through StreamingPerception instead of process_video().

    The default eval path is `process_video()` -> `video_holistic()`, a single detector
    processing frames sequentially. That means it CANNOT see anything about the live
    path's parallel perception -- frames are fed to `PERCEPTION_WORKERS` detectors in
    chunks there, so landmarks differ at every chunk boundary. Without this branch a
    perception change can look validated when it was never exercised.

    Frames are pushed as fast as they read rather than at wall-clock camera rate: this
    measures the TEXT, and pacing would only make the run take as long as the footage.
    Latency conclusions must come from live_worker_probe.py, not from here.
    """
    from streaming_perception import StreamingPerception, stride_from_env

    stride = stride_from_env()
    cap = cv2.VideoCapture(path)
    stream = StreamingPerception(
        config['mediapipe_face_model_path'],
        config['mediapipe_hands_model_path'],
        embed_config=config,
    )
    try:
        read_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if read_count % stride == 0:
                stream.add_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            read_count += 1
        cap.release()
        frames, landmarks, embeddings = stream.finish()
        return processor.process_frames(
            frames, landmarks=landmarks,
            mediapipe_seconds=stream.busy_seconds,
            embeddings=embeddings,
            embed_seconds=stream.embed_busy_seconds)
    finally:
        stream.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="label for the results file")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two existing results files, no inference")
    ap.add_argument("--no-trim", action="store_true",
                    help="score the raw clips instead of trimming to the moving span")
    ap.add_argument("--rescore", metavar="FILE",
                    help="recompute metrics from a saved results file (no inference)")
    ap.add_argument("--streaming", action="store_true",
                    help="translate through StreamingPerception (the live path) instead of "
                         "process_video, so parallel-perception changes are exercised")
    ap.add_argument("--fresh", action="store_true",
                    help="discard any partial run for this tag and start from clip 1")
    args = ap.parse_args()

    if args.rescore:
        data = json.load(open(args.rescore))
        items = [{"id": o["id"], "category": o["category"],
                  "reference": o["reference"]} for o in data["outputs"]]
        hyps = [o["hypothesis"] for o in data["outputs"]]
        res = score(items, hyps)
        report(res, f"{data.get('tag', '')} (rescored)")
        data["results"] = res
        json.dump(data, open(args.rescore, "w"), indent=2)
        print(f"updated {args.rescore}")
        return

    if args.compare:
        a = json.load(open(args.compare[0]))
        b = json.load(open(args.compare[1]))
        print(f"\n{'metric':22s} {'A':>10s} {'B':>10s} {'delta':>10s}")
        print("-" * 56)
        for key in ("bleu", "chrf", "proper_noun_recall"):
            av, bv = a["results"].get(key), b["results"].get(key)
            if av is None or bv is None:
                continue
            print(f"{key:22s} {av:10.2f} {bv:10.2f} {bv - av:+10.2f}")
        for cat in sorted(a["results"]["by_category"]):
            av = a["results"]["by_category"][cat]["bleu"]
            bv = b["results"]["by_category"].get(cat, {}).get("bleu")
            if bv is not None:
                print(f"{'BLEU ' + cat:22s} {av:10.2f} {bv:10.2f} {bv - av:+10.2f}")
        return

    items = load_manifest()
    print(f"{len(items)} clips in eval set")

    # Imported here, not at module scope, so --compare/--rescore stay torch-free.
    import dinov2_features
    import crop_jitter
    from features import SHuBERTProcessor

    # Everything that changes the output and therefore must not vary across a resume.
    run_cfg = {
        "frame_stride": os.environ.get("FRAME_STRIDE", "2"),
        "use_onnx_perception": os.environ.get("USE_ONNX_PERCEPTION", "0"),
        "mediapipe_video_mode": os.environ.get("MEDIAPIPE_VIDEO_MODE", "1"),
        "streaming": args.streaming,
        # Only affects output when --streaming: the sequential path uses one detector.
        "perception_workers": (os.environ.get("PERCEPTION_WORKERS", "2")
                               if args.streaming else "n/a"),
        "perception_chunk": (os.environ.get("PERCEPTION_CHUNK", "30")
                             if args.streaming else "n/a"),
        "dinov2_hands_dtype": str(dinov2_features.HANDS_DTYPE),
        "dinov2_face_dtype": str(dinov2_features.FACE_DTYPE),
        "byt5_dtype": os.environ.get("BYT5_DTYPE", "bfloat16"),
        "byt5_device": os.environ.get("BYT5_DEVICE", "cuda"),
        "byt5_num_beams": os.environ.get("BYT5_NUM_BEAMS", "4"),
        "byt5_max_length": os.environ.get("BYT5_MAX_LENGTH", "768"),
        # Perturbation condition. Load-bearing in the resume guard: resuming a
        # clean run into a jittered one would produce a BLEU score describing no
        # configuration at all, while still looking valid.
        "crop_jitter_px": os.environ.get("CROP_JITTER_PX", "0"),
        "crop_jitter_scale": os.environ.get("CROP_JITTER_SCALE", "0"),
        "crop_jitter_seed": os.environ.get("CROP_JITTER_SEED", "0"),
        "crop_jitter_mode": os.environ.get("CROP_JITTER_MODE", "perframe"),
        # Canonicalised, not the raw env string: "hands" and "left_hand,right_hand" are the
        # same condition and must not look like different ones to the resume guard.
        "crop_jitter_streams": crop_jitter.streams_spec(),
        "no_trim": args.no_trim,
    }

    part_path = partial_path(args.tag)
    if args.fresh and os.path.exists(part_path):
        os.remove(part_path)
        print(f"--fresh: discarded {os.path.basename(part_path)}")
    is_new = not os.path.exists(part_path)
    done = load_partial(part_path, run_cfg)

    os.makedirs(config['temp_dir'], exist_ok=True)
    remaining = [it for it in items if it["id"] not in done]
    if done:
        print(f"resuming from {os.path.basename(part_path)}: "
              f"{len(done)} clips done, {len(remaining)} to go")

    # Skipping warmup when there is nothing left means a resume that only needs the
    # scoring pass never loads the models at all.
    if remaining:
        processor = SHuBERTProcessor(config)
        t0 = time.time()
        processor.warmup()
        print(f"warmup {time.time() - t0:.1f}s")

    with open(part_path, "a") as pf:
        if is_new:
            append_partial(pf, {"_config": run_cfg})
        for it in remaining:
            path = os.path.join(EVAL_DIR, it["file"])
            trimmed = None
            if not args.no_trim:
                trimmed = os.path.join(config['temp_dir'], f"eval_{it['id']}.mp4")
                kept, total = trim_to_motion(path, trimmed)
                if kept and kept < total:
                    print(f"[{it['id']}] trimmed {total} -> {kept} frames")
                    path = trimmed
                else:
                    trimmed = None

            t0 = time.time()
            try:
                hyp = (score_streaming(processor, path) if args.streaming
                       else processor.process_video(path))
            except Exception as e:
                hyp = ""
                print(f"[{it['id']}] FAILED {type(e).__name__}: {e}")
            dt = time.time() - t0
            if trimmed:
                try:
                    os.remove(trimmed)
                except OSError:
                    pass
            rec = {"id": it["id"], "category": it["category"],
                   "reference": it["reference"], "hypothesis": hyp,
                   "seconds": round(dt, 1)}
            append_partial(pf, rec)
            done[it["id"]] = rec
            print(f"[{it['id']}] {it['category']:12s} {dt:5.1f}s")
            print(f"      ref: {it['reference']}")
            print(f"      hyp: {hyp}")

    # Rebuilt in manifest order, not completion order, so a resumed run scores
    # identically to one that ran straight through.
    hyps = [done[it["id"]]["hypothesis"] for it in items]
    times = [done[it["id"]]["seconds"] for it in items]

    res = score(items, hyps)
    report(res, args.tag)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"results_{args.tag + '_' if args.tag else ''}{stamp}.json"
    out_path = os.path.join(EVAL_DIR, name)
    with open(out_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "timestamp": stamp,
            **run_cfg,
            "mean_seconds_per_clip": sum(times) / len(times) if times else 0,
            "results": res,
            "outputs": [
                {"id": it["id"], "category": it["category"],
                 "reference": it["reference"], "hypothesis": h, "seconds": round(t, 1)}
                for it, h, t in zip(items, hyps, times)
            ],
        }, f, indent=2)
    print(f"\nwrote {out_path}")

    # Only now, with the real results safely written.
    try:
        os.remove(part_path)
    except OSError:
        pass


if __name__ == "__main__":
    main()
