"""Score the pipeline against the personal eval set.

Run this BEFORE any QLoRA work to get a baseline, and again afterwards with the adapter
loaded. Without a baseline there is no way to tell whether fine-tuning helped, and the
whole point of the eval set is to answer one question: is the weakness fingerspelling
specifically, or general translation quality?

    python3 run_eval.py                     # score current pipeline
    python3 run_eval.py --tag baseline      # label the results file
    python3 run_eval.py --compare a.json b.json

Reports corpus BLEU and chrF overall and per category, plus a proper-noun recall check
that is the actual metric of interest for the fingerspelling items -- BLEU barely moves
when one name in a sentence is wrong, but that one name is the whole failure.
"""
import argparse
import json
import os
import re
import time

import sacrebleu

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "eval_set")
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


def score(items, hyps):
    refs = [it["reference"] for it in items]
    out = {
        "n": len(items),
        "bleu": sacrebleu.corpus_bleu(hyps, [refs]).score,
        "chrf": sacrebleu.corpus_chrf(hyps, [refs]).score,
        "by_category": {},
    }
    cats = sorted({it["category"] for it in items})
    for cat in cats:
        idxs = [i for i, it in enumerate(items) if it["category"] == cat]
        c_refs = [refs[i] for i in idxs]
        c_hyps = [hyps[i] for i in idxs]
        out["by_category"][cat] = {
            "n": len(idxs),
            "bleu": sacrebleu.corpus_bleu(c_hyps, [c_refs]).score,
            "chrf": sacrebleu.corpus_chrf(c_hyps, [c_refs]).score,
        }

    # Proper-noun recall: of the fingerspelled names in the references, how many appear
    # anywhere in the corresponding hypothesis?
    hit = total = 0
    misses = []
    for it, hyp in zip(items, hyps):
        want = proper_nouns(it["reference"])
        if not want:
            continue
        got = {w for w in re.findall(r"[A-Za-z']+", hyp.lower())}
        for w in want:
            total += 1
            if w in got:
                hit += 1
            else:
                misses.append((it["id"], w, hyp))
    out["proper_noun_recall"] = (hit / total * 100) if total else None
    out["proper_noun_total"] = total
    out["proper_noun_misses"] = misses
    return out


def report(res, tag=""):
    print("\n" + "=" * 72)
    print(f"EVAL RESULTS {tag}".rstrip())
    print("=" * 72)
    print(f"  clips: {res['n']}")
    print(f"  BLEU : {res['bleu']:.2f}")
    print(f"  chrF : {res['chrf']:.2f}")
    print("\n  by category:")
    for cat, c in sorted(res["by_category"].items()):
        print(f"    {cat:12s} n={c['n']:3d}  BLEU {c['bleu']:6.2f}  chrF {c['chrf']:6.2f}")
    if res["proper_noun_recall"] is not None:
        print(f"\n  proper-noun recall: {res['proper_noun_recall']:.1f}% "
              f"({res['proper_noun_total']} names)")
        if res["proper_noun_misses"]:
            print("  missed names:")
            for cid, word, hyp in res["proper_noun_misses"][:12]:
                print(f"    [{cid}] '{word}' -> {hyp}")
    print("=" * 72)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="", help="label for the results file")
    ap.add_argument("--compare", nargs=2, metavar=("A", "B"),
                    help="compare two existing results files, no inference")
    args = ap.parse_args()

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

    from features import SHuBERTProcessor
    os.makedirs(config['temp_dir'], exist_ok=True)
    processor = SHuBERTProcessor(config)
    t0 = time.time()
    processor.warmup()
    print(f"warmup {time.time() - t0:.1f}s")

    hyps, times = [], []
    for it in items:
        path = os.path.join(EVAL_DIR, it["file"])
        t0 = time.time()
        try:
            hyp = processor.process_video(path)
        except Exception as e:
            hyp = ""
            print(f"[{it['id']}] FAILED {type(e).__name__}: {e}")
        dt = time.time() - t0
        hyps.append(hyp)
        times.append(dt)
        print(f"[{it['id']}] {it['category']:12s} {dt:5.1f}s")
        print(f"      ref: {it['reference']}")
        print(f"      hyp: {hyp}")

    res = score(items, hyps)
    report(res, args.tag)

    stamp = time.strftime("%Y%m%d-%H%M%S")
    name = f"results_{args.tag + '_' if args.tag else ''}{stamp}.json"
    out_path = os.path.join(EVAL_DIR, name)
    with open(out_path, "w") as f:
        json.dump({
            "tag": args.tag,
            "timestamp": stamp,
            "frame_stride": os.environ.get("FRAME_STRIDE", "2"),
            "use_onnx_perception": os.environ.get("USE_ONNX_PERCEPTION", "0"),
            "mean_seconds_per_clip": sum(times) / len(times) if times else 0,
            "results": res,
            "outputs": [
                {"id": it["id"], "category": it["category"],
                 "reference": it["reference"], "hypothesis": h, "seconds": round(t, 1)}
                for it, h, t in zip(items, hyps, times)
            ],
        }, f, indent=2)
    print(f"\nwrote {out_path}")


if __name__ == "__main__":
    main()
