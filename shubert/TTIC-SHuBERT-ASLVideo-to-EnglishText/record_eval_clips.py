"""Guided recorder for building a personal evaluation set.

Purpose: before spending cluster time on QLoRA, we need to know whether fine-tuning on
How2Sign would actually fix the failure we keep seeing (fingerspelled names) or only the
general fluency that is already mostly fine. That needs a labelled set of OUR OWN
signing, since every accuracy claim so far has been eyeballed on unlabelled clips.

Usage:
    python3 record_eval_clips.py            # record any prompts not yet done
    python3 record_eval_clips.py --redo 007 # re-record one item

Keys:  SPACE start/stop recording   r redo last   s skip   q quit (progress is saved)

Clips are stored at the camera's native 30fps -- do NOT pre-subsample. The pipeline
applies FRAME_STRIDE itself, and keeping the raw frames means the set stays valid if
that setting ever changes.
"""
import argparse
import json
import os
import sys
import time

import cv2

HERE = os.path.dirname(os.path.abspath(__file__))
EVAL_DIR = os.path.join(HERE, "eval_set")
CLIP_DIR = os.path.join(EVAL_DIR, "clips")
MANIFEST = os.path.join(EVAL_DIR, "manifest.jsonl")
PROMPTS = os.path.join(EVAL_DIR, "prompts.tsv")

CAMERA_INDEX = 0

# Categories are chosen so the eval can answer the question that actually matters:
# is the weakness fingerspelling specifically, or general translation quality?
# Keep 'fingerspell' items to names you will sign identically every time.
DEFAULT_PROMPTS = [
    ("fingerspell", "Hello, my name is Brian."),
    ("fingerspell", "My friend's name is Sarah."),
    ("fingerspell", "I go to Ohio State University."),
    ("fingerspell", "My teacher is Mr. Smith."),
    ("fingerspell", "I live in Columbus."),
    ("fingerspell", "Her name is Emily and she is deaf."),
    ("fingerspell", "I am learning ASL with Kevin."),
    ("fingerspell", "The doctor's name is Patel."),

    ("numbers", "I have three brothers and two sisters."),
    ("numbers", "The class starts at nine in the morning."),
    ("numbers", "I am twenty years old."),
    ("numbers", "There are fifteen students in the room."),
    ("numbers", "My birthday is June tenth."),
    ("numbers", "It costs forty five dollars."),

    ("general", "Nice to meet you."),
    ("general", "What is your name?"),
    ("general", "I am hungry, let's eat lunch."),
    ("general", "Can you help me please?"),
    ("general", "I do not understand, sign slowly."),
    ("general", "Thank you very much."),
    ("general", "Where is the bathroom?"),
    ("general", "I am learning sign language."),

    ("complex", "Yesterday I went to the store and bought food for dinner."),
    ("complex", "My mother said she will visit us next weekend."),
    ("complex", "I want to become a teacher because I like helping children."),
    ("complex", "The weather is cold today so I am wearing a jacket."),
    ("complex", "If you finish your homework, we can watch a movie later."),
    ("complex", "I have been studying sign language for two years now."),
]


def ensure_prompts():
    os.makedirs(CLIP_DIR, exist_ok=True)
    if not os.path.exists(PROMPTS):
        with open(PROMPTS, "w") as f:
            f.write("# category\treference English sentence\n")
            f.write("# Edit freely -- these are only a starting point. Sign the ASL\n")
            f.write("# equivalent; the reference is the English you expect back.\n")
            for cat, text in DEFAULT_PROMPTS:
                f.write(f"{cat}\t{text}\n")
        print(f"wrote starter prompts to {PROMPTS}")

    items = []
    with open(PROMPTS) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            cat, _, text = line.partition("\t")
            if text:
                items.append((cat.strip(), text.strip()))
    return items


def load_done():
    done = {}
    if os.path.exists(MANIFEST):
        with open(MANIFEST) as f:
            for line in f:
                line = line.strip()
                if line:
                    rec = json.loads(line)
                    done[rec["id"]] = rec
    return done


def save_manifest(records):
    with open(MANIFEST, "w") as f:
        for rec in sorted(records.values(), key=lambda r: r["id"]):
            f.write(json.dumps(rec) + "\n")


def wrap(text, width=52):
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--redo", help="re-record a single id, e.g. 007")
    args = ap.parse_args()

    prompts = ensure_prompts()
    records = load_done()

    todo = []
    for i, (cat, text) in enumerate(prompts, start=1):
        cid = f"{i:03d}"
        if args.redo:
            if cid == args.redo:
                todo.append((cid, cat, text))
        elif cid not in records:
            todo.append((cid, cat, text))

    if not todo:
        print(f"nothing to record -- {len(records)}/{len(prompts)} already done")
        print(f"manifest: {MANIFEST}")
        return

    print(f"{len(todo)} clip(s) to record ({len(records)}/{len(prompts)} already done)")

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    if not cap.isOpened():
        print(f"ERROR: could not open camera {CAMERA_INDEX}")
        return

    idx = 0
    recording = False
    frames = []
    started = 0.0

    try:
        while idx < len(todo):
            cid, cat, text = todo[idx]
            ok, frame = cap.read()
            if not ok:
                break
            if recording:
                frames.append(frame.copy())

            disp = frame.copy()
            cv2.rectangle(disp, (0, 0), (640, 96), (0, 0, 0), -1)
            cv2.putText(disp, f"[{cid}] {cat}  ({idx + 1}/{len(todo)})", (10, 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 255), 1)
            for j, line in enumerate(wrap(text)):
                cv2.putText(disp, line, (10, 48 + j * 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

            if recording:
                cv2.circle(disp, (615, 20), 9, (0, 0, 255), -1)
                cv2.putText(disp, f"REC {time.time() - started:.1f}s  {len(frames)}f",
                            (450, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)
            else:
                cv2.putText(disp, "SPACE=record  s=skip  r=redo last  q=quit",
                            (10, 470), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

            cv2.imshow("Record eval clips", disp)
            key = cv2.waitKey(1) & 0xFF

            if key == ord("q"):
                break
            elif key == ord(" "):
                if not recording:
                    recording, frames, started = True, [], time.time()
                    print(f"[{cid}] recording...")
                else:
                    recording = False
                    if len(frames) < 10:
                        print(f"[{cid}] too short ({len(frames)} frames), discarded")
                        continue
                    path = os.path.join(CLIP_DIR, f"{cid}.mp4")
                    h, w = frames[0].shape[:2]
                    writer = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"),
                                             30.0, (w, h))
                    for fr in frames:
                        writer.write(fr)
                    writer.release()
                    records[cid] = {
                        "id": cid,
                        "file": os.path.relpath(path, EVAL_DIR),
                        "category": cat,
                        "reference": text,
                        "frames": len(frames),
                        "seconds": round(len(frames) / 30.0, 2),
                    }
                    save_manifest(records)
                    print(f"[{cid}] saved {len(frames)} frames -> {path}")
                    idx += 1
            elif key == ord("s") and not recording:
                print(f"[{cid}] skipped")
                idx += 1
            elif key == ord("r") and not recording and idx > 0:
                idx -= 1
                print(f"redo [{todo[idx][0]}]")
    finally:
        cap.release()
        cv2.destroyAllWindows()
        save_manifest(records)

    print(f"\n{len(records)}/{len(prompts)} recorded -> {MANIFEST}")
    if len(records) < len(prompts):
        print("run again to continue where you left off")


if __name__ == "__main__":
    main()
