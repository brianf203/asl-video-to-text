"""Build an OpenASL benchmark subset for evaluation.

Why: our own eval set is signed by a beginner, so its scores measure signer and model
together and cannot say how good the translator actually is. OpenASL is native ASL with
English references verified by professional ASL interpreters, and SHuBERT reports results
on it -- so it is both a clean control and a check that our pipeline is faithful.

OpenASL is distributed as annotations only (YouTube IDs + timestamps + text); the video
has to be fetched from YouTube, which is what this script does. Videos are ranked by how
many test sentences they contain so that a handful of downloads yields many clips.

    python3 build_openasl_subset.py --videos 6
    EVAL_DIR=eval_set_openasl python3 run_eval.py --tag openasl

Licence: OpenASL is CC BY-NC-ND 4.0 (non-commercial research). Clips stay local and
gitignored; nothing is redistributed.
"""
import argparse
import csv
import json
import os
import subprocess
import sys
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, "eval_set_openasl")
CLIP_DIR = os.path.join(OUT_DIR, "clips")
CACHE_DIR = os.path.join(OUT_DIR, "_video_cache")
TSV_PATH = os.path.join(OUT_DIR, "openasl-v1.0.tsv")
TSV_URL = ("https://raw.githubusercontent.com/chevalierNoir/OpenASL/main/"
           "data/openasl-v1.0.tsv")

YTDLP = os.path.join(HERE, "shubert_venv", "bin", "yt-dlp")
# Cap at 480p so resolution is not a confounder against the other eval sets. In practice
# YouTube serves these as format 18 (640x360) and every clip built so far is 640x360, so
# treat 360p as the actual, consistent resolution of this set.
FORMAT = "bv*[height<=480]+ba/b[height<=480]/worst"

# YouTube 403s the media fetch for the default player client (android_vr returns URLs that
# are then refused; web/tv report a spurious "DRM protected"; ios/mweb offer no matching
# format). android still serves format 18 fine. Checked 2026-08-10 on the current yt-dlp
# release (2026.07.04) — this is a YouTube-side change, not a stale yt-dlp. If downloads
# start 403ing again, re-test the clients before assuming an upgrade is the fix.
EXTRACTOR_ARGS = "youtube:player_client=android"


def hhmmss_to_seconds(t):
    h, m, s = t.split(":")
    return int(h) * 3600 + int(m) * 60 + float(s)


def load_rows(split):
    if not os.path.exists(TSV_PATH):
        os.makedirs(OUT_DIR, exist_ok=True)
        print(f"downloading annotations -> {TSV_PATH}")
        urllib.request.urlretrieve(TSV_URL, TSV_PATH)
    rows = []
    with open(TSV_PATH, newline="") as f:
        for r in csv.DictReader(f, delimiter="\t"):
            if r["split"] == split:
                rows.append(r)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--videos", type=int, default=6,
                    help="how many source videos to download")
    ap.add_argument("--split", default="test")
    ap.add_argument("--max-clips", type=int, default=60)
    ap.add_argument("--keep-cache", action="store_true",
                    help="keep the downloaded source videos")
    args = ap.parse_args()

    os.makedirs(CLIP_DIR, exist_ok=True)
    os.makedirs(CACHE_DIR, exist_ok=True)

    rows = load_rows(args.split)
    by_video = {}
    for r in rows:
        by_video.setdefault(r["yid"], []).append(r)
    ranked = sorted(by_video.items(), key=lambda kv: -len(kv[1]))
    print(f"{len(rows)} {args.split} sentences across {len(by_video)} videos")

    records = []
    used_videos = 0
    for yid, items in ranked:
        if used_videos >= args.videos or len(records) >= args.max_clips:
            break
        src = os.path.join(CACHE_DIR, f"{yid}.mp4")
        if not os.path.exists(src):
            print(f"downloading {yid} ({len(items)} clips) ...", flush=True)
            rc = subprocess.run(
                [YTDLP, "-f", FORMAT, "--merge-output-format", "mp4",
                 "--extractor-args", EXTRACTOR_ARGS,
                 "-o", src, "--no-playlist", "--quiet", "--no-warnings",
                 f"https://www.youtube.com/watch?v={yid}"],
            ).returncode
            if rc != 0 or not os.path.exists(src):
                print(f"  SKIP {yid}: download failed (video may be gone)")
                continue
        used_videos += 1

        for i, r in enumerate(sorted(items, key=lambda x: x["start"]), start=1):
            if len(records) >= args.max_clips:
                break
            start = hhmmss_to_seconds(r["start"])
            end = hhmmss_to_seconds(r["end"])
            if end - start < 0.8:
                continue
            cid = f"{yid}_{i:02d}"
            dst = os.path.join(CLIP_DIR, f"{cid}.mp4")
            cut = subprocess.run(
                ["ffmpeg", "-y", "-loglevel", "error", "-ss", str(start),
                 "-to", str(end), "-i", src, "-an", dst],
            ).returncode
            if cut != 0 or not os.path.exists(dst):
                print(f"  cut failed for {cid}")
                continue
            records.append({
                "id": cid,
                "file": os.path.relpath(dst, OUT_DIR),
                "category": "openasl",
                "reference": r["raw-text"].strip(),
                "seconds": round(end - start, 2),
                "source": f"https://www.youtube.com/watch?v={yid}",
            })
        print(f"  {yid}: {len(records)} clips so far")

        if not args.keep_cache:
            try:
                os.remove(src)
            except OSError:
                pass

    manifest = os.path.join(OUT_DIR, "manifest.jsonl")
    with open(manifest, "w") as f:
        for rec in records:
            f.write(json.dumps(rec) + "\n")

    print(f"\n{len(records)} clips from {used_videos} videos -> {manifest}")
    if not records:
        print("nothing built -- check network / YouTube availability")
        return 1
    print("score with:")
    print(f"  EVAL_DIR={os.path.basename(OUT_DIR)} python3 run_eval.py --tag openasl")
    return 0


if __name__ == "__main__":
    sys.exit(main())
