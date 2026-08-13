"""Deliberately perturb the crop boxes, to measure how much landmark error the frozen
stack absorbs. OFF by default; this is a measurement tool, not a pipeline feature.

WHY THIS EXISTS
---------------
Replacing MediaPipe with a faster/better pose estimator (POSE_SWAP_SCOPING.md) means
feeding differently-framed crops to `dinov2hand.pth`, `dinov2face.pth` and SHuBERT -- all
frozen, all trained on MediaPipe-derived crops. Whether that is survivable is THE question
that decides the port, and it can be answered without porting anything: displace the crop
boxes by a known amount and measure what the translation does.

The precedent that motivates it: the ONNX perception port matched MediaPipe landmarks to
within 1-2px and still lost content words ("31 workers" -> "Wild workers"), and raising its
hand-detection rate from 71% to 84% reproduced the errors verbatim -- so the damage came
from landmark PRECISION shifting the crop box, not from missed detections.

WHAT IS PERTURBED
-----------------
Hands: the square box built from the 21 hand landmarks, before it is clamped to the frame.
Face: the three sub-boxes (left eye, right eye, mouth) that get composited onto the grey
background -- independently, since a different estimator would displace each separately.

Scale is applied about the box centre and then the box is translated, so a jitter of k px
means "the landmarks moved k px" rather than "the crop grew".

CALIBRATION OF THE UNITS
------------------------
At 640x480 a hand crop is typically 60-120px on a side, so 8px is a ~10% framing error --
roughly the disagreement to expect between two different estimators. 1-2px is the
disagreement the ONNX port had with MediaPipe.

DETERMINISM IS LOAD-BEARING
---------------------------
The offset is a pure function of (seed, stream, frame index), not a stateful RNG. The eval
is crash-resumable and clips are processed by several workers, so a sequential RNG would
give a resumed run different perturbations from the one it resumed -- silently mixing two
conditions into one score. Same reason the runs record their config.
"""
import os

import numpy as np

# Maximum displacement in pixels, applied independently to x and y, uniform in [-k, +k].
CROP_JITTER_PX = float(os.environ.get("CROP_JITTER_PX", "0"))
# Multiplicative size jitter, uniform in [1-s, 1+s]. 0.1 = +/-10% box size.
CROP_JITTER_SCALE = float(os.environ.get("CROP_JITTER_SCALE", "0"))
# Change to get a different perturbation at the same magnitude -- i.e. to check that a
# result is about the magnitude and not about one unlucky draw.
CROP_JITTER_SEED = int(os.environ.get("CROP_JITTER_SEED", "0"))

# "perframe" (default): a fresh displacement every frame -- shimmer.
# "static": ONE displacement per stream, held for every frame of every clip -- a constant
#           reframing.
#
# WHY BOTH EXIST. The per-frame sweep found -2.95 BLEU at 8px, but that is not the error a
# real estimator makes. A different model frames crops with a CONSISTENT offset and scale
# bias plus a little noise; per-frame jitter is pure noise. SHuBERT is a temporal encoder,
# so frame-to-frame inconsistency may cost far more than a uniform shift the DINOv2 crop
# encoders could simply absorb. If static 8px is benign while per-frame 8px is not, then
# the per-frame sweep overstates the cost of swapping extractors and the conclusion flips.
#
# NOTE ON EXPERIMENT DESIGN: static mode draws ONCE per stream, so a single run is a sample
# of size one -- an unlucky draw could show either answer. Run it at two or more seeds
# before believing it.
CROP_JITTER_MODE = os.environ.get("CROP_JITTER_MODE", "perframe").lower()

ALL_STREAMS = ("left_hand", "right_hand", "left_eye", "right_eye", "mouth")
_GROUPS = {
    "all": ALL_STREAMS,
    "hands": ("left_hand", "right_hand"),
    "face": ("left_eye", "right_eye", "mouth"),
}

# Which streams to perturb; the rest are left exactly as MediaPipe framed them.
# "all" (default), a group name ("hands" / "face"), or a comma-separated list of stream
# names ("left_hand" alone, say).
#
# WHY. Two static seeds could not say whether the face or the hand crops carry the
# sensitivity: seed 1 was healthier than seed 0 on every metric, but it displaced its face
# AND its left hand less, so the two explanations were confounded. More seeds cannot
# separate them -- perturbing one stream at a time can.
#
# THE DECOMPOSITION PROPERTY, which is why this is worth doing rather than just running
# more seeds: `_offsets` is a pure function of (seed, stream, index), so removing a stream
# from this set does NOT change what the remaining streams draw. hands-only at seed 0 gives
# the left hand the same 9.97px it got in the full static seed-0 run. So hands-only and
# face-only are the two HALVES of that exact run, and their deltas can be read against it.
def _parse_streams(spec):
    names = []
    for part in spec.replace(" ", "").split(","):
        if not part:
            continue
        if part in _GROUPS:
            names.extend(_GROUPS[part])
        elif part in ALL_STREAMS:
            names.append(part)
        else:
            # Loud, not silent. A typo here would perturb nothing and produce a clean run
            # wearing a jittered run's tag -- the exact silent-wrongness this tool exists
            # to avoid, and it would look like "the swap is free".
            raise SystemExit(
                f"CROP_JITTER_STREAMS: unknown stream {part!r}. "
                f"Valid: {', '.join(ALL_STREAMS)} or groups {', '.join(_GROUPS)}.")
    if not names:
        raise SystemExit("CROP_JITTER_STREAMS is empty -- that is a clean run, not a "
                         "perturbed one. Leave it unset or set CROP_JITTER_PX=0.")
    return frozenset(names)


CROP_JITTER_STREAMS = _parse_streams(os.environ.get("CROP_JITTER_STREAMS", "all").lower())


def enabled():
    return CROP_JITTER_PX > 0 or CROP_JITTER_SCALE > 0


def _offsets(stream, index):
    """Deterministic (dx, dy, scale) for one box. Pure function of its arguments."""
    if stream not in CROP_JITTER_STREAMS:
        return 0.0, 0.0, 1.0
    if CROP_JITTER_MODE == "static":
        # Drop the frame index: the same displacement for every frame of every clip.
        index = 0
    # A cheap integer mix, then a fresh RandomState. Not cryptographic -- it only has to
    # decorrelate neighbouring frames and the two hands from each other.
    h = (CROP_JITTER_SEED * 2654435761
         + index * 40503
         + sum(ord(c) for c in stream) * 1000003) & 0xFFFFFFFF
    rng = np.random.RandomState(h)
    dx = rng.uniform(-CROP_JITTER_PX, CROP_JITTER_PX) if CROP_JITTER_PX else 0.0
    dy = rng.uniform(-CROP_JITTER_PX, CROP_JITTER_PX) if CROP_JITTER_PX else 0.0
    scale = (1.0 + rng.uniform(-CROP_JITTER_SCALE, CROP_JITTER_SCALE)
             if CROP_JITTER_SCALE else 1.0)
    return dx, dy, scale


def jitter_xywh(box, stream, index):
    """Perturb an (x, y, w, h) box. Returns it unchanged when jitter is off."""
    if not enabled():
        return box
    x, y, w, h = box
    dx, dy, scale = _offsets(stream, index)
    cx, cy = x + w / 2.0, y + h / 2.0
    w, h = w * scale, h * scale
    cx, cy = cx + dx, cy + dy
    return int(round(cx - w / 2.0)), int(round(cy - h / 2.0)), int(round(w)), int(round(h))


def jitter_corners(box, stream, index, image_shape=None):
    """Perturb an (x1, y1, x2, y2) box, keeping it inside image_shape if given."""
    if not enabled():
        return box
    x1, y1, x2, y2 = box
    dx, dy, scale = _offsets(stream, index)
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    w, h = (x2 - x1) * scale, (y2 - y1) * scale
    cx, cy = cx + dx, cy + dy
    nx1, ny1 = int(round(cx - w / 2.0)), int(round(cy - h / 2.0))
    nx2, ny2 = int(round(cx + w / 2.0)), int(round(cy + h / 2.0))
    if image_shape is not None:
        ih, iw = image_shape[0], image_shape[1]
        # Clamp, and keep at least one pixel -- a collapsed box would crop an empty array
        # and turn a measurement of framing error into a measurement of blank inputs.
        nx1 = max(0, min(nx1, iw - 1))
        ny1 = max(0, min(ny1, ih - 1))
        nx2 = max(nx1 + 1, min(nx2, iw))
        ny2 = max(ny1 + 1, min(ny2, ih))
    return nx1, ny1, nx2, ny2


def streams_spec():
    """Canonical, order-stable name for the selected set -- for the run record, where it
    has to compare equal across runs and across resumes."""
    return ",".join(s for s in ALL_STREAMS if s in CROP_JITTER_STREAMS)


def describe():
    if not enabled():
        return "off"
    return (f"px={CROP_JITTER_PX} scale={CROP_JITTER_SCALE} seed={CROP_JITTER_SEED} "
            f"mode={CROP_JITTER_MODE} streams={streams_spec()}")


def realized_offsets(streams=ALL_STREAMS):
    """The actual displacement each stream gets. Only meaningful in static mode, where it
    is the whole experimental condition and therefore belongs in the write-up. Unselected
    streams show as (0, 0, 1.0), which is what they actually get."""
    return {s: tuple(round(v, 2) for v in _offsets(s, 0)) for s in streams}
