"""Live ASL auto-segmentation with a persistent in-process translation worker.

Supersedes auto_segment_v3.py (subprocess per clip) and
auto_segment_shubert_threaded.py (in-process worker, frame-count thresholds).

v3 shelled out to `python3 run_shubert.py` for every clip, so each translation
paid the full cold-start cost — the ~13s ByT5 checkpoint load, both DINOv2 loads
and the torch import — on top of the actual work. Here the models are loaded once
at startup (SHuBERTProcessor.warmup) and reused by a background worker thread, so
per-clip latency drops to the compute alone. The camera loop never blocks on
translation; clips are handed off through a queue.

Measured on my_please.mp4: ~57s cold -> ~41s warm per clip.
"""
import json
import os
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
import fit_thresholds
from features import SHuBERTProcessor
from hand_trigger import HAND_NEEDED, HAND_WINDOW, HandPresenceTrigger
from motion_gate import MotionGate, PROFILE_PATH, load_profile
from streaming_perception import StreamingPerception, stride_from_env

# Scale the start/stop thresholds to the room's measured noise floor instead of using the
# absolute constants below. Those constants are raw pixel-difference units fitted to one
# camera, room and lighting setup, so they do not transfer: a dimmer room or a more distant
# signer never reaches the start threshold, a noisier one trips it constantly. Simulated
# over calib.jsonl across a 32x range of noise levels (simulate_segmentation.py --scale):
# fixed thresholds recorded NOTHING at 0.25-0.5x and produced false starts at 2x and above,
# while adaptive held 97.9% coverage with 0 false starts at every scale.
# Set ADAPTIVE_MOTION=0 to restore the old fixed-threshold behaviour.
ADAPTIVE_MOTION = os.environ.get("ADAPTIVE_MOTION", "1") not in ("0", "false", "False")

# Require hands to be visible before a clip starts. The pixel metric cannot tell a signer
# from anything else that moves pixels; hand presence is semantic and separates far better
# (measured on calib.mp4: 7.7% of still frames vs 79-84% of signing frames, and 0.0% across
# the 452-frame post-signing still span). Acts as a pure VETO on top of the existing pixel
# threshold, which is NOT loosened -- see hand_trigger.py for why lowering it was rejected.
# On clean calibration data the hybrid is identical to pixel-only (4 clips, 97.9% coverage,
# 0 false starts) and additionally rejects a 20x background-motion burst that makes
# pixel-only false-start. Set LANDMARK_TRIGGER=0 to disable.
LANDMARK_TRIGGER = os.environ.get("LANDMARK_TRIGGER", "1") not in ("0", "false", "False")

# Promote hand presence from VETO to TRIGGER: the detector runs on every idle frame and a
# start needs only its confirmation, with no pixel threshold in the decision at all.
#
# "auto" (the default) turns this on exactly when calibration reports a hand-gated fit --
# i.e. when it measured that signing does not clear ordinary movement on the pixel metric,
# which is what happens as soon as the signer is far enough away that their hands are a
# small share of the frame. Measured 2026-08-16, standing: signing peaked at 0.497 smoothed
# against 0.514 for deliberate non-signing movement and 0.227 for stillness, and hand
# presence over the same recording was 0 detections in 184 non-signing checks against 32 in
# 101 signing checks. Pixels had no usable boundary; hands had a clean one.
#
# The pre-gated hybrid does not work in that regime and the reason is worth keeping: a
# pixel threshold safe against stillness is cleared by only one 0.53s burst in 9s of
# signing, so the detector is fed for 0.53s of every 9 and its 5-verdict window never
# fills. Simulated over last_calibration.jsonl, the pre-gated path fired 0 starts on 9s of
# real signing.
#
# COST, and it is real: ~102ms/frame of one core, continuously, WHILE IDLE. That is the
# cost hand_trigger.py's pre-gate exists to avoid, and it competes with the CPU-bound
# perception of any clip still translating. It buys the only working trigger at this
# distance, so it is on only where the cheap path has been measured to fail. During
# RECORDING the detector goes quiet again -- the stop side is still the pixel threshold.
HAND_TRIGGER_PRIMARY = os.environ.get("HAND_TRIGGER_PRIMARY", "auto").lower()

# Run MediaPipe on each frame as it is captured rather than after the cut. Perception is
# ~4.5x slower than realtime so it never finishes early, but it absorbs the whole recording
# duration -- ~14s off a ~55s clip. Landmarks are identical, not approximated: a fresh
# detector sees the same frames in the same order either way. Set STREAM_PERCEPTION=0 to
# fall back to writing an mp4 and processing it after the cut.
STREAM_PERCEPTION = os.environ.get("STREAM_PERCEPTION", "1") not in ("0", "false", "False")

# Second stage: also crop and run DINOv2 as landmarks land, pipelined behind perception.
# DINOv2 is GPU work while MediaPipe is CPU-bound, so it hides almost entirely in the
# shadow of the perception backlog -- it drops to 0.0s on the critical path. Measured over
# 5 clips against the non-streaming path, decoded text identical on 5/5:
#   003 23.6->11.0s (2.14x), 004 49.4->22.2s (2.22x), 005 45.0->19.8s (2.27x),
#   006 51.1->22.8s (2.24x), 001 39.3->18.7s (2.10x).
# Set STREAM_DINOV2=0 to disable.
STREAM_DINOV2 = os.environ.get("STREAM_DINOV2", "1") not in ("0", "false", "False")

# How many clips may be streaming at once. Each live StreamingPerception holds a MediaPipe
# detector, its retained stride-2 RGB frames (~193MB for a 419-frame clip) and its own
# DINOv2 thread -- and a signer does not wait for the translation before starting the next
# sentence. In the first real signing session six streams piled up and CUDA OOM'd 5 of 9
# clips; every prior validation had fed clips sequentially, so it never saw more than one.
# Past this cap a clip still records, it just buffers frames and does its perception after
# the cut (slower for that clip, but bounded). gpu_serial.py bounds the device side.
MAX_LIVE_STREAMS = int(os.environ.get("MAX_LIVE_STREAMS", "2"))
_live_lock = threading.Lock()
_live_streams = [0]


def _take_stream_slot():
    """Reserve one of the MAX_LIVE_STREAMS slots. False if they are all taken."""
    with _live_lock:
        if _live_streams[0] >= MAX_LIVE_STREAMS:
            return False
        _live_streams[0] += 1
        return True


def _release_stream_slot():
    with _live_lock:
        _live_streams[0] -= 1


# The stream cap above bounds how many clips do perception CONCURRENTLY. It does not bound
# what actually grows: the translation QUEUE. Every queued clip keeps its stride-2 RGB
# frames alive until the worker reaches it, and a deferred clip keeps all of them with
# perception not even started. At 640x480x3 that is 0.88 MB per retained frame.
#
# The 2026-08-11 live session OOM'd 4 of 8 clips with the stream cap working exactly as
# designed: clip_1 (232 frames) failed while clip_2 was streaming and clips 3 and 4 were
# both buffering deferred -- roughly 650 retained frames, ~575MB, plus ByT5 on the GPU.
# The FIRST clip failing is the proof that concurrency was never the binding constraint.
#
# So: bound total retained frames across queue + deferred + live streams, and refuse to
# START a new clip past the budget. Refusing is deliberate. The camera thread must never
# block (that freezes the preview and drops frames), and dropping one sentence with a
# visible on-screen warning beats OOMing the clips already in flight.
MAX_RETAINED_FRAMES = int(os.environ.get("MAX_RETAINED_FRAMES", "450"))
# Second, independent guard -- an EMERGENCY BACKSTOP, deliberately set low. Retained frames
# are only part of what fills memory (models, embeddings, MediaPipe native buffers), so also
# check what actually binds. Unified memory means GPU allocations show up here too -- see
# the meminfo note in live_worker_probe.py.
#
# Do not raise this much. Healthy probe runs with zero failures bottomed out at 217-752MB
# available (the minimum lands during ByT5 generate), so a floor anywhere near 1GB refuses
# clips during perfectly normal signing -- measured: at 1100MB the machine stopped starting
# clips and never resumed. The frame budget above is the principled limit; this only
# catches a genuine emergency the frame count did not predict.
MIN_AVAILABLE_MB = int(os.environ.get("MIN_AVAILABLE_MB", "350"))
_retained_lock = threading.Lock()
_retained_frames = [0]


def _retain_frame():
    with _retained_lock:
        _retained_frames[0] += 1


def _release_frames(n):
    with _retained_lock:
        _retained_frames[0] = max(0, _retained_frames[0] - n)


def _retained_count():
    with _retained_lock:
        return _retained_frames[0]


def _available_mb():
    """MemAvailable, the metric that binds on this unified-memory box."""
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except OSError:
        pass
    return None


def _capture_budget():
    """(ok, reason) -- whether there is room to start another clip."""
    retained = _retained_count()
    if retained >= MAX_RETAINED_FRAMES:
        return False, f"backlog {retained} frames"
    avail = _available_mb()
    if avail is not None and avail < MIN_AVAILABLE_MB:
        return False, f"{avail}MB free"
    return True, ""

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
os.makedirs(config['temp_dir'], exist_ok=True)

# Segmentation thresholds, CALIBRATED 2026-08-10 against a labelled recording of this
# signer and camera (calibrate_motion.py -> calib.jsonl; 85s of still/sign/fingerspell).
#
# The previous values (start 3.0, stop 1.5, still 1.5s) were inherited from v3 and had
# never been measured. Replaying the state machine over that recording, they recorded
# **11% of the time the user was actually signing**, with a false start and a mid-sentence
# cut. Actual signing scores a median of 0.45 and fingerspelling 0.45, so a stop threshold
# of 1.5 classified ~92% of fingerspelling frames as motionless -- which is exactly the
# reported "it cuts off in the middle of fingerspelling", while whole-frame noise
# (auto-exposure hunting) crossing 3.0 is the reported "it thinks I'm starting a sign".
# These values give 95.8% coverage, 0 false starts and 0 fragments on the same recording.
#
# Two structural changes go with them, both necessary -- thresholds alone are not enough:
#   * the score is SMOOTHED over MOTION_SMOOTHING_FRAMES. No instantaneous metric can
#     separate signing from stillness (measured: every candidate overlapped, because
#     signing contains holds and stillness contains twitches). The separation only exists
#     over time.
#   * a start needs START_DEBOUNCE_FRAMES sustained, and the clip is then back-dated to
#     the real motion onset from the pre-roll ring buffer. Without the back-dating the
#     debounce plus the smoothing lag would eat ~2s off the front of every sentence --
#     trading a truncated tail for a truncated head.
#
# NOTE these are specific to this camera, room and lighting. Re-run calibrate_motion.py
# after any of those change; an adaptive noise floor would be the principled fix.
MOTION_START_THRESHOLD = 1.74
MOTION_STOP_THRESHOLD = 0.29
STILL_DURATION_SECONDS = 2.5
MIN_CLIP_DURATION_SECONDS = 1.0
MOTION_SMOOTHING_FRAMES = 15    # 0.5s trailing mean
START_DEBOUNCE_FRAMES = 3       # sustained frames above start threshold before recording

# Pre-roll: how far back the ring buffer can reach when back-dating a clip to motion onset.
PRE_ROLL_MAX_SECONDS = 2.5
# Mirror of TAIL_PAD_SECONDS at the front, so the first handshape is not clipped.
LEAD_PAD_SECONDS = 0.25

# The STILL_DURATION_SECONDS of stillness that *triggers* the cut also gets recorded,
# so every clip used to carry ~45 dead frames at 30fps. Latency is ~0.41s/frame end to
# end (measured across 105/87/149/181-frame clips), so that tail cost ~18s per clip and
# gave ByT5 nothing to translate — a tail-only clip once produced a fluent hallucination.
# Trim back to the last motion, keeping a short pad so a final handshape hold (which
# carries meaning in ASL) isn't clipped off.
TAIL_PAD_SECONDS = 0.25

# Push-to-record gets the same trim, for the same two reasons. A key press cannot be
# simultaneous with the sign: the hand has to leave the keyboard before the first sign and
# come back after the last one, so both ends of a manual clip hold dead air the signer did
# not intend to record.
#   Quality -- and on the live path this is the ONLY reason. Near-still frames are exactly
#   what makes ByT5 invent a fluent sentence rather than return nothing (a 49-frame
#   near-still clip once decoded as "I am here to show you how to do the heart artery";
#   live on 2026-08-21 a clip the signer deliberately held still through decoded as "I'm
#   fluent in Spain").
#   NOT latency, though it was justified that way and the claim stood until it was
#   measured. The original figure -- ~0.85s of post-cut per second of captured video on a
#   ~3.4s floor, from probe_w2_a / probe_w2_control -- comes from the EVAL path, where
#   perception runs after the cut over exactly the kept frames. On the LIVE streamed path
#   StreamingPerception.finish() drains every queue and joins all workers BEFORE it slices
#   [start:end], so trimmed frames have already been through MediaPipe and DINOv2 by the
#   time they are dropped (its docstring says so for the head, and the same holds for the
#   tail). Post-cut = drain + ByT5, and the drain is set by the perception BACKLOG at the
#   cut, which trimming the FRONT cannot reduce -- those are the oldest frames and the
#   first processed. What is left is a shorter SHuBERT/ByT5 input, too small to measure
#   against ByT5 generate time, which tracks OUTPUT length. See the 2026-08-21 section of
#   PROJECT_CONTEXT.md.
# MANUAL_TRIM=0 restores the previous behaviour of keeping every frame between the presses.
MANUAL_TRIM = os.environ.get("MANUAL_TRIM", "1") not in ("0", "false", "False")
# Hard bound on what either end may lose. The trim runs on a threshold fitted to the room's
# noise floor, and if that estimate is wrong this is the difference between a clip that is
# slightly short and a clip with no signing left in it. Nothing is trimmed beyond this.
#
# It doubles as the window the onset search may look in, which has a consequence worth
# knowing: a head with MORE than this much dead air is not trimmed at ALL, rather than
# trimmed back to the bound. That is deliberate. Absence of motion is not evidence of
# absence of signing on a whole-frame pixel mean -- the calibration work measured most
# fingerspelling frames reading as "still" -- so a long quiet head is as likely to be a slow
# sentence as a pause, and only the tail has a positive motion detection (last_motion_time)
# to anchor against. Press SPACE close to when you start signing and this never applies.
MANUAL_TRIM_MAX_SECONDS = float(os.environ.get("MANUAL_TRIM_MAX_SECONDS", "2.5"))
# A quiet stretch this long separates "settling into position" from signing. Same 0.3s the
# pre-roll onset search uses, but expressed in TIME rather than frames: these frames are
# strided (~15fps) while the pre-roll ring is not (30fps).
HEAD_QUIET_SECONDS = 0.3

# A clip is normally ended by STILL_DURATION_SECONDS of stillness -- so a signer who never
# pauses is never cut. The 2026-08-11 live session produced a 542-frame / 35.8s clip that
# way, which cost 477MB of retained frames on its own and decoded to a degenerate loop
# ("...IN MY LEADERSHIP" x8, 26.2s in ByT5). Long clips hurt three ways at once: memory,
# latency (ByT5 scales with encoder length) and quality. So cap the length and force a cut.
#
# A forced cut is NOT the same as a stillness cut: motion is still going, so there is no
# still tail to trim and every frame is signing. Recording restarts immediately at the cut
# point, with no pre-roll back-dating -- the previous clip already covers everything up to
# now, so seeding from the ring buffer would duplicate frames across the boundary.
# The cut lands mid-sentence, which will read as a broken sentence; that is the intended
# trade against losing the clip entirely to OOM.
MAX_CLIP_SECONDS = float(os.environ.get("MAX_CLIP_SECONDS", "15"))

# How long to wait on quit for the worker to finish what it is translating. Generous
# because abandoning it destroys that clip AND tears the detector's thread pool down while
# it is still in use; a deferred clip at the 15s cap can need a couple of minutes on this
# box. Ctrl-C during the wait abandons deliberately.
QUIT_DRAIN_SECONDS = float(os.environ.get("QUIT_DRAIN_SECONDS", "180"))

# A clip is only worth translating if enough of what it keeps is actually moving. Measured
# on calib.jsonl (smoothed score vs a fitted stop threshold of 0.241): SIGN 87.4% of frames
# above it, FINGERSPELL 93.9%, versus 47.0% / 62.6% for the still phases and 34.5% for the
# settled-still span. Anything from 0.70 down separates those, but the default sits well
# below the signing side rather than at the midpoint: this exists to discard the
# essentially-empty case, and dropping a real sentence costs the signer far more than
# translating a spurious one. Raise it if false starts keep reaching the worker.
MIN_MOTION_FRACTION = float(os.environ.get("MIN_MOTION_FRACTION", "0.5"))
# The same test, used in manual mode to WARN rather than to reject (a key press is not a
# false start, but it is not evidence of signing either). Its own knob so the warning can be
# made quieter or louder without touching what the automatic path REJECTS. See the comment
# at the use site for the recording that fixes the default.
MANUAL_EMPTY_FRACTION = float(os.environ.get("MANUAL_EMPTY_FRACTION",
                                             str(MIN_MOTION_FRACTION)))

# Push-to-record: SPACE starts a clip, SPACE ends it. No motion detection at all.
#
# This is the DEFAULT because segmentation is not what the project is trying to solve. A
# global pixel-difference mean cannot reliably separate signing from ordinary movement
# (measured: the two overlap at 1.0-1.05x separation), so automatic segmentation costs a
# calibration ritual, a threshold that varies run to run, and a class of failure -- false
# starts, missed starts, cuts mid-fingerspelling -- that contaminates every latency and
# quality measurement taken through it. A key press has none of that.
#
# It also REMOVES LATENCY: automatic mode waits STILL_DURATION_SECONDS (2.5s) of stillness
# before it will cut, and every one of those seconds is perceived latency on every
# sentence. Pressing space ends the clip immediately.
#
# RECORD_MODE=motion restores the automatic path (calibration, thresholds, hand veto).
MANUAL_RECORD = os.environ.get("RECORD_MODE", "manual").lower() != "motion"

# Calibrate at every launch, in this window, before signing starts. The whole point is
# that anyone can sit down in any room and have the machine fit itself to THEM: their
# lighting, their camera distance, how big they sign. A saved profile is still written and
# is reused when this is off (STARTUP_CALIBRATION=0), which is also how an automated probe
# runs without a human to prompt.
STARTUP_CALIBRATION = os.environ.get("STARTUP_CALIBRATION", "1") not in ("0", "false",
                                                                        "False")

# Kept SHORT deliberately: this runs before every session, and it is hidden inside the
# ~20s the worker spends loading models, so it costs nothing as long as it stays under
# that. The long-form version in calibrate_motion.py (7 phases, ~100s) exists for
# analysis -- more phases, saved footage, per-metric comparison -- not for daily use.
#
# Three phases is the minimum that measures the real decision: what the room does when
# nobody signs (SIT STILL), what the SIGNER does when they are not signing (MOVE -- the
# phase that matters, see fit_thresholds.py), and what signing looks like.
CALIB_PHASES = [
    ("STAND STILL", "Stand still and look at the camera", 6.0),
    ("MOVE (NO SIGNING)", "Move but DO NOT sign - shift around, scratch your face", 7.0),
    ("SIGN", "Sign anything - a sentence, your name, fingerspelling", 9.0),
]
# Instruction shown on a static screen before each phase, so reading it does not land in
# the measured window.
CALIB_LEAD_SECONDS = 2.0

CAMERA_INDEX = 0
# One window for calibration, loading and signing: reusing the title means the same OS
# window is reused rather than a new one popping up at each stage.
WINDOW_TITLE = 'Auto ASL Translation (v5 - persistent worker)'

# How much bigger than the captured frame to DISPLAY the preview, for demos where the
# screen is a few feet away.
#
# This upscales the display copy only. Capture stays at 640x480, and that is the point:
# every frame that reaches the pipeline is unchanged, so nothing downstream shifts. Raising
# cv2.CAP_PROP_FRAME_* instead would (a) invalidate the motion thresholds, which are fitted
# against the mean absolute difference of 640x480 frames, (b) make MediaPipe and the DINOv2
# crops slower on every clip, and (c) inflate every retained frame against the
# MAX_RETAINED_FRAMES budget, whose 0.88MB/frame figure assumes 640x480x3.
#
# Cost of the upscale, measured on this box at 1.6x (640x480 -> 1024x768): 1.6 ms/frame
# with OpenCV's default 6 threads, 7.3 ms pinned to one. The capture loop has ~33 ms per
# frame, and it is not what bounds latency -- perception is, on its own threads. Set
# WINDOW_SCALE=1 to get the old behaviour and spend nothing.
#
# Doing it here rather than with resizeWindow is not a stylistic choice: this OpenCV is
# built against Qt5, and a WINDOW_NORMAL resize there PADS the window instead of scaling
# the image (measured -- a 400px circle stayed 408px in a 1024-wide window).
WINDOW_SCALE = float(os.environ.get("WINDOW_SCALE", "1.6"))
# INTER_NEAREST is ~3x cheaper (0.5 vs 1.6 ms) and keeps the overlay text crisp at the cost
# of jagged edges; INTER_LINEAR reads better on the video itself. The overlays are drawn at
# 640x480 coordinates either way and are magnified with everything else.
WINDOW_INTERP = (cv2.INTER_NEAREST
                 if os.environ.get("WINDOW_INTERP", "linear").lower() == "nearest"
                 else cv2.INTER_LINEAR)


def show(frame):
    """Draw one frame into the preview window, upscaled for readability."""
    if WINDOW_SCALE != 1.0:
        h, w = frame.shape[:2]
        frame = cv2.resize(frame, (int(w * WINDOW_SCALE), int(h * WINDOW_SCALE)),
                           interpolation=WINDOW_INTERP)
    cv2.imshow(WINDOW_TITLE, frame)


def _gray_for_motion(frame):
    """The exact preprocessing the live loop uses. Shared so the two cannot drift apart:
    a threshold fitted on one blur radius and applied against another is silently wrong."""
    return cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (21, 21), 0)


def _draw_calibration(frame, prompt, banner, seconds_left, score, models_loading):
    disp = frame.copy()
    cv2.putText(disp, banner, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
    cv2.putText(disp, prompt, (10, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
    cv2.putText(disp, f"{seconds_left:.0f}", (10, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)
    if models_loading:
        cv2.putText(disp, "loading models in the background...", (10, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    if score is not None:
        cv2.putText(disp, f"motion {score:.2f}", (520, 455),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)
    return disp


# Every calibration writes its labelled frames here, overwritten each launch. It is the
# only recording that contains a labelled MOVE (NO SIGNING) phase -- no existing capture
# does -- which is exactly the class needed to answer whether a HAND-LANDMARK motion score
# separates signing from fidgeting where the pixel mean does not. Costs one small file per
# launch and saves having to stage a whole recording session to find out.
CALIB_LOG_PATH = os.environ.get("CALIB_LOG", "last_calibration.jsonl")


def run_startup_calibration(cap, hand_gate=None):
    """Prompt through the short guided calibration.

    Returns (profile, keep_going, retry_worthy). `profile` is None if the fit failed, in
    which case the caller falls back to inferring thresholds online. `keep_going` is False
    only when the user pressed q, which means quit the program -- not "carry on
    uncalibrated". `retry_worthy` says whether running it AGAIN could plausibly help: only
    when the recording holds no evidence anyone signed. A fit that failed because signing
    and ordinary movement overlap will fail again identically, and re-prompting a signer
    who did exactly what was asked is worse than useless.

    Runs in the live preview window and uses the live loop's own motion metric, so what is
    fitted is exactly what will be compared against. Deliberately placed while the
    translation models load: the wait already exists, so calibration is free.
    """
    rows = []
    prev_gray = None
    total = sum(s for _, _, s in CALIB_PHASES) + CALIB_LEAD_SECONDS * len(CALIB_PHASES)
    print(f"Calibrating to this room and signer (~{total:.0f}s) — follow the prompts.")

    for index, (label, prompt, seconds) in enumerate(CALIB_PHASES, start=1):
        banner = f"CALIBRATION {index}/{len(CALIB_PHASES)}"
        for measuring in (False, True):
            phase_end = time.time() + (seconds if measuring else CALIB_LEAD_SECONDS)
            while time.time() < phase_end:
                ret, frame = cap.read()
                if not ret:
                    print("Calibration aborted: camera read failed.")
                    return None, True, False
                gray = _gray_for_motion(frame)
                score = None
                if prev_gray is not None:
                    score = float(np.mean(cv2.absdiff(prev_gray, gray)))
                    if measuring:
                        row = {"current": score, "phase": label, "t": time.time()}
                        if hand_gate is not None:
                            # Whatever the detector last produced. It lags ~100-200ms and
                            # arrives at ~10fps, so consecutive rows repeat a result --
                            # deliberate: dropping to the detector's rate here would make
                            # the pixel and landmark series different lengths.
                            row["hands"] = hand_gate.latest()
                        rows.append(row)
                prev_gray = gray
                if hand_gate is not None:
                    # Keep the detector fed so the landmark log covers the whole session,
                    # and so its CPU cost is present during calibration exactly as it will
                    # be during signing -- the pixel score depends on the frame interval,
                    # so calibrating without this load would fit the wrong thing.
                    hand_gate.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

                show(_draw_calibration(frame,
                                       prompt if measuring else f"NEXT: {prompt}",
                                       banner if measuring else "GET READY",
                                       phase_end - time.time(), score,
                                       not models_ready.is_set()))
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Calibration cancelled.")
                    return None, False, False

    try:
        with open(CALIB_LOG_PATH, "w") as fh:
            for row in rows:
                fh.write(json.dumps(row) + "\n")
    except OSError as e:
        print(f"(could not save {CALIB_LOG_PATH}: {e})")

    start, stop, rep = fit_thresholds.fit_rows(rows, metric="current",
                                               window=MOTION_SMOOTHING_FRAMES)
    print(fit_thresholds.describe(start, stop, rep))
    if start is None:
        # Only reachable now when the recording holds no evidence of signing at all. The
        # noise-floor fallback is a genuinely bad outcome (floor x 9 produced an
        # unreachable 1.7 in the 2026-08-16 standing session and 7.06 before that), so it
        # is stated as the failure it is rather than as a graceful degradation.
        print("Calibration could not fit thresholds — falling back to the noise floor, "
              "which infers the signing side rather than measuring it.")
        return None, True, rep.get("retry_worthy", True)

    if rep.get("hand_gated"):
        print("\n*** This room/distance cannot be segmented on pixel motion alone. ***")
        print("Thresholds are fitted to STILLNESS only and ordinary movement will cross "
              "them; hand presence is what has to reject it, so LANDMARK_TRIGGER must "
              "stay on. Expect occasional false starts.")
        print("Standing closer to the camera, or framing it tighter on your upper body, "
              "restores the normal fit — signing moves a larger share of the frame.\n")

    profile = {
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "metric": "current", "smoothing_frames": MOTION_SMOOTHING_FRAMES, "fps": 30.0,
        "start_threshold": start, "stop_threshold": stop,
        # No online floor from a window this short: the floor estimator needs a sustained
        # quiet run and 6s of stillness may not contain one. The live gate simply skips
        # drift correction when this is absent, which is right -- calibration just ran.
        "floor": None,
        "separation": rep.get("separation"), "separable": rep.get("ok", False),
        # Recorded so the live gate can refuse to run this profile without the veto, and
        # so a session log says which of the two fits produced these numbers.
        "hand_gated": bool(rep.get("hand_gated")),
        "n_frames": len(rows),
        "stats": {k: rep.get(k) for k in ("neg_max", "neg_p50", "weakest_move_peak",
                                          "still_max", "stop_still_quiet_frames",
                                          "stop_signing_quiet_frames")},
    }
    try:
        with open(PROFILE_PATH, "w") as fh:
            json.dump(profile, fh, indent=2)
    except OSError as e:
        print(f"(could not save {PROFILE_PATH}: {e})")
    return profile, True, False


def wait_for_models(cap):
    """Hold the preview open until the worker has loaded, so the camera never freezes."""
    while not models_ready.is_set():
        ret, frame = cap.read()
        if not ret:
            break
        disp = frame.copy()
        cv2.putText(disp, "Loading translation models...", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 255), 2)
        show(disp)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            return False
    return True

def _begin_perception():
    """Start perception for a new clip. Returns (stream, deferred).

    Shared by the IDLE->RECORDING transition and the forced mid-sentence cut, so both
    honour the live-stream cap the same way.
    """
    if STREAM_PERCEPTION and _take_stream_slot():
        # One detector per clip -- never reused, see streaming_perception.py.
        return StreamingPerception(
            config['mediapipe_face_model_path'],
            config['mediapipe_hands_model_path'],
            embed_config=config if STREAM_DINOV2 else None,
        ), False
    if STREAM_PERCEPTION:
        # Over the cap: buffer stride-2 RGB, the same frames the stream would have
        # retained, so this fallback costs no more memory than streaming would have.
        return None, True
    return None, False


clip_queue = queue.Queue()
models_ready = threading.Event()
state_lock = threading.Lock()
# What the camera window shows about the worker. Deliberately NOT the translated text:
# translations go to the TERMINAL, one "Signer: ..." line per clip, so the video window
# stays clean enough to record. This only ever holds short status strings.
worker_status = ["Loading models..."]
# Clips the cut-time checks flagged as holding no detectable signing, keyed by label. The
# camera thread writes, the worker pops when it prints the translation. A side channel
# rather than a queue field because the legacy path queues a bare mp4 path, which has
# nowhere to carry one. Guarded by state_lock, like worker_status.
suspect_clips = {}


def flag_suspect(label, note):
    """Mark a queued clip so its translation is printed with a warning attached."""
    with state_lock:
        suspect_clips[label] = note
clip_counter = [0]


def translation_worker(processor):
    """Load models once, then translate queued clips until the sentinel arrives."""
    try:
        t0 = time.time()
        processor.warmup()
        print(f"[worker] Models loaded in {time.time() - t0:.1f}s")
        with state_lock:
            worker_status[0] = "Ready to sign."
    except Exception as e:
        print(f"[worker] FATAL: model preload failed: {e}")
        with state_lock:
            worker_status[0] = f"Model load failed: {e}"
        return
    finally:
        models_ready.set()

    while True:
        item = clip_queue.get()
        if item is None:
            break
        # Three shapes arrive here:
        #   ("stream", label, stream, head, keep, retained)  StreamingPerception to drain
        #   ("frames", label, frames, retained)        buffered frames, perception not
        #                                    started (the over-cap fallback; no mp4 round
        #                                    trip)
        #   "<path>.mp4"                     legacy, only when STREAM_PERCEPTION=0
        # `retained` is this clip's contribution to the global retained-frame budget; the
        # finally releases it, which is what lets the camera thread start a clip again.
        kind = item[0] if isinstance(item, tuple) else "path"
        label = item[1] if isinstance(item, tuple) else item
        streamed = kind == "stream"
        stream = None
        # Frames this clip is keeping alive, released in the finally once they are no
        # longer referenced -- that release is what lets the camera thread start again.
        retained = item[-1] if isinstance(item, tuple) else 0
        # Popped here, not at print time, so a clip that raises cannot leave an entry
        # behind for a later clip to inherit.
        with state_lock:
            suspect = suspect_clips.pop(label, None)
        try:
            print(f"[worker] Translating {label} ...")
            t0 = time.time()
            if streamed:
                _, _, stream, head, keep, _ = item
                # Drain here rather than on the camera thread: perception runs ~4.5x
                # slower than realtime, so at the cut there is still a backlog worth
                # seconds, and blocking the capture loop on it would freeze the preview
                # and drop frames of whatever the signer does next.
                drain_start = time.time()
                frames, landmarks, embeddings = stream.finish(keep, start=head)
                drain = time.time() - drain_start
                print(f"[worker] drained perception backlog in {drain:.1f}s "
                      f"({stream.processed_frames} frames processed, "
                      f"{stream.busy_seconds:.1f}s inside MediaPipe)")
                result = processor.process_frames(
                    frames, landmarks=landmarks,
                    mediapipe_seconds=stream.busy_seconds,
                    embeddings=embeddings,
                    embed_seconds=stream.embed_busy_seconds)
            elif kind == "frames":
                # Over the stream cap: landmarks were never started, so this pays the full
                # perception cost here. Still cheaper than the legacy path, which wrote and
                # re-read an mp4 and buffered full-rate BGR.
                result = processor.process_frames(item[2])
            else:
                result = processor.process_video(item)
            elapsed = time.time() - t0
            with state_lock:
                worker_status[0] = "Ready to sign."
            # The translation itself goes ONLY here, on its own line and prefixed, so a
            # terminal recording reads as a transcript rather than as a log.
            print(f"[worker] {label} translated in {elapsed:.1f}s")
            # The warning rides INSIDE the Signer line rather than on one of its own,
            # because demo_transcript.py shows only these lines -- a caveat printed
            # anywhere else is invisible exactly where it matters most, on a demo screen.
            marker = f"[{suspect} — likely invented] " if suspect else ""
            print(f"Signer: {marker}{result}", flush=True)
        except Exception as e:
            # Report rather than swallow — v2 discarded stderr and made failures invisible.
            print(f"[worker] Error translating {label}: {type(e).__name__}: {e}")
            # State at the moment of failure, so an OOM can be attributed instead of
            # guessed at. The 2026-08-11 adaptive session OOM'd with NO backlog, which
            # only became clear afterwards -- without these numbers the queue-growth fix
            # looked like it had failed when it simply did not apply.
            print(f"[worker] at failure: retained={_retained_count()} frames, "
                  f"available={_available_mb()}MB, live_streams={_live_streams[0]}, "
                  f"queue={clip_queue.qsize()}")
            with state_lock:
                # A failure is status, not a translation, so it does belong on screen --
                # showing "Ready to sign." after a clip died would hide it.
                worker_status[0] = f"[translation failed: {type(e).__name__}]"
        finally:
            # MediaPipe native objects must be released per clip or they exhaust the
            # Jetson's shared pool after a few clips.
            if stream is not None:
                stream.close()
                _release_stream_slot()
            # Drop the reference to the buffered frames before accounting for them, so
            # the budget reflects memory actually reclaimable rather than optimism.
            # `label` still holds the mp4 path for the legacy branch below.
            item = None
            _release_frames(retained)
            clip_queue.task_done()
            if kind == "path":
                try:
                    os.remove(label)
                except OSError:
                    pass


def find_motion_onset(ring, stop_threshold=None):
    """Walk back through the pre-roll ring to where this burst of motion began.

    The trigger fires late by construction (a debounce plus a 0.5s trailing mean), so the
    clip has to be back-dated or the front of every sentence is lost. Onset = the end of
    the last sustained quiet stretch in the buffer; anything before that belongs to the
    previous stillness, not to this sentence. This is the mirror image of the tail trim,
    which ends the clip at the last motion plus a pad.

    `ring` is a deque of (frame, timestamp, raw_score). Returns an index into it.

    `stop_threshold` must be passed when thresholds are adaptive -- defaulting to the
    module constant would back-date using this room's fitted value while the trigger used
    an adapted one, so the two halves of the decision would disagree.
    """
    if stop_threshold is None:
        stop_threshold = MOTION_STOP_THRESHOLD
    quiet_needed = max(1, int(0.3 * 30))    # 0.3s of quiet marks a real boundary
    quiet = 0
    for i in range(len(ring) - 1, -1, -1):
        if ring[i][2] <= stop_threshold:
            quiet += 1
            if quiet >= quiet_needed:
                return min(i + quiet_needed, len(ring) - 1)
        else:
            quiet = 0
    return 0    # the whole buffer is motion; keep all of it


def find_signing_onset(times, raw_scores, stop_threshold, window_seconds):
    """Index of the frame where signing begins, searching only the first `window_seconds`.

    For push-to-record, where the clip starts at a key press. The head of such a clip is
    the hand coming back from the keyboard, then usually a short settle, then the sign --
    so taking the first MOVING frame would trim nothing, because the reach is real motion.
    Onset is instead where motion RESUMES after the FIRST sustained quiet stretch, which is
    that settle. A signer who goes straight from the key into the sign leaves no quiet
    stretch to find, and this returns 0: keep everything, the mirror of find_motion_onset's
    "the whole buffer is motion" case.

    FIRST, not last, and that is not the arbitrary choice it looks like. find_motion_onset
    walks BACK to the last quiet stretch because everything after it is known to be one
    burst of motion. Here the search runs forward into unknown territory, and signing is
    full of sub-threshold holds -- the calibration work measured 87-99% of fingerspelling
    frames reading as "still" on this whole-frame metric. Taking the last quiet stretch in
    the window therefore lands INSIDE the sentence: on a clip spliced from calib.jsonl it
    cut 2.27s, about 1.3s of which was signing. The first stretch cannot do that, because
    the sign has not started yet; and every way it can be wrong keeps more, not less.

    Searching only a window at the front bounds the damage if the head holds no motion at
    all -- without it, a clip whose first real motion is the sentence itself could be
    trimmed by however long the signer hesitated.

    `raw_scores` must be RAW, not smoothed: a 0.5s trailing mean carries stillness forward
    into the sign and would move the boundary later by its own lag, eating the first
    handshape. Entries may be None (pre-roll frames carry no comparable score); those are
    skipped rather than treated as quiet.
    """
    if not times:
        return 0
    deadline = times[0] + window_seconds
    onset = 0
    quiet_start = None
    for i, (t, score) in enumerate(zip(times, raw_scores)):
        if t > deadline:
            break
        if score is None:
            continue
        if score <= stop_threshold:
            if quiet_start is None:
                quiet_start = t
        else:
            if quiet_start is not None and t - quiet_start >= HEAD_QUIET_SECONDS:
                return i
            quiet_start = None
    return onset


def manual_head_trim(times, raw_scores, gate):
    """Frames to drop from the front of a push-to-record clip, or 0 to keep them all.

    Split out of the manual-stop path so a FORCED cut can use it too. That cut lands
    mid-sentence, so its TAIL is all signing and must not be touched -- but its HEAD is the
    same key-press dead air every manual clip carries, and skipping the trim there fed a
    signer's ~7s pause straight to the model on 2026-08-21.
    """
    if not (MANUAL_TRIM and times):
        return 0
    if gate.floor is None:
        # The floor needs ~60 sustained-quiet samples (~2s of stillness,
        # motion_gate.FLOOR_MIN_SAMPLES) and manual mode runs no calibration, so pressing
        # SPACE straight after launch lands here. Say it out loud: the clip otherwise
        # reports "trimmed 0 lead-in" and reads as the onset search finding nothing to cut.
        print("Trim skipped — no room floor yet (needs ~2s of stillness after launch)")
        return 0
    onset = find_signing_onset(times, raw_scores, gate.stop_threshold,
                               MANUAL_TRIM_MAX_SECONDS)
    # Back off by the same pad the automatic path uses, so the first handshape is not
    # clipped. The head cannot run away: the onset search only looks inside the first
    # MANUAL_TRIM_MAX_SECONDS.
    lead_cutoff = times[onset] - LEAD_PAD_SECONDS
    while onset > 0 and times[onset - 1] >= lead_cutoff:
        onset -= 1
    return onset


def write_clip(frames, path, fps=30.0):
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    for f in frames:
        out.write(f)
    out.release()


def main():
    print("Starting persistent translation worker...")
    processor = SHuBERTProcessor(config)
    worker = threading.Thread(target=translation_worker, args=(processor,), daemon=True)
    worker.start()

    cap = cv2.VideoCapture(CAMERA_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print(f"ERROR: could not open camera at index {CAMERA_INDEX}")
        return

    # AUTOSIZE so the window fits whatever show() hands it -- the frame is already
    # upscaled by then, so the window follows WINDOW_SCALE without any resize call.
    cv2.namedWindow(WINDOW_TITLE, cv2.WINDOW_AUTOSIZE)

    state = "IDLE"
    recorded_frames = []
    frame_times = []
    # Smoothed motion score per retained frame, None for frames seeded from the pre-roll
    # (those carry the raw score, a different quantity, and are the lead-in by
    # construction). Used only to reject clips that contain essentially no motion.
    frame_scores = []
    # The same frames' RAW scores, kept in parallel for the manual head trim: locating a
    # boundary needs the unsmoothed signal, while deciding whether to cut needs the
    # smoothed one. Pre-roll frames DO carry a comparable value here (the raw score is a
    # difference between consecutive camera frames either way; the stride only decides
    # which of them are retained), which is why this list has no None holes where
    # frame_scores does.
    frame_raw_scores = []
    record_start_time = None
    last_motion_time = None
    prev_gray = None
    stream = None          # StreamingPerception for the clip being recorded
    deferred = False       # recording without a stream, because the cap was reached
    raw_frame_index = 0    # raw frames since RECORDING began, for applying the stride
    retained_this_clip = 0  # frames this clip has added to the global retained budget
    # Started before calibration so its MediaPipe graph loads during it, rather than
    # costing the first sentence a ~1s load at the moment it is needed.
    # The hand trigger only ever gated STARTS, so manual mode has no use for it -- and
    # skipping it frees a core, which matters: detection is ~102ms/frame on a 6-core box
    # whose bottleneck is CPU-bound MediaPipe perception.
    hand_gate = None
    if LANDMARK_TRIGGER and not MANUAL_RECORD:
        hand_gate = HandPresenceTrigger(config['mediapipe_hands_model_path'])
        hand_gate.start()
        print("Landmark trigger ON — a start also requires hands to be visible.")

    # Calibrate this signer in this room, now, while the models load. Thresholds fitted to
    # measured still / non-signing-motion / signing distributions beat inferring signing
    # from the noise floor: that inference put the start threshold at 7.06 in a noisier
    # room, above anything signing can produce. STARTUP_CALIBRATION=0 reuses the saved
    # profile instead (for unattended probes), and ADAPTIVE_MOTION=0 drops to fixed
    # constants outright.
    profile = None
    if MANUAL_RECORD:
        print("Push-to-record: press SPACE to start a clip, SPACE again to end it.")
    elif ADAPTIVE_MOTION and STARTUP_CALIBRATION:
        # One retry, and ONLY when a second run could plausibly differ -- i.e. the
        # recording holds no evidence anyone signed, the case where the signer missed the
        # prompt. When the fit failed because signing and ordinary movement overlap, a
        # second run reproduces it exactly: the 2026-08-16 standing session sat through
        # calibration twice and was then told to "sign as you normally would", which is
        # what it had been doing. Bounded at two attempts so an empty room cannot loop.
        for attempt in (1, 2):
            profile, keep_going, retry_worthy = run_startup_calibration(cap, hand_gate)
            if not keep_going:
                cap.release()
                cv2.destroyAllWindows()
                clip_queue.put(None)
                return
            if profile or attempt == 2 or not retry_worthy:
                break
            # The fit's own reason, not a fixed line: the two failures that reach here
            # need opposite corrections from the signer (sign during SIGN / hold still
            # during STILL), and the old message only ever gave one of them.
            print("\nRunning calibration once more.\n")
    elif ADAPTIVE_MOTION and not MANUAL_RECORD:
        profile = load_profile()
    gate = MotionGate(MOTION_SMOOTHING_FRAMES,
                      MOTION_START_THRESHOLD, MOTION_STOP_THRESHOLD,
                      adaptive=ADAPTIVE_MOTION, profile=profile)
    # Hand presence drives the START when the pixel metric has been MEASURED unable to (a
    # hand-gated fit), unless the env var forces the answer. It needs the detector, so
    # LANDMARK_TRIGGER=0 or manual mode rules it out whatever the profile says.
    if HAND_TRIGGER_PRIMARY in ("1", "true", "yes", "on"):
        hand_primary = hand_gate is not None
    elif HAND_TRIGGER_PRIMARY in ("0", "false", "no", "off"):
        hand_primary = False
    else:
        hand_primary = bool((profile or {}).get("hand_gated")) and hand_gate is not None
    if hand_primary:
        print("[gate] HAND-PRIMARY start: pixel motion does not separate signing at this "
              "distance, so a clip starts on hand presence alone "
              f"({HAND_NEEDED} of {HAND_WINDOW} detector verdicts). The detector now runs "
              "continuously while idle (~1 core); the pixel stop threshold still ends the "
              "clip. HAND_TRIGGER_PRIMARY=0 restores the pre-gated veto.")

    if profile and not MANUAL_RECORD:
        print(f"[gate] calibrated {profile.get('created', '?')} — "
              f"start {gate.start_threshold:.2f}, stop {gate.stop_threshold:.2f} "
              f"(separation {profile.get('separation') or 0:.2f}x)")
        if profile.get("hand_gated") and hand_gate is None:
            # These thresholds sit inside the non-signing movement distribution by
            # construction -- they are only a pre-filter, and the veto is the actual
            # decision. Running them without it is a false start on every fidget.
            print("[gate] WARNING: these thresholds were fitted for hand-gated use but "
                  "LANDMARK_TRIGGER is OFF. Ordinary movement WILL start clips. Turn the "
                  "trigger on, or use push-to-record (RECORD_MODE=manual, the default).")
        elif not profile.get("separable", True):
            print("[gate] WARNING: calibration could not cleanly separate signing from "
                  "ordinary movement — expect occasional false starts; the hand veto and "
                  "the empty-clip test cover some of them.")
    elif ADAPTIVE_MOTION and not MANUAL_RECORD:
        print("[gate] no calibration — inferring thresholds from the room's noise floor.")

    # Do not open for signing before the models exist: a clip queued now would sit unread
    # while the signer assumes it was taken.
    if not wait_for_models(cap):
        cap.release()
        cv2.destroyAllWindows()
        clip_queue.put(None)
        return
    # The pre-roll ring exists only to back-date an automatic trigger that fires late.
    # Manual mode has nothing to back-date, and the ring is not free: 2.5s of 640x480 BGR
    # is ~66MB held permanently on an 8GB box that OOMs under memory pressure.
    pre_roll = deque(maxlen=0 if MANUAL_RECORD else int(PRE_ROLL_MAX_SECONDS * 30))
    above_start = 0        # consecutive frames over the start threshold (the debounce)
    # SPACE, read at the bottom of the previous iteration. waitKey has to follow imshow to
    # render, so the press is acted on one frame (~33ms) later -- irrelevant next to a
    # translation, and it keeps all the key handling in one place.
    toggle_pressed = False
    stride = stride_from_env()
    if STREAM_PERCEPTION:
        print(f"Streaming perception ON (stride {stride}) — landmarks are extracted "
              f"during recording, not after the cut.")

    if MANUAL_RECORD:
        print("Ready. SPACE starts and ends each clip; 'q' quits.")
    else:
        print("Auto-segmentation active. Sign naturally, pause briefly between sentences.")
        print("Press 'q' to quit.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        now = time.time()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (21, 21), 0)

        raw_motion = 0.0
        if prev_gray is not None:
            raw_motion = float(np.mean(cv2.absdiff(prev_gray, gray)))
        prev_gray = gray

        # Decisions run on the smoothed score; the raw one is kept for locating motion
        # onset in the pre-roll, where the smoothing lag would blur the boundary.
        # The floor is only sampled while idle -- during a clip the score IS signing, and
        # folding that in would progressively deafen the trigger.
        was_ready = gate.ready
        had_floor = gate.floor is not None
        motion_score = gate.update(raw_motion, recording=(state == "RECORDING"))
        if ADAPTIVE_MOTION and not profile and gate.ready and not was_ready:
            # Print once, when the floor is first established: without this the chosen
            # thresholds exist only on the preview overlay and a session log cannot be
            # diagnosed afterwards. Guarded on `not profile` because with one the gate
            # becomes ready on the ARMING timer instead, and the floor may still be None
            # at that moment -- whether it is depends on whether the room happened to be
            # quiet, so without the guard this crashes only sometimes.
            print(f"[gate] noise floor {gate.floor:.3f} -> start {gate.start_threshold:.2f}, "
                  f"stop {gate.stop_threshold:.2f} "
                  f"(fixed fallback was {MOTION_START_THRESHOLD}/{MOTION_STOP_THRESHOLD})")
        elif profile and gate.floor is not None and not had_floor:
            # With a profile the gate is armed immediately, so the floor's arrival is a
            # drift report rather than a threshold announcement.
            calib_floor = profile.get("floor")
            print(f"[gate] room floor {gate.floor:.3f} vs "
                  + (f"{calib_floor:.3f}" if calib_floor else "no reference")
                  + f" at calibration — drift x{gate.drift:.2f}, start "
                  f"{gate.start_threshold:.2f}, stop {gate.stop_threshold:.2f}"
                  + ("  [CLAMPED: the room has changed enough to be worth re-calibrating]"
                     if gate.drift_clamped else ""))
        pre_roll.append((frame, now, raw_motion))

        display_frame = frame.copy()
        with state_lock:
            status_line = worker_status[0][:70]
        queued_count = clip_queue.qsize()
        queue_line = f"{queued_count} translating/queued" if queued_count else ""
        if queue_line:
            status_line += f"  ({queue_line})"

        # Don't start recording until the models are up, otherwise clips pile up
        # in the queue while the 2.68GB checkpoint is still loading.
        if not models_ready.is_set():
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        elif state == "IDLE" and MANUAL_RECORD:
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(display_frame, "SPACE to start recording", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
            start_now = toggle_pressed
            toggle_pressed = False
            if start_now:
                budget_ok, budget_reason = _capture_budget()
                if not budget_ok:
                    # Same refusal as the automatic path: the camera thread must never
                    # block, and a visibly dropped clip beats OOMing the ones in flight.
                    start_now = False
                    print(f"Not starting — {budget_reason}")
            if start_now:
                state = "RECORDING"
                retained_this_clip = 0
                # No back-dating: the signer pressed the key before signing, so there is
                # no lead-in sitting in the past to recover. That is the whole appeal --
                # the pre-roll, the onset search and the lead pad exist to undo a trigger
                # that fires late, and a key press does not fire late.
                record_start_time = now
                last_motion_time = now
                raw_frame_index = 0
                recorded_frames = []
                frame_times = []
                frame_scores = []
                frame_raw_scores = []
                stream, deferred = _begin_perception()
                extra = (f" (deferred — {MAX_LIVE_STREAMS} streams already live)"
                         if deferred else "")
                print(f">>> RECORDING{extra} <<<")

        elif state == "IDLE":
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            above_start = above_start + 1 if motion_score > gate.start_threshold else 0
            if hand_primary:
                # Hand presence IS the trigger; the pixel score no longer gates anything
                # on the start side. Feed the detector every idle frame -- submit() is
                # non-blocking and drops what it cannot keep up with, so this runs at the
                # detector's own ~10fps -- and never call note_absent(): every verdict in
                # the window now comes from an actual check, which is what makes 3-of-5
                # mean what hand_trigger.py measured it to mean.
                hand_gate.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            elif hand_gate is not None:
                if above_start:
                    # Only while the pixel gate is hot: detection is ~102ms and must not
                    # run continuously. submit() is non-blocking and drops frames the
                    # detector cannot keep up with.
                    hand_gate.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    # Decay, so a hit from minutes ago cannot confirm a later start.
                    hand_gate.note_absent()
            armed = gate.ready or not ADAPTIVE_MOTION
            if not armed:
                # No floor estimate yet -- refuse to record. Without this interlock a
                # noisy room latches on within a few frames, and because the floor is
                # only sampled while idle it would then never be established at all.
                # It applies in hand-primary mode too: the START does not use the floor
                # there, but the STOP still does, and a clip that cannot end is worse
                # than one that starts a second later.
                above_start = 0
                # With a profile this is only the short arming window; without one it is
                # the floor estimate being built, which takes as long as it takes.
                waiting = "arming..." if profile else "calibrating noise floor..."
                cv2.putText(display_frame, waiting, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            if hand_primary:
                # No pixel debounce: at this distance signing does not reliably clear any
                # threshold that stillness stays under, so requiring both means requiring
                # the impossible. The debounce's job -- rejecting a one-frame blip -- is
                # done instead by needing HAND_NEEDED of HAND_WINDOW detector verdicts.
                start_due = armed and hand_gate.confirmed
                # What the trigger is actually seeing, on screen. Without it a signer who
                # is standing too far away or out of frame has no way to tell whether the
                # detector can see their hands, which is now the entire start decision.
                hits, seen = hand_gate.recent_hits
                cv2.putText(display_frame,
                            f"hands {hits}/{seen} (need {HAND_NEEDED} of {HAND_WINDOW})",
                            (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                            (0, 255, 0) if hits >= HAND_NEEDED else (0, 200, 255), 2)
            else:
                start_due = (armed and above_start >= START_DEBOUNCE_FRAMES
                             and (hand_gate is None or hand_gate.confirmed))
            # Checked only when a start is actually due, not every idle frame: this reads
            # /proc/meminfo, and the capture loop runs 30x a second.
            budget_ok, budget_reason = _capture_budget() if start_due else (True, "")
            if not budget_ok:
                # Refuse to start rather than pile another clip's frames onto a backlog
                # that is already at the limit. Say so on screen: a dropped sentence the
                # signer can see and repeat beats an OOM that silently loses clips
                # already in flight.
                above_start = 0
                if hand_primary:
                    hand_gate.reset()
                cv2.putText(display_frame, f"BUSY - backlog full ({budget_reason})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif start_due:
                above_start = 0
                if hand_gate is not None:
                    hand_gate.reset()
                state = "RECORDING"
                retained_this_clip = 0
                # Back-date to motion onset. The trigger is late by the debounce plus the
                # smoothing window, measured at ~2s on the calibration recording -- all of
                # which is signing, so it must come from the ring buffer rather than being
                # dropped.
                onset = find_motion_onset(pre_roll,
                                          stop_threshold=gate.stop_threshold)
                # Extend back by the lead pad, the mirror of TAIL_PAD_SECONDS: cutting
                # exactly at onset would clip the start of the first handshape.
                onset = max(0, onset - int(LEAD_PAD_SECONDS * 30))
                seed = list(pre_roll)[onset:]

                record_start_time = seed[0][1] if seed else now
                last_motion_time = now
                raw_frame_index = 0
                recorded_frames = []
                frame_times = []
                frame_scores = []
                # Stream only if a slot is free. Over the cap the clip is still recorded,
                # just buffered at stride and translated after the cut — degraded latency
                # for that one clip instead of an OOM that loses it entirely.
                stream, deferred = _begin_perception()

                if not STREAM_PERCEPTION:
                    # Legacy path keeps every frame and strides later, at video read.
                    recorded_frames = [f for f, _, _ in seed]
                    frame_times = [t for _, t, _ in seed]
                    frame_scores = [None] * len(frame_times)
                    frame_raw_scores = [r for _, _, r in seed]
                    for _ in recorded_frames:
                        _retain_frame()
                    retained_this_clip += len(recorded_frames)
                else:
                    # Replay the pre-roll through the same stride the live branch uses, so
                    # seeded and live frames form one evenly-sampled sequence.
                    for f, t, r in seed:
                        if raw_frame_index % stride == 0:
                            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            if stream is not None:
                                stream.add_frame(rgb)
                            else:
                                recorded_frames.append(rgb)
                            frame_times.append(t)
                            frame_scores.append(None)
                            frame_raw_scores.append(r)
                            _retain_frame()
                            retained_this_clip += 1
                        raw_frame_index += 1

                extra = "" if stream is not None else (
                    f" (deferred — {MAX_LIVE_STREAMS} streams already live)"
                    if deferred else "")
                print(f">>> RECORDING STARTED{extra} — backdated {len(seed)} frames "
                      f"({(now - record_start_time):.2f}s) to motion onset <<<")

        elif state == "RECORDING":
            # The stride is applied here for both streamed and deferred clips: features.py
            # only strides at video read, which neither of them goes through. Without this
            # the model would get 30fps instead of the ~15fps SHuBERT expects.
            if stream is not None:
                if raw_frame_index % stride == 0:
                    stream.add_frame(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_times.append(now)
                    frame_scores.append(motion_score)
                    frame_raw_scores.append(raw_motion)
                    _retain_frame()
                    retained_this_clip += 1
                raw_frame_index += 1
            elif deferred:
                if raw_frame_index % stride == 0:
                    recorded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_times.append(now)
                    frame_scores.append(motion_score)
                    frame_raw_scores.append(raw_motion)
                    _retain_frame()
                    retained_this_clip += 1
                raw_frame_index += 1
            else:
                recorded_frames.append(frame)
                frame_times.append(now)
                frame_scores.append(motion_score)
                frame_raw_scores.append(raw_motion)
                _retain_frame()
                retained_this_clip += 1
            if motion_score > gate.stop_threshold:
                last_motion_time = now

            still_duration = now - last_motion_time
            elapsed = now - record_start_time

            if stream is not None:
                rec_label = (f"RECORDING - {len(frame_times)} frames, {elapsed:.1f}s "
                             f"(perception {stream.processed_frames}/{stream.queued_frames})")
            else:
                rec_label = (f"RECORDING{' [deferred]' if deferred else ''} - "
                             f"{len(recorded_frames)} frames, {elapsed:.1f}s")
            if MANUAL_RECORD:
                rec_label += "  [SPACE to stop]"
            cv2.putText(display_frame, rec_label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            # While recording, the second line carries the backlog only. The worker's
            # status belongs to the idle screen, and repeating "Ready to sign." under a
            # RECORDING banner just reads as a contradiction.
            if queue_line:
                cv2.putText(display_frame, queue_line, (10, 60),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            force_cut = elapsed >= MAX_CLIP_SECONDS
            manual_stop = MANUAL_RECORD and toggle_pressed
            if manual_stop:
                toggle_pressed = False
            if manual_stop or force_cut or (
                    not MANUAL_RECORD and still_duration >= STILL_DURATION_SECONDS):
                head = 0
                empty_note = None
                if force_cut:
                    # The cut lands mid-sentence, so the TAIL is all signing and stays.
                    # Measure to now.
                    keep = len(frame_times)
                    signing_duration = now - record_start_time
                    print(f">>> FORCED CUT at {elapsed:.1f}s "
                          f"(MAX_CLIP_SECONDS={MAX_CLIP_SECONDS}) <<<")
                    # The HEAD is a different question, and this branch used to skip it.
                    # In manual mode the clip began at a key press whatever ended it, so it
                    # carries the same dead air as any other manual clip -- measured live
                    # 2026-08-21, where a ~7s pause after the press went to the model whole.
                    # No span guard here: the tail is known signing, so the head trim cannot
                    # empty the clip the way it can on a manual stop.
                    if MANUAL_RECORD:
                        head = manual_head_trim(frame_times, frame_raw_scores, gate)
                elif manual_stop:
                    # A key press brackets the sentence loosely at BOTH ends -- the hand
                    # has to leave the keyboard before the first sign and come back after
                    # the last -- so trim the dead air the signer did not mean to record.
                    # Every guard here fails safe by keeping more, never less:
                    #   * MANUAL_TRIM off, or no room-fitted floor yet, so the threshold is
                    #     the untuned fallback constant -> trim nothing at all;
                    #   * neither end may lose more than MANUAL_TRIM_MAX_SECONDS;
                    #   * a trim that would leave less than a MIN_CLIP_DURATION_SECONDS
                    #     span is discarded whole, because a near-empty clip is the one
                    #     input that reliably makes ByT5 hallucinate (and its decode can
                    #     run for minutes -- see the 40-frame "Hmm." clip).
                    # Acceptance still measures press-to-press: the human asked for this
                    # clip, so trimming must not be able to reject it.
                    keep = len(frame_times)
                    signing_duration = now - record_start_time
                    onset = manual_head_trim(frame_times, frame_raw_scores, gate)
                    if MANUAL_TRIM and gate.floor is not None and frame_times:
                        # Taking the LATER of the two cutoffs is what bounds the tail:
                        # last_motion_time only advances above the stop threshold, so a
                        # sentence signed below it (measured on the eval clips:
                        # fingerspelling reads far lower than a whole-frame mean suggests)
                        # would otherwise trim back to almost nothing.
                        tail_cutoff = max(last_motion_time + TAIL_PAD_SECONDS,
                                          frame_times[-1] - MANUAL_TRIM_MAX_SECONDS)
                        tail = sum(1 for t in frame_times if t <= tail_cutoff)

                        span = frame_times[tail - 1] - frame_times[onset] if tail > onset \
                            else 0.0
                        if span >= MIN_CLIP_DURATION_SECONDS:
                            head, keep = onset, tail
                        else:
                            # The trim just measured that essentially nothing between the
                            # two presses was signing. It keeps every frame anyway -- the
                            # guard fails safe toward KEEPING, because the alternative is
                            # truncating a real sentence -- but that is exactly the input
                            # that makes ByT5 invent one, and MIN_MOTION_FRACTION does not
                            # run in manual mode to catch it. Live 2026-08-21: a clip the
                            # signer deliberately held still through came back as "I'm
                            # fluent in Spain". So carry the finding to the translation
                            # instead of dropping it here. Warn, do not reject: a rejected
                            # clip is invisible to the signer, a flagged one is not.
                            # Two decimals, not one: the guard trips at span <
                            # MIN_CLIP_DURATION_SECONDS (1.0), so a span of 0.97 printed
                            # at .1f reads "would leave 1.0s (< MIN_CLIP_DURATION_SECONDS)"
                            # -- a message that contradicts itself. Seen live 2026-08-21.
                            empty_note = f"no signing detected, trim span {span:.2f}s"
                            print(f"Trim skipped — would leave {span:.2f}s "
                                  f"(< MIN_CLIP_DURATION_SECONDS={MIN_CLIP_DURATION_SECONDS}"
                                  f"). This clip looks EMPTY; any translation of it is "
                                  f"probably invented.")
                else:
                    # Drop the trailing stillness; keep everything up to the last motion
                    # plus TAIL_PAD_SECONDS. Gate on signing time rather than wall-clock
                    # elapsed, so a clip that is nothing but the still tail scores ~0s and
                    # is rejected instead of being handed to ByT5 to hallucinate over.
                    cutoff = last_motion_time + TAIL_PAD_SECONDS
                    keep = sum(1 for t in frame_times if t <= cutoff)
                    signing_duration = last_motion_time - record_start_time

                # How much of what is being KEPT is actually moving. A false start --
                # auto-exposure hunting, someone crossing behind, a shadow -- clears the
                # duration gate whenever the disturbance lasts a second or two, and then
                # ByT5 invents a fluent sentence over the dead air rather than returning
                # nothing (measured: a 49-frame near-still clip produced "I am here to show
                # you how to do the heart artery"). Frames seeded from the pre-roll are
                # excluded: they are the lead-in by construction.
                scored = [s for s in frame_scores[head:keep] if s is not None]
                motion_fraction = (
                    sum(1 for s in scored if s > gate.stop_threshold) / len(scored)
                    if scored else 1.0)

                # In manual mode the signer explicitly asked for this clip, so the
                # motion test does not REJECT -- it exists to catch FALSE STARTS, which
                # cannot happen when a key press starts the clip.
                enough_motion = MANUAL_RECORD or motion_fraction >= MIN_MOTION_FRACTION
                # It still MEASURES, though, and in manual mode that measurement is the
                # best empty-clip signal available -- better than the trim span, which is
                # the distance between the two cutoffs and says nothing about the signing
                # between them. Live 2026-08-21: a 5.5s clip the signer held still through
                # kept a wide span (a key-press motion blip at each end) and went out
                # UNFLAGGED, decoding to "...Spring of 2018 and 2017 Spring of 2018". The
                # span check caught the other empty clip that day only because it was short.
                # The threshold is the same MIN_MOTION_FRACTION, which the frozen
                # calibration recordings support as a separator (fraction of smoothed
                # frames above the stop threshold, measured 2026-08-21):
                #     sitting  (calib.jsonl)                 still 37.0%  sign 76.5%
                #                                                         fingerspell 78.2%
                #     standing (fixtures/standing_...jsonl)  still 30.5%  sign 55.6%
                # Clean sitting, marginal standing -- so expect the occasional false
                # warning on a standing signer, which is why this WARNS and never rejects.
                if (MANUAL_RECORD and empty_note is None and scored
                        and motion_fraction < MANUAL_EMPTY_FRACTION):
                    empty_note = f"only {100 * motion_fraction:.0f}% of frames moving"
                    print(f"This clip looks EMPTY — only {100 * motion_fraction:.0f}% of "
                          f"its frames are above the stop threshold "
                          f"(< {100 * MANUAL_EMPTY_FRACTION:.0f}%); any translation of it "
                          f"is probably invented.")
                if signing_duration >= MIN_CLIP_DURATION_SECONDS and enough_motion:
                    clip_counter[0] += 1
                    if stream is not None:
                        label = f"clip_{clip_counter[0]} (streamed)"
                        if empty_note:
                            flag_suspect(label, empty_note)
                        # Hand the whole stream over; the worker drains, closes it,
                        # releases the slot and releases this clip's retained frames.
                        clip_queue.put(("stream", label, stream, head, keep,
                                        retained_this_clip))
                        print(f"Queued {label} ({keep - head} frames, "
                              f"{signing_duration:.1f}s signing; trimmed {head} lead-in + "
                              f"{len(frame_times) - keep} still frames; "
                              f"{stream.processed_frames} already through MediaPipe)")
                        stream = None
                        # Ownership passes to the worker, which releases them in its
                        # finally. Zero it here so the quit path cannot double-release.
                        retained_this_clip = 0
                    elif deferred:
                        label = f"clip_{clip_counter[0]} (deferred)"
                        if empty_note:
                            flag_suspect(label, empty_note)
                        clip_queue.put(("frames", label, recorded_frames[head:keep],
                                        retained_this_clip))
                        print(f"Queued {label} ({keep - head} frames, "
                              f"{signing_duration:.1f}s signing; trimmed {head} lead-in + "
                              f"{len(recorded_frames) - keep} still frames; perception "
                              f"not started)")
                        # As above: the worker owns these frames from here.
                        retained_this_clip = 0
                    else:
                        clip_path = os.path.join(config['temp_dir'],
                                                 f"clip_{clip_counter[0]}.mp4")
                        write_clip(recorded_frames[head:keep], clip_path)
                        if empty_note:
                            flag_suspect(clip_path, empty_note)
                        clip_queue.put(clip_path)
                        print(f"Queued {clip_path} ({keep - head} frames, "
                              f"{signing_duration:.1f}s signing; trimmed {head} lead-in + "
                              f"{len(recorded_frames) - keep} still frames)")
                        # The legacy path hands over an mp4 path, not the frames, so the
                        # buffer is dropped here rather than by the worker.
                        _release_frames(retained_this_clip)
                        retained_this_clip = 0
                else:
                    why = ("too short" if signing_duration < MIN_CLIP_DURATION_SECONDS
                           else f"only {100 * motion_fraction:.0f}% of frames moving")
                    print(f"Ignored — {why} ({signing_duration:.1f}s signing, "
                          f"{len(frame_times)} frames).")
                    if stream is not None:
                        stream.close()
                        _release_stream_slot()
                        stream = None
                    # Nothing was queued, so nothing downstream will release these.
                    _release_frames(retained_this_clip)
                    retained_this_clip = 0

                recorded_frames = []
                frame_times = []
                frame_scores = []
                deferred = False
                if force_cut and signing_duration >= MIN_CLIP_DURATION_SECONDS:
                    # Still mid-sentence. Restart immediately, anchored at now: the clip
                    # just queued covers everything up to this point, so seeding from the
                    # pre-roll would duplicate frames across the boundary. Falls back to
                    # IDLE if the backlog is now full, which is the same refusal the
                    # IDLE branch makes.
                    # CONFIRMED CORRECT FOR MANUAL MODE TOO (user's call, 2026-08-21): queue
                    # the capped clip for translation AND resume recording at once, because
                    # a signer cut at MAX_CLIP_SECONDS is probably mid-sentence. Do not
                    # "fix" this to go IDLE in push-to-record. The known cost is a trailing
                    # clip when the signer stops instead of continuing -- that is what
                    # produced run 2's unasked-for clip_2 -- and the empty-clip warning is
                    # what covers it.
                    budget_ok, budget_reason = _capture_budget()
                    if budget_ok:
                        stream, deferred = _begin_perception()
                        record_start_time = now
                        last_motion_time = now
                        raw_frame_index = 0
                        retained_this_clip = 0
                        state = "RECORDING"
                    else:
                        print(f"Not continuing after forced cut: {budget_reason}")
                        state = "IDLE"
                else:
                    state = "IDLE"

        # Two decimals: the thresholds are now 1.74 / 0.29, so one decimal cannot show
        # whether the score is near either of them.
        cv2.putText(display_frame,
                    f"motion: {motion_score:.2f} (raw {raw_motion:.2f})  "
                    f"start {gate.start_threshold:.2f} stop {gate.stop_threshold:.2f}"
                    + (f"  floor {gate.floor:.3f}" if gate.floor is not None
                       else "  floor --"),
                    (10, 470),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
        show(display_frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord(' '):
            toggle_pressed = True
        if key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Quitting mid-recording leaves a detector holding MediaPipe native resources.
    if stream is not None:
        stream.close()
        _release_stream_slot()
    _release_frames(retained_this_clip)
    if hand_gate is not None:
        hand_gate.close()

    # Drain properly instead of abandoning whatever is mid-flight.
    #
    # This used to be `worker.join(timeout=5)` followed by main() returning. Five seconds
    # is far less than a clip takes, so the interpreter would start tearing down while the
    # worker was still inside MediaPipe -- shutting its ThreadPoolExecutor down underneath
    # it. Every remaining frame then failed with "cannot schedule new futures after
    # shutdown" (measured: 377 frames of a 542-frame clip, landmarks all None, clip lost),
    # and the process died with "terminate called without an active exception" as native
    # threads were destroyed while still joinable. Reproduced with the real classes in
    # scratchpad/quit_repro.py.
    pending = clip_queue.qsize()
    if pending:
        print(f"Finishing {pending} queued clip(s) — press Ctrl-C to abandon them.")
    clip_queue.put(None)
    t0 = time.time()
    try:
        worker.join(timeout=QUIT_DRAIN_SECONDS)
        if worker.is_alive():
            # Say so rather than tearing down silently: the clip IS lost at this point,
            # and the old code lost it without ever admitting to it.
            print(f"Worker still busy after {QUIT_DRAIN_SECONDS:.0f}s; abandoning the "
                  f"clip in progress. Its translation is lost.")
        else:
            print(f"All clips finished ({time.time() - t0:.1f}s).")
    except KeyboardInterrupt:
        print("Abandoning queued clips at user request.")


if __name__ == "__main__":
    main()
