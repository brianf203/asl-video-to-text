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
import os
os.environ.setdefault("PYTORCH_NO_CUDA_MEMORY_CACHING", "1")

import cv2
import numpy as np
import threading
import queue
import time
from collections import deque
from features import SHuBERTProcessor
from hand_trigger import HandPresenceTrigger
from motion_gate import MotionGate, load_profile
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

CAMERA_INDEX = 0

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
latest_translation = ["Loading models..."]
clip_counter = [0]


def translation_worker(processor):
    """Load models once, then translate queued clips until the sentinel arrives."""
    try:
        t0 = time.time()
        processor.warmup()
        print(f"[worker] Models loaded in {time.time() - t0:.1f}s")
        with state_lock:
            latest_translation[0] = "Ready. Waiting for signing..."
    except Exception as e:
        print(f"[worker] FATAL: model preload failed: {e}")
        with state_lock:
            latest_translation[0] = f"Model load failed: {e}"
        return
    finally:
        models_ready.set()

    while True:
        item = clip_queue.get()
        if item is None:
            break
        # Three shapes arrive here:
        #   ("stream", label, stream, keep, retained)  live StreamingPerception to drain
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
        try:
            print(f"[worker] Translating {label} ...")
            t0 = time.time()
            if streamed:
                _, _, stream, keep, _ = item
                # Drain here rather than on the camera thread: perception runs ~4.5x
                # slower than realtime, so at the cut there is still a backlog worth
                # seconds, and blocking the capture loop on it would freeze the preview
                # and drop frames of whatever the signer does next.
                drain_start = time.time()
                frames, landmarks, embeddings = stream.finish(keep)
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
                latest_translation[0] = result
            print(f"[worker] ({elapsed:.1f}s) {result}")
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
                latest_translation[0] = f"[translation failed: {type(e).__name__}]"
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

    state = "IDLE"
    recorded_frames = []
    frame_times = []
    # Smoothed motion score per retained frame, None for frames seeded from the pre-roll
    # (those carry the raw score, a different quantity, and are the lead-in by
    # construction). Used only to reject clips that contain essentially no motion.
    frame_scores = []
    record_start_time = None
    last_motion_time = None
    prev_gray = None
    stream = None          # StreamingPerception for the clip being recorded
    deferred = False       # recording without a stream, because the cap was reached
    raw_frame_index = 0    # raw frames since RECORDING began, for applying the stride
    retained_this_clip = 0  # frames this clip has added to the global retained budget
    # Thresholds come from calibration when a profile exists (run calibrate_motion.py),
    # because measuring both classes beats inferring signing from the noise floor: a
    # multiplier fitted in one room put the start threshold at 7.06 in a noisier one,
    # above anything signing can produce. Without a profile this falls back to the online
    # floor x multiplier, and ADAPTIVE_MOTION=0 restores the fixed constants outright.
    profile = load_profile() if ADAPTIVE_MOTION else None
    gate = MotionGate(MOTION_SMOOTHING_FRAMES,
                      MOTION_START_THRESHOLD, MOTION_STOP_THRESHOLD,
                      adaptive=ADAPTIVE_MOTION, profile=profile)
    if profile:
        print(f"[gate] calibrated {profile.get('created', '?')} — "
              f"start {gate.start_threshold:.2f}, stop {gate.stop_threshold:.2f} "
              f"(separation {profile.get('separation') or 0:.2f}x)")
        if not profile.get("separable", True):
            print("[gate] WARNING: calibration found signing and ordinary movement "
                  "overlapping — expect occasional false starts; the hand veto covers "
                  "some of them. Re-run calibrate_motion.py if the room changed.")
    elif ADAPTIVE_MOTION:
        print("[gate] no motion_profile.json — inferring thresholds from the noise floor. "
              "Run `python3 calibrate_motion.py` once for measured ones.")
    # Started here so its MediaPipe graph loads while the models warm up, rather than
    # costing the first sentence a ~1s load at the moment it is needed.
    hand_gate = None
    if LANDMARK_TRIGGER:
        hand_gate = HandPresenceTrigger(config['mediapipe_hands_model_path'])
        hand_gate.start()
        print("Landmark trigger ON — a start also requires hands to be visible.")
    pre_roll = deque(maxlen=int(PRE_ROLL_MAX_SECONDS * 30))
    above_start = 0        # consecutive frames over the start threshold (the debounce)
    stride = stride_from_env()
    if STREAM_PERCEPTION:
        print(f"Streaming perception ON (stride {stride}) — landmarks are extracted "
              f"during recording, not after the cut.")

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
        if ADAPTIVE_MOTION and gate.ready and not was_ready:
            # Print once, when the floor is first established: without this the chosen
            # thresholds exist only on the preview overlay and a session log cannot be
            # diagnosed afterwards.
            print(f"[gate] noise floor {gate.floor:.3f} -> start {gate.start_threshold:.2f}, "
                  f"stop {gate.stop_threshold:.2f} "
                  f"(fixed fallback was {MOTION_START_THRESHOLD}/{MOTION_STOP_THRESHOLD})")
        elif profile and gate.floor is not None and not had_floor:
            # With a profile the gate is armed immediately, so the floor's arrival is a
            # drift report rather than a threshold announcement.
            print(f"[gate] room floor {gate.floor:.3f} vs {profile.get('floor')} at "
                  f"calibration — drift x{gate.drift:.2f}, start "
                  f"{gate.start_threshold:.2f}, stop {gate.stop_threshold:.2f}"
                  + ("  [CLAMPED: the room has changed enough to be worth re-calibrating]"
                     if gate.drift_clamped else ""))
        pre_roll.append((frame, now, raw_motion))

        display_frame = frame.copy()
        with state_lock:
            status_line = latest_translation[0][:70]
        queued_count = clip_queue.qsize()
        if queued_count > 0:
            status_line += f"  ({queued_count} translating/queued)"

        # Don't start recording until the models are up, otherwise clips pile up
        # in the queue while the 2.68GB checkpoint is still loading.
        if not models_ready.is_set():
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        elif state == "IDLE":
            cv2.putText(display_frame, status_line, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            above_start = above_start + 1 if motion_score > gate.start_threshold else 0
            if hand_gate is not None:
                if above_start:
                    # Only while the pixel gate is hot: detection is ~102ms and must not
                    # run continuously. submit() is non-blocking and drops frames the
                    # detector cannot keep up with.
                    hand_gate.submit(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                else:
                    # Decay, so a hit from minutes ago cannot confirm a later start.
                    hand_gate.note_absent()
            if not gate.ready and ADAPTIVE_MOTION:
                # No floor estimate yet -- refuse to record. Without this interlock a
                # noisy room latches on within a few frames, and because the floor is
                # only sampled while idle it would then never be established at all.
                above_start = 0
                # With a profile this is only the short arming window; without one it is
                # the floor estimate being built, which takes as long as it takes.
                waiting = "arming..." if profile else "calibrating noise floor..."
                cv2.putText(display_frame, waiting, (10, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)
            # Checked only when a start is actually due, not every idle frame: this reads
            # /proc/meminfo, and the capture loop runs 30x a second.
            budget_ok, budget_reason = (
                _capture_budget() if above_start >= START_DEBOUNCE_FRAMES else (True, ""))
            if not budget_ok:
                # Refuse to start rather than pile another clip's frames onto a backlog
                # that is already at the limit. Say so on screen: a dropped sentence the
                # signer can see and repeat beats an OOM that silently loses clips
                # already in flight.
                above_start = 0
                cv2.putText(display_frame, f"BUSY - backlog full ({budget_reason})",
                            (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            elif above_start >= START_DEBOUNCE_FRAMES and (
                    hand_gate is None or hand_gate.confirmed):
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
                    for _ in recorded_frames:
                        _retain_frame()
                    retained_this_clip += len(recorded_frames)
                else:
                    # Replay the pre-roll through the same stride the live branch uses, so
                    # seeded and live frames form one evenly-sampled sequence.
                    for f, t, _ in seed:
                        if raw_frame_index % stride == 0:
                            rgb = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
                            if stream is not None:
                                stream.add_frame(rgb)
                            else:
                                recorded_frames.append(rgb)
                            frame_times.append(t)
                            frame_scores.append(None)
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
                    _retain_frame()
                    retained_this_clip += 1
                raw_frame_index += 1
            elif deferred:
                if raw_frame_index % stride == 0:
                    recorded_frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame_times.append(now)
                    frame_scores.append(motion_score)
                    _retain_frame()
                    retained_this_clip += 1
                raw_frame_index += 1
            else:
                recorded_frames.append(frame)
                frame_times.append(now)
                frame_scores.append(motion_score)
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
            cv2.putText(display_frame, rec_label,
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.putText(display_frame, status_line, (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            force_cut = elapsed >= MAX_CLIP_SECONDS
            if still_duration >= STILL_DURATION_SECONDS or force_cut:
                if force_cut:
                    # Still signing: there is no still tail to trim and every frame counts
                    # as signing, so keep the lot and measure the duration to now.
                    keep = len(frame_times)
                    signing_duration = now - record_start_time
                    print(f">>> FORCED CUT at {elapsed:.1f}s "
                          f"(MAX_CLIP_SECONDS={MAX_CLIP_SECONDS}) <<<")
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
                scored = [s for s in frame_scores[:keep] if s is not None]
                motion_fraction = (
                    sum(1 for s in scored if s > gate.stop_threshold) / len(scored)
                    if scored else 1.0)

                if (signing_duration >= MIN_CLIP_DURATION_SECONDS
                        and motion_fraction >= MIN_MOTION_FRACTION):
                    clip_counter[0] += 1
                    if stream is not None:
                        label = f"clip_{clip_counter[0]} (streamed)"
                        # Hand the whole stream over; the worker drains, closes it,
                        # releases the slot and releases this clip's retained frames.
                        clip_queue.put(("stream", label, stream, keep,
                                        retained_this_clip))
                        print(f"Queued {label} ({keep} frames, {signing_duration:.1f}s "
                              f"signing; trimmed {len(frame_times) - keep} still frames; "
                              f"{stream.processed_frames} already through MediaPipe)")
                        stream = None
                        # Ownership passes to the worker, which releases them in its
                        # finally. Zero it here so the quit path cannot double-release.
                        retained_this_clip = 0
                    elif deferred:
                        label = f"clip_{clip_counter[0]} (deferred)"
                        clip_queue.put(("frames", label, recorded_frames[:keep],
                                        retained_this_clip))
                        print(f"Queued {label} ({keep} frames, {signing_duration:.1f}s "
                              f"signing; trimmed {len(recorded_frames) - keep} still "
                              f"frames; perception not started)")
                        # As above: the worker owns these frames from here.
                        retained_this_clip = 0
                    else:
                        clip_path = os.path.join(config['temp_dir'],
                                                 f"clip_{clip_counter[0]}.mp4")
                        write_clip(recorded_frames[:keep], clip_path)
                        clip_queue.put(clip_path)
                        print(f"Queued {clip_path} ({keep} frames, {signing_duration:.1f}s "
                              f"signing; trimmed {len(recorded_frames) - keep} still frames)")
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
        cv2.imshow('Auto ASL Translation (v5 - persistent worker)', display_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
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
