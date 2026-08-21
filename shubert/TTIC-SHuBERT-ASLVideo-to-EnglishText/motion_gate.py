"""Motion smoothing and start/stop thresholds, from calibration where one exists.

THREE MODES, IN ORDER OF PREFERENCE
-----------------------------------
1. CALIBRATED (a `motion_profile.json` from `calibrate_motion.py`): thresholds are fitted
   to measured still / non-signing-motion / signing distributions, and the online floor
   only applies a drift correction clamped to +/-40%. This is the one to use.
2. ADAPTIVE (no profile): floor x START_MULTIPLIER, described below. It measures only the
   still side and infers the signing side from a multiplier fitted in one room, which
   fails when a room's noise floor is high -- see the 7.06 case in PROFILE_PATH's comment.
3. FIXED (adaptive=False): the hand-fitted constants, valid only in the room they came
   from.

Everything below documents mode 2, which remains the fallback when nobody has calibrated.

WHY MODE 2 EXISTS
-----------------
`auto_segment_v5.py` used absolute thresholds (start 1.74, stop 0.29) fitted by
`calibrate_motion.py` on one camera, one room and one lighting setup. Those numbers are
raw pixel-difference units, so they do not transfer: a darker room raises the sensor noise
floor, auto-exposure hunting shifts every pixel at once, a signer further from the camera
moves fewer pixels, and a different sensor has a different noise floor entirely. Ship the
constants elsewhere and the machine either never starts or never stops.

The separation between stillness and signing is a RATIO, not a distance. Measured on
calib.jsonl (2534 labelled frames, this room): smoothed still median 0.264 vs signing
median 0.525, i.e. only 1.98x apart -- but the shipped thresholds sit at fixed multiples
of the still median:

    start 1.74 = 6.58 x still-smoothed-median
    stop  0.29 = 1.10 x still-smoothed-median

So estimate the still median online and keep those multipliers. In this room that
reproduces today's behaviour exactly; in another room it tracks whatever the floor is.

WHAT THIS CANNOT FIX
--------------------
It rescales thresholds, it does not improve the underlying signal. If a room changes the
signal-to-noise RATIO -- a moving background, a signer so far away their motion falls into
the noise -- the 6.58x/1.10x multipliers stop being right and no rescaling saves them. The
principled fix for that is a semantic trigger (hand presence/velocity from MediaPipe
landmarks) rather than a global pixel mean; see the notes on arXiv 2607.09611's FSM.

Note the start threshold sits ABOVE the still smoothed maximum (1.571) and near the top of
the signing distribution (p99 1.68), which is deliberate: it triggers rarely and precisely,
and back-dating to motion onset recovers the lead-in. Do not "fix" it to trigger more
readily without re-running the end-to-end acceptance test -- optimising a proxy picked the
wrong config once already.
"""
import json
import os
import time
from collections import deque

# A calibration profile, when one exists, REPLACES the floor x multiplier inference: the
# thresholds come from measured still / non-signing-motion / signing distributions
# (see calibrate_motion.py and fit_thresholds.py). The online floor is still tracked, but
# only to correct for DRIFT since calibration -- the room dimming, the camera gaining a
# different auto-exposure operating point.
#
# Why this exists: inferring the signing side from the still side was measured to fail.
# Live 2026-08-12, a floor of 0.785 x 9.0 put the start threshold at 7.06 while signing's
# smoothed maximum is 1.889 -- the second sentence of the session could not be started at
# all. A multiplier fitted in one room cannot know how hard signing moves in another; only
# a measurement can.
PROFILE_PATH = os.environ.get("MOTION_PROFILE", "motion_profile.json")

# Drift correction is CLAMPED to this factor either way. Past a 1.4x change in the noise
# floor the room is not the room that was calibrated, and silently rescaling a measured
# threshold by an arbitrary amount is how the floor x 9.0 design produced an unreachable
# 7.06. Beyond the clamp the gate holds the calibrated value and says so.
DRIFT_MAX = float(os.environ.get("MOTION_DRIFT_MAX", "1.4"))


# A profile arms the gate immediately -- the thresholds are already known -- but not
# INSTANTLY. Whoever launched the program is settling into frame, and that motion reaches
# signing levels: simulating the calibrated config over calib.jsonl produced a spurious
# clip at t+1.5s, inside the first SIT STILL span, which the adaptive path avoided only
# because its floor interlock happened to refuse the first ~2s. The hand veto does not
# help here (hands are visible while you settle), so hold the same window deliberately
# rather than inheriting it by accident.
ARMING_SECONDS = float(os.environ.get("MOTION_ARMING_SECONDS", "2.0"))


def load_profile(path=PROFILE_PATH):
    """Return the calibration profile dict, or None if there is not a usable one."""
    try:
        with open(path) as fh:
            profile = json.load(fh)
    except (OSError, ValueError):
        return None
    if not profile.get("start_threshold"):
        return None
    return profile

# Multipliers on the ESTIMATED ONLINE FLOOR -- note that is not the same quantity as the
# global still median. Fitting against the global median (0.264) gave 6.58x, but the
# rolling estimator actually produces ~0.155 here, so 6.58x put the threshold at 1.04 and
# produced a false start. Always fit a multiplier against the estimator you ship.
#
# Swept over calib.jsonl with simulate_segmentation.py (0 false starts required):
#     6.58x -> 1 false start        <- too low
#     8.0x - 11.3x -> 0 false starts, 97.5-98.0% coverage
#     11.3x -> threshold 1.74, i.e. exactly reproduces the hand-fitted config
#     12.0x -> 0 clips: never triggers at all       <- cliff
# 9.0 is the log-midpoint of that band (sqrt(6.58 * 12) ~= 8.9), so it carries the most
# margin in BOTH directions -- which is the whole point when the room is unknown. Sitting
# at 11.3 would reproduce this room exactly but leave almost no headroom before the cliff.
# The 6:1 start:stop ratio is preserved from the hand-fitted pair (1.74 / 0.29).
START_MULTIPLIER = float(os.environ.get("MOTION_START_MULT", "9.0"))
STOP_MULTIPLIER = float(os.environ.get("MOTION_STOP_MULT", "1.5"))

# Seconds of QUIET score history the floor is estimated from. Long enough to be stable
# across a pause in signing, short enough to follow a lighting change.
FLOOR_WINDOW_SECONDS = float(os.environ.get("MOTION_FLOOR_WINDOW", "10"))

# Only frames at or below stop_threshold * this factor feed the floor.
#
# THIS IS THE LOAD-BEARING PART. Sampling every idle frame -- which the first version did
# -- creates a runaway: while the signer is moving but has not triggered yet, those moving
# frames raise the floor, which raises the threshold, which makes triggering harder, so
# they keep moving and it climbs further. Reported from a live session as "it kept going up
# the more I was moving and I could never start a sign until I held still for several
# seconds". Measured on calib.jsonl: sustained motion drags a median-of-idle floor to 0.525,
# putting the start threshold at 4.72 -- while the highest a signing frame ever reaches is
# 1.889. The trigger becomes unreachable by construction, which is exactly what was
# reported. A low quantile alone does NOT fix it (p10 still gives 2.21, also unreachable).
# "Idle" is not "still", so the floor must be fed only frames that are actually quiet.
QUIET_ACCEPT_FACTOR = float(os.environ.get("MOTION_QUIET_FACTOR", "1.5"))

# A frame only counts toward the floor if it is part of a SUSTAINED quiet run this long.
#
# Per-frame acceptance is not enough, and the reason is worth keeping: signing contains
# holds and transitions with genuinely near-zero motion, so a good fraction of moving
# frames slip under any per-frame bound. Those inflate the floor, which raises the bound,
# which admits more of them -- the same runaway one level down. Measured: per-frame
# acceptance alone still drove the start threshold to 4.43 (unreachable).
#
# Requiring N contiguous quiet frames fixes it because fidgeting cannot produce a long
# quiet run by definition. During motion the floor simply FREEZES at its last good value,
# which is exactly the wanted behaviour: the threshold stays where it was and stays
# reachable.
QUIET_RUN_SECONDS = float(os.environ.get("MOTION_QUIET_RUN", "2.0"))
# If no quiet run has been seen for this long the room probably got genuinely noisier, so
# widen the bound step by step until one is found again. Without this the floor can never
# rise and a noisy room false-starts forever. Widening the BOUND (not the floor) keeps the
# floor itself sourced only from real quiet runs.
QUIET_STALE_SECONDS = float(os.environ.get("MOTION_QUIET_STALE", "15"))
QUIET_RELAX_STEP = float(os.environ.get("MOTION_QUIET_RELAX", "1.5"))
# Until this many idle samples exist there is no floor estimate, so the fixed fallback
# thresholds are used -- which is what makes startup behave sanely.
FLOOR_MIN_SAMPLES = int(os.environ.get("MOTION_FLOOR_MIN_SAMPLES", "60"))
# The BOOTSTRAP estimate is a low percentile, not the median, because that window accepts
# every frame -- there is no threshold yet to judge "quiet" against -- so it contains
# whatever the signer was doing at launch. A median of a contaminated window tracks the
# contamination; a low percentile tracks the quiet part of it, which is what a noise floor
# is. Measured on calib.jsonl (2026-08-21), inflation of the bootstrap estimate against a
# clean window, as the window fills with GENUINELY moving frames:
#     contamination     25%     50%     75%
#     p10 (this)       1.16x   2.06x   4.43x
#     median (before)  3.51x   5.51x   8.36x
# So this is RELATIVE robustness, not immunity -- p10 still inflates once most of the window
# is motion. An earlier version of this comment claimed p10 was flat at 0.9x regardless;
# that came from a fixture whose "motion" was the signer reading the calibration prompt
# (below threshold), i.e. not motion at all. Same trap test_manual_trim.py hit. The sweep
# above is in test_motion_gate.py so the claim cannot rot.
# Live evidence this matters (run 4, 2026-08-21): a launch with motion in it bootstrapped
# the floor to 0.900 against a converged value of ~0.157, and while it drained, a genuinely
# SIGNED clip measured 53% moving against the 50% empty-clip cutoff -- three points from
# having a real sentence marked "likely invented".
BOOTSTRAP_PERCENTILE = float(os.environ.get("MOTION_BOOTSTRAP_PERCENTILE", "10"))
# A floor of ~0 (a synthetic or perfectly static feed) would drive both thresholds to zero
# and latch the machine on. Clamp it.
FLOOR_MIN = float(os.environ.get("MOTION_FLOOR_MIN", "0.02"))


def _percentile(values, pct):
    """Nearest-rank percentile. Small windows, no numpy dependency in this module."""
    ordered = sorted(values)
    if not ordered:
        return 0.0
    k = int(round(pct / 100.0 * (len(ordered) - 1)))
    return ordered[max(0, min(len(ordered) - 1, k))]


def _median(values):
    ordered = sorted(values)
    n = len(ordered)
    if n == 0:
        return 0.0
    mid = n // 2
    if n % 2:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


class MotionGate:
    """Turns a raw per-frame motion score into smoothed value + start/stop thresholds.

    Used by both the live path and the offline simulator, so what gets validated is the
    code that runs -- not a reimplementation of it.
    """

    def __init__(self, smoothing_frames, fixed_start, fixed_stop, fps=30.0,
                 adaptive=True, profile=None):
        self._smooth = deque(maxlen=max(1, smoothing_frames))
        self._fixed_start = fixed_start
        self._fixed_stop = fixed_stop
        self.adaptive = adaptive
        self.profile = profile
        self._profile_floor = (profile or {}).get("floor")
        # Floor samples are the SMOOTHED score, because that is what the thresholds are
        # compared against. Estimating on the raw score would be a different quantity.
        self._floor_samples = deque(maxlen=max(FLOOR_MIN_SAMPLES,
                                               int(FLOOR_WINDOW_SECONDS * fps)))
        self.smoothed = 0.0
        self.floor = None
        self._frames_seen = 0
        # Accept-everything phase, only until the first floor estimate exists -- and only
        # when there is no profile. With one, a meaningful stop threshold exists from the
        # first frame, so the sustained-quiet-run rule can be applied immediately. That
        # matters: bootstrapping accepts whatever the signer happens to be doing at launch,
        # and if that is motion the floor starts inflated. Bounded by the drift clamp, but
        # there is no reason to take the error at all when calibration already ran.
        self._bootstrapping = profile is None
        self._last_quiet_t = None
        self._quiet_run = 0      # consecutive frames under the quiet bound
        self._relax = 1.0        # widens the bound if the room got genuinely noisier
        self._fps = fps

    def update(self, raw_score, recording, now=None):
        """Feed one frame. Returns the smoothed score.

        `recording` excludes frames belonging to a clip -- during one the score IS
        signing. But that alone is not enough: idle frames can be full of motion too, so
        only frames that look QUIET feed the floor. See QUIET_ACCEPT_FACTOR.
        """
        self._smooth.append(raw_score)
        self.smoothed = sum(self._smooth) / len(self._smooth)
        self._frames_seen += 1
        if now is None:
            now = time.time()

        if not recording:
            if self._bootstrapping:
                # At startup there is no threshold to judge "quiet" against, so accept
                # everything briefly. The user has just launched the program and is at the
                # keyboard, not signing. Once a floor exists this stops.
                self._floor_samples.append(self.smoothed)
                self._last_quiet_t = now
                if len(self._floor_samples) >= FLOOR_MIN_SAMPLES:
                    # Low percentile, NOT the median: see BOOTSTRAP_PERCENTILE. Steady
                    # state keeps the median, because those samples are already filtered to
                    # sustained quiet runs and do not need the robustness.
                    self.floor = max(FLOOR_MIN,
                                     _percentile(self._floor_samples, BOOTSTRAP_PERCENTILE))
                    self._bootstrapping = False
            else:
                bound = self.stop_threshold * QUIET_ACCEPT_FACTOR * self._relax
                run_needed = max(1, int(QUIET_RUN_SECONDS * self._fps))
                if self.smoothed <= bound:
                    self._quiet_run += 1
                    # Only a SUSTAINED quiet run counts. Motion cannot fake one, so the
                    # floor freezes during motion instead of chasing it upward.
                    if self._quiet_run >= run_needed:
                        self._floor_samples.append(self.smoothed)
                        self._last_quiet_t = now
                        self._relax = 1.0
                        self.floor = max(FLOOR_MIN, _median(self._floor_samples))
                else:
                    self._quiet_run = 0
                    if (self._last_quiet_t is not None
                            and now - self._last_quiet_t > QUIET_STALE_SECONDS):
                        # Room seems permanently noisier than the current bound allows.
                        self._relax *= QUIET_RELAX_STEP
                        self._last_quiet_t = now
        return self.smoothed

    @property
    def drift(self):
        """Multiplier applied to calibrated thresholds, clamped to [1/DRIFT_MAX, DRIFT_MAX].

        1.0 when there is no profile floor to compare against, or before the online floor
        is established -- in both cases the calibrated thresholds are used unmodified,
        which is the whole point of having measured them.
        """
        if not self.profile or not self._profile_floor or self.floor is None:
            return 1.0
        ratio = self.floor / self._profile_floor
        return min(DRIFT_MAX, max(1.0 / DRIFT_MAX, ratio))

    @property
    def drift_clamped(self):
        """True when the room has moved further than the clamp allows -- worth reporting."""
        if not self.profile or not self._profile_floor or self.floor is None:
            return False
        ratio = self.floor / self._profile_floor
        return ratio > DRIFT_MAX or ratio < 1.0 / DRIFT_MAX

    @property
    def ready(self):
        """False until enough idle samples exist; the fixed fallback is in use.

        The caller must REFUSE TO START RECORDING while this is False (in adaptive mode).
        Without that interlock a noisy room latches the machine on within the first few
        frames, and since the floor is only sampled while idle it then never gets
        established at all -- measured: at 4x the calibration noise level the floor stayed
        unset for the whole recording and the gate silently ran on the fixed fallback
        thresholds forever. A ~2s pause before the first clip is a trivial price.

        With a calibration profile the thresholds are already known, so the wait is only
        ARMING_SECONDS -- long enough for whoever launched the program to stop settling
        into frame, which is motion at signing levels.
        """
        if self.profile is not None:
            return self._frames_seen >= int(ARMING_SECONDS * self._fps)
        return self.floor is not None

    @property
    def start_threshold(self):
        if self.profile:
            return self.profile["start_threshold"] * self.drift
        if self.adaptive and self.floor is not None:
            return self.floor * START_MULTIPLIER
        return self._fixed_start

    @property
    def stop_threshold(self):
        if self.profile:
            return self.profile["stop_threshold"] * self.drift
        if self.adaptive and self.floor is not None:
            return self.floor * STOP_MULTIPLIER
        return self._fixed_stop
