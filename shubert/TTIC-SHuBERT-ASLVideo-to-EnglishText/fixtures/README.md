# Frozen calibration recordings

`last_calibration.jsonl` in the parent directory is **regenerated on every launch** and is
gitignored, so any analysis written against it is silently invalidated by the next run.
That happened once: the 2026-08-16 standing-distance analysis was written against a
recording that a demo run overwrote a minute later, and `test_hand_primary.py` then failed
against numbers the note said it should pass.

Recordings that a test or a written finding depends on get copied here and committed.

## standing_calibration_20260816.jsonl

The standing session that motivated the hand-primary trigger (2026-08-16 15:43, 663
labelled frames, 30fps). Fitted profile: start 0.491 / stop 0.190, separation 0.98x,
`hand_gated: true`.

| phase | frames | detector verdicts | with hands |
|---|---|---|---|
| STAND STILL | 181 | 96 | 13, all within the first 1.30s |
| MOVE (NO SIGNING) | 211 | 123 | 0 |
| SIGN | 271 | 107 | 55 |

Signing peaks *below* deliberate non-signing movement, which is the whole point of the
recording: at this distance no pixel threshold separates the two classes.

The 13 STAND STILL detections are the signer settling in front of the camera at launch,
with their hands still up. They are inside `motion_gate.ARMING_SECONDS` (2.0s), which the
live path refuses to start within, so they are not reachable false starts.

## demo_20260816_motion_handprimary.log

The 2026-08-16 15:43 live session (`RECORD_MODE=motion`, hand-primary active), frozen from
`demo_full_log.txt` — which `demo_transcript.py` OVERWRITES on every run, the same trap that
cost the calibration recording above. This is the session the 2026-08-16 section's live-result
table is computed from: 4 starts, 3 accepted, one rejected by `MIN_MOTION_FRACTION`, post-cut
23.5s / 12.0s / 7.5s.

## manual_trim_live_20260821.log

The first live push-to-record session with `MANUAL_TRIM` on (2026-08-21, all defaults).
8 clips, 8 translated, 0 failures. Head trim fired on 4 (22 / 6 / 13 / 4 frames), skipped
correctly on 2 no-pause clips, was bypassed by a forced cut on clip_7 (15.0s cap) and
refused by the span guard on clip_8. This is the log the 2026-08-21 section's table and its
"the latency claim does not survive" finding are computed from.

Two things the log does NOT show, both from the signer: clip_7 had ~7s of dead air at the
head (a forced cut skips the head trim, and 7s exceeds MANUAL_TRIM_MAX_SECONDS anyway), and
clip_8 contained NO SIGNING AT ALL — "I'm fluent in Spain" is a hallucination on 60 still
frames. Do not read the clip_8 row as a translation.
