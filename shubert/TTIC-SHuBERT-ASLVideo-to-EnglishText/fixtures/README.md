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
