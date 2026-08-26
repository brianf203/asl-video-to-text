"""Append-only session transcript, written and fsync'd one line at a time.

Why this exists: until now a session's translations existed ONLY in the terminal's
scrollback. This board dies silently -- caught on telemetry 2026-08-24 going down at
5.2W idle with every alarm flat, so it is a power-delivery fault and no software change
prevents it -- and when it does, the whole session's output goes with it. There is no
shutdown hook to hang a save on, because there is no shutdown: the power simply stops.

That constraint drives every decision here:

* **fsync, not just flush.** `flush()` only moves bytes from Python's buffer into the
  page cache, which is RAM. A clean kill survives that; a power cut does not. `fsync`
  is what puts the line on the eMMC. Cost is one sync per clip, i.e. once per ~20s of
  translation work, against a pipeline that spends seconds per clip -- unmeasurable.
* **Open, append, fsync, close per line.** A long-lived handle buys nothing when the
  process can vanish between any two instructions, and it costs the one failure mode
  that matters: a handle opened at startup and never re-checked can be silently invalid
  by the time the first clip lands (removable media, a deleted directory). Reopening
  makes every write independently succeed or fail.
* **The transcript is written at the same moment the line is printed**, not at exit.
  Anything queued for later is exactly what a cut takes.

Failures here NEVER propagate. A full disk must not cost a translation, so a write error
is reported once and the log disables itself.
"""
import os
import time

# Where sessions land. One file per run, named for when the run started, so a crashed
# session and the one that follows it never collide.
TRANSCRIPT_DIR = os.environ.get("TRANSCRIPT_DIR", "transcripts")
# Exact path override, for probes that want a known filename.
TRANSCRIPT_PATH = os.environ.get("TRANSCRIPT_PATH", "")
# TRANSCRIPT=0 turns the whole thing off.
TRANSCRIPT_ENABLED = os.environ.get("TRANSCRIPT", "1") not in ("0", "false", "False")

# Same prefix the terminal uses, so `grep 'Signer: '` works identically on a live session
# and on a saved transcript.
PREFIX = "Signer: "


class TranscriptLog:
    """One session's transcript. Thread-compatible: only the worker thread writes."""

    def __init__(self, path=None, enabled=None):
        self.enabled = TRANSCRIPT_ENABLED if enabled is None else enabled
        if path is None:
            path = TRANSCRIPT_PATH or os.path.join(
                TRANSCRIPT_DIR, f"session_{time.strftime('%Y%m%d-%H%M%S')}.txt")
        self.path = path
        self.started = time.time()
        self.lines = 0
        # The file is created on the FIRST line, not at startup: a session that opens the
        # camera and quits without signing should not leave an empty file behind, and
        # there are a lot of those (calibration checks, threshold probes, restarts).
        self._opened = False

    def _write(self, text):
        """Append one line durably. Returns False once the log has given up."""
        if not self.enabled:
            return False
        try:
            directory = os.path.dirname(self.path)
            if directory:
                os.makedirs(directory, exist_ok=True)
            if not self._opened:
                header = (f"# ASL session transcript — started "
                          f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.started))}\n"
                          f"# One line per clip, fsync'd as it is printed.\n\n")
                text = header + text
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(text)
                fh.flush()
                # The whole point of the file. See the module docstring.
                os.fsync(fh.fileno())
            if not self._opened:
                self._opened = True
                print(f"[transcript] saving to {self.path}", flush=True)
            return True
        except OSError as e:
            # Once, then stay quiet: a full disk would otherwise print this per clip and
            # bury the translations it is supposed to be protecting.
            print(f"[transcript] DISABLED — cannot write {self.path}: "
                  f"{type(e).__name__}: {e}", flush=True)
            self.enabled = False
            return False

    def record(self, label, text, seconds=None, note=None):
        """Save one translation. `note` is the suspect-clip warning, if any."""
        stamp = time.strftime("%H:%M:%S")
        took = f" ({seconds:.1f}s)" if seconds is not None else ""
        # The warning rides inside the line for the same reason it does on the terminal
        # (auto_segment_v5.py:668): a caveat stored anywhere else is not there when
        # someone reads the transcript back as a record of what was signed.
        marker = f"[{note} — likely invented] " if note else ""
        if self._write(f"[{stamp}] {label}{took} {PREFIX}{marker}{text}\n"):
            self.lines += 1

    def record_error(self, label, message):
        """Save a clip that failed. A gap in a transcript should be explained, not blank."""
        stamp = time.strftime("%H:%M:%S")
        self._write(f"[{stamp}] {label} ERROR {message}\n")

    def close(self):
        """Footer, on the paths where we do get to exit cleanly. Never required."""
        if not self._opened or not self.enabled:
            return
        minutes = (time.time() - self.started) / 60.0
        self._write(f"\n# session ended {time.strftime('%Y-%m-%d %H:%M:%S')} — "
                    f"{self.lines} translation(s) over {minutes:.1f} min\n")
