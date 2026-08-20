"""Show ONLY the translated sentences, for demos.

auto_segment_v5.py prints a lot: model loads, per-clip perception timings, queue depth,
gate thresholds, memory accounting. All of that is diagnostic -- it is what makes a latency
regression or an OOM attributable afterwards -- but in front of an audience it buries the
one line that matters. The translation is already printed on its own prefixed line
(`Signer: ...`, flushed immediately) precisely so it can be separated out, so this filters
the stream rather than changing what the pipeline logs.

    python3 demo_transcript.py                    # launches auto_segment_v5.py itself
    python3 auto_segment_v5.py | python3 demo_transcript.py     # or filter a pipe

The camera window still opens and behaves exactly as usual (SPACE to start/stop a clip,
'q' to quit) -- only the terminal output is filtered. Nothing is thrown away: the full
log is written to a file so a demo failure is still diagnosable.

Options:
    --bare              print the sentence without the "Signer: " prefix
    --log PATH          where to tee the full output (default demo_full_log.txt)
    --no-log            do not write a log file
    -- CMD [ARGS...]    run something other than `python3 -u auto_segment_v5.py`
"""
import os
import subprocess
import sys

PREFIX = "Signer: "
DEFAULT_LOG = os.environ.get("DEMO_LOG", "demo_full_log.txt")


def parse_args(argv):
    bare = False
    log_path = DEFAULT_LOG
    command = None
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == "--bare":
            bare = True
        elif arg == "--no-log":
            log_path = None
        elif arg == "--log":
            i += 1
            if i >= len(argv):
                sys.exit("--log needs a path")
            log_path = argv[i]
        elif arg == "--":
            command = argv[i + 1:]
            break
        elif arg in ("-h", "--help"):
            print(__doc__)
            sys.exit(0)
        else:
            sys.exit(f"unknown argument {arg!r} (see --help)")
        i += 1
    return bare, log_path, command


def emit(line, bare):
    """Print one translation. Flushed: stdout may itself be a pipe or a recording."""
    text = line[len(PREFIX):] if bare else line
    print(text, flush=True)


def filter_stream(stream, bare, log):
    for line in stream:
        if log is not None:
            log.write(line)
            log.flush()    # so the log survives a Ctrl-C or a kill mid-demo
        line = line.rstrip("\n")
        if line.startswith(PREFIX):
            emit(line, bare)


def main():
    bare, log_path, command = parse_args(sys.argv[1:])
    log = open(log_path, "w", buffering=1) if log_path else None

    try:
        if command is None and not sys.stdin.isatty():
            # Being piped into: `auto_segment_v5.py | demo_transcript.py`. The upstream
            # process owns the terminal's stdin and the camera window either way.
            filter_stream(sys.stdin, bare, log)
            return 0

        if command is None:
            # -u because stdout is a pipe here, so Python would otherwise buffer the
            # child's diagnostics in 8KB blocks. The translation line is flushed
            # explicitly upstream, but the log file should stay readable live too.
            command = [sys.executable, "-u",
                       os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "auto_segment_v5.py")]

        # stderr folded into stdout: a traceback is the one thing worth seeing during a
        # demo, and it lands in the log rather than scrolling past the transcript.
        proc = subprocess.Popen(command, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            filter_stream(proc.stdout, bare, log)
            return proc.wait()
        except KeyboardInterrupt:
            # Ctrl-C already reached the child (same process group). Wait for it rather
            # than exiting underneath it: quitting mid-clip has a drain path that
            # finishes queued translations, and killing the pipe here would cut it off.
            try:
                return proc.wait(timeout=200)
            except (KeyboardInterrupt, subprocess.TimeoutExpired):
                proc.kill()
                return 130
    finally:
        if log is not None:
            log.close()
            print(f"(full log: {log_path})", file=sys.stderr)


if __name__ == "__main__":
    sys.exit(main())
