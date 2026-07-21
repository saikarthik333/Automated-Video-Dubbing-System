"""Small shared helpers: timestamped progress logging and time formatting."""

import time


_start_time = time.time()


def log(step: str, message: str) -> None:
    """Print a progress line with an elapsed-time prefix.

    Example: [  12.4s] TRANSCRIBE  Detected language: de
    """
    elapsed = time.time() - _start_time
    print(f"[{elapsed:7.1f}s] {step.upper():<12} {message}", flush=True)


def format_hhmmss(seconds: float) -> str:
    """Convert seconds -> H:MM:SS.mmm, used for SRT-style debug output."""
    h = int(seconds // 3600)
    m = int((seconds % 3600) // 60)
    s = seconds % 60
    return f"{h}:{m:02d}:{s:06.3f}"
