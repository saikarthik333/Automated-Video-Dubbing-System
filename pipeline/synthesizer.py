"""Step 3: synthesize natural English speech for each translated segment.

Uses edge-tts (Microsoft Edge's neural TTS, free, no API key) for natural-
sounding voices. Two things make the dub feel matched to the original:

1. Voice gender pick: we estimate the speaker's average pitch (F0) from the
   original audio and choose a male or female neural voice accordingly.
   (Single voice for the whole video -- multi-speaker voice assignment is
   the stretch goal, not attempted here.)
2. Duration matching: TTS output rarely lands exactly on the original
   segment's duration. We synthesize once at normal rate, measure the
   result, then re-synthesize with a speaking-rate adjustment so the clip
   fits its time slot -- keeping the dub in sync with the video.
"""

import asyncio
import os
import time
from typing import List

import edge_tts
import numpy as np
import soundfile as sf
from pydub import AudioSegment

from .transcriber import Segment
from .utils import log

VOICE_MALE = "en-US-AndrewNeural"
VOICE_FEMALE = "en-US-AvaNeural"

# Rate adjustment bounds so we never distort speech into gibberish.
MIN_RATE_PCT, MAX_RATE_PCT = -40, 60

# edge-tts calls a remote WebSocket; flaky connections occasionally drop mid
# -call. Retry a few times with backoff rather than losing a whole long run
# to one transient blip.
MAX_TTS_RETRIES = 4
RETRY_BACKOFF_SECONDS = 3


def detect_voice(video_path: str) -> str:
    """Estimate whether the source speaker is closer to a typical male or
    female pitch range, using median fundamental frequency. Falls back to
    the male voice if pitch can't be estimated (e.g. librosa missing).
    """
    try:
        import librosa

        y, sr = librosa.load(video_path, sr=16000, duration=120)  # sample first 2 min
        f0, voiced_flag, _ = librosa.pyin(y, fmin=65, fmax=400, sr=sr)
        f0 = f0[voiced_flag]
        if len(f0) == 0:
            raise ValueError("no voiced frames detected")
        median_f0 = float(np.nanmedian(f0))
        voice = VOICE_FEMALE if median_f0 > 165 else VOICE_MALE
        log("synthesize", f"Estimated median pitch {median_f0:.0f} Hz -> using {voice}")
        return voice
    except Exception as e:
        log("synthesize", f"Pitch detection skipped ({e}); defaulting to {VOICE_MALE}")
        return VOICE_MALE


async def _synthesize_one(text: str, voice: str, rate_pct: int, out_path: str) -> None:
    rate_str = f"{rate_pct:+d}%"
    last_error = None
    for attempt in range(1, MAX_TTS_RETRIES + 1):
        try:
            communicate = edge_tts.Communicate(text, voice, rate=rate_str)
            await communicate.save(out_path)
            return
        except Exception as e:
            last_error = e
            if attempt < MAX_TTS_RETRIES:
                log(
                    "synthesize",
                    f"TTS attempt {attempt}/{MAX_TTS_RETRIES} failed ({e}); "
                    f"retrying in {RETRY_BACKOFF_SECONDS}s...",
                )
                await asyncio.sleep(RETRY_BACKOFF_SECONDS)
    raise last_error


def _clip_duration(path: str) -> float:
    info = sf.info(path)
    return info.frames / info.samplerate


def synthesize_segment(text: str, voice: str, target_duration: float, out_path: str) -> AudioSegment:
    """Synthesize `text` so its spoken duration fits `target_duration` seconds
    as closely as possible, writing the final clip to `out_path`.
    """
    tmp_path = out_path + ".tmp.mp3"
    asyncio.run(_synthesize_one(text, voice, 0, tmp_path))
    natural_duration = _clip_duration(tmp_path)

    if natural_duration > 0.05 and target_duration > 0.05:
        # positive rate = speak faster = shorter clip
        rate_pct = int(round((natural_duration / target_duration - 1) * 100))
        rate_pct = max(MIN_RATE_PCT, min(MAX_RATE_PCT, rate_pct))
    else:
        rate_pct = 0

    if rate_pct != 0:
        asyncio.run(_synthesize_one(text, voice, rate_pct, tmp_path))

    clip = AudioSegment.from_file(tmp_path)
    clip.export(out_path, format="wav")
    os.remove(tmp_path)
    return clip


def synthesize_all(
    segments: List[Segment], texts: List[str], video_path: str, work_dir: str
) -> List[tuple[float, AudioSegment]]:
    """Synthesize English audio for every segment. Returns list of
    (start_time, AudioSegment) ready to be placed on the output timeline.
    """
    os.makedirs(work_dir, exist_ok=True)
    voice = detect_voice(video_path)

    results = []
    for i, (seg, text) in enumerate(zip(segments, texts)):
        target = max(seg.end - seg.start, 0.3)
        out_path = os.path.join(work_dir, f"seg_{i:04d}.wav")
        try:
            clip = synthesize_segment(text, voice, target, out_path)
            log(
                "synthesize",
                f"({i+1}/{len(segments)}) target={target:.2f}s got={len(clip)/1000:.2f}s: {text!r}",
            )
        except Exception as e:
            # Don't lose a multi-hour run to one stubborn segment -- drop in
            # silence for this line and keep going; it'll show up as a gap
            # in the final dub rather than a crash.
            log("synthesize", f"({i+1}/{len(segments)}) FAILED after retries ({e}); using silence")
            clip = AudioSegment.silent(duration=int(target * 1000))
        results.append((seg.start, clip))

    return results