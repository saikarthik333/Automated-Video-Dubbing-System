"""Step 1b: transcribe speech in the source video, in its original language.

Uses faster-whisper (a CTranslate2 reimplementation of OpenAI Whisper) because
it is significantly faster and lighter on GPU memory than the original
openai-whisper package, while producing the same accuracy. It also gives us
segment-level timestamps for free, which we need later to time-align the
dubbed audio.
"""

import subprocess
from dataclasses import dataclass
from typing import List, Optional

from faster_whisper import WhisperModel

from .utils import log, format_hhmmss


@dataclass
class Segment:
    start: float
    end: float
    text: str


def _media_duration_seconds(path: str) -> float:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", path,
        ],
        capture_output=True, text=True, check=True,
    )
    return float(result.stdout.strip())


def transcribe_video(
    video_path: str,
    model_size: str = "medium",
    device: str = "cuda",
    compute_type: str = "float16",
    language: Optional[str] = None,
) -> tuple[List[Segment], str]:
    """Transcribe `video_path` and return (segments, detected_language_code).

    faster-whisper reads audio directly out of the video container via
    ffmpeg, so no separate audio extraction step is needed.

    `language`: force a specific ISO 639-1 code (e.g. "te") instead of
    relying on auto-detection, useful if the source has background music
    that confuses language ID on a short clip.
    """
    log("transcribe", f"Loading Whisper model '{model_size}' on {device}")
    try:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception as e:
        # Graceful fallback if no compatible GPU/CTranslate2 build is found.
        log("transcribe", f"GPU load failed ({e}); falling back to CPU int8")
        model = WhisperModel(model_size, device="cpu", compute_type="int8")

    log("transcribe", "Running speech recognition (this can take a while)...")
    raw_segments, info = model.transcribe(
        video_path,
        language=language,
        vad_filter=True,  # skip silence, avoids hallucinated text in gaps
        vad_parameters={"min_silence_duration_ms": 300},
        # condition_on_previous_text=True (the default) can send Whisper into
        # a hallucination/repetition loop on noisy audio (movie background
        # score, sound effects), which can cause it to stall and stop
        # yielding segments well before the audio actually ends. Disabling
        # it makes each segment decode independently, trading a little
        # cross-segment context for much more reliable full-length coverage.
        condition_on_previous_text=False,
        beam_size=5,
        best_of=5,
        # Discourage the kind of repeated-phrase hallucination visible in
        # noisy/musical audio.
        repetition_penalty=1.1,
        no_repeat_ngram_size=3,
    )

    segments: List[Segment] = []
    for seg in raw_segments:
        text = seg.text.strip()
        if not text:
            continue
        segments.append(Segment(start=seg.start, end=seg.end, text=text))
        log(
            "transcribe",
            f"[{format_hhmmss(seg.start)} -> {format_hhmmss(seg.end)}] {text}",
        )

    log(
        "transcribe",
        f"Detected language: {info.language} (p={info.language_probability:.2f}), "
        f"{len(segments)} segments",
    )

    # Sanity check: warn loudly if transcription stopped well short of the
    # actual media length -- this is the single biggest cause of "the dub
    # only covers the first N seconds" bugs.
    try:
        total_duration = _media_duration_seconds(video_path)
        covered = segments[-1].end if segments else 0.0
        if total_duration > 0 and covered < total_duration * 0.9:
            log(
                "transcribe",
                f"WARNING: transcription only covers {covered:.1f}s of "
                f"{total_duration:.1f}s total ({covered/total_duration*100:.0f}%). "
                f"Try a larger --whisper-model or a cleaner (less background "
                f"music/SFX) source video for full coverage.",
            )
    except Exception:
        pass  # duration check is a diagnostic nicety, not critical path

    return segments, info.language