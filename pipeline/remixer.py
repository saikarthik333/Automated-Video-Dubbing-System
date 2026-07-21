"""Step 4: assemble the dubbed audio timeline and swap it into the video.

We place each synthesized segment at its original start timestamp (silence
elsewhere), then use ffmpeg with `-c:v copy` to replace only the audio
stream -- the video is remuxed, not re-encoded, so this step is fast and
lossless for the picture.
"""

import os
import subprocess
from typing import List, Tuple

from pydub import AudioSegment

from .utils import log


def build_audio_timeline(
    segments_audio: List[Tuple[float, AudioSegment]], total_duration_ms: int
) -> AudioSegment:
    """Overlay each (start_time_seconds, clip) onto a silent track of the
    full video's length, so every clip lands at the right timestamp.
    """
    timeline = AudioSegment.silent(duration=total_duration_ms, frame_rate=24000)
    for start_sec, clip in segments_audio:
        position_ms = int(start_sec * 1000)
        timeline = timeline.overlay(clip, position=position_ms)
    return timeline


def get_video_duration_ms(video_path: str) -> int:
    result = subprocess.run(
        [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", video_path,
        ],
        capture_output=True, text=True, check=True,
    )
    return int(float(result.stdout.strip()) * 1000)


def mux_final_video(video_path: str, dubbed_audio_path: str, output_path: str) -> None:
    """Replace the audio track of `video_path` with `dubbed_audio_path`,
    copying the video stream untouched.
    """
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-i", dubbed_audio_path,
        "-map", "0:v:0", "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "192k",
        "-shortest",
        output_path,
    ]
    log("remix", "Running ffmpeg to mux dubbed audio into final video...")
    subprocess.run(cmd, check=True, capture_output=True)
    log("remix", f"Final dubbed video written to {output_path}")


def remix(
    segments_audio: List[Tuple[float, AudioSegment]],
    video_path: str,
    work_dir: str,
    output_path: str,
) -> str:
    duration_ms = get_video_duration_ms(video_path)
    log("remix", f"Building {duration_ms/1000:.1f}s audio timeline from {len(segments_audio)} clips")
    timeline = build_audio_timeline(segments_audio, duration_ms)

    dubbed_audio_path = os.path.join(work_dir, "dubbed_audio.wav")
    timeline.export(dubbed_audio_path, format="wav")

    mux_final_video(video_path, dubbed_audio_path, output_path)
    return output_path
