#!/usr/bin/env python3
"""Automated Video Dubbing System.

Takes a YouTube URL in any spoken language and produces an English-dubbed
version: same speaker's approximate voice, same video, English audio synced
to the original timing.

Usage:
    python main.py "https://www.youtube.com/watch?v=XXXXXXXX"
    python main.py "<url>" --output-dir out --whisper-model medium --device cuda

Pipeline (see pipeline/ for each stage's implementation):
    1. download.py    yt-dlp        fetch the source video
    2. transcriber.py faster-whisper  speech -> timestamped source-language text
    3. translator.py  NLLB-200        source-language text -> English text
    4. synthesizer.py edge-tts        English text -> timed English speech
    5. remixer.py      ffmpeg         swap audio track into the original video
"""

import argparse
import os
import sys
import time

from pipeline.downloader import download_video
from pipeline.transcriber import transcribe_video
from pipeline.translator import translate_segments
from pipeline.synthesizer import synthesize_all
from pipeline.remixer import remix
from pipeline.utils import log


def parse_args():
    p = argparse.ArgumentParser(description="Dub a YouTube video into English.")
    p.add_argument("url", help="YouTube video URL")
    p.add_argument("--output-dir", default="output", help="Where to write results")
    p.add_argument(
        "--whisper-model",
        default="medium",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (bigger = more accurate, slower). "
        "'medium' is a good accuracy/speed tradeoff for a GPU machine.",
    )
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument(
        "--language",
        default=None,
        help="Force source language as an ISO 639-1 code (e.g. 'te' for "
        "Telugu, 'hi' for Hindi). Skips auto-detection, which can be "
        "unreliable on short or musically-heavy clips.",
    )
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = os.path.join(args.output_dir, "work")
    os.makedirs(work_dir, exist_ok=True)

    t0 = time.time()
    log("pipeline", f"Starting dub job for {args.url}")

    # 1. Download
    video_path = download_video(args.url, args.output_dir)

    # 2. Transcribe (original language)
    segments, source_lang = transcribe_video(
        video_path, model_size=args.whisper_model, device=args.device, language=args.language
    )
    if not segments:
        log("pipeline", "No speech detected -- nothing to dub. Exiting.")
        sys.exit(1)

    # 3. Translate to English
    english_texts = translate_segments(segments, source_lang, device=args.device)

    # 4. Synthesize timed English speech
    segments_audio = synthesize_all(segments, english_texts, video_path, work_dir)

    # 5. Remix into final video
    output_path = os.path.join(args.output_dir, "dubbed_output.mp4")
    remix(segments_audio, video_path, work_dir, output_path)

    elapsed = time.time() - t0
    log("pipeline", f"Done in {elapsed/60:.1f} min. Output: {output_path}")


if __name__ == "__main__":
    main()