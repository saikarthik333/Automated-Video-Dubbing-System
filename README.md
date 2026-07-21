# Automated Video Dubbing System

Takes a YouTube URL in any spoken language and produces an English-dubbed
version: same video, English audio synced to the original timing, in an
approximately matching voice.

## Structure
```
video_dubbing/
├── main.py                 # CLI entrypoint — orchestrates all 5 stages
├── requirements.txt         # All Python dependencies, pinned
├── README.md                 # Architecture + setup docs
└── pipeline/
    ├── __init__.py
    ├── downloader.py         # Stage 1: yt-dlp download
    ├── transcriber.py         # Stage 2: faster-whisper transcription
    ├── translator.py          # Stage 3: NLLB-200 translation
    ├── synthesizer.py          # Stage 4: edge-tts speech synthesis
    ├── remixer.py               # Stage 5: ffmpeg audio/video remux
    └── utils.py                  # Shared helpers (logging, time formatting)
```

## Architecture

```
YouTube URL
   |
   v
[1] downloader.py   yt-dlp        -> source.mp4
   |
   v
[2] transcriber.py  faster-whisper -> [Segment(start, end, text), ...] + source language
   |
   v
[3] translator.py   NLLB-200       -> English text per segment
   |
   v
[4] synthesizer.py  edge-tts       -> timed English audio clips per segment
   |
   v
[5] remixer.py      ffmpeg         -> dubbed_output.mp4  (video stream copied, audio replaced)
```

Each stage is a separate module under `pipeline/` with a single clear
responsibility, so any piece can be swapped independently (e.g. a different
TTS engine, or IndicTrans2 instead of NLLB for Indic languages) without
touching the rest.

## Key design decisions

- **faster-whisper over openai-whisper**: same accuracy, noticeably faster
  and lighter on GPU memory (CTranslate2 backend). VAD filtering is enabled
  to skip silent gaps, which avoids Whisper's tendency to hallucinate text
  in silence.
- **Transcribe-then-translate, not Whisper's built-in `translate` task**:
  keeping transcription and translation as separate stages gives an
  intermediate source-language transcript (useful for debugging/QA) and
  lets translation be evaluated and improved independently of ASR quality.
- **NLLB-200 for translation**: one open-source model covers ~200
  languages, so German, French, Hindi, etc. all go through the same code
  path with no per-language branching. It translates full sentences, not
  word-by-word, so output reads naturally rather than literally.
  For Indian-language sources specifically, AI4Bharat's **IndicTrans2**
  is purpose-built and can outperform NLLB -- see the swap-in note at the
  top of `translator.py`.
- **edge-tts for synthesis**: free, no API key, and the neural voices sound
  natural rather than robotic.
- **Voice matching (lightweight)**: rather than full voice cloning, we
  estimate the source speaker's median pitch (via `librosa.pyin`) once per
  video and pick a male or female neural voice accordingly. This is a
  reasonable approximation for a single-speaker video; true voice cloning
  (e.g. Coqui XTTS) is listed as a stretch goal in the brief and is not
  attempted here to keep the core pipeline solid.
- **Timing sync via rate adjustment**: each segment's English line is
  synthesized once at normal speaking rate, measured, then re-synthesized
  with a speed adjustment (`edge-tts --rate`) so it fits the original
  segment's duration, bounded to -40%/+60% so speech doesn't get
  distorted. Clips are placed on a silent full-length timeline at their
  original start timestamps before muxing, so overall sync tracks the
  source even where per-line matching isn't exact.
- **`-c:v copy` on remux**: the video stream is never re-encoded, only the
  audio is replaced, so this step is fast and there's no quality loss to
  the picture.

## Setup

```bash
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
# ffmpeg must also be installed on the system (see requirements.txt)
```

## Usage

```bash
python main.py "https://www.youtube.com/watch?v=XXXXXXXX"

# with options
python main.py "<url>" \
  --output-dir out \
  --whisper-model medium \
  --device cuda
```

Output:
- `output/source.mp4` -- the original downloaded video
- `output/dubbed_output.mp4` -- the final English-dubbed video
- `output/work/` -- intermediate per-segment audio clips (kept for debugging)

Progress for every stage (download, transcription per segment, translation
per segment, synthesis per segment, remux) is printed to the terminal with
elapsed-time stamps.

## Known limitations / what I'd improve next

- Single voice for the whole video -- no multi-speaker diarization or
  per-speaker voice assignment yet (this is the brief's stretch goal).
- Duration matching is a single-pass rate adjustment, not an iterative
  fit -- very long translated sentences relative to short original clips
  can still hit the rate cap and drift slightly out of sync.
- NLLB-200-distilled-600M is the smaller/faster variant; the 1.3B or 3.3B
  variants would likely improve translation quality at the cost of speed.
