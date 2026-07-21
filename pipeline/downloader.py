"""Step 1a: download the source video from a YouTube URL with yt-dlp.

yt-dlp is preferred over pytube etc. because it's actively maintained and
handles YouTube's frequent player/format changes much more reliably.
"""

import os
import yt_dlp

from .utils import log


def download_video(url: str, output_dir: str) -> str:
    """Download the best available mp4 (video+audio muxed) for `url`.

    Returns the local filepath of the downloaded video.
    """
    os.makedirs(output_dir, exist_ok=True)
    out_template = os.path.join(output_dir, "source.%(ext)s")

    ydl_opts = {
        # best mp4 video + best m4a audio, falling back to best single file.
        "format": "bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best",
        "outtmpl": out_template,
        "merge_output_format": "mp4",
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
    }

    log("download", f"Fetching info for {url}")
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        filepath = ydl.prepare_filename(info)
        # merge_output_format can change the extension after download
        base, _ = os.path.splitext(filepath)
        mp4_path = base + ".mp4"
        if os.path.exists(mp4_path):
            filepath = mp4_path

    duration = info.get("duration", 0)
    log("download", f"Done: '{info.get('title')}' ({duration // 60:.0f} min) -> {filepath}")
    return filepath
