"""Download a YouTube video, cut it into 9:16 shorts with ClippedAI and write a manifest."""

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import TYPE_CHECKING, List, Optional

import yt_dlp

if TYPE_CHECKING:
    from main import ClipResult

INPUT_DIR = Path("input")
OUTPUT_DIR = Path("output")


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def parse_args(argv: List[str] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="YouTube video URL")
    parser.add_argument("--max-clips", type=positive_int, required=True)
    parser.add_argument("--cookies", type=Path, help="Netscape cookies file for yt-dlp")
    return parser.parse_args(argv)


def download_video(url: str, cookies: Optional[Path]) -> Path:
    options = {
        "format": "bv*[ext=mp4][height<=1080]+ba[ext=m4a]/b[ext=mp4]",
        "merge_output_format": "mp4",
        "outtmpl": str(INPUT_DIR / "%(id)s.%(ext)s"),
        "noplaylist": True,
    }
    if cookies:
        options["cookiefile"] = str(cookies)
    with yt_dlp.YoutubeDL(options) as ydl:
        info = ydl.extract_info(url, download=True)
    return INPUT_DIR / f"{info['id']}.mp4"


def write_manifest(output_dir: Path, source_url: str, clips: List["ClipResult"]) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "video_id": output_dir.name,
        "source_url": source_url,
        "clips": [asdict(clip) for clip in clips],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run(url: str, max_clips: int, cookies: Optional[Path]) -> Path:
    # Imported lazily: main loads torch and downloads NLTK data at import time
    from main import VideoJob, get_transcription_file_path, process_video

    video_path = download_video(url, cookies)
    output_dir = OUTPUT_DIR / video_path.stem
    job = VideoJob(
        input_path=str(video_path.resolve()),
        transcription_path=get_transcription_file_path(str(video_path)),
        max_clips=max_clips,
        output_dir=str(output_dir),
    )
    clips = process_video(job)
    if not clips:
        raise RuntimeError(f"No clips produced for {url}")
    return write_manifest(output_dir, url, clips)


if __name__ == "__main__":
    args = parse_args()
    print(f"Manifest written to {run(args.url, args.max_clips, args.cookies)}")
