"""Search YouTube without an API key and rank long videos whose speakers stay in the foreground."""

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import List, Optional

import yt_dlp

SAMPLE_POSITIONS = (0.15, 0.3, 0.5, 0.7, 0.85)
MIN_FACE_AREA = 0.015
MIN_FACE_FRAMES = 3


@dataclass(frozen=True)
class Candidate:
    video_id: str
    title: str
    channel: str
    views: int
    duration_minutes: float
    url: str
    face_frames: int = 0


def positive_int(value: str) -> int:
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return number


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--query", required=True, help="Search terms, in the target market's language")
    parser.add_argument("--top", type=positive_int, default=10, help="How many clippable videos to return")
    parser.add_argument("--min-minutes", type=positive_int, default=20)
    parser.add_argument("--search-size", type=positive_int, default=40, help="Search results to inspect")
    parser.add_argument("--output", type=Path, default=Path("candidates.json"))
    return parser.parse_args(argv)


def search(query: str, size: int) -> List[Candidate]:
    with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True}) as ydl:
        entries = ydl.extract_info(f"ytsearch{size}:{query}", download=False)["entries"]
    return [
        Candidate(
            video_id=entry["id"],
            title=entry.get("title") or "",
            channel=entry.get("channel") or "",
            views=entry.get("view_count") or 0,
            duration_minutes=round((entry.get("duration") or 0) / 60, 1),
            url=f"https://www.youtube.com/watch?v={entry['id']}",
        )
        for entry in entries
    ]


def rank_long_videos(candidates: List[Candidate], min_minutes: int) -> List[Candidate]:
    long_videos = [c for c in candidates if c.duration_minutes >= min_minutes]
    return sorted(long_videos, key=lambda c: c.views, reverse=True)


def is_clippable(face_areas: List[float]) -> bool:
    return sum(area >= MIN_FACE_AREA for area in face_areas) >= MIN_FACE_FRAMES


def sample_face_areas(candidate: Candidate, face_detector) -> List[float]:
    """Largest face area (fraction of the frame) at each sample position, read from the stream without downloading."""
    import cv2
    import numpy as np

    with yt_dlp.YoutubeDL({"format": "bv*[height<=360]/b[height<=360]", "quiet": True}) as ydl:
        info = ydl.extract_info(candidate.url, download=False)
    headers = "".join(f"{key}: {value}\r\n" for key, value in info.get("http_headers", {}).items())
    areas = []
    for position in SAMPLE_POSITIONS:
        frame_bytes = subprocess.run(
            ["ffmpeg", "-v", "error", "-headers", headers, "-ss", str(info["duration"] * position),
             "-i", info["url"], "-frames:v", "1", "-f", "image2pipe", "-vcodec", "png", "-"],
            check=True, capture_output=True, timeout=60,
        ).stdout
        frame = cv2.imdecode(np.frombuffer(frame_bytes, np.uint8), cv2.IMREAD_COLOR)
        detections = face_detector.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)).detections or []
        boxes = [d.location_data.relative_bounding_box for d in detections]
        areas.append(max((box.width * box.height for box in boxes), default=0.0))
    return areas


def find_clippable(ranked: List[Candidate], top: int) -> List[Candidate]:
    import mediapipe as mp

    clippable = []
    # model_selection=1 also detects small, distant faces, so a corner facecam is measured rather than missed
    with mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5) as detector:
        for candidate in ranked:
            areas = sample_face_areas(candidate, detector)
            face_frames = sum(area >= MIN_FACE_AREA for area in areas)
            print(f"{face_frames}/{len(areas)} face frames  {candidate.views:>11,}  {candidate.title[:70]}")
            if is_clippable(areas):
                clippable.append(Candidate(**{**asdict(candidate), "face_frames": face_frames}))
            if len(clippable) == top:
                break
    return clippable


def write_candidates(path: Path, query: str, candidates: List[Candidate]) -> None:
    payload = {"query": query, "candidates": [asdict(c) for c in candidates]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    ranked = rank_long_videos(search(args.query, args.search_size), args.min_minutes)
    print(f"{len(ranked)} videos >= {args.min_minutes} min; checking faces...")
    clippable = find_clippable(ranked, args.top)
    write_candidates(args.output, args.query, clippable)
    print(f"{len(clippable)} clippable candidates written to {args.output}")


if __name__ == "__main__":
    main()
