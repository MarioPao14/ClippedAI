"""Turn a team's latest match highlights into a vertical edit of its loudest moments, strongest first,
with a 9:16 window that follows the ball."""

import argparse
import json
import re
import subprocess
from dataclasses import asdict, dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import List, Optional, Tuple
from urllib.parse import quote_plus

import cv2
import numpy as np
import yt_dlp
from scipy import sparse
from scipy.sparse.linalg import spsolve

from pipeline import OUTPUT_DIR, download_video

SAMPLE_RATE = 8000
WINDOW_SECONDS = 0.5
PRE_ROLL = 4.0
POST_ROLL = 2.0
MIN_SHOT_AROUND_PEAK = 1.0
SAME_PLAY_GAP = 10.0
MIN_PROMINENCE_DB = 6.0
SKIP_INTRO = 5.0
SKIP_OUTRO = 10.0
SOURCE_MIN_SECONDS = 120
SOURCE_MAX_SECONDS = 900
SCENE_THRESHOLD = 0.35
FFMPEG_THREADS = "2"
# Fedora's ffmpeg-free ships without libx264; OpenH264 is the H.264 encoder it does include
VIDEO_ENCODER = "libopenh264"
OUT_WIDTH, OUT_HEIGHT = 1080, 1920
# Roboflow's football ball detector (github.com/roboflow/sports): ~3x more confident hits than generic YOLO on
# broadcast shots and ~5x fewer false positives, but ~4 s per frame on CPU, hence the sampling step
BALL_MODEL = Path("models/football-ball-detection.pt")
BALL_MODEL_DRIVE_ID = "1isw4wx-MK9h9LMr36VvIWlJD6ppUvw7V"
BALL_IMAGE_SIZE = 1280
BALL_CLASS = 0
TRACK_STEP = 4
PLAYER_MODEL = "yolov8n.pt"
PLAYER_IMAGE_SIZE = 640
PERSON = 0
MIN_DETECTION_CONF = 0.1
REACQUIRE_CONF = 0.5
MAX_BALL_JUMP = 350.0
BRIDGE_SAMPLES = 2
# Ball detections in a window this many samples long are median filtered to drop one-off false positives
SPIKE_FILTER_SAMPLES = 5
# Camera smoothness: oscillations shorter than this are flattened out of the planned path
SMOOTHING_SECONDS = 1.5
# Fraction of the crop width kept between the ball and the edge of the vertical window
EDGE_MARGIN = 0.25
PLAN_ITERATIONS = 20


@dataclass(frozen=True)
class Moment:
    start: float
    end: float
    loudness: float


def positive_number(value: str) -> float:
    number = float(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be > 0")
    return number


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--team", required=True, help='Team to search highlights for, e.g. "PSG"')
    parser.add_argument("--duration", type=positive_number, default=30.0, help="Max edit length in seconds")
    parser.add_argument("--max-age-days", type=lambda v: int(positive_number(v)), default=7)
    return parser.parse_args(argv)


def find_highlights(team: str, max_age_days: int) -> dict:
    """Most viewed recent highlights video of the team; the flat search has no dates, so each is checked in turn."""
    search_url = f"https://www.youtube.com/results?search_query={quote_plus(team + ' highlights')}&sp=CAI%253D"
    with yt_dlp.YoutubeDL({"extract_flat": True, "quiet": True, "playlistend": 20}) as ydl:
        entries = ydl.extract_info(search_url, download=False)["entries"]
    videos = [e for e in entries if SOURCE_MIN_SECONDS <= (e.get("duration") or 0) <= SOURCE_MAX_SECONDS]
    cutoff = (date.today() - timedelta(days=max_age_days)).strftime("%Y%m%d")
    with yt_dlp.YoutubeDL({"quiet": True}) as ydl:
        for entry in sorted(videos, key=lambda e: e.get("view_count") or 0, reverse=True):
            url = f"https://www.youtube.com/watch?v={entry['id']}"
            info = ydl.extract_info(url, download=False, process=False)
            if (info.get("upload_date") or "") >= cutoff:
                return {"id": entry["id"], "title": entry.get("title") or "", "url": url}
    raise RuntimeError(f"No {team} highlights between 2 and 15 minutes from the last {max_age_days} days")


def loudness_envelope(source: Path, window_seconds: float) -> np.ndarray:
    """RMS loudness in dB per window of the mono audio track."""
    pcm = subprocess.run(
        ["ffmpeg", "-v", "error", "-threads", FFMPEG_THREADS, "-i", str(source),
         "-vn", "-ac", "1", "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"],
        check=True, capture_output=True,
    ).stdout
    samples = np.frombuffer(pcm, np.int16).astype(np.float64)
    window = int(SAMPLE_RATE * window_seconds)
    frames = samples[: len(samples) // window * window].reshape(-1, window)
    return 20 * np.log10(np.sqrt((frames ** 2).mean(axis=1)) + 1e-9)


def scene_cuts(source: Path) -> List[float]:
    result = subprocess.run(
        ["ffmpeg", "-v", "info", "-threads", FFMPEG_THREADS, "-i", str(source), "-an",
         "-vf", f"scale=320:-2,select='gt(scene,{SCENE_THRESHOLD})',showinfo", "-f", "null", "-"],
        check=True, capture_output=True, text=True,
    )
    return [float(t) for t in re.findall(r"pts_time:([\d.]+)", result.stderr)]


def moment_around(peak: float, cuts: List[float], video_length: float, loudness: float) -> Moment:
    """Window around a peak, trimmed to the shot boundaries so the edit never jumps mid-shot."""
    starts = [c for c in cuts if peak - PRE_ROLL <= c <= peak - MIN_SHOT_AROUND_PEAK]
    ends = [c for c in cuts if peak + MIN_SHOT_AROUND_PEAK <= c <= peak + POST_ROLL]
    start = max(starts, default=peak - PRE_ROLL)
    end = min(ends, default=peak + POST_ROLL)
    return Moment(max(start, 0.0), min(end, video_length), loudness)


def pick_moments(envelope: np.ndarray, window_seconds: float, cuts: List[float], budget: float) -> List[Moment]:
    video_length = len(envelope) * window_seconds
    threshold = float(np.median(envelope)) + MIN_PROMINENCE_DB
    peaks: List[float] = []
    moments: List[Moment] = []
    for index in np.argsort(envelope)[::-1]:
        loudness = float(envelope[index])
        if loudness < threshold:
            break
        peak = index * window_seconds
        if peak < SKIP_INTRO or peak > video_length - SKIP_OUTRO or any(abs(peak - p) < SAME_PLAY_GAP for p in peaks):
            continue
        moment = moment_around(peak, cuts, video_length, loudness)
        if sum(m.end - m.start for m in moments) + moment.end - moment.start > budget:
            break
        peaks.append(peak)
        moments.append(moment)
    return moments


@dataclass(frozen=True)
class TrackSamples:
    """Per sampled frame: tracked ball x and median player x, NaN where not found."""
    frames: np.ndarray
    ball_x: np.ndarray
    players_x: np.ndarray
    frame_count: int


@dataclass(frozen=True)
class FrameGeometry:
    fps: float
    width: int
    crop_width: int


def render_edit(source: Path, moments: List[Moment], output_path: Path) -> None:
    concat_path = output_path.with_name("concat.mp4")
    concat_moments(source, moments, concat_path)
    capture = cv2.VideoCapture(str(concat_path))
    frame_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    geometry = FrameGeometry(capture.get(cv2.CAP_PROP_FPS), int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
                             round(frame_height * 9 / 16 / 2) * 2)
    capture.release()
    samples = track_samples(concat_path)
    # Frame counts per moment can drift by one from rounding; the last segment absorbs the difference
    durations = [round((m.end - m.start) * geometry.fps) for m in moments]
    bounds = np.minimum(np.cumsum([0] + durations), samples.frame_count)
    bounds[-1] = samples.frame_count
    centers = np.concatenate([
        segment_path(samples, (a, b), geometry) for a, b in zip(bounds[:-1], bounds[1:]) if b > a
    ])
    lefts = window_lefts(centers, geometry.width, geometry.crop_width)
    render_vertical(concat_path, lefts, geometry.crop_width, output_path)
    concat_path.unlink()


def concat_moments(source: Path, moments: List[Moment], output_path: Path) -> None:
    inputs = []
    for m in moments:
        inputs += ["-ss", f"{m.start:.3f}", "-t", f"{m.end - m.start:.3f}", "-i", str(source)]
    count = len(moments)
    graph = "".join(f"[{i}:v][{i}:a]" for i in range(count)) + f"concat=n={count}:v=1:a=1[v][a]"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex", graph, "-map", "[v]", "-map", "[a]",
         "-c:v", VIDEO_ENCODER, "-b:v", "12M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-threads", FFMPEG_THREADS,
         str(output_path)],
        check=True,
    )


def track_samples(video: Path) -> TrackSamples:
    """Stage 1: ball and player positions every TRACK_STEP frames, so the camera is planned knowing the whole play."""
    import torch
    from ultralytics import YOLO

    # Torch grabs every core by default, which starves the desktop on a 4-core machine
    torch.set_num_threads(int(FFMPEG_THREADS))
    if not BALL_MODEL.exists():
        raise FileNotFoundError(
            f"{BALL_MODEL} missing. Download it from the link in roboflow/sports examples/soccer/setup.sh: "
            f'gdown -O {BALL_MODEL} "https://drive.google.com/uc?id={BALL_MODEL_DRIVE_ID}"'
        )
    ball_model, player_model = YOLO(str(BALL_MODEL)), YOLO(PLAYER_MODEL)
    capture = cv2.VideoCapture(str(video))
    ball_candidates, players_xs, index = [], [], 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        if index % TRACK_STEP == 0:
            balls = ball_model.predict(frame, imgsz=BALL_IMAGE_SIZE, conf=MIN_DETECTION_CONF,
                                       classes=[BALL_CLASS], verbose=False)[0].boxes
            ball_candidates.append(list(zip(balls.xywh[:, 0].tolist(), balls.conf.tolist())))
            players = player_model.predict(frame, imgsz=PLAYER_IMAGE_SIZE, classes=[PERSON], verbose=False)[0].boxes
            players_xs.append(float(np.median(players.xywh[:, 0].tolist())) if len(players) else np.nan)
        index += 1
    capture.release()
    sampled = np.arange(0, index, TRACK_STEP)
    return TrackSamples(sampled, track_ball(ball_candidates), np.array(players_xs), index)


def track_ball(candidates_per_sample: List[List[Tuple[float, float]]]) -> np.ndarray:
    """Ball x per sample, NaN where unknown. Weak detections may only bridge a short blur after a confident one;
    letting them chain on their own makes the tracker follow a white boot for half a second."""
    xs, last_ball, since_confident = [], None, BRIDGE_SAMPLES
    for balls in candidates_per_sample:
        ball = choose_ball(balls, last_ball if since_confident < BRIDGE_SAMPLES else None)
        since_confident = 0 if any(conf >= REACQUIRE_CONF for _, conf in balls) else since_confident + 1
        last_ball = ball if ball is not None else last_ball
        xs.append(np.nan if ball is None else ball)
    return np.array(xs, dtype=float)


def choose_ball(balls: List[Tuple[float, float]], last_ball: Optional[float]) -> Optional[float]:
    """A confident detection always wins; weak ones only bridge blurry frames near the last ball.
    Preferring the nearest weak one instead lets the tracker lock onto a sock or a line for seconds."""
    strong = [x for x, conf in balls if conf >= REACQUIRE_CONF]
    if strong:
        return strong[0] if last_ball is None else min(strong, key=lambda x: abs(x - last_ball))
    near = [x for x, _ in balls if last_ball is not None and abs(x - last_ball) <= MAX_BALL_JUMP]
    return min(near, key=lambda x: abs(x - last_ball)) if near else None


def segment_path(samples: TrackSamples, bounds: Tuple[int, int], geometry: FrameGeometry) -> np.ndarray:
    """Stage 2: camera center for every frame of one moment, never planned across the cut into the next."""
    targets, follows_ball = segment_targets(samples, bounds, geometry.width)
    reach = geometry.crop_width / 2 - geometry.crop_width * EDGE_MARGIN if follows_ball else np.inf
    return plan_path(targets, reach, geometry.fps)


def segment_targets(samples: TrackSamples, bounds: Tuple[int, int], frame_width: int) -> Tuple[np.ndarray, bool]:
    """Ball if it appears anywhere in the moment, gliding straight across the gaps where it is out of shot;
    players only when the ball is never found, since their median jumps as people enter and leave the shot."""
    start, end = bounds
    in_segment = (samples.frames >= start) & (samples.frames < end)
    frames = np.arange(start, end)
    for values, follows_ball in ((samples.ball_x, True), (samples.players_x, False)):
        known = in_segment & ~np.isnan(values)
        if known.any():
            return np.interp(frames, samples.frames[known], drop_spikes(values[known])), follows_ball
    return np.full(end - start, frame_width / 2), False


def drop_spikes(values: np.ndarray) -> np.ndarray:
    # With fewer detections than the window, a median would average real positions together
    if len(values) < SPIKE_FILTER_SAMPLES:
        return values
    half = SPIKE_FILTER_SAMPLES // 2
    return np.array([np.median(values[max(0, i - half):i + half + 1]) for i in range(len(values))])


def plan_path(targets: np.ndarray, reach: float, fps: float) -> np.ndarray:
    """Smoothest camera path that keeps every target within reach of the center, planned over the whole moment.

    Whittaker smoother: minimizes squared distance to the targets plus lambda times squared second differences.
    Frames where the result would leave the target out of reach get their weight raised until it fits."""
    count = len(targets)
    if count < 3:
        return targets.copy()
    lam = (fps * SMOOTHING_SECONDS / (2 * np.pi)) ** 4
    second_diff = sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(count - 2, count))
    penalty = lam * (second_diff.T @ second_diff)
    weights = np.ones(count)
    for _ in range(PLAN_ITERATIONS):
        path = spsolve((sparse.diags(weights) + penalty).tocsc(), weights * targets)
        outside = np.abs(path - targets) > reach
        if not outside.any():
            return path
        weights[outside] *= 10
    return np.clip(path, targets - reach, targets + reach)


def window_lefts(centers: np.ndarray, frame_width: int, crop_width: int) -> np.ndarray:
    return np.clip(np.round(centers - crop_width / 2), 0, frame_width - crop_width).astype(int)


def render_vertical(video: Path, lefts: np.ndarray, crop_width: int, output_path: Path) -> None:
    """Stage 3: crop every frame where the planned path says and encode the vertical edit."""
    capture = cv2.VideoCapture(str(video))
    fps = capture.get(cv2.CAP_PROP_FPS)
    encoder = subprocess.Popen(
        ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{OUT_WIDTH}x{OUT_HEIGHT}",
         "-r", str(fps), "-i", "-", "-i", str(video), "-map", "0:v", "-map", "1:a",
         "-c:v", VIDEO_ENCODER, "-b:v", "8M", "-pix_fmt", "yuv420p", "-c:a", "aac", "-threads", FFMPEG_THREADS,
         "-shortest", "-movflags", "+faststart", str(output_path)],
        stdin=subprocess.PIPE,
    )
    for left in lefts:
        ok, frame = capture.read()
        if not ok:
            break
        crop = frame[:, left:left + crop_width]
        encoder.stdin.write(cv2.resize(crop, (OUT_WIDTH, OUT_HEIGHT), interpolation=cv2.INTER_LANCZOS4).tobytes())
    capture.release()
    encoder.stdin.close()
    if encoder.wait() != 0:
        raise RuntimeError(f"ffmpeg failed to encode {output_path}")


def write_manifest(output_dir: Path, video: dict, moments: List[Moment]) -> Path:
    manifest_path = output_dir / "manifest.json"
    manifest = {
        "file": "edit.mp4",
        "source": {"video_id": video["id"], "title": video["title"], "url": video["url"]},
        "duration": round(sum(m.end - m.start for m in moments), 3),
        "moments": [asdict(m) for m in moments],
    }
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest_path


def run(team: str, budget: float, max_age_days: int) -> Path:
    video = find_highlights(team, max_age_days)
    print(f"Source: {video['title']} ({video['url']})")
    source = download_video(video["url"], None)
    moments = pick_moments(loudness_envelope(source, WINDOW_SECONDS), WINDOW_SECONDS, scene_cuts(source), budget)
    if not moments:
        raise RuntimeError(f"No moment stands out {MIN_PROMINENCE_DB} dB above the rest of {video['url']}")
    output_dir = OUTPUT_DIR / video["id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    render_edit(source, moments, output_dir / "edit.mp4")
    return write_manifest(output_dir, video, moments)


if __name__ == "__main__":
    args = parse_args()
    print(f"Manifest written to {run(args.team, args.duration, args.max_age_days)}")
