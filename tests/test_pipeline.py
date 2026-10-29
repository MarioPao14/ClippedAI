import json
from pathlib import Path
from dataclasses import dataclass

import pytest

from pipeline import parse_args, write_manifest


@dataclass(frozen=True)
class FakeClip:
    file: str
    title: str
    start: float
    end: float
    duration: float


def test_manifest_round_trip_keeps_every_field(tmp_path):
    output_dir = tmp_path / "abc123"
    output_dir.mkdir()
    clip = FakeClip(file="01 Título 🔥.mp4", title="Título 🔥", start=12.5, end=70.25, duration=57.75)

    manifest = json.loads(write_manifest(output_dir, "https://youtu.be/abc123", [clip]).read_text(encoding="utf-8"))

    assert manifest["video_id"] == "abc123"
    assert manifest["source_url"] == "https://youtu.be/abc123"
    assert manifest["clips"] == [
        {"file": "01 Título 🔥.mp4", "title": "Título 🔥", "start": 12.5, "end": 70.25, "duration": 57.75}
    ]


def test_manifest_with_no_clips_has_empty_list(tmp_path):
    manifest_path = write_manifest(tmp_path, "u", [])
    assert json.loads(manifest_path.read_text(encoding="utf-8"))["clips"] == []


def test_parse_args_reads_url_and_max_clips():
    args = parse_args(["--url", "https://youtu.be/x", "--max-clips", "3"])
    assert (args.url, args.max_clips, args.cookies) == ("https://youtu.be/x", 3, None)


def test_parse_args_reads_cookies_path():
    args = parse_args(["--url", "u", "--max-clips", "1", "--cookies", "c.txt"])
    assert args.cookies == Path("c.txt")


@pytest.mark.parametrize("value", ["0", "-1", "abc"])
def test_parse_args_rejects_invalid_max_clips(value):
    with pytest.raises(SystemExit):
        parse_args(["--url", "u", "--max-clips", value])
