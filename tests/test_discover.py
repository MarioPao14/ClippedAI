import json

import pytest

from discover import Candidate, is_clippable, parse_args, rank_long_videos, write_candidates


def candidate(video_id: str, views: int, minutes: float) -> Candidate:
    return Candidate(video_id, f"title {video_id}", "channel", views, minutes, f"https://youtu.be/{video_id}")


def test_rank_drops_videos_shorter_than_minimum():
    ranked = rank_long_videos([candidate("short", 9_000_000, 19.9), candidate("long", 10, 20)], 20)
    assert [c.video_id for c in ranked] == ["long"]


def test_rank_orders_by_views_descending():
    ranked = rank_long_videos([candidate("a", 5, 60), candidate("b", 50, 60), candidate("c", 20, 60)], 20)
    assert [c.video_id for c in ranked] == ["b", "c", "a"]


def test_clippable_when_three_frames_have_a_large_face():
    assert is_clippable([0.02, 0.015, 0.05, 0.0, 0.001])


def test_not_clippable_when_only_two_frames_have_a_large_face():
    assert not is_clippable([0.02, 0.05, 0.014, 0.0, 0.0])


def test_corner_facecam_is_not_clippable():
    assert not is_clippable([0.006, 0.007, 0.006, 0.008, 0.007])


def test_write_candidates_round_trip(tmp_path):
    path = tmp_path / "candidates.json"
    original = Candidate("abc", "Título 🔥", "Canal", 1_234_567, 95.5, "https://youtu.be/abc", 4)

    write_candidates(path, "podcast entrevista", [original])

    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload == {
        "query": "podcast entrevista",
        "candidates": [{
            "video_id": "abc", "title": "Título 🔥", "channel": "Canal", "views": 1_234_567,
            "duration_minutes": 95.5, "url": "https://youtu.be/abc", "face_frames": 4,
        }],
    }


def test_parse_args_defaults():
    args = parse_args(["--query", "podcast"])
    assert (args.top, args.min_minutes, args.search_size) == (10, 20, 40)


@pytest.mark.parametrize("flag", ["--top", "--min-minutes", "--search-size"])
def test_parse_args_rejects_non_positive(flag):
    with pytest.raises(SystemExit):
        parse_args(["--query", "q", flag, "0"])
