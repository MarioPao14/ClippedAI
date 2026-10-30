import json
import subprocess

import cv2
import numpy as np
import pytest

from edits import (
    BRIDGE_SAMPLES, FrameGeometry, Moment, TrackSamples, choose_ball, drop_spikes, loudness_envelope,
    moment_around, parse_args, pick_moments, plan_path, render_edit, render_vertical, scene_cuts, segment_path,
    segment_targets, track_ball, window_lefts, write_manifest,
)

WINDOW = 0.5


def envelope_with_peaks(length_seconds: float, peaks: dict) -> np.ndarray:
    envelope = np.full(int(length_seconds / WINDOW), -40.0)
    for second, loudness in peaks.items():
        envelope[int(second / WINDOW)] = loudness
    return envelope


def test_moment_without_cuts_spans_preroll_and_postroll():
    assert moment_around(60.0, [], 300.0, -5.0) == Moment(56.0, 62.0, -5.0)


def test_moment_starts_at_last_scene_cut_inside_preroll():
    assert moment_around(60.0, [55.0, 57.5, 58.5], 300.0, -5.0).start == 58.5


def test_moment_ignores_cut_too_close_to_peak():
    assert moment_around(60.0, [59.5], 300.0, -5.0).start == 56.0


def test_moment_ends_at_first_scene_cut_inside_postroll():
    assert moment_around(60.0, [61.5, 61.8], 300.0, -5.0).end == 61.5


def test_moment_is_clamped_to_video_bounds():
    assert moment_around(299.0, [], 300.0, -5.0) == Moment(295.0, 300.0, -5.0)


def test_pick_moments_puts_loudest_first():
    envelope = envelope_with_peaks(300, {60: -10.0, 120: -2.0, 200: -6.0})
    moments = pick_moments(envelope, WINDOW, [], 30.0)
    assert [round(m.start + 4) for m in moments] == [120, 200, 60]


def test_pick_moments_merges_peaks_of_the_same_play():
    envelope = envelope_with_peaks(300, {60: -2.0, 63: -3.0, 150: -5.0})
    moments = pick_moments(envelope, WINDOW, [], 30.0)
    assert [round(m.start + 4) for m in moments][:2] == [60, 150]


def test_pick_moments_respects_duration_budget():
    envelope = envelope_with_peaks(600, {s: -1.0 - s / 1000 for s in range(30, 580, 40)})
    moments = pick_moments(envelope, WINDOW, [], 30.0)
    assert sum(m.end - m.start for m in moments) <= 30.0
    assert len(moments) == 5


def test_pick_moments_skips_intro_and_outro():
    envelope = envelope_with_peaks(300, {2: 0.0, 295: 0.0, 100: -10.0})
    moments = pick_moments(envelope, WINDOW, [], 30.0)
    assert [round(m.start + 4) for m in moments] == [100]


def test_manifest_round_trip(tmp_path):
    moments = [Moment(116.0, 122.0, -2.5), Moment(56.25, 62.0, -9.0)]
    path = write_manifest(tmp_path, {"id": "abc", "title": "PSG 4-2", "url": "https://youtu.be/abc"}, moments)
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "file": "edit.mp4",
        "source": {"video_id": "abc", "title": "PSG 4-2", "url": "https://youtu.be/abc"},
        "duration": 11.75,
        "moments": [
            {"start": 116.0, "end": 122.0, "loudness": -2.5},
            {"start": 56.25, "end": 62.0, "loudness": -9.0},
        ],
    }


def test_parse_args_defaults():
    args = parse_args(["--team", "PSG"])
    assert (args.team, args.duration, args.max_age_days) == ("PSG", 30.0, 7)


@pytest.mark.parametrize("flag", ["--duration", "--max-age-days"])
def test_parse_args_rejects_non_positive(flag):
    with pytest.raises(SystemExit):
        parse_args(["--team", "PSG", flag, "0"])


@pytest.fixture(scope="module")
def synthetic_match(tmp_path_factory):
    """20 s video: quiet tone with a loud burst at 12 s, and a hard visual cut at 10 s."""
    path = tmp_path_factory.mktemp("match") / "match.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=320x180:d=10:r=25",
         "-f", "lavfi", "-i", "color=c=white:s=320x180:d=10:r=25",
         "-f", "lavfi", "-i", "sine=f=440:d=20:sample_rate=8000",
         "-filter_complex",
         "[0][1]concat=n=2:v=1:a=0[v];[2]volume='if(between(t,12,12.5),1,0.01)':eval=frame[a]",
         "-map", "[v]", "-map", "[a]", "-c:v", "libopenh264", "-c:a", "aac", str(path)],
        check=True,
    )
    return path


def test_loudness_peak_is_found_in_real_audio(synthetic_match):
    envelope = loudness_envelope(synthetic_match, WINDOW)
    assert int(np.argmax(envelope)) * WINDOW == 12.0


def test_scene_cut_is_found_in_real_video(synthetic_match):
    assert [round(cut) for cut in scene_cuts(synthetic_match)] == [10]


def probe(path) -> dict:
    output = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "stream=codec_type,width,height:format=duration",
         "-of", "json", str(path)],
        check=True, capture_output=True, text=True,
    ).stdout
    return json.loads(output)


def test_render_produces_vertical_edit_with_audio_and_moment_length(synthetic_match, tmp_path):
    output = tmp_path / "edit.mp4"
    render_edit(synthetic_match, [Moment(11.0, 14.0, -1.0), Moment(2.0, 4.0, -9.0)], output)

    info = probe(output)
    video = next(s for s in info["streams"] if s["codec_type"] == "video")
    assert (video["width"], video["height"]) == (1080, 1920)
    assert any(s["codec_type"] == "audio" for s in info["streams"])
    assert float(info["format"]["duration"]) == pytest.approx(5.0, abs=0.15)


def test_confident_ball_beats_a_nearer_weak_one():
    assert choose_ball([(860.0, 0.29), (1224.0, 0.60)], 850.0) == 1224.0


def test_weak_ball_near_last_position_bridges_a_blurry_frame():
    assert choose_ball([(400.0, 0.2), (1020.0, 0.2)], 1000.0) == 1020.0


def test_nearest_of_several_confident_balls_wins():
    assert choose_ball([(300.0, 0.9), (1050.0, 0.6)], 1000.0) == 1050.0


def test_low_confidence_jump_is_rejected():
    assert choose_ball([(70.0, 0.2)], 1300.0) is None


def test_high_confidence_jump_is_accepted():
    assert choose_ball([(70.0, 0.7)], 1300.0) == 70.0


def test_first_ball_needs_high_confidence():
    assert choose_ball([(500.0, 0.3)], None) is None


def test_weak_detections_bridge_a_short_blur_after_a_confident_one():
    track = track_ball([[(1000.0, 0.8)], [(1010.0, 0.2)], [(1020.0, 0.2)]])
    assert track.tolist() == [1000.0, 1010.0, 1020.0]


def test_weak_detections_cannot_chain_beyond_the_bridge():
    frames = [[(1000.0, 0.8)]] + [[(1000.0 + 8 * i, 0.2)] for i in range(1, BRIDGE_SAMPLES + 4)]
    track = track_ball(frames)
    assert np.isnan(track[BRIDGE_SAMPLES + 1:]).all() and not np.isnan(track[:BRIDGE_SAMPLES + 1]).any()


def test_weak_detections_alone_are_never_trusted():
    assert np.isnan(track_ball([[(500.0, 0.3)], [(505.0, 0.3)]])).all()


def samples(ball, players, frame_count=100):
    frames = np.arange(0, frame_count, 10)
    return TrackSamples(frames, np.array(ball, dtype=float), np.array(players, dtype=float), frame_count)


def test_camera_glides_straight_across_ball_gap_ignoring_players():
    nan = np.nan
    track = samples([500, nan, nan, nan, nan, nan, nan, nan, nan, 900], [100, 1800, 50, 1700, 0, 1900, 30, 1800, 40, 1700])
    targets, follows_ball = segment_targets(track, (0, 91), 1920)
    assert follows_ball
    assert np.allclose(targets, np.linspace(500, 900, 91))


def test_players_are_followed_only_when_moment_has_no_ball():
    nan = np.nan
    targets, follows_ball = segment_targets(samples([nan] * 10, [800] * 10), (0, 100), 1920)
    assert (follows_ball, targets[50]) == (False, 800.0)


def test_empty_moment_stays_centered():
    nan = np.nan
    targets, follows_ball = segment_targets(samples([nan] * 10, [nan] * 10), (0, 100), 1920)
    assert (follows_ball, targets[0]) == (False, 960.0)


def test_segment_ignores_samples_from_other_moments():
    nan = np.nan
    targets, _ = segment_targets(samples([100, 100, 100, 100, 100, 1500, nan, nan, nan, nan], [nan] * 10), (50, 100), 1920)
    assert np.allclose(targets, 1500.0)


def test_fast_ball_never_leaves_the_vertical_window():
    geometry = FrameGeometry(fps=25.0, width=1920, crop_width=608)
    ball = np.concatenate([np.full(5, 200.0), np.full(5, 1700.0)])
    track = TrackSamples(np.arange(0, 100, 10), ball, np.full(10, np.nan), 100)
    centers = segment_path(track, (0, 100), geometry)
    lefts = window_lefts(centers, geometry.width, geometry.crop_width)
    targets, _ = segment_targets(track, (0, 100), geometry.width)
    assert np.all((targets >= lefts) & (targets <= lefts + geometry.crop_width))


def test_spike_filter_drops_one_off_false_positive():
    assert drop_spikes(np.array([900.0, 905, 910, 74, 920, 925, 930])).tolist()[3] == 910.0


def test_spike_filter_keeps_a_real_fast_move():
    shot = np.array([850.0, 900, 1000, 1100, 1200, 1300, 1300, 1300, 1300])
    assert drop_spikes(shot)[5] == 1300.0


def test_spike_filter_leaves_sparse_detections_untouched():
    assert drop_spikes(np.array([500.0, 900.0])).tolist() == [500.0, 900.0]


def test_planned_path_keeps_a_still_target_still():
    assert np.allclose(plan_path(np.full(50, 640.0), 200.0, 25.0), 640.0)


def test_planned_path_flattens_jitter():
    jitter = 640.0 + np.tile([40.0, -40.0], 50)
    assert np.abs(plan_path(jitter, 200.0, 25.0) - 640.0).max() < 5.0


def test_planned_path_never_lets_target_out_of_reach():
    shot = np.concatenate([np.full(40, 300.0), np.linspace(300, 1600, 8), np.full(40, 1600.0)])
    assert np.abs(plan_path(shot, 152.0, 25.0) - shot).max() <= 152.0 + 1e-6


def test_planned_path_starts_moving_before_the_shot():
    shot = np.concatenate([np.full(40, 300.0), np.full(40, 1600.0)])
    assert plan_path(shot, 152.0, 25.0)[35] > 310.0


def test_window_never_leaves_the_frame():
    lefts = window_lefts(np.array([0.0, 960.0, 1920.0]), 1920, 608)
    assert lefts.tolist() == [0, 656, 1312]


@pytest.fixture(scope="module")
def split_frame_video(tmp_path_factory):
    """Black left half, white right half, with silent audio."""
    path = tmp_path_factory.mktemp("split") / "split.mp4"
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y",
         "-f", "lavfi", "-i", "color=c=black:s=640x360:d=2:r=25,drawbox=x=320:y=0:w=320:h=360:color=white:t=fill",
         "-f", "lavfi", "-i", "anullsrc=r=8000:cl=mono", "-t", "2",
         "-c:v", "libopenh264", "-c:a", "aac", str(path)],
        check=True,
    )
    return path


def mean_brightness(path) -> float:
    capture = cv2.VideoCapture(str(path))
    ok, frame = capture.read()
    capture.release()
    assert ok
    return float(frame.mean())


@pytest.mark.parametrize("left, expected", [(0, "dark"), (640 - 202, "bright")])
def test_vertical_render_shows_the_window_it_was_given(split_frame_video, tmp_path, left, expected):
    output = tmp_path / "vertical.mp4"
    render_vertical(split_frame_video, np.full(50, left), 202, output)
    brightness = mean_brightness(output)
    assert (brightness > 200) if expected == "bright" else (brightness < 30)
