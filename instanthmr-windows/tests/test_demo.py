"""demo.py's argument checks and its runners, with the models stubbed out.

``build_all`` is replaced by a stub pipeline and a stub visualizer and
``cv2.VideoCapture`` by a stub that serves black frames, so the video and
camera loops run in milliseconds and no weights are needed.
"""

from __future__ import annotations

import json
import sys

import cv2
import numpy as np
import pytest

import demo
from instanthmr.pipeline import FrameResult
from tests.test_mocap import make_prediction, standing_skeleton


def parse(monkeypatch, *argv):
    monkeypatch.setattr(sys, "argv", ["demo.py", *argv])
    return demo.parse_args()


def test_parse_args_rejects_bad_export_before_any_model_loads(monkeypatch, capsys, tmp_path):
    with pytest.raises(SystemExit):
        parse(monkeypatch, "--video", "x.mp4", "--export", str(tmp_path / "no" / "such" / "clip"))
    assert "does not exist" in capsys.readouterr().err
    for folder in (str(tmp_path) + "/", str(tmp_path), ""):
        with pytest.raises(SystemExit):
            parse(monkeypatch, "--video", "x.mp4", "--export", folder)
        assert "file-name prefix" in capsys.readouterr().err, folder
    args = parse(monkeypatch, "--video", "x.mp4", "--export", str(tmp_path / "clip"))
    assert args.export == str(tmp_path / "clip")


def test_parse_args_rejects_frame_skip_below_one_and_warns_on_unused_flags(monkeypatch, capsys):
    with pytest.raises(SystemExit):
        parse(monkeypatch, "--video", "x.mp4", "--frame-skip", "0")
    assert "--frame-skip" in capsys.readouterr().err
    parse(monkeypatch, "--video", "x.mp4", "--no-floor")
    assert "do nothing without --export" in capsys.readouterr().out
    parse(monkeypatch, "--video", "x.mp4")
    assert "do nothing" not in capsys.readouterr().out


class StubCapture:
    def __init__(self, n: int, fps: float = 25.0):
        self.n, self.fps, self.i = n, fps, 0

    def isOpened(self) -> bool:
        return True

    def get(self, prop):
        return {cv2.CAP_PROP_FPS: self.fps, cv2.CAP_PROP_FRAME_COUNT: self.n}.get(prop, 640)

    def read(self):
        if self.i >= self.n:
            return False, None
        self.i += 1
        return True, np.zeros((480, 640, 3), dtype=np.uint8)

    def release(self) -> None:
        pass


class StubPipeline:
    def __init__(self, fail_at: int | None = None):
        self.calls, self.fail_at = 0, fail_at

    def predict(self, rgb):
        self.calls += 1
        if self.calls - 1 == self.fail_at:
            raise ValueError("a bad frame")
        person = make_prediction(standing_skeleton(), [100, 100, 200, 400])
        return FrameResult(persons=[person], detector_ms=1.0, hmr_ms=1.0, total_ms=2.0)


class StubViz:
    def log_frame(self, *args, **kwargs) -> float:
        return 0.0


def stub_models(monkeypatch, pipeline: StubPipeline, n_frames: int) -> None:
    monkeypatch.setattr(demo, "build_all", lambda args: (pipeline, StubViz(), None))
    monkeypatch.setattr(demo.cv2, "VideoCapture", lambda src: StubCapture(n_frames))


def test_run_video_exports_what_was_captured_before_an_error(monkeypatch, tmp_path):
    video = tmp_path / "clip.mp4"
    video.touch()
    stub_models(monkeypatch, StubPipeline(fail_at=3), n_frames=10)
    args = parse(monkeypatch, "--video", str(video), "--export", str(tmp_path / "clip"),
                 "--export-formats", "json")
    with pytest.raises(SystemExit) as e:
        demo.run_video(args)
    assert e.value.code == 1
    doc = json.loads((tmp_path / "clip.mocap.json").read_text())
    assert [f["frame"] for f in doc["tracks"][0]["frames"]] == [0, 1, 2]


def test_run_camera_time_stamps_start_at_zero(monkeypatch, tmp_path):
    stub_models(monkeypatch, StubPipeline(), n_frames=5)
    args = parse(monkeypatch, "--camera", "0", "--export", str(tmp_path / "cam"),
                 "--export-formats", "json", "--smoothing-filter")
    demo.run_camera(args)          # the stub capture runs dry, like a camera unplugged
    doc = json.loads((tmp_path / "cam.mocap.json").read_text())
    times = [f["time"] for f in doc["tracks"][0]["frames"]]
    assert len(times) == 5 and 0.0 <= times[0] < 1.0 and times == sorted(times)
