"""PosePipeline's detector fallback, with stub models (no weights needed).

The detector is stubbed with a list of boxes per detector CALL, and InstantHMR
with a function returning a fixed skeleton for whatever box it is given.
"""

from __future__ import annotations

import numpy as np

import instanthmr.pipeline as pipeline_mod
from instanthmr.pipeline import PosePipeline
from tests.test_mocap import make_prediction, standing_skeleton

BOX = [100.0, 100.0, 200.0, 400.0]
IMAGE = np.zeros((480, 640, 3), dtype=np.uint8)


class StubDetector:
    def __init__(self, boxes_per_call: list[list[list[float]]]):
        self.boxes_per_call = boxes_per_call
        self.calls = 0

    def warmup(self) -> None:
        pass

    def detect(self, image_rgb):
        boxes = self.boxes_per_call[self.calls]
        self.calls += 1
        return [{"bbox": np.asarray(b, dtype=np.float32), "confidence": 0.9} for b in boxes]


class StubHMR:
    derives_joints_3d = False
    has_prev_input = False

    def __init__(self, *args, **kwargs):
        pass

    def warmup(self, max_batch_size: int) -> None:
        pass

    def predict(self, image_rgb, bbox, confidence, prev=None):
        return make_prediction(standing_skeleton(), bbox, confidence=confidence)


def make_pipeline(monkeypatch, boxes_per_call, **kwargs) -> PosePipeline:
    monkeypatch.setattr(pipeline_mod, "InstantHMR", StubHMR)
    monkeypatch.setattr(pipeline_mod, "RFDETRDetector",
                        lambda **kw: StubDetector(boxes_per_call))
    return PosePipeline("stub.onnx", max_persons=1, **kwargs)


def test_detector_miss_is_a_miss_not_a_ghost(monkeypatch):
    # Stride 1: the detector runs every frame. A person seen on frame 0 and
    # gone from frame 1 on must yield no persons from frame 1 on; the old
    # box must NOT be re-fed to HMR (that produced a confident-looking
    # skeleton for an empty crop until the end of the clip).
    pipe = make_pipeline(monkeypatch, [[BOX], [], [], []])
    assert len(pipe.predict(IMAGE).persons) == 1
    for _ in range(3):
        assert pipe.predict(IMAGE).persons == []


def test_fallback_bridges_only_stride_skipped_frames(monkeypatch):
    # Stride 2: frame 1 is skipped, so frame 0's box is reused there (expanded,
    # confidence x0.9); on frame 2 the detector runs and sees nobody, and the
    # skipped frame 3 must not resurrect the box either.
    pipe = make_pipeline(monkeypatch, [[BOX], []], detector_stride=2)
    first = pipe.predict(IMAGE).persons
    bridged = pipe.predict(IMAGE).persons
    assert len(first) == 1 and len(bridged) == 1
    assert bridged[0].confidence == np.float32(0.9) * np.float32(0.9) or \
        abs(bridged[0].confidence - 0.81) < 1e-6
    assert bridged[0].bbox[0] < first[0].bbox[0] and bridged[0].bbox[2] > first[0].bbox[2]
    assert pipe.predict(IMAGE).persons == []
    assert pipe.predict(IMAGE).persons == []
