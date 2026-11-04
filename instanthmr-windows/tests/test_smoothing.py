"""TemporalSmoother contracts the mocap recorder relies on: every emitted
prediction carries a track id, in every mode; foot contact passes through;
two persons keep distinct ids; a track survives ``max_age`` absent frames."""

from __future__ import annotations

import numpy as np
import pytest

from instanthmr.mocap import MocapRecorder
from instanthmr.smoothing import TemporalSmoother
from tests.test_mocap import make_prediction, standing_skeleton

BOX_A = [100, 100, 200, 400]
BOX_B = [600, 100, 700, 400]


def test_track_ids_stamped_in_every_mode():
    pose = standing_skeleton()
    for mode in ("centered", "causal", "1euro"):
        sm = TemporalSmoother(window=5, mode=mode)
        out = []
        for i in range(6):
            frame = sm.push([make_prediction(pose, BOX_A)], timestamp=i / 30.0)
            if frame is not None:
                out.extend(frame)
        out.extend(p for frame in sm.flush() for p in frame)
        assert len(out) == 6 and all(p.track_id == 0 for p in out), mode
        # A stamped stream never makes the recorder build its own tracker.
        rec = MocapRecorder(fps=30.0)
        for i, p in enumerate(out):
            rec.push(i, i / 30.0, [p])
        assert rec._tracker is None and [t.track_id for t in rec.finish()] == [0]


def test_contact_prob_passes_through():
    sm = TemporalSmoother(window=3, mode="causal")
    contact = np.array([0.9, 0.1], dtype=np.float32)
    (out,) = sm.push([make_prediction(standing_skeleton(), BOX_A, contact=contact)])
    np.testing.assert_allclose(out.contact_prob, [0.9, 0.1])


def test_two_persons_keep_distinct_ids():
    pose = standing_skeleton()
    sm = TemporalSmoother(window=3, mode="causal")
    for _ in range(10):
        out = sm.push([make_prediction(pose, BOX_A), make_prediction(pose, BOX_B)])
        assert [p.track_id for p in out] == [0, 1]


def test_track_survives_max_age_absent_frames():
    # The demo passes max_age=10 (= MocapRecorder.max_gap) so a 10-frame
    # absence keeps the id; with the default 5 the person comes back as a new
    # track. Frames 10-19 are empty.
    pose = standing_skeleton()
    for max_age, expected in ((5, {0, 1}), (10, {0})):
        sm = TemporalSmoother(window=1, mode="causal", max_age=max_age)
        ids = set()
        for i in range(30):
            present = i < 10 or i >= 20
            out = sm.push([make_prediction(pose, BOX_A)] if present else [])
            ids |= {p.track_id for p in out}
        assert ids == expected, (max_age, ids)


def test_1euro_error_leaves_nothing_to_flush():
    # demo.py flushes the smoother after an error in the frame loop. An error
    # inside the 1euro filter (here the MHR keypoint backend, on frame 2) used
    # to leave that frame counted as pushed but not emitted, and flush() then
    # raised IndexError on its empty buffer, hiding the real error.
    def keypoints(mhr_params, shape_params):
        keypoints.calls += 1
        if keypoints.calls == 2:
            raise RuntimeError("non-finite pose parameters")
        return np.zeros((70, 3), np.float32)
    keypoints.calls = 0
    sm = TemporalSmoother(mode="1euro", keypoint_fn=keypoints)
    sm.push([make_prediction(standing_skeleton(), BOX_A)], timestamp=0.0)
    with pytest.raises(RuntimeError):
        sm.push([make_prediction(standing_skeleton(), BOX_A)], timestamp=1 / 30)
    assert list(sm.flush()) == []
