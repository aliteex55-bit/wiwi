"""Tests for instanthmr.mocap on synthetic skeletons with a known ground truth.

Every skeleton here is built by forward kinematics from
``mocap_skeleton.REST_DIRECTIONS`` and a table of plausible bone lengths, so
bone lengths, floor height and foot positions are known exactly. No model
files are needed.
"""

from __future__ import annotations

import csv
import json

import numpy as np
import pytest

from instanthmr.inference import HMRPrediction
from instanthmr.mocap import (
    MocapRecorder, _rotation_about_x, anchor_root, estimate_floor, estimate_pitch,
    fill_gaps, normalise_bone_lengths, planted_feet, to_world,
)
from instanthmr.mocap_skeleton import (
    CAM_TO_WORLD, FOOT_NODES, NODE_NAMES, NUM_NODES, PARENTS, REST_DIRECTIONS, ROOT,
    TOPOLOGICAL_ORDER, add_root, bone_lengths,
)

# Plausible bone length (metres) of the bone ending at each node.
LENGTHS = np.zeros(NUM_NODES)
for _nodes, _len in [
    ((9, 10), 0.10),                     # hips -> hip
    ((11, 12), 0.42),                    # thigh
    ((13, 14), 0.42),                    # shin
    ((17, 20), 0.08),                    # heel
    ((15, 18), 0.20),                    # big toe
    ((16, 19), 0.18),                    # small toe
    ((69,), 0.50),                       # spine (hips -> neck)
    ((0,), 0.15),                        # neck -> nose
    ((1, 2), 0.04), ((3, 4), 0.08),      # eyes, ears
    ((5, 6), 0.18),                      # neck -> shoulder
    ((67, 68), 0.05),                    # acromion
    ((7, 8), 0.28),                      # upper arm
    ((63, 64, 65, 66), 0.03),            # elbow landmarks
    ((62, 41), 0.26),                    # forearm
    (tuple(range(21, 41)) + tuple(range(42, 62)), 0.03),   # finger segments
]:
    LENGTHS[list(_nodes)] = _len
assert all(LENGTHS[n] > 0 for n in range(NUM_NODES) if n != ROOT)


def forward_kinematics(lengths: np.ndarray, directions: np.ndarray = REST_DIRECTIONS,
                       root=(0.0, 0.0, 0.0)) -> np.ndarray:
    """``(71, 3)`` skeleton: each node = parent + direction * length."""
    pos = np.zeros((NUM_NODES, 3))
    pos[ROOT] = root
    for node in TOPOLOGICAL_ORDER[1:]:
        pos[node] = pos[PARENTS[node]] + directions[node] * lengths[node]
    return pos


FOOT_LIST = list(FOOT_NODES["left"]) + list(FOOT_NODES["right"])


def standing_skeleton() -> np.ndarray:
    """T-pose skeleton with its lowest foot keypoint exactly on y = 0."""
    pose = forward_kinematics(LENGTHS)
    pose[:, 1] -= pose[FOOT_LIST, 1].min()
    return pose


def foot_mean(joints: np.ndarray, side: str) -> np.ndarray:
    return joints[:, list(FOOT_NODES[side])].mean(axis=1)


def xz_steps(points: np.ndarray) -> np.ndarray:
    """Frame-to-frame movement of ``(N, 3)`` points in the floor plane (x, z):
    the anchoring pins planted feet there and leaves y to the floor levelling."""
    return np.linalg.norm(np.diff(points[:, [0, 2]], axis=0), axis=-1)


# ----------------------------------------------------------------------
def test_to_world_flips_y_and_z_and_keeps_handedness():
    cam = np.array([[1.0, 2.0, 3.0], [0.5, -0.5, 4.0]])
    world = to_world(cam)
    np.testing.assert_allclose(world, [[1.0, -2.0, -3.0], [0.5, 0.5, -4.0]])
    assert np.linalg.det(CAM_TO_WORLD) == pytest.approx(1.0)
    assert to_world(np.zeros((2, 5, 3))).shape == (2, 5, 3)


def test_add_root_is_hip_midpoint():
    j = np.random.default_rng(0).normal(size=(4, 70, 3))
    out = add_root(j)
    assert out.shape == (4, 71, 3)
    np.testing.assert_allclose(out[:, ROOT], 0.5 * (j[:, 9] + j[:, 10]))


# ----------------------------------------------------------------------
def test_fill_gaps_interpolates_short_and_holds_long():
    samples = np.array([0, 1, 3, 4, 10])
    arrays = {"x": np.array([0.0, 1.0, 3.0, 4.0, 10.0], dtype=np.float32),
              "v": np.arange(5, dtype=np.float32)[:, None] * np.ones((5, 3), np.float32)}
    out_samples, out, valid = fill_gaps(samples, arrays, max_gap=2)
    np.testing.assert_array_equal(out_samples, np.arange(11))
    np.testing.assert_array_equal(valid, [1, 1, 0, 1, 1, 0, 0, 0, 0, 0, 1])
    assert out["x"][2] == pytest.approx(2.0)                   # midpoint of 1 and 3
    np.testing.assert_array_equal(out["x"][5:10], 4.0)          # held over the long gap
    assert out["x"][10] == 10.0
    assert out["v"].shape == (11, 3) and out["v"].dtype == np.float32
    np.testing.assert_allclose(out["v"][2], 1.5)


def test_fill_gaps_two_frame_gap_within_max_gap():
    _, out, valid = fill_gaps(np.array([0, 3]), {"x": np.array([0.0, 3.0])}, max_gap=2)
    np.testing.assert_allclose(out["x"], [0.0, 1.0, 2.0, 3.0])
    np.testing.assert_array_equal(valid, [1, 0, 0, 1])


# ----------------------------------------------------------------------
def test_normalise_bone_lengths_hits_targets_and_keeps_directions():
    rng = np.random.default_rng(1)
    # Random bone directions and wrong lengths in every frame.
    frames = []
    for _ in range(5):
        d = rng.normal(size=(NUM_NODES, 3))
        d /= np.linalg.norm(d, axis=-1, keepdims=True)
        frames.append(forward_kinematics(LENGTHS * rng.uniform(0.7, 1.3, NUM_NODES), d,
                                         root=rng.normal(size=3)))
    nodes = np.stack(frames)
    out = normalise_bone_lengths(nodes, LENGTHS)
    assert out.shape == nodes.shape
    lengths = bone_lengths(out)[:, :ROOT]
    np.testing.assert_allclose(lengths, np.broadcast_to(LENGTHS[:ROOT], lengths.shape), atol=1e-6)
    np.testing.assert_allclose(out[:, ROOT], nodes[:, ROOT])
    for node in range(NUM_NODES):
        if node == ROOT:
            continue
        old = nodes[:, node] - nodes[:, PARENTS[node]]
        new = out[:, node] - out[:, PARENTS[node]]
        cos = (old * new).sum(-1) / np.linalg.norm(old, axis=-1) / np.linalg.norm(new, axis=-1)
        np.testing.assert_allclose(cos, 1.0, atol=1e-9)


def test_normalise_bone_lengths_zero_bone_takes_rest_direction():
    lengths = LENGTHS.copy()
    lengths[15] = 0.0      # a leaf (left big toe)
    lengths[13] = 0.0      # an internal node (left ankle) with children
    nodes = forward_kinematics(lengths)[None]
    out = normalise_bone_lengths(nodes, LENGTHS)
    for node in (15, 13):
        d = out[0, node] - out[0, PARENTS[node]]
        np.testing.assert_allclose(d, REST_DIRECTIONS[node] * LENGTHS[node], atol=1e-9)
    np.testing.assert_allclose(bone_lengths(out)[0, 1:ROOT], LENGTHS[1:ROOT], atol=1e-9)


# ----------------------------------------------------------------------
def test_estimate_floor_on_standing_skeleton():
    pose = standing_skeleton()
    pose[:, 1] += 0.37                       # lowest foot keypoint now at y = 0.37
    joints = np.repeat(pose[None], 20, axis=0)
    valid = np.ones(20, dtype=bool)
    assert estimate_floor([joints], [valid]) == pytest.approx(0.37, abs=1e-9)
    # Invalid frames are ignored: sink them and the answer must not change.
    joints2 = joints.copy()
    joints2[::2, :, 1] -= 1.0
    valid2 = np.ones(20, dtype=bool)
    valid2[::2] = False
    assert estimate_floor([joints2], [valid2]) == pytest.approx(0.37, abs=1e-9)


# ----------------------------------------------------------------------
JITTER = 0.008   # metres per axis, uniform; at FPS it stays under speed_thresh
FPS = 10.0


def jittered_standing(n: int, seed: int = 2, amp: float = JITTER) -> tuple[np.ndarray, np.ndarray]:
    """Standing skeleton with the SAME random offset (uniform, +-``amp`` per
    axis) on every joint of a frame -- what cam_trans jitter looks like.
    Returns (joints, jitter)."""
    rng = np.random.default_rng(seed)
    jitter = rng.uniform(-amp, amp, size=(n, 3))
    joints = standing_skeleton()[None] + jitter[:, None, :]
    return joints, jitter


def test_anchor_root_removes_jitter_on_standing_skeleton():
    joints, jitter = jittered_standing(60)
    out, offset = anchor_root(joints, FPS)
    assert out.shape == joints.shape and offset.shape == (60, 3)
    # Planted feet stand still in x and z (both feet planted from the start).
    for side in ("left", "right"):
        assert xz_steps(foot_mean(out, side)).max() < 1e-6
    # Height is not anchored: it follows the model, so the floor levelling holds.
    np.testing.assert_array_equal(offset[:, 1], 0.0)
    np.testing.assert_allclose(out[:, :, 1], joints[:, :, 1])
    root_before = xz_steps(joints[:, ROOT]).mean()
    root_after = xz_steps(out[:, ROOT]).mean()
    assert root_after < 0.2 * root_before
    # The offset is continuous: it never jumps by more than the jitter did.
    max_jitter_step = np.linalg.norm(np.diff(jitter, axis=0), axis=-1).max()
    assert np.linalg.norm(np.diff(offset, axis=0), axis=-1).max() <= max_jitter_step + 1e-12
    # Bone lengths are untouched (a pure translation per frame).
    np.testing.assert_allclose(bone_lengths(out), bone_lengths(joints), atol=1e-9)


def test_anchor_root_leaves_airborne_translation_alone():
    n = 30
    pose = standing_skeleton()
    pose[:, 1] += 0.5                                  # feet well above height_thresh
    t = np.arange(n) / FPS
    joints = pose[None] + np.stack([t, np.zeros(n), np.zeros(n)], axis=1)[:, None, :]  # 1 m/s in x
    out, offset = anchor_root(joints, FPS)
    np.testing.assert_array_equal(out, joints)
    np.testing.assert_array_equal(offset, 0.0)


def test_anchor_root_two_feet_to_one_foot_has_no_jump():
    n = 40
    joints, jitter = jittered_standing(n, seed=3)
    # From frame 20 the left foot lifts (up 0.3 m, forward 0.2 m) and swings.
    lift = np.zeros((n, 3))
    lift[20:] = [0.0, 0.3, 0.2]
    lift[20:, 2] += np.arange(n - 20) * 0.05
    joints = joints.copy()
    joints[:, list(FOOT_NODES["left"])] += lift[:, None, :]
    out, offset = anchor_root(joints, FPS)
    max_jitter_step = np.linalg.norm(np.diff(jitter, axis=0), axis=-1).max()
    assert np.linalg.norm(np.diff(offset, axis=0), axis=-1).max() <= max_jitter_step + 1e-12
    # The right foot stays planted and still on both sides of the transition. On
    # the transition frame itself it is re-locked at its current output position
    # (that is what keeps the offset continuous), so there it may move by at
    # most one frame of jitter. The foot speed is a central difference over
    # +-1/15 s, at this 10 fps +-1 frame, so the left foot's lift at frame 20
    # is seen from frame 19 on.
    step = xz_steps(foot_mean(out, "right"))
    moving = np.flatnonzero(step > 1e-6)
    assert len(moving) == 1 and 17 <= moving[0] <= 19, moving
    assert step.max() <= max_jitter_step + 1e-12
    # The lifted foot follows its own motion relative to the planted one.
    assert foot_mean(out, "left")[-1, 1] - foot_mean(out, "right")[-1, 1] == pytest.approx(0.3, abs=1e-9)


def test_anchor_root_with_contact_probabilities():
    joints, _ = jittered_standing(30)
    contact = np.zeros((30, 2))
    contact[:, 1] = 1.0                                # only the right foot is "in contact"
    out, _ = anchor_root(joints, FPS, contact)
    assert xz_steps(foot_mean(out, "right")).max() < 1e-6


def test_anchor_root_adds_no_height_drift_over_many_plants():
    # Clean alternating gait: the swinging foot is lifted 8 cm, 60 plants per
    # foot at 30 fps, and the body's height carries a slow +-2 cm drift (the
    # model's depth error, which scales the whole body) plus 2 mm of jitter.
    fps, per_plant, plants = 30.0, 15, 60
    n = 2 * per_plant * plants
    t = np.arange(n)
    joints = np.repeat(standing_skeleton()[None], n, axis=0)
    for side, k in (("left", 0), ("right", 1)):
        swinging = ((t // per_plant) % 2) != k
        joints[:, list(FOOT_NODES[side]), 1] += np.where(swinging, 0.08, 0.0)[:, None]
    rng = np.random.default_rng(5)
    drift = 0.01 * np.sin(2 * np.pi * t / (7.3 * fps)) + 0.01 * np.sin(2 * np.pi * t / (2.9 * fps) + 1.0)
    joints[:, :, 1] += (drift + rng.normal(0.0, 0.002, size=n))[:, None]
    out, offset = anchor_root(joints, fps)
    assert planted_feet(joints, fps).mean() > 0.3     # ~0.38: the central difference blurs each plant by 2 frames
    # Were y anchored too, every plant would add its start-minus-end height
    # error to a sum that never decays: +5 cm after these 120 plants. Instead
    # the output height is exactly the input height.
    np.testing.assert_array_equal(offset[:, 1], 0.0)
    low_in = joints[:, FOOT_LIST, 1].min(axis=1)
    low_out = out[:, FOOT_LIST, 1].min(axis=1)
    np.testing.assert_allclose(low_out, low_in)
    assert np.abs(low_out).max() < 0.03


def test_planted_feet_survive_centimetre_jitter_at_30fps():
    # 1 cm of white jitter per axis at 30 fps reads as 0.7 m/s over one frame,
    # twice the plant threshold; over the +-2 frame central difference it is a
    # quarter of that. A standing foot must still count as planted, and a
    # foot sliding at walking speed (1.2 m/s, kept low) must not.
    fps, n = 30.0, 300
    rng = np.random.default_rng(6)
    joints = np.repeat(standing_skeleton()[None], n, axis=0)
    joints[:, list(FOOT_NODES["left"]), 2] += (np.arange(n) / fps * 1.2)[:, None]
    joints += rng.normal(0.0, 0.01, size=(n, 1, 3))
    planted = planted_feet(joints, fps)
    assert planted[:, 1].mean() > 0.8, planted[:, 1].mean()      # right: standing
    assert planted[:, 0].mean() < 0.05, planted[:, 0].mean()     # left: sliding


def walk(n: int, fps: float, stance: float = 0.5, jitter: float = 0.005,
         seed: int = 0) -> np.ndarray:
    """``(n, 71, 3)``: a walk along +z at 1.2 m/s. Each foot stands still for
    ``stance`` seconds, then swings forward for as long, lifted up to 8 cm, so
    each foot is planted half the time. Plus cam_trans-like Gaussian jitter
    (``jitter`` metres per axis, the same on every joint of a frame)."""
    base = standing_skeleton()
    t = np.arange(n) / fps
    step = 1.2 * stance                          # distance the body covers in one stance
    joints = np.repeat(base[None], n, axis=0)
    joints[:, :, 2] += (t * 1.2)[:, None]
    for side, phase in (("left", 0.0), ("right", 0.5)):
        nodes = list(FOOT_NODES[side])
        u = (t / (2 * stance) + phase) % 1.0     # where in its stride the foot is
        z_plant = (np.floor(t / (2 * stance) + phase) + phase) * 2 * step
        swing = np.clip(2 * u - 1, 0.0, 1.0)     # 0 -> 1 during the swing
        z = np.where(u < 0.5, z_plant, z_plant + swing * 2 * step)
        joints[:, nodes, 2] = base[nodes, 2] + z[:, None]
        joints[:, nodes, 1] = base[nodes, 1] + np.where(u < 0.5, 0.0, 0.08 * np.sin(np.pi * swing))[:, None]
    rng = np.random.default_rng(seed)
    return joints + rng.normal(0.0, jitter, size=(n, 1, 3))


def test_planted_feet_at_low_frame_rates():
    # A 30 fps walk pushed at 30, 15 and 10 fps (--frame-skip 1, 2, 3, or a
    # 10 fps security camera). A +-2 SAMPLE speed window is 0.4 s at 10 fps,
    # as long as a stance: it reached into the swing at both ends of every
    # stance and left 1% of foot-frames planted, so the anchoring did nothing.
    # Fixed at +-1/15 s it measures the same motion at every rate.
    joints = walk(1800, 30.0)
    for k in (1, 2, 3):
        planted = planted_feet(joints[::k], 30.0 / k)
        assert 0.3 < planted.mean() <= 0.5, (k, planted.mean())


# ----------------------------------------------------------------------
def make_prediction(joints_world: np.ndarray, bbox, confidence=0.9, contact=None,
                    track_id=None) -> HMRPrediction:
    """An HMRPrediction whose joints_3d_cam is ``joints_world[:70]`` in the camera frame."""
    cam = (joints_world[:70] @ CAM_TO_WORLD.T).astype(np.float32)   # the flip is its own inverse
    p = HMRPrediction(
        bbox=np.asarray(bbox, dtype=np.float32),
        confidence=confidence,
        joints_3d_local=cam - cam[ROOT - 1],
        joints_3d_cam=cam,
        joints_2d=np.tile(np.asarray(bbox[:2], np.float32), (70, 1)),
        cam_trans=cam[ROOT - 1].copy(),
        focal_length=np.array([1000.0, 1000.0], np.float32),
        principal_point=np.array([640.0, 360.0], np.float32),
        image_shape=(720, 1280),
        mhr_params=np.zeros(204, np.float32),
        shape_params=np.ones(45, np.float32),
        contact_prob=contact,
    )
    if track_id is not None:
        p.track_id = track_id
    return p


def two_person_recorder(fps=25.0, frame_step=1, n=10, missing=(5,), **kw) -> MocapRecorder:
    """Person A (stamped track_id=3) stands at x=0 with cam_trans-like jitter
    and is missing on ``missing`` frames; person B (no track_id, so the
    recorder's tracker calls it 0) stands still at x=1.5. Frames are pushed at
    25 per second whatever ``fps`` says. The export numbers the tracks 0, 1,
    ... in order of first appearance, so A comes out as track 1."""
    rec = MocapRecorder(fps=fps, frame_step=frame_step, **kw)
    a, _ = jittered_standing(n, seed=4, amp=0.003)    # 3 mm: under speed_thresh at 25 fps
    b = standing_skeleton()[None].repeat(n, axis=0)
    b[:, :, 0] += 1.5
    for i in range(n):
        frame_idx = i * frame_step
        persons = [make_prediction(b[i], [600, 100, 700, 400], confidence=0.8)]
        if i not in missing:
            persons.append(make_prediction(a[i], [100, 100, 200, 400], track_id=3))
        rec.push(frame_idx, i / 25.0, persons)
    return rec


def test_recorder_end_to_end():
    rec = two_person_recorder()
    tracks = rec.finish()
    assert [t.track_id for t in tracks] == [0, 1]
    assert rec.finish() is tracks                       # idempotent
    b, a = tracks
    for t in (a, b):
        np.testing.assert_array_equal(t.frames, np.arange(10))
        assert t.frames.dtype == np.int64
        assert t.joints_world.shape == (10, 71, 3) and t.joints_world.dtype == np.float32
        assert t.joints_cam.shape == (10, 71, 3) and t.joints_2d.shape == (10, 70, 2)
        assert t.mhr_params.shape == (10, 204) and t.shape_params.shape == (10, 45)
        assert t.contact is None and t.root_offset.shape == (10, 3)
        assert t.bone_lengths.shape == (71,) and t.bone_lengths[ROOT] == 0.0
        np.testing.assert_allclose(t.bone_lengths[:ROOT], LENGTHS[:ROOT], atol=1e-5)
        np.testing.assert_allclose(bone_lengths(t.joints_world)[:, :ROOT],
                                   np.broadcast_to(LENGTHS[:ROOT], (10, ROOT)), atol=1e-5)
        np.testing.assert_allclose(t.times, np.arange(10) / 25.0)
    np.testing.assert_array_equal(a.valid, [1, 1, 1, 1, 1, 0, 1, 1, 1, 1])
    assert b.valid.all()
    assert a.confidence[0] == pytest.approx(0.9) and b.confidence[0] == pytest.approx(0.8)
    # joints_cam is the prediction as received, Y down; joints_world is Y up.
    assert a.joints_cam[0, 0, 1] < 0 < a.joints_world[0, 0, 1]
    # Floor found at y = 0 (the synthetic feet stand on it): one clip-wide 5th
    # percentile, so person A's 3 mm jitter can pull it that far under person B.
    assert abs(rec.floor_y) < 0.003
    assert abs(b.joints_world[:, FOOT_LIST, 1].min()) < 0.003
    # Person A's jitter was removed in x and z: both feet planted, root still
    # there; its height follows the model (the floor levelling owns y).
    assert xz_steps(a.joints_world[:, ROOT].astype(np.float64)).max() < 1e-5
    assert np.abs(a.root_offset[:, [0, 2]]).max() > 0 and np.abs(b.root_offset).max() < 1e-6
    np.testing.assert_array_equal(a.root_offset[:, 1], 0.0)
    # A standing person covers no depth, so no camera pitch is estimated.
    assert rec.camera_pitch_deg == 0.0


def test_recorder_export_json_npz_csv(tmp_path):
    rec = two_person_recorder()
    paths = rec.export(tmp_path / "dance", formats=("json", "npz", "csv"))
    assert [p.name for p in paths] == ["dance.mocap.json", "dance.mocap.npz",
                                       "dance.track0.csv", "dance.track1.csv"]
    assert all(p.exists() for p in paths)

    doc = json.loads(paths[0].read_text())
    assert doc["format"] == "instanthmr-mocap" and doc["version"] == 1
    for key in ("fps", "frame_step", "num_frames", "units", "coordinate_system",
                "joint_names", "parents", "floor_y", "footlock", "tracks"):
        assert key in doc
    assert doc["joint_names"] == NODE_NAMES and len(doc["joint_names"]) == 71
    assert doc["parents"] == PARENTS and doc["fps"] == 25.0 and doc["num_frames"] == 10
    assert [t["id"] for t in doc["tracks"]] == [0, 1]
    t1 = doc["tracks"][1]
    assert (t1["first_frame"], t1["last_frame"], t1["num_frames"], t1["num_filled"]) == (0, 9, 10, 1)
    assert len(t1["bone_lengths"]) == 71 and len(t1["frames"]) == 10
    f = t1["frames"][5]
    assert f["valid"] is False and f["frame"] == 5
    assert len(f["joints_world"]) == 71 and len(f["joints_world"][0]) == 3
    assert len(f["joints_2d"]) == 70 and len(f["mhr_params"]) == 204 and f["contact"] is None

    npz = np.load(paths[1])
    assert set(npz.files) >= {"fps", "frame_step", "floor_y", "joint_names", "parents", "track_ids"}
    assert list(npz["track_ids"]) == [0, 1] and list(npz["joint_names"]) == NODE_NAMES
    assert npz["track1_joints_world"].shape == (10, 71, 3)
    assert npz["track1_valid"].dtype == bool and npz["track1_bone_lengths"].shape == (71,)
    assert "track1_contact" not in npz.files

    rows = list(csv.reader(paths[3].open()))
    assert len(rows) == 11 and all(len(r) == 4 + 71 * 3 for r in rows)
    assert rows[0][:5] == ["frame", "time", "valid", "confidence", "nose_x"]
    assert rows[6][0] == "5" and rows[6][2] == "0" and rows[1][2] == "1"


def test_recorder_export_bvh(tmp_path):
    pytest.importorskip("instanthmr.bvh")
    rec = two_person_recorder()
    paths = rec.export(tmp_path / "dance", formats=("bvh",))
    assert [p.name for p in paths] == ["dance.track0.bvh", "dance.track1.bvh"]
    text = paths[1].read_text()
    assert "Frames: 10" in text and "HIERARCHY" in text


def test_export_unknown_format_raises():
    with pytest.raises(ValueError):
        two_person_recorder().export("/nonexistent/x", formats=("json", "fbx"))


def test_recorder_measures_fps_and_frame_step(tmp_path):
    # A 50 fps video read with --frame-skip 2: frames 0, 2, 4, ... pushed at 25 per second.
    rec = two_person_recorder(fps=None, frame_step=2, missing=())
    tracks = rec.finish()
    assert rec.fps == pytest.approx(25.0)
    for t in tracks:
        np.testing.assert_array_equal(t.frames, np.arange(0, 20, 2))
        assert t.valid.all() and len(t.frames) == 10
    doc = json.loads(rec.export(tmp_path / "c", ("json",))[0].read_text())
    assert doc["num_frames"] == 19 and doc["frame_step"] == 2 and doc["fps"] == 25.0


def test_recorder_times_advance_over_long_gap():
    # A hole longer than max_gap holds the pose, but time must keep advancing
    # (it is linear in the frame number) or the CSV's time column freezes for
    # the hole and then jumps.
    rec = MocapRecorder(fps=25.0, max_gap=10)
    pose = standing_skeleton()
    for i in list(range(3)) + list(range(20, 23)):
        rec.push(i, i / 25.0, [make_prediction(pose, [100, 100, 200, 400], track_id=0)])
    (t,) = rec.finish()
    np.testing.assert_array_equal(t.frames, np.arange(23))
    np.testing.assert_allclose(t.times, np.arange(23) / 25.0)
    assert np.all(np.diff(t.times) > 0)
    assert (~t.valid).sum() == 17 and np.all(t.joints_world[3:20] == t.joints_world[2])


def test_recorder_one_track_across_max_gap():
    # Without stamped ids the recorder's own tracker keeps a person alive for
    # max_gap missed frames -- exactly the holes fill_gaps interpolates -- so a
    # blink of the detector does not split one person into two files.
    def record(gap):
        rec = MocapRecorder(fps=25.0, max_gap=10)
        pose = standing_skeleton()
        for i in range(30):
            present = i < 10 or i >= 10 + gap
            rec.push(i, i / 25.0, [make_prediction(pose, [100, 100, 200, 400])] if present else [])
        return rec.finish()
    (t,) = record(10)
    assert t.track_id == 0 and (~t.valid).sum() == 10 and t.valid[:10].all() and t.valid[20:].all()
    assert np.all(np.diff(t.times) > 0)
    assert [t.track_id for t in record(11)] == [0, 1]


def test_recorder_track_ends_when_detections_stop():
    rec = MocapRecorder(fps=25.0)
    pose = standing_skeleton()
    for i in range(30):
        rec.push(i, i / 25.0, [make_prediction(pose, [100, 100, 200, 400])] if i < 10 else [])
    (t,) = rec.finish()
    np.testing.assert_array_equal(t.frames, np.arange(10))
    assert t.valid.all()


def test_recorder_finish_tolerates_torn_push():
    # A Ctrl+C between the appends of one push leaves `samples` (and here
    # `times`) one longer than the other fields; that frame is dropped and the
    # ten complete ones are exported.
    rec = two_person_recorder(fps=None, missing=())
    rec._tracks[3]["samples"].append(10)
    rec._tracks[3]["times"].append(10 / 25.0)
    tracks = rec.finish()
    assert rec.fps == pytest.approx(25.0)
    assert [len(t.frames) for t in tracks] == [10, 10]
    np.testing.assert_array_equal(tracks[1].frames, np.arange(10))


def pitched_camera_walk(tilt_deg: float, n: int = 150, fps: float = 30.0) -> MocapRecorder:
    """A person walking 3 m towards a camera 1.2 m above the floor that is
    pitched down by ``tilt_deg``; on every 10th-to-12th frame the whole body
    is 8 cm up in the air (a hop, which the floor fit must ignore)."""
    th = np.radians(tilt_deg)
    pitch = np.array([[1, 0, 0], [0, np.cos(th), -np.sin(th)], [0, np.sin(th), np.cos(th)]])
    base = standing_skeleton()
    base[:, 1] -= 1.2
    rec = MocapRecorder(fps=fps, footlock=False, min_frames=1)
    for i in range(n):
        w = base.copy()
        w[:, 2] += -5.0 + 3.0 * i / (n - 1)
        if i % 10 < 3:
            w[:, 1] += 0.08
        # What a camera pitched down sees: into the camera frame, pitch, back.
        seen = (w @ CAM_TO_WORLD) @ pitch.T @ CAM_TO_WORLD
        rec.push(i, i / fps, [make_prediction(seen, [100, 100, 200, 400], track_id=1)])
    return rec


def test_recorder_levels_a_pitched_camera():
    for tilt in (10.0, 20.0):
        rec = pitched_camera_walk(tilt)
        (t,) = rec.finish()
        assert rec.camera_pitch_deg == pytest.approx(tilt, abs=0.1)
        low = t.joints_world[:, FOOT_LIST, 1].min(axis=1)
        on_floor = np.arange(len(low)) % 10 >= 3
        # Without the levelling the floor would slope by 3 m * tan(tilt): 52 cm
        # at 10 degrees. After it the feet are on y = 0 for the whole walk,
        # and 8 cm above it during the hops.
        assert np.abs(low[on_floor]).max() < 0.005
        assert np.abs(low[~on_floor] - 0.08).max() < 0.005
    rec = pitched_camera_walk(0.0)
    rec.finish()
    assert abs(rec.camera_pitch_deg) < 0.01
    # A person has to cover a metre of depth or the slope is noise: a
    # standing person with 1 cm of jitter gets no correction at all, and two
    # standing persons at different depths (1.5 m apart) do not add up to one.
    joints, _ = jittered_standing(100, amp=0.01)
    valid = np.ones(100, dtype=bool)
    assert estimate_pitch([joints], [valid]) == 0.0
    far = joints.copy()
    far[:, :, 1] += 0.3                 # a bad depth estimate also mis-places the height
    far[:, :, 2] -= 1.5
    assert estimate_pitch([joints, far], [valid, valid]) == 0.0


def test_estimate_pitch_ignores_a_frame_with_a_bad_depth():
    # Level camera 1.2 m above the floor, a person standing 4 m away for 300
    # frames with 1% depth jitter. A depth error slides the whole person along
    # the camera ray, so it also moves them in height. One frame 1.3x too far
    # (a person half out of the picture) used to open the 1 m depth gate on
    # its own and then, as the one far-off point, tilt the fit by 17 degrees.
    rng = np.random.default_rng(7)
    person = standing_skeleton() + [0.0, -1.2, -4.0]
    scale = 1.0 + rng.normal(0.0, 0.01, size=300)
    scale[150] = 1.3
    joints = np.stack([to_world(to_world(person) * s) for s in scale])   # to_world is its own inverse
    assert estimate_pitch([joints], [np.ones(300, dtype=bool)]) == 0.0


def test_camera_pitch_is_written_and_relates_the_files_to_the_camera(tmp_path):
    # joints_world is the camera's view rotated level and moved to the floor;
    # with the pitch in the files a reader can undo both (no foot anchoring here).
    rec = pitched_camera_walk(10.0)
    json_path, npz_path = rec.export(tmp_path / "walk", ("json", "npz"))
    doc = json.loads(json_path.read_text())
    assert doc["camera_pitch_deg"] == pytest.approx(10.0, abs=0.1)
    assert "camera_pitch_deg" in doc["coordinate_system"]["joints_world"]
    npz = np.load(npz_path)
    assert float(npz["camera_pitch_deg"]) == doc["camera_pitch_deg"]
    world = npz["track0_joints_world"].astype(np.float64) + [0.0, float(npz["floor_y"]), 0.0]
    level = _rotation_about_x(-np.radians(float(npz["camera_pitch_deg"])))
    np.testing.assert_allclose(world @ level, to_world(npz["track0_joints_cam"]), atol=1e-4)


def test_fill_gaps_turns_root_rotation_the_short_way():
    # Back to the camera, the root's extrinsic-XYZ Euler angles sit at +-pi:
    # frames 0 and 5 are 2.8 degrees apart as rotations but (-179, 1, 180)
    # and (179, -1, 180) degrees as numbers. Interpolated number by number,
    # the filled frames 1-4 turned 72, 143, 145 and 74 degrees away.
    def rotation(xyz_rad):
        a, b, c = xyz_rad
        rx = np.array([[1, 0, 0], [0, np.cos(a), -np.sin(a)], [0, np.sin(a), np.cos(a)]])
        ry = np.array([[np.cos(b), 0, np.sin(b)], [0, 1, 0], [-np.sin(b), 0, np.cos(b)]])
        rz = np.array([[np.cos(c), -np.sin(c), 0], [np.sin(c), np.cos(c), 0], [0, 0, 1]])
        return rz @ ry @ rx

    def degrees_between(r1, r2):
        return np.degrees(np.arccos(np.clip((np.trace(r1.T @ r2) - 1.0) / 2.0, -1.0, 1.0)))

    rec = MocapRecorder(fps=25.0)
    ends = {0: np.radians([-179.0, 1.0, 180.0]), 5: np.radians([179.0, -1.0, 180.0])}
    pushed = {}
    for i, root in ends.items():
        p = make_prediction(standing_skeleton(), [100, 100, 200, 400], track_id=0)
        p.mhr_params[3:6] = root
        p.mhr_params[10] = -3.1          # a joint angle near -pi: must not jump either
        pushed[i] = p.mhr_params.copy()
        rec.push(i, i / 25.0, [p])
    (t,) = rec.finish()
    assert t.mhr_params.dtype == np.float32 and list(t.valid) == [1, 0, 0, 0, 0, 1]
    for i in ends:                       # the pushed rows are untouched
        np.testing.assert_array_equal(t.mhr_params[i], pushed[i])
    r0, r5 = rotation(t.mhr_params[0, 3:6]), rotation(t.mhr_params[5, 3:6])
    total = degrees_between(r0, r5)
    assert total == pytest.approx(2.83, abs=0.01)
    for i in range(1, 5):
        r = rotation(t.mhr_params[i, 3:6])
        assert degrees_between(r0, r) <= total + 1e-3 and degrees_between(r, r5) <= total + 1e-3
        assert abs(t.mhr_params[i, 10] + 3.1) < 1e-6


def test_report_no_tracks():
    rec = MocapRecorder(fps=30.0)
    text = rec.report()
    assert "no person was tracked" in text and "floor" not in text and "finger" not in text
    rec = MocapRecorder(fps=30.0)          # min_frames=2: a single frame is dropped
    rec.push(0, 0.0, [make_prediction(standing_skeleton(), [100, 100, 200, 400])])
    text = rec.report()
    assert "no person was tracked for 2 frame(s)" in text and "1 short detection" in text
    assert "floor" not in text and "finger" not in text


def test_report_flags_short_low_confidence_track():
    rec = two_person_recorder(n=40, missing=())
    blob = standing_skeleton() + [0, 0, -3]
    for i in range(5):
        rec.push(i, i / 25.0, [make_prediction(blob, [1000, 50, 1050, 150], confidence=0.55)])
    lines = rec.report().splitlines()
    flagged = [ln for ln in lines if "may be a false detection" in ln]
    assert len(flagged) == 1 and flagged[0].startswith("  track 1:"), lines


def test_recorder_drops_short_tracks_and_reports():
    rec = two_person_recorder(min_frames=3)
    # A one-frame false positive, far from both persons.
    rec.push(4, 4 / 25.0, [make_prediction(standing_skeleton() + [0, 0, -3], [1000, 50, 1050, 150])])
    tracks = rec.finish()
    assert [t.track_id for t in tracks] == [0, 1]
    text = rec.report()
    for word in ("track 0", "track 1", "floor", "finger", "1 dropped"):
        assert word in text, text


def test_first_person_is_track0_after_a_dropped_false_detection(tmp_path):
    # A one-frame false detection (a reflection) on frame 0 takes the tracker's
    # id 0; the person walks in on frame 2. The files must still say track0.
    rec = MocapRecorder(fps=25.0)
    rec.push(0, 0.0, [make_prediction(standing_skeleton() + [0, 0, -3], [1000, 50, 1050, 150])])
    for i in range(2, 20):
        rec.push(i, i / 25.0, [make_prediction(standing_skeleton(), [100, 100, 200, 400])])
    paths = rec.export(tmp_path / "shop", ("csv", "bvh"))
    assert [p.name for p in paths] == ["shop.track0.csv", "shop.track0.bvh"]
    assert "track 0: frames 2-19" in rec.report()


def test_no_floor_turns_the_foot_anchoring_off():
    # Without the floor step y = 0 is the camera's height, so with the camera
    # 1.4 m above the floor every foot passed the "low" test and a foot lifted
    # slowly (0.26 m/s) counted as planted: it held still and dragged the
    # standing foot and the body 15 cm back. The anchoring needs the floor.
    fps, n = 30.0, 45
    joints = np.repeat((standing_skeleton() + [0.0, -1.4, -3.0])[None], n, axis=0)
    lift = np.linspace(0.0, 1.0, n)[:, None] * [0.0, 0.25, 0.30]   # up 25 cm, forward 30 cm in 1.5 s
    joints[:, list(FOOT_NODES["right"])] += lift[:, None, :]
    for floor in (True, False):
        rec = MocapRecorder(fps=fps, floor=floor)
        for i in range(n):
            rec.push(i, i / fps, [make_prediction(joints[i], [100, 100, 200, 400], track_id=0)])
        (t,) = rec.finish()
        left = foot_mean(t.joints_world.astype(np.float64), "left")
        assert np.abs(left[:, [0, 2]] - left[0, [0, 2]]).max() < 0.01, floor
    assert rec.footlock is False and "foot anchoring: off" in rec.report()


def test_report_mentions_tracks_floor_and_fingers():
    rec = two_person_recorder()
    text = rec.report()
    assert "track 0" in text and "track 1" in text
    assert "floor" in text and "finger" in text and "ordered as expected" in text
    assert "camera pitch not measured" in text and "false detection" not in text
