"""Tests for ``instanthmr.bvh``: the Euler / swing / Kabsch building blocks,
and write -> read -> forward-kinematics round trips on the real mocap tree."""

from __future__ import annotations

import numpy as np
import pytest

from instanthmr import bvh
from instanthmr.mocap_skeleton import (
    CHILDREN, NODE_NAMES, PARENTS, REST_DIRECTIONS, ROOT, TOPOLOGICAL_ORDER,
)

SCALE = 100.0           # write_bvh default: metres -> centimetres
#: The file has one ``<fork>_to_<child>`` joint per child of every fork but
#: the fork's main child, which hangs on the fork directly.
N_FORK_JOINTS = sum(len(c) - 1 for c in CHILDREN if len(c) >= 2)


def tree_positions(parsed: bvh.Bvh) -> np.ndarray:
    """``fk(parsed)`` for the 71 tree nodes only, in ``NODE_NAMES`` order, in
    file units (the file holds them in its own order, plus the fork joints)."""
    return bvh.fk(parsed)[:, [parsed.names.index(n) for n in NODE_NAMES]]


def random_rotation(rng: np.random.Generator) -> np.ndarray:
    """Uniformly random rotation matrix, from a random unit quaternion."""
    w, x, y, z = rng.normal(size=4) / 1.0
    n = np.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


#: Plausible bone lengths in metres for the real tree, by node name. Nodes
#: not listed are finger joints and take ``_FINGER_LENGTHS`` by suffix.
_LENGTHS = {
    "left_hip": 0.10, "right_hip": 0.10,
    "left_knee": 0.42, "right_knee": 0.42,
    "left_ankle": 0.42, "right_ankle": 0.42,
    "left_heel": 0.08, "right_heel": 0.08,
    "left_big_toe_tip": 0.20, "right_big_toe_tip": 0.20,
    "left_small_toe_tip": 0.18, "right_small_toe_tip": 0.18,
    "neck": 0.50, "nose": 0.18,
    "left_eye": 0.05, "right_eye": 0.05, "left_ear": 0.10, "right_ear": 0.10,
    "left_shoulder": 0.18, "right_shoulder": 0.18,
    "left_acromion": 0.04, "right_acromion": 0.04,
    "left_elbow": 0.28, "right_elbow": 0.28,
    "left_olecranon": 0.04, "right_olecranon": 0.04,
    "left_cubital_fossa": 0.03, "right_cubital_fossa": 0.03,
    "left_wrist": 0.26, "right_wrist": 0.26,
}
_FINGER_LENGTHS = {"_third_joint": 0.08, "_second_joint": 0.04, "_first_joint": 0.025, "_tip": 0.02}


def real_rest_offsets() -> np.ndarray:
    """``(71, 3)`` rest offsets in metres: REST_DIRECTIONS times the table above."""
    lengths = np.zeros(len(NODE_NAMES))
    for j, name in enumerate(NODE_NAMES):
        if j == ROOT:
            continue
        lengths[j] = _LENGTHS.get(name) or next(
            v for suffix, v in _FINGER_LENGTHS.items() if name.endswith(suffix))
    assert (lengths[np.arange(len(NODE_NAMES)) != ROOT] > 0).all()
    return REST_DIRECTIONS * lengths[:, None]


def animate(parents, order, rest_offsets, rng, n_frames):
    """A rigid animation: every joint gets a random local rotation per frame
    and the root moves. Returns ``positions (N, J, 3)`` built hierarchically
    the way a BVH reader would, and the global rotations ``(N, J, 3, 3)``."""
    J = len(parents)
    positions = np.zeros((n_frames, J, 3))
    global_rot = np.zeros((n_frames, J, 3, 3))
    for t in range(n_frames):
        for j in order:
            L = random_rotation(rng)
            p = parents[j]
            if p < 0:
                global_rot[t, j] = L
                positions[t, j] = [0.1 * t, 1.0 + 0.05 * np.sin(t), 0.02 * t]
            else:
                global_rot[t, j] = global_rot[t, p] @ L
                positions[t, j] = positions[t, p] + global_rot[t, p] @ rest_offsets[j]
    return positions, global_rot


def assert_proper_rotation(R):
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-12)
    assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-12)


# --------------------------------------------------------------------------
# building blocks


def test_euler_round_trip():
    rng = np.random.default_rng(0)
    for _ in range(500):
        R = random_rotation(rng)
        z, x, y = bvh.matrix_to_euler_zxy(R)
        np.testing.assert_allclose(bvh.euler_zxy_to_matrix(z, x, y), R, atol=1e-9)


@pytest.mark.parametrize("x", [90.0, -90.0])
def test_euler_gimbal(x):
    R = bvh.euler_zxy_to_matrix(33.0, x, -71.0)
    z2, x2, y2 = bvh.matrix_to_euler_zxy(R)
    assert np.isfinite([z2, x2, y2]).all()
    assert x2 == pytest.approx(x, abs=1e-6)
    np.testing.assert_allclose(bvh.euler_zxy_to_matrix(z2, x2, y2), R, atol=1e-9)


def test_swing():
    rng = np.random.default_rng(1)
    for _ in range(200):
        u, v = rng.normal(size=(2, 3))
        u, v = u / np.linalg.norm(u), v / np.linalg.norm(v)
        R = bvh.swing(u, v)
        assert_proper_rotation(R)
        np.testing.assert_allclose(R @ u, v, atol=1e-12)
    for _ in range(20):
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        R = bvh.swing(u, -u)
        assert_proper_rotation(R)
        np.testing.assert_allclose(R @ u, -u, atol=1e-12)
        np.testing.assert_allclose(bvh.swing(u, u), np.eye(3), atol=1e-12)


def test_kabsch_recovers_rotation():
    rng = np.random.default_rng(2)
    for _ in range(50):
        R = random_rotation(rng)
        a = rng.normal(size=(3, 3))
        a /= np.linalg.norm(a, axis=1, keepdims=True)
        b = a @ R.T                      # b_k = R a_k
        np.testing.assert_allclose(bvh.kabsch(a, b), R, atol=1e-12)


# --------------------------------------------------------------------------
# round trips through a file


def test_rigid_round_trip_real_tree(tmp_path):
    rng = np.random.default_rng(3)
    rest = real_rest_offsets()
    positions, true_global = animate(PARENTS, TOPOLOGICAL_ORDER, rest, rng, 20)
    path = tmp_path / "rigid.bvh"
    bvh.write_bvh(path, NODE_NAMES, PARENTS, rest, positions, fps=30.0)

    # the file itself
    text = path.read_text()
    assert text.startswith("HIERARCHY")
    assert "MOTION" in text and "Frames: 20" in text and "Frame Time: 0.03333333" in text
    assert "ROOT hips\n{\n\tOFFSET 0.000000 0.000000 0.000000\n\tCHANNELS 6 " in text
    lines = text.splitlines()
    frame_lines = lines[lines.index("Frames: 20") + 2:]
    assert len(frame_lines) == 20
    assert N_FORK_JOINTS == 25          # 71 tree nodes -> 96 joints in the file
    assert all(len(line.split()) == 6 + 3 * (70 + N_FORK_JOINTS) for line in frame_lines)
    n_leaves = sum(1 for j in range(len(NODE_NAMES)) if j not in PARENTS)
    assert n_leaves == 26 and text.count("End Site") == n_leaves

    # read back: joints come in file order, so match them by name
    parsed = bvh.read_bvh(path)
    assert parsed.names[0] == "hips" and parsed.parents[0] == -1
    assert parsed.frame_time == pytest.approx(1 / 30)
    assert parsed.motion.shape == (20, 6 + 3 * (70 + N_FORK_JOINTS))
    fork_joints = [i for i, n in enumerate(parsed.names) if "_to_" in n]
    assert len(fork_joints) == N_FORK_JOINTS
    tree = [i for i in range(len(parsed.names)) if i not in fork_joints]
    perm = [NODE_NAMES.index(parsed.names[i]) for i in tree]
    assert sorted(perm) == list(range(len(NODE_NAMES)))
    np.testing.assert_allclose(parsed.offsets[tree][1:], rest[perm][1:] * SCALE, atol=1e-6)
    np.testing.assert_array_equal(parsed.offsets[fork_joints], 0.0)
    # every fork joint sits between a fork and one of its children
    for i in fork_joints:
        fork, child = parsed.names[i].split("_to_")
        assert parsed.names[parsed.parents[i]] == fork
        assert [parsed.names[c] for c, p in enumerate(parsed.parents) if p == i] == [child]
    # ... and every fork keeps exactly one child of its own, the main child
    for j in range(len(parsed.names)):
        kids = [c for c, p in enumerate(parsed.parents) if p == j]
        if len(kids) >= 2:
            assert sum(1 for c in kids if c not in fork_joints) == 1, parsed.names[j]
    # No zero-length bones the way Blender's importer draws them: a joint's
    # bone ends at the average of its children's rest heads (an End Site for
    # a leaf). A zero-length one it replaces with a copy of the parent bone,
    # which drew 30-50 cm spikes out of the head and through the floor.
    rest_head = np.zeros((len(parsed.names), 3))
    for i, p in enumerate(parsed.parents):
        if p >= 0:
            rest_head[i] = rest_head[p] + parsed.offsets[i]
    for j in range(len(parsed.names)):
        kids = [c for c, p in enumerate(parsed.parents) if p == j]
        if kids:
            tail = rest_head[kids].mean(axis=0)
            assert np.linalg.norm(tail - rest_head[j]) > 0.1, parsed.names[j]   # cm
    err = np.linalg.norm(bvh.fk(parsed)[:, tree] - positions[:, perm] * SCALE, axis=-1)
    assert err.max() < 1e-3             # file units (cm), i.e. 0.01 mm
    # rigid motion: the fork joints add no rotation of their own
    rot = parsed.motion[:, 6:].reshape(20, -1, 3)    # one (z, x, y) triple per joint, file order
    assert np.abs(rot[:, [i - 1 for i in fork_joints]]).max() < 1e-6

    # at a fork the Kabsch solve recovers the true orientation, twist included
    forks = [j for j in range(len(NODE_NAMES)) if PARENTS.count(j) >= 2]
    assert ROOT in forks and len(forks) == 11   # pelvis, neck, nose, 2x shoulder/elbow/wrist/ankle
    for t in (0, 7, 19):
        solved, _ = bvh.solve_frame(PARENTS, rest, positions[t], perm)   # perm: parents first
        for j in forks:
            np.testing.assert_allclose(solved[j], true_global[t, j], atol=1e-9)


def test_noisy_input_is_bounded(tmp_path):
    """1 cm of noise in a random direction on every joint of every frame. The
    file keeps bone directions and rest lengths, so the error is the length
    mismatch accumulated down each chain: bounded, not zero (observed: max
    4.4 cm, mean 1.4 cm; 4.9 / 1.6 cm before the fork joints)."""
    rng = np.random.default_rng(4)
    rest = real_rest_offsets()
    clean, _ = animate(PARENTS, TOPOLOGICAL_ORDER, rest, rng, 20)
    direction = rng.normal(size=clean.shape)
    noisy = clean + 0.01 * direction / np.linalg.norm(direction, axis=-1, keepdims=True)
    path = tmp_path / "noisy.bvh"
    bvh.write_bvh(path, NODE_NAMES, PARENTS, rest, noisy, fps=25.0)
    err_cm = np.linalg.norm(tree_positions(bvh.read_bvh(path)) - noisy * SCALE, axis=-1)
    print(f"noisy round trip: max {err_cm.max():.2f} cm, mean {err_cm.mean():.2f} cm")
    assert err_cm.max() < 6.0
    assert err_cm.mean() < 2.0


def _tilted(d: np.ndarray, degrees: float, rng: np.random.Generator) -> np.ndarray:
    """Unit vector ``d`` turned by ``degrees`` about a random axis normal to it."""
    axis = np.cross(d, rng.normal(size=3))
    axis /= np.linalg.norm(axis)
    t = np.radians(degrees)
    return d * np.cos(t) + np.cross(axis, d) * np.sin(t)


def _turn_subtree(pose: np.ndarray, node: int, pivot: int, R: np.ndarray) -> np.ndarray:
    """``pose`` with ``node`` and everything below it rotated by ``R`` about ``pivot``."""
    below, stack = [], [node]
    while stack:
        below.append(stack.pop())
        stack.extend(CHILDREN[below[-1]])
    out = pose.copy()
    out[below] = (pose[below] - pose[pivot]) @ R.T + pose[pivot]
    return out


def test_exact_lengths_round_trip_exactly_at_forks(tmp_path):
    """Bone lengths equal the rest lengths, no noise, but the children of the
    forks are NOT where one rotation of the rest pose would put them. Before
    the fork joints the file was off by 3.8 cm (bones 10 degrees off the rest
    directions), 6.1 cm (head turned 60 degrees: the face) and 2.4 cm (a
    15 degree side bend over a level pelvis: one whole leg); with them every
    position comes back exactly."""
    rng = np.random.default_rng(6)
    rest = real_rest_offsets()
    lengths = np.linalg.norm(rest, axis=1)
    anatomy = np.zeros_like(rest)        # this person's bones: 10 degrees off the T-pose
    for j in range(len(NODE_NAMES)):
        if j != ROOT:
            anatomy[j] = _tilted(REST_DIRECTIONS[j], 10.0, rng)
    standing = np.zeros_like(rest)
    for j in TOPOLOGICAL_ORDER[1:]:
        standing[j] = standing[PARENTS[j]] + anatomy[j] * lengths[j]
    c, s = np.cos(np.radians(60.0)), np.sin(np.radians(60.0))
    turn = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])          # about y
    c, s = np.cos(np.radians(15.0)), np.sin(np.radians(15.0))
    bend = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])          # about z
    neck, nose = NODE_NAMES.index("neck"), NODE_NAMES.index("nose")
    moving = _turn_subtree(_turn_subtree(standing, nose, neck, turn), neck, ROOT, bend)
    positions = np.stack([standing, moving]) + [0.3, 1.0, -2.0]
    path = tmp_path / "forks.bvh"
    bvh.write_bvh(path, NODE_NAMES, PARENTS, rest, positions, fps=30.0)
    err_cm = np.linalg.norm(tree_positions(bvh.read_bvh(path)) - positions * SCALE, axis=-1)
    assert err_cm.max() < 1e-3, err_cm.max()


def test_chain_only_tree(tmp_path):
    """root -> a -> b -> c: no forks, so only the swing path is exercised."""
    rng = np.random.default_rng(5)
    names, parents = ["root", "a", "b", "c"], [-1, 0, 1, 2]
    rest = np.array([[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.4, 0.0], [0.3, 0.0, 0.0]])
    positions, _ = animate(parents, [0, 1, 2, 3], rest, rng, 10)
    path = tmp_path / "chain.bvh"
    bvh.write_bvh(path, names, parents, rest, positions, fps=60.0)
    parsed = bvh.read_bvh(path)
    assert parsed.names == names and parsed.parents == parents
    assert [len(c) for c in parsed.channels] == [6, 3, 3, 3]
    assert list(parsed.end_sites) == [3]
    # a quarter of the 30 cm bone would be 7.5 cm: capped at 1 cm, same direction
    np.testing.assert_allclose(parsed.end_sites[3], [1.0, 0.0, 0.0], atol=1e-6)
    bvh.write_bvh(path, names, parents, rest * 0.1, positions * 0.1, fps=60.0)
    # a 3 cm bone keeps its quarter, 0.75 cm
    np.testing.assert_allclose(bvh.read_bvh(path).end_sites[3], rest[3] * 0.1 * SCALE * 0.25,
                               atol=1e-6)
    err = np.linalg.norm(bvh.fk(parsed) - positions * SCALE, axis=-1)
    assert err.max() < 1e-3
