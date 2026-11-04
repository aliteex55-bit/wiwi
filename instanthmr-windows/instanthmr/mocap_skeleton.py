"""The motion-capture skeleton: a tree over the MHR70 keypoints.

``skeleton.py`` describes the 70 keypoints as an undirected drawing (the lines
the viewer draws, including the hip-hip-shoulder loop). Motion-capture formats
such as BVH need a *tree*: every joint has exactly one parent, there is one
root, and every bone has a rest-pose direction and a fixed length. This module
is that tree. Everything in ``mocap.py`` and ``bvh.py`` builds on it.

Node numbering
--------------
Nodes ``0..69`` are the MHR70 keypoints in ``skeleton.JOINT_NAMES`` order.
Node ``70`` is a virtual root called ``hips``: the midpoint of ``left_hip``
(9) and ``right_hip`` (10). It is added because the model predicts no pelvis
joint, and a mocap rig needs one bone to carry the body's translation.

Hand chains
-----------
The finger keypoints are named from the tip inwards: ``*_tip``,
``*_first_joint``, ``*_second_joint``, ``*_third_joint`` -- so the *third*
joint is the one next to the wrist (the Goliath / Sapiens naming that MHR70
follows). The chains below are therefore ``wrist -> third -> second -> first
-> tip``. :func:`check_finger_order` tests that assumption against real
predictions and says so if the data disagree.

Coordinate systems
------------------
The model works in the camera frame: X right, Y **down**, Z away from the
camera (OpenCV). Animation tools expect Y **up**; the usual BVH convention is
Y up with the character facing +Z. ``CAM_TO_WORLD`` flips Y and Z, which turns
the camera frame into exactly that: X right, Y up, Z towards the camera, so a
person who faces the camera faces +Z. It is a proper rotation (determinant
+1), so left and right are preserved.

Units: metres everywhere in this package. ``bvh.py`` scales to centimetres
when it writes a file, because that is what most BVH readers expect.

Rest pose
---------
``REST_DIRECTIONS`` gives each bone's unit direction (from parent to child, in
the Y-up world frame) in a T-pose: standing on the floor, facing +Z, arms
straight out to the sides, palms down. Multiplied by a bone length it becomes
the BVH ``OFFSET``. The directions are only a convention: they decide what
"zero rotation" means in the exported file, not where the joints are.
"""

from __future__ import annotations

import numpy as np

from .skeleton import JOINT_NAMES, NUM_JOINTS

#: Index of the virtual root node.
ROOT = NUM_JOINTS            # 70
NUM_NODES = NUM_JOINTS + 1   # 71

#: Node names: the 70 MHR70 keypoints, then the virtual root.
NODE_NAMES: list[str] = list(JOINT_NAMES) + ["hips"]

#: Parent of every node (``-1`` for the root). Built from the bone list below
#: so the two can never disagree.
_BONES: list[tuple[int, int]] = [
    # (parent, child)
    # pelvis and legs
    (ROOT, 9), (9, 11), (11, 13), (13, 17), (13, 15), (13, 16),
    (ROOT, 10), (10, 12), (12, 14), (14, 20), (14, 18), (14, 19),
    # spine and head
    (ROOT, 69), (69, 0), (0, 1), (0, 2), (0, 3), (0, 4),
    # left arm
    (69, 5), (5, 67), (5, 7), (7, 63), (7, 65), (7, 62),
    # right arm
    (69, 6), (6, 68), (6, 8), (8, 64), (8, 66), (8, 41),
    # left hand: wrist -> third -> second -> first -> tip
    (62, 45), (45, 44), (44, 43), (43, 42),      # thumb
    (62, 49), (49, 48), (48, 47), (47, 46),      # index
    (62, 53), (53, 52), (52, 51), (51, 50),      # middle
    (62, 57), (57, 56), (56, 55), (55, 54),      # ring
    (62, 61), (61, 60), (60, 59), (59, 58),      # pinky
    # right hand
    (41, 24), (24, 23), (23, 22), (22, 21),      # thumb
    (41, 28), (28, 27), (27, 26), (26, 25),      # index
    (41, 32), (32, 31), (31, 30), (30, 29),      # middle
    (41, 36), (36, 35), (35, 34), (34, 33),      # ring
    (41, 40), (40, 39), (39, 38), (38, 37),      # pinky
]

PARENTS: list[int] = [-2] * NUM_NODES
PARENTS[ROOT] = -1
for _p, _c in _BONES:
    assert PARENTS[_c] == -2, f"node {_c} ({NODE_NAMES[_c]}) has two parents"
    PARENTS[_c] = _p
assert -2 not in PARENTS, [NODE_NAMES[i] for i, p in enumerate(PARENTS) if p == -2]
assert len(_BONES) == NUM_NODES - 1

#: Children of every node, in bone-list order.
CHILDREN: list[list[int]] = [[] for _ in range(NUM_NODES)]
for _p, _c in _BONES:
    CHILDREN[_p].append(_c)

#: Nodes ordered parents-first (the root is first), which is the order a
#: forward-kinematics pass -- and a BVH file -- visits them in.
TOPOLOGICAL_ORDER: list[int] = []


def _visit(node: int) -> None:
    TOPOLOGICAL_ORDER.append(node)
    for child in CHILDREN[node]:
        _visit(child)


_visit(ROOT)
assert len(TOPOLOGICAL_ORDER) == NUM_NODES

#: Each finger as ``(wrist, third, second, first, tip)``, used by
#: :func:`check_finger_order`.
FINGER_CHAINS: list[tuple[int, int, int, int, int]] = [
    (62, 45, 44, 43, 42), (62, 49, 48, 47, 46), (62, 53, 52, 51, 50),
    (62, 57, 56, 55, 54), (62, 61, 60, 59, 58),
    (41, 24, 23, 22, 21), (41, 28, 27, 26, 25), (41, 32, 31, 30, 29),
    (41, 36, 35, 34, 33), (41, 40, 39, 38, 37),
]

#: The keypoints of each foot (ankle, big toe, small toe, heel).
FOOT_NODES: dict[str, tuple[int, int, int, int]] = {
    "left": (13, 15, 16, 17),
    "right": (14, 18, 19, 20),
}

#: Camera frame (X right, Y down, Z forward) -> world frame (X right, Y up,
#: Z towards the camera). Apply as ``xyz_world = xyz_cam @ CAM_TO_WORLD``.
CAM_TO_WORLD = np.diag([1.0, -1.0, -1.0])


def _unit(x: float, y: float, z: float) -> np.ndarray:
    v = np.array([x, y, z], dtype=np.float64)
    return v / np.linalg.norm(v)


# T-pose bone directions, world frame (X right = the person's LEFT when they
# face the camera, Y up, Z towards the camera). Left-side entries are written
# out; the right side is mirrored in X below.
_LEFT_DIRECTIONS: dict[int, np.ndarray] = {
    9:  _unit(1.0, 0.0, 0.0),        # hips -> left_hip
    11: _unit(0.0, -1.0, 0.0),       # left_hip -> left_knee
    13: _unit(0.0, -1.0, 0.0),       # left_knee -> left_ankle
    17: _unit(0.0, -0.5, -1.0),      # left_ankle -> left_heel (down, back)
    15: _unit(0.1, -0.5, 1.0),       # left_ankle -> left_big_toe_tip (down, forward)
    16: _unit(0.4, -0.5, 0.9),       # left_ankle -> left_small_toe_tip (outward too)
    69: _unit(0.0, 1.0, 0.0),        # hips -> neck
    0:  _unit(0.0, 0.8, 0.6),        # neck -> nose (up, forward)
    1:  _unit(0.4, 0.5, -0.7),       # nose -> left_eye (up, back, outward)
    3:  _unit(0.9, 0.3, -0.5),       # nose -> left_ear
    5:  _unit(1.0, -0.2, 0.0),       # neck -> left_shoulder
    67: _unit(0.5, 0.85, 0.0),       # left_shoulder -> left_acromion (top of the shoulder)
    7:  _unit(1.0, 0.0, 0.0),        # left_shoulder -> left_elbow (arm out)
    63: _unit(0.0, 0.0, -1.0),       # left_elbow -> left_olecranon (back of the elbow)
    65: _unit(0.0, 0.0, 1.0),        # left_elbow -> left_cubital_fossa (front of the elbow)
    62: _unit(1.0, 0.0, 0.0),        # left_elbow -> left_wrist
    # fingers, palm down: thumb forward, pinky back
    45: _unit(0.7, 0.0, 0.7), 44: _unit(0.7, 0.0, 0.7), 43: _unit(0.7, 0.0, 0.7), 42: _unit(0.7, 0.0, 0.7),
    49: _unit(1.0, 0.0, 0.3), 48: _unit(1.0, 0.0, 0.3), 47: _unit(1.0, 0.0, 0.3), 46: _unit(1.0, 0.0, 0.3),
    53: _unit(1.0, 0.0, 0.0), 52: _unit(1.0, 0.0, 0.0), 51: _unit(1.0, 0.0, 0.0), 50: _unit(1.0, 0.0, 0.0),
    57: _unit(1.0, 0.0, -0.3), 56: _unit(1.0, 0.0, -0.3), 55: _unit(1.0, 0.0, -0.3), 54: _unit(1.0, 0.0, -0.3),
    61: _unit(0.9, 0.0, -0.5), 60: _unit(0.9, 0.0, -0.5), 59: _unit(0.9, 0.0, -0.5), 58: _unit(0.9, 0.0, -0.5),
}

#: Left node -> right node, for everything that has a mirror image.
MIRROR: dict[int, int] = {
    9: 10, 11: 12, 13: 14, 17: 20, 15: 18, 16: 19,
    1: 2, 3: 4, 5: 6, 67: 68, 7: 8, 63: 64, 65: 66, 62: 41,
    45: 24, 44: 23, 43: 22, 42: 21, 49: 28, 48: 27, 47: 26, 46: 25,
    53: 32, 52: 31, 51: 30, 50: 29, 57: 36, 56: 35, 55: 34, 54: 33,
    61: 40, 60: 39, 59: 38, 58: 37,
}

#: ``(NUM_NODES, 3)`` unit direction of the bone ending at each node, parent
#: -> node, in the T-pose described in the module docstring. Zero for the root.
REST_DIRECTIONS = np.zeros((NUM_NODES, 3), dtype=np.float64)
for _node, _d in _LEFT_DIRECTIONS.items():
    REST_DIRECTIONS[_node] = _d
    if _node in MIRROR:
        REST_DIRECTIONS[MIRROR[_node]] = _d * np.array([-1.0, 1.0, 1.0])
# mid-line nodes that are not in the left table: 69 (neck) and 0 (nose) are;
# everything else must now be filled.
for _node in range(NUM_NODES):
    if _node != ROOT:
        assert np.linalg.norm(REST_DIRECTIONS[_node]) > 0.99, NODE_NAMES[_node]


def add_root(joints: np.ndarray) -> np.ndarray:
    """Append the virtual root to ``(..., 70, 3)`` joints -> ``(..., 71, 3)``."""
    joints = np.asarray(joints)
    hips = 0.5 * (joints[..., 9, :] + joints[..., 10, :])
    return np.concatenate([joints, hips[..., None, :]], axis=-2)


def bone_lengths(nodes: np.ndarray) -> np.ndarray:
    """Length of the bone ending at each node, ``(..., 71)``; 0 for the root."""
    nodes = np.asarray(nodes)
    out = np.zeros(nodes.shape[:-1], dtype=np.float64)
    for node, parent in enumerate(PARENTS):
        if parent >= 0:
            out[..., node] = np.linalg.norm(nodes[..., node, :] - nodes[..., parent, :], axis=-1)
    return out


def check_finger_order(joints: np.ndarray) -> list[str]:
    """Test the ``wrist -> third -> second -> first -> tip`` assumption on data.

    ``joints`` is ``(N, 70, 3)`` or ``(70, 3)``. For every finger the median
    distance from the wrist must grow along the chain. Returns one message per
    finger that violates it (empty list = the assumption holds on this data).
    """
    joints = np.asarray(joints, dtype=np.float64)
    if joints.ndim == 2:
        joints = joints[None]
    problems = []
    for chain in FINGER_CHAINS:
        wrist = joints[:, chain[0]]
        dists = [float(np.median(np.linalg.norm(joints[:, k] - wrist, axis=-1)))
                 for k in chain[1:]]
        if any(b < a for a, b in zip(dists, dists[1:])):
            names = " -> ".join(JOINT_NAMES[k] for k in chain)
            problems.append(f"{names}: distances from the wrist are "
                            + ", ".join(f"{d * 100:.1f} cm" for d in dists))
    return problems
