"""Write a BVH motion file from joint positions, and read one back to check it.

BVH (Biovision Hierarchy) is the plain-text motion-capture format that
Blender, MotionBuilder, Unity (through Blender) and most animation tools read.
A file has two parts::

    HIERARCHY
    ROOT hips
    {
        OFFSET 0.000000 0.000000 0.000000
        CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
        JOINT left_hip
        {
            OFFSET 10.000000 0.000000 0.000000
            CHANNELS 3 Zrotation Xrotation Yrotation
            JOINT left_knee
            {
                ...
                End Site
                {
                    OFFSET 0.000000 -10.500000 0.000000
                }
            }
        }
        ...
    }
    MOTION
    Frames: 20
    Frame Time: 0.03333333
    <one line per frame: every channel of every joint, in hierarchy order>

``HIERARCHY`` is the skeleton in its rest pose: a tree of joints, each with an
``OFFSET`` (where it sits relative to its parent when every rotation is zero)
and a list of ``CHANNELS`` (the numbers each ``MOTION`` line supplies for it).
``MOTION`` is the animation: one value per channel per frame. A reader
rebuilds a pose by forward kinematics, visiting joints parents-first::

    R_global[j]  = R_global[parent] @ R_local[j]        # R_local from j's channels
    position[j]  = position[parent] + R_global[parent] @ OFFSET[j]

Note that a joint is moved by its *parent's* rotation; its own rotation moves
its children. So the file stores bone *lengths* once, in the offsets, and
bone *directions* per frame, as rotations. That is the difficulty of
exporting a positional skeleton: the model gives positions, the file wants
rotations. :func:`solve_frame` does that conversion (see below).

Conventions this writer uses
----------------------------
* Units: callers pass metres (this package's unit everywhere); ``scale``
  converts to file units. The default 100 writes centimetres, which is what
  most BVH files in the wild use and what most readers assume. (Blender's
  importer has a Scale option: 0.01 brings a centimetre file back to its
  metres.)
* Axes: whatever the caller passes; nothing is enforced. ``mocap.py`` passes
  the Y-up world frame of ``mocap_skeleton.CAM_TO_WORLD`` (X right, Y up, Z
  towards the camera), which is the usual BVH convention.
* The root has ``OFFSET 0 0 0`` and six channels: three positions, then the
  rotations in the classic Biovision order ``Zrotation Xrotation Yrotation``.
  Its position channels carry the root's absolute position in file units,
  not a delta from the offset (the offset is zero).
* Every other joint has ``OFFSET`` = its rest-pose vector from its parent in
  file units, and the same three rotation channels. The file also has one
  extra joint per child of every fork but one, ``<fork>_to_<child>``, with
  ``OFFSET 0 0 0`` (see "Forks" below), so it holds more joints than the
  caller's tree: 96 for the 71-node mocap tree.
* Every leaf gets an ``End Site``: a stub continuing the leaf's own bone for
  ``end_site_fraction`` of its length, at most ``end_site_max`` (1 cm), so
  readers that draw a bone from each joint to its child do not make a
  zero-length bone at the tips. The cap keeps the stubs short where the leaf's
  bone is long: a quarter of the ankle-to-toe bone ran 4-5 cm past the toes,
  down along the foot, and ended 1.4-2.5 cm lower than the toe, under the
  floor whenever the toe stood on it.
* Angles are in degrees, as the format requires. The channel order fixes how a
  reader composes them -- channels multiply in the listed order, so
  ``Zrotation Xrotation Yrotation`` builds ``R = Rz(z) @ Rx(x) @ Ry(y)`` --
  and :func:`matrix_to_euler_zxy` is the exact inverse of that.
* Joints are written depth-first, children in ascending index order. The
  reader returns joints in file order, which is generally *not* the caller's
  index order; match them up by name.

From positions to rotations
---------------------------
For every joint ``j`` the writer needs the global rotation ``G_j`` that takes
each child's rest offset onto the child's current direction from ``j``. It
visits joints parents-first. With ``a_c`` the unit rest direction of child
``c``, ``b_c`` its unit current direction ``pos[c] - pos[j]`` and ``G_p`` the
parent's solved global rotation (identity for the root):

* **Fork** -- two or more children whose rest directions are not collinear:
  ``G_j`` is the Kabsch rotation, the proper rotation (determinant +1) that
  minimises ``sum_c L_c^2 |G a_c - b_c|^2``, where ``L_c`` is the rest length
  of bone ``c`` -- in other words, the rotation that best maps the rest
  offsets onto the current directions drawn at rest length. Two non-collinear
  directions pin a rotation down completely, so for a rigid motion of the
  skeleton every fork -- pelvis, neck, shoulders, elbows, wrists, ankles,
  head -- gets its true orientation, twist included. The length weighting
  only matters on noisy input, and there it matters a lot: with 1 cm of
  noise a 3 cm landmark (the cubital fossa) is 20 degrees off while the 26 cm
  forearm next to it is under 3 degrees off. With equal weights the landmark
  steers the elbow's orientation and that error is multiplied by the forearm
  and the whole hand
  (measured on the test skeleton with 1 cm-per-axis Gaussian noise:
  fingertips up to 19.5 cm off with equal weights, 8.4 cm with length
  weights).
* **Chain** -- one child (or several collinear ones): a single bone direction
  says nothing about the twist about that bone, so the twist is inherited
  from the parent: ``G_j = swing(G_p a_c -> b_c) @ G_p``, where ``swing(u ->
  v)`` is the smallest rotation taking unit ``u`` to unit ``v`` (Rodrigues'
  formula; the identity when ``u ~ v``, a half turn about any axis
  perpendicular to ``u`` when ``u ~ -v``). ``G_p`` already moves the bone to
  ``G_p a_c``; the swing fixes only the remaining direction error.
* **Leaf**: nothing to measure, ``G_j = G_p``.

Forks
-----
One rotation at a fork places every child only if the angles between the
children's current directions equal the angles between their rest
directions, and on real data they do not. The rest directions are a
hand-written T-pose, not this person's anatomy, and ordinary motion moves the
children of a fork against each other: the head nods against the shoulders
at the neck, the spine bends against the hip line at the pelvis, the thumb
moves against the palm. Whatever the Kabsch fit leaves over would be stored
nowhere: the child would sit off its measured position and carry its whole
subtree with it. Measured with exact bone lengths and no noise: bone
directions 10 degrees off the rest ones put the wrists 2.8 cm off in every
frame of a person standing still, a 60 degree head turn put the face 6.1 cm
off, and a torso sway with planted feet made the feet bob 3.2 cm in the file
although they did not move at all in the input.

So :func:`write_bvh` puts a joint ``<fork>_to_<child>`` between a fork and
each of its children except one, the *main* child: the child that continues
the fork's own bone most nearly straight on (the head at the neck, the
middle finger at the wrist; for the root, the spine). A ``<fork>_to_<child>``
joint sits on the fork (``OFFSET 0 0 0``) and has the one child, so it is a
chain joint: its rotation turns that child's bone onto the child's measured
direction. The fork's Kabsch fit looks through these joints to the real
children, and is then turned by the small remaining swing that puts the main
child exactly on its measured position. So every bone has its own rotation,
the fork keeps its Kabsch orientation up to that small correction (none for
a rigid motion, where every ``<fork>_to_<child>`` rotation is the identity
too), and every joint lands exactly.

The main child is why the fork joint itself still has a real bone in readers
that draw one from each joint towards its children: Blender's importer ends
a joint's bone at the average of its children's heads, which for a fork is
the fork plus the main child's offset divided by the number of children --
a short bone pointing at the main child. (With a ``<fork>_to_<child>`` joint
for every child that average was the fork itself, a zero-length bone, which
Blender replaces with a copy of the parent bone: spikes of 30 to 50 cm out
of the head, through the floor at the ankles and past the fingertips.)
The 71-node mocap tree has 11 forks (the root among them) with 36 children,
so the file has 71 + 36 - 11 = 96 joints.

The file stores the local rotation ``L_j = G_p^T @ G_j`` (``L = G`` for the
root) as ZXY Euler angles. With the fork joints above, every bone keeps its
measured *direction* exactly and gets its *rest length*, so whenever the
input's bone lengths equal the rest lengths -- which ``mocap.py``'s
bone-length normalisation guarantees -- ``fk(read_bvh(path))`` returns the
input positions to rounding error, whatever the pose. Input whose lengths
disagree with the rest offsets (per-frame noise) comes back with those length
errors accumulated along each chain. Clean the positions first; that is
``mocap.py``'s job.

:func:`read_bvh` and :func:`fk` are a small parser and forward-kinematics
pass for checking a written file. They handle any channel order, and joint
names must not contain whitespace.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

_AXES = {"X": 0, "Y": 1, "Z": 2}


def _axis_rotation(axis: str, degrees) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` about one axis, angles in degrees."""
    t = np.radians(np.asarray(degrees, dtype=np.float64))
    c, s = np.cos(t), np.sin(t)
    R = np.zeros(t.shape + (3, 3))
    i = _AXES[axis]
    j, k = (i + 1) % 3, (i + 2) % 3     # the two axes that turn
    R[..., i, i] = 1.0
    R[..., j, j] = c
    R[..., j, k] = -s
    R[..., k, j] = s
    R[..., k, k] = c
    return R


def euler_zxy_to_matrix(z: float, x: float, y: float) -> np.ndarray:
    """``Rz(z) @ Rx(x) @ Ry(y)`` for degrees: what a BVH reader builds for the
    channels ``Zrotation Xrotation Yrotation``."""
    return _axis_rotation("Z", z) @ _axis_rotation("X", x) @ _axis_rotation("Y", y)


def matrix_to_euler_zxy(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of :func:`euler_zxy_to_matrix`: ``(z, x, y)`` in degrees.

    Writing out ``Rz Rx Ry``, the last row is ``(-cos x sin y, sin x, cos x
    cos y)`` and the middle column is ``(-sin z cos x, cos z cos x, sin x)``,
    so ``x`` comes from ``R[2, 1]``, ``y`` from the last row and ``z`` from the
    middle column. When ``cos x ~ 0`` (``x = +-90`` degrees, gimbal lock) the
    Z and Y rotations act about the same axis and only their sum (or
    difference) is determined; we put all of it in ``z`` and set ``y = 0``,
    which still reproduces the matrix. The threshold 1e-8 is where the two
    formulas are equally precise (both lose about ``eps / cos x``).
    """
    R = np.asarray(R, dtype=np.float64)
    cx = np.hypot(R[2, 0], R[2, 2])
    x = np.arctan2(R[2, 1], cx)
    if cx < 1e-8:
        z = np.arctan2(R[1, 0], R[0, 0])
        y = 0.0
    else:
        z = np.arctan2(-R[0, 1], R[1, 1])
        y = np.arctan2(-R[2, 0], R[2, 2])
    return float(np.degrees(z)), float(np.degrees(x)), float(np.degrees(y))


def swing(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The smallest rotation taking unit vector ``u`` onto unit vector ``v``.

    Rodrigues' formula about the axis ``u x v``: with ``K`` the cross-product
    matrix of that (unnormalised) axis and ``c = u . v``, ``R = I + K + K^2 /
    (1 + c)``. When ``u ~ -v`` the axis vanishes and ``1 + c`` goes to zero,
    so that case is a half turn about an axis perpendicular to ``u`` (any
    one works; we take ``u x X``, or ``u x Y`` if ``u`` is nearly ``X``).
    """
    u = np.asarray(u, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    c = float(u @ v)
    if c < -1.0 + 1e-10:
        p = np.cross(u, [1.0, 0.0, 0.0] if abs(u[0]) < 0.9 else [0.0, 1.0, 0.0])
        p /= np.linalg.norm(p)
        return 2.0 * np.outer(p, p) - np.eye(3)
    ax = np.cross(u, v)
    K = np.array([[0.0, -ax[2], ax[1]], [ax[2], 0.0, -ax[0]], [-ax[1], ax[0], 0.0]])
    return np.eye(3) + K + K @ K / (1.0 + c)


def kabsch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """The proper rotation ``R`` minimising ``sum_k |R a_k - b_k|^2`` for
    matched rows ``a`` and ``b`` of shape ``(K, 3)``.

    Kabsch's algorithm: with ``H = a^T b = U S V^T``, ``R = V U^T``. If that
    has determinant -1 it is a reflection, and the best *rotation* flips the
    least significant singular direction instead.
    """
    U, _, Vt = np.linalg.svd(np.asarray(a, dtype=np.float64).T @ np.asarray(b, dtype=np.float64))
    d = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0.0 else -1.0
    return Vt.T @ np.diag([1.0, 1.0, d]) @ U.T


def solve_frame(parents: Sequence[int], rest_offsets: np.ndarray, pos: np.ndarray,
                order: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Rotations for one frame of joint positions: ``(global, local)``, each
    ``(J, 3, 3)``. ``order`` lists the joints parents-first. The rules are in
    the module docstring ("From positions to rotations")."""
    rest_offsets = np.asarray(rest_offsets, dtype=np.float64)
    pos = np.asarray(pos, dtype=np.float64)
    children: list[list[int]] = [[] for _ in parents]
    for c, p in enumerate(parents):
        if p >= 0:
            children[p].append(c)
    global_rot = np.zeros((len(parents), 3, 3))
    local_rot = np.zeros((len(parents), 3, 3))
    for j in order:
        p = parents[j]
        G_p = global_rot[p] if p >= 0 else np.eye(3)
        rest, cur, lengths = [], [], []      # unit rest / current directions, rest lengths
        for c in children[j]:
            # A zero-offset joint with one child sits on j (see "Forks"):
            # measure through it, to the child it leads to.
            g = children[c][0] if not rest_offsets[c].any() and len(children[c]) == 1 else c
            a, b = rest_offsets[g], pos[g] - pos[j]
            na, nb = np.linalg.norm(a), np.linalg.norm(b)
            if na > 0.0 and nb > 0.0:
                rest.append(a / na)
                cur.append(b / nb)
                lengths.append(na)
        if len(rest) >= 2 and np.linalg.matrix_rank(np.array(rest), tol=1e-6) >= 2:
            w = np.array(lengths)[:, None]
            G = kabsch(np.array(rest) * w, np.array(cur) * w)
            # A fork whose main child hangs on it directly (write_bvh's layout,
            # "Forks"): turn the fit by the little that is left so that child
            # lands exactly; the other children have joints of their own.
            direct = [c for c in children[j]
                      if rest_offsets[c].any() or len(children[c]) != 1]
            if len(direct) == 1:
                m = direct[0]
                a, b = rest_offsets[m], pos[m] - pos[j]
                na, nb = np.linalg.norm(a), np.linalg.norm(b)
                if na > 0.0 and nb > 0.0:
                    G = swing(G @ (a / na), b / nb) @ G
        elif rest:
            G = swing(G_p @ rest[0], cur[0]) @ G_p
        else:
            G = G_p
        global_rot[j] = G
        local_rot[j] = G_p.T @ G
    return global_rot, local_rot


def _fmt(values) -> str:
    # ``+ 0.0`` turns -0.0 into 0.0 so the file never says "-0.000000".
    return " ".join(f"{float(v) + 0.0:.6f}" for v in values)


def _main_child(j: int, kids: list[list[int]], parents: Sequence[int],
                rest_offsets: np.ndarray) -> int:
    """The child of fork ``j`` that hangs on it directly in the file: the one
    that continues ``j``'s own bone most nearly straight on (the middle
    finger at the wrist, the head at the neck). The root has no bone of its
    own, so it keeps the child with the most joints below it (the spine).
    Ties go to the larger subtree, then the lower index."""
    def size(c: int) -> int:
        return 1 + sum(size(g) for g in kids[c])

    def unit(v: np.ndarray) -> np.ndarray:
        n = np.linalg.norm(v)
        return v / n if n > 0.0 else v

    own = unit(rest_offsets[j]) if parents[j] >= 0 else np.zeros(3)
    return max(kids[j], key=lambda c: (round(float(unit(rest_offsets[c]) @ own), 6), size(c), -c))


def write_bvh(path, names: Sequence[str], parents: Sequence[int], rest_offsets: np.ndarray,
              positions: np.ndarray, fps: float, scale: float = 100.0,
              end_site_fraction: float = 0.25, end_site_max: float = 0.01) -> None:
    """Write ``positions`` ``(N, J, 3)`` (metres) of the tree ``names`` /
    ``parents`` (exactly one ``parents[j] == -1``) as a BVH file at ``path``.

    ``rest_offsets[j]`` is the rest-pose vector from ``parents[j]`` to ``j``
    in metres (zero for the root); it becomes the joint's ``OFFSET`` after
    ``scale``. ``scale`` converts metres to file units (100 = centimetres).
    A leaf's ``End Site`` is ``end_site_fraction`` of its rest offset, at most
    ``end_site_max`` metres long.
    See the module docstring for the file layout, the rotation solve, and the
    ``<fork>_to_<child>`` joints the file gets on top of ``names``.
    """
    names, parents = list(names), list(parents)
    rest_offsets = np.asarray(rest_offsets, dtype=np.float64)
    positions = np.asarray(positions, dtype=np.float64)
    J = len(names)
    roots = [j for j, p in enumerate(parents) if p < 0]
    if len(roots) != 1 or len(parents) != J:
        raise ValueError(f"expected one root and {J} parents, got roots {roots}")
    if rest_offsets.shape != (J, 3) or positions.ndim != 3 or positions.shape[1:] != (J, 3):
        raise ValueError(f"rest_offsets {rest_offsets.shape} / positions {positions.shape} "
                         f"do not match {J} joints")
    # A zero-offset joint between every fork and each of its children but the
    # main one, so every bone gets its own rotation (module docstring, "Forks").
    kids = [[c for c, p in enumerate(parents) if p == j] for j in range(J)]
    rest_list, pos_list = list(rest_offsets), list(positions.transpose(1, 0, 2))
    for j in range(J):
        if len(kids[j]) >= 2:
            main = _main_child(j, kids, parents, rest_offsets)
            for c in kids[j]:
                if c == main:
                    continue
                names.append(f"{names[j]}_to_{names[c]}")
                parents.append(j)
                parents[c] = len(names) - 1
                rest_list.append(np.zeros(3))
                pos_list.append(pos_list[j])
    rest_offsets, positions, J = np.array(rest_list), np.stack(pos_list, axis=1), len(names)
    root = roots[0]
    children: list[list[int]] = [[] for _ in range(J)]
    for c, p in enumerate(parents):
        if p >= 0:
            children[p].append(c)         # ascending index order

    lines = ["HIERARCHY"]
    order: list[int] = []                 # depth-first, the order the channels appear in

    def visit(j: int, depth: int) -> None:
        order.append(j)
        tab = "\t" * depth
        if j == root:
            lines.append(f"{tab}ROOT {names[j]}")
            lines.append(f"{tab}{{")
            lines.append(f"{tab}\tOFFSET 0.000000 0.000000 0.000000")
            lines.append(f"{tab}\tCHANNELS 6 Xposition Yposition Zposition "
                         "Zrotation Xrotation Yrotation")
        else:
            lines.append(f"{tab}JOINT {names[j]}")
            lines.append(f"{tab}{{")
            lines.append(f"{tab}\tOFFSET {_fmt(rest_offsets[j] * scale)}")
            lines.append(f"{tab}\tCHANNELS 3 Zrotation Xrotation Yrotation")
        for c in children[j]:
            visit(c, depth + 1)
        if not children[j]:
            lines.append(f"{tab}\tEnd Site")
            lines.append(f"{tab}\t{{")
            stub = rest_offsets[j] * end_site_fraction
            stub *= min(1.0, end_site_max / max(float(np.linalg.norm(stub)), 1e-12))
            lines.append(f"{tab}\t\tOFFSET {_fmt(stub * scale)}")
            lines.append(f"{tab}\t}}")
        lines.append(f"{tab}}}")

    visit(root, 0)

    lines += ["MOTION", f"Frames: {len(positions)}", f"Frame Time: {1.0 / fps:.8f}"]
    for pos in positions:
        _, local = solve_frame(parents, rest_offsets, pos, order)
        values = list(pos[root] * scale)
        for j in order:
            values.extend(matrix_to_euler_zxy(local[j]))
        lines.append(_fmt(values))
    Path(path).write_text("\n".join(lines) + "\n")


@dataclass
class Bvh:
    """A parsed BVH file. Joints are in file order (parents before children),
    offsets and motion in file units."""

    names: list[str]
    parents: list[int]
    offsets: np.ndarray                  # (J, 3)
    channels: list[list[str]]            # per joint, e.g. ["Zrotation", "Xrotation", "Yrotation"]
    end_sites: dict[int, np.ndarray]     # joint index -> End Site offset
    frame_time: float
    motion: np.ndarray                   # (N, C): one row per frame, columns in channel order


def _expect(tokens: list[str], i: int, word: str) -> None:
    if tokens[i] != word:
        raise ValueError(f"BVH parse error: expected {word!r}, got {tokens[i]!r}")


def read_bvh(path) -> Bvh:
    """Parse a BVH file. Reads what :func:`write_bvh` writes plus any channel
    order; whitespace is the only separator, so joint names cannot contain it."""
    tokens = Path(path).read_text().split()
    names: list[str] = []
    parents: list[int] = []
    offsets: list[np.ndarray] = []
    channels: list[list[str]] = []
    end_sites: dict[int, np.ndarray] = {}
    stack: list[int] = []                # joints whose "{" is still open
    in_end_site = False
    _expect(tokens, 0, "HIERARCHY")
    i = 1
    while tokens[i] != "MOTION":
        tok = tokens[i]
        i += 1
        if tok in ("ROOT", "JOINT"):
            names.append(tokens[i])
            parents.append(stack[-1] if stack else -1)
            offsets.append(np.zeros(3))
            channels.append([])
            _expect(tokens, i + 1, "{")
            i += 2
            stack.append(len(names) - 1)
        elif tok == "OFFSET":
            offset = np.array([float(t) for t in tokens[i:i + 3]])
            i += 3
            if in_end_site:
                end_sites[stack[-1]] = offset
            else:
                offsets[stack[-1]] = offset
        elif tok == "CHANNELS":
            n = int(tokens[i])
            channels[stack[-1]] = tokens[i + 1:i + 1 + n]
            i += 1 + n
        elif tok == "End":                # "End Site {"
            _expect(tokens, i + 1, "{")
            i += 2
            in_end_site = True
        elif tok == "}":
            if in_end_site:
                in_end_site = False
            else:
                stack.pop()
        else:
            raise ValueError(f"BVH parse error: unexpected token {tok!r}")
    i += 1
    _expect(tokens, i, "Frames:")
    n_frames = int(tokens[i + 1])
    _expect(tokens, i + 2, "Frame")
    _expect(tokens, i + 3, "Time:")
    frame_time = float(tokens[i + 4])
    n_channels = sum(len(c) for c in channels)
    motion = np.array(tokens[i + 5:], dtype=np.float64).reshape(n_frames, n_channels)
    return Bvh(names, parents, np.array(offsets), channels, end_sites, frame_time, motion)


def fk(bvh: Bvh) -> np.ndarray:
    """Joint positions ``(N, J, 3)`` in file units by standard BVH forward
    kinematics (the two lines in the module docstring). Position channels add
    to the joint's offset; rotation channels multiply in the listed order."""
    N, J = len(bvh.motion), len(bvh.names)
    positions = np.zeros((N, J, 3))
    global_rot = np.zeros((N, J, 3, 3))
    col = 0
    for j in range(J):                   # file order: parents come first
        offset = np.tile(bvh.offsets[j], (N, 1))
        R = np.tile(np.eye(3), (N, 1, 1))
        for ch in bvh.channels[j]:
            values = bvh.motion[:, col]
            col += 1
            if ch.lower().endswith("position"):
                offset[:, _AXES[ch[0].upper()]] += values
            else:
                R = R @ _axis_rotation(ch[0].upper(), values)
        p = bvh.parents[j]
        if p < 0:
            positions[:, j] = offset
            global_rot[:, j] = R
        else:
            global_rot[:, j] = global_rot[:, p] @ R
            positions[:, j] = positions[:, p] + np.einsum("nij,nj->ni", global_rot[:, p], offset)
    return positions
