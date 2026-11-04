"""Motion capture from per-frame InstantHMR predictions.

``demo.py`` runs the model frame by frame and shows the result. This module is
the layer that turns those per-frame, per-person predictions into motion-capture
*tracks* -- one clean skeleton animation per person -- and writes them to
standard files (BVH for Blender, plus JSON / NPZ / CSV).

How a clip becomes a track
--------------------------
:class:`MocapRecorder` receives every processed frame through :meth:`push`,
groups the predictions by person (``track_id``), and :meth:`finish` runs these
steps on each person, in this order:

1. **Fill gaps** (:func:`fill_gaps`). The detector misses a person now and then,
   so a track has holes. Short holes (``max_gap`` frames or fewer) are linearly
   interpolated; longer ones hold the last value. ``TrackData.valid`` is False on
   every filled-in frame so downstream code can tell real data from guesses.
   The rig's angles (``mhr_params[3:130]``) are interpolated the short way
   round, so a hole across a +-pi wrap does not spin the body. The time
   stamps are the exception: time is linear in the frame number, so
   they are interpolated whatever the length of the hole. (The tracker that
   gives out ids retires a person after ``max_gap`` missed frames, so from the
   demo a longer hole arrives as a new track rather than as a held pose.)

2. **Add the root and go to a Y-up world** (:func:`mocap_skeleton.add_root`,
   :func:`to_world`). The model predicts no pelvis, so a virtual ``hips`` node
   (the midpoint of the two hips) is appended as node 70; that is the bone that
   carries the body's translation in a rig. The prediction is in the camera
   frame (X right, Y down, Z away from the camera, metres); ``CAM_TO_WORLD``
   flips Y and Z so the result is X right, Y up, Z towards the camera -- the
   frame animation tools expect, with a person facing the camera facing +Z.

3. **Normalise bone lengths** (:func:`normalise_bone_lengths`). Every frame is
   estimated on its own, so a thigh that is 42.0 cm in one frame is 41.3 cm in
   the next; a rig cannot animate that -- a bone has ONE length. The median
   length of each bone over the person's valid frames is taken as that person's
   skeleton, and every frame is rebuilt with those lengths, keeping each bone's
   predicted direction. The root keeps its predicted position.

4. **Level the floor and put it at y = 0** (:func:`estimate_pitch`,
   :func:`estimate_floor`). A camera on a shelf or a tripod usually looks down
   a little, and in the camera's frame the floor is then a slope: far feet sit
   higher than near ones, so one floor height would be wrong for most of the
   clip and a walking person would sink into the ground. When a person
   covered at least a metre of depth (5th to 95th percentile), the slope of
   the lowest foot keypoint against depth gives the camera's pitch and the
   whole world is rotated back level; with less depth the slope is noise, and
   the camera is assumed level.
   Then one floor height for the whole clip: the 5th percentile, over all
   tracks and valid frames, of the lowest foot keypoint. A percentile rather
   than the minimum so that one bad frame cannot sink the floor. Subtracting
   it makes "standing on the ground" mean y = 0, which is what the next step
   and every animation tool assume.

5. **Anchor the root with the feet** (:func:`anchor_root`). This is the step
   that makes the result usable. ``cam_trans`` -- the model's estimate of where
   the person is in depth -- is a free 3-vector with little tying it to pixels
   (see ``smoothing.py``), so it wobbles by centimetres from frame to frame and
   a standing person appears to slide back and forth. A foot that is standing
   on the floor is the one thing we know is NOT moving, so for every frame the
   whole body is shifted by whatever amount keeps the planted foot exactly where
   it was. The shift is held while both feet are in the air, so walking and
   jumping still move the body as predicted. Only x and z are shifted: the
   height is left to the floor levelling, because a shift that is held from
   one plant to the next would add the model's height error at every plant to
   a running sum that never decays, and the body would float away from the
   floor it was just put on. The per-frame shift is kept in
   ``TrackData.root_offset`` so the correction is always recoverable. This
   step needs step 4: "standing on the floor" means y near 0, so with
   ``floor=False`` it is off too.

Coordinate frames in :class:`TrackData`:

* ``joints_cam``   -- the prediction as received: camera frame, Y down, metres.
* ``joints_world`` -- the mocap result: X right, Y up, Z towards the camera,
  bone lengths normalised, camera pitch removed, floor at y = 0, root anchored
  by the feet in x and z.
* ``joints_2d``    -- full-frame pixels, origin top-left.

Frames and samples: the demo may process every k-th video frame
(``--frame-skip k``) and then pushes frame numbers 0, k, 2k, ... Inside this
module a *sample* is ``frame_idx // frame_step``, so that "consecutive" means
consecutive processed frames; ``TrackData.frames`` is reported in the ORIGINAL
frame numbers so it lines up with the video.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .inference import HMRPrediction
from .mocap_skeleton import (
    CAM_TO_WORLD, FOOT_NODES, NODE_NAMES, NUM_NODES, PARENTS, REST_DIRECTIONS,
    ROOT, TOPOLOGICAL_ORDER, add_root, bone_lengths, check_finger_order,
)

#: Per-frame fields stored for every pushed person, with the dtype they are
#: kept in. ``contact`` is only present when the model has it.
_FIELD_DTYPES: dict[str, type] = {
    "times": np.float64,
    "confidence": np.float32,
    "bbox": np.float32,
    "joints_2d": np.float32,
    "joints_3d_cam": np.float32,
    "cam_trans": np.float32,
    "mhr_params": np.float32,
    "shape_params": np.float32,
    "contact": np.float32,
}

_FEET = (FOOT_NODES["left"], FOOT_NODES["right"])


@dataclass
class TrackData:
    """One person's clip after processing. ``N`` frames, consecutive."""

    track_id: int             # 0, 1, ... in order of first appearance
    frames: np.ndarray        # (N,) int64, ORIGINAL frame numbers, steps of frame_step
    times: np.ndarray         # (N,) float64 seconds
    valid: np.ndarray         # (N,) bool, False on frames that were filled in
    confidence: np.ndarray    # (N,) float32
    bbox: np.ndarray          # (N,4) float32
    joints_2d: np.ndarray     # (N,70,2) float32 pixels
    joints_cam: np.ndarray    # (N,71,3) float32 camera frame, Y down, root appended
    joints_world: np.ndarray  # (N,71,3) float32 Y-up world, normalised, levelled, floor y=0, anchored
    cam_trans: np.ndarray     # (N,3) float32
    mhr_params: np.ndarray    # (N,204) float32
    shape_params: np.ndarray  # (N,45) float32
    contact: np.ndarray | None  # (N,2) float32 foot-contact probability, or None
    bone_lengths: np.ndarray  # (71,) float64 median bone length per node, 0 for the root
    root_offset: np.ndarray   # (N,3) float32 translation added by the foot anchoring


# ----------------------------------------------------------------------
# Pure functions, one per processing step
# ----------------------------------------------------------------------

def to_world(xyz: np.ndarray) -> np.ndarray:
    """Camera frame (Y down, Z forward) -> world frame (Y up, Z towards the
    camera). Works on any ``(..., 3)`` array."""
    return np.asarray(xyz) @ CAM_TO_WORLD


def fill_gaps(
    samples: np.ndarray, arrays: dict[str, np.ndarray], max_gap: int,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    """Make a track's sample range consecutive.

    ``samples`` is ``(M,)`` ints (any order) and every array in ``arrays`` is
    ``(M, ...)`` in the same order. Returns the consecutive samples
    ``first..last`` (``N`` of them), the arrays resampled to ``(N, ...)``, and a
    ``(N,)`` mask that is True where a sample was really pushed. Gaps of at most
    ``max_gap`` missing samples are linearly interpolated; longer gaps hold the
    value before the gap. If a sample was pushed twice the later one wins.
    """
    samples = np.asarray(samples, dtype=np.int64)
    order = np.argsort(samples, kind="stable")
    samples = samples[order]
    first, last = int(samples[0]), int(samples[-1])
    n = last - first + 1
    valid = np.zeros(n, dtype=bool)
    valid[samples - first] = True

    out: dict[str, np.ndarray] = {}
    for name, arr in arrays.items():
        arr = np.asarray(arr)[order]
        filled = np.empty((n,) + arr.shape[1:], dtype=arr.dtype)
        filled[samples - first] = arr
        for i in range(len(samples) - 1):
            gap = int(samples[i + 1] - samples[i]) - 1
            if gap <= 0:
                continue
            lo = int(samples[i]) - first + 1
            if gap <= max_gap:
                a0 = arr[i].astype(np.float64)
                a1 = arr[i + 1].astype(np.float64)
                w = np.arange(1, gap + 1, dtype=np.float64) / (gap + 1)
                w = w.reshape((gap,) + (1,) * (arr.ndim - 1))
                filled[lo:lo + gap] = a0 + w * (a1 - a0)
            else:
                filled[lo:lo + gap] = arr[i]
        out[name] = filled
    return np.arange(first, last + 1, dtype=np.int64), out, valid


def normalise_bone_lengths(nodes: np.ndarray, lengths: np.ndarray) -> np.ndarray:
    """Rebuild ``(..., 71, 3)`` skeletons so every bone has ``lengths[node]``.

    Walks the tree parents-first: the root keeps its position, every other node
    is placed at its (new) parent plus its *predicted* direction times the
    target length. A bone of zero predicted length has no direction, so it
    takes its T-pose direction (``REST_DIRECTIONS``) instead.
    """
    nodes = np.asarray(nodes, dtype=np.float64)
    shape = nodes.shape
    nodes = nodes.reshape(-1, NUM_NODES, 3)
    lengths = np.asarray(lengths, dtype=np.float64)
    out = nodes.copy()
    for node in TOPOLOGICAL_ORDER[1:]:
        parent = PARENTS[node]
        d = nodes[:, node] - nodes[:, parent]
        norm = np.linalg.norm(d, axis=-1, keepdims=True)
        unit = np.where(norm > 0.0, d / np.where(norm > 0.0, norm, 1.0),
                        REST_DIRECTIONS[node])
        out[:, node] = out[:, parent] + unit * lengths[node]
    return out.reshape(shape)


def estimate_floor(
    joints_world_list: Sequence[np.ndarray], valid_list: Sequence[np.ndarray],
    percentile: float = 5.0,
) -> float:
    """Floor height (world y) for a clip: the ``percentile``-th percentile, over
    all tracks and valid frames, of the lowest foot keypoint. 0.0 when there is
    no valid frame at all."""
    foot_nodes = list(FOOT_NODES["left"]) + list(FOOT_NODES["right"])
    lows = [np.asarray(j)[np.asarray(v)][:, foot_nodes, 1].min(axis=-1)
            for j, v in zip(joints_world_list, valid_list)]
    lows = np.concatenate(lows) if lows else np.zeros(0)
    return float(np.percentile(lows, percentile)) if lows.size else 0.0


def _lowest_foot_points(joints_world: np.ndarray, valid: np.ndarray) -> np.ndarray:
    """``(K, 3)``: the lowest foot keypoint of every valid frame of one track."""
    foot_nodes = list(FOOT_NODES["left"]) + list(FOOT_NODES["right"])
    feet = np.asarray(joints_world, dtype=np.float64)[np.asarray(valid)][:, foot_nodes]
    return feet[np.arange(len(feet)), feet[:, :, 1].argmin(axis=1)]


def estimate_pitch(
    joints_world_list: Sequence[np.ndarray], valid_list: Sequence[np.ndarray],
    min_depth_range: float = 1.0,
) -> float:
    """The camera's pitch in radians, positive when it looks down, from how the
    lowest foot keypoint's height changes with depth (world z).

    A pitched camera sees a level floor as a slope ``y = slope * z + c``; a
    straight-line fit of the lowest foot keypoints gives the slope, refitted
    once without the points more than 5 cm above the line (feet in the air).
    Only pitch is estimated, not roll: a camera on a shelf or a tripod is
    tilted down, not sideways. Only a track whose own points span at least
    ``min_depth_range`` metres between the 5th and 95th percentile of depth
    takes part, and the result is 0.0 when none does: a person standing still
    gives a slope that is all noise, two people standing at different depths
    give none at all, and a level camera is the safer assumption.

    Percentiles, not the minimum and maximum, and only the points between
    them enter the fit: a depth error moves a person ALONG the camera ray,
    which in the (z, y) plane is itself a slope. With max - min, one frame
    whose depth is badly wrong (a person half out of the picture at the start
    of a track) opens the gate for a standing person, and as the one far-off
    point it then decides the slope of the whole fit: one frame 1.3x too far
    among 300 of a person standing still read as a camera pitched 17 degrees.
    """
    points = []
    for j, v in zip(joints_world_list, valid_list):
        p = _lowest_foot_points(j, v)
        if len(p) < 2:
            continue
        lo, hi = np.percentile(p[:, 2], [5, 95])
        if hi - lo >= min_depth_range:
            points.append(p[(p[:, 2] >= lo) & (p[:, 2] <= hi)])
    if not points:
        return 0.0
    points = np.concatenate(points)
    z, y = points[:, 2], points[:, 1]
    slope, intercept = np.polyfit(z, y, 1)
    on_floor = y - (slope * z + intercept) < 0.05
    if on_floor.sum() >= 2:
        slope, intercept = np.polyfit(z[on_floor], y[on_floor], 1)
    return float(-np.arctan(slope))


def _rotation_about_x(angle: float) -> np.ndarray:
    """``(3, 3)`` rotation by ``angle`` radians about the x axis; apply to row
    vectors as ``xyz @ R.T``."""
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def _foot_positions(joints: np.ndarray) -> np.ndarray:
    """``(N, 2, 3)``: each foot as the mean of its 4 keypoints, left then right."""
    return np.stack([joints[:, list(f)].mean(axis=1) for f in _FEET], axis=1)


def _foot_speeds(feet: np.ndarray, fps: float) -> np.ndarray:
    """``(N, 2)`` speed in m/s of each foot, as a central difference over
    +-1/15 s: +-2 samples at 30 fps, +-4 at 60, +-1 at 22 fps or less
    (one-sided at the ends of the clip).

    Not over one frame: the per-frame jitter of ``cam_trans`` reads as a speed
    that grows with the frame rate -- 1 cm of white jitter is 0.6 m/s at 30 fps
    and 0.85 m/s at 60 fps, above ``speed_thresh`` -- so no foot would ever
    count as planted on exactly the input the anchoring exists for. Over four
    frames the same jitter reads as a quarter of that, while a real, steady
    motion reads the same either way. And fixed in time, not in samples: at
    10 fps (a security camera, or ``--frame-skip 3``) +-2 samples is 0.4 s,
    as long as a whole stance, so near both ends of every stance the window
    reaches into the swing, the plant left over is shorter than
    ``min_frames`` and is erased, and the anchoring silently does nothing.
    """
    n = len(feet)
    speed = np.zeros(feet.shape[:2])
    if n > 1:
        k = min(max(1, round(fps / 15.0)), n - 1)
        i = np.arange(n)
        hi, lo = np.minimum(i + k, n - 1), np.maximum(i - k, 0)
        speed = np.linalg.norm(feet[hi] - feet[lo], axis=-1) * fps / (hi - lo)[:, None]
    return speed


def planted_feet(
    joints: np.ndarray, fps: float, contact: np.ndarray | None = None,
    speed_thresh: float = 0.35, height_thresh: float = 0.10,
    min_frames: int = 3, release_factor: float = 1.5,
) -> np.ndarray:
    """``(N, 2)`` bool: is the left / right foot standing on the floor?

    Without ``contact``: a foot becomes planted when its speed is below
    ``speed_thresh`` (m/s) and its height above the floor (y, floor at 0) is
    below ``height_thresh``; it stays planted until its speed exceeds
    ``speed_thresh * release_factor`` or it rises above ``height_thresh``
    (hysteresis, so a foot does not flicker at the threshold). With ``contact``
    (the model's foot-contact probability, prev-token graphs), planted is
    ``contact > 0.5``. Either way a plant shorter than ``min_frames`` consecutive
    frames is ignored.
    """
    feet = _foot_positions(np.asarray(joints, dtype=np.float64))
    n = len(feet)
    planted = np.zeros((n, 2), dtype=bool)
    if contact is not None:
        planted[:] = np.asarray(contact)[:, :2] > 0.5
    else:
        speed = _foot_speeds(feet, fps)
        low = feet[:, :, 1] < height_thresh
        for foot in range(2):
            down = False
            for t in range(n):
                limit = speed_thresh * release_factor if down else speed_thresh
                down = bool(low[t, foot] and speed[t, foot] < limit)
                planted[t, foot] = down
    # Drop plants shorter than min_frames: find each run of True and clear it.
    for foot in range(2):
        start = None
        for t in range(n + 1):
            on = t < n and planted[t, foot]
            if on and start is None:
                start = t
            elif not on and start is not None:
                if t - start < min_frames:
                    planted[start:t, foot] = False
                start = None
    return planted


def anchor_root(
    joints: np.ndarray, fps: float, contact: np.ndarray | None = None,
    speed_thresh: float = 0.35, height_thresh: float = 0.10,
    min_frames: int = 3, release_factor: float = 1.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Foot-anchored root motion: shift every frame so planted feet stand still.

    ``joints`` is ``(N, 71, 3)`` in the world frame with the floor at y = 0.
    Returns the shifted joints and the ``(N, 3)`` offset that was added to each
    frame. See :func:`planted_feet` for what "planted" means; the thresholds
    are passed through to it.

    The offset is a running value, zero at the start. Whenever the set of
    planted feet changes (a foot lands, lifts, or two planted feet become one)
    every planted foot is locked at its current OUTPUT position, i.e. where it
    is after the offset so far -- that keeps the offset continuous, so a foot
    landing never pops the body. While feet are planted the offset is whatever
    keeps them at their locked positions (the mean over planted feet, when
    both are down); while none is planted the offset is held and the body
    moves as predicted.

    Only x and z are shifted; the offset's y is always 0. In x and z the
    anchoring is odometry -- the planted foot is the only thing we know is not
    moving, so its drift must be the body's error. In y we know more: the
    floor is at 0. A held y offset would add, at every plant, the difference
    between the model's height error at its start and at its end to a running
    sum that never decays, and after a minute of walking the whole body floats
    centimetres above the floor that step 4 just levelled.
    """
    joints = np.asarray(joints, dtype=np.float64)
    planted = planted_feet(joints, fps, contact, speed_thresh, height_thresh,
                           min_frames, release_factor)
    feet = _foot_positions(joints)
    offsets = np.zeros((len(joints), 3))
    offset = np.zeros(3)
    locked = np.zeros((2, 3))
    prev = np.zeros(2, dtype=bool)
    for t in range(len(joints)):
        down = planted[t]
        if not np.array_equal(down, prev):
            locked[down] = feet[t, down] + offset
            prev = down.copy()
        if down.any():
            offset = (locked[down] - feet[t, down]).mean(axis=0)
            offset[1] = 0.0
        offsets[t] = offset
    return joints + offsets[:, None, :], offsets


# ----------------------------------------------------------------------
# File writers
# ----------------------------------------------------------------------

def _rounded(x: np.ndarray) -> list:
    return np.round(np.asarray(x, dtype=np.float64), 5).tolist()


def write_json(path: str | Path, tracks: Sequence[TrackData], meta: dict) -> None:
    """``instanthmr-mocap`` JSON: everything in :class:`TrackData`, 5 decimals."""
    doc = {
        "format": "instanthmr-mocap",
        "version": 1,
        "fps": meta["fps"],
        "frame_step": meta["frame_step"],
        "num_frames": meta["num_frames"],
        "units": "metres",
        "coordinate_system": {
            "joints_world": "X right, Y up, Z horizontal towards the camera; rotated "
                            "about X by -camera_pitch_deg to level the floor; floor at y=0",
            "joints_cam": "camera frame, X right, Y down, Z forward (OpenCV)",
            "joints_2d": "pixels, origin top-left",
        },
        "joint_names": list(NODE_NAMES),
        "parents": list(PARENTS),
        "floor_y": meta["floor_y"],
        "camera_pitch_deg": meta["camera_pitch_deg"],
        "footlock": meta["footlock"],
        "tracks": [],
    }
    for t in tracks:
        frames = []
        for i in range(len(t.frames)):
            frames.append({
                "frame": int(t.frames[i]),
                "time": round(float(t.times[i]), 5),
                "valid": bool(t.valid[i]),
                "confidence": round(float(t.confidence[i]), 5),
                "bbox": _rounded(t.bbox[i]),
                "joints_world": _rounded(t.joints_world[i]),
                "joints_cam": _rounded(t.joints_cam[i]),
                "joints_2d": _rounded(t.joints_2d[i]),
                "cam_trans": _rounded(t.cam_trans[i]),
                "root_offset": _rounded(t.root_offset[i]),
                "mhr_params": _rounded(t.mhr_params[i]),
                "shape_params": _rounded(t.shape_params[i]),
                "contact": _rounded(t.contact[i]) if t.contact is not None else None,
            })
        doc["tracks"].append({
            "id": int(t.track_id),
            "first_frame": int(t.frames[0]),
            "last_frame": int(t.frames[-1]),
            "num_frames": int(len(t.frames)),
            "num_filled": int((~t.valid).sum()),
            "bone_lengths": _rounded(t.bone_lengths),
            "frames": frames,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(doc, f)


def write_npz(path: str | Path, tracks: Sequence[TrackData], meta: dict) -> None:
    """Compressed NPZ: clip-level arrays plus ``track<id>_<field>`` per track."""
    arrays = {
        "fps": np.float64(meta["fps"]),
        "frame_step": np.int64(meta["frame_step"]),
        "floor_y": np.float64(meta["floor_y"]),
        "camera_pitch_deg": np.float64(meta["camera_pitch_deg"]),
        "joint_names": np.array(NODE_NAMES),
        "parents": np.asarray(PARENTS, dtype=np.int64),
        "track_ids": np.array([t.track_id for t in tracks], dtype=np.int64),
    }
    for t in tracks:
        p = f"track{t.track_id}_"
        for name in ("frames", "times", "valid", "confidence", "bbox", "joints_2d",
                     "joints_cam", "joints_world", "cam_trans", "root_offset",
                     "mhr_params", "shape_params", "bone_lengths"):
            arrays[p + name] = getattr(t, name)
        if t.contact is not None:
            arrays[p + "contact"] = t.contact
    np.savez_compressed(path, **arrays)


def write_csv(path: str | Path, track: TrackData) -> None:
    """One row per frame: frame, time, valid (1/0), confidence, then x/y/z of
    the 71 world-frame nodes, 5 decimals."""
    header = "frame,time,valid,confidence," + ",".join(
        f"{n}_x,{n}_y,{n}_z" for n in NODE_NAMES)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(header + "\n")
        for i in range(len(track.frames)):
            row = ",".join(f"{v:.5f}" for v in track.joints_world[i].ravel())
            f.write(f"{int(track.frames[i])},{track.times[i]:.5f},{int(track.valid[i])},"
                    f"{track.confidence[i]:.5f},{row}\n")


# ----------------------------------------------------------------------
# The recorder
# ----------------------------------------------------------------------

class MocapRecorder:
    """Collects per-frame predictions and turns them into :class:`TrackData`.

    Args:
        fps: rate of the frames that are PUSHED (a video read with
            ``--frame-skip 2`` pushes at ``video_fps / 2``). ``None`` measures
            it from the pushed timestamps in :meth:`finish` (live camera).
        frame_step: pushed frame numbers advance by this much (the demo pushes
            0, 2, 4, ... with ``--frame-skip 2``).
        footlock: anchor the root motion with the feet (step 5 above).
            Only with ``floor``, which it needs.
        floor: estimate the floor and move it to y = 0 (step 4 above).
        max_gap: longest run of missing frames that is interpolated; longer
            gaps hold the last value.
        min_frames: tracks with fewer pushed frames than this are dropped --
            a detector false positive is typically one or two frames long.

    Identity: a prediction carrying a ``track_id`` (the smoother stamps one)
    is filed under it; predictions without one are matched with the same
    greedy IoU tracker the smoother uses, kept alive for ``max_gap`` missed
    frames so that every hole it bridges is one :func:`fill_gaps` interpolates.
    A stream is expected to be all one or all the other -- tracker ids start
    at 0 and could collide with stamped ones in a mixed stream.
    """

    def __init__(self, fps: float | None, frame_step: int = 1, footlock: bool = True,
                 floor: bool = True, max_gap: int = 10, min_frames: int = 2):
        self.fps = fps
        self.frame_step = int(frame_step)
        # planted_feet's height test needs the floor at y = 0; without the
        # floor step y = 0 is the camera's height and a lifted foot "stands".
        self.footlock = footlock and floor
        self.floor = floor
        self.max_gap = int(max_gap)
        self.min_frames = int(min_frames)
        self.floor_y = 0.0
        self.camera_pitch_deg = 0.0   # positive: the camera looked down
        self._tracker = None
        self._tracks: dict[int, dict[str, list]] = {}
        self._last_frame = -1
        self._dropped = 0
        self._result: list[TrackData] | None = None

    def push(self, frame_idx: int, timestamp: float, persons: Sequence[HMRPrediction]) -> None:
        """Record the persons of one frame. Copies every array."""
        persons = [p for p in persons if np.shape(p.joints_3d_cam) == (70, 3)]
        ids = [getattr(p, "track_id", None) for p in persons]
        untracked = [i for i, t in enumerate(ids) if t is None]
        if untracked and self._tracker is None:
            from .smoothing import _Tracker
            self._tracker = _Tracker(max_age=self.max_gap)
        if self._tracker is not None:
            # Called every frame once it exists, so unmatched tracks age out.
            for i, t in zip(untracked, self._tracker.assign([persons[i] for i in untracked])):
                ids[i] = t

        sample = int(frame_idx) // self.frame_step
        self._last_frame = max(self._last_frame, int(frame_idx))
        for track_id, p in zip(ids, persons):
            rec = self._tracks.setdefault(int(track_id), {"samples": []})
            rec["samples"].append(sample)
            values = {
                "times": float(timestamp),
                "confidence": float(p.confidence),
                "bbox": p.bbox, "joints_2d": p.joints_2d, "joints_3d_cam": p.joints_3d_cam,
                "cam_trans": p.cam_trans, "mhr_params": p.mhr_params,
                "shape_params": p.shape_params, "contact": p.contact_prob,
            }
            for name, v in values.items():
                rec.setdefault(name, []).append(
                    None if v is None else np.array(v, dtype=_FIELD_DTYPES[name]))

    # ------------------------------------------------------------------
    @staticmethod
    def _complete_frames(rec: dict[str, list]) -> int:
        """Frames of a track that have every field. A Ctrl+C can land between
        the appends of one :meth:`push`, leaving its last frame torn (one list
        longer than the others); that frame is dropped rather than crashing
        the export of everything before it."""
        return min(len(rec.get(name, ())) for name in ("samples", *_FIELD_DTYPES))

    def _measure_fps(self) -> float:
        """Pushed-frame rate from the timestamps: 1 / median time per sample.
        A nominal 30 when no track has two samples."""
        steps = []
        for rec in self._tracks.values():
            n = self._complete_frames(rec)
            s = np.asarray(rec["samples"][:n], dtype=np.float64)
            t = np.asarray(rec["times"][:n], dtype=np.float64)
            if len(s) > 1:
                steps.append(np.diff(t) / np.diff(s))
        if not steps:
            return 30.0
        step = float(np.median(np.concatenate(steps)))
        return 1.0 / step if step > 0.0 else 30.0

    def finish(self) -> list[TrackData]:
        """Run the processing steps (see the module docstring) once; cached."""
        if self._result is not None:
            return self._result
        if self.fps is None:
            self.fps = self._measure_fps()

        working = []
        for track_id in sorted(self._tracks):
            rec = self._tracks[track_id]
            n = self._complete_frames(rec)
            if n < self.min_frames:
                self._dropped += 1
                continue
            pushed = np.asarray(rec["samples"][:n], dtype=np.int64)
            arrays = {name: np.stack(rec[name][:n]) for name in _FIELD_DTYPES
                      if name not in ("times", "contact")}
            if all(c is not None for c in rec["contact"][:n]):
                arrays["contact"] = np.stack(rec["contact"][:n])
            order = np.argsort(pushed, kind="stable")
            # mhr_params[3:130] are angles in radians (root Euler, then the
            # joint angles), so a hole is filled the short way round: with its
            # back to the camera a person's root angles sit at +-pi and flip
            # sign from frame to frame, and interpolated one number at a time
            # the filled frames would spin half a turn and back. Only the
            # filled rows are replaced, so the pushed ones stay bit-identical.
            angles = arrays["mhr_params"][:, 3:130].astype(np.float64)
            angles[order] = np.unwrap(angles[order], axis=0)
            samples, arrays, valid = fill_gaps(pushed, arrays, self.max_gap)
            a = fill_gaps(pushed, {"a": angles}, self.max_gap)[1]["a"]
            a -= 2 * np.pi * np.round(a / (2 * np.pi))          # back into [-pi, pi]
            arrays["mhr_params"][~valid, 3:130] = a[~valid]
            # Time is linear in the frame number, so a hole's time stamps are
            # interpolated whatever its length: fill_gaps would hold the last
            # stamp over a long hole, and time must keep advancing.
            arrays["times"] = np.interp(
                samples, pushed[order], np.asarray(rec["times"][:n], dtype=np.float64)[order])

            joints_cam = add_root(arrays.pop("joints_3d_cam").astype(np.float64))
            joints_world = to_world(joints_cam)
            lengths = np.median(bone_lengths(joints_world[valid]), axis=0)
            joints_world = normalise_bone_lengths(joints_world, lengths)
            working.append((track_id, samples, valid, arrays, joints_cam, joints_world, lengths))

        if self.floor:
            valids = [w[2] for w in working]
            pitch = estimate_pitch([w[5] for w in working], valids)
            self.camera_pitch_deg = float(np.degrees(pitch))
            if pitch:
                # Undo the pitch: rotate the world back up by the same angle.
                level = _rotation_about_x(-pitch)
                for w in working:
                    w[5][:] = w[5] @ level.T
            self.floor_y = estimate_floor([w[5] for w in working], valids)
            for w in working:
                w[5][..., 1] -= self.floor_y

        tracks = []
        for _, samples, valid, arrays, joints_cam, joints_world, lengths in working:
            contact = arrays.get("contact")
            if self.footlock:
                joints_world, offset = anchor_root(joints_world, self.fps, contact)
            else:
                offset = np.zeros((len(samples), 3))
            tracks.append(TrackData(
                # 0, 1, ... in order of appearance, not the tracker's id: that
                # also counted the short detections dropped above, and the
                # guide promises the first person is track0.
                track_id=len(tracks),
                frames=samples * self.frame_step,
                times=arrays["times"],
                valid=valid,
                confidence=arrays["confidence"],
                bbox=arrays["bbox"],
                joints_2d=arrays["joints_2d"],
                joints_cam=joints_cam.astype(np.float32),
                joints_world=joints_world.astype(np.float32),
                cam_trans=arrays["cam_trans"],
                mhr_params=arrays["mhr_params"],
                shape_params=arrays["shape_params"],
                contact=contact,
                bone_lengths=lengths,
                root_offset=offset.astype(np.float32),
            ))
        self._result = tracks
        return tracks

    # ------------------------------------------------------------------
    def report(self) -> str:
        """A few lines for a human: tracks, floor, what the anchoring did, and
        whether the finger chains are ordered as the skeleton assumes."""
        tracks = self.finish()
        if not tracks:
            return (f"mocap: no person was tracked for {self.min_frames} frame(s) or more "
                    "- nothing to animate, so no .csv or .bvh is written"
                    + (f" ({self._dropped} short detection(s) dropped)" if self._dropped else ""))
        lines = [f"mocap: {len(tracks)} track(s)"
                 + (f", {self._dropped} dropped (fewer than {self.min_frames} frames)"
                    if self._dropped else "")]
        longest = max(len(t.frames) for t in tracks)
        for t in tracks:
            line = (f"  track {t.track_id}: frames {t.frames[0]}-{t.frames[-1]}, "
                    f"{len(t.frames)} samples, {int((~t.valid).sum())} filled in, "
                    f"mean confidence {float(t.confidence.mean()):.2f}")
            # A detector false positive is typically brief and half-hearted; a
            # person is neither, so the user is told which files to doubt.
            if float(t.confidence.mean()) < 0.7 or len(t.frames) < 0.25 * longest:
                line += " - short or low confidence: may be a false detection, not a person"
            lines.append(line)
        if self.floor:
            line = f"floor: y = {self.floor_y:+.3f} m before levelling (now y = 0)"
            if self.camera_pitch_deg:
                line += (f"; camera pitched {abs(self.camera_pitch_deg):.1f} deg "
                         f"{'down' if self.camera_pitch_deg > 0 else 'up'}, floor levelled")
            else:
                line += ("; camera pitch not measured (needs a person moving 1 m or more "
                         "in depth), assumed level")
            lines.append(line)
        else:
            lines.append("floor: not levelled (floor=False)")

        if self.footlock:
            lines.append("foot anchoring: root movement per frame (real motion counts too), "
                         "speed of planted feet")
            for t in tracks:
                after = t.joints_world.astype(np.float64)
                before = after - t.root_offset[:, None, :].astype(np.float64)
                planted = planted_feet(before, self.fps, t.contact)[1:]
                rb = np.linalg.norm(np.diff(before[:, ROOT], axis=0), axis=-1).mean() if len(after) > 1 else 0.0
                ra = np.linalg.norm(np.diff(after[:, ROOT], axis=0), axis=-1).mean() if len(after) > 1 else 0.0
                less = 100.0 * (1.0 - ra / rb) if rb > 0.0 else 0.0
                line = (f"  track {t.track_id}: root {rb * 1000:.1f} -> {ra * 1000:.1f} mm/frame "
                        f"({less:.0f}% less)")
                if planted.any():
                    sb = _foot_speeds(_foot_positions(before), self.fps)[1:][planted].mean()
                    sa = _foot_speeds(_foot_positions(after), self.fps)[1:][planted].mean()
                    line += (f"; planted feet {sb:.3f} -> {sa:.3f} m/s "
                             f"(a foot is planted on {planted.mean():.0%} of foot-frames)")
                else:
                    line += "; no foot was ever planted"
                lines.append(line)
        else:
            lines.append("foot anchoring: off")

        cams = [t.joints_cam[t.valid][:, :ROOT] for t in tracks if t.valid.any()]
        problems = check_finger_order(np.concatenate(cams)) if cams else []
        lines.append("finger chains ordered as expected" if not problems
                     else "finger chains NOT ordered as expected:\n  " + "\n  ".join(problems))
        return "\n".join(lines)

    # ------------------------------------------------------------------
    def export(self, prefix: str | Path,
               formats: Sequence[str] = ("json", "npz", "csv", "bvh")) -> list[Path]:
        """Write ``<prefix>.mocap.json``, ``<prefix>.mocap.npz``,
        ``<prefix>.track<id>.csv`` and ``<prefix>.track<id>.bvh``. Returns the
        paths written."""
        unknown = [f for f in formats if f not in ("json", "npz", "csv", "bvh")]
        if unknown:
            raise ValueError(f"unknown export format(s): {unknown}")
        tracks = self.finish()
        prefix = Path(prefix)
        meta = {"fps": float(self.fps), "frame_step": self.frame_step,
                "num_frames": self._last_frame + 1, "floor_y": float(self.floor_y),
                "camera_pitch_deg": float(self.camera_pitch_deg),
                "footlock": bool(self.footlock)}
        written: list[Path] = []
        if "json" in formats:
            written.append(prefix.with_name(prefix.name + ".mocap.json"))
            write_json(written[-1], tracks, meta)
        if "npz" in formats:
            written.append(prefix.with_name(prefix.name + ".mocap.npz"))
            write_npz(written[-1], tracks, meta)
        if "csv" in formats:
            for t in tracks:
                written.append(prefix.with_name(f"{prefix.name}.track{t.track_id}.csv"))
                write_csv(written[-1], t)
        if "bvh" in formats:
            from .bvh import write_bvh  # lazy: mocap.py must import without bvh.py
            for t in tracks:
                written.append(prefix.with_name(f"{prefix.name}.track{t.track_id}.bvh"))
                write_bvh(written[-1], NODE_NAMES, PARENTS,
                          REST_DIRECTIONS * t.bone_lengths[:, None],
                          t.joints_world, self.fps)
        return written
