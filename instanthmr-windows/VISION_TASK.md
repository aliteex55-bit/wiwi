# Task for a Claude session on the PC: InstantHMR motion capture in Vision

Written for a Claude Code session running on the user's Windows PC. The user
asked for this from their phone: "build this new GitHub mocap on what we did,
if it is better" and "build a new .exe".

**Vision** is Vision.exe (formerly Jarvis Studio): Electron + Vite + three.js,
source in `C:\Users\xboxa\Documents\Codex\Jarvis\desktop`. It opens fullscreen
on the live camera, draws a 3D lattice over it, and takes mouse, MediaPipe hand
tracking (local, CPU) and PS5 controller input. The Python voice line
`backtalk` talks to it over a loopback hub on 127.0.0.1:8795.

**InstantHMR** (this folder) is full-body 3D pose: RF-DETR finds each person,
a small ONNX model returns 70 keypoints per person in metres (body, feet, both
hands with fingers), about 30 fps on an RTX 4070 with the detector on every
frame. This bundle adds Windows setup scripts, smoothing, persistent person ids
and motion-capture export (BVH / JSON / CSV); see GUIDE_WINDOWS.md and
BUNDLE_NOTES.md.

The goal: make Vision's motion capture better with InstantHMR, keep what is
already better, and ship a new Vision.exe.

## 0. Protect existing work first

1. In the Jarvis folder, run `git status` (if it is a git repo). On 23 Oct
   another session left uncommitted changes in `backtalk/backtalk/{ears,config,main,brain}.py`
   and in `desktop/renderer/vision/`. Do not overwrite them. Commit them as
   they are on a branch named `wip-before-instanthmr` (or, without git, copy
   the folders to `Jarvis-backup-<date>`), then do the work on a new branch.
2. Keep the current Vision.exe: copy the built app folder to
   `Vision-previous` before rebuilding.
3. Leave `mic_device` / `mic_hostapi` in `backtalk/backtalk.json` as they are.
4. The user lifts the 23 Oct production hold for this task only.

## 1. Install InstantHMR

```
git clone -b claude/tender-dijkstra-q388pa https://github.com/aliteex55-bit/wiwi C:\InstantHMR-src
xcopy /e /i C:\InstantHMR-src\instanthmr-windows C:\InstantHMR
```

Then double-click `C:\InstantHMR\setup_windows.bat` (or run it in a
terminal). It installs a private Python, the GPU packages, downloads the
weights from HuggingFace and writes `config.bat` with `DEVICE` and `DETECTOR`.
Report those two values. Run `git pull` in `C:\InstantHMR-src` and copy again
before step 4: final fixes may still land on that branch.

## 2. Measure before deciding "better"

Use the camera Vision uses. Record a 20-30 s clip at the desk: hands in front
of the camera doing Vision's gestures, then standing back so the whole body is
in view.

- InstantHMR: `run_video.bat` on the clip, then `export_mocap.bat` on it; read
  the fps in the console and the report (tracks, floor, jitter).
- Vision's current MediaPipe hand tracking on the same clip (or live, side by
  side): fps, how often each hand is found, how steady the fingertips are.

Write a short table: fps on this PC, hands found (% of frames), fingertip
jitter, what only InstantHMR gives (whole body in 3D, several people, metric
depth, BVH recording).

**Decision rule.** Keep MediaPipe for anything it measures better (expected:
quick finger gestures at close range, because InstantHMR sees the whole person
in one 224 px crop). Add InstantHMR for the body, and for the hands only if it
measures as good or better. If InstantHMR runs under 15 fps on this PC with
`DETECTOR=nano` and `--detector-stride 3`, stop and tell the user before
building.

## 3. Design

- Run InstantHMR as a sidecar Python process, like backtalk, started and
  stopped by Vision. Use `C:\InstantHMR\.venv\Scripts\python.exe` and the
  `instanthmr` package (`PosePipeline`, `TemporalSmoother` in 1euro mode,
  `MocapRecorder`).
- Stream one JSON message per frame over a local WebSocket on
  `127.0.0.1:8796` (not 8795, that is backtalk's hub): timestamp, and per
  person `track_id`, confidence, the 70 joints in 3D (metres, use the
  bundle's world frame: Y up, floor at 0) and in 2D (video pixels).
  Joint names and bone list: `instanthmr/skeleton.py` and
  `instanthmr/mocap_skeleton.py`.
- Camera: Windows often lets only one program open a webcam. Test whether
  Vision and the sidecar can both open it. If not, Vision keeps the camera and
  sends frames to the sidecar over the same WebSocket (JPEG, newest frame
  wins, drop the rest).
- Renderer: draw the skeleton over the camera picture with three.js, a menu
  toggle to show or hide it, and keep mouse, MediaPipe hands and the PS5 pad
  working as before.
- Optional menu item "Record motion": start/stop a `MocapRecorder` in the
  sidecar and write BVH/JSON next to Vision's other outputs.

## 4. Build and check

1. Build the new Vision.exe with the project's own script (see
   `desktop/package.json`).
2. Start it, stand in front of the camera, and confirm the skeleton follows
   you. Take a screenshot.
3. Commit the Vision changes on the new branch.

## 5. Report to the user

Plain words: the measurement table and the choice it led to, what changed in
Vision, how to start it, how to switch back to `Vision-previous`, and what is
not done.
