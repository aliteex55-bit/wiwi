# Bundle notes (for developers)

This folder is upstream InstantHMR
(https://github.com/Naturalpad-studio/InstantHMR, commit in
`UPSTREAM_COMMIT.txt`) with `.git` removed and a thin Windows layer on top.
Apart from `instanthmr/detector.py` (see the rfdetr bullet below) and the
motion-capture layer (its own section below: three new modules, four new
`demo.py` flags, a `track_id` field), nothing under `instanthmr/`, `demo.py`,
`tools/`, training or benchmark code was touched.

## Changes to upstream files

`install.py` (surgical, same style):

- `main()`: `elif sysname in ("Linux", "Windows")` - Windows takes the Linux
  branch. Upstream exited with "unsupported platform" on Windows.
- torch is pinned to `torch==2.8.0 torchvision==0.23.0` on every non-mac
  branch. 2.8.0 satisfies rfdetr (>=2.2) and transformers 5 (>=2.6), and it is
  the ABI `pymomentum` (optional MHR mesh) is built against, so the mesh step
  never has to replace torch (pip backtracking against the pytorch index,
  which has no PEP 658 metadata, downloads whole wheels per candidate).
- CUDA branches, keyed on `nvidia-smi --query-gpu=compute_cap`:
  - `>= 7.5` (Turing .. Blackwell): torch from `/whl/cu128` +
    `onnxruntime-gpu==1.26.0`. ORT 1.21-1.26 Windows wheels are built with
    CUDA 12.8 (`onnxruntime/capi/build_and_package_info.py` in the 1.26.0
    wheel: `cuda_version = '12.8'`) and ORT's `preload_dlls()` loads
    cudart/cublas/cuDNN from `torch\lib` when torch's `+cuNNN` major matches,
    so the torch wheel MUST ship CUDA 12.8 DLLs: upstream's cu124 branch
    (torch 2.6.0+cu124, CUDA 12.4 / cuDNN 9.1 DLLs) made the CUDA EP fail to
    load and setup silently chose `device=cpu` on RTX 30/40 cards.
  - `< 7.5` (Pascal / Volta): torch from `/whl/cu126` (cu128 has no sm_6x /
    sm_70 kernels) + the CPU `onnxruntime` (the cu126 wheel has no CUDA 12.8
    DLLs for ORT). RF-DETR runs on the GPU, the small ONNX regressor on the
    CPU; setup then writes `DEVICE=cpu DETECTOR=medium`.
- `onnxruntime-gpu>=1.19` became `onnxruntime-gpu==1.26.0`: 1.27+ (June 2026)
  is built against CUDA 13 and silently falls back to CPU next to a cu12 torch.
  This also fixes Linux and matches the repo's own CLAUDE.md note.
- The `nvidia-cudnn-cu12` / `nvidia-cuda-runtime-cu12` / `nvidia-cublas-cu12`
  (+ `nvidia-cufft-cu12`, which the 1.26.0 CUDA EP also imports) wheels are
  installed on Linux only. On Windows ORT never consults `site-packages/nvidia`
  when torch's CUDA major matches its own, so they were ~1 GB of unused
  download.
- Re-running on an existing .venv switches builds: `install_torch(tag)` reads
  the installed torch's local label (`importlib.metadata.version('torch')`,
  `2.8.0+cpu` -> `cpu`, no label -> `cpu` since PyPI's Windows wheel is
  CPU-only) and runs `pip uninstall -y torch torchvision` first when it differs
  from the wanted index tag, because pip treats an installed `2.8.0+cpu` as
  satisfying `torch==2.8.0` (PEP 440 ignores the local label) and would keep
  the CPU build forever after a driver install. `install_onnxruntime(gpu)`
  uninstalls the other flavour (`onnxruntime` vs `onnxruntime-gpu`) before
  installing, since both unpack into the same `onnxruntime/` directory and the
  CUDA EP silently disappears when they are layered. Uninstall-then-install
  was chosen over `--force-reinstall --no-deps` because on Linux the cu128
  torch wheel pulls the `nvidia-*` runtime wheels as dependencies.
- `rfdetr>=1.0` became `rfdetr>=1.8` (also in `requirements.txt`): 1.8.0 is
  the oldest version verified here to honour `RF_HOME`. There is no upper
  bound. rfdetr 1.11.0 (2026-09-23) removed `optimize_for_inference()`
  (deprecated alias of `inference()` since 1.9.0), which upstream's
  `instanthmr/detector.py` calls by default -> `AttributeError` at pipeline
  construction; the bundle's `detector.py` looks for `inference()` first and
  falls back to the old name (verified here with rfdetr 1.11.2 on CPU: the
  detector constructs and finds the same boxes). The `AttributeError` can
  therefore only come back if `detector.py` is replaced by the upstream file;
  the fix is to restore the bundle's copy, not to pin rfdetr.
- The two "Linux ..." messages print `platform.system()`; the GPU-branch
  messages use `->` instead of the U+2192 arrow (not encodable in cp1252 if
  output is ever redirected to a file).
- `--dry-run` and the Darwin branch are unchanged; `--force-cpu` behaves as
  before, only its help text now says `Linux / Windows`. Verified here
  on Linux (dry-run, both flags) and with `platform.system` monkeypatched to
  `Windows` against a fake `nvidia-smi` reporting 8.9 / 12.0 / 7.5 / 6.1 and
  against none.

`requirements-mhr.txt`: comment block only. It no longer tells people to
install both requirement files in one pip call: pymomentum-gpu's Windows wheel
(0.1.112+) declares no torch requirement, so that recipe did not pin torch at
all; only pymomentum-cpu still pins `torch>=2.8,<2.9`.

## Added files

All `.bat` files are ASCII with CRLF line endings, `setlocal`, a UNC guard
(`set "ROOT=%~dp0"` / `if "%ROOT:~0,2%"=="\\" goto :on_network`: `cd /d` fails
on `\\server\share` with the working directory left at `C:\Windows`, so setup
would otherwise try to create the venv there), then `cd /d "%~dp0"`, no
delayed expansion, linear flow with `goto` labels, every path quoted (also
inside `echo` text, so a `&` in a file name cannot split the line), `pause`
only at the end. Exit codes are tested with
`if not "%errorlevel%"=="0"` (and `if "%errorlevel%"=="0"`), never
`if errorlevel 1`: a process that dies hard on Windows (access violation
0xC0000005, missing DLL 0xC0000135, Ctrl+C 0xC000013A) exits with a NEGATIVE
code, which `errorlevel 1` / `not errorlevel 1` would count as success. Every
script refuses to run when `demo.py` is not next to it (double-clicked inside
Explorer's zip view, or copied alone). They could not be executed on the build
machine (no cmd.exe) - they were reviewed by hand only; the Python one-liners
inside them were run here against the CPU onnxruntime wheel.

- `setup_windows.bat [--cpu | /cpu]` - finds `py -3.12-64`, `py -3.11-64`, then
  `python` (version-checked, 3.13 rejected because rfdetr/pymomentum wheels
  lag; only 64-bit accepted - `-64` for the launcher, `sys.maxsize > 2**32` for
  `python` and for an existing `.venv` - because torch and onnxruntime have no
  32-bit wheels);
  `python -m venv .venv`; `.venv\Scripts\python.exe -m pip install --upgrade
  pip`; `python install.py [--force-cpu]`; `call download_weights.bat nopause`;
  (before `install.py`: refuses with a link to `vc_redist.x64.exe` unless
  `%SystemRoot%\System32\msvcp140.dll` and `vcruntime140_1.dll` exist - torch's
  and ORT's Windows wheels link against them and the python.org installer ships
  only `vcruntime140.dll`); then two independent GPU tests whose stderr goes to
  `gpu_check.log`:
  `DEVICE=cuda` if an ORT session on `models/instanthmr.onnx` with
  `CUDAExecutionProvider` first reports that provider as active (ORT falls back
  to CPU with a warning rather than raising, so `session.get_providers()[0]`
  is the only truthful check; torch is deliberately NOT imported first, which
  mirrors `pipeline.py` constructing `InstantHMR` before `RFDETRDetector`), and
  `DETECTOR=medium` if `torch.cuda.is_available()`; writes `config.bat`;
  pre-downloads the RF-DETR checkpoint (`RF_HOME=models\rfdetr`, honoured by
  rfdetr >= 1.8 `get_model_cache_dir`, weights come from
  storage.googleapis.com) through the same call path the demo uses
  (`RFDETRDetector(variant=...)` with the default optimisation pass, plus
  `warmup()`), so an rfdetr API break fails at setup, not on first use;
  verifies `from instanthmr import PosePipeline`, constructs
  `InstantHMR('models/instanthmr.onnx', device=<DEVICE>)` + `warmup(max_batch_size=2)`
  in its own process (torch not imported first, matching `pipeline.py`) and
  prints `active_provider`, so a damaged weights file or a CUDA failure at
  execution time fails setup rather than the first drag-and-drop; then prints
  ORT/torch versions.
  The "Python not found" text links the 3.12.10 installer directly
  (python.org's 3.12.11+ entries are source-only security releases, PEP 693).
- `download_weights.bat [nopause]` - `curl.exe -L -f --retry 3` to
  `models\instanthmr.onnx.part`, PowerShell `Invoke-WebRequest` fallback (it
  reads the URL and path as `$env:URL` / `$env:TMP_OUT`, inherited from cmd's
  `set`, instead of splicing them into single-quoted PowerShell strings, which
  broke for a user folder containing an apostrophe; same in
  `download_mhr_assets.bat` with `$env:ZIP_URL` / `$env:ZIP_PART`),
  `move` into place only if >= 50,000,000 bytes (the file is ~77 MB; smaller
  means an HTML error page). Skips when a >= 50 MB file exists.
- `download_mhr_assets.bat` - optional MHR mesh. Requires a 3.12 venv, `git`
  (for `mhr @ git+https://github.com/facebookresearch/MHR.git`) and `tar.exe`.
  Order: checks -> `pip freeze` of the installed `torch==`/`torchvision==`
  lines into a pip constraints file in `%TEMP%` (deleted again on every exit
  path, success or failure), refusing unless torch is 2.8.x ->
  download `assets.zip` (v1.0.1, 198,943,157 bytes, verified here: a single
  top-level `assets/` folder with 19 files, 4,767,087,611 bytes unpacked) ->
  `tar.exe -xf ... --strip-components=1` so `models\mhr_assets\` holds
  `lod*.fbx`, `compact_v6_1.model`, `mhr_model.pt`, the corrective blendshape
  `.npz` files and `LICENSE.txt` directly -> `pip install -r
  requirements-mhr.txt -c pins.txt --extra-index-url
  https://download.pytorch.org/whl/<tag from torch.version.cuda>` (CPU venvs:
  `pymomentum-cpu` + `mhr` with the same constraints). `requirements.txt` is
  deliberately NOT passed: its `onnxruntime>=1.19` line would install the CPU
  wheel on top of `onnxruntime-gpu` (same `onnxruntime/` directory, CUDA EP
  gone, silent CPU fallback). Verification: imports, `PosePipeline`, ORT
  providers must still contain `CUDAExecutionProvider` when `config.bat` says
  `DEVICE=cuda`, and `MHRRenderer('models/mhr_assets', device=<DEVICE>)` is
  constructed once (a mismatched pymomentum build segfaults here instead of on
  the user's first run). Only then is `models\mhr_assets\mhr_ready.txt`
  written; it is deleted at the start of every run of this script.
- `run_image.bat`, `run_video.bat`, `record_video.bat`, `run_camera.bat` -
  take `%1` (drag-and-drop works; `%~f1` is captured before `cd`), default
  `DEVICE=cuda` / `DETECTOR=medium`, `call config.bat` overrides, set
  `RF_HOME`, add `--mhr-assets models\mhr_assets` only when
  `models\mhr_assets\mhr_ready.txt` exists (not when `lod3.fbx` exists: an
  unpacked zip with a failed package install would otherwise break every run).
  A missing `config.bat` is reported (run setup again) and the defaults are
  used. `run_camera.bat` rejects an existing path or a non-numeric `%1` (a
  video dragged onto it by mistake) with a message pointing at `run_video.bat`.
  All four set `PYTHONWARNINGS=ignore` before `demo.py` (so does
  `setup_windows.bat` before its detector test): rfdetr's TorchScript pass
  prints dozens of `TracerWarning` blocks at every start, which a
  non-developer reads as errors; the exit code still decides success.
  Flags, all checked against `demo.py`'s argparse: image -> none; video ->
  `--smoothing-filter --smoothing-mode 1euro --detector-stride 2`; camera ->
  `--smoothing-filter --detector-stride 2` (demo.py forces 1euro for
  `--camera`); record -> video flags plus `--no-spawn --save-rrd "%~dpn1.rrd"`,
  then `.venv\Scripts\rerun.exe` on the file. `rr.save()` REPLACES the gRPC
  sink that `rr.init(spawn=True)` installed (rerun-sdk 0.41, verified here:
  `rr.save(a); log(world); rr.save(b); log(x)` leaves only `world` in a.rrd and
  only `x` in b.rrd), so with a spawned viewer the viewer receives nothing and
  the file misses the static `world` ViewCoordinates that `visualizer.py` logs
  before `save()`; with `--no-spawn` the SDK buffers until `save()` and the file
  gets everything (verified: `init(spawn=False); log(world); save(c); log(x)`
  gives both). `run_camera.bat`, `run_video.bat`, `record_video.bat` and
  `export_mocap.bat` treat exit code -1073741510 (0xC000013A, Ctrl+C during model loading, before
  demo.py's `except KeyboardInterrupt`) as "Stopped", not as an error.
- `open_recording.bat` - drag a `.rrd` onto it: runs `.venv\Scripts\rerun.exe
  file.rrd` (the `rerun` console script ships with `rerun-sdk`; viewer and SDK
  versions must match).
- `setup_linux_mac.sh [--cpu]` - same steps in bash (`python3.12`,
  `python3.11`, `python3`, `python` with a version check; `.venv/bin/python`;
  `coreml` on macOS when the CoreML EP is present, `cuda` / `medium` on Linux
  via the same two tests, stderr to `gpu_check.log`; same detector warm-up;
  a failed `python -m venv` prints the `apt install python3.12-venv` hint).
  Prints the `demo.py` commands with the detected device instead of writing a
  config file.
- `GUIDE_WINDOWS.md` - end-user guide (unpacking the zip, security dialogs,
  Smart App Control, the firewall prompt for `rerun.exe`, how to stop the
  camera, disk-space and pip-cache notes, `gpu_check.log`).

## Motion-capture layer

New in this bundle, not upstream: `instanthmr/mocap_skeleton.py`,
`instanthmr/mocap.py`, `instanthmr/bvh.py`, `tests/` (pytest, no model files
needed), the `--export` flags in `demo.py` and `export_mocap.bat`. Plain
numpy; no new package. `demo.py` only ever showed each frame's prediction;
this layer turns the per-frame, per-person predictions of a run into one
cleaned skeleton animation per person and writes it to files.

### Modules

- `mocap_skeleton.py` - the contract the other two build on: a TREE over the
  70 MHR70 keypoints plus a virtual root `hips` (node 70, the midpoint of the
  two hips; the model predicts no pelvis and a rig needs one bone to carry the
  translation). `NODE_NAMES` (71), `PARENTS` (-1 for the root), `CHILDREN`,
  `TOPOLOGICAL_ORDER` (parents first), `REST_DIRECTIONS` (71,3) unit T-pose
  bone directions, `CAM_TO_WORLD` = diag(1,-1,-1) applied as `xyz_cam @
  CAM_TO_WORLD` (camera X right / Y down / Z forward -> world X right / Y up /
  Z towards the camera; determinant +1, so left stays left and a person facing
  the camera faces +Z, the usual BVH convention), `FOOT_NODES`,
  `FINGER_CHAINS`, `add_root()`, `bone_lengths()`, `check_finger_order()`.
  Metres everywhere. The hand chains assume the MHR70 names are tip-first
  (`*_third_joint` is next to the wrist) - see "Not verified".
- `mocap.py` - `MocapRecorder(fps, frame_step, footlock, floor, max_gap=10,
  min_frames=2)`. `push(frame_idx, timestamp, persons)` once per processed
  frame; `finish()` then runs, per person, in this order: **gap fill** (holes
  of <= `max_gap` samples linearly interpolated, longer ones hold the last
  value, `TrackData.valid` False on filled samples; the time stamps are
  interpolated over any hole, since time is linear in the frame number;
  tracks with fewer than `min_frames` complete frames are dropped - a
  detector false positive is one or two frames long - and a push torn by a
  Ctrl+C between its appends loses only that frame) -> **add root** ->
  **world frame** (`CAM_TO_WORLD`) -> **bone-length normalisation** (median
  length of every bone over the valid frames; every frame rebuilt
  parents-first with its predicted direction and that length, because a rig
  has ONE length per bone) -> **floor** (`estimate_pitch`: the camera's pitch
  from a straight-line fit of the lowest foot keypoint's height against depth
  z, refitted once without points more than 5 cm above the line, undone by a
  rotation about x of every track; skipped, and the camera assumed level,
  when the points span less than 1 m of depth, where the slope is noise - a
  standing person gets no correction; then one clip-wide 5th percentile of
  the lowest foot keypoint over all tracks, subtracted so the floor is y = 0;
  a percentile so one bad frame cannot sink it) -> **foot anchoring** (a foot
  is "planted" when its speed is below 0.35 m/s and its height below 0.10 m,
  with hysteresis x1.5 and a 3-frame minimum, or `contact_prob > 0.5` when
  the graph has it; the speed is a +-2 frame central difference, because
  over one frame 1 cm of white jitter reads as 0.6 m/s at 30 fps and 0.85 at
  60 fps and no foot would ever plant; a running root offset in x and z keeps
  every planted foot at its locked output position, is held while no foot is
  planted, and is re-locked whenever the planted set changes so it never
  jumps; y is never anchored, because a held y offset adds each plant's
  start-minus-end height error to a sum that never decays and floats the
  body off the levelled floor; stored per frame in `TrackData.root_offset`,
  so `joints_world - root_offset` is the un-anchored skeleton). This last step
  is the one that matters: `cam_trans` is a free 3-vector with little tying
  it to pixels (`smoothing.py`), so a standing person slides back and forth
  by centimetres, and a planted foot is the one thing known not to move.
  `report()` prints the numbers (tracks, floor, root movement and
  planted-foot speed before -> after, `check_finger_order`); `export(prefix,
  formats)` writes `<prefix>.mocap.json`, `<prefix>.mocap.npz`,
  `<prefix>.track<id>.csv`, `<prefix>.track<id>.bvh`.
- `bvh.py` - `write_bvh(path, names, parents, rest_offsets, positions, fps,
  scale=100)`: positions in, rotations out, because a BVH file stores bone
  lengths once (`OFFSET`) and directions per frame as rotations. Per joint,
  parents first: a **fork** (>= 2 children with non-collinear rest directions:
  pelvis, neck, shoulders, elbows, wrists, ankles, head) gets the Kabsch
  rotation of its children's rest offsets onto their current directions,
  weighted by rest bone length so a 3 cm elbow landmark cannot steer the 26
  cm forearm (with 1 cm noise, equal weights put fingertips up to 20 cm off,
  length weights 8 cm); a **chain** joint (one child) gets the smallest swing
  taking the parent-rotated rest direction onto the current one, twist
  inherited from the parent; a **leaf** inherits its parent's rotation. The
  local rotation `G_parent^T @ G` is written as ZXY Euler degrees (`CHANNELS 3
  Zrotation Xrotation Yrotation`, the root also `Xposition Yposition
  Zposition`), offsets and positions x100 = centimetres, every leaf gets an
  `End Site`. `read_bvh()` and `fk()` are the checking parser and forward
  kinematics.

### demo.py and the .bat

Flags: `--export PREFIX` (for a video `--export C:/clips/dance` puts the files
next to the clip; a folder, an empty name or a missing parent folder is
rejected in `parse_args`, before any model loads), `--export-formats
json,npz,csv,bvh` (validated there too), `--no-footlock`, `--no-floor`.
`demo.MAX_GAP` (10) is passed both as the smoother's tracker `max_age` and as
the recorder's `max_gap`, so every hole the tracker bridges is one the
recorder interpolates (the defaults were 5 and 10: a 6-10 frame blink split
one person into two files). `pipeline.py` no longer re-feeds the last box
when the detector ran and saw nobody - only on stride-skipped frames - so a
person who leaves the picture ends their track instead of producing a
confident-looking skeleton of an empty crop until the end of the clip.
`build_recorder()` makes the `MocapRecorder`, `export_mocap()` runs
finish / report / export and prints the paths under `motion capture
written:`. Video: the recorder gets `fps / frame_skip` and `frame_step =
frame_skip`, `emit()` pushes `(frame_idx, frame_idx / fps)` after
`viz.log_frame`, and the export runs after the smoother flush (the centred
filter holds 2 frames back) and also after Ctrl+C. Camera: `fps=None` (the
recorder measures 1 / median dt from the pushed timestamps, which are
`time.perf_counter() - t_start`, so the exported times start at 0 like a
video's), export after the `try` block, so Ctrl+C - the only way a camera run
ends - still writes the files. Any other exception inside either loop is
printed with its traceback, the export still runs, and the function then
exits with code 1 (so the .bat reaches `:failed`): on a CPU a run takes 5-15x
the clip's length and losing the capture is the worst outcome. Image: one
frame pushed as `(0, 0.0)` with `min_frames=1` (a one-frame track: floor
from that frame, nothing to anchor).

Track ids: `HMRPrediction.track_id` is a new field (default `None`).
`TemporalSmoother._finalize` stamps the smoother's greedy-IoU id on every
prediction it emits, so with `--smoothing-filter` the recorder files
predictions under the smoother's ids. Without the smoother they carry `None`
and the recorder builds its own `smoothing._Tracker` on first use and calls
it on every push so stale tracks age out. A mixed stream (some stamped, some
not) is not expected; both id spaces start at 0 and could collide.

`export_mocap.bat`: `record_video.bat`'s structure and guards, running
`demo.py --video "%INPUT%" ... --smoothing-filter --smoothing-mode centered
--detector-stride %STRIDE% --no-spawn --save-rrd "%~dpn1.rrd" --export
"%~dpn1" %MHR%` with `STRIDE=1` on `DEVICE=cuda` or `DETECTOR=medium` (the card runs the detector)
and 2 otherwise. Centred
rather than 1euro because the clip is offline: the 2-frame lookahead costs
nothing and has no lag, while 1euro's heavy rest-time smoothing would
round off slow motion in the file. Afterwards it lists `dir /b
"%~dp1%~n1.*"`, prints the Blender import steps and opens the `.rrd` in
`.venv\Scripts\rerun.exe`.

### Verified here (Linux, 4 CPUs, no GPU, no real weights)

- `python -m pytest tests -q`: 26 passed (18 in `test_mocap.py`, 8 in
  `test_bvh.py`), synthetic skeletons with known truth.
- End to end: `demo.py --video` on a 30-frame 1280x497 clip of a real person
  with RF-DETR nano and a stand-in ONNX graph (real input/output names and
  shapes, constant outputs; `joints_3d` is `uniform(-0.6, 0.6)` noise) with
  `--smoothing-filter --smoothing-mode centered --export`: 9 fps, `mocap: 2
  track(s)` (the second: 15 real + 15 filled frames of a low-confidence
  detection), all six files written - `clip.mocap.json` 451,687 B,
  `clip.mocap.npz` 41,004 B, `clip.track{0,1}.csv` 59,109 B each,
  `clip.track{0,1}.bvh` 76,911 B each - and exit code 0. `root 0.0 -> 0.0
  mm/frame` and `finger chains NOT ordered as expected` are what constant
  random outputs must give.
- The written BVH read back with `read_bvh()`: 71 joints, 30 frames, frame
  time 0.0667 s (the clip is 15 fps), `fk()` reproduces the root exactly. On
  the stand-in the other joints are up to 169 cm off, and that is the
  stand-in, not the writer: its sibling bones at a fork are not a rigid
  rotation of the rest pose (angle neck->nose / neck->left_shoulder 25 deg
  vs 99 deg in the T-pose), so no rotation fits. The same recorder -> export
  -> `read_bvh` -> `fk` round trip on a plausible synthetic human (T-pose
  lengths, a swinging arm, a bending knee, 4 mm per-axis jitter, 60 frames):
  max 5.5 cm, mean 0.8 cm against `joints_world`, and `joints_world` itself is
  within 3.0 cm max / 0.9 cm mean of the noise-free truth.
- `demo.py --help` lists the four flags; `--export-formats json,xyz` exits
  with `unknown format(s) xyz; choose among json, npz, csv, bvh` before any
  model loads.
- `export_mocap.bat` passes the ASCII / CRLF check; read, not run.

### Not verified

- Real model weights (huggingface.co is blocked here): every number above is
  from the stand-in or synthetic data. In particular the **finger-order
  assumption** of `mocap_skeleton.py` (chains `wrist -> third -> second ->
  first -> tip`) is only testable on real predictions: `report()` prints
  `finger chains NOT ordered as expected` with the per-finger wrist distances
  when it fails. If it fails on a real clip the hand entries of
  `mocap_skeleton._BONES` and `FINGER_CHAINS` must be reversed.
- Blender import on a real PC: the file passes our parser and the 20tab `bvh`
  PyPI parser (same joints, offsets, parents, channel lists and values), the
  Blender axis statement in `GUIDE_WINDOWS.md` (file +Z = Blender -Y with the
  importer's default `-Z forward, Y up`) is from the importer's documented
  defaults, not from a run.
- `export_mocap.bat` on Windows (no cmd.exe here).

### Limitations

- One bone length per bone per clip: a very short clip, or a person mostly
  occluded, gives a wrong median and so a wrong rig for every frame.
- Anchoring is a translation only; it cannot fix a rotated or scaled body,
  and a person who never plants a foot (jumping, dancing on the toes) keeps
  the model's depth wobble. A foot that really slides (skating) is pinned.
- A person lost for more than `max_gap` (10) processed frames comes back as
  a new track id (the tracker retires them; `fill_gaps`'s hold-the-last-pose
  branch is only reachable from a caller that stamps its own ids), and two
  people crossing can swap ids (greedy IoU, no appearance model).
- The world frame is the camera's, flipped and then de-pitched: only the
  camera's pitch is estimated (from the floor's slope along depth), not its
  roll, so a phone held crooked still tilts the floor sideways; and only when
  someone covered a metre of depth, so a clip of a standing person in front
  of a pitched camera keeps the pitch (harmless there: the floor height is
  right at that depth).
- At most `--max-persons` (2) tracks per frame.

## Why the weights are not in the bundle

`huggingface.co` is blocked from the machine that produced this bundle, so
`instanthmr.onnx` (and the optional `.pth`) could not be fetched. The scripts
download it on the user's PC. Everything else the install needs (PyPI,
download.pytorch.org, storage.googleapis.com for RF-DETR, github.com for the
MHR assets) is reachable from a normal PC.

## Not verified here

- No Windows machine, cmd.exe or wine: the `.bat` files were read, not run.
- No GPU: the CUDA branches of `install.py` and the CUDA session test were only
  dry-run / monkeypatched.
- `download.pytorch.org` is blocked here, so the Windows wheel resolution
  (`torch==2.8.0 torchvision==0.23.0` from `/whl/cu128`, `/whl/cu126` and
  `/whl/cpu` for cp311/cp312) was not exercised. The 2.8.0 / 0.23.0 Windows
  cp311/cp312 wheels on PyPI exist, but PyPI's Windows wheel is the CPU build;
  the `+cu128` / `+cu126` Windows wheels live only on download.pytorch.org and
  could not be checked from here. torchvision 0.23.0 hard-pins `torch==2.8.0`.
- The MHR mesh path on Windows (pymomentum-gpu 0.1.114 cp312 win_amd64 against
  torch 2.8.0+cu128 / +cu126) is best effort: the wheel records no torch or
  CUDA-minor requirement; the renderer docstring says 0.1.113 needs torch
  2.8.x. The script fails safely (no `mhr_ready.txt`) if it does not load.
- Python 3.12.10 being the last 3.12 with a Windows installer comes from PEP
  693; python.org is blocked here, so the link was not clicked.
