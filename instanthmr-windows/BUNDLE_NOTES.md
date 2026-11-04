# Bundle notes (for developers)

This folder is upstream InstantHMR
(https://github.com/Naturalpad-studio/InstantHMR, commit in
`UPSTREAM_COMMIT.txt`) with `.git` removed and a thin Windows layer on top.
Nothing under `instanthmr/`, `demo.py`, `tools/`, training or benchmark code
was touched.

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
- `rfdetr>=1.0` became `rfdetr>=1.8,<1.11` (also in `requirements.txt`):
  rfdetr 1.11.0 (2026-09-23) removed `optimize_for_inference()` (deprecated
  alias of `inference()` since 1.9.0), which `instanthmr/detector.py` calls by
  default -> `AttributeError` at pipeline construction on every fresh install.
  1.8.0 is the oldest version verified here to honour `RF_HOME`.
- The two "Linux ..." messages print `platform.system()`; the GPU-branch
  messages use `->` instead of the U+2192 arrow (not encodable in cp1252 if
  output is ever redirected to a file).
- `--dry-run`, `--force-cpu` and the Darwin branch are unchanged. Verified here
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

- `setup_windows.bat [--cpu]` - finds `py -3.12`, `py -3.11`, then `python`
  (version-checked, 3.13 rejected because rfdetr/pymomentum wheels lag);
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
  lines into a pip constraints file, refusing unless torch is 2.8.x ->
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
  gives both). `run_camera.bat`, `run_video.bat` and `record_video.bat` treat
  exit code -1073741510 (0xC000013A, Ctrl+C during model loading, before
  demo.py's `except KeyboardInterrupt`) as "Stopped", not as an error.
- `open_recording.bat` - drag a `.rrd` onto it: runs `.venv\Scripts\rerun.exe
  file.rrd` (the `rerun` console script ships with `rerun-sdk`; viewer and SDK
  versions must match).
- `setup_linux_mac.sh [--cpu]` - same steps in bash (`python3.12`,
  `python3.11`, `python3`, `python` with a version check; `.venv/bin/python`;
  `coreml` on macOS when the CoreML EP is present, `cuda` / `medium` on Linux
  via the same two tests, stderr to `gpu_check.log`; same detector warm-up).
  Prints the `demo.py` commands with the detected device instead of writing a
  config file.
- `GUIDE_WINDOWS.md` - end-user guide (unpacking the zip, security dialogs,
  Smart App Control, the firewall prompt for `rerun.exe`, how to stop the
  camera, disk-space and pip-cache notes, `gpu_check.log`).

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
  `/whl/cpu` for cp311/cp312) was not exercised; the 2.8.0 / 0.23.0 Windows
  cp311/cp312 wheels exist on PyPI and torchvision 0.23.0 hard-pins
  `torch==2.8.0`.
- The MHR mesh path on Windows (pymomentum-gpu 0.1.114 cp312 win_amd64 against
  torch 2.8.0+cu128 / +cu126) is best effort: the wheel records no torch or
  CUDA-minor requirement; the renderer docstring says 0.1.113 needs torch
  2.8.x. The script fails safely (no `mhr_ready.txt`) if it does not load.
- Python 3.12.10 being the last 3.12 with a Windows installer comes from PEP
  693; python.org is blocked here, so the link was not clicked.
