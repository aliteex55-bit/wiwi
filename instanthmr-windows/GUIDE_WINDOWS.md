# InstantHMR on Windows - plain-language guide

## What this is

InstantHMR turns a photo, a video or a live webcam picture of a person into a
moving 3D skeleton (70 joints: body, hands, feet) - a lightweight motion-capture
tool. It first finds the people in the picture with a detector (RF-DETR), then
estimates each person's 3D pose with the InstantHMR model, and shows the result
in a 3D viewer called Rerun: the original picture with the skeleton drawn on it,
the skeleton in 3D that you can rotate with the mouse, and a small speed graph.
Optionally it can also draw the full body surface (a "mesh") over the skeleton,
and it can write the motion of a video as animation files for Blender (see
"Motion capture files" below).

This folder is a ready-to-use bundle. You do not need to type any code: you
double-click one file to install, and drag files onto other files to run.

## What you need

- Windows 10 or 11, 64-bit.
- Python 3.12, 64-bit (3.11 also works; NOT 3.13 or 3.14). Free, see Step 1
  below.
- The Microsoft Visual C++ runtime (free, 25 MB). Most PCs already have it.
  Setup checks for it; if it is missing, install it from
  https://aka.ms/vs/17/release/vc_redist.x64.exe (run it, click **Install**,
  then **Close**) and run `setup_windows.bat` again.
- Internet during setup: about 4 to 5 GB of downloads with an NVIDIA graphics
  card, about 1 GB without.
- Free space on drive C: about 10 GB with an NVIDIA card, about 3 GB without.
  The installer keeps a copy of everything it downloads in a cache on C: even
  if the bundle is on another drive. After setup you can free roughly half of
  it: open a Command Prompt in this folder (click in the folder window's
  address bar, type `cmd`, press Enter) and run
  `.venv\Scripts\python.exe -m pip cache purge`. The optional body-mesh step
  needs another 8 GB.
- An NVIDIA graphics card is OPTIONAL. GeForce GTX 16-series / RTX 20-series
  or newer, with a driver from 2023 or later, runs everything on the card. A
  GTX 10-series card runs the person detector on the card and the pose model
  on the processor. Without an NVIDIA card everything still works on the
  processor, but more slowly. Setup detects all of this by itself.
- A webcam if you want live capture.

Tips before you start:

- Put this folder somewhere simple, for example `C:\InstantHMR`, not inside a
  OneDrive-synced folder, not on a network drive (the scripts refuse to run
  from a `\\server\...` location) and not too deep (very long paths can break
  installs).
- When you double-click a `.bat` file, Windows may ask **Open File - Security
  Warning** (click **Run**) or say **Windows protected your PC** (click **More
  info**, then **Run anyway**). Unblocking the zip in Step 0 avoids most of
  this. If instead it says **Smart App Control blocked...** or **This app has
  been blocked**, open Windows Security > App & browser control > Smart App
  Control settings and set it to **Off** (Windows will not let you turn it back
  on later without reinstalling Windows, so this is your call; the scripts are
  plain text you can read in Notepad).

## Step 0 - unpack the bundle

You received a `.zip` file (if it came by e-mail or as a download link, it is
in your Downloads folder). Windows can look inside a zip without unpacking
it, but the scripts only work from a real folder:

1. Right-click the zip file, choose **Properties**, tick **Unblock** at the
   bottom (if that box is shown), click **OK**. This stops Windows from warning
   about every file inside.
2. Right-click the zip again, choose **Extract All...**, type a simple
   destination such as `C:\InstantHMR`, click **Extract**.
3. Open the extracted folder. If it contains just one folder, open that one
   too. You must see `setup_windows.bat`, `demo.py` and a `models` folder next
   to each other. Everything below happens in this folder.

If you skip this and double-click a script inside the zip, Windows quietly
copies only that one file to a temporary folder and runs it there; the script
notices and stops with `demo.py is not next to this file`.

## Step 1 - install Python 3.12

1. Download the installer from this exact link (Python 3.12.10 is the newest
   3.12 that still has a Windows installer; the newer 3.12.x entries on
   python.org are "security releases" without one):
   https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
   (If that ever stops working, use Python 3.11.9:
   https://www.python.org/ftp/python/3.11.9/python-3.11.9-amd64.exe)
2. Run it. On the first screen tick **"Add python.exe to PATH"** (leave the
   other box as it is) and click **Install Now**. The `py` launcher that setup
   looks for is included automatically.
3. On the last screen, if you see a button **"Disable path length limit"**,
   click it (it avoids the "path too long" error mentioned in Troubleshooting),
   then click **Close** when it says "Setup was successful".

If you are not sure whether Python 3.12 is installed, go straight to Step 2:
it tells you if Python is missing.

## Step 2 - run the setup (once)

Double-click **`setup_windows.bat`**.

A black window opens and works through 8 steps. It can take 10 to 40 minutes,
most of it downloading. Plug the laptop in and do not let it go to sleep while
it runs. What it does:

1. Finds Python 3.12 (or 3.11).
2. Creates a private folder `.venv` with its own copy of Python - nothing else
   on your PC is changed.
3. Upgrades pip (the package installer).
4. Runs `install.py`, which picks the right packages: GPU versions if an NVIDIA
   card is found (`nvidia-smi` works), otherwise CPU versions. The line
   starting with `[install] platform:` tells you which it chose. (Just before
   this it checks that the Microsoft Visual C++ runtime is installed and stops
   with a download link if not - see "What you need".)
5. Downloads the model weights `models\instanthmr.onnx` (about 77 MB) from
   HuggingFace. **The weights are not inside the bundle** - the machine that
   built this bundle could not reach HuggingFace, so your PC fetches them on
   first setup. `download_weights.bat` does this and can be re-run on its own.
6. Tests whether the graphics card really works - once for the pose model
   (`DEVICE=cuda` or `cpu`; CUDA is NVIDIA's name for running programs on the
   graphics card) and once for the person detector
   (`DETECTOR=medium` or `nano`) - and writes the result into `config.bat`. The
   run scripts read that file. Any error messages from these tests go into
   `gpu_check.log` in this folder.
7. Downloads the person detector (RF-DETR, about 400 MB) into `models\rfdetr`
   and runs it once on a blank picture.
8. Checks that everything loads and runs the pose model once on the device
   written to `config.bat`.

A good result ends with lines like these, and then "Press any key to continue":

```
[8/8] Verifying ...
ok
pose model ok, running on CUDAExecutionProvider
onnxruntime 1.26.0 providers: ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
torch 2.8.0+cu128 cuda available: True

===== SETUP FINISHED - everything is ready =====
  device=cuda  detector=medium  (change them in config.bat if needed)
```

In the `providers:` line only the word `CUDAExecutionProvider` matters; the
other names vary. On a PC without an NVIDIA card that line reads
`['AzureExecutionProvider', 'CPUExecutionProvider']` and you will see
`device=cpu  detector=nano`, `pose model ok, running on CPUExecutionProvider`,
`torch 2.8.0+cpu` and `cuda available: False` - that is normal. On a GTX
10-series card you get `device=cpu  detector=medium` and `cuda available: True`.

If instead you see `***** SETUP FAILED while ... *****`, scroll up, read the
last error lines, and look in "Troubleshooting" below. Running
`setup_windows.bat` again is always safe; it continues where it can.

## Step 3 - use it

**How to "drag a file onto" a script**: open a second window showing your
photo or video (or use the Desktop), press and hold the left mouse button on
it, move it over the `run_image.bat` icon until that icon highlights, and let
go. Drag one file at a time. A black window opens with `Running InstantHMR on
photo: "C:\...\photo.jpg"`. It then prints a line starting with `Warming up
models` and finally `Ready.` -
this pause is normal: 10 to 30 seconds with a graphics card, one or two minutes
without. Then the **Rerun viewer** opens (it may open behind the black window -
look in the taskbar). It has three areas: **Camera** (your picture with the
skeleton drawn on it), **3D Scene** (the skeleton in space: left-drag rotates,
right-drag pans, mouse wheel zooms) and **Latency (ms)** (a speed graph).

The first time the viewer opens, Windows may show **Windows Security Alert -
Windows Defender Firewall has blocked some features of this app** for
`rerun.exe`. Click **Allow access** (clicking **Cancel** also works: the viewer
only talks to this PC).

Each run opens two things: the black window (progress and messages) and the
Rerun viewer window (the result). Close the viewer when you are done, then
press a key in the black window (for the webcam, see below). If a viewer
window is still open from an earlier run, the new run appears inside it as a
second recording (listed on the left); close the old viewer first if that
confuses you.

**A photo**: drag a `.jpg` or `.png` file onto **`run_image.bat`**.
The viewer shows the photo with the skeleton drawn on it, and the 3D skeleton
you can rotate with the mouse. The black window prints a line like
`image: persons= 1  RF-DETR= 45.2 ms  InstantHMR=  5.1 ms  total= 51.0 ms ( 19.6 fps)`.

**A video file**: drag a `.mp4`, `.mov` or `.avi` onto **`run_video.bat`**.
The viewer fills up as the clip is processed; drag the time slider at the
bottom to scrub (move back and forth in the clip). How long it takes: with a
graphics card about as long as the clip itself, on the processor 5 to 15 times
the clip's length (a 30-second
phone clip: well under a minute with a card, several minutes without). The
black window first prints `video: clip.mp4, 900 frames @ 30.0 fps` and then a
progress line every 30 frames, for example
`frame 30/900 ... total= 18.6 ms ( 53.8 fps)` - as long as those keep coming,
it is working. Keep clips to a few minutes: the viewer holds every frame in
memory and starts dropping the oldest ones when memory is full. Smoothing is
on (the `1euro` filter: no lag, less jitter) and the person detector runs on
every 2nd frame to go faster. To stop early, click on the black window and
press **Ctrl+C**; at `Terminate batch job (Y/N)?` type **N** and press Enter so
the window stays open and you can read the `summary` line (if you press Ctrl+C
while the
models are still loading, it just says `Stopped before the clip was
processed.`). At the end it prints a `summary` line with the average speed,
and a short jitter report.

**The webcam**: double-click **`run_camera.bat`**. It uses the default camera.
If the wrong camera opens (or none), open a Command Prompt in this folder (click
in the folder window's address bar, type `cmd`, press Enter) and run
`run_camera.bat 1` (or 2, 3 ...). To stop: click once on the black window (the
viewer has the keyboard after it opens), then press **Ctrl+C**. Windows asks
`Terminate batch job (Y/N)?` - type **N** and press Enter to keep the window
open (it prints `Stopped.`), or **Y** to close it. Closing the viewer window
does **not** stop the camera; closing the black window with its X does. If the
picture freezes right after you clicked in the black window, you selected text
by accident (Windows 10): press Esc in that window and it continues.
`run_camera.bat` takes a camera number, not a file: dragging a video onto it
only prints a message (use `run_video.bat` for files).

**Record a video to a file you can reopen**: drag the video onto
**`record_video.bat`** instead of `run_video.bat`. Unlike `run_video.bat`, no
viewer opens while the clip is processed: the black window prints a progress
line every 30 frames (`frame 30/900 ...`), and when it is finished the viewer
opens with the result. The recording is saved next to the video with the same
name and the ending `.rrd` (for example `C:\clips\dance.mp4` gives
`C:\clips\dance.rrd`). Copy the video to a folder on this PC first (for example
`C:\clips`): the recording is written next to the video, which does not work
from a phone, camera or network location. To open a recording later, drag the
`.rrd` file onto **`open_recording.bat`**. The viewer opens exactly as it was,
without re-processing the video. The recording is only readable by the viewer
that is installed in `.venv` (the Rerun version must match), so keep the bundle.

**Is the graphics card being used?** Look at the `RF-DETR=` numbers printed
every 30 frames: a graphics card gives roughly 10 to 50 ms, the processor 300
to 1500 ms. The first run of the day is slower: the detector is loaded and the
model warms up before the first frame appears.

**Changing the settings**: `config.bat` is a small text file (two comment
lines and two settings). Right-click it and choose **Edit** (on Windows 11 you
may first have to click **Show more options**); do not double-click it, that
runs it and does nothing visible. `DEVICE` is `cuda` (NVIDIA card) or `cpu`.
`DETECTOR` is the person detector size: `nano` (fastest), `small`, `medium`
(default on a GPU), `base` or `large` (most accurate, slowest). After changing
`DETECTOR`, the next run first downloads that detector (about 400 MB, once).
If `config.bat` is missing, the run scripts say so and use `cuda` / `medium`
until you run `setup_windows.bat` again.

## Motion capture files (BVH for Blender)

`run_video.bat` and `record_video.bat` only show the result. To get the
motion as files you can use in Blender or another 3D program, drag the video
onto **`export_mocap.bat`** instead. It processes the clip like
`record_video.bat` (no viewer until the end; with a graphics card it is
somewhat slower, because the person detector runs on every frame; on the
processor it takes 5 to 15 times the clip's length), then turns the per-frame
skeletons into one steady animation per person and writes these files next
to the video. For `C:\clips\dance.mp4`:

- `dance.track0.bvh` - the animation, for Blender and most other 3D tools
  (BVH is the standard motion-capture text format). One file per person:
  `track0` is the first person found, a second person gets `track1`. An
  extra track can also be a false detection (a coat on a hook, a reflection):
  the report, see below, marks a track that is short or low in confidence
  with `may be a false detection` - ignore its files.
- `dance.track0.csv` - the same animation as a spreadsheet (opens in Excel):
  one row per video frame, with the x / y / z position of every joint in
  metres.
- `dance.mocap.json` and `dance.mocap.npz` - everything (all persons, the raw
  and the cleaned-up positions, the model's own parameters), for programmers.
- `dance.rrd` - the viewer recording, like the one `record_video.bat` makes
  (with the export's steadier smoothing). The viewer opens with it when the
  script is finished.

Copy the video to a folder on this PC first (for example `C:\clips`): the
files are written next to the video, which does not work from a phone, camera
or network location.

While it runs, the black window prints a progress line every 30 frames, as
`record_video.bat` does. To stop early, click the window, press Ctrl+C and
answer N at `Terminate batch job (Y/N)?`; the files then cover the part of
the clip processed so far. At the end it prints a short report, for example:

```
mocap: 1 track(s)
  track 0: frames 0-899, 900 samples, 3 filled in, mean confidence 0.91
floor: y = -0.812 m before levelling (now y = 0); camera pitched 8.0 deg down, floor levelled
foot anchoring: root movement per frame (real motion counts too), speed of planted feet
  track 0: root 6.3 -> 2.1 mm/frame (67% less); planted feet 0.141 -> 0.009 m/s (a foot is planted on 72% of foot-frames)
finger chains ordered as expected
motion capture written:
  C:\clips\dance.mocap.json
  C:\clips\dance.mocap.npz
  C:\clips\dance.track0.csv
  C:\clips\dance.track0.bvh
```

(The numbers are an invented example.) `samples` is the number of frames in
the animation, `mean confidence` is how sure the detector was, on average,
that this is a person (0 to 1), and `foot-frames` counts each foot in each
frame separately, so 72% means a foot was on the floor about three quarters
of the time. `filled in` is the number of frames where the person was not
found and the skeleton was filled in from the frames around it. `camera
pitched ... down, floor levelled` means the camera was looking down a little
and the program straightened the floor; `camera pitch not measured` means the
person did not move far enough towards or away from the camera to tell (one
metre is needed), and the camera is taken to be level. `root ... mm/frame`
is how much the body moved from one frame to
the next, before and after the feet were used to steady it; `planted feet` is
how fast a foot that stands on the floor was still moving - after the
clean-up it should be close to 0. A line `finger chains NOT ordered as
expected` means the hand points came out in an order the program did not
expect: the body is fine, only the fingers in the files may be wrong - tell
the developer. Then `motion capture written:` lists the files and the viewer
opens.

**Opening the animation in Blender** (version 4.x, free from
https://www.blender.org): **File > Import > Motion Capture (.bvh)**, choose
the `.track0.bvh` file. In the panel on the right of the file window set
**Scale** to `0.01` (the file is in centimetres and Blender works in metres;
the scale only changes the size, never the motion), and in the **Animation**
part of the same panel tick **Update Scene FPS** and **Update Scene
Duration** (otherwise Blender keeps its own 24 frames per second and
250-frame timeline: a 30 fps clip then plays in slow motion and stops after 8
seconds), then click **Import BVH**. A stick figure appears. Press **Space** to play it and drag the time
slider at the bottom to scrub. The floor is at height 0, and the person faces
the +Z direction of the file (towards the camera that filmed them), which with
Blender's default import settings puts them facing you in the Front view
(numpad 1).

**What to expect.** The skeleton has 70 points: body, face, both hands with
all fingers, and the feet. The body proportions are measured once over the
whole clip and held constant, so the figure does not change size from frame
to frame. The feet are used to steady the body: a foot that stands on the
floor is held in place on the floor, which removes most of the back-and-forth
sliding the model would otherwise produce. Fast motion, a person partly out of the
picture, two people crossing, or a person leaving and re-entering the frame
can still produce glitches: a jump, a twisted limb for a few frames, or one
person split into `track0` and `track1`. Nothing here is perfect - it is the
model's estimate of each single frame, cleaned up, not a studio motion
capture.

## Optional - draw the body surface (mesh)

By default you get a skeleton. If you want the full body surface drawn over it:

- You need Python **3.12** (not 3.11), **Git for Windows**
  (https://git-scm.com/download/win, default options), 8 GB of free disk and
  a 199 MB download. An NVIDIA card is strongly recommended; on the CPU it is
  slow.
- Double-click **`download_mhr_assets.bat`**. It downloads Meta's MHR body
  model files (`assets.zip`), unpacks them into `models\mhr_assets`, installs
  two more packages (`pymomentum` and `mhr`) and then loads the mesh model once
  to check it works. PyTorch stays at version 2.8, the version the mesh package
  was built for (setup installed exactly that; the script refuses to continue
  if it finds another version).
- A good result ends with `torch 2.8.0+cu128 ... cuda available: True` on an
  NVIDIA PC (`+cpu ... False` without one), an `onnxruntime ... providers:`
  line that still contains `CUDAExecutionProvider` on an NVIDIA PC,
  `mesh model loads ok` and `===== Body-mesh setup finished. =====`.
- From then on, every `run_*.bat` adds the mesh automatically: it checks for
  `models\mhr_assets\mhr_ready.txt`, which is written only when
  `download_mhr_assets.bat` finishes successfully. To go back to skeleton-only,
  delete that `mhr_ready.txt` file (or the whole `models\mhr_assets` folder).

If the window instead says `MHR SETUP FAILED while verifying`, the mesh
packages do not work on this PC: skip the mesh - the skeleton demo is
unaffected and the run scripts ignore the mesh until this script succeeds. If
the error mentions `DLL load failed` or `undefined symbol`, the installed
PyTorch is not 2.8.x: delete the `.venv` folder, run `setup_windows.bat`, then
`download_mhr_assets.bat` again. If the skeleton demo misbehaves after a
failed mesh install, delete the `.venv` folder and run `setup_windows.bat`
again to get back the known-good packages.

## Troubleshooting

**"64-bit Python 3.12 (or 3.11) was not found"** - Do Step 1 with the exact
link given there (it is the 64-bit installer; a 32-bit Python is not accepted
because the packages only exist for 64-bit). In the installer the "Add
python.exe to PATH" box must be ticked. If Python is installed but still not
found, uninstall and reinstall it with that box ticked. Python 3.13 and 3.14
are not accepted on purpose: some packages have no version for them yet.

**"The .venv folder was made with a different Python version (or a 32-bit
one)"** - delete the `.venv` folder inside the bundle and run
`setup_windows.bat` again.

**"This folder is on a network location"** - the bundle sits on a network
share (`\\server\...`), where the scripts cannot run. Copy the whole folder
to this PC, for example `C:\InstantHMR`, and use the scripts there.

**"The Microsoft Visual C++ runtime is missing"**, **`OSError: [WinError 126]
The specified module could not be found. Error loading ...\torch\lib\c10.dll`**,
**`DLL load failed while importing onnxruntime_pybind11_state`** or
**`Microsoft Visual C++ Redistributable is not installed`** - the Microsoft
Visual C++ runtime is missing. Install
https://aka.ms/vs/17/release/vc_redist.x64.exe (run it, click **Install**, then
**Close**), then run `setup_windows.bat` again (no need to delete `.venv`).

**`can't open file ... install.py`** or **`demo.py is not next to this file`**
- the scripts were started from inside the zip, or one script was copied
somewhere on its own. Do Step 0 and use the scripts in the extracted folder.

**Weights download failed / "only N bytes, that is an error page"** - your PC
could not reach huggingface.co (no internet, a company firewall, a proxy -
a company gateway that filters internet traffic).
Try again later or on another network, or download the file in a browser:
https://huggingface.co/momolesang/InstantHMR/resolve/main/instanthmr.onnx
and save it as `models\instanthmr.onnx` inside the bundle (about 77 MB). Then
run `setup_windows.bat` again.

**`[error] InstantHMR ONNX not found: models\instanthmr.onnx`** - the model
weights were never downloaded (setup step 5 failed). Double-click
`download_weights.bat`; when it prints `Downloaded models\instanthmr.onnx (...
bytes). OK.` try the run script again. See the weights entry above if it fails.

**`[error] cv2.imread failed for ...`** or **`[error] cv2.VideoCapture failed
for ...`** - the photo or video is in a format the program cannot read. Phone
photos are often `.heic`: on an iPhone set Settings > Camera > Formats >
**Most Compatible** before taking the picture, or open the picture in the
Windows Photos app and use **Save as** to make a `.jpg`. Videos must be
`.mp4`, `.mov` or `.avi`. If you dragged a `.rrd`, `.bvh` or `.json` onto a
run script by mistake: a `.rrd` goes onto `open_recording.bat`, the others
are results, not inputs.

**"unsupported platform"** - this message comes from the original `install.py`,
which only knew Linux and macOS. The `install.py` in this bundle handles
Windows. If you see it, you are running an old copy; use this bundle's file.

**`AttributeError: ... has no attribute 'optimize_for_inference'`** - the file
`instanthmr\detector.py` in this bundle was replaced by the original upstream
one. Upstream's copy calls a function that newer versions of the detector
package no longer have; the bundle's copy works with old and new versions.
Put the bundle's `instanthmr\detector.py` back (extract it from the zip again).
Do not downgrade or pin the detector package.

**Setup chose `device=cpu` although this PC has an NVIDIA card / the demo is
slow** - Scroll up in the setup window to the line starting with
`[install] platform:`. `Windows without NVIDIA GPU` means no NVIDIA driver is
installed: install it from https://www.nvidia.com/drivers, restart the PC, then
run `setup_windows.bat` again - it notices that the processor-only packages are
installed, prints `installed torch is the cpu build - replacing it with cu128`
and swaps them for the graphics-card versions (another 3 to 4 GB download; no
need to delete `.venv`). `Windows + NVIDIA (compute capability ...)` (compute
capability is NVIDIA's number for the card's generation; 7.5 and up is GTX 16 /
RTX 20 or newer) means the card was found but could not be used: open
`gpu_check.log` in Notepad.
`Require cuDNN 9.* and CUDA 12.*`, `CUDA driver version is insufficient` or
`no kernel image is available` means the driver is too old or the card is
too old for the CUDA 12.8 packages: update the driver from
https://www.nvidia.com/drivers (version 528 or newer, i.e. any driver from
2023 on), restart the PC, then run `setup_windows.bat` again - do NOT delete
`.venv` in this case: the right packages are already installed, setup only
repeats the test and rewrites `config.bat`. If you cannot update, stay on the
CPU: `config.bat` already says `cpu` and `nano`. A compute capability below
7.5 (GTX 10-series) is expected to give `device=cpu  detector=medium` - the
card then runs only the detector. Deleting `.venv` and running setup again is
the last resort for a badly broken install; it downloads everything again.

**Slow on a PC without a graphics card** - the detector is the expensive part.
`config.bat` already uses the `nano` detector on CPU, and the video and camera
scripts run it only every 2nd frame. Expect a few frames per second with
one person. Smaller videos (720p) help. Running on the CPU is normal for
checking a clip; it is not fast enough for smooth live capture.

**"cannot open camera index 0"** - another program (Teams, Zoom, the Camera
app) is using the webcam, or the camera has another number. Close the other
program, or run `run_camera.bat 1` from a Command Prompt. Also check Windows
Settings > Privacy > Camera: desktop apps must be allowed to use the camera.

**The black window only says `Usage: drag a photo (jpg or png) onto
run_image.bat`** (or the video equivalent) - you double-clicked a run script
without giving it a file. Drag a photo or video onto the script instead (see
Step 3), or run it from a Command Prompt with the file path in quotes.

**A black window flashes and disappears** - Windows (Smart App Control) or an
antivirus stopped the script before it could print anything. See the tips at
the start of this guide and the antivirus entry below.

**"Smart App Control blocked an app that may be unsafe" / "This app has been
blocked"** - open Windows Security > App & browser control > Smart App Control
settings and set it to **Off**. Windows will not let you turn it back on later
without reinstalling Windows, so this is your call; the scripts are plain text
you can read in Notepad.

**`No space left on device` / `There is not enough space on the disk`** - free
space on drive C: (see "What you need"), then run setup again.

**Installing packages fails with a red error mentioning "pip" or a package
name** - usually a dropped internet connection or a company proxy. Run
`setup_windows.bat` again; pip resumes with what is already installed. If it
mentions "Long path" or "path too long", move the bundle to `C:\InstantHMR`
and run setup again.

**Antivirus deletes or blocks files** - some antivirus tools quarantine newly
downloaded `.exe` / `.dll` files inside `.venv`. Add the bundle folder to the
antivirus exceptions and run setup again.

**Nothing is detected in the picture** - the detector needs a mostly visible
person, roughly upright, not too small in the frame. Only the 2 most confident
people are processed per frame.

**`mocap: no person was tracked for 2 frame(s) or more` in the report, and
`motion capture written:` lists only the `.mocap.json` and `.mocap.npz`
files - no `.bvh` or `.csv` appeared** - there was no animation to write (the
`.mocap.json` and `.mocap.npz` files are still written, without tracks). See
"Nothing is detected in the picture". **`Failed to create file: Access is
denied` (or `Permission denied`, `No such file or directory`) right after
`Ready.`, before any `frame` line** - the recording and the files go next to
the video, so the video must be in a folder this PC can write to: copy it to
`C:\clips` and drag it from there.

**`MHR SETUP FAILED`** - see the end of "Optional - draw the body surface".

## What was built where

- Source: https://github.com/Naturalpad-studio/InstantHMR at commit
  `811a486e3b73a0d44e02c48f73780bbe0d204e8a` (also in `UPSTREAM_COMMIT.txt`).
  The original README.md in this folder is the upstream one and describes the
  project in developer terms.
- Changed for Windows: `install.py` now treats Windows like Linux (NVIDIA
  detection through `nvidia-smi`, CUDA or CPU packages), installs PyTorch 2.8.0
  (CUDA 12.8 build for GTX 16 / RTX 20-series and newer, CUDA 12.6 build for
  GTX 10-series, CPU build otherwise), pins `onnxruntime-gpu` to 1.26.0 (the
  last version built for CUDA 12) and asks for the detector package `rfdetr`
  1.8 or newer (`requirements.txt` says the same). When it is run again on an
  existing `.venv` it replaces a PyTorch or onnxruntime of the wrong kind (CPU
  vs. graphics card) instead of keeping it.
- Changed for newer detector packages: `instanthmr\detector.py` accepts both
  the old and the new name of the detector's start-up function
  (`optimize_for_inference` / `inference`), so any `rfdetr` from 1.8 on works.
  Keep this file: see the `AttributeError` entry in Troubleshooting.
- Added, all plain text, all re-runnable:
  - `setup_windows.bat` - the one-click installer described in Step 2. It
    refuses to run from a network location, checks for the Visual C++ runtime
    and ends by loading the pose model once on the chosen device.
  - `download_weights.bat` - fetches `models\instanthmr.onnx` (curl, or
    PowerShell as fallback), refuses files under 50 MB.
  - `download_mhr_assets.bat` - the optional body-mesh step.
  - `run_image.bat`, `run_video.bat`, `run_camera.bat`, `record_video.bat` -
    run `demo.py` with sensible motion-capture settings; they read `config.bat`.
    `record_video.bat` processes the clip without a viewer and opens the saved
    `.rrd` in the viewer afterwards.
  - `export_mocap.bat` - like `record_video.bat`, but also writes the
    motion-capture files (`.bvh`, `.csv`, `.mocap.json`, `.mocap.npz`) next
    to the video; see "Motion capture files".
  - `instanthmr\mocap_skeleton.py`, `instanthmr\mocap.py`, `instanthmr\bvh.py`
    and the `tests` folder - the code behind `export_mocap.bat` (new, not
    from upstream; `BUNDLE_NOTES.md` describes it).
  - `open_recording.bat` - opens a saved `.rrd` recording in the viewer.
  - `config.bat` - written by setup: `DEVICE` and `DETECTOR`.
  - `gpu_check.log` - written by setup: the error output of the GPU tests.
  - `setup_linux_mac.sh` - the same setup for Linux / macOS, from a terminal:
    `bash setup_linux_mac.sh`.
  - `GUIDE_WINDOWS.md` (this file) and `BUNDLE_NOTES.md` (technical notes).
- Everything the setup downloads stays inside this folder: `.venv` (packages),
  `models\instanthmr.onnx` (model), `models\rfdetr` (detector),
  `models\mhr_assets` (optional mesh) - except pip's download cache in
  `%LocalAppData%\pip\Cache` (see "What you need" for how to empty it).
  Deleting the folder removes everything else.
- The model weights are licensed under the SAM license and the code under
  Apache 2.0 - see `LICENSE` and `models\README.md`.
