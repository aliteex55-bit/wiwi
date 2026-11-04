#!/usr/bin/env bash
# InstantHMR setup for Linux and macOS - the equivalent of setup_windows.bat.
# Run it from a terminal inside this folder:
#     bash setup_linux_mac.sh          (add --cpu to force a CPU-only install)
# It finds Python 3.12 / 3.11, creates .venv, runs install.py, downloads the
# model weights with curl, checks the GPU, downloads the person detector and
# verifies that everything loads. Re-running it is safe.
set -u
cd "$(dirname "$0")" || exit 1
ROOT="$(pwd)"
FORCE_CPU=0
[ "${1:-}" = "--cpu" ] && FORCE_CPU=1
URL="https://huggingface.co/momolesang/InstantHMR/resolve/main/instanthmr.onnx"
MIN_BYTES=50000000

STEP="starting"
fail() {
  echo
  echo "***** SETUP FAILED while $STEP. *****"
  echo "Read the error lines above. GUIDE_WINDOWS.md, section 'Troubleshooting', applies here too."
  exit 1
}
is_311_312() {
  "$1" -c 'import sys; sys.exit(0 if sys.version_info[:2] in ((3, 11), (3, 12)) else 1)' >/dev/null 2>&1
}
size_of() { wc -c < "$1" | tr -d ' '; }

echo
echo "===== InstantHMR setup ($(uname -s)) ====="
echo "Folder: $ROOT"
[ "$FORCE_CPU" = 1 ] && echo "CPU-only install requested."
echo

# ---------- 1. Python ----------
STEP="finding Python 3.12 or 3.11"
PYTHON=""
for c in python3.12 python3.11 python3 python; do
  if command -v "$c" >/dev/null 2>&1 && is_311_312 "$c"; then PYTHON="$c"; break; fi
done
if [ -z "$PYTHON" ]; then
  echo "***** Python 3.12 (or 3.11) was not found. *****"
  echo "  Ubuntu / Debian:   sudo apt install python3.12 python3.12-venv"
  echo "  macOS (Homebrew):  brew install python@3.12"
  echo "  or download it from https://www.python.org/downloads/"
  echo "Then run this script again."
  exit 1
fi
echo "[1/7] Using $PYTHON ($("$PYTHON" --version 2>&1))"

# ---------- 2. .venv ----------
STEP="creating the Python environment (.venv)"
VPY="$ROOT/.venv/bin/python"
if [ -x "$VPY" ]; then
  echo "[2/7] The .venv folder already exists, reusing it."
  if ! is_311_312 "$VPY"; then
    echo "***** .venv was made with a different Python version. Delete the .venv folder and run again. *****"
    exit 1
  fi
else
  echo "[2/7] Creating the .venv folder ..."
  if ! "$PYTHON" -m venv .venv; then
    echo "If the error above mentions ensurepip: on Ubuntu/Debian run  sudo apt install python3.12-venv  (or python3.11-venv), then run this script again."
    fail
  fi
fi

# ---------- 3. pip ----------
STEP="upgrading pip"
echo "[3/7] Upgrading pip ..."
"$VPY" -m pip install --upgrade pip || fail

# ---------- 4. packages ----------
STEP="installing the packages with install.py"
echo "[4/7] Installing packages (1 to 5 GB download, be patient) ..."
if [ "$FORCE_CPU" = 1 ]; then
  "$VPY" install.py --force-cpu || fail
else
  "$VPY" install.py || fail
fi

# ---------- 5. model weights ----------
STEP="downloading the model weights"
echo "[5/7] Downloading the InstantHMR model weights ..."
mkdir -p models
if [ -f models/instanthmr.onnx ] && [ "$(size_of models/instanthmr.onnx)" -ge "$MIN_BYTES" ]; then
  echo "      models/instanthmr.onnx is already there ($(size_of models/instanthmr.onnx) bytes)."
else
  rm -f models/instanthmr.onnx models/instanthmr.onnx.part
  if ! curl -L -f --retry 3 --retry-delay 3 -o models/instanthmr.onnx.part "$URL"; then
    rm -f models/instanthmr.onnx.part
    echo "Download failed. Check the internet connection / proxy, or open"
    echo "  $URL"
    echo "in a browser and save it as models/instanthmr.onnx."
    fail
  fi
  if [ "$(size_of models/instanthmr.onnx.part)" -lt "$MIN_BYTES" ]; then
    rm -f models/instanthmr.onnx.part
    echo "The downloaded file is too small - an error page, not the model. It was deleted."
    fail
  fi
  mv models/instanthmr.onnx.part models/instanthmr.onnx
  echo "      downloaded models/instanthmr.onnx ($(size_of models/instanthmr.onnx) bytes)."
fi

# ---------- 6. device ----------
STEP="checking the GPU"
echo "[6/7] Checking which device can run the model ..."
DEVICE=cpu
DETECTOR=nano
if [ "$FORCE_CPU" != 1 ]; then
  if [ "$(uname -s)" = "Darwin" ]; then
    if "$VPY" -c "import sys, onnxruntime as ort; sys.exit(0 if 'CoreMLExecutionProvider' in ort.get_available_providers() else 1)" 2>/dev/null; then
      DEVICE=coreml
    fi
  else
    # DEVICE: can onnxruntime run the pose model on the GPU? (same order as demo.py:
    # onnxruntime loads its CUDA libraries before torch is imported)
    if "$VPY" -c "import sys, onnxruntime as ort; getattr(ort, 'preload_dlls', lambda: None)(); s = ort.InferenceSession('models/instanthmr.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); ok = s.get_providers()[0] == 'CUDAExecutionProvider'; print('onnxruntime on the GPU:', ok); sys.exit(0 if ok else 1)" 2>gpu_check.log; then
      DEVICE=cuda
    fi
    # DETECTOR: can torch (the person detector) use the GPU? medium if yes, nano if not.
    if "$VPY" -c "import sys, torch; ok = torch.cuda.is_available(); print('torch on the GPU:', ok); sys.exit(0 if ok else 1)" 2>>gpu_check.log; then
      DETECTOR=medium
    fi
  fi
fi
echo "      device=$DEVICE  detector=$DETECTOR   (details in gpu_check.log)"

# ---------- 7. detector + verification ----------
STEP="downloading the RF-DETR person detector"
export RF_HOME="$ROOT/models/rfdetr"
echo "[7/7] Downloading the RF-DETR person detector ($DETECTOR) into models/rfdetr and verifying ..."
# Same call path as demo.py (optimisation pass + one dummy frame), so a package problem shows up here.
"$VPY" -c "from instanthmr import RFDETRDetector; d = RFDETRDetector(variant='$DETECTOR'); d.warmup(); print('detector ok')" || fail
STEP="verifying the installation"
"$VPY" -c "from instanthmr import PosePipeline; print('ok')" || fail
# Same path as the demo (preload_dlls, session, one run) on the detected device.
STEP="loading the pose model on device=$DEVICE"
"$VPY" -c "from instanthmr import InstantHMR; m = InstantHMR('models/instanthmr.onnx', device='$DEVICE'); m.warmup(max_batch_size=2); print('pose model ok, running on', m.active_provider)" || fail
STEP="verifying the installation"
"$VPY" -c "import onnxruntime as ort; print('onnxruntime', ort.__version__, 'providers:', ort.get_available_providers())" || fail
"$VPY" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())" || fail

cat <<MSG

===== SETUP FINISHED - everything is ready =====
Run the demo from this folder (RF_HOME keeps the detector weights inside it):

  export RF_HOME="$ROOT/models/rfdetr"
  .venv/bin/python demo.py --image photo.jpg --device $DEVICE --detector-variant $DETECTOR
  .venv/bin/python demo.py --video clip.mp4 --device $DEVICE --detector-variant $DETECTOR --smoothing-filter --smoothing-mode 1euro --detector-stride 2
  .venv/bin/python demo.py --camera 0     --device $DEVICE --detector-variant $DETECTOR --smoothing-filter --detector-stride 2
  .venv/bin/python demo.py --video clip.mp4 --device $DEVICE --detector-variant $DETECTOR --smoothing-filter --export clip
      (motion capture: writes clip.track0.bvh for Blender, clip.track0.csv, clip.mocap.json and .npz next to clip.mp4)

Add  --save-rrd clip.rrd  to keep a recording; reopen it later with:  .venv/bin/rerun clip.rrd
MSG
