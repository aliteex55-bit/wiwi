#!/usr/bin/env python3
"""Platform-aware installer for InstantHMR.

Picks the right Torch + ONNX Runtime wheels for the current machine:

  - macOS (Apple Silicon / Intel) — stock ``torch`` + ``onnxruntime``
    (which bundles the CoreML execution provider).
  - Linux + NVIDIA GPU — Torch 2.8.0 built for CUDA 12.8 (cu128, native
    kernels for Turing .. Blackwell) plus ``onnxruntime-gpu`` 1.26.0 (the
    last CUDA 12 build; it needs the CUDA 12.8 DLLs that the cu128 torch
    wheel ships). Pascal / Volta cards get the cu126 torch wheel and the
    CPU build of onnxruntime. On Linux the ``nvidia-*-cu12`` runtime
    wheels are added so a system CUDA install isn't required.
  - Linux without an NVIDIA GPU — Torch 2.8.0 CPU wheel + ``onnxruntime``.
  - Windows — same branches as Linux (NVIDIA detected through
    ``nvidia-smi`` -> CUDA wheels, otherwise CPU wheels).

torch is pinned to 2.8.0 everywhere (torchvision 0.23.0): it satisfies
rfdetr (>=2.2) and transformers 5 (>=2.6), and it is the ABI the optional
``pymomentum`` body-mesh package is built against, so the optional MHR
step never has to replace torch.

Run *after* you've created and activated a fresh Python environment::

    python install.py

For a dry run that just prints the commands::

    python install.py --dry-run
"""

from __future__ import annotations

import argparse
import platform
import subprocess
import sys
from typing import Optional


def run_pip(args: list[str], dry_run: bool = False) -> None:
    cmd = [sys.executable, "-m", "pip", "install"] + args
    print(">>", " ".join(cmd))
    if not dry_run:
        subprocess.check_call(cmd)


def run_pip_uninstall(packages: list[str], dry_run: bool = False) -> None:
    """``pip uninstall -y`` - the exit code is ignored (not installed is fine)."""
    cmd = [sys.executable, "-m", "pip", "uninstall", "-y"] + packages
    print(">>", " ".join(cmd))
    if not dry_run:
        subprocess.call(cmd)


def installed_torch_build() -> Optional[str]:
    """``'cu128'`` / ``'cu126'`` / ``'cpu'`` for the installed torch, ``None`` if absent.

    Wheels from download.pytorch.org carry the build as a local version
    label (``2.8.0+cu128``, ``2.8.0+cpu``). A label-less version means a
    PyPI wheel, which on Windows is CPU-only (on Linux it is a CUDA build,
    but install.py never installs torch from PyPI there, so at worst it is
    replaced once by the same thing from the index).
    """
    try:
        from importlib.metadata import version
        v = version("torch")
    except Exception:
        return None
    return v.split("+", 1)[1] if "+" in v else "cpu"


def install_torch(tag: str, dry_run: bool) -> None:
    """torch 2.8.0 + torchvision 0.23.0 from ``download.pytorch.org/whl/<tag>``.

    pip considers an installed ``2.8.0+cpu`` as already satisfying
    ``torch==2.8.0`` (PEP 440 ignores the local label), so a .venv first
    built without a GPU driver would keep the CPU build forever. If a
    different build is installed it is removed first, so re-running the
    setup after a driver install really switches to the CUDA wheels.
    """
    have = installed_torch_build()
    if have is not None and have != tag:
        print(f"[install]   installed torch is the {have} build - replacing it with {tag}")
        run_pip_uninstall(["torch", "torchvision"], dry_run)
    run_pip([
        "--index-url", f"https://download.pytorch.org/whl/{tag}",
        "torch==2.8.0", "torchvision==0.23.0",
    ], dry_run)


def install_onnxruntime(gpu: bool, dry_run: bool) -> None:
    """``onnxruntime-gpu`` 1.26.0 (CUDA 12) or the CPU ``onnxruntime``.

    Both wheels unpack into the same ``onnxruntime/`` folder, so installing
    one on top of the other leaves a file-by-file mix (no import error, but
    the CUDA provider silently disappears). The other flavour is removed
    first, which makes switching between ``--force-cpu`` and GPU installs
    on the same .venv safe.
    """
    run_pip_uninstall(["onnxruntime"] if gpu else ["onnxruntime-gpu"], dry_run)
    if gpu:
        # Last CUDA 12 build. 1.27+ is built against CUDA 13 and silently
        # falls back to CPU next to the cu12 torch wheel.
        run_pip(["onnxruntime-gpu==1.26.0"], dry_run)
    else:
        run_pip(["onnxruntime>=1.19"], dry_run)


def detect_cuda_compute_cap() -> Optional[float]:
    """Return the highest CUDA compute capability across visible GPUs.

    Returns ``None`` if ``nvidia-smi`` is missing, fails, or reports no
    GPU.
    """
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=compute_cap", "--format=csv,noheader"],
            text=True, stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    caps: list[float] = []
    for line in out.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            caps.append(float(line))
        except ValueError:
            pass
    return max(caps) if caps else None


def install_mac(dry_run: bool) -> None:
    print("[install] platform: macOS — using stock torch + onnxruntime (CoreML EP)")
    run_pip(["torch>=2.2", "torchvision"], dry_run)
    run_pip(["onnxruntime>=1.19"], dry_run)


def install_linux_cpu(dry_run: bool) -> None:
    print(f"[install] platform: {platform.system()} without NVIDIA GPU - using CPU wheels")
    install_torch("cpu", dry_run)
    install_onnxruntime(False, dry_run)


def install_linux_cuda(compute_cap: float, dry_run: bool) -> None:
    print(f"[install] platform: {platform.system()} + NVIDIA (compute capability {compute_cap})")
    if compute_cap >= 7.5:
        # Turing .. Blackwell (sm_75 .. sm_120). onnxruntime-gpu 1.21-1.26 is
        # built with CUDA 12.8 and, on Windows, loads cudart/cublas/cuDNN from
        # torch\lib, so torch MUST be a cu128 build. 2.8.x is also the ABI the
        # optional pymomentum-gpu (MHR body mesh) wheel is built against.
        print("[install]   -> torch 2.8.0 cu128 + onnxruntime-gpu 1.26.0")
        install_torch("cu128", dry_run)
        install_onnxruntime(True, dry_run)
        if platform.system() != "Windows":
            # Linux: ORT finds these in site-packages/nvidia. On Windows the
            # cu128 torch wheel ships every DLL in torch\lib and ORT loads
            # them from there, so the ~1 GB of extra wheels would be unused.
            run_pip([
                "nvidia-cudnn-cu12>=9.0,<10",
                "nvidia-cuda-runtime-cu12",
                "nvidia-cublas-cu12",
                "nvidia-cufft-cu12",
            ], dry_run)
    else:
        # Pascal / Volta (GTX 10-series, Titan V): the cu128 wheels carry no
        # sm_6x / sm_70 kernels and the cu126 wheels carry no CUDA 12.8 DLLs
        # for onnxruntime-gpu, so RF-DETR runs on the GPU and the small ONNX
        # regressor runs on the CPU.
        print("[install]   -> pre-Turing GPU: torch 2.8.0 cu126 + CPU onnxruntime")
        install_torch("cu126", dry_run)
        install_onnxruntime(False, dry_run)


def install_common(dry_run: bool) -> None:
    print("[install] common deps")
    run_pip([
        "numpy>=1.24",
        "opencv-python>=4.8",
        "pillow>=10",
        # >=1.8 for RF_HOME (weights cache dir). detector.py copes with the
        # 1.9+ rename of optimize_for_inference() to inference().
        "rfdetr>=1.8",
        "rerun-sdk>=0.21",
    ], dry_run)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Print pip commands but don't run them.")
    parser.add_argument("--force-cpu", action="store_true",
                        help="Force the CPU-only path on Linux / Windows even if an NVIDIA GPU is detected.")
    args = parser.parse_args()

    sysname = platform.system()
    machine = platform.machine()
    print(f"[install] python={platform.python_version()}  system={sysname}  machine={machine}")

    if sysname == "Darwin":
        install_mac(args.dry_run)
    elif sysname in ("Linux", "Windows"):
        if args.force_cpu:
            install_linux_cpu(args.dry_run)
        else:
            cap = detect_cuda_compute_cap()
            if cap is None:
                install_linux_cpu(args.dry_run)
            else:
                install_linux_cuda(cap, args.dry_run)
    else:
        sys.exit(f"[install] unsupported platform: {sysname}. "
                 f"Install torch + onnxruntime manually, then `pip install -r requirements.txt`.")

    install_common(args.dry_run)
    print("\n[install] done. verify with: python -c 'from instanthmr import PosePipeline; print(\"ok\")'")


if __name__ == "__main__":
    main()
