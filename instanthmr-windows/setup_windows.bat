@echo off
rem ==================================================================
rem  InstantHMR - one-click setup for Windows 10 / 11
rem
rem  Double-click this file. It
rem    1. finds Python 3.12 (or 3.11) on this PC
rem    2. creates a private Python environment in the folder .venv
rem    3. installs the packages (install.py picks GPU or CPU versions)
rem    4. downloads the model weights (download_weights.bat)
rem    5. checks whether the NVIDIA GPU works and writes config.bat
rem    6. downloads and test-runs the person detector, test-runs the pose
rem       model and checks everything loads
rem
rem  Re-running it is safe. To force a CPU-only install, open a
rem  Command Prompt in this folder and run:  setup_windows.bat --cpu
rem ==================================================================
setlocal
set "ROOT=%~dp0"
if "%ROOT:~0,2%"=="\\" goto :on_network
cd /d "%ROOT%"
if not exist "%ROOT%install.py" goto :not_extracted
if not exist "%ROOT%demo.py" goto :not_extracted
set "VPY=%~dp0.venv\Scripts\python.exe"
set "FORCE_CPU=0"
if /i "%~1"=="--cpu" set "FORCE_CPU=1"
if /i "%~1"=="/cpu" set "FORCE_CPU=1"

echo.
echo ===== InstantHMR setup =====
echo Folder: "%ROOT%"
if "%FORCE_CPU%"=="1" echo CPU-only install requested.
echo.

rem ---------- 1. find Python 3.12 or 3.11 ----------
rem ("%errorlevel%"=="0" rather than "not errorlevel 1": a program that crashes
rem  hard exits with a NEGATIVE code, which "not errorlevel 1" would count as success)
set "STEP=finding Python 3.12 or 3.11"
set "PYTHON="
py -3.12 -c "import sys" >nul 2>&1
if "%errorlevel%"=="0" set "PYTHON=py -3.12"
if defined PYTHON goto :python_found
py -3.11 -c "import sys" >nul 2>&1
if "%errorlevel%"=="0" set "PYTHON=py -3.11"
if defined PYTHON goto :python_found
python -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor in (11, 12) else 1)" >nul 2>&1
if "%errorlevel%"=="0" set "PYTHON=python"
if defined PYTHON goto :python_found
goto :no_python

:python_found
echo [1/8] Using Python: %PYTHON%
%PYTHON% --version
if not "%errorlevel%"=="0" goto :error

rem ---------- 2. create the private environment ----------
set "STEP=creating the Python environment (.venv)"
if exist "%VPY%" goto :venv_exists
echo [2/8] Creating the .venv folder ...
%PYTHON% -m venv ".venv"
if not "%errorlevel%"=="0" goto :error
goto :venv_ready
:venv_exists
echo [2/8] The .venv folder already exists, reusing it.
"%VPY%" -c "import sys; sys.exit(0 if sys.version_info.major == 3 and sys.version_info.minor in (11, 12) else 1)" >nul 2>&1
if not "%errorlevel%"=="0" goto :venv_wrong_version
:venv_ready

rem ---------- 3. pip ----------
set "STEP=upgrading pip"
echo [3/8] Upgrading pip ...
"%VPY%" -m pip install --upgrade pip
if not "%errorlevel%"=="0" goto :error

rem ---------- Visual C++ runtime (torch and onnxruntime need it) ----------
set "STEP=checking the Microsoft Visual C++ runtime"
if exist "%SystemRoot%\System32\msvcp140.dll" if exist "%SystemRoot%\System32\vcruntime140_1.dll" goto :vcredist_ok
echo.
echo ***** The Microsoft Visual C++ runtime is missing (torch and onnxruntime need it). *****
echo   1. Download and run https://aka.ms/vs/17/release/vc_redist.x64.exe (click Install, then Close).
echo   2. Double-click setup_windows.bat again.
echo.
pause
exit /b 1
:vcredist_ok

rem ---------- 4. packages ----------
set "STEP=installing the packages with install.py"
echo [4/8] Installing packages. This downloads 1 to 5 GB - it can take 10 to 40 minutes ...
set "INSTALL_ARGS="
if "%FORCE_CPU%"=="1" set "INSTALL_ARGS=--force-cpu"
"%VPY%" install.py %INSTALL_ARGS%
if not "%errorlevel%"=="0" goto :error

rem ---------- 5. model weights ----------
set "STEP=downloading the model weights"
echo [5/8] Downloading the InstantHMR model weights ...
call "%~dp0download_weights.bat" nopause
if not "%errorlevel%"=="0" goto :error

rem ---------- 6. GPU check -> config.bat ----------
set "STEP=checking the GPU"
echo [6/8] Checking whether the NVIDIA GPU can be used (details go to gpu_check.log) ...
set "DEVICE=cpu"
set "DETECTOR=nano"
if "%FORCE_CPU%"=="1" goto :write_config
rem DEVICE: can onnxruntime run the pose model on the GPU? Same order as demo.py
rem (onnxruntime loads its CUDA DLLs before torch is imported).
"%VPY%" -c "import sys, onnxruntime as ort; getattr(ort, 'preload_dlls', lambda: None)(); s = ort.InferenceSession('models/instanthmr.onnx', providers=['CUDAExecutionProvider', 'CPUExecutionProvider']); ok = s.get_providers()[0] == 'CUDAExecutionProvider'; print('onnxruntime on the GPU:', ok); sys.exit(0 if ok else 1)" 2> "gpu_check.log"
if "%errorlevel%"=="0" set "DEVICE=cuda"
rem DETECTOR: can torch (the person detector) use the GPU? medium if yes, nano if not.
"%VPY%" -c "import sys, torch; ok = torch.cuda.is_available(); print('torch on the GPU:', ok); sys.exit(0 if ok else 1)" 2>> "gpu_check.log"
if "%errorlevel%"=="0" set "DETECTOR=medium"
:write_config
> "config.bat" echo rem InstantHMR settings, written by setup_windows.bat. Right-click this file and choose Edit to change them.
>> "config.bat" echo rem DEVICE is cuda (NVIDIA GPU) or cpu. DETECTOR is nano, small, medium, base or large.
>> "config.bat" echo set "DEVICE=%DEVICE%"
>> "config.bat" echo set "DETECTOR=%DETECTOR%"
echo       device=%DEVICE%  detector=%DETECTOR%  - saved in config.bat
if "%DETECTOR%"=="nano" echo       No usable NVIDIA GPU: the demo will run on the CPU, which is slower.
if "%DETECTOR%"=="nano" if "%FORCE_CPU%"=="0" echo       (gpu_check.log in this folder says why - only interesting if this PC has an NVIDIA card)
if "%DEVICE%"=="cpu" if "%DETECTOR%"=="medium" echo       The NVIDIA card runs the person detector; the pose model runs on the CPU (older card, see gpu_check.log).

rem ---------- 7. person detector ----------
set "STEP=downloading the RF-DETR person detector"
echo [7/8] Downloading the RF-DETR person detector (%DETECTOR%) into models\rfdetr and test-running it ...
set "RF_HOME=%~dp0models\rfdetr"
rem Same call path as demo.py (optimisation pass + one dummy frame), so a package problem shows up here.
"%VPY%" -c "from instanthmr import RFDETRDetector; d = RFDETRDetector(variant='%DETECTOR%'); d.warmup(); print('detector ok')"
if not "%errorlevel%"=="0" goto :error

rem ---------- 8. verification ----------
set "STEP=verifying the installation"
echo [8/8] Verifying ...
"%VPY%" -c "from instanthmr import PosePipeline; print('ok')"
if not "%errorlevel%"=="0" goto :error
set "STEP=loading the pose model on device=%DEVICE%"
rem Same path as the run scripts (preload_dlls, session, one run) on the device written to config.bat,
rem so a damaged weights file or a CUDA problem that only shows when the model runs fails here, not later.
"%VPY%" -c "from instanthmr import InstantHMR; m = InstantHMR('models/instanthmr.onnx', device='%DEVICE%'); m.warmup(max_batch_size=2); print('pose model ok, running on', m.active_provider)"
if not "%errorlevel%"=="0" goto :error
set "STEP=verifying the installation"
"%VPY%" -c "import onnxruntime as ort; print('onnxruntime', ort.__version__, 'providers:', ort.get_available_providers())"
if not "%errorlevel%"=="0" goto :error
"%VPY%" -c "import torch; print('torch', torch.__version__, 'cuda available:', torch.cuda.is_available())"
if not "%errorlevel%"=="0" goto :error

echo.
echo ===== SETUP FINISHED - everything is ready =====
echo   device=%DEVICE%  detector=%DETECTOR%  (change them in config.bat if needed)
echo.
echo Next:
echo   - drag a photo onto run_image.bat
echo   - drag a video onto run_video.bat   (record_video.bat also saves a .rrd file)
echo   - double-click run_camera.bat for the webcam
echo See GUIDE_WINDOWS.md for details.
echo.
pause
exit /b 0

:on_network
echo.
echo ***** This folder is on a network location: "%ROOT%" *****
echo The scripts and the packages they install cannot run from a network share.
echo Copy the whole folder to this PC first (for example to C:\InstantHMR) and
echo double-click setup_windows.bat there.
echo.
pause
exit /b 1

:not_extracted
echo.
echo ***** install.py and demo.py are not next to this file. *****
echo You probably opened the zip without extracting it, or copied only this file.
echo Right-click the zip, choose "Extract All...", open the extracted folder and
echo double-click setup_windows.bat there. See GUIDE_WINDOWS.md, Step 0.
echo.
pause
exit /b 1

:no_python
echo.
echo ***** Python 3.12 (or 3.11) was not found on this PC. *****
echo.
echo   1. Download Python 3.12.10 (the newest 3.12 that has a Windows installer):
echo      https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe
echo   2. Run it. On the first screen tick "Add python.exe to PATH", then click "Install Now".
echo      On the last screen click "Disable path length limit" if it is offered, then Close.
echo   3. Double-click setup_windows.bat again.
echo.
pause
exit /b 1

:venv_wrong_version
echo.
echo ***** The .venv folder was made with a different Python version. *****
echo Delete the folder .venv inside "%ROOT%" and double-click setup_windows.bat again.
echo.
pause
exit /b 1

:error
echo.
echo ***** SETUP FAILED while %STEP%. *****
echo Scroll up and read the last error lines.
echo See GUIDE_WINDOWS.md, section "Troubleshooting". Re-running setup_windows.bat is safe.
echo.
pause
exit /b 1
