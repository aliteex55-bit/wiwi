@echo off
rem ==================================================================
rem  OPTIONAL - body-mesh rendering with Meta's MHR body model.
rem
rem  The skeleton demo works WITHOUT this. Only run it if you want the
rem  full body surface drawn in the 3D view (slower, especially on CPU).
rem
rem  Needs:  setup_windows.bat already run with Python 3.12 (not 3.11).
rem            Setup installs torch 2.8, the version the mesh package is
rem            built for; torch is NOT changed by this script.
rem          Git for Windows installed  (https://git-scm.com/download/win)
rem          a 199 MB download that unpacks to 4.7 GB in models\mhr_assets
rem  Safe to re-run. The run_*.bat scripts use the mesh only after this
rem  script has finished successfully: it then writes the marker file
rem  models\mhr_assets\mhr_ready.txt. Delete that file to go back to
rem  skeleton-only.
rem ==================================================================
setlocal
set "ROOT=%~dp0"
if "%ROOT:~0,2%"=="\\" goto :on_network
cd /d "%~dp0"
if not exist "demo.py" goto :not_extracted
set "VPY=%~dp0.venv\Scripts\python.exe"
set "ZIP_URL=https://github.com/facebookresearch/MHR/releases/download/v1.0.1/assets.zip"
set "ZIP=%~dp0models\mhr_assets.zip"
set "ZIP_PART=%~dp0models\mhr_assets.zip.part"
set "DEST=%~dp0models\mhr_assets"
set "READY=%~dp0models\mhr_assets\mhr_ready.txt"
set "PINS=%TEMP%\instanthmr_pins.txt"
set "MIN_ZIP_BYTES=150000000"

echo.
echo ===== InstantHMR - optional body-mesh (MHR) setup =====
echo This downloads a 199 MB file that unpacks to 4.7 GB, then installs extra packages.
echo Close this window now if you do not want that. Otherwise
pause

rem ---------- checks ----------
set "STEP=checking the requirements"
if not exist "%VPY%" goto :no_venv
"%VPY%" -c "import sys; sys.exit(0 if sys.version_info[:2] == (3, 12) else 1)" >nul 2>&1
if not "%errorlevel%"=="0" goto :need_312
where git >nul 2>&1
if not "%errorlevel%"=="0" goto :no_git
where tar.exe >nul 2>&1
if not "%errorlevel%"=="0" goto :no_tar
if not exist "models\" mkdir "models"
rem The run scripts use the mesh only while the marker exists: remove it until every check below passes.
if exist "%READY%" del /f /q "%READY%"
rem The mesh package (pymomentum) is built against torch 2.8.x and its Windows wheel does not
rem tell pip which torch it needs. setup_windows.bat installs torch 2.8.0; the installed torch
rem and torchvision are written to a pin file so pip keeps exactly those builds.
"%VPY%" -m pip freeze --all 2>nul | findstr /i /b "torch== torchvision==" > "%PINS%"
findstr /c:"torch==2.8." "%PINS%" >nul
if not "%errorlevel%"=="0" goto :need_torch28
set "DEVICE=cpu"
if exist "config.bat" call "config.bat"

rem ---------- 1. download ----------
set "STEP=downloading assets.zip"
if exist "%DEST%\lod3.fbx" goto :assets_present
if not exist "%ZIP%" goto :download_zip
set "SIZE=0"
for %%A in ("%ZIP%") do set "SIZE=%%~zA"
if %SIZE% GEQ %MIN_ZIP_BYTES% goto :have_zip
del /f /q "%ZIP%"
:download_zip
echo [1/3] Downloading assets.zip (199 MB) from
echo   %ZIP_URL%
if exist "%ZIP_PART%" del /f /q "%ZIP_PART%"
where curl.exe >nul 2>&1
if not "%errorlevel%"=="0" goto :zip_powershell
curl.exe -L -f --retry 3 --retry-delay 3 -o "%ZIP_PART%" "%ZIP_URL%"
if not "%errorlevel%"=="0" goto :zip_powershell
goto :zip_check
:zip_powershell
echo curl did not work - trying PowerShell instead ...
if exist "%ZIP_PART%" del /f /q "%ZIP_PART%"
rem ZIP_URL and ZIP_PART are read by PowerShell from the environment ($env:) - splicing a path
rem into a quoted PowerShell string breaks when the user name contains an apostrophe.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri $env:ZIP_URL -OutFile $env:ZIP_PART -UseBasicParsing } catch { Write-Host $_; exit 1 }"
if not "%errorlevel%"=="0" goto :error
:zip_check
if not exist "%ZIP_PART%" goto :error
set "SIZE=0"
for %%A in ("%ZIP_PART%") do set "SIZE=%%~zA"
if %SIZE% LSS %MIN_ZIP_BYTES% goto :zip_too_small
move /y "%ZIP_PART%" "%ZIP%" >nul
if not "%errorlevel%"=="0" goto :error
:have_zip

rem ---------- 2. unpack ----------
set "STEP=unpacking assets.zip"
echo [2/3] Unpacking into models\mhr_assets (4.7 GB - takes a few minutes) ...
if not exist "%DEST%\" mkdir "%DEST%"
rem The zip has one top-level folder "assets\"; --strip-components=1 drops it
rem so the files land directly in models\mhr_assets.
tar.exe -xf "%ZIP%" -C "%DEST%" --strip-components=1
if not "%errorlevel%"=="0" goto :error
if not exist "%DEST%\lod3.fbx" goto :unpack_bad
del /f /q "%ZIP%"
echo       unpacked OK, assets.zip deleted.
:assets_present

rem ---------- 3. packages ----------
set "STEP=installing the MHR packages (pymomentum + mhr)"
echo [3/3] Installing the MHR packages (pymomentum and mhr). The installed torch is kept.
echo       This can take a while ...
rem Which torch build is installed (cu128, cu126 or cpu)? Its PyTorch index is added
rem so that anything pip looks up for torch is found in the right place.
"%VPY%" -c "import torch; v = torch.version.cuda; print('cpu' if v is None else 'cu' + v.replace('.', ''))" > "%TEMP%\instanthmr_torch_tag.txt"
if not "%errorlevel%"=="0" goto :error
set "TORCH_TAG="
set /p TORCH_TAG=<"%TEMP%\instanthmr_torch_tag.txt"
del /f /q "%TEMP%\instanthmr_torch_tag.txt"
echo       installed torch build: %TORCH_TAG%
if "%TORCH_TAG%"=="cpu" goto :pip_cpu
"%VPY%" -m pip install -r requirements-mhr.txt -c "%PINS%" --extra-index-url https://download.pytorch.org/whl/%TORCH_TAG%
if not "%errorlevel%"=="0" goto :error
goto :pip_done
:pip_cpu
echo       No CUDA torch installed - using the CPU build of pymomentum (slow, but works).
"%VPY%" -m pip install pymomentum-cpu "mhr @ git+https://github.com/facebookresearch/MHR.git" -c "%PINS%" --extra-index-url https://download.pytorch.org/whl/cpu
if not "%errorlevel%"=="0" goto :error
:pip_done
del /f /q "%PINS%"

set "STEP=verifying the MHR install"
"%VPY%" -c "import torch, torchvision, pymomentum, mhr; print('torch', torch.__version__, 'torchvision', torchvision.__version__, 'cuda available:', torch.cuda.is_available()); print('pymomentum ok, mhr ok')"
if not "%errorlevel%"=="0" goto :error
"%VPY%" -c "from instanthmr import PosePipeline; print('instanthmr ok')"
if not "%errorlevel%"=="0" goto :error
rem The pose model must still run where config.bat says (the CUDA build of onnxruntime must not have been replaced).
"%VPY%" -c "import sys, onnxruntime as ort; p = ort.get_available_providers(); print('onnxruntime', ort.__version__, 'providers:', p); sys.exit(0 if '%DEVICE%' != 'cuda' or 'CUDAExecutionProvider' in p else 1)"
if not "%errorlevel%"=="0" goto :ort_lost
rem Load the mesh model exactly as the run scripts will. A wrong pymomentum build crashes here, not later.
"%VPY%" -c "from instanthmr.mhr_renderer import MHRRenderer; MHRRenderer('models/mhr_assets', device='%DEVICE%'); print('mesh model loads ok')"
if not "%errorlevel%"=="0" goto :error
> "%READY%" echo ok

echo.
echo ===== Body-mesh setup finished. =====
echo The run_*.bat scripts now add --mhr-assets models\mhr_assets automatically
echo (they check for models\mhr_assets\mhr_ready.txt; delete that file to go back
echo to skeleton-only).
echo.
pause
exit /b 0

:no_venv
echo ***** Run setup_windows.bat first. *****
pause
exit /b 1

:on_network
echo ***** This folder is on a network location: "%ROOT%" *****
echo Copy the whole folder to this PC (for example C:\InstantHMR) and use the scripts there.
pause
exit /b 1

:not_extracted
echo ***** demo.py is not next to this file - you are probably running it from INSIDE the zip. *****
echo Right-click the zip, choose "Extract All...", then use the scripts in the new folder.
pause
exit /b 1

:need_312
echo.
echo ***** Body-mesh rendering needs Python 3.12 - the .venv folder uses another version. *****
echo Install Python 3.12.10 from https://www.python.org/ftp/python/3.12.10/python-3.12.10-amd64.exe ,
echo delete the .venv folder, run setup_windows.bat again, then run this file again.
echo.
pause
exit /b 1

:need_torch28
if exist "%PINS%" del /f /q "%PINS%"
echo.
echo ***** The installed PyTorch is not 2.8.x, which the body-mesh package needs. *****
echo Delete the .venv folder, run setup_windows.bat again (it installs torch 2.8),
echo then run this file again.
echo.
pause
exit /b 1

:no_git
echo.
echo ***** Git was not found. The mhr package is installed straight from GitHub and needs it. *****
echo Install "Git for Windows" from https://git-scm.com/download/win (default options),
echo then run this file again.
echo.
pause
exit /b 1

:no_tar
echo.
echo ***** tar.exe was not found. It ships with Windows 10 (1803) and later. *****
echo Unpack models\mhr_assets.zip by hand instead: right-click it, "Extract All...", then
echo move the files from the "assets" folder inside it into models\mhr_assets, so that
echo models\mhr_assets\lod3.fbx exists. Then run this file again.
echo.
pause
exit /b 1

:zip_too_small
del /f /q "%ZIP_PART%"
if exist "%PINS%" del /f /q "%PINS%"
echo.
echo ***** The downloaded file is only %SIZE% bytes - that is an error page, not the zip. *****
echo It has been deleted. Check the internet connection and try again, or download
echo   %ZIP_URL%
echo in a web browser and save it as models\mhr_assets.zip, then run this file again.
echo.
pause
exit /b 1

:unpack_bad
if exist "%PINS%" del /f /q "%PINS%"
echo.
echo ***** Unpacking finished but models\mhr_assets\lod3.fbx is missing. *****
echo The zip may be damaged. Delete models\mhr_assets.zip and models\mhr_assets, then run again.
echo.
pause
exit /b 1

:ort_lost
if exist "%PINS%" del /f /q "%PINS%"
echo.
echo ***** onnxruntime lost its GPU support (config.bat says DEVICE=cuda). *****
echo Open a Command Prompt in this folder and run:
echo   .venv\Scripts\python.exe -m pip install --force-reinstall --no-deps onnxruntime-gpu==1.26.0
echo then run this file again. The skeleton demo still works meanwhile, but on the CPU.
echo.
pause
exit /b 1

:error
if exist "%PINS%" del /f /q "%PINS%"
echo.
echo ***** MHR SETUP FAILED while %STEP%. *****
echo Scroll up and read the last error lines. The skeleton demo still works without the
echo mesh, and the run scripts ignore the mesh until this script finishes successfully.
echo If the error says "DLL load failed" or "undefined symbol", the installed torch is not
echo the 2.8.x build the mesh package needs: delete the .venv folder, run setup_windows.bat,
echo then this file again.
echo.
pause
exit /b 1
