@echo off
rem ==================================================================
rem  LIVE motion capture from the WEBCAM.
rem  Double-click this file (uses camera 0, the default webcam), or give
rem  another camera number from a Command Prompt:
rem    run_camera.bat 1
rem  Smoothing (1euro filter) is on and the person detector runs every
rem  2nd frame for speed. To stop: click this window, press Ctrl+C.
rem ==================================================================
setlocal
set "ROOT=%~dp0"
if "%ROOT:~0,2%"=="\\" goto :on_network
cd /d "%~dp0"
if not exist "demo.py" goto :not_extracted
set "CAM=%~1"
if "%CAM%"=="" set "CAM=0"
if not exist ".venv\Scripts\python.exe" goto :no_setup

set "DEVICE=cuda"
set "DETECTOR=medium"
if exist "config.bat" call "config.bat"
set "RF_HOME=%~dp0models\rfdetr"
set "MHR="
rem mhr_ready.txt is written by download_mhr_assets.bat only after all its checks pass.
if exist "models\mhr_assets\mhr_ready.txt" set "MHR=--mhr-assets models\mhr_assets"

echo Starting live capture from camera %CAM%
echo   device=%DEVICE%  detector=%DETECTOR%  smoothing=1euro  detector-stride=2  %MHR%
echo   A Rerun viewer window opens. Closing the viewer does NOT stop the camera.
echo   To stop: click THIS window, press Ctrl+C, then answer N to "Terminate batch job"
echo   so this window stays open (Y closes it at once).
echo   If the wrong camera opens, or none, try: run_camera.bat 1
echo.
".venv\Scripts\python.exe" demo.py --camera %CAM% --device %DEVICE% --detector-variant %DETECTOR% --smoothing-filter --detector-stride 2 %MHR%
rem -1073741510 is the Windows code for "stopped with Ctrl+C" (while the models were still loading).
if "%errorlevel%"=="-1073741510" goto :stopped
if not "%errorlevel%"=="0" goto :failed
:stopped
echo.
echo Stopped.
pause
exit /b 0

:no_setup
echo ***** The .venv folder is missing. Double-click setup_windows.bat first. *****
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

:failed
echo.
echo ***** InstantHMR stopped with an error. Scroll up to read it. *****
echo If it says "cannot open camera index", another program may be using the webcam,
echo or try a different number: run_camera.bat 1
echo See GUIDE_WINDOWS.md, section "Troubleshooting".
pause
exit /b 1
