@echo off
rem ==================================================================
rem  Like run_video.bat, but SAVES the result as a Rerun recording (.rrd)
rem  next to the video, with the same name:
rem    C:\clips\dance.mp4  ->  C:\clips\dance.rrd
rem  No viewer opens while the clip is processed (--no-spawn: with a live
rem  viewer, rr.save() would take over as the only sink and the viewer would
rem  stay empty); the viewer opens with the finished recording afterwards.
rem  Reopen a recording later by dragging the .rrd onto open_recording.bat.
rem  Drag a video onto this file, or run from a Command Prompt:
rem    record_video.bat "C:\path\to\clip.mp4"
rem ==================================================================
setlocal
set "INPUT=%~f1"
set "RRD=%~dpn1.rrd"
set "ROOT=%~dp0"
if "%ROOT:~0,2%"=="\\" goto :on_network
cd /d "%~dp0"
if not exist "demo.py" goto :not_extracted
if "%~1"=="" goto :usage
if not exist "%INPUT%" goto :missing
if not exist ".venv\Scripts\python.exe" goto :no_setup

set "DEVICE=cuda"
set "DETECTOR=medium"
if exist "config.bat" call "config.bat"
if not exist "config.bat" echo   (config.bat is missing - run setup_windows.bat again to detect the GPU; using device=cuda detector=medium for now)
set "RF_HOME=%~dp0models\rfdetr"
set "MHR="
rem mhr_ready.txt is written by download_mhr_assets.bat only after all its checks pass.
if exist "models\mhr_assets\mhr_ready.txt" set "MHR=--mhr-assets models\mhr_assets"

echo Running InstantHMR on video: "%INPUT%"
echo   recording to: "%RRD%"
echo   (the video must be in a normal folder on this PC - a phone, camera or
echo    network location is read-only and the recording cannot be saved there)
echo   device=%DEVICE%  detector=%DETECTOR%  smoothing=1euro  detector-stride=2  %MHR%
echo   No viewer opens while the clip is processed - watch the frame lines here.
echo   The viewer opens with the result when it is finished.
echo   To stop early: click this window, press Ctrl+C, then at "Terminate batch job (Y/N)?"
echo   answer N and press Enter so this window stays open.
echo.
rem Hides the harmless TracerWarning blocks that rfdetr's TorchScript pass prints at every start.
set "PYTHONWARNINGS=ignore"
".venv\Scripts\python.exe" demo.py --video "%INPUT%" --device %DEVICE% --detector-variant %DETECTOR% --smoothing-filter --smoothing-mode 1euro --detector-stride 2 --no-spawn --save-rrd "%RRD%" %MHR%
rem -1073741510 is the Windows code for "stopped with Ctrl+C" (while the models were still loading).
if "%errorlevel%"=="-1073741510" goto :stopped
if not "%errorlevel%"=="0" goto :failed
echo.
echo Done. The recording was saved to:
echo   "%RRD%"
echo Opening it in the Rerun viewer now. Close the viewer, then press any key to end this window.
echo To open it again later, drag that .rrd file onto open_recording.bat.
".venv\Scripts\rerun.exe" "%RRD%"
pause
exit /b 0

:stopped
echo.
echo Stopped before the clip was processed. The .rrd recording is incomplete.
pause
exit /b 0

:usage
echo Usage: drag a video file onto record_video.bat
echo    or: record_video.bat "C:\path\to\clip.mp4"
pause
exit /b 1

:missing
echo ***** File not found: "%INPUT%" *****
pause
exit /b 1

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
echo If it says "Failed to create file: Access is denied" (or "Permission denied") for the
echo .rrd file, copy the video to a folder on this PC first (for example C:\clips) and drag it from there.
echo See GUIDE_WINDOWS.md, section "Troubleshooting".
pause
exit /b 1
