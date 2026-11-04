@echo off
rem ==================================================================
rem  Run InstantHMR on a VIDEO FILE (motion capture of a clip).
rem  Drag a video (mp4 / mov / avi) onto this file, or run from a Command Prompt:
rem    run_video.bat "C:\path\to\clip.mp4"
rem  A Rerun viewer window opens and fills up as the clip is processed.
rem  Smoothing (1euro filter) is on, and the person detector runs every
rem  2nd frame for speed. Use record_video.bat to also save a .rrd file.
rem ==================================================================
setlocal
set "INPUT=%~f1"
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
echo   device=%DEVICE%  detector=%DETECTOR%  smoothing=1euro  detector-stride=2  %MHR%
echo   To stop early: click this window, press Ctrl+C, then at "Terminate batch job (Y/N)?"
echo   answer N and press Enter so this window stays open.
echo.
rem Hides the harmless TracerWarning blocks that rfdetr's TorchScript pass prints at every start.
set "PYTHONWARNINGS=ignore"
".venv\Scripts\python.exe" demo.py --video "%INPUT%" --device %DEVICE% --detector-variant %DETECTOR% --smoothing-filter --smoothing-mode 1euro --detector-stride 2 %MHR%
rem -1073741510 is the Windows code for "stopped with Ctrl+C" (while the models were still loading).
if "%errorlevel%"=="-1073741510" goto :stopped
if not "%errorlevel%"=="0" goto :failed
echo.
echo Done. Close the Rerun viewer window when you have finished looking.
pause
exit /b 0

:stopped
echo.
echo Stopped before the clip was processed.
pause
exit /b 0

:usage
echo Usage: drag a video file onto run_video.bat
echo    or: run_video.bat "C:\path\to\clip.mp4"
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
echo See GUIDE_WINDOWS.md, section "Troubleshooting".
pause
exit /b 1
