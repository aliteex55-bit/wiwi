@echo off
rem ==================================================================
rem  MOTION CAPTURE: turn a video into animation files, written next to
rem  the video with the same name. For C:\clips\dance.mp4 it writes:
rem    dance.track0.bvh   the animation of the first person, for Blender
rem                       (File > Import > Motion Capture (.bvh)) and most
rem                       other 3D tools; a second person gets dance.track1.bvh
rem    dance.track0.csv   the same as a spreadsheet: one row per frame, the
rem                       x / y / z position of every joint in metres
rem    dance.mocap.json   everything, for programmers (also dance.mocap.npz)
rem    dance.rrd          the recording for the Rerun viewer (open_recording.bat)
rem  It works like record_video.bat: no viewer opens while the clip is
rem  processed, the viewer opens with the finished recording at the end.
rem  The exported skeleton is cleaned up: one set of bone lengths for the whole
rem  clip, the floor at height 0, and the body position steadied on whichever
rem  foot is standing on the floor. Smoothing is the centred moving average
rem  (it looks 2 frames ahead, so the exported motion does not lag).
rem  Drag a video onto this file, or run from a Command Prompt:
rem    export_mocap.bat "C:\path\to\clip.mp4"
rem  A photo (.jpg / .png) works too and gives a one-frame pose.
rem ==================================================================
setlocal
set "INPUT=%~f1"
set "PREFIX=%~dpn1"
set "RRD=%~dpn1.rrd"
set "OUTDIR=%~dp1"
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
rem The person detector runs on every frame when a graphics card runs it (steadier
rem tracking; setup writes DETECTOR=medium only when torch sees the card), on the
rem processor only on every 2nd frame, for speed.
set "STRIDE=2"
if "%DEVICE%"=="cuda" set "STRIDE=1"
if "%DETECTOR%"=="medium" set "STRIDE=1"

echo Motion capture of the video or photo: "%INPUT%"
echo   When it is finished these files are next to the video:
echo     "%PREFIX%.track0.bvh"   the animation, for Blender (one file per person: track0, track1, ...)
echo     "%PREFIX%.track0.csv"   joint positions per frame, as a spreadsheet
echo     "%PREFIX%.mocap.json"   everything, for programmers (also .mocap.npz)
echo     "%RRD%"   the recording for the Rerun viewer
echo     (the .rrd is large: about 5 GB per minute of Full-HD video - delete it if you only need the .bvh)
echo   (a track - even track0 - can be a false detection, not a person; the report marks it)
echo   (the video must be in a normal folder on this PC - a phone, camera or
echo    network location is read-only and nothing can be written there)
echo   device=%DEVICE%  detector=%DETECTOR%  smoothing=centered  detector-stride=%STRIDE%  %MHR%
echo   No viewer opens while the clip is processed - watch the frame lines here.
echo   At the end it prints a short report and "motion capture written:" with the
echo   files, then the viewer opens with the recording.
echo   To stop early: click this window, press Ctrl+C, then at "Terminate batch job (Y/N)?"
echo   answer N and press Enter so this window stays open. The files then cover
echo   the part of the clip processed so far.
echo.
rem Hides the harmless TracerWarning blocks that rfdetr's TorchScript pass prints at every start.
set "PYTHONWARNINGS=ignore"
rem A photo goes in as --image: read as a video it is one frame, too short to be kept as a person.
set "SRC=--video"
for %%E in (.jpg .jpeg .png .bmp .webp) do if /i "%~x1"=="%%E" set "SRC=--image"
".venv\Scripts\python.exe" demo.py %SRC% "%INPUT%" --device %DEVICE% --detector-variant %DETECTOR% --smoothing-filter --smoothing-mode centered --detector-stride %STRIDE% --no-spawn --save-rrd "%RRD%" --export "%PREFIX%" %MHR%
rem -1073741510 is the Windows code for "stopped with Ctrl+C" (while the models were still loading).
if "%errorlevel%"=="-1073741510" goto :stopped
if not "%errorlevel%"=="0" goto :failed
if not exist "%PREFIX%.track0.bvh" goto :no_person
echo.
echo Done. The new files are listed above under "motion capture written:" (in "%OUTDIR%").
echo.
echo To use the animation in Blender (4 or 5): File ^> Import ^> Motion Capture (.bvh), choose the
echo .track0.bvh file (or, if the report marks track 0 as a false detection, the first track it
echo does not mark). In the panel on the right set Scale to 0.01 (centimetres to metres), set
echo Start Frame to the number printed above under that file, tick "Update Scene FPS" (leave
echo "Update Scene Duration" unticked), and click Import BVH. Then set End at the bottom of the
echo timeline to the number printed above, and press Space to play. See GUIDE_WINDOWS.md,
echo section "Motion capture files".
echo.
echo Opening the recording in the Rerun viewer now. Close the viewer, then press any key to end this window.
".venv\Scripts\rerun.exe" "%RRD%"
pause
exit /b 0

:stopped
echo.
echo Stopped. If it was stopped while writing, the motion-capture files may be incomplete.
pause
exit /b 0

:no_person
echo.
echo ***** No person was found in the video or photo - no animation was written (see the report above). *****
echo See GUIDE_WINDOWS.md, section "Troubleshooting".
pause
exit /b 1

:usage
echo Usage: drag a video file onto export_mocap.bat
echo    or: export_mocap.bat "C:\path\to\clip.mp4"
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
echo If "motion capture written:" appears above, those files were still saved and cover the clip up to the error.
echo If it says "Failed to create file: Access is denied" (or "Permission denied") right after
echo Ready., copy the video to a folder on this PC first (for example C:\clips) and drag it from there.
echo If it says "cv2.VideoCapture failed", the file you dragged is not a video (.mp4 / .mov / .avi);
echo a .rrd recording goes onto open_recording.bat instead.
echo If it says "could not write" or "cannot write", that file is open in another program (Excel):
echo close it and run again.
echo See GUIDE_WINDOWS.md, section "Troubleshooting".
pause
exit /b 1
