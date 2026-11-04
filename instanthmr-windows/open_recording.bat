@echo off
rem ==================================================================
rem  Opens a saved recording (.rrd file, made by record_video.bat) in the
rem  Rerun viewer, without processing the video again.
rem  Drag a .rrd file onto this file, or run from a Command Prompt:
rem    open_recording.bat "C:\clips\dance.rrd"
rem ==================================================================
setlocal
set "INPUT=%~f1"
set "ROOT=%~dp0"
if "%ROOT:~0,2%"=="\\" goto :on_network
cd /d "%~dp0"
if not exist "demo.py" goto :not_extracted
if "%~1"=="" goto :usage
if not exist "%INPUT%" goto :missing
if not exist ".venv\Scripts\rerun.exe" goto :no_setup
echo Opening "%INPUT%" in the Rerun viewer ...
".venv\Scripts\rerun.exe" "%INPUT%"
if not "%errorlevel%"=="0" goto :failed
exit /b 0

:usage
echo Usage: drag a .rrd file onto open_recording.bat
echo    or: open_recording.bat "C:\clips\dance.rrd"
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
echo ***** The viewer could not open the recording. Scroll up to read why. *****
echo A recording can only be opened by the viewer version that made it, so keep the bundle.
pause
exit /b 1
