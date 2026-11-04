@echo off
rem ==================================================================
rem  Downloads the InstantHMR model weights (instanthmr.onnx, about
rem  77 MB) from HuggingFace into the models folder.
rem  Safe to re-run: it skips the download if the file is already there.
rem  setup_windows.bat calls this; you can also double-click it.
rem ==================================================================
setlocal
set "ROOT=%~dp0"
if "%ROOT:~0,2%"=="\\" goto :on_network
cd /d "%~dp0"
set "URL=https://huggingface.co/momolesang/InstantHMR/resolve/main/instanthmr.onnx"
set "OUT=%~dp0models\instanthmr.onnx"
set "TMP_OUT=%~dp0models\instanthmr.onnx.part"
set "MIN_BYTES=50000000"
set "NOPAUSE=0"
if /i "%~1"=="nopause" set "NOPAUSE=1"

if not exist "models\" mkdir "models"

if not exist "%OUT%" goto :download
set "SIZE=0"
for %%A in ("%OUT%") do set "SIZE=%%~zA"
if %SIZE% GEQ %MIN_BYTES% goto :already_there
echo The existing models\instanthmr.onnx is only %SIZE% bytes - a failed download. Downloading again.
del /f /q "%OUT%"

:download
echo Downloading instanthmr.onnx (about 77 MB) from
echo   %URL%
if exist "%TMP_OUT%" del /f /q "%TMP_OUT%"
where curl.exe >nul 2>&1
if not "%errorlevel%"=="0" goto :use_powershell
curl.exe -L -f --retry 3 --retry-delay 3 -o "%TMP_OUT%" "%URL%"
if not "%errorlevel%"=="0" goto :use_powershell
goto :check

:use_powershell
echo curl did not work - trying PowerShell instead ...
if exist "%TMP_OUT%" del /f /q "%TMP_OUT%"
rem URL and TMP_OUT are read by PowerShell from the environment ($env:) - splicing a path
rem into a quoted PowerShell string breaks when the user name contains an apostrophe.
powershell -NoProfile -ExecutionPolicy Bypass -Command "$ProgressPreference = 'SilentlyContinue'; [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; try { Invoke-WebRequest -Uri $env:URL -OutFile $env:TMP_OUT -UseBasicParsing } catch { Write-Host $_; exit 1 }"
if not "%errorlevel%"=="0" goto :failed

:check
if not exist "%TMP_OUT%" goto :failed
set "SIZE=0"
for %%A in ("%TMP_OUT%") do set "SIZE=%%~zA"
if %SIZE% LSS %MIN_BYTES% goto :too_small
move /y "%TMP_OUT%" "%OUT%" >nul
if not "%errorlevel%"=="0" goto :failed
echo Downloaded models\instanthmr.onnx (%SIZE% bytes). OK.
goto :done_ok

:already_there
echo models\instanthmr.onnx is already there (%SIZE% bytes). Nothing to download.
goto :done_ok

:too_small
del /f /q "%TMP_OUT%"
echo.
echo ***** The downloaded file is only %SIZE% bytes. That is an error page, not the model. *****
echo It has been deleted. Possible causes: no internet, a firewall or proxy blocking
echo huggingface.co, or the file was moved. Try again later, or download it in a
echo web browser from
echo   %URL%
echo and save it as models\instanthmr.onnx inside this folder.
goto :done_fail

:failed
if exist "%TMP_OUT%" del /f /q "%TMP_OUT%"
echo.
echo ***** The download failed. *****
echo Possible causes: no internet, a firewall or proxy blocking huggingface.co.
echo Try again later, or download the file in a web browser from
echo   %URL%
echo and save it as models\instanthmr.onnx inside this folder.
goto :done_fail

:on_network
echo ***** This folder is on a network location: "%ROOT%" *****
echo Copy the whole folder to this PC (for example C:\InstantHMR) and use the scripts there.
pause
exit /b 1

:done_ok
if "%NOPAUSE%"=="0" pause
exit /b 0

:done_fail
if "%NOPAUSE%"=="0" pause
exit /b 1
