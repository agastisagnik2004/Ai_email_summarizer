@echo off
setlocal EnableExtensions
title Embrify - AI Email Summarizer
cd /d "%~dp0"

rem =====================================================================================
rem  Embrify - AI Email Summarizer: one-click install + start for Windows.
rem  First run: finds or installs Python 3.12, creates a private environment in
rem  %LOCALAPPDATA%\Embrify, installs the packages and downloads missing models.
rem  Later runs: starts straight away. Close this window to stop the app.
rem =====================================================================================

set "PORT=7860"
set "URL=http://127.0.0.1:%PORT%"
set "ENVDIR=%LOCALAPPDATA%\Embrify"
set "VENV=%ENVDIR%\venv"
set "VPY=%VENV%\Scripts\python.exe"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "HF_HUB_DISABLE_SYMLINKS_WARNING=1"

echo.
echo   ==============================================
echo     Embrify - AI Email Summarizer
echo   ==============================================
echo.

rem ---- already running? then just open it
powershell -NoProfile -Command "try { Invoke-WebRequest -UseBasicParsing '%URL%/api/info' -TimeoutSec 2 | Out-Null; exit 0 } catch { exit 1 }" >nul 2>&1
if %errorlevel%==0 (
    echo [ok] Embrify is already running - opening it in your browser.
    start "" "%URL%"
    timeout /t 3 >nul
    exit /b 0
)

if not exist "app.py" (
    echo [!!] app.py not found. Keep this file inside the Ai_email_summarizer folder.
    goto :fail
)

rem ---- 1. Python 3.10 - 3.12 (the voice package Kokoro does not support 3.13 yet)
if exist "%VPY%" goto :have_venv
call :find_python
if defined PY goto :make_venv

echo [..] Python 3.10-3.12 was not found. Installing Python 3.12 for this user...
where winget >nul 2>&1
if errorlevel 1 goto :no_python
winget install -e --id Python.Python.3.12 --scope user --accept-package-agreements --accept-source-agreements
call :find_python
if not defined PY if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" set "PY=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
if not defined PY goto :no_python

:make_venv
echo [..] Creating the app environment in %VENV%
if not exist "%ENVDIR%" mkdir "%ENVDIR%"
"%PY%" -m venv "%VENV%"
if errorlevel 1 (
    echo [!!] Could not create the Python environment.
    goto :fail
)

:have_venv
rem ---- 2. packages: installed on the first run, and again only when requirements.txt changes
fc /b "requirements.txt" "%ENVDIR%\installed-requirements.txt" >nul 2>&1
if not errorlevel 1 goto :models

echo [..] Installing packages - first run only, this can take 5-15 minutes...
"%VPY%" -m pip install --upgrade pip --quiet
if errorlevel 1 goto :pip_failed

where nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [..] No NVIDIA GPU found - installing PyTorch for the CPU. The app works, but more slowly.
    "%VPY%" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cpu
) else (
    echo [..] NVIDIA GPU found - installing PyTorch with CUDA support, about 2.5 GB.
    "%VPY%" -m pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu124
)
if errorlevel 1 goto :pip_failed

"%VPY%" -m pip install -r requirements.txt
if errorlevel 1 goto :pip_failed
copy /y "requirements.txt" "%ENVDIR%\installed-requirements.txt" >nul
echo [ok] Packages installed.

:models
rem ---- 3. models: download whatever is missing
"%VPY%" setup_models.py
if errorlevel 1 (
    echo [!!] The original model could not be downloaded. Check the internet connection and try again.
    goto :fail
)

rem ---- 4. start the app; it opens the browser by itself
echo.
echo [ok] Starting Embrify at %URL%
echo      The browser opens automatically. Keep this window open - closing it stops the app.
echo.
"%VPY%" app.py --port %PORT%
echo.
echo [--] Embrify has stopped.
pause
exit /b 0

rem =====================================================================================
:find_python
set "PY="
for %%V in (3.12 3.11 3.10) do (
    if not defined PY (
        for /f "delims=" %%P in ('py -%%V -c "import sys; print(sys.executable)" 2^>nul') do set "PY=%%P"
    )
)
if defined PY exit /b 0
for /f "delims=" %%P in ('python -c "import sys; assert (3, 10) <= sys.version_info[:2] <= (3, 12); print(sys.executable)" 2^>nul') do set "PY=%%P"
exit /b 0

:no_python
echo [!!] Could not find or install Python 3.10-3.12.
echo      Install Python 3.12 from https://www.python.org/downloads/ - tick "Add python.exe to PATH" -
echo      then run Start Embrify.bat again.
goto :fail

:pip_failed
echo [!!] Installing packages failed. Check the internet connection and run Start Embrify.bat again.
echo      If it keeps failing, delete the folder %ENVDIR% and try once more.
goto :fail

:fail
echo.
pause
exit /b 1
