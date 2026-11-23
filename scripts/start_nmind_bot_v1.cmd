@echo off
rem N-MIND Bot V1 one-click launcher (Windows).
rem Works from any working directory: resolves repo root from this script's location.
setlocal
set "SCRIPT_DIR=%~dp0"
set "REPO_ROOT=%SCRIPT_DIR%.."
set "BOT_ROOT=%REPO_ROOT%\results\bot_v1"

rem Prefer the Python that actually has nmind installed (editable install).
set "PYEXE="
if exist "C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe" (
  set "PYEXE=C:\Users\Administrator\AppData\Local\Programs\Python\Python311\python.exe"
) else (
  where py >nul 2>nul && set "PYEXE=py -3"
)
if "%PYEXE%"=="" set "PYEXE=python"

cd /d "%REPO_ROOT%"
%PYEXE% -m nmind.cli bot ui --offline --profile safe-local --root "%BOT_ROOT%"
if errorlevel 1 (
  echo.
  echo [N-MIND Bot V1] startup failed with code %errorlevel%.
  echo Check: python -m nmind.cli bot doctor
  pause
)
endlocal
