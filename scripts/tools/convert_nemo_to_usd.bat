@echo off
rem Convert legged_rl NEMO MJCF (nemo5.xml + assets/*.stl) to nemo.usd for isaaclab_assets.
rem Requires Isaac Lab's Isaac Sim python (isaaclab.bat -p).
rem
rem If headless MJCF import crashes, run WITHOUT --headless and use --exit-after-convert
rem (see scripts/tools/convert_mjcf.py). Prefer short paths on Windows (e.g. C:\temp\nemo_mjcf).

setlocal
set "ISAACLAB_ROOT=%~dp0..\.."
set "NEMO_MJCF=%ISAACLAB_ROOT%\legged_rl-main\legged_rl-main\models\nemo\nemo5.xml"
set "OUT_USD=%ISAACLAB_ROOT%\source\isaaclab_assets\data\Robots\Nemo\nemo.usd"

if not exist "%NEMO_MJCF%" (
  echo [ERROR] MJCF not found: "%NEMO_MJCF%"
  exit /b 1
)

call "%ISAACLAB_ROOT%\isaaclab.bat" -p "%ISAACLAB_ROOT%\scripts\tools\convert_mjcf.py" --exit-after-convert --headless "%NEMO_MJCF%" "%OUT_USD%"
exit /b %ERRORLEVEL%
