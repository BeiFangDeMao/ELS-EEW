@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if defined ABLATION_PYTHON goto launch
set "ABLATION_PYTHON=%USERPROFILE%\.conda\envs\lightweight\python.exe"
if exist "%ABLATION_PYTHON%" goto launch
echo Set ABLATION_PYTHON to your lightweight python.exe.
pause
exit /b 1
:launch
"%ABLATION_PYTHON%" -u "%~dp0run_ablation.py"
set "ablation_exit=%ERRORLEVEL%"
echo.
echo Ablation exit code: %ablation_exit%
pause
exit /b %ablation_exit%
