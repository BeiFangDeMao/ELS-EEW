@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if defined SCALING_PYTHON goto launch
set "SCALING_PYTHON=%USERPROFILE%\.conda\envs\lightweight\python.exe"
if exist "%SCALING_PYTHON%" goto launch
echo lightweight Python was not found. Set SCALING_PYTHON to your python.exe.
pause
exit /b 1
:launch
echo Starting full sequential scaling experiment. This can take many hours.
"%SCALING_PYTHON%" -u "%~dp0run_scaling_experiment.py"
set "study_exit=%ERRORLEVEL%"
echo.
echo Experiment exit code: %study_exit%
echo Logs and results: discussion\scaling_study\results
pause
exit /b %study_exit%
