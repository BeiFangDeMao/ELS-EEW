@echo off
setlocal
chcp 65001 >nul
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
cd /d "%~dp0"
if not defined COMPARISON_PYTHON set "COMPARISON_PYTHON=%USERPROFILE%\.conda\envs\lightweight\python.exe"
"%COMPARISON_PYTHON%" train_comparison.py --check
pause
