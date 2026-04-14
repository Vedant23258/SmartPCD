@echo off
set PYTHON_EXE=%~dp0python-runtime\python.exe

if not exist "%PYTHON_EXE%" (
    echo Portable Python runtime not found.
    pause
    exit /b 1
)

echo ================================================
echo Smart Pothole and Crack Detection Project
echo ================================================
echo 1. Train Model
echo 2. Predict Single Image
set /p choice=Enter your choice (1 or 2): 

if "%choice%"=="1" (
    "%PYTHON_EXE%" src\main.py train
    goto end
)

if "%choice%"=="2" (
    set /p imgpath=Enter image path: 
    "%PYTHON_EXE%" src\main.py predict --image "%imgpath%"
    goto end
)

echo Invalid choice.

:end
pause
