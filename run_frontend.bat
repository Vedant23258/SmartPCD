@echo off
set PYTHON_EXE=%~dp0python-runtime\python.exe

if not exist "%PYTHON_EXE%" (
    echo Portable Python runtime not found.
    pause
    exit /b 1
)

cd /d "%~dp0"
"%PYTHON_EXE%" -m streamlit run app.py --server.address 0.0.0.0 --server.port 8501
pause
