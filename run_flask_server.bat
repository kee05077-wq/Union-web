@echo off
cd /d %~dp0

echo [SYSTEM] Installing required Python packages if needed...
python -m ensurepip --upgrade
python -m pip install -r requirements.txt

echo [SYSTEM] Starting Union-web Flask server...
python app.py

echo.
echo [SYSTEM] Server process ended. Press any key to close this window.
pause >nul
