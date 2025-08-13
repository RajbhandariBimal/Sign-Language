@echo off
echo ============================================
echo ASL Alphabet Recognition Setup Script
echo ============================================
echo.

REM Check if Python is installed
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python is not installed or not in PATH
    echo Please install Python 3.8 or higher from https://python.org
    pause
    exit /b 1
)

echo Python detected. Installing requirements...
echo.

REM Install Python requirements
pip install -r requirements.txt
if errorlevel 1 (
    echo.
    echo ERROR: Failed to install requirements
    echo Please check your internet connection and try again
    pause
    exit /b 1
)

echo.
echo ============================================
echo Installation completed successfully!
echo ============================================
echo.
echo Next steps:
echo 1. Run: python main.py (for interactive menu)
echo 2. Or run: python setup_dataset.py (to download dataset)
echo 3. Then run: python train_model.py (to train the model)
echo 4. Finally run: python detect_asl.py (for real-time detection)
echo.
echo For enhanced detection with hand tracking:
echo pip install mediapipe
echo python detect_asl_enhanced.py
echo.
pause
