@echo off
setlocal
chcp 65001 >nul

cd /d "%~dp0"

echo [1/3] Python 확인 중...
python --version
if errorlevel 1 (
    echo.
    echo Python을 찾지 못했습니다.
    echo Python 3.11을 설치한 뒤 다시 실행하세요.
    goto :fail
)

echo.
echo [2/3] 필요한 패키지 설치 중...
pip install -r requirements.txt
if errorlevel 1 goto :fail

echo.
echo [3/3] MediaPipe 모델 다운로드 확인 중...
python download_mediapipe_models.py
if errorlevel 1 goto :fail

echo.
echo 설정이 완료되었습니다.
echo 이제 word_model 폴더로 이동해서 record_dataset.py를 실행하면 됩니다.
echo 예시:
echo   cd word_model
echo   python record_dataset.py --person-index 1
pause
exit /b 0

:fail
echo.
echo 설정 중 문제가 발생했습니다.
echo 위 메시지를 확인한 뒤 다시 실행해주세요.
pause
exit /b 1
