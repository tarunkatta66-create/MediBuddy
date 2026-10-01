@echo off
echo Starting MediPredict environment setup and pipeline...

python scripts/download_data.py
if errorlevel 1 exit /b %errorlevel%

python -m medipredict.train
if errorlevel 1 exit /b %errorlevel%

python scripts/generate_results_summary.py

pytest tests/
if errorlevel 1 exit /b %errorlevel%

echo Starting FastAPI Backend Server on http://127.0.0.1:8000 ...
start uvicorn medipredict.api.app:app --host 127.0.0.1 --port 8000

echo Starting Frontend Dashboard on http://localhost:3000 ...
cd frontend
npm run dev
