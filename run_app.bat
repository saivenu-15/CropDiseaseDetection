@echo off
cd /d "%~dp0"
echo Activating virtual environment...
call venv\Scripts\activate
echo Running Streamlit App...
python -m streamlit run app.py
pause
