@echo off
:: OSP Anlaufanalyse — Startscript (Windows)

cd /d "%~dp0"
echo === OSP Anlaufanalyse Dashboard ===
echo.

:: Virtuelle Umgebung erstellen oder aktualisieren
if not exist "venv" (
    echo Erstelle virtuelle Umgebung...
    python -m venv venv
)

call venv\Scripts\activate.bat

echo Installiere/aktualisiere Abhaengigkeiten...
python -m pip install --quiet --upgrade pip
python -m pip install --quiet -r requirements.txt

echo.
echo Dashboard startet -- Browser oeffnet sich automatisch...
echo URL: http://localhost:8501
echo Passwort: OSP2024
echo.
echo Zum Beenden: Strg+C
echo.

streamlit run streamlit_dashboard.py --server.headless false --server.port 8501 --browser.gatherUsageStats false
pause
