#!/bin/bash
# OSP Anlaufanalyse — Startscript (Mac/Linux)

set -e
cd "$(dirname "$0")"

echo "=== OSP Anlaufanalyse Dashboard ==="
echo ""

# Virtuelle Umgebung erstellen oder aktualisieren
if [ ! -d "venv" ]; then
    echo "Erstelle virtuelle Umgebung..."
    python3 -m venv venv
fi

source venv/bin/activate

echo "Installiere/aktualisiere Abhängigkeiten..."
pip install --quiet --upgrade pip
pip install --quiet -r requirements.txt

echo ""
echo "Dashboard startet — Browser öffnet sich automatisch..."
echo "URL: http://localhost:8501"
echo "Passwort: OSP2024"
echo ""
echo "Zum Beenden: Strg+C"
echo ""

streamlit run streamlit_dashboard.py \
    --server.headless false \
    --server.port 8501 \
    --browser.gatherUsageStats false
