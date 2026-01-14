# Online-Deployment Anleitung

## Streamlit Cloud (Empfohlen - Kostenlos!)

### Vorteile:
- ✅ **100% Kostenlos** für öffentliche Apps
- ✅ **Sehr einfach** - nur wenige Klicks
- ✅ **Automatische Updates** bei Code-Änderungen
- ✅ **Kein Server-Management** nötig

### Schritt-für-Schritt Anleitung:

#### 1. GitHub Repository erstellen

1. Gehe zu [github.com](https://github.com) und erstelle einen Account (falls noch nicht vorhanden)
2. Klicke auf "New Repository"
3. Name: z.B. `osp-anlaufanalyse`
4. Wähle "Public" (für kostenloses Hosting)
5. Klicke "Create repository"

#### 2. Code auf GitHub hochladen

**Option A: GitHub Desktop (einfach)**
1. Lade [GitHub Desktop](https://desktop.github.com/) herunter
2. Öffne GitHub Desktop und melde dich an
3. "Add Local Repository" → Wähle den `OSP_New` Ordner
4. "Publish repository" klicken

**Option B: Kommandozeile**
```bash
cd /Pfad/zu/OSP_New
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/DEIN-USERNAME/osp-anlaufanalyse.git
git push -u origin main
```

**WICHTIG:** Die `venv/` Ordner sollte NICHT hochgeladen werden!  
Erstelle eine `.gitignore` Datei mit folgendem Inhalt:
```
venv/
__pycache__/
*.pyc
.DS_Store
*.log
```

#### 3. Streamlit Cloud einrichten

1. Gehe zu [share.streamlit.io](https://share.streamlit.io)
2. Klicke "Sign in with GitHub"
3. Klicke "New app"
4. Wähle dein Repository: `osp-anlaufanalyse`
5. Branch: `main`
6. Main file path: `streamlit_dashboard.py`
7. Klicke "Deploy!"

#### 4. Fertig! 🎉

Nach ca. 2-3 Minuten ist deine App online unter:
```
https://DEIN-USERNAME-osp-anlaufanalyse.streamlit.app
```

Diesen Link kannst du an deinen Kunden weitergeben!

---

## Alternative: Render.com (Auch kostenlos)

### Anleitung:

1. Gehe zu [render.com](https://render.com) und erstelle einen Account
2. Klicke "New +" → "Web Service"
3. Verbinde dein GitHub Repository
4. Einstellungen:
   - **Name:** osp-anlaufanalyse
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `streamlit run streamlit_dashboard.py --server.port=$PORT --server.address=0.0.0.0`
5. Klicke "Create Web Service"

---

## Lokales Testen vor Deployment

Bevor du online gehst, teste die Streamlit-App lokal:

```bash
cd OSP_New
source venv/bin/activate  # Mac/Linux
# oder: venv\Scripts\activate  # Windows

pip install streamlit
streamlit run streamlit_dashboard.py
```

Die App öffnet sich automatisch im Browser unter `http://localhost:8501`

---

## Troubleshooting

### Problem: "ModuleNotFoundError"
**Lösung:** Stelle sicher, dass alle Pakete in `requirements.txt` aufgelistet sind.

### Problem: "FileNotFoundError: Input files"
**Lösung:** Die `Input files/` Ordner müssen im GitHub Repository enthalten sein!

### Problem: App lädt sehr langsam
**Lösung:** 
- Streamlit Cloud hat begrenzte Ressourcen im kostenlosen Plan
- Erwäge, nur eine Auswahl der Dateien hochzuladen (z.B. nur Drei M/W)
- Oder nutze Caching (`@st.cache_data`)

### Problem: App schläft nach Inaktivität
**Lösung:** Das ist normal im kostenlosen Plan. Beim ersten Aufruf dauert es 10-20 Sekunden, dann läuft sie wieder.

---

## Neue Features

### ✅ Datei-Upload für Kunden
- Kunden können ihre eigenen `.dat` Dateien hochladen
- Analyse mit allen trainierten Modellen (Kalman, SSA, Neural Network)
- Gleiche Qualitätsmetriken wie bei lokalen Dateien

### ✅ Passwortschutz
- Integrierter Passwortschutz (Standard: `OSP2024`)
- Konfigurierbar über Streamlit Secrets oder Umgebungsvariablen

### ✅ Beispiel-Dateien
- Alle Beispiel-Dateien (Yamal, Osazee, etc.) sind als Demo verfügbar
- Kunden können zwischen Beispielen und eigenen Uploads wechseln

## Datenschutz & Sicherheit

**Wichtig für sensible Daten:**
- Streamlit Cloud Public Apps sind **öffentlich zugänglich**
- **Passwortschutz ist bereits implementiert!** (Standard: `OSP2024`)
- Wenn die Daten sehr vertraulich sind, nutze:
  - **Private Repository** (kostet bei Streamlit Cloud)
  - **Eigenes Passwort** setzen (siehe oben)
  - **Eigenen Server** (z.B. Render.com mit Authentication)

**Passwortschutz ist bereits implementiert!**

Das Dashboard hat bereits einen integrierten Passwortschutz. Standard-Passwort: `OSP2024`

**Passwort ändern:**

**Option 1: Streamlit Secrets (Empfohlen für Streamlit Cloud)**
1. In Streamlit Cloud: "Manage app" → "Secrets"
2. Füge hinzu:
```toml
OSP_DASHBOARD_PASSWORD = "DEIN_PASSWORT"
```

**Option 2: Umgebungsvariable (für lokale/Server-Deployments)**
```bash
export OSP_DASHBOARD_PASSWORD="DEIN_PASSWORT"
```

**Option 3: Lokale secrets.toml (für lokales Testen)**
Erstelle `.streamlit/secrets.toml`:
```toml
OSP_DASHBOARD_PASSWORD = "DEIN_PASSWORT"
```

---

## Kosten-Übersicht

| Plattform | Kostenlos | Limits | Bezahlt |
|-----------|-----------|--------|---------|
| **Streamlit Cloud** | ✅ Ja | 1 GB RAM, Public | $20/Monat (Private) |
| **Render.com** | ✅ Ja | 512 MB RAM, schläft nach 15min | $7/Monat (immer aktiv) |
| **Heroku** | ❌ Nein | - | Ab $7/Monat |
| **AWS/Azure** | ⚠️ Free Tier | Komplex | Ab $10/Monat |

**Empfehlung:** Starte mit Streamlit Cloud (kostenlos), upgrade nur wenn nötig.

---

## Support & Updates

**Automatische Updates:**
- Jedes Mal, wenn du Code auf GitHub pushst, wird die App automatisch neu deployed
- Dauert ca. 2-3 Minuten

**Logs ansehen:**
- In Streamlit Cloud: Klicke auf "Manage app" → "Logs"
- Hier siehst du Fehler und Debug-Informationen

**App neu starten:**
- In Streamlit Cloud: Klicke auf "Manage app" → "Reboot"

---

## Kontakt & Hilfe

- **Streamlit Docs:** https://docs.streamlit.io
- **Streamlit Forum:** https://discuss.streamlit.io
- **GitHub Issues:** Erstelle Issues in deinem Repository

---

**Viel Erfolg mit dem Deployment! 🚀**

