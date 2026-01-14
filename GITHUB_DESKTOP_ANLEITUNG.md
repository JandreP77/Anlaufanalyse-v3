# 🚀 GitHub Desktop Anleitung

## Schritt 1: GitHub Desktop installieren

1. Gehe zu: https://desktop.github.com/
2. Klicke **"Download for macOS"**
3. Installiere die App
4. Öffne GitHub Desktop

---

## Schritt 2: Repository hinzufügen

### Option A: Neues Repository erstellen

1. In GitHub Desktop: **File** → **New Repository**
2. **Name:** `osp-anlaufanalyse-v2`
3. **Local Path:** `/Users/andreparduhn/Documents/OSP_New`
4. **GitHub Account:** Mit deinem Account anmelden
5. ✅ **"Initialize this repository with a README"** NICHT ankreuzen
6. Klicke **"Create Repository"**

### Option B: Existierendes Repository verbinden

1. In GitHub Desktop: **File** → **Add Local Repository**
2. Klicke **"Choose..."** und wähle: `/Users/andreparduhn/Documents/OSP_New`
3. Falls gefragt: **"This directory appears to be a Git repository"** → **"Add"**
4. Falls gefragt: **"Publish repository"** → Repository-Name: `osp-anlaufanalyse-v2`

---

## Schritt 3: Dateien prüfen

Im GitHub Desktop solltest du sehen:
- ✅ Alle Python-Dateien (`.py`)
- ✅ Alle Modell-Dateien (`.pt`, `.pkl`)
- ✅ `requirements.txt`
- ✅ `.gitignore`
- ✅ `osp_logo.png`
- ✅ `Input files/` Ordner

**NICHT sichtbar sein sollten:**
- ❌ `venv/` (wird von `.gitignore` ausgeschlossen)
- ❌ `*_BACKUP.py` (falls in `.gitignore`)
- ❌ `__pycache__/`

---

## Schritt 4: Commit und Push

1. Im GitHub Desktop siehst du alle geänderten/neuen Dateien
2. Unten links: **Summary:** z.B. "Alle Dateien für Deployment"
3. Klicke **"Commit to main"**
4. Klicke **"Push origin"** (oben rechts)
5. Warte bis "Pushed" erscheint ✅

---

## Schritt 5: Auf GitHub prüfen

Gehe zu: https://github.com/JandreP77/-osp-anlaufanalyse-v2

Du solltest jetzt ALLE Dateien sehen! 🎉

---

## Falls Probleme auftreten

**"Repository not found":**
- Stelle sicher, dass das Repository auf GitHub existiert: https://github.com/JandreP77/-osp-anlaufanalyse-v2

**"Authentication failed":**
- In GitHub Desktop: **GitHub Desktop** → **Preferences** → **Accounts**
- Stelle sicher, dass du eingeloggt bist

**Dateien fehlen:**
- Prüfe `.gitignore` - vielleicht werden Dateien ausgeschlossen
- Füge sie manuell hinzu: Rechtsklick → **"Add to Git"**

---

## Nach erfolgreichem Push

Weiter mit **Streamlit Cloud Deployment** (siehe `DEPLOYMENT.md`)

