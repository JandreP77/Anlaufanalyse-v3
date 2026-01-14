# 📦 Dateien für Deployment

## ✅ MUSS hochgeladen werden:

### 1. Python-Module (ALLE!)
- ✅ `streamlit_dashboard.py` - Haupt-Dashboard
- ✅ `analyze_movement_data.py` - Datenanalyse-Modul
- ✅ `kalman_ssa_interpolator.py` - Kalman/SSA Interpolation
- ✅ `neural_interpolator.py` - Neural Network Interpolation
- ✅ `data_cleaner.py` - Datenbereinigung

### 2. Modelle (WICHTIG!)
- ✅ `neural_interpolator_model.pt` - Neural Network Modell
- ✅ `global_ssa_models.pkl` - SSA Modelle
- ✅ `hybrid_ssa_models.pkl` - Hybrid Modelle

### 3. Konfiguration
- ✅ `requirements.txt` - Python-Abhängigkeiten (KRITISCH!)
- ✅ `.gitignore` - Git-Konfiguration

### 4. Assets (optional, aber empfohlen)
- ✅ `osp_logo.png` - Logo für Dashboard

### 5. Beispiel-Dateien (optional, für Demo)
- ✅ `Input files/` - Ordner mit Beispiel-Dateien (Yamal, etc.)

---

## ❌ NICHT hochladen (werden ignoriert):

- ❌ `*_BACKUP.py` - Backup-Dateien
- ❌ `venv/` - Virtuelle Umgebung
- ❌ `__pycache__/` - Python-Cache
- ❌ `.streamlit/secrets.toml` - Secrets (nur lokal!)
- ❌ `*.log` - Log-Dateien

---

## 🚀 Schnell-Checkliste:

```bash
# Diese Dateien MÜSSEN auf GitHub sein:
streamlit_dashboard.py
analyze_movement_data.py
kalman_ssa_interpolator.py
neural_interpolator.py
data_cleaner.py
neural_interpolator_model.pt
global_ssa_models.pkl
hybrid_ssa_models.pkl
requirements.txt
.gitignore
osp_logo.png
Input files/  # Ordner mit Beispiel-Dateien
```

---

## ⚠️ WICHTIG:

**Ohne diese Dateien funktioniert das Dashboard NICHT:**
- ❌ `requirements.txt` → App kann nicht starten
- ❌ `neural_interpolator_model.pt` → Neural Network funktioniert nicht
- ❌ `global_ssa_models.pkl` → SSA funktioniert nicht
- ❌ `analyze_movement_data.py` → Dashboard kann nicht starten
- ❌ `kalman_ssa_interpolator.py` → Kalman/SSA funktioniert nicht

**Lade ALLE Dateien hoch, nicht nur das Dashboard!**

