# Push-Anleitung für GitHub

## Option 1: GitHub Desktop (EINFACHSTE METHODE)

1. Öffne **GitHub Desktop**
2. **File** → **Add Local Repository**
3. Wähle den Ordner: `/Users/andreparduhn/Documents/OSP_New`
4. Klicke **"Publish repository"**
5. Wähle das Repository: `-osp-anlaufanalyse-v2`
6. Fertig! ✅

---

## Option 2: Kommandozeile mit Token

### Schritt 1: Personal Access Token erstellen

1. Gehe zu: https://github.com/settings/tokens
2. Klicke **"Generate new token (classic)"**
3. Name: `OSP Deployment`
4. Scopes: ✅ **repo** (alle Unterpunkte)
5. Klicke **"Generate token"**
6. **WICHTIG:** Token sofort kopieren! (wird nur einmal angezeigt)

### Schritt 2: Push ausführen

```bash
cd /Users/andreparduhn/Documents/OSP_New
git push -u origin main
```

Wenn nach Username gefragt wird:
- **Username:** `JandreP77`
- **Password:** [Dein Token einfügen]

---

## Option 3: SSH (falls eingerichtet)

```bash
cd /Users/andreparduhn/Documents/OSP_New
git remote set-url origin git@github.com:JandreP77/-osp-anlaufanalyse-v2.git
git push -u origin main
```

---

## Nach erfolgreichem Push

Gehe zu: https://github.com/JandreP77/-osp-anlaufanalyse-v2

Du solltest jetzt alle Dateien sehen! 🎉

Dann weiter mit **Streamlit Cloud Deployment** (siehe DEPLOYMENT.md)

