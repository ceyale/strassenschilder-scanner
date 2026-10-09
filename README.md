# Schilder-Scanner

Erkennt einfache deutsche Verkehrszeichen live mit der Handykamera im Browser. Bestätigte Schilderinformationen und ein stark komprimiertes JPEG-Bild (maximal 480 Pixel breit) werden an das PC-Backend gesendet. Bei laufender Kamera wird bis zu viermal pro Sekunde ein aktualisiertes Bild übertragen. Laufzeitbibliotheken und Modelle werden bei Bedarf geladen.

Erkannte Schilder werden im Browser gefunden. Die mittleren Erkennungsschwellen balancieren Erkennungsrate und Fehlalarme: Sättigung startet bei 0,38, das CNN verlangt mindestens 0,22 Confidence und 0,05 Abstand zur nächstbesten Klasse. Ein Stopp-Schild wird vom Verkehrszeichen-CNN klassifiziert. Für Tempolimits liefert dasselbe CNN direkt die Zahl aus seiner GTSRB-Klasse. Für Texte auf blauen Hinweisschildern und Ortstafeln liest zusätzlich ein alphanumerisches CNN einzelne Zeichen; deutsche OCR ergänzt ganze Wörter und Umlaute. Vertikale Reihen aus roten, gelben und grünen Lichtfeldern werden als Ampeln angezeigt. Wenn ein Backend konfiguriert ist, werden Schilddaten und ein komprimiertes JPEG mit Erkennungsrahmen zusätzlich dorthin gesendet. Beim ersten Start ist Internet nötig, um ONNX Runtime Web, Tesseract.js und ihre Modelle/WASM-Dateien zu laden.

Das Verkehrszeichen-CNN verlangt mindestens `confidence` 0,22 und Klassenabstand 0,05. Die vorgeschaltete Formsuche akzeptiert Kandidaten ab Formvertrauen 0,38.

## Schnellstart

**Lokaler PC mit Backend (empfohlen):**
1. Python 3 installieren.
2. Im Projektordner `python server.py` starten.
3. In einem zweiten Terminal `python gui.py` starten, um die offene Python-Anzeige zu öffnen.
4. `http://localhost:8765` öffnen. Die Website sendet bestätigte Erkennungen an den PC; unter `http://localhost:8765/dashboard` werden sie ebenfalls angezeigt.

**Auf dem Handy (GitHub Pages):**
1. Alle Dateien in ein GitHub-Repository hochladen (Ordnerstruktur beibehalten).
2. *Settings → Pages → Deploy from a branch → `main` / `(root)`* wählen.
3. Die Seite `https://<name>.github.io/<repo>/` am Handy öffnen, unter „PC-Backend“ die PC-Adresse eintragen und „Kamera starten“ tippen.

Beim Zugriff über GitHub Pages benötigt die PC-Adresse HTTPS, damit ein HTTPS-Browser sie kontaktieren darf; das lokale Backend stellt standardmäßig nur HTTP bereit. Für einen einfachen Start nutze Website und Backend gemeinsam unter `http://localhost:8765` am PC. Für Handyzugriff braucht der PC-Backendserver zusätzlich eine HTTPS-Konfiguration und eine erreichbare Netzwerkadresse.

**Temporäre öffentliche URL mit Cloudflare Quick Tunnel:**
1. Starte `python server.py` auf dem PC.
2. In einem zweiten Terminal `npx wrangler tunnel quick-start http://localhost:8765` ausführen (oder `cloudflared tunnel --url http://localhost:8765`, falls installiert).
3. Die ausgegebene `https://….trycloudflare.com`-Adresse öffnen und teilen. Website, Dashboard und Backend laufen über dieselbe HTTPS-Adresse.

Die zufällige Adresse ist öffentlich erreichbar, solange der Tunnel läuft. Wer die URL kennt, kann die Website und den Empfangs-Endpunkt erreichen. Beim Stoppen des Tunnelprozesses wird die URL ungültig; beim nächsten Start wird eine neue Adresse vergeben. Quick Tunnels sind für Entwicklung und Tests gedacht.

**Nur statische Website ohne Backend:**
```
python -m http.server 8000
```
Dann `http://localhost:8000` öffnen. `localhost` gilt als sicher, die Webcam funktioniert. Über die WLAN-IP am Handy geht die Kamera **nicht**, dafür braucht der Browser HTTPS.

**Ohne Kamera:** „Foto wählen“ analysiert ein einzelnes Bild.

## Projektstruktur

| Datei | Inhalt |
|---|---|
| `index.html` | Seitengerüst |
| `src/detector.js` | Erkennung (ohne Browser-Abhängigkeit, auch in Node nutzbar) |
| `src/app.js` | Kamera, Foto-Import, Anzeige und Backend-Übertragung |
| `server.py` | Lokaler Website- und API-Server auf Port 8765 |
| `gui.py` | Offenes Python-Fenster mit aktuellem Empfang, Protokoll und JPEG-Vorschau (benötigt Pillow) |
| `backend/dashboard.html` | Live-Ansicht der zuletzt empfangenen Schilder |
| `backend/latest-frame.jpg` | Letztes komprimiertes Kamerabild (wird beim Empfang überschrieben) |
| `src/style.css` | Gestaltung |
| `tests/detector.test.js` | Test mit synthetisch gezeichneten Schildern (`node tests/detector.test.js`) |
| `docs/DOKUMENTATION.md` | Ausführliche Doku: Algorithmus, Schwellwerte, Grenzen, Erweiterungen |

Lizenz: MIT.
