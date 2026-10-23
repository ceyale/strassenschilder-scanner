# Schilder-Scanner

Erkennt einfache deutsche Verkehrszeichen live mit der Handykamera im Browser. Bestätigte Schilderinformationen und ein stark komprimiertes JPEG-Bild (maximal 480 Pixel breit) werden an das PC-Backend gesendet. Bei laufender Kamera wird bis zu viermal pro Sekunde ein aktualisiertes Bild übertragen. Laufzeitbibliotheken und Modelle werden bei Bedarf geladen.

Im Frontend kann zwischen **Handy erkennt** und **PC erkennt · YOLO26s** umgeschaltet werden. Im PC-Modus überträgt das Handy ausschließlich das unverarbeitete Kamerabild als JPEG an `/api/frame`; Erkennung und Klassifizierung laufen im Python-Backend. Die lokale Browser-Erkennung wird in diesem Modus nicht gestartet. Die Backend-Adresse akzeptiert wie zuvor `/api/signs`; der PC-Modus leitet sie automatisch auf `/api/frame` um.

**PC-Erkennung installieren:** Im Projektverzeichnis `python -m pip install -r requirements-pc-vision.txt` ausführen und anschließend `python server.py` starten. Beim ersten Start lädt Ultralytics automatisch `yolo26s.pt` herunter, falls es lokal noch fehlt. Das ist ein allgemeines COCO-Modell: Es erkennt unter anderem Ampeln und Stoppschilder, aber keine vollständige Auswahl deutscher Verkehrszeichen. Das zusätzliche GTSRB-Klassifizierungsmodell kann nur bereits erkannte Schild-Ausschnitte einordnen. Für die vollständige Erkennung muss ein trainierter Detektor als `models/yolo26s-signs.pt` bereitgestellt werden (oder der Pfad über `SIGN_YOLO_MODEL` gesetzt werden). Ein solcher Detektor ist diesem Projekt derzeit nicht beigelegt. OCR für Ortstafeln ist optional und benötigt ein passendes PaddlePaddle-Paket; ohne OCR werden Ortstafeln nicht ausgelesen.

Erkannte Schilder werden im Browser gefunden. Die Erkennung nutzt mittlere Schwellen: Sättigung startet bei 0,40, das CNN verlangt mindestens 0,40 Confidence und 0,08 Abstand zur nächstbesten Klasse. STOP wird strenger bestätigt (Confidence 0,60, Abstand 0,15), um runde Logos weniger oft als Stoppschilder einzustufen. Der CNN-Ausschnitt enthält 12 % Rand um das Schild; sichtbare Boxen haben zusätzlich einen kleinen Außenabstand und werden stärker geglättet. Für Texte auf blauen Hinweisschildern und Ortstafeln liest zusätzlich ein alphanumerisches CNN einzelne Zeichen; deutsche OCR ergänzt ganze Wörter und Umlaute. Vertikale Reihen aus roten, gelben und grünen Lichtfeldern werden als Ampeln angezeigt. Wenn ein Backend konfiguriert ist, werden Schilddaten und ein komprimiertes JPEG mit Erkennungsrahmen zusätzlich dorthin gesendet. Beim ersten Start ist Internet nötig, um ONNX Runtime Web, Tesseract.js und ihre Modelle/WASM-Dateien zu laden.

Das Verkehrszeichen-CNN verlangt mindestens `confidence` 0,40 und Klassenabstand 0,08. STOP verlangt 0,60 und 0,15. Die vorgeschaltete Formsuche akzeptiert Kandidaten ab Formvertrauen 0,42.

## Schnellstart

Die Steuerungskette verbindet den Handy-Scanner per Internet oder lokalem WLAN mit dem Laptop-Backend. Das Backend speist die Fahrwerte in einen virtuellen Bluetooth-Classic-HID-Gamepad ein, den der EV3 wie einen PS3-Controller liest. Port A ist die Lenkung; B und C treiben gemeinsam an. TCP-Port 8766 wird nur intern zwischen Windows-Backend und der Bluetooth-Brücke in WSL verwendet; der EV3 erhält die Steuerung über Bluetooth.

**Lokaler PC mit Backend (empfohlen):**
1. Python 3 installieren.
2. Im Projektordner `python server.py` starten.
3. Optional in einem zweiten Terminal `python gui.py` starten, um Erkennungen und den aktuellen EV3-Befehl anzuzeigen.
4. `http://localhost:8765` öffnen. Die Website sendet bestätigte Erkennungen an den PC; unter `http://localhost:8765/dashboard` werden sie ebenfalls angezeigt.

**Bluetooth-Classic-Verbindung zum EV3:**
1. WSL2, Ubuntu, BlueZ und `usbipd-win` sind auf diesem Laptop eingerichtet. `start-bluetooth-gamepad.ps1` startet WSL, übergibt den eingebauten Intel-Bluetooth-Adapter an Ubuntu und startet den virtuellen Gamepad-Dienst. Beim Übergeben ist der Bluetooth-Adapter vorübergehend nicht mehr für Windows selbst verfügbar.
2. Auf dem Laptop `python server.py` starten. Es hostet die Scanner-Website und das Backend auf Port 8765. Die WSL-Brücke verbindet sich intern mit dem Befehlsstrom auf Port 8766.
3. Den EV3 einmalig per Bluetooth Classic mit **EV3 Truck Gamepad** koppeln. Das Gamepad wird von BlueZ bereitgestellt; es erscheint wie ein normaler HID-Controller.
4. `main.py` auf den EV3 kopieren und starten. Das Programm sucht das Gamepad in `/proc/bus/input/devices`; falls die EV3-Version keinen Gerätenamen meldet, nutzt es wie das Beispiel `/dev/input/event3`.
5. Das Handy öffnet die Scanner-Seite. Für eine gehostete HTTPS-Seite muss im Feld „PC-Backend“ eine erreichbare HTTPS-Adresse des PC-Backends plus `/api/signs` eingetragen sein. Das Handy und der Laptop brauchen eine Datenverbindung zum Backend; der EV3 braucht keine WLAN-Verbindung.

Ein erkanntes Tempolimit bleibt als Prozent Motorleistung (maximal 100 %) bis zum nächsten Limit aktiv. Ein neu erkanntes Stoppschild hält fünf Sekunden an. Rot und Gelb halten an; Grün gibt die Fahrt frei. Links-/Rechts-Gebotsschilder lösen einen weichen Lenkbogen aus (Startwert ±45°, 1,5 Sekunden). Ohne erkannte Ampel fährt der LKW mit dem zuletzt gesetzten Limit (Startwert 30 %). Wenn 1,5 Sekunden keine Kameradaten eintreffen, sendet das Backend neutral. Fehlen Bluetooth-Gamepad-Eingaben länger als 1,5 Sekunden, stoppt `main.py` beide Antriebsmotoren und zentriert die Lenkung. Die Gamepad- und Backend-Verbindungen bleiben dabei offen und können weiterfahren, sobald wieder gültige Befehle eintreffen.

Die Lenkung muss beim Programmstart mechanisch mittig stehen. Wenn der LKW falsch herum lenkt oder zu stark einschlägt, `STEER_LEFT_DEG`, `STEER_RIGHT_DEG` und `STEER_SPEED` in `main.py` am stehenden, angehobenen Fahrzeug anpassen. B und C erhalten stets denselben Fahrwert.

**Auf dem Handy (GitHub Pages):**
1. Alle Dateien in ein GitHub-Repository hochladen (Ordnerstruktur beibehalten).
2. *Settings → Pages → Deploy from a branch → `main` / `(root)`* wählen.
3. Die Seite `https://<name>.github.io/<repo>/` am Handy öffnen, unter „PC-Backend“ die PC-Adresse eintragen und „Kamera starten“ tippen.

Beim Zugriff über GitHub Pages benötigt die PC-Adresse HTTPS, damit ein HTTPS-Browser sie kontaktieren darf; das lokale Backend stellt standardmäßig nur HTTP bereit. Für einen einfachen Start nutze Website und Backend gemeinsam unter `http://localhost:8765` am PC. Für Handyzugriff braucht der PC-Backendserver zusätzlich eine HTTPS-Konfiguration und eine erreichbare Netzwerkadresse.

**Temporäre öffentliche URL mit Cloudflare Quick Tunnel:**
`start-bluetooth-gamepad.ps1` startet den PC-Server und den Cloudflare Quick Tunnel mit Windows-Anmeldung. Die aktuelle Adresse steht in `backend/cloudflare-url.txt` (Endpunkt `/api/signs`). Öffne alternativ die ausgegebene `https://….trycloudflare.com`-Adresse direkt am Handy; Website, Dashboard und API werden gemeinsam darüber bereitgestellt.

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
| `main.py` | EV3-Hauptprogramm: liest das Bluetooth-HID-Gamepad und steuert A/B/C mit Watchdog |
| `bluetooth_gamepad/` | BlueZ Classic-HID-Dienst und WSL-Installationsdateien für den virtuellen Controller |
| `start-bluetooth-gamepad.ps1` | WSL starten, Bluetooth-USB-Adapter anhängen und den HID-Dienst starten |
| `backend/dashboard.html` | Live-Ansicht der zuletzt empfangenen Schilder |
| `backend/latest-frame.jpg` | Letztes komprimiertes Kamerabild (wird beim Empfang überschrieben) |
| `src/style.css` | Gestaltung |
| `tests/detector.test.js` | Test mit synthetisch gezeichneten Schildern (`node tests/detector.test.js`) |
| `docs/DOKUMENTATION.md` | Ausführliche Doku: Algorithmus, Schwellwerte, Grenzen, Erweiterungen |

Lizenz: MIT.
