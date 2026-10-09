# Schilder-Scanner

Erkennt einfache deutsche Verkehrszeichen live mit der Handykamera – direkt im Browser, ohne Server und ohne Installation. Laufzeitbibliotheken und Modelle werden bei Bedarf geladen.

Erkannte Schilder müssen zu Farbe und Form auch eine passende Innenfläche haben. In der Kamera werden Treffer erst nach mehreren aufeinanderfolgenden Bildern angezeigt, um Fehlalarme zu verringern. Ein Stopp-Schild wird erst eingeblendet, wenn das Verkehrszeichen-CNN es bestätigt. Für Tempolimits liefert dasselbe CNN direkt die Zahl aus seiner GTSRB-Klasse; es gibt keine nachgeschaltete Ziffernsegmentierung oder OCR-Prüfung. Für Texte auf blauen Hinweisschildern und Ortstafeln liest zusätzlich ein alphanumerisches CNN einzelne Zeichen; deutsche OCR ergänzt ganze Wörter und Umlaute. Vertikale Reihen aus roten, gelben und grünen Lichtfeldern werden als Ampeln angezeigt. Das Bild wird nicht hochgeladen. Beim ersten Start ist Internet nötig, um ONNX Runtime Web, Tesseract.js und ihre Modelle/WASM-Dateien zu laden.

## Schnellstart

**Auf dem Handy (GitHub Pages):**
1. Alle Dateien in ein GitHub-Repository hochladen (Ordnerstruktur beibehalten).
2. *Settings → Pages → Deploy from a branch → `main` / `(root)`* wählen.
3. Die Seite `https://<name>.github.io/<repo>/` am Handy öffnen, „Kamera starten“ tippen und Zugriff erlauben.

**Auf dem PC zum Testen:**
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
| `src/app.js` | Kamera, Foto-Import, Anzeige |
| `src/style.css` | Gestaltung |
| `tests/detector.test.js` | Test mit synthetisch gezeichneten Schildern (`node tests/detector.test.js`) |
| `docs/DOKUMENTATION.md` | Ausführliche Doku: Algorithmus, Schwellwerte, Grenzen, Erweiterungen |

Lizenz: MIT.
