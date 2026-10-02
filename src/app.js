/**
 * app.js – Oberfläche: Kamera, Foto-Import, Zeichnen der Ergebnisse.
 * Die eigentliche Erkennung passiert in detector.js.
 */
(function () {
  'use strict';
  const D = SignDetector;
  const $ = id => document.getElementById(id);
  const view = $('view'), ctx = view.getContext('2d');           // sichtbares Bild + Rahmen
  const work = document.createElement('canvas');                  // kleines Bild für die Analyse
  const wctx = work.getContext('2d', { willReadFrequently: true });
  const maskCv = document.createElement('canvas');                // Farbmasken-Ansicht (Debug)
  const video = $('video'), list = $('list'), statusEl = $('status');

  const VIEW_W = 640;       // Breite der Anzeige
  const WORK_W = 320;       // Mehr Auflösung für kleine und entfernte Schilder
  const INTERVAL_MS = 70;   // bis zu rund 14 Analysen pro Sekunde

  // OCR wird erst geladen, wenn tatsächlich ein Schild erkannt wurde.
  let ocrWorkerPromise = null, ocrQueue = Promise.resolve();
  function getOcrWorker() {
    if (!window.Tesseract) return Promise.reject(new Error('OCR-Bibliothek nicht verfügbar'));
    if (!ocrWorkerPromise) {
      ocrWorkerPromise = window.Tesseract.createWorker('eng').catch(error => {
        ocrWorkerPromise = null;
        throw error;
      });
    }
    return ocrWorkerPromise;
  }

  let stream = null, running = false, still = null, lastRun = 0, lastKey = '', dims = '';
  let tracker = D.createTracker(), current = [], mask = null;

  /** Canvas-Größen an die Bildquelle anpassen (Seitenverhältnis beibehalten). */
  function setSize(sw, sh) {
    if (dims === sw + 'x' + sh) return;
    dims = sw + 'x' + sh;
    view.width = VIEW_W; view.height = Math.round(VIEW_W * sh / sw);
    work.width = maskCv.width = WORK_W; work.height = maskCv.height = Math.round(WORK_W * sh / sw);
  }

  /** Erkennung auf einer Bildquelle ausführen (Video oder Foto). */
  function analyse(source, useTracker) {
    const t0 = performance.now();
    wctx.drawImage(source, 0, 0, work.width, work.height);
    const img = wctx.getImageData(0, 0, work.width, work.height);
    const res = D.detect(img.data, work.width, work.height, { minSaturation: +$('sat').value });
    current = useTracker ? tracker.update(res.detections) : res.detections;
    mask = res.mask;
    queueOcr(source, current, work.width, work.height);
    statusEl.textContent = (running ? 'Kamera läuft' : 'Foto analysiert') + ' · ' + Math.round(performance.now() - t0) + ' ms pro Analyse';
    renderList();
  }

  /** Text innerhalb eines Schildes lesen; Worker und Erkennung laufen asynchron. */
  function queueOcr(source, detections, analysisW, analysisH) {
    for (const detection of detections) {
      if (detection.ocrQueued || detection.ocrText || detection.label !== 'verbot') continue;
      detection.ocrQueued = true;
      const sx = source.videoWidth || source.naturalWidth || source.width;
      const sy = source.videoHeight || source.naturalHeight || source.height;
      // Großzügig ausschneiden: auch dreistellige Limits reichen fast über
      // die gesamte weiße Innenfläche. Den roten Ring entfernen wir unten per Farbe.
      const insetX = detection.w * 0.08, insetY = detection.h * 0.08;
      const x = Math.max(0, Math.floor((detection.x + insetX) * sx / analysisW));
      const y = Math.max(0, Math.floor((detection.y + insetY) * sy / analysisH));
      const right = Math.min(sx, Math.ceil((detection.x + detection.w - insetX) * sx / analysisW));
      const bottom = Math.min(sy, Math.ceil((detection.y + detection.h - insetY) * sy / analysisH));
      const crop = document.createElement('canvas');
      crop.width = Math.max(1, (right - x) * 5);
      crop.height = Math.max(1, (bottom - y) * 5);
      const cctx = crop.getContext('2d', { willReadFrequently: true });
      cctx.imageSmoothingEnabled = true;
      cctx.drawImage(source, x, y, right - x, bottom - y, 0, 0, crop.width, crop.height);
      const pixels = cctx.getImageData(0, 0, crop.width, crop.height);
      for (let i = 0; i < pixels.data.length; i += 4) {
        const red = pixels.data[i], green = pixels.data[i + 1], blue = pixels.data[i + 2];
        const isRed = red > green * 1.3 && red > blue * 1.3;
        const gray = red * 0.299 + green * 0.587 + blue * 0.114;
        const value = !isRed && gray < 170 ? 0 : 255;
        pixels.data[i] = pixels.data[i + 1] = pixels.data[i + 2] = value;
      }
      cctx.putImageData(pixels, 0, 0);

      ocrQueue = ocrQueue.then(async () => {
        const worker = await getOcrWorker();
        await worker.setParameters({ tessedit_char_whitelist: '0123456789', tessedit_pageseg_mode: '7' });
        const { data } = await worker.recognize(crop);
        const digits = (data.text || '').replace(/\D/g, '');
        detection.ocrText = digits ? digits + ' km/h' : '';
        renderList();
      }).catch(() => {
        detection.ocrText = '';
        detection.ocrFailed = true;
        detection.ocrQueued = false;
        renderList();
      });
    }
  }

  /** Liste unter dem Bild – nur neu aufbauen, wenn sich die Schildtypen ändern. */
  function renderList() {
    const labels = [...new Set(current.map(t => t.label))];
    const key = labels.map(l => l + ':' + current.filter(t => t.label === l).map(t => t.ocrText || '').join('|')).join();
    if (key === lastKey) return;
    lastKey = key;
    list.replaceChildren();
    if (!labels.length) {
      const empty = document.createElement('li');
      empty.className = 'leer';
      empty.textContent = 'Noch kein Schild erkannt. Halte ein Schild ruhig und frontal ins Bild.';
      list.append(empty);
      return;
    }
    for (const label of labels) {
      const sign = D.SIGNS[label], li = document.createElement('li');
      li.style.setProperty('--c', sign.hex);
      const title = document.createElement('b'), number = document.createElement('span'), note = document.createElement('p');
      title.textContent = sign.name;
      number.textContent = sign.zeichen;
      note.textContent = sign.note;
      li.append(title, number, note);
      const texts = [...new Set(current.filter(t => t.label === label).map(t => t.ocrText).filter(Boolean))];
      if (texts.length) {
        const read = document.createElement('p');
        read.className = 'ocr-text';
        read.textContent = 'Gelesener Text: ' + texts.join(' · ');
        li.append(read);
      }
      list.append(li);
    }
  }

  /** Debug: erkannte Farbflächen halbtransparent einblenden. */
  function drawMask() {
    const m = maskCv.getContext('2d'), img = m.createImageData(maskCv.width, maskCv.height);
    const pal = [null, [211, 35, 47], [20, 103, 184], [242, 194, 0]];
    for (let i = 0; i < mask.length; i++) if (mask[i]) img.data.set([...pal[mask[i]], 255], i * 4);
    m.putImageData(img, 0, 0);
    ctx.imageSmoothingEnabled = false; ctx.globalAlpha = 0.6;
    ctx.drawImage(maskCv, 0, 0, view.width, view.height);
    ctx.globalAlpha = 1; ctx.imageSmoothingEnabled = true;
  }

  /** Rahmen + Beschriftung; Koordinaten vom Analysebild auf die Anzeige hochrechnen. */
  function drawBoxes() {
    const k = view.width / work.width;
    ctx.lineWidth = 3; ctx.textBaseline = 'top';
    ctx.font = '600 15px Bahnschrift, "DIN Alternate", system-ui, sans-serif';
    for (const t of current) {
      const s = D.SIGNS[t.label], x = t.x * k, y = t.y * k, w = t.w * k, h = t.h * k;
      const txt = (t.ocrText ? t.ocrText + ' · ' : '') + s.name + ' ' + Math.round(t.conf * 100) + ' %', tw = Math.min(view.width - x, ctx.measureText(txt).width + 10);
      const ty = y > 22 ? y - 22 : y + h + 2;
      ctx.strokeStyle = s.hex; ctx.strokeRect(x, y, w, h);
      ctx.fillStyle = s.hex; ctx.fillRect(x, ty, tw, 20);
      ctx.fillStyle = s.text; ctx.fillText(txt, x + 5, ty + 3);
    }
  }

  const draw = () => { if ($('maskCb').checked && mask) drawMask(); drawBoxes(); };

  /** Kamera-Schleife: jedes Bild anzeigen, aber nur alle INTERVAL_MS analysieren. */
  function frame(now) {
    if (!running) return;
    if (video.videoWidth) {
      setSize(video.videoWidth, video.videoHeight);
      ctx.drawImage(video, 0, 0, view.width, view.height);
      if (now - lastRun >= INTERVAL_MS) { lastRun = now; analyse(video, true); }
      draw();
    }
    requestAnimationFrame(frame);
  }

  async function startCamera() {
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
      statusEl.textContent = 'Kein Kamerazugriff. Die Seite muss über HTTPS (z. B. GitHub Pages) geöffnet werden.';
      return;
    }
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1280 } }, audio: false });
      video.srcObject = stream; await video.play();
      still = null; tracker = D.createTracker(2, 3); running = true;
      $('camBtn').textContent = 'Kamera stoppen';
      requestAnimationFrame(frame);
    } catch (err) {
      statusEl.textContent = 'Kamera nicht verfügbar (' + err.name + '). Zugriff im Browser erlauben?';
    }
  }

  function stopCamera() {
    running = false;
    if (stream) stream.getTracks().forEach(t => t.stop());
    stream = null; $('camBtn').textContent = 'Kamera starten';
  }

  /** Einzelnes Foto analysieren (ohne Tracker, jeder Treffer zählt sofort). */
  function showStill() {
    setSize(still.naturalWidth, still.naturalHeight);
    ctx.drawImage(still, 0, 0, view.width, view.height);
    analyse(still, false); draw();
  }

  $('camBtn').addEventListener('click', () => (running ? stopCamera() : startCamera()));
  $('file').addEventListener('change', e => {
    const f = e.target.files[0];
    if (!f) return;
    stopCamera();
    const img = new Image();
    img.onload = () => { still = img; lastKey = ''; showStill(); URL.revokeObjectURL(img.src); };
    img.src = URL.createObjectURL(f);
  });
  ['sat', 'maskCb'].forEach(id => $(id).addEventListener('input', () => {
    $('satOut').textContent = (+$('sat').value).toFixed(2);
    if (still) showStill();
  }));
  renderList();
})();
