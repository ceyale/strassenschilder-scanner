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

  // Kleines MNIST-CNN; die 25 KB Gewichte werden erst bei einem Tempolimit geladen.
  let digitSessionPromise = null, characterSessionPromise = null, digitQueue = Promise.resolve();
  function getDigitSession() {
    if (!window.ort) return Promise.reject(new Error('ONNX Runtime Web nicht verfügbar'));
    if (!digitSessionPromise) {
      window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
      digitSessionPromise = window.ort.InferenceSession.create('models/mnist-12.onnx', {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all'
      }).catch(error => {
        digitSessionPromise = null;
        throw error;
      });
    }
    return digitSessionPromise;
  }
  function getCharacterSession() {
    if (!window.ort) return Promise.reject(new Error('ONNX Runtime Web nicht verfügbar'));
    if (!characterSessionPromise) {
      window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
      characterSessionPromise = window.ort.InferenceSession.create('models/emnist-alphanumeric.onnx', {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all'
      }).catch(error => {
        characterSessionPromise = null;
        throw error;
      });
    }
    return characterSessionPromise;
  }
  let textWorkerPromise = null, textQueue = Promise.resolve();
  function getTextWorker() {
    if (!window.Tesseract) return Promise.reject(new Error('Texterkennung nicht verfügbar'));
    if (!textWorkerPromise) textWorkerPromise = window.Tesseract.createWorker('deu').catch(error => {
      textWorkerPromise = null;
      throw error;
    });
    return textWorkerPromise;
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

  function refreshStill() {
    if (!still || running) return;
    ctx.drawImage(still, 0, 0, view.width, view.height);
    draw();
  }

  /** Ziffern in der weißen Innenfläche eines runden Tempolimits mit einem CNN lesen. */
  function queueOcr(source, detections, analysisW, analysisH) {
    for (const detection of detections) {
      if (detection.ocrQueued || detection.ocrText || detection.ocrFailed) continue;
      if (detection.label === 'hinweis' || detection.label === 'ortstafel') {
        queueSignText(source, detection, analysisW, analysisH);
        continue;
      }
      if (detection.label !== 'verbot') continue;
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
      crop.width = Math.max(1, (right - x) * 2);
      crop.height = Math.max(1, (bottom - y) * 2);
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

      digitQueue = digitQueue.then(async () => {
        const session = await getDigitSession();
        const chars = segmentDigits(pixels, crop.width, crop.height);
        let digits = '';
        for (const char of chars) digits += await classifyDigit(session, char);
        detection.ocrText = digits ? digits + ' km/h' : '';
        renderList();
        refreshStill();
      }).catch(() => {
        detection.ocrText = '';
        detection.ocrFailed = true;
        renderList();
        refreshStill();
      });
    }
  }

  /** Ganze Schrift auf rechteckigen Hinweisschildern und Ortstafeln per OCR lesen. */
  function queueSignText(source, detection, analysisW, analysisH) {
    detection.ocrQueued = true;
    const sx = source.videoWidth || source.naturalWidth || source.width;
    const sy = source.videoHeight || source.naturalHeight || source.height;
    const padX = detection.w * 0.025, padY = detection.h * 0.06;
    const x = Math.max(0, Math.floor((detection.x + padX) * sx / analysisW));
    const y = Math.max(0, Math.floor((detection.y + padY) * sy / analysisH));
    const right = Math.min(sx, Math.ceil((detection.x + detection.w - padX) * sx / analysisW));
    const bottom = Math.min(sy, Math.ceil((detection.y + detection.h - padY) * sy / analysisH));
    const crop = document.createElement('canvas');
    crop.width = Math.max(1, (right - x) * 2);
    crop.height = Math.max(1, (bottom - y) * 2);
    const cropCtx = crop.getContext('2d', { willReadFrequently: true });
    cropCtx.drawImage(source, x, y, right - x, bottom - y, 0, 0, crop.width, crop.height);
    const binary = cropCtx.getImageData(0, 0, crop.width, crop.height);
    for (let i = 0; i < binary.data.length; i += 4) {
      const r = binary.data[i], g = binary.data[i + 1], b = binary.data[i + 2];
      const gray = r * 0.299 + g * 0.587 + b * 0.114;
      const foreground = detection.label === 'hinweis'
        ? Math.min(r, g, b) > 165
        : gray < 145;
      const value = foreground ? 0 : 255;
      binary.data[i] = binary.data[i + 1] = binary.data[i + 2] = value;
    }
    cropCtx.putImageData(binary, 0, 0);

    digitQueue = digitQueue.then(async () => {
      const session = await getCharacterSession();
      const rows = segmentTextCharacters(binary, crop.width, crop.height);
      const output = [];
      for (const row of rows) {
        let text = '';
        for (let i = 0; i < row.length; i++) {
          if (i && row[i].x - (row[i - 1].x + row[i - 1].w) > row[i - 1].w * 0.85) text += ' ';
          text += await classifyCharacter(session, { image: binary, width: crop.width, part: row[i] });
        }
        if (text.trim()) output.push(text.trim());
      }
      detection.cnnText = output.join(' ');
      renderList();
      refreshStill();
    }).catch(() => {
      detection.cnnFailed = true;
      renderList();
    });

    // Full-word OCR bleibt als Ergänzung für Umlaute und unklare Segmentierung.
    textQueue = textQueue.then(async () => {
      const worker = await getTextWorker();
      await worker.setParameters({ preserve_interword_spaces: '1' });
      const { data } = await worker.recognize(crop);
      detection.ocrText = (data.text || '').replace(/[^\p{L}\p{N}\s.,'’/-]/gu, ' ').replace(/\s+/g, ' ').trim();
      renderList();
      refreshStill();
    }).catch(() => {
      detection.ocrFailed = true;
      renderList();
      refreshStill();
    });
  }

  /** Einzelne Zeichen der Texttafel nach Zeilen gruppieren und horizontal sortieren. */
  function segmentTextCharacters(image, width, height) {
    const binary = new Uint8Array(width * height), seen = new Uint8Array(width * height);
    for (let p = 0, i = 0; i < binary.length; i++, p += 4) binary[i] = image.data[p] < 128 ? 1 : 0;
    const queue = new Int32Array(binary.length), parts = [];
    for (let start = 0; start < binary.length; start++) {
      if (!binary[start] || seen[start]) continue;
      let head = 0, tail = 0, x0 = width, x1 = 0, y0 = height, y1 = 0;
      queue[tail++] = start; seen[start] = 1;
      while (head < tail) {
        const p = queue[head++], x = p % width, y = (p / width) | 0;
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const next = ny * width + nx;
          if (binary[next] && !seen[next]) { seen[next] = 1; queue[tail++] = next; }
        }
      }
      const w = x1 - x0 + 1, h = y1 - y0 + 1;
      if (h >= height * 0.07 && w >= width * 0.004 && tail >= width * height * 0.00015) parts.push({ x: x0, y: y0, w, h, area: tail });
    }
    const rows = [];
    for (const part of parts.sort((a, b) => a.y - b.y || a.x - b.x)) {
      const center = part.y + part.h / 2;
      let row = rows.find(r => Math.abs(r.center - center) < Math.max(r.height, part.h) * 0.65);
      if (!row) { row = { center, height: part.h, parts: [] }; rows.push(row); }
      row.parts.push(part);
      row.center = row.parts.reduce((sum, p) => sum + p.y + p.h / 2, 0) / row.parts.length;
      row.height = Math.max(row.height, part.h);
    }
    return rows.sort((a, b) => a.center - b.center).map(row => row.parts.sort((a, b) => a.x - b.x));
  }

  /** Alphanumerische EMNIST-Zeichen auf das Eingabeformat des ONNX-CNN bringen. */
  async function classifyCharacter(session, char) {
    const { image, width, part } = char;
    const side = 76, scale = Math.min(side / part.w, side / part.h);
    const drawW = Math.max(1, Math.round(part.w * scale)), drawH = Math.max(1, Math.round(part.h * scale));
    const left = Math.round((96 - drawW) / 2), top = Math.round((96 - drawH) / 2);
    const input = new Float32Array(96 * 96).fill(1);
    for (let y = 0; y < drawH; y++) for (let x = 0; x < drawW; x++) {
      const sx = part.x + Math.min(part.w - 1, Math.floor((x + 0.5) * part.w / drawW));
      const sy = part.y + Math.min(part.h - 1, Math.floor((y + 0.5) * part.h / drawH));
      if (image.data[(sy * width + sx) * 4] < 128) input[(top + y) * 96 + left + x] = -1;
    }
    const tensor = new window.ort.Tensor('float32', input, [1, 1, 96, 96]);
    const result = await session.run({ [session.inputNames[0]]: tensor });
    const scores = result[session.outputNames[0]].data;
    let best = 0;
    for (let i = 1; i < 37; i++) if (scores[i] > scores[best]) best = i;
    if (best === 36) return '';
    return '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[best];
  }

  /** Verbundene dunkle Ziffernformen im kontrastverstärkten Ausschnitt segmentieren. */
  function segmentDigits(image, width, height) {
    const binary = new Uint8Array(width * height), seen = new Uint8Array(width * height);
    for (let p = 0, i = 0; i < binary.length; i++, p += 4) binary[i] = image.data[p] < 128 ? 1 : 0;
    const queue = new Int32Array(binary.length), parts = [];
    for (let start = 0; start < binary.length; start++) {
      if (!binary[start] || seen[start]) continue;
      let head = 0, tail = 0, x0 = width, x1 = 0, y0 = height, y1 = 0;
      queue[tail++] = start; seen[start] = 1;
      while (head < tail) {
        const p = queue[head++], x = p % width, y = (p / width) | 0;
        x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= width || ny < 0 || ny >= height) continue;
          const next = ny * width + nx;
          if (binary[next] && !seen[next]) { seen[next] = 1; queue[tail++] = next; }
        }
      }
      const w = x1 - x0 + 1, h = y1 - y0 + 1;
      const centered = x0 > width * 0.04 && x1 < width * 0.96 && y0 > height * 0.08 && y1 < height * 0.94;
      if (centered && h >= height * 0.18 && w >= width * 0.025 && tail >= width * height * 0.0015) {
        parts.push({ x: x0, y: y0, w, h, area: tail });
      }
    }
    // Small flecks are filtered above; favor the three sign glyphs if texture remains.
    const tallest = Math.max(0, ...parts.map(part => part.h));
    return parts.filter(part => part.h >= tallest * 0.55).sort((a, b) => a.x - b.x).slice(0, 3).map(part => ({ image, width, part }));
  }

  /** Ein segmentiertes Zeichen MNIST-konform auf 28×28 bringen und per ONNX-CNN klassifizieren. */
  async function classifyDigit(session, char) {
    const { image, width, part } = char;
    const scale = Math.min(20 / part.w, 20 / part.h);
    const drawW = Math.max(1, Math.round(part.w * scale)), drawH = Math.max(1, Math.round(part.h * scale));
    const left = Math.round((28 - drawW) / 2), top = Math.round((28 - drawH) / 2);
    const glyph = new Float32Array(28 * 28);
    for (let y = 0; y < drawH; y++) for (let x = 0; x < drawW; x++) {
      const sx = part.x + Math.min(part.w - 1, Math.floor((x + 0.5) * part.w / drawW));
      const sy = part.y + Math.min(part.h - 1, Math.floor((y + 0.5) * part.h / drawH));
      if (image.data[(sy * width + sx) * 4] < 128) glyph[(top + y) * 28 + left + x] = 1;
    }
    // MNIST-Zeichen anhand des Tinten-Schwerpunkts, nicht nur am Rahmen zentrieren.
    let mass = 0, sumX = 0, sumY = 0;
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const ink = glyph[y * 28 + x]; mass += ink; sumX += x * ink; sumY += y * ink;
    }
    const shiftX = mass ? Math.round(13.5 - sumX / mass) : 0;
    const shiftY = mass ? Math.round(13.5 - sumY / mass) : 0;
    const input = new Float32Array(28 * 28);
    for (let y = 0; y < 28; y++) for (let x = 0; x < 28; x++) {
      const dx = x + shiftX, dy = y + shiftY;
      if (dx >= 0 && dx < 28 && dy >= 0 && dy < 28) input[dy * 28 + dx] = glyph[y * 28 + x];
    }
    const tensor = new window.ort.Tensor('float32', input, [1, 1, 28, 28]);
    const result = await session.run({ [session.inputNames[0]]: tensor });
    const scores = result[session.outputNames[0]].data;
    let best = 0;
    for (let i = 1; i < 10; i++) if (scores[i] > scores[best]) best = i;
    return String(best);
  }

  /** Liste unter dem Bild – nur neu aufbauen, wenn sich die Schildtypen ändern. */
  function renderList() {
    const labels = [...new Set(current.map(t => t.label))];
    const key = labels.map(l => l + ':' + current.filter(t => t.label === l)
      .map(t => [t.ocrText || '', t.cnnText || '', t.state || '', t.ocrFailed || '', t.cnnFailed || ''].join('/')).join('|')).join();
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
      } else if (current.some(t => t.label === label && t.ocrFailed)) {
        const failed = document.createElement('p');
        failed.textContent = 'Texterkennung konnte nicht geladen werden.';
        li.append(failed);
      }
      const cnnTexts = [...new Set(current.filter(t => t.label === label).map(t => t.cnnText).filter(Boolean))];
      if (cnnTexts.length) {
        const cnn = document.createElement('p');
        cnn.className = 'cnn-text';
        cnn.textContent = 'CNN-Zeichen: ' + cnnTexts.join(' · ');
        li.append(cnn);
      } else if (current.some(t => t.label === label && t.cnnFailed)) {
        const failed = document.createElement('p');
        failed.textContent = 'Zeichen-CNN konnte nicht geladen werden.';
        li.append(failed);
      }
      const lightStates = [...new Set(current.filter(t => t.label === label).map(t => t.state).filter(Boolean))];
      if (lightStates.length) {
        const state = document.createElement('p');
        state.textContent = 'Hellstes Lichtfeld: ' + lightStates.join(', ');
        li.append(state);
      }
      list.append(li);
    }
  }

  /** Debug: erkannte Farbflächen halbtransparent einblenden. */
  function drawMask() {
    const m = maskCv.getContext('2d'), img = m.createImageData(maskCv.width, maskCv.height);
    const pal = [null, [211, 35, 47], [20, 103, 184], [242, 194, 0], [35, 185, 86]];
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
      const txt = (t.ocrText ? t.ocrText + ' · ' : '') + (t.state ? t.state + ' · ' : '') + s.name + ' ' + Math.round(t.conf * 100) + ' %', tw = Math.min(view.width - x, ctx.measureText(txt).width + 10);
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
    img.onerror = () => { statusEl.textContent = 'Das Bild konnte nicht geöffnet werden.'; URL.revokeObjectURL(img.src); };
    img.src = URL.createObjectURL(f);
  });
  ['sat', 'maskCb'].forEach(id => $(id).addEventListener('input', () => {
    $('satOut').textContent = (+$('sat').value).toFixed(2);
    if (still) showStill();
  }));
  renderList();
})();
