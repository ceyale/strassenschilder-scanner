/**
 * app.js – Oberfläche: Kamera, Foto-Import, Zeichnen der Ergebnisse.
 * Die eigentliche Erkennung passiert in detector.js.
 */
(function () {
  'use strict';
  const D = SignDetector;
  const CNN_SIGN_LABELS = new Set(['stop', 'verbot', 'einfahrtVerboten', 'vorfahrtGewaehren', 'warnung', 'gebot', 'vorfahrtstrasse']);
  const TEXT_SIGN_LABELS = new Set(['hinweis', 'ortstafel']);
  const GTSRB_SPEED_CLASSES = new Map([[0, 20], [1, 30], [2, 50], [3, 60], [4, 70], [5, 80], [7, 100], [8, 120]]);
  const GTSRB_WARNING_CLASSES = new Set([11, ...Array.from({ length: 14 }, (_, i) => i + 18)]);
  // Lower thresholds accept more uncertain CNN predictions (and may add false positives).
  const MIN_SIGN_CONFIDENCE = 0.4;
  const MIN_SIGN_MARGIN = 0.08;
  const $ = id => document.getElementById(id);
  const view = $('view'), ctx = view.getContext('2d');           // sichtbares Bild + Rahmen
  const backendUrlInput = $('backendUrl'), backendStatus = $('backendStatus');
  const work = document.createElement('canvas');                  // kleines Bild für die Analyse
  const wctx = work.getContext('2d', { willReadFrequently: true });
  const maskCv = document.createElement('canvas');                // Farbmasken-Ansicht (Debug)
  const video = $('video'), list = $('list'), statusEl = $('status');

  const VIEW_W = 640;       // Breite der Anzeige
  const WORK_W = 640;       // hohe Analyseauflösung für kleine und entfernte Schilder
  const INTERVAL_MS = 100;  // Rechenzeit für die größere Analysefläche einplanen

  // Das GTSRB-CNN liefert bei Tempolimits direkt die passende Zahl als Klasse.
  let characterSessionPromise = null, signSessionPromise = null;
  let signQueue = Promise.resolve(), characterQueue = Promise.resolve();
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
  function getSignSession() {
    if (!window.ort) return Promise.reject(new Error('ONNX Runtime Web nicht verfügbar'));
    if (!signSessionPromise) {
      window.ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.22.0/dist/';
      signSessionPromise = window.ort.InferenceSession.create('models/gtsrb-sign-cnn.onnx', {
        executionProviders: ['wasm'], graphOptimizationLevel: 'all'
      }).catch(error => {
        signSessionPromise = null;
        throw error;
      });
    }
    return signSessionPromise;
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
  let tracker = D.createTracker(4, 2), current = [], mask = null;
  let lastBackendKey = '', backendRetryAt = 0, lastBackendSentAt = 0;

  backendUrlInput.value = localStorage.getItem('signScannerBackendUrl') || '';
  backendUrlInput.addEventListener('change', () => {
    localStorage.setItem('signScannerBackendUrl', backendUrlInput.value.trim());
    lastBackendKey = '';
    publishBackendSnapshot();
  });

  /** Send only confirmed, currently visible signs to the configured PC service. */
  function publishBackendSnapshot() {
    const signs = current.filter(t => !t.rejected && (!CNN_SIGN_LABELS.has(t.label) || t.cnnVerified) && (!TEXT_SIGN_LABELS.has(t.label) || t.textVerified));
    const items = signs.map(t => ({
      label: t.label,
      name: D.SIGNS[t.label].name,
      value: t.label === 'verbot' ? Number((t.cnnText || '').match(/\d+/)?.[0]) || null : null,
      text: t.ocrText || t.cnnText || t.state || '',
      confidence: Number(t.conf.toFixed(3))
    }));
    const key = JSON.stringify(items);
    if ((key === lastBackendKey && (!running || Date.now() - lastBackendSentAt < 250)) || Date.now() < backendRetryAt) return;
    const endpoint = backendUrlInput.value.trim() || (location.protocol.startsWith('http') ? location.origin + '/api/signs' : '');
    if (!endpoint) { backendStatus.textContent = 'Adresse fehlt (Backend läuft auf dem PC).'; return; }
    backendStatus.textContent = 'Sende erkannte Schilder …';
    const jpegFrame = (running || still) ? createAnnotatedJpeg() : null;
    fetch(endpoint, {
      method: 'POST', headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ source: 'strassenschilder-scanner', detectedAt: new Date().toISOString(), signs: items, imageJpeg: jpegFrame })
    }).then(response => {
      if (!response.ok) throw new Error('HTTP ' + response.status);
      lastBackendKey = key;
      lastBackendSentAt = Date.now();
      backendStatus.textContent = 'Mit PC-Backend verbunden · ' + items.length + ' Schild(er)';
    }).catch(error => {
      backendRetryAt = Date.now() + 5000;
      backendStatus.textContent = 'Backend nicht erreichbar: ' + error.message;
    });
  }

  /** Encode the current analysis frame with the same detection boxes as the preview. */
  function createAnnotatedJpeg() {
    const canvas = document.createElement('canvas');
    const outputScale = Math.min(1, 480 / work.width);
    canvas.width = Math.round(work.width * outputScale);
    canvas.height = Math.round(work.height * outputScale);
    const previewCtx = canvas.getContext('2d');
    previewCtx.setTransform(outputScale, 0, 0, outputScale, 0, 0);
    previewCtx.drawImage(work, 0, 0);
    const markScale = Math.max(1, work.width / 320);
    previewCtx.lineWidth = markScale * 2;
    previewCtx.font = `600 ${Math.max(12, Math.round(work.width / 48))}px system-ui, sans-serif`;
    previewCtx.textBaseline = 'top';
    for (const detection of current) {
      if (detection.rejected || (CNN_SIGN_LABELS.has(detection.label) && !detection.cnnVerified) || (TEXT_SIGN_LABELS.has(detection.label) && !detection.textVerified)) continue;
      const sign = D.SIGNS[detection.label];
      if (!sign) continue;
      const marginX = detection.w * 0.06, marginY = detection.h * 0.06;
      const x = Math.max(0, detection.x - marginX), y = Math.max(0, detection.y - marginY);
      const w = Math.min(work.width - x, detection.w + marginX * 2);
      const h = Math.min(work.height - y, detection.h + marginY * 2);
      const label = (detection.ocrText ? detection.ocrText + ' · ' : '') + sign.name + ' ' + Math.round(detection.conf * 100) + '%';
      const labelWidth = Math.min(work.width - x, previewCtx.measureText(label).width + markScale * 8);
      const labelY = y > markScale * 24 ? y - markScale * 24 : y + h;
      previewCtx.strokeStyle = sign.hex;
      previewCtx.strokeRect(x, y, w, h);
      previewCtx.fillStyle = sign.hex;
      previewCtx.fillRect(x, labelY, labelWidth, markScale * 23);
      previewCtx.fillStyle = sign.text;
      previewCtx.fillText(label, x + markScale * 4, labelY + markScale * 3);
    }
    return canvas.toDataURL('image/jpeg', 0.35);
  }

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

  /** Schildkandidaten mit dem GTSRB-CNN klassifizieren; Speedklassen liefern direkt den Zahlenwert. */
  function queueOcr(source, detections, analysisW, analysisH) {
    for (const detection of detections) {
      if (detection.ocrQueued || detection.ocrText || detection.ocrFailed) continue;
      if (detection.label === 'hinweis' || detection.label === 'ortstafel') {
        queueSignText(source, detection, analysisW, analysisH);
        continue;
      }
      // Jede rote Formheuristik bleibt zunächst unsichtbar, bis das CNN den
      // Kandidaten einer passenden GTSRB-Klasse zugeordnet hat.
      if (!CNN_SIGN_LABELS.has(detection.label)) continue;
      detection.ocrQueued = true;
      const sx = source.videoWidth || source.naturalWidth || source.width;
      const sy = source.videoHeight || source.naturalHeight || source.height;
      // Keep a visible margin so the model also sees the sign border and context.
      const padX = detection.w * 0.12, padY = detection.h * 0.12;
      const signX = Math.max(0, Math.floor((detection.x - padX) * sx / analysisW));
      const signY = Math.max(0, Math.floor((detection.y - padY) * sy / analysisH));
      const signRight = Math.min(sx, Math.ceil((detection.x + detection.w + padX) * sx / analysisW));
      const signBottom = Math.min(sy, Math.ceil((detection.y + detection.h + padY) * sy / analysisH));
      const signCrop = document.createElement('canvas');
      signCrop.width = signCrop.height = 96;
      signCrop.getContext('2d').drawImage(source, signX, signY, signRight - signX, signBottom - signY, 0, 0, 96, 96);
      signQueue = signQueue.then(async () => {
        // Das Verkehrszeichen-CNN klassifiziert direkt den gesamten Kandidaten.
        const signSession = await getSignSession();
        const signClass = await classifyTrafficSign(signSession, signCrop);
        if (signClass.index === 14 && signClass.confidence >= 0.6 && signClass.margin >= 0.15) {
          detection.label = 'stop';
          detection.cnnVerified = true;
          detection.stopVerified = true;
          detection.ocrText = 'STOP';
          detection.cnnText = 'STOP';
          renderList();
          refreshStill();
          return;
        }
        if (signClass.index === 17 && signClass.confidence >= MIN_SIGN_CONFIDENCE && signClass.margin >= MIN_SIGN_MARGIN) {
          detection.label = 'einfahrtVerboten';
          detection.cnnVerified = true;
          renderList();
          refreshStill();
          return;
        }
        if (signClass.index === 13 && signClass.confidence >= MIN_SIGN_CONFIDENCE && signClass.margin >= MIN_SIGN_MARGIN) {
          detection.label = 'vorfahrtGewaehren';
          detection.cnnVerified = true;
          renderList();
          refreshStill();
          return;
        }
        if (GTSRB_WARNING_CLASSES.has(signClass.index) && signClass.confidence >= MIN_SIGN_CONFIDENCE && signClass.margin >= MIN_SIGN_MARGIN) {
          detection.label = 'warnung';
          detection.cnnVerified = true;
          renderList();
          refreshStill();
          return;
        }
        if (signClass.index === 12 && signClass.confidence >= MIN_SIGN_CONFIDENCE && signClass.margin >= MIN_SIGN_MARGIN) {
          detection.label = 'vorfahrtstrasse';
          detection.cnnVerified = true;
          renderList();
          refreshStill();
          return;
        }
        if (signClass.index >= 33 && signClass.index <= 40 && signClass.confidence >= MIN_SIGN_CONFIDENCE && signClass.margin >= MIN_SIGN_MARGIN) {
          detection.label = 'gebot';
          detection.cnnVerified = true;
          renderList();
          refreshStill();
          return;
        }
        const expectedSpeed = GTSRB_SPEED_CLASSES.get(signClass.index);
        if (!expectedSpeed || signClass.confidence < MIN_SIGN_CONFIDENCE || signClass.margin < MIN_SIGN_MARGIN) {
          detection.rejected = true;
          renderList();
          refreshStill();
          return;
        }
        detection.label = 'verbot';
        detection.cnnVerified = true;
        detection.cnnText = expectedSpeed + ' km/h';
        detection.ocrText = detection.cnnText;
        renderList();
        refreshStill();
      }).catch(() => {
        detection.cnnText = '';
        detection.rejected = true;
        detection.ocrFailed = true;
        renderList();
        refreshStill();
      });
    }
  }

  /** Verkehrszeichen-CNN klassifiziert den gesamten roten Kandidaten statt Glyphen. */
  async function classifyTrafficSign(session, signCrop) {
    const size = 32, input = new Float32Array(3 * size * size);
    const canvas = document.createElement('canvas');
    canvas.width = canvas.height = size;
    const c = canvas.getContext('2d', { willReadFrequently: true });
    c.imageSmoothingEnabled = true;
    c.drawImage(signCrop, 0, 0, size, size);
    const pixels = claheLuminance(c.getImageData(0, 0, size, size).data, size);
    const mean = [0.41359345, 0.38223032, 0.39460091];
    const std = [0.27057118, 0.26105214, 0.26897097];
    for (let i = 0; i < size * size; i++) {
      input[i] = (pixels[i * 4] / 255 - mean[0]) / std[0];
      input[size * size + i] = (pixels[i * 4 + 1] / 255 - mean[1]) / std[1];
      input[2 * size * size + i] = (pixels[i * 4 + 2] / 255 - mean[2]) / std[2];
    }
    const tensor = new window.ort.Tensor('float32', input, [1, 3, size, size]);
    const result = await session.run({ [session.inputNames[0]]: tensor });
    const logits = result[session.outputNames[0]].data;
    let max = -Infinity, index = -1, second = -Infinity, secondIndex = -1;
    for (let i = 0; i < logits.length; i++) {
      if (logits[i] > max) { second = max; secondIndex = index; max = logits[i]; index = i; }
      else if (logits[i] > second) { second = logits[i]; secondIndex = i; }
    }
    const exps = Array.from(logits, value => Math.exp(value - max));
    const sum = exps.reduce((a, b) => a + b, 0);
    return { index, confidence: exps[index] / Math.max(sum, 1e-12), margin: (exps[index] - exps[secondIndex]) / Math.max(sum, 1e-12) };
  }

  /** CLAHE-Näherung auf 4×4 Helligkeits-Kacheln, passend zur CNN-Vorverarbeitung. */
  function claheLuminance(source, size) {
    const tiles = 4, tileSize = size / tiles, area = tileSize * tileSize;
    const luma = new Uint8Array(size * size), luts = [];
    for (let i = 0; i < luma.length; i++) {
      const r = source[i * 4], g = source[i * 4 + 1], b = source[i * 4 + 2];
      luma[i] = Math.round(0.299 * r + 0.587 * g + 0.114 * b);
    }
    const clipLimit = Math.max(1, Math.floor(2 * area / 256));
    for (let ty = 0; ty < tiles; ty++) for (let tx = 0; tx < tiles; tx++) {
      const hist = new Uint32Array(256);
      for (let y = ty * tileSize; y < (ty + 1) * tileSize; y++) for (let x = tx * tileSize; x < (tx + 1) * tileSize; x++) hist[luma[y * size + x]]++;
      let excess = 0;
      for (let i = 0; i < 256; i++) if (hist[i] > clipLimit) { excess += hist[i] - clipLimit; hist[i] = clipLimit; }
      const batch = Math.floor(excess / 256), residual = excess - batch * 256;
      for (let i = 0; i < 256; i++) hist[i] += batch;
      const step = residual ? Math.max(1, Math.floor(256 / residual)) : 256;
      for (let i = 0; i < 256 && i / step < residual; i += step) hist[i]++;
      const lut = new Uint8Array(256); let cumulative = 0;
      for (let i = 0; i < 256; i++) { cumulative += hist[i]; lut[i] = Math.max(0, Math.min(255, Math.round(cumulative * 255 / area))); }
      luts.push(lut);
    }
    const out = new Uint8ClampedArray(source.length);
    for (let y = 0; y < size; y++) for (let x = 0; x < size; x++) {
      const fx = x / tileSize - 0.5, fy = y / tileSize - 0.5;
      const xr = Math.floor(fx), yr = Math.floor(fy), ax = fx - xr, ay = fy - yr;
      const x0 = Math.max(0, Math.min(tiles - 1, xr)), x1 = Math.max(0, Math.min(tiles - 1, xr + 1));
      const y0 = Math.max(0, Math.min(tiles - 1, yr)), y1 = Math.max(0, Math.min(tiles - 1, yr + 1));
      const value = luma[y * size + x];
      const top = luts[y0 * tiles + x0][value] * (1 - ax) + luts[y0 * tiles + x1][value] * ax;
      const bottom = luts[y1 * tiles + x0][value] * (1 - ax) + luts[y1 * tiles + x1][value] * ax;
      const equalized = top * (1 - ay) + bottom * ay, ratio = value > 2 ? equalized / value : 1;
      const i = (y * size + x) * 4;
      out[i] = Math.min(255, source[i] * ratio); out[i + 1] = Math.min(255, source[i + 1] * ratio);
      out[i + 2] = Math.min(255, source[i + 2] * ratio); out[i + 3] = 255;
    }
    return out;
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
    crop.width = Math.max(1, (right - x) * 3);
    crop.height = Math.max(1, (bottom - y) * 3);
    const cropCtx = crop.getContext('2d', { willReadFrequently: true });
    cropCtx.drawImage(source, x, y, right - x, bottom - y, 0, 0, crop.width, crop.height);
    const binary = cropCtx.getImageData(0, 0, crop.width, crop.height);
    const histogram = new Uint32Array(256), grayValues = new Uint8Array(crop.width * crop.height);
    for (let i = 0, p = 0; i < binary.data.length; i += 4, p++) {
      const r = binary.data[i], g = binary.data[i + 1], b = binary.data[i + 2];
      grayValues[p] = Math.round(r * 0.299 + g * 0.587 + b * 0.114);
      histogram[grayValues[p]]++;
    }
    const otsu = otsuThreshold(histogram);
    const threshold = detection.label === 'hinweis'
      ? Math.max(150, otsu)
      : Math.max(75, Math.min(185, otsu));
    for (let i = 0, p = 0; i < binary.data.length; i += 4, p++) {
      const foreground = detection.label === 'hinweis' ? grayValues[p] > threshold : grayValues[p] < threshold;
      const value = foreground ? 0 : 255;
      binary.data[i] = binary.data[i + 1] = binary.data[i + 2] = value;
    }
    cropCtx.putImageData(binary, 0, 0);

    characterQueue = characterQueue.then(async () => {
      const session = await getCharacterSession();
      const rows = segmentTextCharacters(binary, crop.width, crop.height);
      const output = [];
      for (const row of rows) {
        let text = '';
        for (let i = 0; i < Math.min(row.length, 24); i++) {
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
      await worker.setParameters({
        preserve_interword_spaces: '1', tessedit_char_whitelist: '',
        tessedit_pageseg_mode: '3', classify_bln_numeric_mode: '0'
      });
      const { data } = await worker.recognize(crop);
      const confidentWords = (data.words || []).filter(word => word.confidence >= 35);
      const text = confidentWords.length ? confidentWords.map(word => word.text).join(' ') : (data.text || '');
      detection.ocrText = text.replace(/[^\p{L}\p{N}\s.,'’/-]/gu, ' ').replace(/\s+/g, ' ').trim();
      detection.textVerified = confidentWords.some(word => word.confidence >= 60 && /[\p{L}\p{N}]{3,}/u.test(word.text));
      if (!detection.textVerified) detection.rejected = true;
      renderList();
      refreshStill();
    }).catch(() => {
      detection.ocrFailed = true;
      detection.rejected = true;
      renderList();
      refreshStill();
    });
  }

  /** Otsu-Schwelle: passt den Kontrastfilter an Belichtung und Schildfarbe an. */
  function otsuThreshold(histogram) {
    let total = 0, sum = 0;
    for (let i = 0; i < 256; i++) { total += histogram[i]; sum += i * histogram[i]; }
    let backgroundWeight = 0, backgroundSum = 0, bestVariance = -1, threshold = 128;
    for (let i = 0; i < 256; i++) {
      backgroundWeight += histogram[i];
      if (!backgroundWeight) continue;
      const foregroundWeight = total - backgroundWeight;
      if (!foregroundWeight) break;
      backgroundSum += i * histogram[i];
      const delta = backgroundSum / backgroundWeight - (sum - backgroundSum) / foregroundWeight;
      const variance = backgroundWeight * foregroundWeight * delta * delta;
      if (variance > bestVariance) { bestVariance = variance; threshold = i; }
    }
    return threshold;
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
    const max = scores[best];
    let denominator = 0;
    for (let i = 0; i < 37; i++) denominator += Math.exp(scores[i] - max);
    if (best === 36 || 1 / denominator < 0.32) return '';
    return '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'[best];
  }

  function renderList() {
    const shown = current.filter(t => !t.rejected && (!CNN_SIGN_LABELS.has(t.label) || t.cnnVerified) && (!TEXT_SIGN_LABELS.has(t.label) || t.textVerified));
    publishBackendSnapshot();
    const labels = [...new Set(shown.map(t => t.label))];
    const key = labels.map(l => l + ':' + shown.filter(t => t.label === l)
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
      const texts = [...new Set(shown.filter(t => t.label === label).map(t => t.ocrText).filter(Boolean))];
      if (texts.length) {
        const read = document.createElement('p');
        read.className = 'ocr-text';
        read.textContent = 'Gelesener Text: ' + texts.join(' · ');
        li.append(read);
      } else if (shown.some(t => t.label === label && t.ocrFailed)) {
        const failed = document.createElement('p');
        failed.textContent = 'Texterkennung konnte nicht geladen werden.';
        li.append(failed);
      }
      const cnnTexts = [...new Set(shown.filter(t => t.label === label).map(t => t.cnnText).filter(Boolean))];
      if (cnnTexts.length && (!texts.length || cnnTexts.some(text => !texts.includes(text)))) {
        const cnn = document.createElement('p');
        cnn.className = 'cnn-text';
        cnn.textContent = label === 'verbot' ? 'CNN-Zahl: ' + cnnTexts.join(' · ') : 'CNN-Zeichen: ' + cnnTexts.join(' · ');
        li.append(cnn);
      } else if (shown.some(t => t.label === label && t.cnnFailed)) {
        const failed = document.createElement('p');
        failed.textContent = 'Zeichen-CNN konnte nicht geladen werden.';
        li.append(failed);
      }
      const lightStates = [...new Set(shown.filter(t => t.label === label).map(t => t.state).filter(Boolean))];
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
    ctx.lineWidth = 4; ctx.textBaseline = 'top';
    ctx.font = '600 15px Bahnschrift, "DIN Alternate", system-ui, sans-serif';
    for (const t of current) {
      if (t.rejected || (CNN_SIGN_LABELS.has(t.label) && !t.cnnVerified) || (TEXT_SIGN_LABELS.has(t.label) && !t.textVerified)) continue;
      const s = D.SIGNS[t.label], mx = t.w * 0.06, my = t.h * 0.06;
      const left = Math.max(0, t.x - mx), top = Math.max(0, t.y - my);
      const x = left * k, y = top * k;
      const w = Math.min(work.width - left, t.w + mx * 2) * k;
      const h = Math.min(work.height - top, t.h + my * 2) * k;
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
    // Modell und WASM parallel zur Kameraberechtigung laden, damit die erste
    // Schilderkennung nicht auf den Kaltstart warten muss.
    getCharacterSession().catch(() => {});
    getSignSession().catch(() => {});
    try {
      stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
      video.srcObject = stream; await video.play();
      still = null; tracker = D.createTracker(4, 2); running = true;
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
