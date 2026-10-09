/**
 * detector.js – Kernlogik der Schilderkennung.
 * Keine DOM-Abhängigkeit → im Browser (window.SignDetector) und in Node (require) nutzbar.
 *
 * Ablauf (Details: docs/DOKUMENTATION.md):
 *   1. buildMask       Pixel nach Farbe klassifizieren (rot / blau / gelb) über HSV
 *   2. findComponents  zusammenhängende Farbflächen finden (Flood-Fill) und grob filtern
 *   3. analyzeShape    Silhouette der Fläche vermessen (Füllgrad, Breitenprofil)
 *   4. shapeOf/labelOf Form + Farbe → Schildtyp
 *   5. createTracker   Treffer über mehrere Bilder stabilisieren (gegen Flackern)
 */
(function (root) {
  'use strict';

  const NONE = 0, RED = 1, BLUE = 2, YELLOW = 3, GREEN = 4;

  /** Schwellwerte. Alle Größen in Pixeln des verkleinerten Analysebilds. */
  const CONFIG = {
    minSaturation: 0.52,
    minValue: 0.2,
    minBox: 10,
    minArea: 38,
    maxFrameShare: 0.92,
    minAspect: 0.7,
    maxAspect: 4.5
  };

  /** Katalog der erkennbaren Schilder (Zeichen-Nummern nach StVO). */
  const SIGNS = {
    stop:              { name: 'Stopp',             zeichen: 'Z 206',     hex: '#d3232f', text: '#fff',    note: 'STOP im Schildinneren per CNN erkannt.' },
    vorfahrtGewaehren: { name: 'Vorfahrt gewähren', zeichen: 'Z 205',     hex: '#d3232f', text: '#fff',    note: 'Rotes Dreieck mit der Spitze nach unten.' },
    warnung:           { name: 'Gefahrzeichen',     zeichen: 'Z 101 ff.', hex: '#d3232f', text: '#fff',    note: 'Rotes Dreieck, Spitze oben. Das Symbol im Inneren wird nicht gelesen.' },
    verbot:            { name: 'Tempolimit / Verbot', zeichen: 'Z 274 ff.', hex: '#d3232f', text: '#fff',    note: 'Die Ziffer im roten Ring wird automatisch gelesen.' },
    einfahrtVerboten:  { name: 'Einfahrt verboten', zeichen: 'Z 267',     hex: '#d3232f', text: '#fff',    note: 'Roter Vollkreis mit weißem Balken.' },
    gebot:             { name: 'Gebotszeichen',     zeichen: 'Z 2xx',     hex: '#1467b8', text: '#fff',    note: 'Blauer Kreis, z. B. vorgeschriebene Fahrtrichtung oder Radweg.' },
    hinweis:           { name: 'Hinweiszeichen',    zeichen: 'Z 3xx',     hex: '#1467b8', text: '#fff',    note: 'Blaues Rechteck, z. B. Parkplatz (Z 314).' },
    vorfahrtstrasse:   { name: 'Vorfahrtstraße',    zeichen: 'Z 306',     hex: '#f2c200', text: '#1e2329', note: 'Gelbe Raute mit weißem Rand.' },
    ortstafel:         { name: 'Ortstafel',         zeichen: 'Z 310',     hex: '#f2c200', text: '#1e2329', note: 'Gelbes Rechteck am Ortseingang.' },
    ampel:             { name: 'Ampel',             zeichen: 'Lichtsignal', hex: '#303941', text: '#fff', note: 'Vertikale Rot-Gelb-Grün-Lichtfelder erkannt.' }
  };

  /** Schritt 1: RGB → Farbklasse. h in Grad (0..360), s und v in 0..1. */
  function classifyPixel(r, g, b, cfg) {
    const max = Math.max(r, g, b), min = Math.min(r, g, b), d = max - min;
    const v = max / 255;
    if (d === 0 || v < cfg.minValue) return NONE;
    const s = d / max;
    if (s < cfg.minSaturation) return NONE;
    let h;
    if (max === r) h = 60 * (((g - b) / d + 6) % 6);
    else if (max === g) h = 60 * ((b - r) / d + 2);
    else h = 60 * ((r - g) / d + 4);
    if (h <= 20 || h >= 337) return RED;          // Rot liegt am Rand des Farbkreises
    if (h >= 30 && h <= 72 && v > 0.35) return YELLOW;
    if (h >= 185 && h <= 265) return BLUE;
    if (h >= 68 && h <= 175) return GREEN;
    return NONE;
  }

  /** Maske für das ganze Bild (RGBA-Array wie von getImageData). */
  function buildMask(data, w, h, cfg) {
    const mask = new Uint8Array(w * h);
    for (let i = 0, p = 0; i < mask.length; i++, p += 4) mask[i] = classifyPixel(data[p], data[p + 1], data[p + 2], cfg);
    return mask;
  }

  /**
   * Schritt 2: Flood-Fill (4er-Nachbarschaft, gleiche Farbklasse).
   * Die Warteschlange `queue` enthält danach alle Pixel; jede Fläche belegt einen
   * zusammenhängenden Abschnitt [start, end) – so braucht man keine Extra-Listen.
   */
  function findComponents(mask, w, h, cfg) {
    const n = w * h, queue = new Int32Array(n), seen = new Uint8Array(n), comps = [];
    let tail = 0;
    for (let i = 0; i < n; i++) {
      const c = mask[i];
      if (!c || seen[i]) continue;
      const start = tail;
      let head = tail, x0 = w, x1 = 0, y0 = h, y1 = 0;
      queue[tail++] = i; seen[i] = 1;
      while (head < tail) {
        const p = queue[head++], x = p % w, y = (p / w) | 0;
        if (x < x0) x0 = x; if (x > x1) x1 = x;
        if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (x > 0     && !seen[p - 1] && mask[p - 1] === c) { seen[p - 1] = 1; queue[tail++] = p - 1; }
        if (x < w - 1 && !seen[p + 1] && mask[p + 1] === c) { seen[p + 1] = 1; queue[tail++] = p + 1; }
        if (y > 0     && !seen[p - w] && mask[p - w] === c) { seen[p - w] = 1; queue[tail++] = p - w; }
        if (y < h - 1 && !seen[p + w] && mask[p + w] === c) { seen[p + w] = 1; queue[tail++] = p + w; }
      }
      const bw = x1 - x0 + 1, bh = y1 - y0 + 1, area = tail - start, aspect = bw / bh;
      if (area < cfg.minArea || bw < cfg.minBox || bh < cfg.minBox) continue;   // zu klein
      if (aspect < cfg.minAspect || aspect > cfg.maxAspect) continue;            // falsches Seitenverhältnis
      if (bw > w * cfg.maxFrameShare || bh > h * cfg.maxFrameShare) continue;    // füllt das Bild
      comps.push({ color: c, x: x0, y: y0, w: bw, h: bh, area, start, end: tail });
    }
    return { comps, queue };
  }

  /**
   * Schritt 3: Silhouette vermessen. Pro Zeile zählt nur der äußerste linke und rechte
   * Pixel – so werden Ringe (rote Ränder) zur gefüllten Form.
   *   solidity = Silhouettenfläche / Rahmenfläche   (Kreis .785, Dreieck/Raute .5, Rechteck 1)
   *   fill     = echte Farbpixel / Silhouettenfläche (Ring niedrig, Vollfläche hoch)
   *   wTop/wBot = mittlere Breite im oberen/unteren Viertel relativ zur Rahmenbreite
   */
  function analyzeShape(c, queue, w) {
    const rowMin = new Int16Array(c.h).fill(32767), rowMax = new Int16Array(c.h).fill(-1);
    for (let k = c.start; k < c.end; k++) {
      const p = queue[k], x = p % w, y = ((p / w) | 0) - c.y;
      if (x < rowMin[y]) rowMin[y] = x;
      if (x > rowMax[y]) rowMax[y] = x;
    }
    const nq = Math.max(1, Math.round(c.h / 4));
    let sil = 0, top = 0, bot = 0;
    for (let y = 0; y < c.h; y++) {
      const width = rowMax[y] >= 0 ? rowMax[y] - rowMin[y] + 1 : 0;
      sil += width;
      if (y < nq) top += width;
      if (y >= c.h - nq) bot += width;
    }
    return { solidity: sil / (c.w * c.h), fill: c.area / sil, wTop: top / nq / c.w, wBot: bot / nq / c.w };
  }

  /** Schritt 4a: Form bestimmen. */
  function shapeOf(s, aspect) {
    if (s.solidity < 0.65) {                                   // Dreieck oder Raute (beide ≈ 0.5)
      if (s.wTop < 0.3 && s.wBot < 0.3) return 'diamond';      // schmal oben UND unten → Raute
      return s.wTop < s.wBot ? 'triangleUp' : 'triangleDown';
    }
    if (s.solidity > 0.92) return 'rect';
    if (aspect > 0.78 && aspect < 1.28 && s.solidity > 0.67) return 'circle';
    return null;
  }

  /** Schritt 4b: Farbe + Form → Schildtyp (oder null). */
  function labelOf(color, shape, fill, stats) {
    if (color === RED) {
      if (shape === 'triangleDown') return 'vorfahrtGewaehren';
      if (shape === 'triangleUp') return 'warnung';
      if (shape === 'circle') return fill > 0.65 ? 'einfahrtVerboten' : 'verbot';
    } else if (color === BLUE) {
      if (shape === 'circle') return 'gebot';
      if (shape === 'rect') return 'hinweis';
    } else if (color === YELLOW) {
      if (shape === 'diamond') return 'vorfahrtstrasse';
      if (shape === 'rect') return 'ortstafel';
    }
    return null;
  }

  /** Schildtypische Form-, Farb- und Innenflächenmerkmale gegen Farbflecken prüfen. */
  function signEvidence(data, mask, imageW, comp, shape) {
    const aspect = comp.w / comp.h;
    const shapeRanges = {
      circle: [0.7, 1.4, 0.67, 0.91],
      triangleUp: [0.7, 1.4, 0.34, 0.67], triangleDown: [0.7, 1.4, 0.34, 0.67],
      diamond: [0.7, 1.4, 0.34, 0.67], rect: [0.85, 4.5, 0.82, 1.02]
    }[shape];
    const stats = analyzeShape(comp, mask.queue, imageW);
    if (!shapeRanges || aspect < shapeRanges[0] || aspect > shapeRanges[1] ||
        stats.solidity < shapeRanges[2] || stats.solidity > shapeRanges[3]) return null;

    let expected = 0, colored = 0, neutral = 0, interior = 0;
    for (let y = comp.y; y < comp.y + comp.h; y++) for (let x = comp.x; x < comp.x + comp.w; x++) {
      const nx = (x - comp.x + 0.5) / comp.w, ny = (y - comp.y + 0.5) / comp.h;
      let inside = false;
      if (shape === 'circle') inside = ((nx - 0.5) / 0.5) ** 2 + ((ny - 0.5) / 0.5) ** 2 <= 1;
      else if (shape === 'triangleUp') inside = ny >= Math.abs(nx - 0.5) * 2;
      else if (shape === 'triangleDown') inside = 1 - ny >= Math.abs(nx - 0.5) * 2;
      else if (shape === 'diamond') inside = Math.abs(nx - 0.5) + Math.abs(ny - 0.5) <= 0.5;
      else inside = true;
      if (!inside) continue;
      expected++;
      const p = y * imageW + x, color = mask.pixels[p];
      if (color === comp.color) colored++;
      const i = p * 4, r = data[i], g = data[i + 1], b = data[i + 2], max = Math.max(r, g, b);
      const saturation = max ? (max - Math.min(r, g, b)) / max : 0;
      const core = (nx - 0.5) ** 2 + (ny - 0.5) ** 2 < 0.12;
      if (core) { interior++; if (saturation < 0.42) neutral++; }
    }
    const colorShare = colored / Math.max(1, expected);
    const neutralShare = neutral / Math.max(1, interior);
    if (colorShare < (comp.color === RED ? 0.07 : 0.24)) return null;
    const needsInteriorDetail = comp.color === RED || comp.color === BLUE || (comp.color === YELLOW && shape === 'rect');
    if (needsInteriorDetail && neutralShare < (comp.color === RED ? 0.12 : 0.08)) return null;

    // Ein echter roter Kreis trägt einen umlaufenden Ring. Einzelne rote
    // Kreisflächen wie Rücklichter bestehen diese Winkelabdeckung meist nicht.
    if (comp.color === RED && shape === 'circle') {
      let covered = 0;
      for (let sector = 0; sector < 16; sector++) {
        const angle = (sector + 0.5) * Math.PI * 2 / 16;
        const x = Math.round(comp.x + comp.w * (0.5 + Math.cos(angle) * 0.39));
        const y = Math.round(comp.y + comp.h * (0.5 + Math.sin(angle) * 0.39));
        let found = false;
        for (let dy = -1; dy <= 1 && !found; dy++) for (let dx = -1; dx <= 1; dx++) {
          const px = x + dx, py = y + dy;
          if (px >= 0 && px < imageW && py >= 0 && py < mask.height && mask.pixels[py * imageW + px] === RED) { found = true; break; }
        }
        if (found) covered++;
      }
      if (covered < 13) return null;
    }
    const shapeConf = confidence(shape, stats.solidity);
    return shapeConf * 0.72 + Math.min(1, colorShare * 1.5) * 0.18 + Math.min(1, neutralShare * 2) * 0.1;
  }

  /** Sicherheit 0..1: wie nah liegt die gemessene Füllung am Idealwert der Form? */
  const IDEAL = { circle: 0.785, triangleUp: 0.5, triangleDown: 0.5, diamond: 0.5, rect: 1 };
  const confidence = (shape, solidity) => Math.max(0, Math.min(1, 1 - Math.abs(solidity - IDEAL[shape]) * 3));

  /** Ein Bild analysieren. `data` ist RGBA (ImageData.data). */
  function detectTrafficLights(comps, queue, data, w) {
    const lamps = comps.filter(c => {
      if (![RED, YELLOW, GREEN].includes(c.color)) return false;
      const aspect = c.w / c.h;
      if (aspect < 0.62 || aspect > 1.4) return false;
      const stats = analyzeShape(c, queue, w);
      return stats.solidity > 0.55 && stats.solidity < 0.98;
    });
    const red = lamps.filter(c => c.color === RED), yellow = lamps.filter(c => c.color === YELLOW), green = lamps.filter(c => c.color === GREEN);
    const out = [];
    for (const r of red) for (const y of yellow) for (const g of green) {
      const ordered = [r, y, g];
      const centers = ordered.map(c => c.x + c.w / 2);
      const sizes = ordered.map(c => (c.w + c.h) / 2);
      const centerX = centers.reduce((a, b) => a + b, 0) / 3;
      const typical = sizes.reduce((a, b) => a + b, 0) / 3;
      if (Math.max(...centers) - Math.min(...centers) > typical * 0.65) continue;
      if (Math.max(...sizes) / Math.max(1, Math.min(...sizes)) > 1.8) continue;
      if (y.y <= r.y || g.y <= y.y) continue;
      const gap1 = y.y - (r.y + r.h), gap2 = g.y - (y.y + y.h);
      if (gap1 > typical * 2.5 || gap2 > typical * 2.5 || gap1 < -typical * 0.4 || gap2 < -typical * 0.4) continue;
      const brightness = ordered.map(c => {
        let sum = 0;
        for (let k = c.start; k < c.end; k++) {
          const p = queue[k] * 4;
          sum += Math.max(data[p], data[p + 1], data[p + 2]);
        }
        return sum / Math.max(1, c.area);
      });
      const brightest = brightness.indexOf(Math.max(...brightness));
      const names = ['Rot', 'Gelb', 'Grün'];
      out.push({ label: 'ampel', state: names[brightest], x: Math.min(...ordered.map(c => c.x)), y: r.y,
        w: Math.max(...ordered.map(c => c.x + c.w)) - Math.min(...ordered.map(c => c.x)),
        h: g.y + g.h - r.y, conf: 0.82 });
      break;
    }
    return out;
  }

  function detect(data, w, h, options) {
    const cfg = Object.assign({}, CONFIG, options);
    const pixels = buildMask(data, w, h, cfg);
    const mask = { pixels, height: h };
    const { comps, queue } = findComponents(pixels, w, h, cfg);
    mask.queue = queue;
    const detections = [];
    for (const c of comps) {
      const s = analyzeShape(c, queue, w);
      const shape = shapeOf(s, c.w / c.h);
      const label = shape && labelOf(c.color, shape, s.fill, s);
      if (!label) continue;
      const conf = signEvidence(data, mask, w, c, shape);
      if (conf >= 0.78) detections.push({ label, shape, x: c.x, y: c.y, w: c.w, h: c.h, conf });
    }
    detections.push(...detectTrafficLights(comps, queue, data, w));
    return { detections: suppressNestedDetections(detections), mask: pixels };
  }

  /** Farbige Innenformen nicht zusätzlich als eigenständige Schilder melden. */
  function suppressNestedDetections(detections) {
    const ranked = detections.slice().sort((a, b) => b.conf - a.conf || (b.w * b.h) - (a.w * a.h));
    const kept = [];
    for (const candidate of ranked) {
      const area = candidate.w * candidate.h;
      const insideExisting = kept.some(parent => {
        const parentArea = parent.w * parent.h;
        const iw = Math.max(0, Math.min(parent.x + parent.w, candidate.x + candidate.w) - Math.max(parent.x, candidate.x));
        const ih = Math.max(0, Math.min(parent.y + parent.h, candidate.y + candidate.h) - Math.max(parent.y, candidate.y));
        const overlap = iw * ih;
        // Eine Rahmenform darf nicht durch ihre Schrift, Symbole oder Farbflecken
        // als mehrere innere Schilder erscheinen. Auch fast deckungsgleiche
        // Mehrfachkonturen werden zusammengeführt.
        return (parentArea >= area * 1.12 && overlap / Math.max(1, area) > 0.68) ||
          (overlap / Math.max(1, area) > 0.88 && overlap / Math.max(1, parentArea) > 0.72);
      });
      if (!insideExisting) kept.push(candidate);
    }
    return kept;
  }

  /** Schritt 5: Treffer über Bilder hinweg verfolgen. Ein Schild gilt erst nach 3 Treffern als sicher. */
  function iou(a, b) {
    const ix = Math.max(0, Math.min(a.x + a.w, b.x + b.w) - Math.max(a.x, b.x));
    const iy = Math.max(0, Math.min(a.y + a.h, b.y + b.h) - Math.max(a.y, b.y));
    const inter = ix * iy;
    return inter / (a.w * a.h + b.w * b.h - inter);
  }

  const CNN_CANDIDATE_LABELS = new Set(['stop', 'verbot', 'einfahrtVerboten', 'vorfahrtGewaehren', 'warnung', 'gebot', 'vorfahrtstrasse']);
  function createTracker(minHits = 3, maxMiss = 3) {
    let tracks = [], nextId = 1;
    return {
      update(dets) {
        tracks.forEach(t => { t.matched = false; });
        for (const d of dets) {
          let best = null, bestIou = 0.25;
          for (const t of tracks) {
            const sameCnnCandidate = CNN_CANDIDATE_LABELS.has(t.label) && CNN_CANDIDATE_LABELS.has(d.label);
            const sameTextCandidate = ['hinweis', 'ortstafel'].includes(t.label) && ['hinweis', 'ortstafel'].includes(d.label);
            if (t.matched || (t.label !== d.label && !sameCnnCandidate && !sameTextCandidate)) continue;
            const o = iou(t, d);
            if (o > bestIou) { bestIou = o; best = t; }
          }
          if (best) {
            best.matched = true; best.hits++; best.miss = 0;
            if (!best.cnnVerified && !best.textVerified) best.label = d.label;
            for (const k of ['x', 'y', 'w', 'h']) best[k] = best[k] * 0.5 + d[k] * 0.5;   // Box glätten
            best.conf = best.conf * 0.7 + d.conf * 0.3;
            if (d.state) best.state = d.state;
          } else tracks.push(Object.assign({ id: nextId++, hits: 1, miss: 0, matched: true }, d));
        }
        tracks.forEach(t => { if (!t.matched) t.miss++; });
        tracks = tracks.filter(t => t.miss <= maxMiss);
        return tracks.filter(t => t.hits >= minHits);
      }
    };
  }

  const api = { CONFIG, SIGNS, detect, createTracker, RED, BLUE, YELLOW };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  else root.SignDetector = api;
})(typeof window !== 'undefined' ? window : globalThis);
