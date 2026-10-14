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
    minSaturation: 0.4,
    minValue: 0.2,
    minBox: 8,
    minArea: 30,
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
    haltverbot:        { name: 'Haltverbot',        zeichen: 'Z 283',     hex: '#1467b8', text: '#fff',    note: 'Blauer Kreis mit rotem Rand und Schrägbalken.' },
    schildKandidat:    { name: 'Schildkandidat',   zeichen: 'Kandidat',  hex: '#a44fe8', text: '#fff',    note: 'Boxvorschlag des lernbaren Farb-/Formmodells.' },
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
    // Ortstafeln sind breite Rechtecke; Schrift und Rahmen brechen die gelbe Fläche
    // auf, deshalb darf deren Masken-Solidity deutlich unter 0.9 liegen.
    if (aspect >= 1.25 && s.solidity >= 0.55) return 'rect';
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
      diamond: [0.7, 1.4, 0.34, 0.67], rect: [1.2, 4.5, 0.55, 1.02]
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
    if (needsInteriorDetail && neutralShare < (comp.color === RED ? 0.12 : (comp.color === YELLOW ? 0.025 : 0.08))) return null;

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

  /** Ampeln auch erkennen, wenn gerade nur ein oder zwei Leuchten sichtbar sind. */
  function detectTrafficLights(comps, queue, data, w) {
    const lamps = comps.filter(c => {
      if (![RED, YELLOW, GREEN].includes(c.color)) return false;
      const aspect = c.w / c.h;
      if (aspect < 0.55 || aspect > 1.6 || c.area < 12) return false;
      const stats = analyzeShape(c, queue, w);
      return stats.solidity > 0.48 && stats.solidity < 0.99;
    }).sort((a, b) => a.y - b.y);
    const brightness = c => {
      let sum = 0;
      for (let k = c.start; k < c.end; k++) {
        const p = queue[k] * 4;
        sum += Math.max(data[p], data[p + 1], data[p + 2]);
      }
      return sum / Math.max(1, c.area);
    };
    const out = [];
    for (let i = 0; i < lamps.length; i++) {
      for (let j = i + 1; j < Math.min(lamps.length, i + 8); j++) {
        const a = lamps[i], b = lamps[j];
        if (a.color === b.color) continue;
        const pair = [a, b], centers = pair.map(c => c.x + c.w / 2);
        const sizes = pair.map(c => (c.w + c.h) / 2);
        const typical = (sizes[0] + sizes[1]) / 2;
        if (Math.abs(centers[0] - centers[1]) > typical * 0.75 ||
            Math.max(...sizes) / Math.max(1, Math.min(...sizes)) > 2.0) continue;
        const gap = b.y - (a.y + a.h);
        if (gap > typical * 3.5 || gap < -typical * 0.5) continue;

        // Prefer a matching third lamp when it is visible; otherwise paired lights
        // remain valid when one LED is off or occluded.
        let group = pair;
        for (let k = j + 1; k < Math.min(lamps.length, j + 8); k++) {
          const c = lamps[k];
          if ([a.color, b.color].includes(c.color)) continue;
          const center = c.x + c.w / 2, size = (c.w + c.h) / 2;
          if (Math.abs(center - (centers[0] + centers[1]) / 2) <= typical * 0.75 &&
              c.y - (b.y + b.h) <= typical * 3.5 && c.y >= b.y) { group = [a, b, c]; break; }
        }
        const active = group.reduce((best, c) => brightness(c) > brightness(best) ? c : best, group[0]);
        const names = { [RED]: 'Rot', [YELLOW]: 'Gelb', [GREEN]: 'Grün' };
        const left = Math.min(...group.map(c => c.x)), top = Math.min(...group.map(c => c.y));
        const right = Math.max(...group.map(c => c.x + c.w)), bottom = Math.max(...group.map(c => c.y + c.h));
        out.push({ label: 'ampel', state: names[active.color], x: left, y: top, w: right - left,
          h: bottom - top, conf: group.length === 3 ? 0.82 : 0.68 });
        break;
      }
      if (out.length) break;
    }
    return out;
  }

  function headPixelColor(r, g, b) {
    const mx=Math.max(r,g,b), mn=Math.min(r,g,b), d=mx-mn, v=mx/255, sat=mx?d/mx:0;
    if (sat < 55/255 || mx < 45) return NONE;
    let h=0;
    if (mx===r) h=60*(((g-b)/d+6)%6); else if (mx===g) h=60*((b-r)/d+2); else h=60*((r-g)/d+4);
    if (h<=22 || h>=338) return RED;
    if (h>=26 && h<=76) return YELLOW;
    if (h>=176 && h<=274) return BLUE;
    if (h>=74 && h<=176) return GREEN;
    return NONE;
  }

  function boxHeadFeatures(data, imageW, imageH, box) {
    const x1=Math.max(0,Math.floor(box.x)), y1=Math.max(0,Math.floor(box.y));
    const x2=Math.min(imageW,Math.ceil(box.x+box.w)), y2=Math.min(imageH,Math.ceil(box.y+box.h));
    const bw=Math.max(1,x2-x1), bh=Math.max(1,y2-y1), area=bw*bh;
    let count=0, satSum=0, valSum=0, edgeCount=0, colors=[0,0,0,0];
    const rowsMin=new Int32Array(bh).fill(bw), rowsMax=new Int32Array(bh).fill(-1);
    const colsMin=new Int32Array(bw).fill(bh), colsMax=new Int32Array(bw).fill(-1);
    const gray=new Uint8Array(area), colored=new Uint8Array(area);
    for(let yy=0;yy<bh;yy++) for(let xx=0;xx<bw;xx++) {
      const i=((y1+yy)*imageW+x1+xx)*4, r=data[i],g=data[i+1],b=data[i+2], mx=Math.max(r,g,b),mn=Math.min(r,g,b),c=headPixelColor(r,g,b);
      const p=yy*bw+xx; gray[p]=Math.round(.299*r+.587*g+.114*b);
      satSum+=mx?(mx-mn)/mx:0; valSum+=mx/255;
      if(c){ colored[p]=1;count++;colors[c===RED?0:c===BLUE?1:c===YELLOW?2:3]++; if(xx<rowsMin[yy])rowsMin[yy]=xx;if(xx>rowsMax[yy])rowsMax[yy]=xx;if(yy<colsMin[xx])colsMin[xx]=yy;if(yy>colsMax[xx])colsMax[xx]=yy; }
    }
    // The row/column extrema form a compact approximation of the convex hull.
    const pts=[];
    for(let y=0;y<bh;y++)if(rowsMax[y]>=0){pts.push([rowsMin[y],y],[rowsMax[y],y]);}
    for(let x=0;x<bw;x++)if(colsMax[x]>=0){pts.push([x,colsMin[x]],[x,colsMax[x]]);}
    pts.sort((a,b)=>a[0]-b[0]||a[1]-b[1]);
    const cross=(o,a,b)=>(a[0]-o[0])*(b[1]-o[1])-(a[1]-o[1])*(b[0]-o[0]);
    const hull=[];
    for(const q of pts){while(hull.length>=2&&cross(hull[hull.length-2],hull[hull.length-1],q)<=0)hull.pop();hull.push(q);}
    const lower=hull.length;
    for(let i=pts.length-2;i>=0;i--){const q=pts[i];while(hull.length>lower&&cross(hull[hull.length-2],hull[hull.length-1],q)<=0)hull.pop();hull.push(q);}
    let hullArea=0;for(let i=0;i<hull.length;i++){const a=hull[i],b=hull[(i+1)%hull.length];hullArea+=a[0]*b[1]-a[1]*b[0];}hullArea=Math.abs(hullArea)/2;
    for(let y=1;y<bh-1;y++)for(let x=1;x<bw-1;x++){const i=y*bw+x,gx=-gray[i-bw-1]+gray[i-bw+1]-2*gray[i-1]+2*gray[i+1]-gray[i+bw-1]+gray[i+bw+1],gy=-gray[i-bw-1]-2*gray[i-bw]-gray[i-bw+1]+gray[i+bw-1]+2*gray[i+bw]+gray[i+bw+1];if(Math.hypot(gx,gy)>210)edgeCount++;}
    const aspect=bw/bh, solidity=count/area, compact=count/Math.max(1,hullArea);
    return [Math.min(1,bw/imageW),Math.min(1,bh/imageH),Math.max(-1,Math.min(1,Math.log(Math.max(aspect,1e-4))/2)),Math.min(1,area/(imageW*imageH)*30),solidity,Math.min(1,compact),satSum/area,valSum/area,
      colors[0]/area,colors[1]/area,colors[2]/area,colors[3]/area,edgeCount/area,Math.min(1,Math.abs(1-aspect)/4),Math.min(1,Math.abs(solidity-.785)*2),Math.min(1,Math.abs(solidity-.5)*2),Math.min(1,colors[0]*colors[1]/(area*area)*8),solidity];
  }

  function runBoxHead(features, model) {
    const normalized=features.map((v,i)=>(v-model.mean[i])/model.std[i]), hidden=[];
    for(let j=0;j<model.hidden_units;j++){let x=model.b1[j];for(let i=0;i<normalized.length;i++)x+=model.w1[j][i]*normalized[i];hidden.push(Math.tanh(x));}
    const out=[];for(let k=0;k<5;k++){let x=model.b2[k];for(let j=0;j<hidden.length;j++)x+=model.w2[k][j]*hidden[j];out.push(x);}
    return {score:1/(1+Math.exp(-Math.max(-30,Math.min(30,out[0])))),offsets:out.slice(1)};
  }

  function applyLearnableBoxHead(data,w,h,detections,model,cfg) {
    if(!model||model.feature_count!==18)return detections;
    const headMask=new Uint8Array(w*h), colorMasks=[new Uint8Array(w*h),new Uint8Array(w*h),new Uint8Array(w*h),new Uint8Array(w*h)];
    for(let p=0,i=0;p<headMask.length;p++,i+=4){const c=headPixelColor(data[i],data[i+1],data[i+2]);if(c){headMask[p]=1;colorMasks[c===RED?0:c===BLUE?1:c===YELLOW?2:3][p]=c;}}
    const hc=Object.assign({},cfg,{minBox:5,minArea:10,maxFrameShare:.8,minAspect:.38,maxAspect:5.2});
    const raw=[];
    for(const m of [headMask,...colorMasks]){const found=findComponents(m,w,h,hc).comps;for(const c of found)raw.push({x:c.x,y:c.y,w:c.w,h:c.h});}
    const proposals=[];
    for(const b of raw)for(const scale of [1,1.18]){const ww=b.w*scale,hh=b.h*scale,cx=b.x+b.w/2,cy=b.y+b.h/2,q={x:Math.max(0,cx-ww/2),y:Math.max(0,cy-hh/2),w:Math.min(w,cx+ww/2)-Math.max(0,cx-ww/2),h:Math.min(h,cy+hh/2)-Math.max(0,cy-hh/2)};if(q.w>=5&&q.h>=5&&!proposals.some(a=>iou(a,q)>.88))proposals.push(q);if(proposals.length>=240)break;}
    const added=[];
    const scoreThreshold=Number.isFinite(cfg.learnableBoxThreshold)?cfg.learnableBoxThreshold:(model.threshold||.55);
    for(const b of proposals){const f=boxHeadFeatures(data,w,h,b),r=runBoxHead(f,model);if(r.score<scoreThreshold)continue;
      const bw=b.w,bh=b.h,cx=b.x+bw/2+r.offsets[0]*bw,cy=b.y+bh/2+r.offsets[1]*bh,ww=bw*Math.exp(Math.max(-.6,Math.min(.6,r.offsets[2]))),hh=bh*Math.exp(Math.max(-.6,Math.min(.6,r.offsets[3])));
      const refined={x:Math.max(0,cx-ww/2),y:Math.max(0,cy-hh/2),w:Math.min(w,cx+ww/2)-Math.max(0,cx-ww/2),h:Math.min(h,cy+hh/2)-Math.max(0,cy-hh/2)};
      const match=detections.find(d=>iou(d,refined)>.28);
      if(match){match.learnedBoxScore=r.score;match.x=match.x*.65+refined.x*.35;match.y=match.y*.65+refined.y*.35;match.w=match.w*.65+refined.w*.35;match.h=match.h*.65+refined.h*.35;continue;}
      if(added.some(a=>iou(a,refined)>.45))continue;
      added.push(Object.assign({label:'schildKandidat',shape:'candidate',conf:r.score,learnedBoxScore:r.score,learnedBox:true},refined));
    }
    return detections.concat(added);
  }

  // Haltverbot hat eine blaue Kreisfläche und ein rotes X. Die X-Balken zerlegen
  // die blaue Fläche in mehrere Komponenten; deshalb verbinden wir Rot und Blau
  // für diesen Schildtyp und prüfen die Farbanteile der gemeinsamen Box.
  function detectHaltverbotUnion(mask, w, h) {
    const seen = new Uint8Array(mask.length), queue = new Int32Array(mask.length), out = [];
    for (let start = 0; start < mask.length; start++) {
      if (seen[start] || (mask[start] !== RED && mask[start] !== BLUE)) continue;
      let head = 0, tail = 0, minX = w, minY = h, maxX = 0, maxY = 0, red = 0, blue = 0;
      queue[tail++] = start; seen[start] = 1;
      while (head < tail) {
        const p = queue[head++], x = p % w, y = (p / w) | 0, c = mask[p];
        if (x < minX) minX = x; if (x > maxX) maxX = x;
        if (y < minY) minY = y; if (y > maxY) maxY = y;
        if (c === RED) red++; else blue++;
        for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
          if (!dx && !dy) continue;
          const nx = x + dx, ny = y + dy;
          if (nx < 0 || nx >= w || ny < 0 || ny >= h) continue;
          const q = ny * w + nx, nc = mask[q];
          if (!seen[q] && (nc === RED || nc === BLUE)) { seen[q] = 1; queue[tail++] = q; }
        }
      }
      const bw = maxX - minX + 1, bh = maxY - minY + 1, boxArea = bw * bh, total = red + blue;
      if (bw < 12 || bh < 12 || bw / bh < .68 || bw / bh > 1.45 || boxArea > w * h * .30) continue;
      if (total / boxArea < .10 || red / total < .045 || blue / total < .12) continue;
      out.push({ label: 'haltverbot', shape: 'circle', x: minX, y: minY, w: bw, h: bh,
        conf: Math.min(.92, .68 + Math.min(red / total, blue / total) * .6) });
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
      if (conf >= 0.42) detections.push({ label, shape, x: c.x, y: c.y, w: c.w, h: c.h, conf });
    }
    detections.push(...detectHaltverbotUnion(pixels, w, h));
    detections.push(...detectTrafficLights(comps, queue, data, w));
    const learned = applyLearnableBoxHead(data,w,h,detections,options.learnableBoxHead,cfg);
    return { detections: suppressNestedDetections(learned), mask: pixels };
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
            for (const k of ['x', 'y', 'w', 'h']) best[k] = best[k] * 0.65 + d[k] * 0.35;   // stärker glätten gegen Kamerajittern
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
