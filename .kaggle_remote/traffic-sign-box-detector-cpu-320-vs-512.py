import csv
import json
import os
import random
import shutil
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 17
random.seed(SEED)
np.random.seed(SEED)
ROOT = Path('/kaggle/working/traffic_sign_data')
INPUT = Path('/kaggle/input')
SYNSET_TRAIN_LIMIT = 5000
SYNSET_VAL_LIMIT = 1000
EPOCHS = 5


def yolo_line(box, width, height):
    x1, y1, x2, y2 = [float(v) for v in box]
    x1, x2 = sorted((max(0.0, min(width, x1)), max(0.0, min(width, x2))))
    y1, y2 = sorted((max(0.0, min(height, y1)), max(0.0, min(height, y2))))
    if x2 <= x1 or y2 <= y1:
        return None
    return f'0 {(x1+x2)/(2*width):.6f} {(y1+y2)/(2*height):.6f} {(x2-x1)/width:.6f} {(y2-y1)/height:.6f}'


def add_image(image_path, boxes, split, source, index):
    try:
        with Image.open(image_path) as im:
            im = im.convert('RGB')
            width, height = im.size
            if width < 16 or height < 16:
                return False
            target = ROOT / 'images' / split / f'{source}_{index:07d}.jpg'
            label = ROOT / 'labels' / split / f'{source}_{index:07d}.txt'
            target.parent.mkdir(parents=True, exist_ok=True)
            label.parent.mkdir(parents=True, exist_ok=True)
            im.save(target, quality=92)
        lines = [line for box in boxes if (line := yolo_line(box, width, height))]
        label.write_text('\n'.join(lines), encoding='utf-8')
        return True
    except Exception as exc:
        print(f'SKIP {image_path}: {exc}')
        return False


def parse_gtsdb():
    roots = [p for p in INPUT.glob('*') if 'gtsdb' in p.name.lower() or 'german-traffic-sign-detection' in p.name.lower()]
    if not roots:
        print('GTSDB input directory not found')
        return 0
    root = roots[0]
    annotations = {}
    for xml in root.rglob('*.xml'):
        try:
            tree = ET.parse(xml).getroot()
            filename = tree.findtext('filename') or (xml.stem + '.jpg')
            boxes = []
            for obj in tree.findall('.//object'):
                b = obj.find('bndbox')
                if b is not None:
                    boxes.append((b.findtext('xmin'), b.findtext('ymin'), b.findtext('xmax'), b.findtext('ymax')))
            annotations[filename.lower()] = boxes
            annotations[xml.stem.lower()] = boxes
        except Exception as exc:
            print('Bad GTSDB XML', xml, exc)
    for txt in root.rglob('*.txt'):
        try:
            for row in csv.reader(txt.open(encoding='utf-8-sig'), delimiter=';'):
                if len(row) >= 5 and row[0].lower().endswith(('.jpg', '.png', '.ppm')):
                    annotations.setdefault(row[0].lower(), []).append(tuple(row[1:5]))
        except Exception:
            pass
    images = [p for p in root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.ppm', '.bmp'}]
    random.shuffle(images)
    count = 0
    for i, path in enumerate(images):
        boxes = annotations.get(path.name.lower(), annotations.get(path.stem.lower(), []))
        split = 'val' if i % 5 == 0 else 'train'
        count += add_image(path, boxes, split, 'gtsdb', i)
    print(f'GTSDB: {len(images)} scene images, {count} used, {sum(not annotations.get(p.name.lower(), annotations.get(p.stem.lower(), [])) for p in images)} candidate negatives')
    return count


def parse_gtsrb():
    roots = [p for p in INPUT.glob('*') if 'gtsrb' in p.name.lower() or 'german-traffic-sign' in p.name.lower()]
    if not roots:
        print('GTSRB input directory not found')
        return 0
    files = []
    for root in roots:
        for ann in root.rglob('*.csv'):
            try:
                with ann.open(encoding='utf-8-sig', newline='') as f:
                    sample = f.read(4096)
                    delimiter = ';' if sample.count(';') > sample.count(',') else ','
                    f.seek(0)
                    rows = list(csv.DictReader(f, delimiter=delimiter))
                for row in rows:
                    lower = {k.strip().lower(): v for k, v in row.items() if k}
                    name = lower.get('filename') or lower.get('path') or lower.get('image')
                    keys = ('roi.x1', 'roi.y1', 'roi.x2', 'roi.y2')
                    if name and all(k in lower for k in keys):
                        img = ann.parent / name
                        if img.exists():
                            files.append((img, tuple(lower[k] for k in keys)))
            except Exception as exc:
                print('Skip CSV', ann, exc)
    random.shuffle(files)
    count = 0
    for i, (img, box) in enumerate(files):
        count += add_image(img, [box], 'val' if i % 5 == 0 else 'train', 'gtsrb', i)
    print(f'GTSRB: {len(files)} ROI images, {count} used (cropped-sign source)')
    return count


def add_synset(split_name, limit, target_split):
    from datasets import load_dataset
    stream = load_dataset('FraunhoferIOSB/Synset-Signset-Germany', 'OGRE', split=split_name, streaming=True)
    count = 0
    for i, row in enumerate(stream):
        if count >= limit:
            break
        image = row['image'].convert('RGB')
        mask = row['mask'].convert('RGB')
        arr = np.asarray(mask)
        # The sign mask is nonzero only on sign pixels; derive a tight rectangle.
        ys, xs = np.where(np.any(arr > 0, axis=2))
        if not len(xs):
            continue
        box = [(xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)]
        tmp = ROOT / f'_synset_{target_split}_{i}.jpg'
        image.save(tmp, quality=92)
        ok = add_image(tmp, box, target_split, 'synset', i)
        tmp.unlink(missing_ok=True)
        count += int(ok)
    print(f'Synset {split_name}: {count} images with mask-derived boxes')
    return count


def main():
    for p in [ROOT / 'images/train', ROOT / 'images/val', ROOT / 'labels/train', ROOT / 'labels/val']:
        p.mkdir(parents=True, exist_ok=True)
    n_gtsdb = parse_gtsdb()
    n_gtsrb = parse_gtsrb()
    n_syn_train = add_synset('train', SYNSET_TRAIN_LIMIT, 'train')
    n_syn_val = add_synset('validation', SYNSET_VAL_LIMIT, 'val')
    ntrain = len(list((ROOT / 'images/train').glob('*')))
    nval = len(list((ROOT / 'images/val').glob('*')))
    if not ntrain or not nval:
        raise RuntimeError(f'No usable train/validation data: train={ntrain}, val={nval}')
    data_yaml = ROOT / 'data.yaml'
    data_yaml.write_text(f"path: {ROOT.as_posix()}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['sign']\n", encoding='utf-8')
    print(json.dumps({'source_counts': {'gtsdb': n_gtsdb, 'gtsrb_roi': n_gtsrb, 'synset_train': n_syn_train, 'synset_val': n_syn_val}, 'train_images': ntrain, 'val_images': nval, 'negatives_in_gtsdb': True}, indent=2))

    from ultralytics import YOLO
    results = []
    for size in (320, 512):
        run = Path('/kaggle/working') / f'yolo11n-sign-{size}'
        print(f'\n=== START CPU TRAINING imgsz={size} ===', flush=True)
        model = YOLO('yolo11n.pt')
        started = time.perf_counter()
        metrics = model.train(data=str(data_yaml), imgsz=size, epochs=EPOCHS, batch=4, workers=2,
                               device='cpu', patience=3, seed=SEED, project='/kaggle/working',
                               name=run.name, exist_ok=True, cache=False, plots=False, verbose=True,
                               amp=False, close_mosaic=2, degrees=8, translate=0.08, scale=0.35,
                               fliplr=0.0, mosaic=0.5, mixup=0.0)
        train_sec = time.perf_counter() - started
        best = run / 'weights' / 'best.pt'
        final_model = YOLO(str(best if best.exists() else run / 'weights' / 'last.pt'))
        val_metrics = final_model.val(data=str(data_yaml), imgsz=size, device='cpu', workers=2, plots=False, verbose=False)
        sample = next((ROOT / 'images/val').glob('*'))
        for _ in range(3):
            final_model.predict(str(sample), imgsz=size, device='cpu', verbose=False)
        times = []
        val_imgs = list((ROOT / 'images/val').glob('*'))[:20]
        for img in val_imgs:
            t0 = time.perf_counter()
            final_model.predict(str(img), imgsz=size, device='cpu', verbose=False)
            times.append((time.perf_counter() - t0) * 1000)
        artifact = Path('/kaggle/working/models') / f'yolo11n-sign-{size}.pt'
        artifact.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(best if best.exists() else run / 'weights' / 'last.pt', artifact)
        row = {'imgsz': size, 'epochs': EPOCHS, 'train_seconds': round(train_sec, 1), 'val_images': len(val_imgs),
               'predict_ms_mean': round(float(np.mean(times)), 1), 'predict_ms_median': round(float(np.median(times)), 1),
               'map50': float(val_metrics.box.map50), 'map50_95': float(val_metrics.box.map),
               'precision': float(val_metrics.box.mp), 'recall': float(val_metrics.box.mr),
               'weights': str(artifact), 'weights_mb': round(artifact.stat().st_size / 1e6, 2)}
        results.append(row)
        print('PERFORMANCE ' + json.dumps(row), flush=True)
        del model, final_model
    out = Path('/kaggle/working/performance.json')
    out.write_text(json.dumps(results, indent=2), encoding='utf-8')
    print('FINAL_PERFORMANCE ' + json.dumps(results), flush=True)


if __name__ == '__main__':
    main()
