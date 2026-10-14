import csv
import json
import pickle
import random
import re
import tarfile
import time
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path

import numpy as np
import requests
import torch
import torch.nn as nn
from PIL import Image, ImageEnhance, ImageFilter, ImageOps
from torch.utils.data import DataLoader, Dataset


SEED = 41
INPUT = Path('/kaggle/input')
WORK = Path('/tmp/sign-binary-cnn-v2')
RAW = WORK / 'raw'
CACHE = WORK / 'images'
OUT = Path('/kaggle/working/sign-binary-cnn-v2-output')
IMAGE_SIZE = 32
MAX_PER_GTSRB_CLASS = 250
VALIDATION_FRACTION = 0.20
MILESTONES = {20, 50, 100, 150}
MAX_EPOCHS = 150
PATIENCE = 20
BATCH_SIZE = 256

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_num_threads(min(4, torch.get_num_threads()))
for p in (RAW, CACHE / 'positive', CACHE / 'negative_road', CACHE / 'negative_objects', CACHE / 'negative_synthetic', CACHE / 'negative_red', OUT / 'checkpoints'):
    p.mkdir(parents=True, exist_ok=True)


def image_paths(root):
    return [p for p in root.rglob('*') if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.ppm', '.bmp'}]


def dataset_roots(needle):
    return [p for p in INPUT.rglob('*') if p.is_dir() and needle in p.name.lower()]


def safe_rgb(source, box=None):
    with Image.open(source) as im:
        im = im.convert('RGB')
        if box:
            x1, y1, x2, y2 = box
            x1, x2 = sorted((max(0, min(im.width - 1, int(x1))), max(1, min(im.width, int(x2)))))
            y1, y2 = sorted((max(0, min(im.height - 1, int(y1))), max(1, min(im.height, int(y2)))))
            if x2 > x1 and y2 > y1:
                im = im.crop((x1, y1, x2, y2))
        return ImageOps.pad(im, (32, 32), color=(127, 127, 127), centering=(0.5, 0.5))


def build_positives():
    roots = list({p for key in ('gtsrb', 'german-traffic-sign') for p in dataset_roots(key)})
    print('Kaggle input roots:', [str(p) for p in INPUT.iterdir()], flush=True)
    rows_by_class = defaultdict(list)
    for root in roots:
        for ann in root.rglob('*.csv'):
            if 'test' in ann.name.lower() or any(part.lower() == 'test' for part in ann.parts):
                continue
            try:
                with ann.open(encoding='utf-8-sig', newline='') as f:
                    sample = f.read(4096)
                    delimiter = ';' if sample.count(';') > sample.count(',') else ','
                    f.seek(0)
                    for row in csv.DictReader(f, delimiter=delimiter):
                        fields = {str(k).strip().lower(): v for k, v in row.items() if k}
                        rel = fields.get('path') or fields.get('filename') or fields.get('image')
                        class_id = fields.get('classid') or fields.get('class_id') or fields.get('label')
                        if not rel or class_id is None:
                            continue
                        relpath = Path(rel.replace('\\', '/'))
                        candidates = (ann.parent / relpath, root / relpath, ann.parent.parent / relpath)
                        img = next((p for p in candidates if p.is_file()), None)
                        if img is None:
                            matches = list(root.rglob(relpath.name))
                            img = matches[0] if matches else None
                        if img is None:
                            continue
                        coords = [fields.get(k) for k in ('roi.x1', 'roi.y1', 'roi.x2', 'roi.y2')]
                        roi = tuple(float(v) for v in coords) if all(v is not None for v in coords) else None
                        rows_by_class[str(class_id)].append((img, roi))
            except Exception as e:
                print(f'SKIP GTSRB annotation {ann}: {e}', flush=True)
    if not rows_by_class:
        for root in roots:
            for img in image_paths(root):
                class_id = next((part for part in reversed(img.parts[:-1]) if part.isdigit() and 0 <= int(part) <= 42), None)
                if class_id is not None:
                    rows_by_class[class_id].append((img, None))
    if not rows_by_class:
        csvs = [str(p) for p in list(INPUT.rglob('*.csv'))[:20]]
        raise RuntimeError(f'GTSRB images/annotations not found. roots={roots}; CSVs={csvs}')

    positives = []
    for class_id in sorted(rows_by_class, key=lambda x: int(re.sub(r'\D', '', x) or 0)):
        items = rows_by_class[class_id]
        random.shuffle(items)
        for img, roi in items[:MAX_PER_GTSRB_CLASS]:
            try:
                cached = CACHE / 'positive' / f'{class_id}_{len(positives):06d}.png'
                safe_rgb(img, roi).save(cached)
                positives.append(cached)
            except Exception:
                continue
    random.shuffle(positives)
    print(f'GTSRB positive crops: {len(positives)} from {len(rows_by_class)} classes; cap={MAX_PER_GTSRB_CLASS}/class', flush=True)
    return positives


def red_blob_ratio(image):
    a = np.asarray(image.convert('RGB'), dtype=np.uint8)
    r, g, b = a[..., 0], a[..., 1], a[..., 2]
    mask = (r > 70) & (r > g * 1.22) & (r > b * 1.18) & (g < 220)
    return float(mask.mean())


def download_cifar_negatives(limit=9000):
    """Ordinary CIFAR objects/scenes, with red-heavy examples kept as hard negatives."""
    archive = RAW / 'cifar-10-python.tar.gz'
    if not archive.exists():
        url = 'https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz'
        print(f'Downloading CIFAR-10 negative images from {url}', flush=True)
        with requests.get(url, stream=True, timeout=120, headers={'User-Agent': 'SignScannerResearch/1.0'}) as response:
            response.raise_for_status()
            with archive.open('wb') as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk: f.write(chunk)
    ordinary, red = [], []
    with tarfile.open(archive, 'r:gz') as tar:
        members = [m for m in tar.getmembers() if re.search(r'/data_batch_[1-5]$', m.name)]
        for member in members:
            stream = tar.extractfile(member)
            if not stream: continue
            batch = pickle.load(stream, encoding='bytes')
            arr = batch[b'data'].reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
            for pixels in arr:
                image = Image.fromarray(pixels)
                is_red = 0.025 <= red_blob_ratio(image) <= 0.65 and len(red) < limit // 2
                pool = red if is_red else ordinary
                if len(ordinary) >= limit - limit // 2 and len(red) >= limit // 2: break
                target = (CACHE / ('negative_red' if is_red else 'negative_objects') / f'cifar_{len(pool):06d}.png')
                image.save(target); pool.append(target)
            if len(ordinary) >= limit - limit // 2 and len(red) >= limit // 2: break
    if len(ordinary) < 1000: raise RuntimeError(f'CIFAR-10 returned too few ordinary negatives: {len(ordinary)}')
    print(f'CIFAR negatives: ordinary={len(ordinary)}, red-heavy={len(red)}', flush=True)
    return ordinary, red


def build_synthetic_negatives(limit=6000):
    """Create varied colored blobs/noise without sign geometry or text."""
    out = CACHE / 'negative_synthetic'
    out.mkdir(parents=True, exist_ok=True)
    saved = []
    palette = [(230,35,35),(245,125,20),(240,210,25),(30,170,65),(25,100,230),(170,50,205),(30,30,30),(235,235,235)]
    for i in range(limit):
        im = Image.new('RGB', (64,64), tuple(random.randint(35,220) for _ in range(3)))
        from PIL import ImageDraw
        d = ImageDraw.Draw(im, 'RGBA')
        for _ in range(random.randint(3,14)):
            x,y=random.randint(-12,58),random.randint(-12,58)
            w,h=random.randint(3,30),random.randint(3,30)
            color=random.choice(palette)+(random.randint(65,220),)
            if random.random()<0.55: d.ellipse((x,y,x+w,y+h),fill=color)
            else: d.polygon([(x,y),(x+w,y+random.randint(-5,8)),(x+random.randint(0,w),y+h)],fill=color)
        im=im.filter(ImageFilter.GaussianBlur(random.uniform(0,2.2)))
        # Add pixel noise so these are not only clean geometric shapes.
        a=np.asarray(im,dtype=np.int16); noise=np.random.normal(0,random.uniform(8,38),a.shape)
        im=Image.fromarray(np.uint8(np.clip(a+noise,0,255))).resize((32,32),Image.Resampling.BILINEAR)
        path=out/f'synthetic_{i:06d}.png'; im.save(path); saved.append(path)
    print(f'Synthetic colored blob/noise negatives: {len(saved)}',flush=True)
    return saved

def iou_crop(crop, box):
    x, y, w, h = crop
    x1, y1 = max(x, box[0]), max(y, box[1])
    x2, y2 = min(x + w, box[2]), min(y + h, box[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area = w * h + max(0, box[2] - box[0]) * max(0, box[3] - box[1]) - inter
    return inter / max(1, area)


def build_road_negatives(limit=12000):
    roots = list({p for key in ('gtsdb', 'german-traffic-sign-detection') for p in dataset_roots(key)})
    if not roots:
        print('GTSDB source missing; using extra safety-sign hard negatives only', flush=True)
        return [], []
    root = roots[0]
    boxes_by_name = defaultdict(list)
    for xml in root.rglob('*.xml'):
        try:
            tree = ET.parse(xml).getroot()
            name = (tree.findtext('filename') or (xml.stem + '.jpg')).lower()
            for obj in tree.findall('.//object'):
                b = obj.find('bndbox')
                if b is not None:
                    boxes_by_name[name].append(tuple(int(float(b.findtext(k, '0'))) for k in ('xmin', 'ymin', 'xmax', 'ymax')))
        except Exception:
            pass
    for txt in root.rglob('*.txt'):
        try:
            with txt.open(encoding='utf-8-sig') as f:
                for row in csv.reader(f, delimiter=';'):
                    if len(row) >= 5 and row[0].lower().endswith(('.jpg', '.png', '.ppm')):
                        boxes_by_name[row[0].lower()].append(tuple(int(float(v)) for v in row[1:5]))
        except Exception:
            pass
    scenes = image_paths(root)
    random.shuffle(scenes)
    negatives, red_negatives = [], []
    for scene_idx, scene in enumerate(scenes):
        if len(negatives) + len(red_negatives) >= limit:
            break
        try:
            with Image.open(scene) as source:
                im = source.convert('RGB')
                width, height = im.size
                if min(width, height) < 64:
                    continue
                boxes = boxes_by_name.get(scene.name.lower(), boxes_by_name.get(scene.stem.lower(), []))
                for _ in range(30):
                    side = random.randint(max(40, min(width, height) // 10), max(41, min(width, height) // 3))
                    side = min(side, width, height)
                    x, y = random.randint(0, width - side), random.randint(0, height - side)
                    crop = (x, y, side, side)
                    if any(iou_crop(crop, b) > 0.0 for b in boxes):
                        continue
                    patch = im.crop((x, y, x + side, y + side))
                    target = CACHE / 'negative_road' / f'road_{scene_idx:04d}_{len(negatives):06d}.png'
                    ImageOps.pad(patch, (32, 32), color=(127, 127, 127)).save(target)
                    negatives.append(target)
                    if len(negatives) + len(red_negatives) >= limit:
                        break
        except Exception:
            continue
    print(f'GTSDB non-sign crops: ordinary={len(negatives)}, red-heavy={len(red_negatives)}', flush=True)
    return negatives, red_negatives


def split(items, fraction=VALIDATION_FRACTION):
    items = list(items)
    random.shuffle(items)
    n = max(1, int(len(items) * fraction))
    return items[n:], items[:n]


def split_by_group(items, group_of, fraction=VALIDATION_FRACTION):
    groups = defaultdict(list)
    for item in items:
        groups[group_of(item)].append(item)
    keys = list(groups)
    random.shuffle(keys)
    n = max(1, int(len(keys) * fraction))
    validation_keys = set(keys[:n])
    train_items = [item for key, values in groups.items() if key not in validation_keys for item in values]
    val_items = [item for key, values in groups.items() if key in validation_keys for item in values]
    return train_items, val_items


class BinaryImages(Dataset):
    def __init__(self, samples, augment):
        self.samples, self.augment = samples, augment

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, label = self.samples[index]
        with Image.open(path) as im:
            im = im.convert('RGB').resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
            if self.augment:
                if random.random() < 0.35:
                    im = im.rotate(random.uniform(-18, 18), resample=Image.Resampling.BILINEAR, fillcolor=(127, 127, 127))
                if random.random() < 0.15:
                    im = ImageOps.mirror(im)
                if random.random() < 0.75:
                    im = ImageEnhance.Brightness(im).enhance(random.uniform(0.7, 1.3))
                    im = ImageEnhance.Color(im).enhance(random.uniform(0.7, 1.3))
                if random.random() < 0.08:
                    im = im.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.1, 0.7)))
            arr = np.asarray(im, dtype=np.float32).transpose(2, 0, 1) / 255.0
        return torch.from_numpy(arr), torch.tensor(label, dtype=torch.long)


class DSBlock(nn.Module):
    def __init__(self, cin, cout, stride=1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(cin, cin, 3, stride, 1, groups=cin, bias=False),
            nn.BatchNorm2d(cin), nn.ReLU6(inplace=True),
            nn.Conv2d(cin, cout, 1, bias=False),
            nn.BatchNorm2d(cout), nn.ReLU6(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class TinySignBinary(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 12, 3, 2, 1, bias=False), nn.BatchNorm2d(12), nn.ReLU6(inplace=True),
            DSBlock(12, 16), DSBlock(16, 24, 2), DSBlock(24, 32, 2), DSBlock(32, 40, 2),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(40, 2)

    def forward(self, x):
        return self.classifier(self.features(x).flatten(1))


def metrics(model, loader, device):
    model.eval()
    correct, total, tp, tn, fp, fn = 0, 0, 0, 0, 0, 0
    loss_sum = 0.0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss_sum += criterion(logits, y).item() * len(y)
            pred = logits.argmax(1)
            total += len(y)
            correct += (pred == y).sum().item()
            tp += ((pred == 1) & (y == 1)).sum().item()
            tn += ((pred == 0) & (y == 0)).sum().item()
            fp += ((pred == 1) & (y == 0)).sum().item()
            fn += ((pred == 0) & (y == 1)).sum().item()
    return {
        'loss': loss_sum / max(1, total), 'accuracy': correct / max(1, total),
        'positive_recall': tp / max(1, tp + fn), 'negative_recall': tn / max(1, tn + fp),
        'positive_precision': tp / max(1, tp + fp), 'count': total,
    }


def save_checkpoint(model, epoch, score, path):
    torch.save({
        'epoch': epoch, 'val_accuracy': score, 'input_size': IMAGE_SIZE,
        'classes': ['not_traffic_sign', 'traffic_sign'], 'state_dict': model.state_dict(),
        'architecture': 'TinySignBinary v2 32px depthwise-separable CNN',
    }, path)


def main():
    positive = build_positives()
    road, road_red = build_road_negatives()
    ordinary, cifar_red = download_cifar_negatives()
    synthetic = build_synthetic_negatives()
    red = road_red + cifar_red

    pos_train, pos_val = split(positive)
    road_train, road_val = split_by_group(road, lambda p: p.stem.split('_')[1]) if road else ([], [])
    red_train, red_val = split(red)
    ordinary_train, ordinary_val = split(ordinary)
    synthetic_train, synthetic_val = split(synthetic)

    # Negatives deliberately exclude all warning/sign images.
    val_negatives = road_val + red_val + ordinary_val + synthetic_val
    val_count = min(len(pos_val), len(val_negatives))
    if val_count < 20:
        raise RuntimeError(f'Not enough validation images: positives={len(pos_val)}, negatives={len(val_negatives)}')
    random.shuffle(pos_val); random.shuffle(val_negatives)
    val_samples = [(p, 1) for p in pos_val[:val_count]] + [(p, 0) for p in val_negatives[:val_count]]

    train_negatives = []
    n = len(pos_train)
    red_count, road_count, synthetic_count = n // 4, n // 4, n // 4
    ordinary_count = n - red_count - road_count - synthetic_count
    pools = ((red_train or ordinary_train, red_count), (road_train or ordinary_train, road_count), (synthetic_train, synthetic_count), (ordinary_train, ordinary_count))
    for pool, count in pools:
        for _ in range(count):
            train_negatives.append(random.choice(pool))
    if len(pos_train) < 500:
        raise RuntimeError(f'Too few positive samples: {len(pos_train)}')
    train_samples = [(p, 1) for p in pos_train] + [(p, 0) for p in train_negatives]
    random.shuffle(train_samples)
    print(json.dumps({
        'positive_source': 'GTSRB', 'positive_train': len(pos_train), 'positive_validation': val_count,
        'negative_train': len(train_negatives), 'negative_train_warning_signs': 0,
        'negative_train_red_blobs': red_count, 'negative_train_road_backgrounds': road_count if road_train else 0,
        'negative_train_synthetic_blobs': synthetic_count, 'negative_train_ordinary_images': ordinary_count + (road_count if not road_train else 0),
        'negative_validation': val_count, 'validation_is_balanced': True,
        'input_resolution': f'{IMAGE_SIZE}x{IMAGE_SIZE}', 'seed': SEED,
    }, indent=2), flush=True)

    train_loader = DataLoader(BinaryImages(train_samples, augment=True), batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=2, pin_memory=False, drop_last=False)
    val_loader = DataLoader(BinaryImages(val_samples, augment=False), batch_size=512, shuffle=False,
                            num_workers=2, pin_memory=False)
    device = torch.device('cpu')
    model = TinySignBinary().to(device)
    params = sum(p.numel() for p in model.parameters())
    print(f'Model: TinySignBinary v2; parameters={params:,}; input={IMAGE_SIZE}x{IMAGE_SIZE}; device=CPU', flush=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.002, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS)
    criterion = nn.CrossEntropyLoss()
    best_acc, best_epoch, stale = 0.0, 0, 0
    log_path = OUT / 'training_log.csv'
    with log_path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['epoch', 'seconds', 'train_loss', 'val_loss', 'val_accuracy', 'positive_recall', 'negative_recall', 'positive_precision', 'learning_rate'])
        writer.writeheader()
        for epoch in range(1, MAX_EPOCHS + 1):
            start = time.perf_counter()
            model.train()
            train_loss, seen = 0.0, 0
            for x, y in train_loader:
                x, y = x.to(device), y.to(device)
                optimizer.zero_grad(set_to_none=True)
                logits = model(x)
                loss = criterion(logits, y)
                loss.backward()
                optimizer.step()
                train_loss += loss.item() * len(y)
                seen += len(y)
            scheduler.step()
            score = metrics(model, val_loader, device)
            elapsed = time.perf_counter() - start
            row = {
                'epoch': epoch, 'seconds': round(elapsed, 2), 'train_loss': round(train_loss / max(1, seen), 6),
                'val_loss': round(score['loss'], 6), 'val_accuracy': round(score['accuracy'], 6),
                'positive_recall': round(score['positive_recall'], 6), 'negative_recall': round(score['negative_recall'], 6),
                'positive_precision': round(score['positive_precision'], 6), 'learning_rate': optimizer.param_groups[0]['lr'],
            }
            writer.writerow(row); f.flush()
            print('EPOCH ' + json.dumps(row), flush=True)

            if score['accuracy'] > best_acc:
                best_acc, best_epoch, stale = score['accuracy'], epoch, 0
                save_checkpoint(model, epoch, best_acc, OUT / 'best.pt')
            else:
                stale += 1
            if epoch in MILESTONES:
                save_checkpoint(model, epoch, score['accuracy'], OUT / 'checkpoints' / f'checkpoint_epoch_{epoch}.pt')
                print(f'CHECKPOINT epoch={epoch} accuracy={score["accuracy"]:.6f}', flush=True)
            if stale >= PATIENCE:
                print(f'Early stopping at epoch {epoch}: no validation-accuracy improvement for {PATIENCE} epochs; best_epoch={best_epoch}', flush=True)
                break

    best = torch.load(OUT / 'best.pt', map_location=device, weights_only=False)
    model.load_state_dict(best['state_dict'])
    model.eval()
    try:
        torch.onnx.export(model, torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE), OUT / 'best.onnx',
                          input_names=['input'], output_names=['logits'], opset_version=12,
                          dynamic_axes={'input': {0: 'batch'}, 'logits': {0: 'batch'}})
        print('ONNX export: best.onnx', flush=True)
    except Exception as e:
        print(f'ONNX export skipped: {e}', flush=True)

    # CPU single-image latency for the final best checkpoint; no accelerator.
    sample = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
    with torch.no_grad():
        for _ in range(20): model(sample)
        times = []
        for _ in range(200):
            t0 = time.perf_counter(); model(sample); times.append((time.perf_counter() - t0) * 1000)
    summary = {
        'best_epoch': int(best['epoch']), 'best_validation_accuracy': float(best['val_accuracy']),
        'parameters': params, 'input_resolution': f'{IMAGE_SIZE}x{IMAGE_SIZE}',
        'cpu_latency_ms_median': float(np.median(times)), 'cpu_latency_ms_p95': float(np.percentile(times, 95)),
        'patience_epochs': PATIENCE, 'max_epochs': MAX_EPOCHS,
        'checkpoint_epochs_saved': [e for e in sorted(MILESTONES) if (OUT / 'checkpoints' / f'checkpoint_epoch_{e}.pt').exists()],
    }
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
    (OUT / 'training.log').write_text(
        f"best_epoch={summary['best_epoch']}\nbest_validation_accuracy={summary['best_validation_accuracy']:.6f}\n"
        f"parameters={params}\ninput_resolution={IMAGE_SIZE}x{IMAGE_SIZE}\n"
        f"cpu_latency_ms_median={summary['cpu_latency_ms_median']:.4f}\n"
        f"cpu_latency_ms_p95={summary['cpu_latency_ms_p95']:.4f}\n"
        f"patience_epochs={PATIENCE}\ncheckpoints={summary['checkpoint_epochs_saved']}\n", encoding='utf-8')
    print('FINAL ' + json.dumps(summary), flush=True)


if __name__ == '__main__':
    main()
