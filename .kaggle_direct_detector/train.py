import csv, json, random, shutil, subprocess, sys, time, xml.etree.ElementTree as ET
from pathlib import Path

try:
    import ultralytics
    from packaging.version import Version
    if Version(ultralytics.__version__) < Version('8.4.14'):
        raise ImportError('YOLO26 support needs a newer Ultralytics release')
    from ultralytics import YOLO
except (ImportError, ModuleNotFoundError):
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q', '--upgrade', 'ultralytics>=8.4.14'])
    from ultralytics import YOLO

SEED=73
random.seed(SEED)
ROOT=Path('/kaggle/input')
OUT=Path('/kaggle/working/german-sign-yolo26n')
DATA=Path('/kaggle/working/gtsdb-one-class')
OUT.mkdir(parents=True,exist_ok=True)

def find_dataset():
    matches=[p for p in ROOT.rglob('*') if p.is_dir() and 'gtsdb-dataset-in-pascal-voc-structure' in p.name.lower()]
    if not matches: matches=[p for p in ROOT.rglob('*') if p.is_dir() and 'gtsdb' in p.name.lower()]
    if not matches: raise RuntimeError('GTSDB dataset not mounted in Kaggle input')
    return matches[0]

def load_annotations(root):
    annotations={}
    for xml in root.rglob('*.xml'):
        try:
            node=ET.parse(xml).getroot()
            name=(node.findtext('filename') or '').strip().lower()
            if not name: name=xml.stem.lower()
            boxes=[]
            for obj in node.findall('.//object'):
                b=obj.find('bndbox')
                if b is not None:
                    xyxy=tuple(float(b.findtext(k,'0')) for k in ('xmin','ymin','xmax','ymax'))
                    if xyxy[2]>xyxy[0] and xyxy[3]>xyxy[1]: boxes.append(xyxy)
            annotations[Path(name).name]=boxes
        except Exception as e: print('skip annotation',xml,str(e)[:120],flush=True)
    for table in root.rglob('*.txt'):
        try:
            with table.open(encoding='utf-8-sig') as f:
                for row in csv.reader(f,delimiter=';'):
                    if len(row)>=5 and row[0].lower().endswith(('.jpg','.png','.ppm')):
                        annotations.setdefault(Path(row[0]).name.lower(),[]).append(tuple(map(float,row[1:5])))
        except Exception: pass
    return annotations

def prepare_data():
    root=find_dataset(); annotations=load_annotations(root)
    images=[]
    for p in root.rglob('*'):
        if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png','.ppm','.bmp'}:
            images.append(p)
    if not images: raise RuntimeError(f'No GTSDB images found under {root}')
    # Keep all unannotated scenes as hard negatives; split by image before copying.
    random.shuffle(images); cut=max(1,int(len(images)*.8)); splits={'train':images[:cut],'val':images[cut:]}
    counts={}
    for split,files in splits.items():
        image_dir=DATA/'images'/split; label_dir=DATA/'labels'/split
        image_dir.mkdir(parents=True,exist_ok=True); label_dir.mkdir(parents=True,exist_ok=True)
        for i,src in enumerate(files):
            name=f'{i:06d}{src.suffix.lower()}'; dst=image_dir/name
            shutil.copy2(src,dst)
            boxes=annotations.get(src.name.lower(),annotations.get(src.stem.lower(),[]))
            try:
                from PIL import Image
                with Image.open(src) as im: w,h=im.size
            except Exception as e:
                print('skip unreadable image',src,str(e)[:100],flush=True); dst.unlink(missing_ok=True); continue
            rows=[]
            for x1,y1,x2,y2 in boxes:
                x1=max(0,min(w,x1));x2=max(0,min(w,x2));y1=max(0,min(h,y1));y2=max(0,min(h,y2))
                if x2<=x1 or y2<=y1: continue
                rows.append(f'0 {(x1+x2)/(2*w):.7f} {(y1+y2)/(2*h):.7f} {(x2-x1)/w:.7f} {(y2-y1)/h:.7f}')
            (label_dir/f'{Path(name).stem}.txt').write_text('\n'.join(rows),encoding='utf-8')
        counts[split]={'images':len(files),'positive':sum(bool(annotations.get(p.name.lower(),annotations.get(p.stem.lower(),[]))) for p in files)}
    data_yaml=DATA/'data.yaml'
    data_yaml.write_text(f"path: {DATA}\ntrain: images/train\nval: images/val\nnames:\n  0: traffic-sign\n",encoding='utf-8')
    print('DATA '+json.dumps({'counts':counts,'annotations':sum(len(v) for v in annotations.values()),'yaml':str(data_yaml)}),flush=True)
    return data_yaml

def main():
    yaml=prepare_data()
    model=YOLO('yolo26n.pt') # COCO-pretrained one-stage detector; fine-tune to one generic sign class.
    start=time.time()
    model.train(data=str(yaml),epochs=100,imgsz=640,batch=4,device='cpu',workers=2,
        project=str(OUT),name='gtsdb-yolo26n',exist_ok=True,pretrained=True,patience=35,
        optimizer='AdamW',lr0=0.001,cos_lr=True,close_mosaic=10,mosaic=1.0,mixup=0.05,
        degrees=8,translate=0.1,scale=0.35,fliplr=0.5,cache=False,plots=False,save_period=20,verbose=True)
    best=OUT/'gtsdb-yolo26n'/'weights'/'best.pt'
    model=YOLO(str(best))
    metrics=model.val(data=str(yaml),imgsz=640,device='cpu',plots=False)
    summary={'map50':float(metrics.box.map50),'map50_95':float(metrics.box.map),'precision':float(metrics.box.mp),
        'recall':float(metrics.box.mr),'best_weights':str(best),'training_seconds':round(time.time()-start,1),
        'target_map50':0.90,'input_size':640,'architecture':'YOLO26n one-class direct detector, COCO-pretrained'}
    (OUT/'final-metrics.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print('FINAL '+json.dumps(summary),flush=True)
    try:
        model.export(format='onnx',imgsz=640,simplify=True,nms=False)
        print('ONNX export complete',flush=True)
    except Exception as e: print('ONNX export failed; best.pt retained:',repr(e),flush=True)

if __name__=='__main__': main()
