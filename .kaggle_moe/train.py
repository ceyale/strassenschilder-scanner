"""GPU training: YOLO26n scene detector + sparse, lightweight GTSRB MoE classifier.

Kaggle inputs: GTSDB (Pascal VOC; includes unannotated/no-sign scenes) and GTSRB.
Outputs: /kaggle/working/models and training_report.json.
"""
import csv, json, os, random, shutil, time, xml.etree.ElementTree as ET
from pathlib import Path

SEED = 37
random.seed(SEED)
ROOT = Path('/kaggle/working/sign_moe')
INPUT = Path('/kaggle/input')
MODEL_DIR = Path('/kaggle/working/models')
IMG_SIZE = 640
DETECT_EPOCHS = int(os.getenv('DETECT_EPOCHS', '80'))
CLASS_EPOCHS = int(os.getenv('CLASS_EPOCHS', '12'))
GTSRB_NAMES = [
 'speed20','speed30','speed50','speed60','speed70','speed80','endSpeed80','speed100','speed120',
 'noPassing','noPassingTrucks','rightOfWayNext','priorityRoad','giveWay','stop','noVehicles',
 'trucksProhibited','noEntry','warning','curveLeft','curveRight','doubleCurve','bumpyRoad',
 'slipperyRoad','roadNarrows','roadWork','trafficSignals','pedestrians','children','bicycles',
 'snow','wildAnimals','endRestrictions','turnRight','turnLeft','straight','straightOrRight',
 'straightOrLeft','keepRight','keepLeft','roundabout','endNoPassing','endNoPassingTrucks']
# Sparse routing: one gate selects exactly one fine-grained sign expert.
GROUPS = {
 'speed': [*range(0,9),32], 'prohibition': [9,10,15,16,17,41,42],
 'priority': [11,12,13,14], 'warning': list(range(18,32)),
 'direction': list(range(33,41)),
}
CLASS_GROUP = {c:g for g, ids in enumerate(GROUPS.values()) for c in ids}

def yolo_box(box, w, h):
    try: x1,y1,x2,y2 = map(float, box)
    except (TypeError, ValueError): return None
    x1,x2 = sorted((max(0,min(w,x1)),max(0,min(w,x2))))
    y1,y2 = sorted((max(0,min(h,y1)),max(0,min(h,y2))))
    if x2 <= x1 or y2 <= y1: return None
    return f'0 {(x1+x2)/(2*w):.6f} {(y1+y2)/(2*h):.6f} {(x2-x1)/w:.6f} {(y2-y1)/h:.6f}'

def prepare_detector():
    from PIL import Image
    ds = ROOT/'detector'; counts={'train':0,'val':0,'negative':0}
    roots=[p for p in INPUT.rglob('*') if p.is_dir() and ('gtsdb' in p.name.lower() or 'traffic-sign-detection' in p.name.lower())]
    if not roots: raise RuntimeError('GTSDB input not found under /kaggle/input')
    root=roots[0]; anns={}
    for xml in root.rglob('*.xml'):
        try:
            node=ET.parse(xml).getroot(); name=(node.findtext('filename') or xml.stem).lower()
            boxes=[]
            for obj in node.findall('.//object'):
                b=obj.find('bndbox')
                if b is not None: boxes.append(tuple(b.findtext(k) for k in ('xmin','ymin','xmax','ymax')))
            anns[name]=boxes; anns[Path(name).stem]=boxes
        except Exception as e: print('skip annotation',xml,e)
    for txt in root.rglob('*.txt'):
        try:
            with txt.open(encoding='utf-8-sig') as f:
                for row in csv.reader(f,delimiter=';'):
                    if len(row)>=5 and row[0].lower().endswith(('.jpg','.png','.ppm')):
                        anns.setdefault(row[0].lower(),[]).append(tuple(row[1:5]))
        except Exception: pass
    imgs=[p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.ppm','.bmp'}]
    random.shuffle(imgs)
    for i,p in enumerate(imgs):
        split='val' if i%5==0 else 'train'
        try:
            with Image.open(p) as im:
                im=im.convert('RGB'); w,h=im.size
                boxes=anns.get(p.name.lower(),anns.get(p.stem.lower(),[]))
                lines=[s for b in boxes if (s:=yolo_box(b,w,h))]
                stem=f'gtsdb_{i:06d}'; (ds/'images'/split).mkdir(parents=True,exist_ok=True)
                (ds/'labels'/split).mkdir(parents=True,exist_ok=True)
                im.save(ds/'images'/split/(stem+'.jpg'),quality=92)
                (ds/'labels'/split/(stem+'.txt')).write_text('\n'.join(lines),encoding='utf8')
                counts[split]+=1; counts['negative']+=not bool(lines)
        except Exception as e: print('skip image',p,e)
    data=ds/'data.yaml'; data.write_text(f"path: {ds}\ntrain: images/train\nval: images/val\nnc: 1\nnames: ['traffic_sign']\n",encoding='utf8')
    return data,counts

def read_gtsrb():
    rows=[]
    for root in INPUT.rglob('*'):
        if not root.is_dir() or not ('gtsrb' in root.name.lower() or 'german-traffic-sign' in root.name.lower()): continue
        for ann in root.rglob('*.csv'):
            try:
                with ann.open(encoding='utf-8-sig',newline='') as f:
                    sample=f.read(2048); delim=';' if sample.count(';')>sample.count(',') else ','; f.seek(0)
                    for row in csv.DictReader(f,delimiter=delim):
                        d={k.strip().lower():v for k,v in row.items() if k}
                        key=next((d[k] for k in ('classid','class_id','label') if k in d),None)
                        name=next((d[k] for k in ('path','filename','image') if k in d),None)
                        if key is None or name is None: continue
                        try: cls=int(key)
                        except ValueError: continue
                        if not 0<=cls<43: continue
                        candidates=[ann.parent/name,root/name]
                        img=next((x for x in candidates if x.is_file()),None)
                        if img is None:
                            matches=list(root.rglob(Path(name).name)); img=matches[0] if matches else None
                        if img: rows.append((img,cls))
            except Exception as e: print('skip csv',ann,e)
    # de-duplicate repeated CSV references; keep source images out of validation leakage
    return list({(str(p),c):(p,c) for p,c in rows}.values())

def train_moe(rows, device):
    import torch
    from torch import nn
    from torch.utils.data import Dataset, DataLoader
    from torchvision import models, transforms
    from PIL import Image
    class Signs(Dataset):
        def __init__(self, items, train):
            self.items=items
            self.tf=transforms.Compose(([transforms.RandomAffine(12,translate=(.08,.08),scale=(.8,1.15)),transforms.ColorJitter(.25,.25,.2,.04)] if train else [])+[transforms.Resize((96,96)),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
        def __len__(self): return len(self.items)
        def __getitem__(self,i):
            p,c=self.items[i]
            with Image.open(p) as im: x=self.tf(im.convert('RGB'))
            return x,c
    random.shuffle(rows); val=[]; train=[]; seen={}
    for item in rows:
        c=item[1]; seen[c]=seen.get(c,0)+1
        (val if seen[c]%5==0 else train).append(item)
    if not train or not val: raise RuntimeError('GTSRB train/validation split is empty')
    def loader(items, aug, shuffle):
        return DataLoader(Signs(items,aug),batch_size=128,shuffle=shuffle,num_workers=2,pin_memory=True)
    def net(outputs):
        try: m=models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
        except Exception as e:
            print('pretrained MobileNet unavailable; training from random initialization:',e)
            m=models.mobilenet_v3_small(weights=None)
        m.classifier[-1]=nn.Linear(m.classifier[-1].in_features,outputs)
        return m.to(device)
    def fit(items, labels, epochs, name, val_items=None):
        label_to_local={c:i for i,c in enumerate(labels)}
        tr=[(p,label_to_local[c]) for p,c in items if c in label_to_local]
        validation = val if val_items is None else val_items
        va=[(p,label_to_local[c]) for p,c in validation if c in label_to_local]
        model=net(len(labels)); opt=torch.optim.AdamW(model.parameters(),lr=0.001,weight_decay=1e-4)
        best=0.0
        for ep in range(epochs):
            model.train()
            for x,y in loader(tr,True,True):
                x=x.to(device,non_blocking=True); y=y.to(device,non_blocking=True)
                opt.zero_grad(set_to_none=True); loss=nn.functional.cross_entropy(model(x),y,label_smoothing=.05); loss.backward(); opt.step()
            model.eval(); ok=n=0
            with torch.no_grad():
                for x,y in loader(va,False,False):
                    pred=model(x.to(device,non_blocking=True)).argmax(1).cpu(); ok+=int((pred==y).sum()); n+=len(y)
            acc=ok/max(1,n); print(f'{name} epoch {ep+1}/{epochs} val_acc={acc:.4f}',flush=True)
            if acc>=best:
                best=acc; torch.save({'state_dict':model.cpu().state_dict(),'labels':labels,'val_accuracy':best},MODEL_DIR/f'{name}.pt'); model.to(device)
        model.load_state_dict(torch.load(MODEL_DIR/f'{name}.pt',map_location=device,weights_only=False)['state_dict']); model.eval()
        dummy=torch.zeros(1,3,96,96,device=device)
        torch.onnx.export(model,dummy,MODEL_DIR/f'{name}.onnx',input_names=['images'],output_names=['logits'],dynamic_axes={'images':{0:'batch'},'logits':{0:'batch'}},opset_version=17)
        return best
    gate_items=[(p,CLASS_GROUP[c]) for p,c in train if c in CLASS_GROUP]
    gate_val=[(p,CLASS_GROUP[c]) for p,c in val if c in CLASS_GROUP]
    gate_labels=list(range(len(GROUPS))); metrics={'gate_accuracy':fit(gate_items,gate_labels,CLASS_EPOCHS,'sign_moe_gate',gate_val)}
    for g,(name,classes) in enumerate(GROUPS.items()):
        metrics[name+'_accuracy']=fit(train,classes,CLASS_EPOCHS,f'sign_moe_expert_{name}')
    (MODEL_DIR/'sign_classes.json').write_text(json.dumps({'gtsrb_classes':GTSRB_NAMES,'experts':GROUPS,'class_to_expert':CLASS_GROUP},indent=2),encoding='utf8')
    return metrics,{'train':len(train),'val':len(val),'classes':len(set(c for _,c in rows))}

def main():
    import torch
    MODEL_DIR.mkdir(parents=True,exist_ok=True)
    device=0 if torch.cuda.is_available() else 'cpu'
    print('device:',torch.cuda.get_device_name(0) if device==0 else 'CPU',flush=True)
    started=time.time(); data,det_counts=prepare_detector()
    from ultralytics import YOLO
    detector=YOLO('yolo26n.pt')
    detector.train(data=str(data),epochs=DETECT_EPOCHS,imgsz=IMG_SIZE,batch=16 if device==0 else 4,
        device=device,workers=4,patience=15,seed=SEED,project='/kaggle/working/runs',name='yolo26n_signs',
        exist_ok=True,cache=False,plots=False,close_mosaic=10,fliplr=0.0,mosaic=0.7,amp=True,verbose=True)
    best=Path('/kaggle/working/runs/yolo26n_signs/weights/best.pt')
    if not best.is_file(): raise RuntimeError(f'YOLO best weights missing: {best}')
    shutil.copy2(best,MODEL_DIR/'yolo26n_sign_detector.pt')
    class_rows=read_gtsrb(); print('GTSRB crops:',len(class_rows),flush=True)
    if not class_rows: raise RuntimeError('No GTSRB class crops found')
    moe_metrics,class_counts=train_moe(class_rows,device)
    report={'device':torch.cuda.get_device_name(0) if device==0 else 'CPU','detector':'YOLO26n','detector_epochs':DETECT_EPOCHS,
      'detector_data':det_counts,'classifier_data':class_counts,'moe_experts':list(GROUPS),'moe':moe_metrics,
      'negative_scene_images':det_counts['negative'],'seconds':round(time.time()-started,1),
      'limitations':['GTSRB has 43 cropped sign classes; no town-name OCR or traffic-light state labels.',
        'GTSDB scene annotations are used as class-agnostic sign boxes; more locally captured full-scene images are needed for broad detection quality.']}
    (Path('/kaggle/working')/'training_report.json').write_text(json.dumps(report,indent=2),encoding='utf8')
    print('TRAINING_REPORT',json.dumps(report),flush=True)

if __name__=='__main__': main()
