import csv, json, random, time, xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

SEED=73; random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
INPUT=Path('/kaggle/input'); OUT=Path('/kaggle/working/learnable-sign-box-head-v4')
OUT.mkdir(parents=True,exist_ok=True); torch.set_num_threads(2)
SIZE=480; MAX_EPOCHS=200; PATIENCE=100; BATCH=512; FEATURE_COUNT=18

def roots_for(key): return [p for p in INPUT.rglob('*') if p.is_dir() and key in p.name.lower()]
def image_files(root): return [p for p in root.rglob('*') if p.suffix.lower() in {'.jpg','.jpeg','.png','.ppm','.bmp'}]
def parse_gtsdb():
    roots=roots_for('gtsdb') or roots_for('german-traffic-sign-detection')
    if not roots: raise RuntimeError('GTSDB input not mounted')
    root=roots[0]; ann={}
    for xml in root.rglob('*.xml'):
        try:
            t=ET.parse(xml).getroot(); name=(t.findtext('filename') or (xml.stem+'.jpg')).lower(); boxes=[]
            for o in t.findall('.//object'):
                b=o.find('bndbox')
                if b is not None: boxes.append(tuple(float(b.findtext(k,'0')) for k in ('xmin','ymin','xmax','ymax')))
            ann[name]=boxes; ann[Path(name).stem]=boxes
        except Exception: pass
    for txt in root.rglob('*.txt'):
        try:
            with txt.open(encoding='utf-8-sig') as f:
                for row in csv.reader(f,delimiter=';'):
                    if len(row)>=5 and row[0].lower().endswith(('.jpg','.png','.ppm')): ann.setdefault(row[0].lower(),[]).append(tuple(map(float,row[1:5])))
        except Exception: pass
    files=image_files(root); random.shuffle(files)
    samples=[]
    for p in files:
        boxes=ann.get(p.name.lower(),ann.get(p.stem.lower(),[]))
        if boxes: samples.append((p,boxes))
        elif p.suffix.lower() in {'.jpg','.jpeg','.png'}: samples.append((p,[]))
    print(f'GTSDB scenes: {len(samples)}, positive scenes: {sum(bool(b) for _,b in samples)}',flush=True)
    return samples

def parse_gtsrb_rois():
    """Use the official GTSRB ROI annotations as additional positive boxes."""
    samples=[]; seen=set()
    for root in roots_for('gtsrb'):
        for table in root.rglob('GT-*.csv'):
            try:
                with table.open(encoding='utf-8-sig',newline='') as f:
                    for row in csv.DictReader(f,delimiter=';'):
                        row={str(k).strip().lower():str(v).strip() for k,v in row.items() if k is not None}
                        name=row.get('filename') or row.get('image') or row.get('path')
                        if not name: continue
                        path=(table.parent/name).resolve()
                        if not path.exists():
                            matches=list(table.parent.rglob(Path(name).name))
                            if not matches: continue
                            path=matches[0]
                        key=str(path).lower()
                        if key in seen: continue
                        box=tuple(float(row[k]) for k in ('roi.x1','roi.y1','roi.x2','roi.y2'))
                        if box[2]>box[0] and box[3]>box[1]: samples.append((path,[box])); seen.add(key)
            except Exception as e: print('GTSRB ROI table skipped',table,str(e)[:100],flush=True)
    print(f'GTSRB ROI images discovered: {len(samples)}',flush=True)
    if len(samples)>6000: samples=random.sample(samples,6000)
    print(f'GTSRB ROI images sampled for CPU: {len(samples)}',flush=True)
    return samples

def hsv_mask(rgb):
    import cv2
    hsv=cv2.cvtColor(rgb,cv2.COLOR_RGB2HSV); h,s,v=cv2.split(hsv)
    red=((h<=11)|(h>=169))&(s>=55)&(v>=45)
    yellow=(h>=13)&(h<=38)&(s>=45)&(v>=55)
    blue=(h>=88)&(h<=137)&(s>=50)&(v>=35)
    green=(h>=37)&(h<=88)&(s>=55)&(v>=35)
    return red,blue,yellow,green,(red|blue|yellow|green)

def proposals(rgb):
    import cv2
    h,w=rgb.shape[:2]; masks=hsv_mask(rgb); union=(masks[-1].astype(np.uint8)*255)
    union=cv2.morphologyEx(union,cv2.MORPH_CLOSE,np.ones((7,7),np.uint8),iterations=1)
    n,labels,stats,cent=cv2.connectedComponentsWithStats(union,8)
    out=[]
    for i in range(1,n):
        x,y,bw,bh,area=map(int,stats[i])
        if area<10 or bw<5 or bh<5 or bw/bh<0.38 or bw/bh>5.2 or bw>w*.8 or bh>h*.8: continue
        out.append((x,y,x+bw,y+bh))
    # Also preserve per-color components when mixed-color gaps split a sign.
    for mask in masks[:4]:
        n,lab,st,_=cv2.connectedComponentsWithStats(mask.astype(np.uint8),8)
        for i in range(1,n):
            x,y,bw,bh,area=map(int,st[i])
            if area>=14 and bw>=6 and bh>=6 and .38<=bw/bh<=5.2 and bw<w*.8 and bh<h*.8:
                out.append((x,y,x+bw,y+bh))
    rb=cv2.morphologyEx((masks[0]|masks[1]).astype(np.uint8)*255,cv2.MORPH_CLOSE,np.ones((9,9),np.uint8),iterations=1)
    n,lab,st,_=cv2.connectedComponentsWithStats(rb,8)
    for i in range(1,n):
        x,y,bw,bh,area=map(int,st[i])
        if area>=20 and bw>=8 and bh>=8 and .62<=bw/bh<=1.55 and bw<w*.8 and bh<h*.8:
            out.append((x,y,x+bw,y+bh))
    # Add modest padding candidates around each base proposal to learn box extent.
    padded=[]
    for x1,y1,x2,y2 in out:
        bw=x2-x1; bh=y2-y1; cx=(x1+x2)/2; cy=(y1+y2)/2
        for scale in (.78,1.0,1.18,1.42):
            ww,hh=bw*scale,bh*scale
            padded.append((max(0,cx-ww/2),max(0,cy-hh/2),min(w,cx+ww/2),min(h,cy+hh/2)))
    return dedupe(padded)

def dedupe(boxes):
    out=[]
    for b in boxes:
        if not any(iou(b,a)>.88 for a in out): out.append(tuple(map(float,b)))
    return out[:240]
def iou(a,b):
    x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]); inter=max(0,x2-x1)*max(0,y2-y1)
    return inter/max(1,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter)
def feats(rgb,box):
    import cv2
    h,w=rgb.shape[:2]; x1,y1,x2,y2=map(int,box); x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
    if x2<=x1 or y2<=y1:return np.zeros(FEATURE_COUNT,np.float32)
    crop=rgb[y1:y2,x1:x2]; hh,ww=crop.shape[:2]; red,blue,yellow,green,mask=hsv_mask(crop)
    area=max(1,hh*ww); aspect=ww/max(1,hh); ys,xs=np.where(mask)
    solidity=len(xs)/area
    color=[np.count_nonzero(z)/area for z in (red,blue,yellow,green)]
    gray=cv2.cvtColor(crop,cv2.COLOR_RGB2GRAY); edges=cv2.Canny(gray,60,150)
    hsv=cv2.cvtColor(crop,cv2.COLOR_RGB2HSV)
    contour=None
    if len(xs):
        pts=np.column_stack([xs,ys]).astype(np.int32).reshape(-1,1,2); contour=cv2.convexHull(pts)
    hull_area=cv2.contourArea(contour) if contour is not None else 0
    compact=len(xs)/max(1,hull_area)
    ar=min(aspect,1/aspect)
    feats=np.array([
      min(1,ww/w),min(1,hh/h),np.clip(np.log(max(aspect,1e-4))/2,-1,1),
      min(1,(ww*hh)/(w*h)*30),solidity,compact,
      min(1,np.mean(hsv[:,:,1])/255),min(1,np.mean(hsv[:,:,2])/255),
      *color,min(1,np.count_nonzero(edges)/area),
      min(1,abs(1-aspect)/4),min(1,abs(solidity-.785)*2),
      min(1,abs(solidity-.5)*2), min(1,(color[0]*color[1])*8),
      min(1,np.count_nonzero(mask)/area)
    ],dtype=np.float32)
    return feats

class Head(nn.Module):
    def __init__(self):
        super().__init__(); self.a=nn.Linear(FEATURE_COUNT,32); self.b=nn.Linear(32,5)
    def forward(self,x): return self.b(torch.tanh(self.a(x)))

def load_data(samples):
    train_rows=[]; val_rows=[]; train_boxes=[]; val_gt={}; base_recalls=[]
    random.shuffle(samples); cut=max(1,int(len(samples)*.8)); tr=samples[:cut]; va=samples[cut:]
    def collect(rows, split, boxes):
        for idx,(path,gts) in enumerate(rows):
            try:
                with Image.open(path) as im:
                    ow,oh=im.size; scale=SIZE/max(ow,oh)
                    nw,nh=max(1,round(ow*scale)),max(1,round(oh*scale))
                    rgb=np.asarray(im.convert('RGB').resize((nw,nh)),dtype=np.uint8)
                # Keep aspect ratio: box shape and color geometry are input features.
                gt=[(b[0]*scale,b[1]*scale,b[2]*scale,b[3]*scale) for b in gts]
                cand=proposals(rgb)
                if split=='train':
                    # GT jitter creates localization supervision when color masks fragment.
                    for b in gt:
                        x1,y1,x2,y2=b; bw=x2-x1;bh=y2-y1;cx=(x1+x2)/2;cy=(y1+y2)/2
                        cand.extend([(max(0,cx-bw*sx/2+dx*bw),max(0,cy-bh*sy/2+dy*bh),min(SIZE,cx+bw*sx/2+dx*bw),min(SIZE,cy+bh*sy/2+dy*bh))
                                     for sx,sy,dx,dy in ((1,1,0,0),(.86,.86,0,0),(1.16,1.16,0,0),(1,1,.08,0),(1,1,-.08,0),(1,1,0,.08),(1,1,0,-.08))])
                cand=dedupe(cand)
                for b in cand:
                    ov=max((iou(b,g) for g in gt),default=0)
                    label=1 if ov>=.28 else (0 if ov<.10 else -1)
                    if label<0: continue
                    f=feats(rgb,b); target=np.zeros(5,np.float32); target[0]=label
                    if label:
                        g=max(gt,key=lambda z:iou(b,z)); bw=max(1,b[2]-b[0]);bh=max(1,b[3]-b[1])
                        target[1]=( (g[0]+g[2]-b[0]-b[2])/2 )/bw
                        target[2]=( (g[1]+g[3]-b[1]-b[3])/2 )/bh
                        target[3]=np.log(max(1,g[2]-g[0])/bw);target[4]=np.log(max(1,g[3]-g[1])/bh)
                    (train_rows if split=='train' else val_rows).append((idx,f,target,b,gt))
                if split=='val' and gt:
                    val_gt[idx]=gt; base_recalls.extend([max((iou(b,g) for b in cand),default=0) for g in gt])
                elif split=='val': val_gt[idx]=[]
            except Exception as e: print('SKIP',path,str(e)[:120],flush=True)
    collect(tr,'train',train_boxes); collect(va,'val',[])
    return train_rows,val_rows,val_gt,base_recalls

def evaluate(model, rows, mean, std, all_gt):
    model.eval(); grouped={gid:{'gt':gts,'preds':[]} for gid,gts in all_gt.items()}
    with torch.no_grad():
        for gid,f,t,b,gts in rows:
            x=torch.tensor(((f-mean)/std)[None],dtype=torch.float32); out=model(x)[0].numpy(); score=1/(1+np.exp(-np.clip(out[0],-30,30)))
            bw=b[2]-b[0];bh=b[3]-b[1];cx=(b[0]+b[2])/2+out[1]*bw;cy=(b[1]+b[3])/2+out[2]*bh
            ww=bw*np.exp(np.clip(out[3],-.6,.6));hh=bh*np.exp(np.clip(out[4],-.6,.6))
            pb=(max(0,cx-ww/2),max(0,cy-hh/2),cx+ww/2,cy+hh/2)
            grouped.setdefault(gid,{'gt':gts,'preds':[]})['preds'].append((score,pb))
    best=(0,0,0,0)
    for threshold in np.arange(.05,.96,.025):
        tp=fp=fn=0
        for item in grouped.values():
            chosen=[]
            for score,box in sorted(item['preds'],key=lambda p:p[0],reverse=True):
                if score<threshold: continue
                if any(iou(box,old)>.50 for old in chosen): continue
                chosen.append(box)
            unmatched=set(range(len(item['gt'])))
            for box in chosen:
                candidates=sorted(((iou(box,item['gt'][j]),j) for j in unmatched),reverse=True)
                if candidates and candidates[0][0]>=.50:
                    tp+=1;unmatched.remove(candidates[0][1])
                else: fp+=1
            fn+=len(unmatched)
        precision=tp/max(1,tp+fp);recall=tp/max(1,tp+fn);f1=2*precision*recall/max(1e-9,precision+recall)
        if f1>best[0]:best=(f1,threshold,precision,recall)
    return {'val_f1_iou50':best[0],'threshold':best[1],'precision_iou50':best[2],'recall_iou50':best[3], 'val_images':len(all_gt)}

def main():
    samples=parse_gtsdb(); samples.extend(parse_gtsrb_rois()); train,val,val_gt,base_recalls=load_data(samples)
    if len(train)<100 or len(val)<20: raise RuntimeError(f'not enough candidates train={len(train)} val={len(val)}')
    X=np.stack([r[1] for r in train]); mean=X.mean(0); std=X.std(0)+1e-5
    y=np.stack([r[2] for r in train]); xt=torch.tensor((X-mean)/std,dtype=torch.float32); yt=torch.tensor(y,dtype=torch.float32)
    loader=DataLoader(TensorDataset(xt,yt),batch_size=BATCH,shuffle=True,num_workers=0)
    model=Head(); opt=torch.optim.AdamW(model.parameters(),lr=.003,weight_decay=.002)
    scheduler=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=120,eta_min=1e-5)
    posw=torch.tensor([1,0,0,0,0],dtype=torch.float32); best=0.;best_epoch=0;stale=0;log=[]
    pos_weight=torch.tensor([min(8.,max(1.,(len(y)-y[:,0].sum())/max(1,y[:,0].sum())))],dtype=torch.float32)
    print(json.dumps({'train_candidates':len(train),'val_candidates':len(val),'val_images':len(val_gt),'val_gt_boxes':sum(len(x) for x in val_gt.values()),'proposal_recall_iou50':float(np.mean(np.asarray(base_recalls)>=.5)) if base_recalls else 0,'train_positive':int(y[:,0].sum()),'train_negative':int(len(y)-y[:,0].sum()),'parameters':sum(p.numel() for p in model.parameters()),'architecture':'Multi-color morphology proposals + 18-32-5 head; 773 parameters','target_val_f1_iou50':0.90,'cpu':True}),flush=True)
    for epoch in range(1,MAX_EPOCHS+1):
        model.train(); total=0
        for x,t in loader:
            opt.zero_grad(set_to_none=True); out=model(x)
            loss=nn.functional.binary_cross_entropy_with_logits(out[:,0],t[:,0],pos_weight=pos_weight)
            pos=t[:,0]>.5
            if pos.any(): loss=loss+.4*nn.functional.smooth_l1_loss(out[pos,1:],t[pos,1:])
            loss.backward(); opt.step(); total+=float(loss.item())
        scheduler.step()
        metric=evaluate(model,val,mean,std,val_gt); row={'epoch':epoch,'train_loss':round(total/max(1,len(loader)),5),**metric};log.append(row)
        row['learning_rate']=opt.param_groups[0]['lr']
        print('EPOCH '+json.dumps(row),flush=True)
        if metric['val_f1_iou50']>best:
            best=metric['val_f1_iou50'];best_epoch=epoch;stale=0;torch.save({'state_dict':model.state_dict(),'mean':mean,'std':std,'epoch':epoch,'metrics':metric},OUT/'best.pt')
        else: stale+=1
        if epoch in (20,50,80,100,150,200): torch.save({'state_dict':model.state_dict(),'epoch':epoch},OUT/f'checkpoint_epoch_{epoch}.pt')
        if stale>=PATIENCE: print(f'EARLY_STOP epoch={epoch} best_epoch={best_epoch}',flush=True);break
    bestck=torch.load(OUT/'best.pt',map_location='cpu',weights_only=False);model.load_state_dict(bestck['state_dict']);metric=evaluate(model,val,mean,std,val_gt)
    a,b=model.a,model.b
    metric['proposal_recall_iou50']=float(np.mean(np.asarray(base_recalls)>=.5)) if base_recalls else 0
    payload={'feature_count':FEATURE_COUNT,'hidden_units':32,'mean':mean.tolist(),'std':std.tolist(),'w1':a.weight.detach().numpy().tolist(),'b1':a.bias.detach().numpy().tolist(),'w2':b.weight.detach().numpy().tolist(),'b2':b.bias.detach().numpy().tolist(),'threshold':metric['threshold'],'box_offset_clip':.6,'metrics':metric,'best_epoch':bestck['epoch'],'parameters':sum(p.numel() for p in model.parameters()),'architecture':'Multi-color morphology proposals + 18-32-5 learned score/regression head'}
    (OUT/'box-head.json').write_text(json.dumps(payload,indent=2),encoding='utf-8')
    (OUT/'training.log').write_text('\n'.join(json.dumps(r) for r in log)+'\nFINAL '+json.dumps(payload),encoding='utf-8')
    print('FINAL '+json.dumps({k:v for k,v in payload.items() if k not in ('w1','w2','b1','b2','mean','std')}),flush=True)

if __name__=='__main__': main()

