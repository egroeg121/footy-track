import json, cv2, numpy as np, pandas as pd, os
HERE=os.path.dirname(os.path.abspath(__file__)); FD="/mnt/storage/footy_data/detection_dataset/roboflow_dataset_11"
fl=pd.read_csv(f"{HERE}/a_flipped_roles_U_beta1.csv"); fl=fl[(fl["gt"]=="referee")&(fl["rescored"]=="player")].head(12)
dets={}
for l in open(f"{HERE}/data/dets_cpu.jsonl"):
    r=json.loads(l); dets[r["image"]]=r
import sys
sys.path.insert(0,HERE)
tiles=[]
for _,r in fl.iterrows():
    q=np.array(dets[r.image]["q"],dtype=float); sc=q[:,4:]; top=sc.argmax(1)
    # candidate index differs after NMS; recover by matching feature-free: use the box with referee score == s_ref
    j=np.argmin(np.abs(q[:,10]-r.s_ref)); cx,cy,w,h=q[j,:4]
    split="test"; img=cv2.imread(f"{FD}/{split}/images/{r.image}"); H,W=img.shape[:2]
    c=img[max(int((cy-h/2)*H),0):int((cy+h/2)*H), max(int((cx-w/2)*W),0):int((cx+w/2)*W)]
    c=cv2.resize(c,(80,160)); cv2.putText(c,r.image.split("_")[3],(2,12),cv2.FONT_HERSHEY_SIMPLEX,.4,(0,255,255),1); tiles.append(c)
cv2.imwrite(f"{HERE}/a_lost_referees_montage.png", np.hstack(tiles))
