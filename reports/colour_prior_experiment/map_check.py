import json, re, os, cv2, numpy as np, sys
cv2.setNumThreads(2)
FD="/mnt/storage/footy_data"
S=json.load(open(f"{FD}/arsenal_mancity/split_video_broadcast_frames/segments.json"))["segments"]
def seg_of(of):
    for s in S:
        if s["start_frame"]<=of<=s["end_frame"]: return s
    return None
td=f"{FD}/detection_dataset/roboflow_dataset_11/test/images"
fs=sorted(f for f in os.listdir(td) if re.match(r"arsenal_mancity_20250925_\d{6}_",f))
cov=0; out=[]
for f in fs:
    n=int(re.match(r"arsenal_mancity_20250925_(\d{6})_",f).group(1)); s=seg_of(25*n)
    out.append((f,n,s["index"] if s else None))
print("test frames",len(fs),"inside a broadcast segment (25n):",sum(1 for o in out if o[2] is not None))
# verify with image similarity for a few
import random
random.seed(1)
for f,n,si in random.sample([o for o in out if o[2] is not None],5):
    s=S[si]; ref=cv2.resize(cv2.imread(f"{td}/{f}"),(160,90)).astype(np.float32)
    cap=cv2.VideoCapture(f"{FD}/arsenal_mancity/split_video_broadcast_frames/{s['file']}")
    best=None
    off=25*n-s["start_frame"]
    for d in range(-30,31,2):
        k=max(off+d,0); cap.set(cv2.CAP_PROP_POS_FRAMES,k); ok,fr=cap.read()
        if not ok: continue
        e=np.abs(cv2.resize(fr,(160,90)).astype(np.float32)-ref).mean()
        if best is None or e<best[0]: best=(e,d)
    cap.release(); print(f,n,"seg",si,"best offset d(frames)",best[1],"err",round(best[0],1))
