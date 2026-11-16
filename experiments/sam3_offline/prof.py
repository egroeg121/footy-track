"""Backbone / per-frame timing on the RTX 2070 after the bf16 patch. Run through the sam3-script flow."""
import sys, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import torch
from sam3_track import Sam3Tracker

def tm(fn, n=4, warm=2):
    for _ in range(warm): fn()
    torch.cuda.synchronize(); t=time.time()
    for _ in range(n): fn()
    torch.cuda.synchronize(); return round((time.time()-t)/n, 4)

for patch in (True,):
    r = Sam3Tracker({"model": "sam3", "patch_turing": patch})
    bb = r.model.detector.backbone
    x = torch.randn(1, 3, 1008, 1008, device="cuda")
    with torch.inference_mode(), r.autocast():
        print("patch", patch, "backbone", tm(lambda: bb.forward_image(x)), flush=True)
        from torch.profiler import profile, ProfilerActivity
        with profile(activities=[ProfilerActivity.CUDA]) as p:
            bb.forward_image(x); torch.cuda.synchronize()
        print(p.key_averages().table(sort_by="cuda_time_total", row_limit=10, max_name_column_width=70), flush=True)
