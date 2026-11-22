"""Label-free anchors for the shared-rig search: ``shared_anchor`` of every Man City segment's middle frame, searched at the
GENERIC prior rig (no GT-derived rig anywhere).  Result: <ROOT>/shared/anchors_mancity_generic.json (resumable).
Run: PYTHONPATH=src nice -n 19 uv run --no-project --with numpy --with scipy --with opencv-python-headless python scripts/ptz_p5/anchors_generic.py"""
import json, sys, time
from pathlib import Path
from footy_track.calibration import ptz
from footy_track.scripts import ptz_temporal as T

root = T.ROOT / "shared"
out = root / "anchors_mancity_generic.json"
rig = ptz.Rig.from_json(sys.argv[1] if len(sys.argv) > 1 else "reports/ptz_v4_assets/generic_gantry_prior_v2.json")
store = json.loads(out.read_text()) if out.exists() else {}
segs = T.match_segments("mancity")
t0 = time.process_time()
for i, sg in enumerate(segs):
    for t in T.sample_frames(sg, 1):
        if f"{sg['file']}:{t}" in store:
            continue
        r = T.shared_anchor("mancity", sg, t, rig, store, root)
        print(i, sg["file"], t, r.get("ok"), r.get("f1"), f"cpu {time.process_time()-t0:.0f}s", flush=True)
    if i % 5 == 4:
        out.write_text(json.dumps(store))
out.write_text(json.dumps(store))
print("done", len(store), sum(v.get("ok", False) for v in store.values()))
