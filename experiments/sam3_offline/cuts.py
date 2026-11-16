"""Scene cuts per spec clip (mini CPU, ffmpeg 'scene' score on 640 px frames) -> adds ``cuts`` (processed-frame indices, first frame of the new shot) to the spec.

    OMP_NUM_THREADS=2 nice python cuts.py --spec spec.json --out spec_cuts.json [--thr 0.30]
"""
import argparse, json, re, subprocess
ap = argparse.ArgumentParser(); ap.add_argument("--spec"); ap.add_argument("--out"); ap.add_argument("--thr", type=float, default=0.30)
a = ap.parse_args()
spec = json.load(open(a.spec))
for name, c in spec["clips"].items():
    cmd = ["ffmpeg", "-v", "info", "-i", c["video"], "-fps_mode", "passthrough", "-vf", f"scale=640:-2,select='gt(scene,{a.thr})',showinfo", "-an", "-f", "null", "-"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    fr = sorted({int(m.group(1)) for m in re.finditer(r"n:\s*(\d+)\s.*?pts_time", r.stderr)})
    # showinfo 'n' counts selected frames; recover original index from pts_time * fps
    ts = [float(m.group(1)) for m in re.finditer(r"pts_time:\s*([\d.]+)", r.stderr)]
    c["cuts"] = sorted({int(round(t * c["fps_orig"])) // c["stride"] for t in ts})
    print(name, c["cuts"], flush=True)
json.dump(spec, open(a.out, "w"))
