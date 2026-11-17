"""Run PnLCalib (single-view weights) zero-shot over a folder of frames (desktop GPU side).

    python run_pnlcalib.py --repo ~/exp/pnlcalib/PnLCalib --weights-dir ~/exp/pnlcalib/weights --frames DIR --manifest M.json --out results.jsonl

Research glue only: imports the PnLCalib clone (GPL-2.0) at run time from ``--repo``; nothing of it is copied here. Per image it
runs the keypoint + line networks once (resized to 960x540 like upstream ``inference.py``) and then four calibrations from the same
detections: camera parameters ``cam`` / ``cam_pnl`` (``heuristic_voting`` without / with ``refine_lines``) and the ground-plane
homography ``gh`` / ``gh_pnl`` (``heuristic_voting_ground``). Output: one JSON line per image (resumable: images already in the
output are skipped) with, per variant, ``P`` (3x4, pixels of the original image, pitch centred, z down) or ``Hinv`` (3x3 pixels ->
centred pitch metres), the voting ``rep_err`` and the seconds it took; plus the network seconds and the kp / line counts.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
import yaml
from PIL import Image


def projection(cp: dict) -> np.ndarray:
    Q = np.array([[cp["x_focal_length"], 0, cp["principal_point"][0]], [0, cp["y_focal_length"], cp["principal_point"][1]], [0, 0, 1]])
    It = np.eye(4)[:-1]
    It[:, -1] = -np.array(cp["position_meters"])
    return Q @ (np.array(cp["rotation_matrix"]) @ It)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", type=Path, required=True)
    ap.add_argument("--weights-dir", type=Path, required=True)
    ap.add_argument("--kp", default="SV_kp")
    ap.add_argument("--lines", default="SV_lines")
    ap.add_argument("--frames", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kp-threshold", type=float, default=0.3434)
    ap.add_argument("--line-threshold", type=float, default=0.7867)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()

    sys.path.insert(0, str(a.repo))
    import os
    os.chdir(a.repo)  # upstream loads config/*.yaml relative to the repo
    from model.cls_hrnet import get_cls_net
    from model.cls_hrnet_l import get_cls_net as get_cls_net_l
    from utils.utils_calib import FramebyFrameCalib
    from utils.utils_heatmap import (complete_keypoints, coords_to_dict, get_keypoints_from_heatmap_batch_maxpool,
                                     get_keypoints_from_heatmap_batch_maxpool_l)

    dev = a.device
    cfg = yaml.safe_load(open(a.repo / "config/hrnetv2_w48.yaml"))
    cfg_l = yaml.safe_load(open(a.repo / "config/hrnetv2_w48_l.yaml"))
    model = get_cls_net(cfg)
    model.load_state_dict(torch.load(a.weights_dir / a.kp, map_location=dev))
    model.to(dev).eval()
    model_l = get_cls_net_l(cfg_l)
    model_l.load_state_dict(torch.load(a.weights_dir / a.lines, map_location=dev))
    model_l.to(dev).eval()
    resize = T.Resize((540, 960))

    items = json.loads(a.manifest.read_text())
    if a.limit:
        items = items[: a.limit]
    done = set()
    if a.out.exists():
        for ln in a.out.read_text().splitlines():
            if ln.strip():
                r = json.loads(ln)
                done.add((r["clip"], r["frame"]))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print(f"{len(items)} images, {len(done)} done, device {dev}, torch {torch.__version__}", flush=True)

    # warm-up (cudnn autotune / first-call cost kept out of the timings)
    with torch.no_grad():
        z = torch.zeros(1, 3, 540, 960, device=dev)
        model(z)
        model_l(z)
    torch.cuda.synchronize()

    t_start = time.time()
    with open(a.out, "a") as fout:
        for n, it in enumerate(items):
            key = (it["clip"], it["frame"])
            if key in done:
                continue
            bgr = cv2.imread(str(a.frames / it["clip"] / f"{it['frame']:06d}.jpg"))
            H0, W0 = bgr.shape[:2]
            img = TF.to_tensor(Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))).float().unsqueeze(0)
            img = img if img.shape[-1] == 960 else resize(img)
            img = img.to(dev)
            _, _, h, w = img.shape
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            with torch.no_grad():
                hm = model(img)
                hml = model_l(img)
            kp_c = get_keypoints_from_heatmap_batch_maxpool(hm[:, :-1])
            ln_c = get_keypoints_from_heatmap_batch_maxpool_l(hml[:, :-1])
            torch.cuda.synchronize()
            t_net = time.perf_counter() - t0
            kp_d = coords_to_dict(kp_c, threshold=a.kp_threshold)
            ln_d = coords_to_dict(ln_c, threshold=a.line_threshold)
            kp_d, ln_d = complete_keypoints(kp_d[0], ln_d[0], w=w, h=h, normalize=True)
            rec = {"clip": it["clip"], "frame": it["frame"], "W": W0, "H": H0, "t_net": t_net, "n_kp": len(kp_d), "n_lines": len(ln_d), "variants": {}}
            for name, ground, refine in (("cam", False, False), ("cam_pnl", False, True), ("gh", True, False), ("gh_pnl", True, True)):
                cam = FramebyFrameCalib(iwidth=W0, iheight=H0, denormalize=True)
                cam.update(copy.deepcopy(kp_d), copy.deepcopy(ln_d))
                t1 = time.perf_counter()
                try:
                    res = cam.heuristic_voting_ground(refine_lines=refine) if ground else cam.heuristic_voting(refine_lines=refine)
                except Exception as e:  # noqa: BLE001 - upstream raises on degenerate configurations; record and move on
                    rec["variants"][name] = {"ok": False, "error": f"{type(e).__name__}: {e}"[:200], "t": time.perf_counter() - t1}
                    continue
                dt = time.perf_counter() - t1
                if res is None:
                    rec["variants"][name] = {"ok": False, "t": dt}
                elif ground:
                    rec["variants"][name] = {"ok": True, "Hinv": np.asarray(res["homography"]).tolist(), "rep_err": float(res["rep_err"]), "t": dt}
                else:
                    cp = res["cam_params"]
                    rec["variants"][name] = {"ok": True, "P": projection(cp).tolist(), "rep_err": float(res["rep_err"]), "mode": res["mode"],
                                             "focal": float(cp["x_focal_length"]), "t": dt}
            fout.write(json.dumps(rec) + "\n")
            fout.flush()
            if n % 20 == 0:
                print(f"{n + 1}/{len(items)} {it['clip']}:{it['frame']} net {t_net * 1000:.0f} ms, kp {len(kp_d)} lines {len(ln_d)}, elapsed {time.time() - t_start:.0f}s", flush=True)
    print("done", flush=True)


if __name__ == "__main__":
    main()
