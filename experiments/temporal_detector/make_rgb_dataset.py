"""Arm A / reference dataset: the B,G,R channels of every 5-channel TIFF as a lossless 3-channel PNG.

Pixels are bit-identical to channels 1:4 of the TIFFs, so arms A and B (and the RT-DETR reference) see exactly the
same frame-t pixels. Labels are copied (not symlinked) so the Ultralytics label cache is per dataset.

    python make_rgb_dataset.py --src ~/exp/temporal/data/k2 --dst ~/exp/temporal/data/k2_rgb
"""

from __future__ import annotations

import argparse
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2

from common import bgr_of, read_tiff5, write_data_yaml


def _one(args):
    src, dst = args
    if dst.exists():
        return 0
    dst.parent.mkdir(parents=True, exist_ok=True)
    bgr = bgr_of(read_tiff5(src))
    tmp = dst.with_name(dst.name + ".tmp.png")
    cv2.imwrite(str(tmp), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 1])
    tmp.replace(dst)
    return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", type=Path, required=True)
    ap.add_argument("--dst", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    cv2.setNumThreads(1)
    jobs = [(p, a.dst / "images" / p.parent.name / (p.stem + ".png")) for p in sorted((a.src / "images").glob("*/*.tiff"))]
    print(f"{len(jobs)} images")
    with ProcessPoolExecutor(a.workers) as ex:
        n = sum(ex.map(_one, jobs, chunksize=8))
    print(f"wrote {n} new PNGs")
    if not (a.dst / "labels").exists():
        shutil.copytree(a.src / "labels", a.dst / "labels")
    shutil.copy(a.src / "meta.jsonl", a.dst / "meta.jsonl")
    write_data_yaml(a.dst, a.dst / "data.yaml", channels=3)
    (a.dst / ".complete").write_text(f"{len(jobs)}\n")


if __name__ == "__main__":
    main()
