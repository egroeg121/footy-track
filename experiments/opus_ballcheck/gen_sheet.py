"""Review sheet of cards where Opus disagrees with George. Usage: gen_sheet.py RUN [--min-conf 0.8] [--out path]

Self-contained HTML (images inlined, strip downscaled). George's verdict is the reference but he can be
wrong, so each card shows both and Opus's reason. Sorted by Opus confidence, ball<->not_ball flips first."""
from __future__ import annotations

import base64
import html
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import score  # noqa: E402

OUT = Path("/mnt/storage/exp/opus_ballcheck")


def b64(p: Path, w: int | None = None) -> str:
    if w:
        import cv2
        im = cv2.imread(str(p))
        s = w / im.shape[1]
        im = cv2.resize(im, (w, round(im.shape[0] * s)), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 80])
        return base64.b64encode(buf.tobytes()).decode()
    return base64.b64encode(p.read_bytes()).decode()


def main() -> None:
    run = sys.argv[1]
    mc = float(sys.argv[sys.argv.index("--min-conf") + 1]) if "--min-conf" in sys.argv else 0.8
    out = Path(sys.argv[sys.argv.index("--out") + 1]) if "--out" in sys.argv else OUT / f"disagreements_{run}.html"
    rows, _ = score.load(run)
    dis = [r for r in rows if r["pred"] != r["george"] and r["pconf"] >= mc]
    flip = lambda r: {r["pred"], r["george"]} == {"ball", "not_ball"}
    dis.sort(key=lambda r: (not flip(r), -r["pconf"]))
    parts = [f"<!doctype html><meta charset=utf-8><title>Opus vs George: {html.escape(run)}</title>"
             "<style>body{font:14px sans-serif;margin:16px;max-width:1500px}.c{border:1px solid #888;margin:12px 0;padding:8px}"
             ".r{display:flex;gap:8px;flex-wrap:wrap}img{max-width:100%}.flip{border-color:#c00;border-width:2px}"
             "code{background:#eee;padding:1px 4px}</style>"
             f"<h2>Opus ({html.escape(run)}) confidently (>= {mc}) disagrees with George: {len(dis)} of {len(rows)} cards</h2>"
             "<p>Red border = ball vs not_ball flip. Yellow box = box judged; magenta = other box. Look at the filmstrip "
             "(t=0 tile is boxed) before deciding who is right.</p>"]
    for r in dis:
        d = OUT / "renders" / r["id"]
        parts.append(
            f"<div class='c{' flip' if flip(r) else ''}'><b>{r['id']}</b> {r['stratum']} {r['clip']} f{r['frame_index']} "
            f"| George: <code>{r['george_raw']}</code> | Opus: <code>{r['pred']}</code> conf {r['pconf']:.2f}"
            f"{' | which_box ' + str(r['which']) if r['which'] else ''}<br><i>{html.escape(r['reason'])}</i>"
            f"<div class=r><img src='data:image/jpeg;base64,{b64(d / 'still.jpg', 360)}'>"
            f"<img src='data:image/jpeg;base64,{b64(d / 'wide.jpg', 360)}'>"
            f"<img src='data:image/jpeg;base64,{b64(d / 'strip.jpg', 640)}'></div></div>")
    out.write_text("".join(parts))
    print(out, len(dis))
    for r in dis:
        print(r["id"], r["stratum"], "G", r["george_raw"], "O", r["pred"], r["pconf"], "FLIP" if flip(r) else "")


if __name__ == "__main__":
    main()
