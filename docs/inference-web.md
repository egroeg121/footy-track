# Inference web app

Upload a match video in the browser. The app cuts it into broadcast shots, tracks
the players, referees and ball on every shot, and shows each result as an overlay
video with a short summary. It is a front end for the existing pipeline: the
cutting is `scripts/split_broadcast_segments.py` and the tracking is
`run_inference` (see [Inference pipeline](inference.md)). Nothing about the
tracking itself is new.

!!! warning "Read the output with care"
    Ball recall is about 0.6, and the ball track can lock onto a false positive
    (for example a player's head) and follow it. Players and referees are tracked
    frame to frame only, with no team and no re-identification. Tracks restart at
    every broadcast cut. The results page repeats this in a banner.

## How a job flows

```mermaid
flowchart LR
    U[Browser upload] --> S[Save file<br/>check it opens]
    S --> Q[Queue]
    Q --> C["Cut into shots<br/>split_broadcast_segments"]
    C --> T["Per clip: run_inference<br/>with overlay"]
    T --> E["Re-encode overlay<br/>to H.264"]
    E --> R[Results page]
```

Jobs run one at a time on a single worker thread, because there is one GPU. A
second upload waits in the queue ("Waiting for the GPU"). The detector is loaded
once and reused for every clip. A clip that fails (for example out of GPU memory)
is marked failed and the other clips still finish.

## Run it on the GPU machine

The app is meant to run on the machine that has the GPU and the detector
weights, and to be opened from other devices over a private network such as
Tailscale. There is no remote-job plumbing: the browser talks straight to the
app.

```bash
# on the GPU machine, in the repo, with the project environment
make infer-web HOST=0.0.0.0 PORT=8080
# or
python -m footy_track.inference.web --host 0.0.0.0 --port 8080
```

Then open `http://<the machine's private name or address>:8080/infer` from a
laptop or phone on the same private network. On WSL, run it inside the WSL
distro; reaching it through Tailscale needs Tailscale to be running where the
app is listening (inside WSL, or on the Windows host with the port forwarded).
Do not forward the port to the public internet.

Needs: the detector checkpoint (`CURRENT_BEST_DETECTOR_CHECKPOINT`, not in git),
the broadcast classifier weights, and `ffmpeg` on the PATH (to cut clips and to
make the overlay playable in browsers). Without the checkpoint the app refuses to
start and says where it looked.

Notes that matter on the WSL box:

- If the WSL virtual machine is down, nothing answers. That looks like a network
  problem but is not one; start WSL from the Windows side first.
- The app does not use dataloader workers, so the "workers must be 0" training
  rule does not apply to it. It runs detection in the app process.
- Run it in `tmux` so it survives an ssh disconnect.

### Try it without a GPU

`make infer-web-fake` (or `FOOTY_INFERENCE_FAKE=1`) swaps in a fake detector and a
cutter that just makes fixed 100-frame chunks, so the whole flow can be tried
with no weights. **Development only**: the boxes it draws are invented and the
chunks are not broadcast shots. The upload page shows a banner while it is on.

## Using it

- **Upload** (`/infer`): drop a file. The page shows uploading, cutting, then
  "tracking clip i of n". Optional quick-run fields cap the frames per clip or
  take every Nth frame.
- **Jobs** (`/infer/jobs`): all jobs, with status and delete.
- **Results** (`/infer/jobs/<id>`): pick a clip, watch the overlay, read the
  summary (frames, fps, player and referee tracklets, ball detected, interpolated
  and none), and download the overlay, the cut clip, `detections.jsonl`,
  `tracklets.jsonl`, `ball_track.jsonl` and `summary.json`. Formats are in
  [Inference pipeline](inference.md#outputs).

The same API works from a script. Upload is the raw file as the request body:

```bash
curl --data-binary @match.mp4 "http://host:8080/infer/api/jobs?name=match.mp4&max_frames=200"
curl http://host:8080/infer/api/jobs/<id>
```

### Demo clip

To let a visitor see the pipeline in one click, point the app at a sample video
kept on the server (not in git):

```bash
FOOTY_INFERENCE_DEMO_CLIP=/path/to/sample.mp4 make infer-web
# or: python -m footy_track.inference.web --demo-clip /path/to/sample.mp4
```

The upload page then shows a "Try a demo clip" button. It starts a normal job
from a copy of that file (cut, track, overlay), with no upload, and honours the
quick-run fields. If the variable is unset or the file is missing, the button is
hidden. The server path is never sent to the browser; `GET /infer/api/config`
reports only `demo` (true or false) and, when true, `demo_name` and
`demo_duration_s`. From a script:

```bash
curl -X POST "http://host:8080/infer/api/demo?max_frames=200"
```

It needs the same token as the rest of the API, and answers 404 when no demo
clip is available.

## Settings

Environment variables, with command-line flags that win over them.

| Variable | Flag | Default | Meaning |
|---|---|---|---|
| `FOOTY_INFERENCE_JOBS_DIR` | `--jobs-dir` | `runs/web` | Where jobs are stored |
| `FOOTY_INFERENCE_WEIGHTS` | `--weights` | `CURRENT_BEST_DETECTOR_CHECKPOINT` | Detector checkpoint |
| `FOOTY_INFERENCE_DEVICE` | `--device` | auto (cuda, mps, cpu) | Detector device |
| `FOOTY_INFERENCE_MAX_UPLOAD_MB` | `--max-upload-mb` | 8192 | Upload size limit |
| `FOOTY_INFERENCE_TOKEN` | none | unset | Optional shared token, see below |
| `FOOTY_INFERENCE_DEMO_CLIP` | `--demo-clip` | unset | Sample video for the demo button |
| `FOOTY_INFERENCE_FAKE` | `--fake` | off | Dev-only fake detector and cutter |

## What is stored

Each job is a folder under the jobs directory with a `job.json` (status of each
stage and clip, counts, timings, errors), the upload, the cut clips and one
output folder per clip. Folder and file names are generated; the uploaded file
name is only shown on the page. The state file is rewritten safely, so finished
jobs are still there after a restart. A job that was running during a restart is
marked failed ("the server restarted"), and half-finished uploads are removed.
Deleting a job removes its folder. A running job must be cancelled first.

## Limits and failure cases

- **No auth beyond an optional token.** Uploads run on a GPU machine, so only
  expose the app on a private network. If `FOOTY_INFERENCE_TOKEN` is set, the API
  and downloads need it (as `Authorization: Bearer`, `?token=`, or a cookie that
  the pages set after asking once). The HTML pages themselves hold no data. It is
  a single shared secret over plain HTTP, not a login system.
- **Upload size** is capped (default 8 GB) and checked while streaming, so an
  oversized upload is cut off and removed. Accepted extensions: mp4, mov, mkv, ts,
  avi, webm, m4v.
- **Bad file**: if OpenCV cannot open it or read a frame, the upload is rejected
  with a clear message and nothing is kept.
- **No broadcast shots found**: the job fails with an explanation (the cutter keeps
  only broadcast footage of at least two seconds).
- **Overlay playback**: OpenCV writes `mp4v`, which most browsers cannot play. The
  app re-encodes each overlay to H.264 with `ffmpeg`. If `ffmpeg` is missing or
  fails, the original is kept, the results page says so, and you can still
  download it and play it locally.
- **Speed**: RT-DETR at 1280 px needs a GPU. A full match has many clips and takes
  a while; use the quick-run fields for a preview. Cancelling stops at the next
  batch of frames.
- One job at a time. Progress on a clip is shown per clip, not per frame.
- Not run end to end with real weights on a GPU as part of writing this; see the
  pull request for what was and was not tested.

## Why a separate app

It is its own small FastAPI app (`footy_track.inference.web`) rather than a
router on the labeller. It is meant to run alone on the GPU machine, and the
labeller imports a large stack that inference does not need. The routes are an
ordinary router (`create_router`), so they can be mounted into the labeller later
if wanted.

## Side fix

`split_broadcast_segments.py` classified frames on a hard-coded `mps` device,
which fails on any non-Apple machine, including the CUDA box this app runs on. It
now uses the same cuda, then mps, then cpu choice as inference.

## Not done yet

- Interactive in-browser overlay (draw the tracks on the original video, toggle
  players and ball) instead of a pre-rendered video.
- A queue across several GPUs or machines.
- Real authentication and per-user jobs.
- Per-frame progress and a live preview while a clip is being tracked.
- Choosing the cutter settings (sample rate, minimum shot length) on the upload
  page.
- Ball-on-player guard and team assignment, which depend on the work listed in
  [Inference pipeline](inference.md#open-decisions-revisit-later).
