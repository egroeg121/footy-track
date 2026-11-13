# Inference web app

Upload a match video in the browser. The app cuts it into broadcast shots, tracks
the players, referees and ball on every shot, and shows each result as an overlay
video with a short summary. It also stitches the whole upload back together into
one "Full video": boxes on the broadcast shots, plain footage for everything the
cutter dropped, in the original order. It is a front end for the existing pipeline: the
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
    E --> L["Work out the timeline<br/>tracked clips + gaps"]
    L --> F["Stitch full.mp4<br/>overlay clips + plain gaps"]
    F --> R[Results page]
    F -. "stitch fails: job still done,<br/>per-clip results kept" .-> R
```

Jobs run one at a time on a single worker thread, because there is one GPU. A
second upload waits in the queue ("Waiting for the GPU"). The detector is loaded
once and reused for every clip. A clip that fails (for example out of GPU memory)
is marked failed and the other clips still finish.

## The full video

The cutter keeps only broadcast shots of at least a couple of seconds. Replays,
crowd shots and short stretches are thrown away, so the clips alone do not add up
to the match. To watch the whole upload in order, the app builds one `full.mp4`
per job once tracking is finished:

- every tracked clip is shown as its overlay (boxes, ids, ball trail);
- every stretch the clips do not cover (before the first clip, between clips,
  after the last) is copied from the original upload as plain footage, with no
  boxes;
- a clip that failed to track, and the part of a clip that was not tracked
  because of the quick-run fields, also show as plain footage.

No detection or tracking runs on the plain stretches, so they cost only an
encode. The results page opens on the "Full video" view by default. Under the
player is a bar showing tracked (blue) and cutaway (dark) stretches; click it to
jump, and a small "cutaway - not tracked" label appears while a plain stretch is
playing. The per-clip views and downloads are unchanged, and `full.mp4` has its
own download button.

Details that matter:

- **Frame exact.** Cuts are made in frame numbers, not rounded seconds, so
  `full.mp4` has the same number of frames as the upload and every frame is in
  its original place. A clip starts at its start time times the frame rate, and
  the plain stretches are what is left between the clips.
- **No sound.** Audio is dropped. The overlay clips never had any, and keeping
  the original track in step with a re-cut video is not worth the risk of drift.
- **Re-encoded.** Every piece is encoded again with the same settings (H.264,
  yuv420p, no B-frames, even width and height, the upload's frame rate) so the
  pieces join cleanly. Tracked clips are therefore encoded twice (once for the
  overlay, once here) and are slightly softer than the per-clip overlay download.
  The size is similar to the upload for the plain parts. `ffmpeg` 5.1 or newer
  is needed.
- **Timeline.** `job.json` (and `GET /infer/api/jobs/<id>`) has a `timeline`: a
  list of spans in time order with `start_s`, `end_s`, `start_frame`,
  `end_frame`, `kind` (`tracked` or `skipped`) and `clip` (the clip id for a
  tracked span, otherwise null). The spans cover the whole video without gaps.
  Odd frame rates such as 29.97 are handled exactly.
- **Failure is not fatal.** If stitching fails (no `ffmpeg`, disk full, anything
  else) the job is still "done", the `stitch` stage is marked failed with a
  message, the timeline is kept and all per-clip results work. The results page
  then simply has no "Full video" button. Cancelling the job also stops a stitch
  in progress.
- `GET /infer/api/jobs/<id>/full.mp4` downloads it, with the same token rules as
  every other download; it answers 404 when there is none (older jobs, failed
  stitch).
- The cutter decides where shots start and end, so a boxed stretch can begin or
  end a few frames away from where a human would cut. The stitch only follows
  what the cutter recorded.

## Preview quality (for slow networks)

The full-quality videos are large: a 1080p overlay is about 5.7 Mbit/s, so a 20
second clip is roughly 15 MB. On a slow or flaky connection (weak Wi-Fi, a phone
on a bad link) the browser can receive that slower than it plays. Safari in
particular then shows the first frame and sits there with the duration shown as
"--:--".

So the app can also make a small **preview** copy of every overlay and of the
full video:

- about 720p (the same shape as the original, never enlarged), H.264, about
  1 to 1.5 Mbit/s, no sound, the same number of frames and the same frame rate;
- written with the index at the front of the file, so playback starts before the
  download has finished;
- made right after each clip's overlay and after the full video is stitched, as
  `overlay_preview.mp4` (per clip) and `full_preview.mp4` (whole job), next to the
  full-quality files, which are never replaced.

On the results page a **Quality: Preview (fast) / Full** switch sits under the
player. It defaults to Preview whenever one exists, remembers your choice in the
browser, and keeps your place in the video when you switch. It applies to both
the "Full video" view and each clip. The download buttons always include the
full-quality files; the preview files are offered as extra downloads.

It is a server option, on by default. Turn it off with
`FOOTY_INFERENCE_PREVIEW=0` or `--no-preview` if you only use fast networks and
want to save the (small) extra encode time and disk space. `GET
/infer/api/config` reports `preview: true` or `false`, and each job lists the
state of its previews (`preview` on a clip, `full_preview` on the job: `done`,
`failed` or `off`).

A preview is a convenience, so it can never break a job: if making one fails
(no `ffmpeg`, disk full) the clip and the job still finish, the failure is logged
and recorded in the job, and the page simply shows no quality switch for that
video. Cancelling a job also stops a preview in progress. Jobs made before this
option existed have no previews and work as before (the switch is hidden).

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
cutter that just makes fixed 100-frame chunks, leaving a 25-frame gap after each
one (so there are cutaways to stitch), so the whole flow can be tried with no
weights. **Development only**: the boxes it draws are invented and the
chunks are not broadcast shots. The upload page shows a banner while it is on.

## Using it

- **Upload** (`/infer`): drop a file. The page shows uploading, cutting, then
  "tracking clip i of n". Optional quick-run fields cap the frames per clip or
  take every Nth frame.
- **Jobs** (`/infer/jobs`): all jobs, with status and delete.
- **Results** (`/infer/jobs/<id>`): watch the "Full video", or pick a clip and watch its overlay, read the
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
| `FOOTY_INFERENCE_PREVIEW` | `--preview` / `--no-preview` | on | Also make small preview videos for slow networks |
| `FOOTY_INFERENCE_FAKE` | `--fake` | off | Dev-only fake detector and cutter |

## What is stored

Each job is a folder under the jobs directory with a `job.json` (status of each
stage and clip, the timeline, counts, timings, errors), the upload, the cut clips,
one output folder per clip, `full.mp4` and the optional `full_preview.mp4` (each clip folder also has `overlay_preview.mp4`). Folder and file names are generated; the uploaded file
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
- **Full video size and time**: encoding every plain stretch and every tracked
  clip again is CPU work: about 5 seconds for a 40 second 1080p video on the
  GPU machine, so expect on the order of ten minutes for a full match, and disk
  space about the size of the upload. It runs after tracking, so per-clip results appear first.
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
