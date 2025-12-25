## Initial


This will be a dev journal with thoughts and designs


Overall design:


stages:
 - Classifier: broadcast frame vs not
 - Detection: detect ball and playesr
 - Track: find continiuos players and ball
 - project: Convert into 2d coordinates on the pitch



Dev ideas:
* Use ultralytics where possible
* Roboflow for data annotation and smart labelling





Day 1:
Get data and scripts to create 1fps start

Use 1fps initially for training/labelling as it's good yield and variation between frames.

Create a 'split_video'. Need to extract frames, but putting a single large video would make it unwieldy. So instead ,we split into 10s intervals and then will extract frames.

Split the video into 10s chunks. Also create a test for 30s to be broken into 10s chunks.
