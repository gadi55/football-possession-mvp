# Football Possession MVP

Browser-based proof of concept for short broadcast football clips. Upload an MP4, identify players and a sports ball with Ultralytics YOLO, cluster team colours, estimate ball-to-player possession, smooth transitions, and download a possession event CSV and annotated video.

**Important:** The default COCO model is *not* football-specific. Small footballs may be missed and proximity is not definitive ball control. This is an experimental baseline, not verified professional match analysis. For reliable football footage, supply a compatible football-trained YOLO weights file through the sidebar (classes mapped to person and ball), then validate outputs manually.

## Run in the cloud (no local installation)

1. Visit https://share.streamlit.io and create an app from `gadi55/football-possession-mvp`, branch `main`, entry point `app.py`.
2. Select Python 3.11 if offered, and deploy. First run downloads YOLO weights and can be slow. Free CPU instances may time out or run out of memory on long clips.
3. Upload a 10–20 second MP4 with visible ball and distinct kits. Limit video to 480px and 5 fps for initial tests.

## Method

- YOLO detects people (COCO class 0) and sports balls (COCO class 32) in sampled frames.
- Crop upper torso of detected people, cluster HSV colours into two teams with OpenCV k-means; spectators and referees can still be misclassified.
- Ball is assigned to the nearest detected player's foot point within a configurable pixel distance relative to image height.
- Uncertain frames hold the prior team briefly; confirmed team changes require multiple samples and are recorded as approximate turnover timestamps.
- A labelled MP4 and CSV are exported. Team labels are arbitrary A/B, not home/away.

## Custom models

The app supports uploading custom Ultralytics YOLO .pt weights. Set the player and ball class IDs in the sidebar. Models with additional referee classes should be configured so the player class excludes them. Do not upload untrusted model weights: PyTorch model files can execute code during deserialization.

## Limits

This is **offline clip processing**, not a live stream. Ball proximity can confuse tackles, passes, aerial balls, camera cuts, bench players, and occlusion. No guaranteed detection accuracy or automatic ground truth. Use short, legally shareable clips.

License: MIT for original application code. Third-party model and package licenses apply independently; check Ultralytics licensing before commercial deployment.
