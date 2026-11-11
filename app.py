import os
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO

st.set_page_config(page_title="Football Possession MVP", page_icon="⚽", layout="wide")
st.title("⚽ Football Possession | Clip Analyzer")
st.caption("Experimental football possession detection · upload a short MP4 · cloud CPU friendly")
st.warning("Prototype: COCO sports-ball detection and proximity-based possession are approximate. Verify events against the original clip.")

with st.sidebar:
    st.header("Analysis settings")
    model_file = st.file_uploader("Optional football-trained YOLO .pt model (trusted files only)", type=["pt"])
    player_class = st.number_input("Player class ID", min_value=0, max_value=200, value=0)
    ball_class = st.number_input("Ball class ID", min_value=0, max_value=200, value=32)
    confidence = st.slider("Detection confidence", 0.05, 0.8, 0.18, 0.01)
    fps_sample = st.slider("Analysis frames per second", 1, 10, 5)
    max_width = st.select_slider("Processing width", options=[320, 480, 640], value=480)
    ball_radius = st.slider("Ball-to-foot threshold (% frame height)", 2, 25, 12)
    confirmation = st.slider("Frames to confirm a turnover", 1, 8, 3)
    st.caption("Team A/B colours are inferred from player crops, not verified team identities.")

video = st.file_uploader("Upload football footage (MP4)", type=["mp4"])
if video:
    st.video(video)


def jersey_feature(frame, box):
    x1, y1, x2, y2 = [int(v) for v in box]
    h, w = frame.shape[:2]
    x1, x2 = max(0, x1), min(w, x2)
    y1, y2 = max(0, y1), min(h, y2)
    bw, bh = x2 - x1, y2 - y1
    if bw < 6 or bh < 15:
        return None
    crop = frame[y1 + int(bh * .17):y1 + int(bh * .52), x1 + int(bw * .22):x2 - int(bw * .22)]
    if crop.size == 0:
        return None
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return np.median(hsv.reshape(-1, 3), axis=0).astype(np.float32)


def run_analysis(input_path, output_path, weights, player_id, ball_id, conf, sample_rate, width, threshold, confirm, progress):
    model = YOLO(weights)
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Cannot open uploaded MP4.")
    src_fps = cap.get(cv2.CAP_PROP_FPS)
    if not np.isfinite(src_fps) or src_fps <= 0:
        src_fps = 25.0
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, round(src_fps / sample_rate))
    actual_fps = src_fps / stride
    source_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    source_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if source_w < 1 or source_h < 1:
        raise ValueError("Video dimensions invalid.")
    out_w = min(width, source_w)
    out_h = max(2, int(source_h * out_w / source_w) // 2 * 2)
    out_w = out_w // 2 * 2
    writer = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*"mp4v"), actual_fps, (out_w, out_h))
    if not writer.isOpened():
        raise ValueError("Unable to create annotated video.")
    records, events, colours = [], [], []
    # Build jersey clusters from an initial sample, then keep centroids fixed.
    samples = []
    cap_idx = 0
    while len(samples) < 24 and cap_idx < min(frame_count if frame_count > 0 else 250, 250):
        ok, frame = cap.read()
        if not ok:
            break
        if cap_idx % stride == 0:
            frame = cv2.resize(frame, (out_w, out_h))
            result = model.predict(frame, conf=conf, imgsz=640, verbose=False)[0]
            for b in result.boxes:
                if int(b.cls.item()) == player_id:
                    feat = jersey_feature(frame, b.xyxy[0].cpu().numpy())
                    if feat is not None:
                        samples.append(feat)
        cap_idx += 1
    if len(samples) < 2:
        cap.release(); writer.release()
        raise ValueError("Not enough players detected to distinguish two teams. Try a different clip or football-trained model.")
    features = np.asarray(samples, dtype=np.float32)
    _, labels, centers = cv2.kmeans(features, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.2), 8, cv2.KMEANS_PP_CENTERS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    current, candidate, candidate_count, missing = None, None, 0, 0
    idx, processed = 0, 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx % stride:
                idx += 1
                continue
            timestamp = idx / src_fps
            frame = cv2.resize(frame, (out_w, out_h))
            result = model.predict(frame, conf=conf, imgsz=640, verbose=False)[0]
            players, balls = [], []
            for b in result.boxes:
                cls = int(b.cls.item())
                xyxy = b.xyxy[0].cpu().numpy()
                if cls == player_id:
                    feat = jersey_feature(frame, xyxy)
                    if feat is not None:
                        # HSV hue wraps at 180; approximate jersey clustering only.
                        team = int(np.argmin(np.linalg.norm(centers - feat, axis=1)))
                        players.append((xyxy, team))
                elif cls == ball_id:
                    balls.append((xyxy, float(b.conf.item())))
            observed = None
            if balls and players:
                ball_box = max(balls, key=lambda z: z[1])[0]
                bx, by = (ball_box[0] + ball_box[2]) / 2, (ball_box[1] + ball_box[3]) / 2
                distances = [np.hypot(bx - (p[0][0] + p[0][2]) / 2, by - p[0][3]) for p in players]
                closest = int(np.argmin(distances))
                if distances[closest] <= threshold / 100 * out_h:
                    observed = players[closest][1]
                cv2.circle(frame, (int(bx), int(by)), 7, (0, 255, 255), 2)
            if observed is None:
                missing += 1
                candidate, candidate_count = None, 0
                if missing > max(confirm * 2, 5):
                    current = None
            else:
                missing = 0
                if observed == current:
                    candidate, candidate_count = None, 0
                elif observed == candidate:
                    candidate_count += 1
                else:
                    candidate, candidate_count = observed, 1
                if candidate_count >= confirm:
                    previous = current
                    current = candidate
                    if previous is not None and previous != current:
                        events.append({"timestamp_s": round(timestamp, 2), "event": "possession_change", "from_team": f"Team {chr(65+previous)}", "to_team": f"Team {chr(65+current)}"})
                    candidate, candidate_count = None, 0
            for p, team in players:
                x1, y1, x2, y2 = [int(v) for v in p]
                color = (240, 130, 30) if team == 0 else (40, 80, 240)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"Team {chr(65+team)}", (x1, max(16, y1-4)), cv2.FONT_HERSHEY_SIMPLEX, .42, color, 1)
            label = f"Possession: Team {chr(65+current)}" if current is not None else "Possession: uncertain"
            cv2.rectangle(frame, (0, 0), (out_w, 32), (16, 22, 34), -1)
            cv2.putText(frame, f"{timestamp:05.1f}s  {label}", (9, 22), cv2.FONT_HERSHEY_SIMPLEX, .55, (255, 255, 255), 2)
            writer.write(frame)
            records.append({"timestamp_s": round(timestamp, 2), "team": f"Team {chr(65+current)}" if current is not None else "Uncertain", "players_detected": len(players), "ball_detected": bool(balls)})
            processed += 1
            if processed % 10 == 0 and frame_count > 0:
                progress.progress(min(1.0, (idx+1) / frame_count))
            idx += 1
    finally:
        cap.release()
        writer.release()
    return pd.DataFrame(records), pd.DataFrame(events, columns=["timestamp_s", "event", "from_team", "to_team"])


if video and st.button("Analyze possession", type="primary"):
    with tempfile.TemporaryDirectory() as temp:
        src = str(Path(temp) / "input.mp4")
        dst = str(Path(temp) / "annotated.mp4")
        with open(src, "wb") as f:
            f.write(video.getbuffer())
        weights = "yolo11n.pt"
        if model_file:
            weights = str(Path(temp) / "custom.pt")
            with open(weights, "wb") as f:
                f.write(model_file.getbuffer())
        bar = st.progress(0, text="Processing clip on cloud CPU...")
        try:
            timeline, events = run_analysis(src, dst, weights, int(player_class), int(ball_class), confidence, fps_sample, max_width, ball_radius, confirmation, bar)
            bar.progress(1.0, text="Analysis complete")
            st.subheader("Possession changes")
            st.metric("Confirmed turnovers", len(events))
            if len(events):
                st.dataframe(events, use_container_width=True, hide_index=True)
            else:
                st.info("No confirmed changes detected. Try a clearer clip or adjust settings.")
            st.subheader("Timeline")
            st.dataframe(timeline, use_container_width=True, hide_index=True)
            st.download_button("Download possession events CSV", events.to_csv(index=False).encode(), file_name="possession_events.csv", mime="text/csv")
            st.download_button("Download frame timeline CSV", timeline.to_csv(index=False).encode(), file_name="possession_timeline.csv", mime="text/csv")
            st.subheader("Annotated footage")
            video_bytes = Path(dst).read_bytes()
            st.video(video_bytes)
            st.download_button("Download annotated MP4", video_bytes, file_name="annotated.mp4", mime="video/mp4")
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")
            st.info("Try a 10-second broadcast clip, lower resolution, or a football-trained YOLO model.")
