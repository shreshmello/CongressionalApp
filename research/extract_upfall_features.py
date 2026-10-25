from pathlib import Path
import argparse
import math

import cv2
import mediapipe as mp
import numpy as np
import pandas as pd


def summarize(values, prefix):
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]

    names = ["mean", "std", "min", "max", "range"]

    if len(values) == 0:
        return {f"{prefix}_{name}": np.nan for name in names}

    return {
        f"{prefix}_mean": float(np.mean(values)),
        f"{prefix}_std": float(np.std(values)),
        f"{prefix}_min": float(np.min(values)),
        f"{prefix}_max": float(np.max(values)),
        f"{prefix}_range": float(np.ptp(values)),
    }


def extract_video_features(video_path, sample_every=2, max_frames=1800):
    capture = cv2.VideoCapture(str(video_path))

    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    fps = capture.get(cv2.CAP_PROP_FPS)

    if not np.isfinite(fps) or fps <= 0:
        fps = 30.0

    pose = mp.solutions.pose.Pose(
        static_image_mode=False,
        model_complexity=1,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    hip_y = []
    torso_angles = []
    torso_distances = []
    visibility_values = []

    frame_index = 0
    frames_read = 0

    landmarks = mp.solutions.pose.PoseLandmark

    try:
        while frames_read < max_frames:
            success, frame = capture.read()

            if not success:
                break

            current_frame = frame_index
            frame_index += 1
            frames_read += 1

            if current_frame % sample_every != 0:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = pose.process(rgb)

            if not result.pose_landmarks:
                continue

            points = result.pose_landmarks.landmark

            required = [
                landmarks.LEFT_SHOULDER.value,
                landmarks.RIGHT_SHOULDER.value,
                landmarks.LEFT_HIP.value,
                landmarks.RIGHT_HIP.value,
            ]

            selected = [points[index] for index in required]
            visibility = [float(point.visibility) for point in selected]

            visibility_values.append(float(np.mean(visibility)))

            # Ignore frames where key body landmarks are poorly detected.
            if min(visibility) < 0.35:
                continue

            left_shoulder, right_shoulder, left_hip, right_hip = selected

            shoulder_x = (left_shoulder.x + right_shoulder.x) / 2
            shoulder_y = (left_shoulder.y + right_shoulder.y) / 2

            hip_x = (left_hip.x + right_hip.x) / 2
            center_hip_y = (left_hip.y + right_hip.y) / 2

            dx = shoulder_x - hip_x
            dy = shoulder_y - center_hip_y

            angle = np.degrees(
                np.arctan2(abs(dx), abs(dy) + 1e-8)
            )

            distance = math.hypot(dx, dy)

            hip_y.append(center_hip_y)
            torso_angles.append(float(angle))
            torso_distances.append(float(distance))

    finally:
        capture.release()
        pose.close()

    features = {
        "frames_read": frames_read,
        "pose_frames": len(hip_y),
        "pose_detection_ratio": len(hip_y) /
            max(1, math.ceil(frames_read / sample_every)),
        "video_duration_seconds": frames_read / fps,
    }

    features.update(summarize(hip_y, "hip_y"))
    features.update(summarize(torso_angles, "torso_angle"))
    features.update(summarize(torso_distances, "torso_distance"))
    features.update(summarize(visibility_values, "visibility"))

    # Estimate vertical hip movement speed between sampled frames.
    if len(hip_y) >= 2:
        time_step = sample_every / fps
        speed = np.abs(np.diff(hip_y)) / time_step
        features.update(summarize(speed, "hip_vertical_speed"))
    else:
        features.update(summarize([], "hip_vertical_speed"))

    return features


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--manifest", required=True)
    parser.add_argument("--data-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sample-every", type=int, default=2)
    parser.add_argument("--max-frames", type=int, default=1800)

    args = parser.parse_args()

    if args.sample_every < 1:
        raise SystemExit("--sample-every must be at least 1")

    manifest = Path(args.manifest).resolve()
    data_root = Path(args.data_root).resolve()

    if not manifest.exists():
        raise SystemExit(f"Manifest not found: {manifest}")

    table = pd.read_csv(manifest)

    required_columns = {
        "video_path",
        "participant_id",
        "label",
    }

    if not required_columns.issubset(table.columns):
        raise SystemExit(
            "Manifest must contain video_path, participant_id, and label."
        )

    rows = []

    for index, record in table.iterrows():
        video = Path(str(record["video_path"]))

        if not video.is_absolute():
            video = data_root / video

        if not video.exists():
            print(f"Skipping missing video: {video}")
            continue

        try:
            features = extract_video_features(
                video,
                sample_every=args.sample_every,
                max_frames=args.max_frames,
            )

            row = {
                "participant_id": str(record["participant_id"]),
                "label": record["label"],
                "video_path": str(video),
                **features,
            }

            rows.append(row)

            print(
                f"Processed {video.name}: "
                f"pose ratio={features['pose_detection_ratio']:.2f}"
            )

        except Exception as error:
            print(f"Could not process {video}: {error}")

    if not rows:
        raise SystemExit("No videos were processed successfully.")

    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)

    result = pd.DataFrame(rows)
    result["label"] = pd.to_numeric(result["label"], errors="coerce")

    result.to_csv(output, index=False)

    print(f"Saved features to: {output}")
    print("Review the labels and pose_detection_ratio before evaluation.")


if __name__ == "__main__":
    main()
