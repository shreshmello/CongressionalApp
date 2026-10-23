from pathlib import Path
import math

import cv2
import numpy as np
import mediapipe as mp


# MediaPipe Pose landmark indices.
LEFT_SHOULDER = 11
RIGHT_SHOULDER = 12

LEFT_HIP = 23
RIGHT_HIP = 24

LEFT_ANKLE = 27
RIGHT_ANKLE = 28

REQUIRED_LANDMARKS = [
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_HIP,
    RIGHT_HIP,
    LEFT_ANKLE,
    RIGHT_ANKLE,
]


def percentile_range(values):
    """
    Calculate the difference between the 90th and 10th percentiles.

    This is less sensitive to extreme values than a simple max-min range.
    Returns None when there is insufficient data.
    """

    values = np.asarray(values, dtype=float)

    if values.size < 3:
        return None

    return float(
        np.percentile(values, 90)
        - np.percentile(values, 10)
    )


def calculate_motion_similarity(left_values, right_values):
    """
    Compare the horizontal movement ranges of the two ankles.

    This is an exploratory image-space similarity measure.

    IMPORTANT:
    It is NOT a validated clinical gait-symmetry measurement.
    """

    left_range = percentile_range(left_values)
    right_range = percentile_range(right_values)

    if left_range is None or right_range is None:
        return None

    denominator = max(left_range, right_range, 1e-6)

    difference = abs(left_range - right_range)

    similarity = 100 * (
        1 - min(difference / denominator, 1)
    )

    return round(float(similarity), 1)


def analyze_video(video_path: Path, model_path: Path) -> dict:
    """
    Analyze a video using MediaPipe Pose Landmarker.

    The output contains experimental image-based features.
    It does not provide a medical diagnosis or validated fall-risk score.
    """

    video_path = Path(video_path)
    model_path = Path(model_path)

    if not video_path.exists():
        raise FileNotFoundError("Video file was not found.")

    if not model_path.exists():
        raise FileNotFoundError(
            "MediaPipe model was not found. "
            "Run python download_model.py."
        )

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise ValueError(
            "Could not open the uploaded video."
        )

    fps = cap.get(cv2.CAP_PROP_FPS)

    if not fps or not math.isfinite(fps) or fps <= 0:
        fps = 30.0

    # Landmark sequences across detected frames.
    hip_mid_x = []
    hip_mid_y = []
    shoulder_mid_x = []

    left_ankle_x = []
    right_ankle_x = []

    visibility_values = []

    frames_seen = 0
    detected_frames = 0

    options = mp.tasks.vision.PoseLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(
            model_asset_path=str(model_path)
        ),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=0.5,
        min_pose_presence_confidence=0.5,
        min_tracking_confidence=0.5,
    )

    try:
        with mp.tasks.vision.PoseLandmarker.create_from_options(
            options
        ) as landmarker:

            while True:
                success, frame = cap.read()

                if not success:
                    break

                frames_seen += 1

                rgb_frame = cv2.cvtColor(
                    frame,
                    cv2.COLOR_BGR2RGB,
                )

                image = mp.Image(
                    image_format=mp.ImageFormat.SRGB,
                    data=rgb_frame,
                )

                timestamp_ms = int(
                    (frames_seen - 1) * 1000 / fps
                )

                output = landmarker.detect_for_video(
                    image,
                    timestamp_ms,
                )

                if not output.pose_landmarks:
                    continue

                landmarks = output.pose_landmarks[0]

                # MediaPipe returns normalized image coordinates.
                # These are not physical distances in meters.
                visible = [
                    float(landmarks[i].visibility)
                    for i in REQUIRED_LANDMARKS
                ]

                average_visibility = float(
                    np.mean(visible)
                )

                visibility_values.append(
                    average_visibility
                )

                if average_visibility < 0.45:
                    continue

                detected_frames += 1

                left_hip = landmarks[LEFT_HIP]
                right_hip = landmarks[RIGHT_HIP]

                left_shoulder = landmarks[LEFT_SHOULDER]
                right_shoulder = landmarks[RIGHT_SHOULDER]

                left_ankle = landmarks[LEFT_ANKLE]
                right_ankle = landmarks[RIGHT_ANKLE]

                # Midpoint between the hips.
                hip_mid_x.append(
                    (left_hip.x + right_hip.x) / 2
                )

                hip_mid_y.append(
                    (left_hip.y + right_hip.y) / 2
                )

                # Midpoint between the shoulders.
                shoulder_mid_x.append(
                    (
                        left_shoulder.x
                        + right_shoulder.x
                    ) / 2
                )

                left_ankle_x.append(
                    left_ankle.x
                )

                right_ankle_x.append(
                    right_ankle.x
                )

    finally:
        cap.release()

    if frames_seen < 10:
        raise ValueError(
            "The video is too short. "
            "Please upload a longer walking clip."
        )

    detection_rate = detected_frames / frames_seen

    # Reject clips that do not contain enough usable landmarks.
    if (
        detected_frames < 10
        or detection_rate < 0.35
    ):
        return {
            "status": "insufficient",
            "message": (
                "The person's pose could not be detected "
                "reliably enough for analysis."
            ),
            "frames_seen": frames_seen,
            "detected_frames": detected_frames,
            "detection_rate": round(
                detection_rate * 100,
                1,
            ),
            "metrics": [],
            "notes": [
                "Use good lighting.",
                "Keep the full body visible.",
                "Keep the camera stationary.",
                "Avoid blocking the legs and feet.",
                "Try a longer video of ordinary walking.",
            ],
        }

    # Calculate preliminary image-space features.
    hip_horizontal_range = percentile_range(
        hip_mid_x
    )

    hip_vertical_range = percentile_range(
        hip_mid_y
    )

    shoulder_horizontal_range = percentile_range(
        shoulder_mid_x
    )

    motion_similarity = calculate_motion_similarity(
        left_ankle_x,
        right_ankle_x,
    )

    metrics = [
        {
            "name": "Hip-center horizontal movement",
            "value": (
                f"{hip_horizontal_range * 100:.2f}%"
                if hip_horizontal_range is not None
                else "Unavailable"
            ),
            "description": (
                "Horizontal variation of the hip midpoint "
                "in normalized image coordinates. This is "
                "not a validated body-sway measurement."
            ),
        },
        {
            "name": "Hip-center vertical movement",
            "value": (
                f"{hip_vertical_range * 100:.2f}%"
                if hip_vertical_range is not None
                else "Unavailable"
            ),
            "description": (
                "Vertical variation of the hip midpoint "
                "in the image. Perspective and camera angle "
                "can affect this measurement."
            ),
        },
        {
            "name": "Shoulder-center horizontal movement",
            "value": (
                f"{shoulder_horizontal_range * 100:.2f}%"
                if shoulder_horizontal_range is not None
                else "Unavailable"
            ),
            "description": (
                "Horizontal variation of the shoulder "
                "midpoint in image coordinates."
            ),
        },
        {
            "name": (
                "Left/right ankle-motion similarity"
            ),
            "value": (
                f"{motion_similarity:.1f}/100"
                if motion_similarity is not None
                else "Unavailable"
            ),
            "description": (
                "Similarity of horizontal ankle-motion "
                "ranges. This is an exploratory feature, "
                "not a validated gait-symmetry score."
            ),
        },
    ]

    return {
        "status": "complete",
        "message": (
            "Pose detection completed and exploratory "
            "movement features were calculated."
        ),
        "frames_seen": frames_seen,
        "detected_frames": detected_frames,
        "detection_rate": round(
            detection_rate * 100,
            1,
        ),
        "metrics": metrics,
        "notes": [
            (
                "These features are experimental and "
                "are not validated medical measurements."
            ),
            (
                "The app does not diagnose mobility "
                "conditions or predict falls."
            ),
            (
                "Camera angle, recording distance, "
                "visibility, and movement direction can "
                "change the results."
            ),
            (
                "Do not interpret these measurements "
                "as a clinical risk category."
            ),
        ],
    }
