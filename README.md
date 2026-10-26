#Motus— Real-Time Movement Observation Prototype

This version replaces video uploads with a browser camera preview and
on-device pose estimation.

## Features

- Opens the camera after the user clicks Start Camera.
- Uses MediaPipe Pose Landmarker in the browser.
- Displays a pose overlay and basic image-derived movement features.
- Flags experimental rapid changes in hip-center position or torso angle.
- Offers a short fall-related screening questionnaire.
- Stores observation summaries and questionnaire responses in the browser's local storage.
- Does not save or upload camera frames.

## Important limitations

The rapid-change thresholds are exploratory and not clinically derived.

This app does not predict that a person is about to fall, cannot reliably
detect all falls, and must not be used as an emergency alert or medical device.

It can produce false positives and false negatives.

The questionnaire does not produce a medical risk score.

## Run locally on Windows

1. Install Python if needed.
2. Open a terminal in this folder.
3. Run:

   ```bash
   python -m http.server 8000
