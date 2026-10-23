# MobilityMind Research Evaluation

## Research datasets to investigate

### 1. UP-Fall Detection Dataset

Official project page:
https://sites.google.com/up.edu.mx/challenge-up-2019/data

It includes video/camera data and simulated falls plus daily activities.
Its participants were healthy young adults.

It is a starting point for fall-event detection research, not evidence that
a model predicts falls in older adults or predicts imminent instability.

### 2. SisFall

Research paper and dataset description:
https://pmc.ncbi.nlm.nih.gov/articles/PMC5298771/

It includes accelerometer and gyroscope signals, along with videos of
activities and simulated falls.

Most signals are wearable-sensor data, so they cannot be directly used
to train the browser's camera-only features.

The paper notes that detection performance can decrease when testing
with older participants.

## CSV format for this script

Provide a labeled feature table with one row per independent window or trial.

Example format:

```csv
subject_id,label,hip_speed,torso_angle_speed,visibility
P01,0,0.12,7.2,0.91
P01,1,0.73,41.0,0.88
P02,0,0.16,8.5,0.95
P02,1,0.62,35.1,0.81
