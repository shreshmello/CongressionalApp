import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21";

"use strict";

// ============================================================
// MOTUS — Live Movement Observation Prototype
//
// Important:
// - Experimental movement observations only.
// - Not a medical device.
// - Not validated for predicting future falls.
// - Hip displacement uses normalized image coordinates,
//   not centimeters or real-world distance.
// ============================================================

const $ = (id) => document.getElementById(id);

const video = $("cameraVideo");
const canvas = $("overlayCanvas");
const ctx = canvas?.getContext("2d");

const HISTORY_KEY = "mobilitymind_history";
const SCREENING_KEY = "mobilitymind_screening";

const state = {
  stream: null,
  landmarker: null,
  running: false,
  observing: false,
  raf: null,

  lastVideoTime: -1,
  startAt: 0,
  endAt: 0,

  timeline: [],
  lastTimelineAt: 0,
  summary: null,

  samples: [],
  events: [],

  lastEventAt: 0,
  candidateStreak: 0,
  lastMetricFeature: null,

  totalFrames: 0,
  goodFrames: 0,

  questionnaire: null
};

// MediaPipe landmark indices.
const LANDMARKS = {
  leftShoulder: 11,
  rightShoulder: 12,
  leftHip: 23,
  rightHip: 24,
  leftKnee: 25,
  rightKnee: 26,
  leftAnkle: 27,
  rightAnkle: 28
};

const IMPORTANT_LANDMARKS = Object.values(LANDMARKS);

const CONNECTIONS = [
  [11, 12],
  [11, 13],
  [13, 15],
  [12, 14],
  [14, 16],
  [11, 23],
  [12, 24],
  [23, 24],
  [23, 25],
  [25, 27],
  [24, 26],
  [26, 28],
  [27, 29],
  [29, 31],
  [28, 30],
  [30, 32]
];

// Experimental event thresholds.
// These are engineering parameters, NOT clinical thresholds.
const SETTINGS = {
  visibilityMinimum: 0.55,

  // Compare poses approximately 150–400 ms apart.
  comparisonMinimumMs: 150,
  comparisonMaximumMs: 400,

  // Normalized image-coordinate change per second.
  hipSpeedThreshold: 0.42,

  // Torso angle change.
  torsoSpeedThreshold: 75,
  torsoAngleThreshold: 13,

  // Require repeated candidate frames before recording an event.
  requiredCandidateFrames: 3,

  // Prevent repeated events from the same continuous movement.
  eventCooldownMs: 3000,

  // Clear old samples to keep calculations recent.
  sampleWindowMs: 2500
};

// ============================================================
// UI HELPERS
// ============================================================

function setMessage(message) {
  const element = $("cameraMessage");

  if (element) {
    element.textContent = message;
  }
}

function setStatus(message, active = false) {
  const element = $("cameraStatus");

  if (!element) return;

  const dot = document.createElement("span");
  dot.className = "status-dot";

  element.replaceChildren(dot, document.createTextNode(` ${message}`));
  element.classList.toggle("active", active);
}

function setText(id, value) {
  const element = $(id);

  if (element) {
    element.textContent = String(value);
  }
}

function setDisabled(id, disabled) {
  const element = $(id);

  if (element) {
    element.disabled = disabled;
  }
}

function setDisplay(id, value) {
  const element = $(id);

  if (element) {
    element.style.display = value;
  }
}

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  })[character]);
}

// ============================================================
// POSE FEATURE CALCULATIONS
// ============================================================

function angleDegrees(hip, shoulder) {
  const dx = shoulder.x - hip.x;
  const dy = shoulder.y - hip.y;

  return Math.atan2(dx, -dy) * 180 / Math.PI;
}

function kneeAngle(hip, knee, ankle) {
  // Angle at the knee between hip and ankle; 180° is a straight leg.
  // Landmarks are normalized per axis, so scale x by the frame's aspect ratio.
  const aspect = video.videoWidth && video.videoHeight
    ? video.videoWidth / video.videoHeight
    : 1;

  const ux = (hip.x - knee.x) * aspect;
  const uy = hip.y - knee.y;
  const vx = (ankle.x - knee.x) * aspect;
  const vy = ankle.y - knee.y;

  const cos = (ux * vx + uy * vy) /
    (Math.hypot(ux, uy) * Math.hypot(vx, vy) + 1e-9);

  return Math.acos(Math.min(1, Math.max(-1, cos))) * 180 / Math.PI;
}

function meanVisibility(landmarks, indices) {
  const values = indices
    .map((index) => {
      const point = landmarks[index];

      return Number(
        point?.visibility ?? point?.presence ?? 0
      );
    })
    .filter(Number.isFinite);

  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0)
    / values.length;
}

function extractFeatures(landmarks, now) {
  const visibility = meanVisibility(
    landmarks,
    IMPORTANT_LANDMARKS
  );

  const leftShoulder = landmarks[LANDMARKS.leftShoulder];
  const rightShoulder = landmarks[LANDMARKS.rightShoulder];

  const leftHip = landmarks[LANDMARKS.leftHip];
  const rightHip = landmarks[LANDMARKS.rightHip];

  const leftKnee = landmarks[LANDMARKS.leftKnee];
  const rightKnee = landmarks[LANDMARKS.rightKnee];

  const leftAnkle = landmarks[LANDMARKS.leftAnkle];
  const rightAnkle = landmarks[LANDMARKS.rightAnkle];

  if (
    !leftShoulder ||
    !rightShoulder ||
    !leftHip ||
    !rightHip ||
    !leftKnee ||
    !rightKnee ||
    !leftAnkle ||
    !rightAnkle
  ) {
    return null;
  }

  const shoulder = {
    x: (leftShoulder.x + rightShoulder.x) / 2,
    y: (leftShoulder.y + rightShoulder.y) / 2
  };

  const hip = {
    x: (leftHip.x + rightHip.x) / 2,
    y: (leftHip.y + rightHip.y) / 2
  };

  const torsoAngle = angleDegrees(hip, shoulder);

  if (
    !Number.isFinite(hip.x) ||
    !Number.isFinite(hip.y) ||
    !Number.isFinite(torsoAngle) ||
    !Number.isFinite(visibility)
  ) {
    return null;
  }

  return {
    time: now,
    visibility,
    hipX: hip.x,
    hipY: hip.y,
    torsoAngle,
    leftKneeAngle: kneeAngle(leftHip, leftKnee, leftAnkle),
    rightKneeAngle: kneeAngle(rightHip, rightKnee, rightAnkle)
  };
}

// Calculate the smallest angular difference.
function angleDifference(a, b) {
  let difference = Math.abs(a - b);

  if (difference > 180) {
    difference = 360 - difference;
  }

  return difference;
}

// ============================================================
// HIP DISPLACEMENT METRIC
// ============================================================

// Measures how far the hip midpoint moved between two detected
// poses. Coordinates are normalized to the image, not centimeters.

function calculateHipDisplacement(current, previous) {
  if (!previous) {
    return null;
  }

  const elapsedSeconds =
    (current.time - previous.time) / 1000;

  if (elapsedSeconds <= 0) {
    return null;
  }

  const displacement = Math.hypot(
    current.hipX - previous.hipX,
    current.hipY - previous.hipY
  );

  return {
    displacement,
    elapsedSeconds,
    speed: displacement / elapsedSeconds
  };
}

function updateHipMetric(features) {
  const previous = state.lastMetricFeature;

  const result = calculateHipDisplacement(
    features,
    previous
  );

  state.lastMetricFeature = features;

  if (!result) {
    setText("hipValue", "—");
    return;
  }

  // Display displacement over the actual interval, rather than
  // distance from the top-left corner of the image.
  setText(
    "hipValue",
    `${result.displacement.toFixed(4)} units`
  );
}

// ============================================================
// DRAW POSE OVERLAY
// ============================================================

function resizeCanvas() {
  if (!video || !canvas || !ctx) return;

  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.max(
    1,
    Math.round(rect.width * dpr)
  );

  canvas.height = Math.max(
    1,
    Math.round(rect.height * dpr)
  );

  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawPose(landmarks) {
  if (!video || !canvas || !ctx) return;

  resizeCanvas();

  const width = video.clientWidth;
  const height = video.clientHeight;

  ctx.clearRect(0, 0, width, height);

  if (!landmarks) return;

  ctx.save();

  // Mirror skeleton to match a mirrored camera preview.
  ctx.translate(width, 0);
  ctx.scale(-1, 1);

  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(221, 242, 221, 0.92)";
  ctx.fillStyle = "rgba(74, 125, 94, 0.95)";

  for (const [a, b] of CONNECTIONS) {
    const p = landmarks[a];
    const q = landmarks[b];

    if (
      p &&
      q &&
      (p.visibility ?? 0) > 0.35 &&
      (q.visibility ?? 0) > 0.35
    ) {
      ctx.beginPath();
      ctx.moveTo(p.x * width, p.y * height);
      ctx.lineTo(q.x * width, q.y * height);
      ctx.stroke();
    }
  }

  for (const point of landmarks) {
    if ((point.visibility ?? 0) > 0.45) {
      ctx.beginPath();

      ctx.arc(
        point.x * width,
        point.y * height,
        3.4,
        0,
        Math.PI * 2
      );

      ctx.fill();
    }
  }

  ctx.restore();
}

// ============================================================
// MEDIAPIPE MODEL
// ============================================================

async function loadPoseModel() {
  if (state.landmarker) return;

  setMessage("Loading the pose-tracking model...");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
  );

  state.landmarker = await PoseLandmarker.createFromOptions(
    vision,
    {
      baseOptions: {
        modelAssetPath:
          "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
      },
      runningMode: "VIDEO",
      numPoses: 1,
      minPoseDetectionConfidence: 0.55,
      minPosePresenceConfidence: 0.55,
      minTrackingConfidence: 0.55
    }
  );
}

// ============================================================
// CAMERA CONTROLS
// ============================================================

async function startCamera() {
  const button = $("startCameraButton");

  if (!button || state.running) return;

  button.disabled = true;

  setMessage("Requesting camera access...");

  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access is unavailable. Open the app using localhost or HTTPS."
      );
    }

    const stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    state.stream = stream;

    video.srcObject = stream;

    await video.play();

    video.style.display = "block";

    setDisplay("cameraPlaceholder", "none");

    $("cameraStage")?.classList.add("camera-active");

    setDisplay("poseBadge", "block");

    state.running = true;
    state.lastVideoTime = -1;
    state.lastMetricFeature = null;

    setStatus("Camera on", true);

    setDisabled("stopButton", false);

    setMessage("Camera is on. Loading pose tracking...");

    try {
      await loadPoseModel();

      if (!state.running) return;

      setDisabled("startObservationButton", false);

      setMessage(
        "Camera and pose tracking are ready. Begin an observation when ready."
      );

      renderLoop();

    } catch (error) {
      console.error("MediaPipe model loading failed:", error);

      setMessage(
        "Camera is on, but pose tracking failed to load. Check your internet connection and Console."
      );

      setText("poseBadge", "Pose model unavailable");
    }

  } catch (error) {
    console.error("Camera startup failed:", error);

    state.stream?.getTracks().forEach((track) => track.stop());

    state.stream = null;
    video.srcObject = null;
    state.running = false;

    setStatus("Camera unavailable");

    if (error.name === "NotAllowedError") {
      setMessage(
        "Camera permission was denied. Allow camera access in Chrome site settings."
      );
    } else if (error.name === "NotFoundError") {
      setMessage("No camera was found.");
    } else if (error.name === "NotReadableError") {
      setMessage(
        "The camera may be in use by another application. Close other camera apps and try again."
      );
    } else {
      setMessage(error.message || "Could not start the camera.");
    }

    button.disabled = false;
  }
}

// ============================================================
// EXPERIMENTAL MOVEMENT-CHANGE DETECTION
// ============================================================

function classifyMovement(features) {
  const priorSamples = state.samples.filter((sample) => {
    const elapsed = features.time - sample.time;

    return (
      elapsed >= SETTINGS.comparisonMinimumMs &&
      elapsed <= SETTINGS.comparisonMaximumMs
    );
  });

  if (!priorSamples.length) {
    state.candidateStreak = 0;
    return { event: false };
  }

  const reference = priorSamples[priorSamples.length - 1];

  const elapsedSeconds =
    (features.time - reference.time) / 1000;

  if (elapsedSeconds <= 0) {
    state.candidateStreak = 0;
    return { event: false };
  }

  const hipDelta = Math.hypot(
    features.hipX - reference.hipX,
    features.hipY - reference.hipY
  );

  const angleDelta = angleDifference(
    features.torsoAngle,
    reference.torsoAngle
  );

  const hipSpeed = hipDelta / elapsedSeconds;
  const angleSpeed = angleDelta / elapsedSeconds;

  const rapidPositionChange =
    hipSpeed > SETTINGS.hipSpeedThreshold;

  const rapidTorsoChange =
    angleSpeed > SETTINGS.torsoSpeedThreshold &&
    angleDelta > SETTINGS.torsoAngleThreshold;

  const candidate =
    features.visibility >= SETTINGS.visibilityMinimum &&
    (rapidPositionChange || rapidTorsoChange);

  // Require multiple consecutive candidate frames to reduce
  // sensitivity to isolated frame-to-frame noise.
  if (candidate) {
    state.candidateStreak++;
  } else {
    state.candidateStreak = 0;
  }

  const enoughFrames =
    state.candidateStreak >= SETTINGS.requiredCandidateFrames;

  const cooldownPassed =
    features.time - state.lastEventAt >
    SETTINGS.eventCooldownMs;

  if (
    enoughFrames &&
    cooldownPassed &&
    state.observing
  ) {
    state.lastEventAt = features.time;

    let detail;
    let short;

    if (rapidPositionChange && rapidTorsoChange) {
      detail =
        "A rapid hip-center shift and torso-angle change were observed.";
      short = "sudden hip shift and upper-body lean";
    } else if (rapidPositionChange) {
      detail =
        "A rapid change in detected hip-center position was observed.";
      short = "sudden shift of the hips";
    } else {
      detail =
        "A rapid change in detected torso angle was observed.";
      short = "sudden lean of the upper body";
    }

    state.events.push({
      at: new Date().toISOString(),
      seconds: (features.time - state.startAt) / 1000,
      label: "Candidate movement change",
      detail,
      short,
      hipSpeed,
      angleSpeed,
      visibility: features.visibility
    });

    setText("eventCount", state.events.length);
    setText("eventHeadline", "Movement change detected");

    setText(
      "eventDescription",
      `${detail} This is an experimental movement flag, not a fall prediction.`
    );

    setText("eventCardTitle", "Candidate event recorded");

    setText(
      "eventCardText",
      `${detail} Review the context; this is not a medical assessment.`
    );

    const card = $("eventCard");

    if (card) {
      card.style.borderColor = "#e6c8ad";
      card.style.background = "#fff8f1";
    }

    // Reset streak so one continuous movement doesn't generate
    // another event on every subsequent frame.
    state.candidateStreak = 0;
  }

  return {
    event: candidate,
    hipDelta,
    angleDelta,
    hipSpeed,
    angleSpeed
  };
}

// ============================================================
// LIVE VIDEO LOOP
// ============================================================

function renderLoop() {
  if (!state.running || !state.landmarker) return;

  try {
    if (
      video.readyState >= 2 &&
      video.currentTime !== state.lastVideoTime
    ) {
      state.lastVideoTime = video.currentTime;

      const now = performance.now();

      const result = state.landmarker.detectForVideo(video, now);

      state.totalFrames++;

      const landmarks = result.landmarks?.[0];

      if (landmarks) {
        drawPose(landmarks);

        const features = extractFeatures(landmarks, now);

        if (features) {
          if (features.visibility >= SETTINGS.visibilityMinimum) {
            state.goodFrames++;
          }

          updateHipMetric(features);

          const movement = classifyMovement(features);

          state.samples.push(features);

          // Keep a lighter whole-observation record (5 per second) for the summary.
          if (
            state.observing &&
            features.visibility >= SETTINGS.visibilityMinimum &&
            now - state.lastTimelineAt >= 200
          ) {
            state.lastTimelineAt = now;

            state.timeline.push({
              t: (now - state.startAt) / 1000,
              left: features.leftKneeAngle,
              right: features.rightKneeAngle,
              lean: Math.abs(features.torsoAngle)
            });
          }

          state.samples = state.samples.filter(
            (sample) =>
              now - sample.time < SETTINGS.sampleWindowMs
          );

          setText(
            "poseBadge",
            features.visibility >= SETTINGS.visibilityMinimum
              ? "Body detected"
              : "Improve visibility"
          );

          setText(
            "torsoValue",
            `${Math.round(features.torsoAngle)}°`
          );

          setText(
            "kneeValue",
            `${Math.round(features.leftKneeAngle)}° / ${Math.round(features.rightKneeAngle)}°`
          );

          setText(
            "visibilityValue",
            `${Math.round(features.visibility * 100)}%`
          );

          const qualityBar = $("qualityBar");

          if (qualityBar) {
            qualityBar.style.width =
              `${Math.round(features.visibility * 100)}%`;
          }

          setText(
            "qualityLabel",
            features.visibility >= 0.75
              ? "Good visibility"
              : features.visibility >= SETTINGS.visibilityMinimum
                ? "Limited visibility"
                : "Low visibility"
          );

          if (!state.observing) {
            setText("eventHeadline", "Body detected");

            setText(
              "eventDescription",
              "Pose tracking is working. Begin an observation to record candidate movement changes."
            );
          } else if (!movement.event) {
            setText("eventHeadline", "Monitoring movement");

            setText(
              "eventDescription",
              "No candidate movement change currently meets the experimental rule."
            );
          }
        }
      } else {
        ctx?.clearRect(0, 0, canvas.width, canvas.height);

        state.lastMetricFeature = null;
        state.candidateStreak = 0;

        setText("poseBadge", "No body detected");
        setText("qualityLabel", "Not available");
        setText("visibilityValue", "—");
        setText("hipValue", "—");
        setText("kneeValue", "—");

        const qualityBar = $("qualityBar");

        if (qualityBar) {
          qualityBar.style.width = "0%";
        }
      }
    }
  } catch (error) {
    console.error("Pose inference failed:", error);

    setMessage(
      "Pose tracking encountered an error. Check the browser Console."
    );
  }

  if (state.observing) {
    setText(
      "elapsedTime",
      formatElapsed(performance.now() - state.startAt)
    );
  }

  if (state.running) {
    state.raf = requestAnimationFrame(renderLoop);
  }
}

// ============================================================
// OBSERVATION CONTROLS
// ============================================================

function beginObservation() {
  if (!state.stream || !state.running || !state.landmarker) {
    setMessage(
      "Start the camera and wait for pose tracking to finish loading."
    );

    return;
  }

  state.observing = true;
  state.startAt = performance.now();

  state.samples = [];
  state.events = [];

  state.lastEventAt = 0;
  state.candidateStreak = 0;
  state.lastMetricFeature = null;

  state.endAt = 0;
  state.timeline = [];
  state.lastTimelineAt = 0;
  state.summary = null;
  renderSummary();

  state.totalFrames = 0;
  state.goodFrames = 0;

  setText("eventCount", "0");
  setText("elapsedTime", "00:00");

  setDisabled("saveObservationButton", false);
  setDisabled("startObservationButton", true);

  setText("eventHeadline", "Monitoring movement");
  setText("eventCardTitle", "No observation events yet");

  setText(
    "eventCardText",
    "Experimental movement changes may be recorded for review."
  );

  const card = $("eventCard");

  if (card) {
    card.style.borderColor = "";
    card.style.background = "";
  }

  setMessage(
    "Observation started. Remain in a safe, comfortable position. No sudden movements are needed."
  );
}

function stopEverything() {
  const wasObserving = state.observing;

  if (wasObserving) {
    state.endAt = performance.now();
  }

  state.observing = false;
  state.running = false;

  if (state.raf !== null) {
    cancelAnimationFrame(state.raf);
  }

  state.raf = null;

  state.stream?.getTracks().forEach((track) => track.stop());
  state.stream = null;

  if (video) {
    video.pause();
    video.srcObject = null;
    video.style.display = "none";
  }

  if (ctx && canvas) {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
  }

  setDisplay("cameraPlaceholder", "block");
  setDisplay("poseBadge", "none");

  setText("poseBadge", "Waiting for camera");

  setDisabled("startCameraButton", false);
  setDisabled("startObservationButton", true);
  setDisabled("stopButton", true);

  setStatus("Camera off");
  setMessage("Camera stopped. Camera frames are not saved.");

  if (state.totalFrames > 0) {
    setDisabled("saveObservationButton", false);
  }

  state.lastMetricFeature = null;
  state.candidateStreak = 0;

  if (wasObserving) {
    state.summary = buildSummary();
    renderSummary();
    $("summaryPanel")?.scrollIntoView({ behavior: "smooth", block: "start" });
  }
}

// ============================================================
// OBSERVATION SUMMARY
// ============================================================

const SUMMARY_LEVELS = {
  none: {
    label: "No warning signs seen",
    icon: "✓",
    text: "The camera didn't pick up any sudden movements or unusual posture during this observation."
  },
  some: {
    label: "Some warning signs",
    icon: "!",
    text: "A few things stood out. They often happen during normal activity, but they're worth keeping an eye on."
  },
  several: {
    label: "Several warning signs",
    icon: "!!",
    text: "Several things stood out. Consider repeating the observation and talking with a doctor or physical therapist about balance and mobility."
  },
  unclear: {
    label: "Not enough data",
    icon: "?",
    text: "The camera couldn't see the body clearly for long enough. Try again with the full body in view for at least 10 seconds."
  }
};

function percentile(values, p) {
  if (!values.length) {
    return null;
  }

  const sorted = [...values].sort((a, b) => a - b);
  const index = Math.min(
    sorted.length - 1,
    Math.max(0, Math.round((p / 100) * (sorted.length - 1)))
  );

  return sorted[index];
}

function smoothedExtremes(values) {
  // A 1-second rolling median drops single-frame tracking glitches
  // but keeps real movements that last a second or more.
  const smoothed = values.map((_, index) =>
    percentile(values.slice(Math.max(0, index - 2), index + 3), 50)
  );

  return {
    min: Math.min(...smoothed),
    max: Math.max(...smoothed)
  };
}

function formatClock(seconds) {
  const whole = Math.max(0, Math.round(seconds));

  return `${Math.floor(whole / 60)}:${String(whole % 60).padStart(2, "0")}`;
}

function formatDuration(seconds) {
  const whole = Math.max(0, Math.round(seconds));
  const minutes = Math.floor(whole / 60);

  return minutes
    ? `${minutes} min ${whole % 60} s`
    : `${whole} s`;
}

function buildSummary() {
  const end = state.endAt || performance.now();
  const durationSeconds = state.startAt
    ? Math.max(0, (end - state.startAt) / 1000)
    : 0;

  const quality = state.totalFrames
    ? state.goodFrames / state.totalFrames
    : 0;

  const points = state.timeline;

  const left = points.map((point) => point.left);
  const right = points.map((point) => point.right);
  const lean = points.map((point) => point.lean);

  const leftKnee = smoothedExtremes(left);
  const rightKnee = smoothedExtremes(right);

  const knees = points.length
    ? {
      leftMin: leftKnee.min,
      leftMax: leftKnee.max,
      rightMin: rightKnee.min,
      rightMax: rightKnee.max
    }
    : null;

  const typicalLean = percentile(lean, 50);
  const maxLean = points.length ? smoothedExtremes(lean).max : null;

  const signs = [];

  if (state.events.length) {
    signs.push(
      `${state.events.length} sudden movement${state.events.length === 1 ? "" : "s"} flagged`
    );
  }

  /*
   * These thresholds are experimental placeholders, like the ones in
   * classifyMovement. They are not clinically derived.
   */
  if (maxLean !== null && maxLean > 20) {
    signs.push(`Upper body leaned far from upright (up to ${Math.round(maxLean)}°)`);
  }

  if (knees) {
    const leftRange = knees.leftMax - knees.leftMin;
    const rightRange = knees.rightMax - knees.rightMin;

    if (
      Math.max(leftRange, rightRange) > 20 &&
      Math.abs(leftRange - rightRange) > 15
    ) {
      signs.push(
        `Knees moved unevenly (left bent through ${Math.round(leftRange)}°, right through ${Math.round(rightRange)}°)`
      );
    }
  }

  // Each flagged event counts as its own sign.
  const signCount = state.events.length + signs.length - (state.events.length ? 1 : 0);

  let level;

  if (durationSeconds < 10 || quality < 0.5 || points.length < 10) {
    level = "unclear";
  } else if (signCount === 0) {
    level = "none";
  } else if (signCount <= 2) {
    level = "some";
  } else {
    level = "several";
  }

  return {
    level,
    durationSeconds: Math.round(durationSeconds),
    quality: Math.round(quality * 100),
    signs,
    events: state.events.map((event) => ({
      seconds: event.seconds,
      short: event.short
    })),
    knees: knees && {
      leftMin: Math.round(knees.leftMin),
      leftMax: Math.round(knees.leftMax),
      rightMin: Math.round(knees.rightMin),
      rightMax: Math.round(knees.rightMax)
    },
    typicalLean: typicalLean === null ? null : Math.round(typicalLean),
    maxLean: maxLean === null ? null : Math.round(maxLean)
  };
}

function qualityText(quality) {
  const word = quality >= 75 ? "Good" : quality >= 55 ? "Fair" : "Poor";

  return `${word}: the full body was clearly visible ${quality}% of the time`;
}

function summaryFacts(summary) {
  const facts = [
    ["Length", formatDuration(summary.durationSeconds)],
    ["Tracking quality", qualityText(summary.quality)]
  ];

  facts.push([
    "Sudden movements",
    summary.events.length
      ? summary.events
        .map((event) => `at ${formatClock(event.seconds)}: ${event.short}`)
        .join("\n")
      : "None flagged"
  ]);

  if (summary.knees) {
    facts.push([
      "Knees",
      `Left moved between ${summary.knees.leftMin}° and ${summary.knees.leftMax}°, ` +
      `right between ${summary.knees.rightMin}° and ${summary.knees.rightMax}° ` +
      "(180° is a straight leg)"
    ]);
  }

  if (summary.maxLean !== null) {
    facts.push([
      "Upper body",
      summary.maxLean <= 10
        ? `Stayed upright (leaned ${summary.maxLean}° at most)`
        : `Usually leaned about ${summary.typicalLean}°, up to ${summary.maxLean}° at most (0° is straight up)`
    ]);
  }

  return facts;
}

function levelBadge(level) {
  const info = SUMMARY_LEVELS[level] || SUMMARY_LEVELS.unclear;

  return `
    <span class="level-badge level-${escapeHtml(level)}">
      <span class="level-icon" aria-hidden="true">${info.icon}</span>
      ${escapeHtml(info.label)}
    </span>
  `;
}

function renderSummary() {
  const panel = $("summaryPanel");
  const summary = state.summary;

  if (!panel) return;

  if (!summary) {
    panel.hidden = true;
    return;
  }

  const info = SUMMARY_LEVELS[summary.level];

  const facts = summaryFacts(summary)
    .map(([name, value]) => `
      <div class="summary-fact">
        <dt>${escapeHtml(name)}</dt>
        <dd>${escapeHtml(value).replace(/\n/g, "<br>")}</dd>
      </div>
    `)
    .join("");

  const signs = summary.signs.length
    ? `
      <h4>What stood out</h4>
      <ul class="summary-signs">
        ${summary.signs.map((sign) => `<li>${escapeHtml(sign)}</li>`).join("")}
      </ul>
    `
    : "";

  const hasChart = state.timeline.length >= 2;

  panel.innerHTML = `
    <div class="summary-head">
      <span class="card-kicker">OBSERVATION SUMMARY</span>
      ${levelBadge(summary.level)}
      <p class="summary-lead">${escapeHtml(info.text)}</p>
    </div>

    ${signs}

    <h4>What we measured</h4>
    <dl class="summary-facts">${facts}</dl>

    ${hasChart ? `
      <h4>Over time</h4>
      <div class="summary-charts">
        <div class="time-chart" id="kneeChart"></div>
        <div class="time-chart" id="leanChart"></div>
      </div>
      <details class="summary-table">
        <summary>Show as table</summary>
        <div class="table-scroll">${timelineTable()}</div>
      </details>
    ` : ""}

    <p class="feature-footnote">
      These are camera-based observations, not a medical diagnosis or a
      fall prediction. Talk to a healthcare professional about any concerns.
    </p>
  `;

  panel.hidden = false;

  if (hasChart) {
    drawSummaryCharts();
  }
}

function drawSummaryCharts() {
  const duration = state.summary.durationSeconds || 1;
  const events = state.summary.events;

  const kneeValues = state.timeline.flatMap((point) => [point.left, point.right]);

  drawTimeChart($("kneeChart"), {
    title: "Knee angle (180° = straight leg)",
    series: [
      { key: "left", name: "Left knee", color: "#2a78d6" },
      { key: "right", name: "Right knee", color: "#eb6834" }
    ],
    yMin: Math.max(0, Math.floor((Math.min(...kneeValues) - 5) / 10) * 10),
    yMax: 180,
    unit: "°",
    duration,
    events
  });

  const leanValues = state.timeline.map((point) => point.lean);

  drawTimeChart($("leanChart"), {
    title: "Upper-body lean (0° = straight up)",
    series: [
      { key: "lean", name: "Lean", color: "#426f5d" }
    ],
    yMin: 0,
    yMax: Math.max(30, Math.ceil((Math.max(...leanValues) + 5) / 10) * 10),
    unit: "°",
    duration,
    events
  });
}

function svgEl(tag, attrs = {}) {
  const el = document.createElementNS("http://www.w3.org/2000/svg", tag);

  Object.entries(attrs).forEach(([name, value]) => {
    el.setAttribute(name, value);
  });

  return el;
}

function drawTimeChart(container, options) {
  const points = state.timeline;

  container.innerHTML = "";

  const heading = document.createElement("div");
  heading.className = "chart-heading";

  const title = document.createElement("strong");
  title.textContent = options.title;
  heading.append(title);

  // Legend only when there's more than one line to tell apart.
  if (options.series.length > 1) {
    const legend = document.createElement("div");
    legend.className = "chart-legend";

    options.series.forEach((series) => {
      const item = document.createElement("span");
      const key = document.createElement("i");
      key.style.background = series.color;
      item.append(key, document.createTextNode(series.name));
      legend.append(item);
    });

    heading.append(legend);
  }

  container.append(heading);

  const width = Math.max(280, container.clientWidth || 600);
  const height = 170;
  const margin = { top: 14, right: 16, bottom: 26, left: 40 };
  const plotW = width - margin.left - margin.right;
  const plotH = height - margin.top - margin.bottom;

  const x = (seconds) => margin.left + (seconds / options.duration) * plotW;
  const y = (value) => margin.top +
    (1 - (value - options.yMin) / (options.yMax - options.yMin)) * plotH;

  const svg = svgEl("svg", {
    width,
    height,
    viewBox: `0 0 ${width} ${height}`,
    role: "img",
    "aria-label": `${options.title} over the observation`
  });

  // Gridlines and y-axis labels on round numbers.
  const tickStep = [5, 10, 20, 30, 45].find(
    (candidate) => (options.yMax - options.yMin) / candidate <= 5
  ) || 60;

  for (let value = options.yMin; value <= options.yMax; value += tickStep) {
    const yPos = y(value);

    svg.append(svgEl("line", {
      x1: margin.left, x2: width - margin.right,
      y1: yPos, y2: yPos,
      class: value === options.yMin ? "chart-axis" : "chart-grid"
    }));

    const label = svgEl("text", {
      x: margin.left - 6, y: yPos + 4,
      "text-anchor": "end",
      class: "chart-tick"
    });
    label.textContent = `${Math.round(value)}${options.unit}`;
    svg.append(label);
  }

  // X-axis time labels.
  // Leave at least ~50px between time labels.
  const step = [5, 10, 15, 30, 60, 120, 300].find(
    (candidate) => (plotW / options.duration) * candidate >= 50
  ) || 600;

  for (let seconds = 0; seconds <= options.duration; seconds += step) {
    const label = svgEl("text", {
      x: x(seconds), y: height - 6,
      "text-anchor": seconds === 0 ? "start" : "middle",
      class: "chart-tick"
    });
    label.textContent = formatClock(seconds);
    svg.append(label);
  }

  // Flagged sudden movements as dashed markers.
  options.events.forEach((event) => {
    svg.append(svgEl("line", {
      x1: x(event.seconds), x2: x(event.seconds),
      y1: margin.top, y2: margin.top + plotH,
      class: "chart-event"
    }));

    const flag = svgEl("text", {
      x: x(event.seconds) + 4, y: margin.top + 10,
      class: "chart-event-label"
    });
    flag.textContent = "!";
    svg.append(flag);
  });

  // Lines break where tracking dropped out for more than a second.
  options.series.forEach((series) => {
    let d = "";
    let previous = null;

    points.forEach((point) => {
      const command = !previous || point.t - previous.t > 1 ? "M" : "L";
      d += `${command}${x(point.t).toFixed(1)},${y(point[series.key]).toFixed(1)}`;
      previous = point;
    });

    svg.append(svgEl("path", {
      d,
      fill: "none",
      stroke: series.color,
      "stroke-width": 2,
      "stroke-linejoin": "round",
      "stroke-linecap": "round"
    }));
  });

  // Hover crosshair and tooltip.
  const crosshair = svgEl("line", {
    y1: margin.top, y2: margin.top + plotH,
    class: "chart-crosshair",
    visibility: "hidden"
  });
  svg.append(crosshair);

  const dots = options.series.map((series) => {
    const dot = svgEl("circle", {
      r: 4,
      fill: series.color,
      stroke: "#fff",
      "stroke-width": 2,
      visibility: "hidden"
    });
    svg.append(dot);
    return dot;
  });

  const hit = svgEl("rect", {
    x: margin.left, y: 0,
    width: plotW, height,
    fill: "transparent"
  });
  svg.append(hit);

  const wrap = document.createElement("div");
  wrap.className = "chart-wrap";

  const tooltip = document.createElement("div");
  tooltip.className = "chart-tooltip";
  tooltip.hidden = true;

  wrap.append(svg, tooltip);
  container.append(wrap);

  const hide = () => {
    crosshair.setAttribute("visibility", "hidden");
    dots.forEach((dot) => dot.setAttribute("visibility", "hidden"));
    tooltip.hidden = true;
  };

  hit.addEventListener("pointerleave", hide);

  hit.addEventListener("pointermove", (pointerEvent) => {
    const box = svg.getBoundingClientRect();
    const seconds = ((pointerEvent.clientX - box.left - margin.left) / plotW) *
      options.duration;

    let nearest = points[0];

    points.forEach((point) => {
      if (Math.abs(point.t - seconds) < Math.abs(nearest.t - seconds)) {
        nearest = point;
      }
    });

    const xPos = x(nearest.t);

    crosshair.setAttribute("x1", xPos);
    crosshair.setAttribute("x2", xPos);
    crosshair.setAttribute("visibility", "visible");

    dots.forEach((dot, index) => {
      dot.setAttribute("cx", xPos);
      dot.setAttribute("cy", y(nearest[options.series[index].key]));
      dot.setAttribute("visibility", "visible");
    });

    tooltip.textContent = "";

    const time = document.createElement("div");
    time.className = "chart-tooltip-time";
    time.textContent = `at ${formatClock(nearest.t)}`;
    tooltip.append(time);

    options.series.forEach((series) => {
      const row = document.createElement("div");
      const key = document.createElement("i");
      key.style.background = series.color;
      const value = document.createElement("strong");
      value.textContent = `${Math.round(nearest[series.key])}${options.unit}`;
      row.append(key, value, document.createTextNode(` ${series.name}`));
      tooltip.append(row);
    });

    const flagged = options.events.find(
      (event) => Math.abs(event.seconds - nearest.t) <= 0.5
    );

    if (flagged) {
      const note = document.createElement("div");
      note.className = "chart-tooltip-event";
      note.textContent = `Flagged: ${flagged.short}`;
      tooltip.append(note);
    }

    tooltip.hidden = false;

    // Flip to the left of the crosshair near the right edge.
    let left = xPos + 12;

    if (left + tooltip.offsetWidth > width) {
      left = xPos - tooltip.offsetWidth - 12;
    }

    tooltip.style.left = `${Math.max(0, left)}px`;
    tooltip.style.top = `${margin.top}px`;
  });
}

function timelineTable() {
  // One row per second keeps the table readable.
  const rows = [];
  let nextSecond = 0;

  state.timeline.forEach((point) => {
    if (point.t >= nextSecond) {
      rows.push(`
        <tr>
          <td>${formatClock(point.t)}</td>
          <td>${Math.round(point.left)}°</td>
          <td>${Math.round(point.right)}°</td>
          <td>${Math.round(point.lean)}°</td>
        </tr>
      `);
      nextSecond = Math.floor(point.t) + 1;
    }
  });

  return `
    <table>
      <thead>
        <tr><th>Time</th><th>Left knee</th><th>Right knee</th><th>Upper-body lean</th></tr>
      </thead>
      <tbody>${rows.join("")}</tbody>
    </table>
  `;
}

// ============================================================
// LOCAL HISTORY
// ============================================================

function getHistory() {
  try {
    return JSON.parse(
      localStorage.getItem(HISTORY_KEY) || "[]"
    );
  } catch (error) {
    console.error("Could not read observation history:", error);
    return [];
  }
}

function saveObservation() {
  // Saving mid-observation summarizes everything up to now.
  if (state.observing || !state.summary) {
    state.summary = buildSummary();
    renderSummary();
  }

  const duration = state.startAt
    ? Math.max(0, (state.endAt || performance.now()) - state.startAt)
    : 0;

  const quality = state.totalFrames
    ? state.goodFrames / state.totalFrames
    : 0;

  const item = {
    id: globalThis.crypto?.randomUUID
      ? crypto.randomUUID()
      : String(Date.now()),

    date: new Date().toISOString(),

    durationSeconds: Math.round(duration / 1000),

    candidateEvents: state.events.length,

    usableFrameRatio: Math.round(quality * 100),

    summary: state.summary,

    eventSummaries: state.events.map((event) => ({
      at: event.at,
      label: event.label,
      detail: event.detail,
      hipSpeed: event.hipSpeed,
      angleSpeed: event.angleSpeed,
      visibility: event.visibility
    })),

    note:
      "Experimental movement-change flags are not a fall prediction or medical assessment."
  };

  const history = getHistory();

  history.unshift(item);

  localStorage.setItem(
    HISTORY_KEY,
    JSON.stringify(history.slice(0, 25))
  );

  renderHistory();

  setMessage(
    "Observation summary saved locally in this browser. No camera frames were saved."
  );
}

function renderHistory() {
  const list = $("historyList");

  if (!list) return;

  const items = getHistory();

  if (!items.length) {
    list.innerHTML = `
      <div class="empty-state">
        <div class="empty-icon" aria-hidden="true">↗</div>
        <h3>Your saved observations will appear here</h3>
        <p>Run an observation and save its summary to begin a local history.</p>
      </div>
    `;

    return;
  }

  list.innerHTML = items.map((item) => {
    const date = new Date(item.date).toLocaleString();

    const seconds = Number(item.durationSeconds) || 0;

    const duration =
      `${Math.floor(seconds / 60)}m ${seconds % 60}s`;

    // Older saved items have no summary; keep their original wording.
    const details = item.summary
      ? `
          ${levelBadge(item.summary.level)}
          <p>${escapeHtml(date)} · ${duration}</p>
          ${item.summary.signs.length
            ? `<p>${item.summary.signs.map(escapeHtml).join(" · ")}</p>`
            : ""}
          ${summaryFacts(item.summary).slice(1).map(([name, value]) =>
            `<p><strong>${escapeHtml(name)}:</strong> ${escapeHtml(value).replace(/\n/g, "; ")}</p>`
          ).join("")}
        `
      : `
          <p>
            ${escapeHtml(date)} · ${duration} ·
            ${Number(item.usableFrameRatio) || 0}% frames met visibility threshold
          </p>
          <p>${Number(item.candidateEvents) || 0} candidate movement event(s)</p>
        `;

    return `
      <article class="history-item">
        <div>
          <h3>Movement observation</h3>
          ${details}
        </div>

        <div class="history-meta">
          <strong>Local summary</strong>

          <button
            class="button button-quiet export-one"
            data-id="${escapeHtml(item.id)}"
            type="button"
          >
            Export JSON
          </button>
        </div>
      </article>
    `;
  }).join("");

  list.querySelectorAll(".export-one").forEach((button) => {
    button.addEventListener("click", () => {
      exportOne(button.dataset.id);
    });
  });
}

function downloadJson(data, filename) {
  const blob = new Blob(
    [JSON.stringify(data, null, 2)],
    { type: "application/json" }
  );

  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");

  link.href = url;
  link.download = filename;

  document.body.appendChild(link);
  link.click();
  link.remove();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportOne(id) {
  const item = getHistory().find(
    (entry) => entry.id === id
  );

  if (item) {
    downloadJson(
      item,
      `motus-observation-${id}.json`
    );
  }
}

// ============================================================
// OPTIONAL SCREENING QUESTIONNAIRE
// ============================================================

function submitScreening(event) {
  event.preventDefault();

  const form = new FormData(event.currentTarget);

  const answers = {
    fallen: form.get("fallen"),
    unsteady: form.get("unsteady"),
    worried: form.get("worried"),
    date: new Date().toISOString()
  };

  state.questionnaire = answers;

  localStorage.setItem(
    SCREENING_KEY,
    JSON.stringify(answers)
  );

  setText(
    "screeningFeedback",
    "Responses saved on this device. No risk score was calculated."
  );
}

// ============================================================
// DEMO DIALOG
// ============================================================

function setupDemo() {
  const dialog = $("demoDialog");

  if (!dialog) return;

  $("demoButton")?.addEventListener("click", () => {
    if (typeof dialog.showModal === "function") {
      dialog.showModal();
    }
  });

  $("closeDemoButton")?.addEventListener("click", () => {
    dialog.close();
  });

  $("closeDemoButton2")?.addEventListener("click", () => {
    dialog.close();
  });
}

// ============================================================
// UTILITIES
// ============================================================

function formatElapsed(milliseconds) {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  return (
    `${String(minutes).padStart(2, "0")}:` +
    `${String(remainingSeconds).padStart(2, "0")}`
  );
}

// ============================================================
// INITIALIZATION
// ============================================================

function initializeApp() {
  const requiredIds = [
    "cameraVideo",
    "overlayCanvas",
    "startCameraButton",
    "startObservationButton",
    "stopButton",
    "saveObservationButton",
    "screeningForm",
    "clearHistoryButton",
    "historyList"
  ];

  const missingIds = requiredIds.filter((id) => !$(id));

  if (missingIds.length) {
    console.error(
      "Motus is missing required HTML elements:",
      missingIds
    );

    setMessage(
      `Page setup error: missing elements ${missingIds.join(", ")}`
    );

    return;
  }

  $("startCameraButton").addEventListener(
    "click",
    startCamera
  );

  $("startObservationButton").addEventListener(
    "click",
    beginObservation
  );

  $("stopButton").addEventListener(
    "click",
    stopEverything
  );

  $("saveObservationButton").addEventListener(
    "click",
    saveObservation
  );

  $("screeningForm").addEventListener(
    "submit",
    submitScreening
  );

  $("clearHistoryButton").addEventListener("click", () => {
    if (
      confirm(
        "Delete all saved observation summaries from this browser?"
      )
    ) {
      localStorage.removeItem(HISTORY_KEY);
      renderHistory();
    }
  });

  window.addEventListener("beforeunload", () => {
    state.stream?.getTracks().forEach((track) => track.stop());
  });

  window.addEventListener("resize", () => {
    if (video?.readyState >= 2) {
      resizeCanvas();
    }

    if (state.summary && state.timeline.length >= 2) {
      drawSummaryCharts();
    }
  });

  setupDemo();
  renderHistory();

  console.info("Motus JavaScript initialized successfully.");
}

initializeApp();
