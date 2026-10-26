import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/vision_bundle.mjs";

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

  if (
    !leftShoulder ||
    !rightShoulder ||
    !leftHip ||
    !rightHip
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
    torsoAngle
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
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.32/wasm"
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

    if (rapidPositionChange && rapidTorsoChange) {
      detail =
        "A rapid hip-center shift and torso-angle change were observed.";
    } else if (rapidPositionChange) {
      detail =
        "A rapid change in detected hip-center position was observed.";
    } else {
      detail =
        "A rapid change in detected torso angle was observed.";
    }

    state.events.push({
      at: new Date().toISOString(),
      label: "Candidate movement change",
      detail,
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
  const duration = state.startAt
    ? Math.max(0, performance.now() - state.startAt)
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

    return `
      <article class="history-item">
        <div>
          <h3>Movement observation</h3>
          <p>
            ${escapeHtml(date)} · ${duration} ·
            ${Number(item.usableFrameRatio) || 0}% frames met visibility threshold
          </p>
          <p>${Number(item.candidateEvents) || 0} candidate movement event(s)</p>
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
  });

  setupDemo();
  renderHistory();

  console.info("Motus JavaScript initialized successfully.");
}

initializeApp();
