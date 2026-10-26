import {
  FilesetResolver,
  PoseLandmarker
} from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21";

const $ = (id) => document.getElementById(id);

const video = $("cameraVideo");
const canvas = $("overlayCanvas");
const ctx = canvas.getContext("2d");

const state = {
  stream: null,
  landmarker: null,
  raf: null,

  running: false,
  observing: false,

  lastVideoTime: -1,
  startAt: 0,

  samples: [],
  events: [],
  lastEventAt: 0,

  currentFeatures: null,
  visibility: 0,

  totalFrames: 0,
  goodFrames: 0,

  questionnaire: null
};

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

function setMessage(text) {
  $("cameraMessage").textContent = text;
}

function setStatus(text, active = false) {
  $("cameraStatus").innerHTML =
    `<span class="status-dot"></span> ${text}`;

  $("cameraStatus").classList.toggle("active", active);
}

function angleDegrees(hip, shoulder) {
  // Approximate torso angle relative to vertical in image coordinates.
  const dx = shoulder.x - hip.x;
  const dy = shoulder.y - hip.y;

  return Math.atan2(dx, -dy) * 180 / Math.PI;
}

function meanVisibility(landmarks, indices) {
  const values = indices
    .map((index) => Number(landmarks[index]?.visibility ?? 0))
    .filter(Number.isFinite);

  if (!values.length) {
    return 0;
  }

  return values.reduce((sum, value) => sum + value, 0) / values.length;
}

function resizeCanvas() {
  const rect = video.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;

  canvas.width = Math.max(1, Math.round(rect.width * dpr));
  canvas.height = Math.max(1, Math.round(rect.height * dpr));

  canvas.style.width = `${rect.width}px`;
  canvas.style.height = `${rect.height}px`;

  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
}

function drawPose(landmarks) {
  resizeCanvas();

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  if (!landmarks) {
    return;
  }

  const width = video.clientWidth;
  const height = video.clientHeight;

  // Mirror the landmark overlay to match the mirrored video preview.
  ctx.save();
  ctx.translate(width, 0);
  ctx.scale(-1, 1);

  ctx.lineWidth = 3;
  ctx.strokeStyle = "rgba(221, 242, 221, .92)";
  ctx.fillStyle = "rgba(74, 125, 94, .95)";

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

function extractFeatures(landmarks, now) {
  const important = [
    11, 12, 23, 24, 25, 26, 27, 28
  ];

  const visibility = meanVisibility(landmarks, important);

  const leftShoulder = landmarks[11];
  const rightShoulder = landmarks[12];

  const leftHip = landmarks[23];
  const rightHip = landmarks[24];

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

  return {
    time: now,
    visibility,
    hipX: hip.x,
    hipY: hip.y,
    torsoAngle
  };
}

function classifyMovement(features) {
  const recentSamples = state.samples.filter(
    (sample) => features.time - sample.time <= 1000
  );

  const priorSamples = recentSamples.filter((sample) => {
    const elapsed = features.time - sample.time;

    return elapsed >= 120 && elapsed <= 350;
  });

  if (!priorSamples.length) {
    return {
      label: "Tracking movement",
      detail: "Collecting enough frames to compare movement over time.",
      event: false
    };
  }

  const reference = priorSamples[priorSamples.length - 1];
  const elapsedSeconds = (features.time - reference.time) / 1000;

  if (elapsedSeconds <= 0) {
    return {
      label: "Tracking movement",
      detail: "Collecting movement frames.",
      event: false
    };
  }

  // Position change is measured in normalized image coordinates.
  const hipDelta = Math.hypot(
    features.hipX - reference.hipX,
    features.hipY - reference.hipY
  );

  let angleDelta = Math.abs(
    features.torsoAngle - reference.torsoAngle
  );

  // Account for wraparound in angle measurements.
  if (angleDelta > 180) {
    angleDelta = 360 - angleDelta;
  }

  const hipChangePerSecond = hipDelta / elapsedSeconds;
  const angleChangePerSecond = angleDelta / elapsedSeconds;

  /*
   * These thresholds are experimental placeholders.
   * They are not clinically derived and are not a fall prediction.
   */
  const rapidPositionChange = hipChangePerSecond > 0.42;

  const rapidTorsoChange =
    angleChangePerSecond > 75 &&
    angleDelta > 13;

  const candidate =
    features.visibility >= 0.55 &&
    (rapidPositionChange || rapidTorsoChange);

  /*
   * Record candidate events only during an active observation.
   * This prevents alerts from accumulating before the user starts.
   */
  if (
    candidate &&
    state.observing &&
    features.time - state.lastEventAt > 3000
  ) {
    state.lastEventAt = features.time;

    let explanation;

    if (rapidPositionChange && rapidTorsoChange) {
      explanation =
        "A rapid hip-center shift and torso-angle change were detected.";
    } else if (rapidPositionChange) {
      explanation =
        "A rapid change in detected hip-center position was observed.";
    } else {
      explanation =
        "A rapid change in detected torso angle was observed.";
    }

    state.events.push({
      at: new Date().toISOString(),
      label: "Candidate movement change",
      detail: explanation
    });

    $("eventCount").textContent = String(state.events.length);

    $("eventHeadline").textContent = "Movement change detected";

    $("eventDescription").textContent =
      explanation +
      " This can happen for many reasons and does not mean a fall is imminent.";

    $("eventCardTitle").textContent = "Candidate event recorded";

    $("eventCardText").textContent =
      explanation +
      " Review the context; this is not a fall prediction.";

    $("eventCard").style.borderColor = "#e6c8ad";
    $("eventCard").style.background = "#fff8f1";
  }

  return {
    label: candidate ? "Movement change detected" : "Tracking movement",
    detail: "Exploratory image-based movement features.",
    event: candidate,
    hipDelta,
    angleDelta
  };
}

async function loadPoseModel() {
  if (state.landmarker) {
    return;
  }

  setMessage("Loading the pose model. This may take a moment the first time.");

  const vision = await FilesetResolver.forVisionTasks(
    "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.21/wasm"
  );

  state.landmarker = await PoseLandmarker.createFromOptions(vision, {
    baseOptions: {
      modelAssetPath:
        "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_lite/float16/1/pose_landmarker_lite.task"
    },

    runningMode: "VIDEO",
    numPoses: 1,

    minPoseDetectionConfidence: 0.55,
    minPosePresenceConfidence: 0.55,
    minTrackingConfidence: 0.55
  });
}

async function startCamera() {
  try {
    if (!navigator.mediaDevices?.getUserMedia) {
      throw new Error(
        "Camera access requires a secure context. Open this site using localhost or HTTPS."
      );
    }

    $("startCameraButton").disabled = true;

    await loadPoseModel();

    state.stream = await navigator.mediaDevices.getUserMedia({
      video: {
        facingMode: "user",
        width: { ideal: 1280 },
        height: { ideal: 720 }
      },
      audio: false
    });

    video.srcObject = state.stream;

    await video.play();

    video.style.display = "block";
    $("cameraPlaceholder").style.display = "none";
    $("cameraStage").classList.add("camera-active");
    $("poseBadge").style.display = "block";

    $("startObservationButton").disabled = false;
    $("stopButton").disabled = false;

    state.running = true;
    state.lastVideoTime = -1;

    setStatus("Camera on", true);

    setMessage(
      "Camera is on. Keep the person in view; begin an observation when ready."
    );

    renderLoop();

  } catch (error) {
    console.error(error);

    setMessage(
      error?.message ||
      "Could not start the camera. Check browser permissions and try again."
    );

    $("startCameraButton").disabled = false;

    setStatus("Camera unavailable");
  }
}

function renderLoop() {
  if (!state.running) {
    return;
  }

  if (
    video.readyState >= 2 &&
    video.currentTime !== state.lastVideoTime
  ) {
    state.lastVideoTime = video.currentTime;

    const now = performance.now();

    try {
      const result = state.landmarker.detectForVideo(video, now);

      state.totalFrames++;

      const landmarks = result.landmarks?.[0];

      if (landmarks) {
        drawPose(landmarks);

        const features = extractFeatures(landmarks, now);

        if (features) {
          state.visibility = features.visibility;

          if (features.visibility >= 0.55) {
            state.goodFrames++;
          }

          state.currentFeatures = features;

          const movement = classifyMovement(features);

          state.samples.push(features);

          // Retain only a short recent window in memory.
          state.samples = state.samples.filter(
            (sample) => now - sample.time < 2500
          );

          $("poseBadge").textContent =
            features.visibility >= 0.55
              ? "Body detected"
              : "Improve visibility";

          $("torsoValue").textContent =
            `${Math.round(features.torsoAngle)}°`;

          $("hipValue").textContent =
            `${Math.hypot(features.hipX, features.hipY).toFixed(3)} image units`;

          $("visibilityValue").textContent =
            `${Math.round(features.visibility * 100)}%`;

          $("qualityBar").style.width =
            `${Math.round(features.visibility * 100)}%`;

          $("qualityLabel").textContent =
            features.visibility >= 0.75
              ? "Good"
              : features.visibility >= 0.55
                ? "Limited"
                : "Low";

          if (!state.observing) {
            $("eventHeadline").textContent = "Body detected";

            $("eventDescription").textContent =
              "Pose tracking is working. Begin an observation to record candidate movement changes.";

          } else if (!movement.event) {
            $("eventHeadline").textContent = "Monitoring movement";

            $("eventDescription").textContent =
              "No candidate movement change detected in this frame. This does not mean the person is safe from falling.";
          }
        }

      } else {
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        $("poseBadge").textContent = "No body detected";
        $("qualityLabel").textContent = "Not available";
        $("qualityBar").style.width = "0%";
      }

    } catch (error) {
      console.error("Pose inference failed:", error);
    }
  }

  if (state.observing) {
    $("elapsedTime").textContent =
      formatElapsed(performance.now() - state.startAt);
  }

  state.raf = requestAnimationFrame(renderLoop);
}

function formatElapsed(milliseconds) {
  const seconds = Math.floor(milliseconds / 1000);
  const minutes = Math.floor(seconds / 60);
  const remainingSeconds = seconds % 60;

  return (
    `${String(minutes).padStart(2, "0")}:` +
    `${String(remainingSeconds).padStart(2, "0")}`
  );
}

function beginObservation() {
  if (!state.stream || !state.running) {
    return;
  }

  state.observing = true;
  state.startAt = performance.now();

  state.samples = [];
  state.events = [];
  state.lastEventAt = 0;

  state.totalFrames = 0;
  state.goodFrames = 0;

  $("eventCount").textContent = "0";
  $("elapsedTime").textContent = "00:00";

  $("saveObservationButton").disabled = false;
  $("startObservationButton").disabled = true;

  $("eventCardTitle").textContent = "No observation events yet";

  $("eventCardText").textContent =
    "The app will flag experimental movement changes that cross its current thresholds.";

  $("eventCard").style.borderColor = "";
  $("eventCard").style.background = "";

  setMessage(
    "Observation started. Stay in a safe, comfortable position; do not attempt to trigger an event."
  );
}

function stopEverything() {
  const wasObserving = state.observing;

  state.observing = false;
  state.running = false;

  if (state.raf) {
    cancelAnimationFrame(state.raf);
  }

  state.raf = null;

  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }

  state.stream = null;

  video.pause();
  video.srcObject = null;
  video.style.display = "none";

  ctx.clearRect(0, 0, canvas.width, canvas.height);

  $("cameraPlaceholder").style.display = "block";
  $("poseBadge").style.display = "none";

  $("startCameraButton").disabled = false;
  $("startObservationButton").disabled = true;
  $("stopButton").disabled = true;

  // Keep Save enabled if a completed observation has data.
  $("saveObservationButton").disabled =
    !(wasObserving || state.totalFrames > 0);

  setStatus("Camera off");

  setMessage(
    "Camera stopped. Camera frames are not saved by this prototype."
  );

  $("poseBadge").textContent = "Waiting for camera";
}

function saveObservation() {
  const duration = state.startAt
    ? Math.max(0, performance.now() - state.startAt)
    : 0;

  const quality = state.totalFrames
    ? state.goodFrames / state.totalFrames
    : 0;

  const item = {
    id: crypto.randomUUID
      ? crypto.randomUUID()
      : String(Date.now()),

    date: new Date().toISOString(),

    durationSeconds: Math.round(duration / 1000),

    candidateEvents: state.events.length,

    usableFrameRatio: Math.round(quality * 100),

    eventSummaries: state.events.map((event) => ({
      at: event.at,
      label: event.label,
      detail: event.detail
    })),

    note:
      "Experimental movement-change flags are not a fall prediction or medical assessment."
  };

  const history = getHistory();

  history.unshift(item);

  localStorage.setItem(
    "mobilitymind_history",
    JSON.stringify(history.slice(0, 25))
  );

  renderHistory();

  setMessage(
    "Observation summary saved locally in this browser. No camera frames were saved."
  );
}

function getHistory() {
  try {
    return JSON.parse(
      localStorage.getItem("mobilitymind_history") || "[]"
    );
  } catch {
    return [];
  }
}

function renderHistory() {
  const list = $("historyList");
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
          <p>
            ${Number(item.candidateEvents) || 0} candidate movement event(s)
          </p>
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

function escapeHtml(value) {
  return String(value).replace(/[&<>"']/g, (character) => ({
    "&": "&amp;",
    "<": "&lt;",
    ">": "&gt;",
    '"': "&quot;",
    "'": "&#39;"
  }[character]));
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
  link.click();

  setTimeout(() => URL.revokeObjectURL(url), 1000);
}

function exportOne(id) {
  const item = getHistory().find((entry) => entry.id === id);

  if (item) {
    downloadJson(
      item,
      `mobilitymind-observation-${id}.json`
    );
  }
}

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
    "mobilitymind_screening",
    JSON.stringify(answers)
  );

  $("screeningFeedback").textContent =
    "Responses saved on this device. No risk score was calculated.";
}

function setupDemo() {
  const dialog = $("demoDialog");

  $("demoButton").addEventListener("click", () => {
    dialog.showModal();
  });

  $("closeDemoButton").addEventListener("click", () => {
    dialog.close();
  });

  $("closeDemoButton2").addEventListener("click", () => {
    dialog.close();
  });
}

$("startCameraButton").addEventListener("click", startCamera);

$("startObservationButton").addEventListener(
  "click",
  beginObservation
);

$("stopButton").addEventListener("click", stopEverything);

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
    localStorage.removeItem("mobilitymind_history");
    renderHistory();
  }
});

window.addEventListener("beforeunload", () => {
  if (state.stream) {
    state.stream.getTracks().forEach((track) => track.stop());
  }
});

window.addEventListener("resize", () => {
  if (video.readyState >= 2) {
    resizeCanvas();
  }
});

setupDemo();
renderHistory();
