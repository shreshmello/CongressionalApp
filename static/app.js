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
  endAt: 0,

  timeline: [],
  lastTimelineAt: 0,
  summary: null,

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

  const leftKnee = landmarks[25];
  const rightKnee = landmarks[26];

  const leftAnkle = landmarks[27];
  const rightAnkle = landmarks[28];

  if (
    !leftShoulder ||
    !rightShoulder ||
    !leftHip ||
    !rightHip ||
    !rightKnee ||
    !leftKnee ||
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
    let short;

    if (rapidPositionChange && rapidTorsoChange) {
      explanation =
        "A rapid hip-center shift and torso-angle change were detected.";
      short = "sudden hip shift and upper-body lean";
    } else if (rapidPositionChange) {
      explanation =
        "A rapid change in detected hip-center position was observed.";
      short = "sudden shift of the hips";
    } else {
      explanation =
        "A rapid change in detected torso angle was observed.";
      short = "sudden lean of the upper body";
    }

    state.events.push({
      at: new Date().toISOString(),
      seconds: (features.time - state.startAt) / 1000,
      label: "Candidate movement change",
      detail: explanation,
      short
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

          // Keep a lighter whole-observation record (5 per second) for the summary.
          if (
            state.observing &&
            features.visibility >= 0.55 &&
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

          $("kneeValue").textContent =
            `${Math.round(features.leftKneeAngle)}° / ${Math.round(features.rightKneeAngle)}°`;

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

  state.endAt = 0;
  state.timeline = [];
  state.lastTimelineAt = 0;
  state.summary = null;
  renderSummary();

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

  if (wasObserving) {
    state.endAt = performance.now();
  }

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

  if (wasObserving) {
    state.summary = buildSummary();
    renderSummary();
    $("summaryPanel").scrollIntoView({ behavior: "smooth", block: "start" });
  }

  $("poseBadge").textContent = "Waiting for camera";
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
    id: crypto.randomUUID
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
          <p>
            ${Number(item.candidateEvents) || 0} candidate movement event(s)
          </p>
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

  if (state.summary && state.timeline.length >= 2) {
    drawSummaryCharts();
  }
});

setupDemo();
renderHistory();
