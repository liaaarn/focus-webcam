/**
 * FocusWebCam V2 — focus.js
 * ===========================
 * Uses Logistic Regression model trained from focus_model.pkl
 * Integrated with Ethical Guardrails
 * 
 * Model coefficients from training_report.txt:
 *   ear         : +1.0494  ↑ increases FOCUS
 *   head_pose   : -2.6625  ↓ decreases FOCUS
 *   mouth_ratio : +2.0005  ↑ increases FOCUS
 */

// ─────────────────────────────────────────────
// MODEL PARAMETERS (from Logistic Regression training)
// ─────────────────────────────────────────────
const MODEL = {
  coef: {
    ear: 1.0494,
    head_pose: -2.6625,
    mouth_ratio: 2.0005
  },
  intercept: -0.5234,
  scaler: {
    ear: { mean: 0.214, std: 0.098 },
    head_pose: { mean: 0.178, std: 0.245 },
    mouth_ratio: { mean: 0.068, std: 0.082 }
  },
  threshold: 0.5
};

// ─────────────────────────────────────────────
// Visual & alert hyperparameters
// ─────────────────────────────────────────────
const CONFIG = {
  ALERT_THRESHOLD: 40,
  ALERT_DURATION: 5,
  ALERT_COOLDOWN: 30,
  STATS_INTERVAL: 1000,
  EAR_OPEN: 0.25,
  EAR_CLOSED: 0.15,
  SMOOTHING_WINDOW: 3
};

// ─────────────────────────────────────────────
// Landmark indices MediaPipe FaceMesh
// ─────────────────────────────────────────────
const LANDMARKS = {
  LEFT_EYE: [362, 385, 387, 263, 373, 380],
  RIGHT_EYE: [33, 160, 158, 133, 153, 144],
  MOUTH_TOP: 13,
  MOUTH_BOTTOM: 14,
  MOUTH_LEFT: 78,
  MOUTH_RIGHT: 308,
  NOSE_TIP: 1,
  FACE_LEFT: 234,
  FACE_RIGHT: 454,
};

// ─────────────────────────────────────────────
// DOM references
// ─────────────────────────────────────────────
const elScore = document.getElementById('scoreNumber');
const elScoreBar = document.getElementById('scoreBarFill');
const elScoreState = document.getElementById('scoreState');
const elEAR = document.getElementById('earValue');
const elHead = document.getElementById('headValue');
const elMouth = document.getElementById('mouthValue');
const elEARBar = document.getElementById('earBar');
const elHeadBar = document.getElementById('headBar');
const elMouthBar = document.getElementById('mouthBar');
const elFaceStatus = document.getElementById('faceStatus');
const elHeaderSt = document.getElementById('headerStatus');
const elDuration = document.getElementById('statDuration');
const elAvg = document.getElementById('statAvg');
const elFocusTime = document.getElementById('statFocusTime');
const elAlerts = document.getElementById('statAlerts');
const elLogList = document.getElementById('logList');
const elToast = document.getElementById('toast');
const elToastMsg = document.getElementById('toastMsg');
const elBtnStart = document.getElementById('btnStart');
const elCameraFrame = document.getElementById('cameraFrame');
const webcamEl = document.getElementById('webcam');
const overlayCanvas = document.getElementById('overlayCanvas');
const ctx = overlayCanvas ? overlayCanvas.getContext('2d') : null;

// ─────────────────────────────────────────────
// Application state
// ─────────────────────────────────────────────
let isRunning = false;
let sessionHistory = [];
let alertCount = 0;
let lowScoreCount = 0;
let lastAlertTime = -Infinity;
let sessionStart = null;
let statsTimer = null;
let camera = null;
let lastScores = [];
let frameCounter = 0;

// Ethical Guardrails instances
let explainabilityEngine = null;
let biasMitigation = null;
let privacyGuard = null;
let safetyOverride = null;
let ethicalCompliance = { consentObtained: false, explainabilityEnabled: true };

// ─────────────────────────────────────────────
// Feature calculation utilities
// ─────────────────────────────────────────────

function dist(a, b) {
  return Math.hypot(a.x - b.x, a.y - b.y);
}

function calcEAR(lm, indices) {
  const p = indices.map(i => lm[i]);
  const A = dist(p[1], p[5]);
  const B = dist(p[2], p[4]);
  const C = dist(p[0], p[3]);
  return C === 0 ? 0 : (A + B) / (2 * C);
}

function calcHeadPose(lm) {
  const nose = lm[LANDMARKS.NOSE_TIP];
  const left = lm[LANDMARKS.FACE_LEFT];
  const right = lm[LANDMARKS.FACE_RIGHT];
  const faceCenter = (left.x + right.x) / 2;
  const faceWidth = Math.abs(right.x - left.x);
  if (faceWidth === 0) return 0;
  return Math.abs(nose.x - faceCenter) / faceWidth;
}

function calcMouth(lm) {
  const top = lm[LANDMARKS.MOUTH_TOP];
  const bottom = lm[LANDMARKS.MOUTH_BOTTOM];
  const left = lm[LANDMARKS.MOUTH_LEFT];
  const right = lm[LANDMARKS.MOUTH_RIGHT];
  const vertical = Math.abs(top.y - bottom.y);
  const horizontal = Math.abs(left.x - right.x);
  return horizontal === 0 ? 0 : vertical / horizontal;
}

// ─────────────────────────────────────────────
// MODEL INFERENCE
// ─────────────────────────────────────────────

function standardize(value, mean, std) {
  return (value - mean) / std;
}

function sigmoid(z) {
  return 1 / (1 + Math.exp(-z));
}

function predictProbability(ear, headPose, mouth) {
  const ear_scaled = standardize(ear, MODEL.scaler.ear.mean, MODEL.scaler.ear.std);
  const head_scaled = standardize(headPose, MODEL.scaler.head_pose.mean, MODEL.scaler.head_pose.std);
  const mouth_scaled = standardize(mouth, MODEL.scaler.mouth_ratio.mean, MODEL.scaler.mouth_ratio.std);
  
  const logit = MODEL.coef.ear * ear_scaled + 
                MODEL.coef.head_pose * head_scaled + 
                MODEL.coef.mouth_ratio * mouth_scaled + 
                MODEL.intercept;
  
  return sigmoid(logit);
}

function smoothScore(newScore) {
  lastScores.push(newScore);
  if (lastScores.length > CONFIG.SMOOTHING_WINDOW) lastScores.shift();
  return lastScores.reduce((a, b) => a + b, 0) / lastScores.length;
}

// ─────────────────────────────────────────────
// Normalization for display (VISUAL ONLY)
// ─────────────────────────────────────────────

function normalizeEARForDisplay(ear) {
  const clamped = Math.max(CONFIG.EAR_CLOSED, Math.min(CONFIG.EAR_OPEN, ear));
  return ((clamped - CONFIG.EAR_CLOSED) / (CONFIG.EAR_OPEN - CONFIG.EAR_CLOSED)) * 100;
}

function normalizeHeadForDisplay(headPose) {
  const clamped = Math.min(headPose, 0.3);
  return (1 - clamped / 0.3) * 100;
}

function normalizeMouthForDisplay(mouth) {
  const clamped = Math.max(0, Math.min(0.5, mouth));
  return (1 - clamped / 0.5) * 100;
}

// ─────────────────────────────────────────────
// Ethical Integration - Calculate with Ethics
// ─────────────────────────────────────────────

function calculateFocusScoreWithEthics(ear, headPose, mouth, brightness = 0.5) {
  if (biasMitigation) {
    ear = biasMitigation.compensateLighting(ear, brightness);
  }
  
  const probability = predictProbability(ear, headPose, mouth);
  const rawScore = probability * 100;
  const smoothedScore = smoothScore(rawScore);
  const finalScore = Math.min(100, Math.max(0, Math.round(smoothedScore)));
  
  let explanation = null;
  if (explainabilityEngine && ethicalCompliance.explainabilityEnabled) {
    explanation = explainabilityEngine.explainPrediction(ear, headPose, mouth, probability, finalScore);
    showExplanationPanel(explanation);
  }
  
  if (safetyOverride) {
    const hazards = safetyOverride.detectSafetyHazard(ear, lowScoreCount, frameCounter);
    hazards.forEach(hazard => {
      safetyOverride.requestIntervention(hazard.message);
      addLog(hazard.message, 'log-alert');
    });
  }
  
  let fairnessWarning = null;
  if (biasMitigation) {
    const fairness = biasMitigation.validateFairness(ear, headPose, mouth);
    if (!fairness.isFair && fairness.warnings.length > 0) {
      fairnessWarning = fairness.warnings[0];
      if (frameCounter % 300 === 0) {
        addLog(`⚠️ Fairness: ${fairnessWarning}`, 'log-system');
      }
    }
  }
  
  const earNorm = normalizeEARForDisplay(ear);
  const headNorm = normalizeHeadForDisplay(headPose);
  const mouthNorm = normalizeMouthForDisplay(mouth);
  
  return { score: finalScore, rawProbability: probability, earNorm, headNorm, mouthNorm, explanation, fairnessWarning };
}

// ─────────────────────────────────────────────
// Explanation Panel
// ─────────────────────────────────────────────

let currentExplanationPanel = null;
let explanationTimeout = null;

function showExplanationPanel(explanation) {
  if (!explanation) return;
  
  if (explanation.score >= 65 && explanation.factors.negative.length === 0) {
    if (currentExplanationPanel) currentExplanationPanel.remove();
    return;
  }
  
  if (!currentExplanationPanel) {
    currentExplanationPanel = document.createElement('div');
    currentExplanationPanel.className = 'explanation-panel';
    document.body.appendChild(currentExplanationPanel);
  }
  
  currentExplanationPanel.innerHTML = `
    <h4>📊 Why score ${explanation.score}?</h4>
    <p>${explanation.explanation}</p>
    <div class="suggestion">💡 ${explanation.suggestion}</div>
    <div style="margin-top: 8px; font-size: 0.6rem; color: var(--text-dim);">
      👁 ${explanation.contributions.eye}% | ↻ ${explanation.contributions.head}% | 👄 ${explanation.contributions.mouth}%
    </div>
  `;
  
  if (explanationTimeout) clearTimeout(explanationTimeout);
  if (explanation.score >= 65) {
    explanationTimeout = setTimeout(() => {
      if (currentExplanationPanel) currentExplanationPanel.remove();
      currentExplanationPanel = null;
    }, 5000);
  }
}

// ─────────────────────────────────────────────
// Alert Engine
// ─────────────────────────────────────────────

function checkAlert(score) {
  const now = Date.now() / 1000;
  if (score < CONFIG.ALERT_THRESHOLD) {
    lowScoreCount++;
  } else {
    lowScoreCount = 0;
  }
  
  if (lowScoreCount >= CONFIG.ALERT_DURATION && now - lastAlertTime >= CONFIG.ALERT_COOLDOWN) {
    triggerAlert(score);
    lastAlertTime = now;
    lowScoreCount = 0;
  }
}

function triggerAlert(score) {
  alertCount++;
  elAlerts.textContent = alertCount;
  elToastMsg.textContent = `Score ${score} for ${CONFIG.ALERT_DURATION} seconds`;
  elToast.classList.add('show');
  playBeep();
  addLog(`⚠ Alert #${alertCount} — score ${score} (model prediction)`, 'log-alert');
  setTimeout(() => elToast.classList.remove('show'), 4000);
}

function playBeep() {
  try {
    const ctx = new AudioContext();
    const osc = ctx.createOscillator();
    const gain = ctx.createGain();
    osc.connect(gain);
    gain.connect(ctx.destination);
    osc.frequency.value = 520;
    osc.type = 'sine';
    gain.gain.setValueAtTime(0.3, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.6);
    osc.start(ctx.currentTime);
    osc.stop(ctx.currentTime + 0.6);
  } catch (e) {}
}

// ─────────────────────────────────────────────
// Update UI
// ─────────────────────────────────────────────

function updateUI(score, earNorm, headNorm, mouthNorm, ear, headPose, mouth) {
  elScore.textContent = score;
  elScoreBar.style.width = score + '%';
  
  elCameraFrame.classList.remove('state-focus', 'state-medium', 'state-unfocus');
  if (score >= 65) {
    elCameraFrame.classList.add('state-focus');
    elScoreState.textContent = 'FOCUS';
    elFaceStatus.textContent = 'Face detected — Focused';
  } else if (score >= 40) {
    elCameraFrame.classList.add('state-medium');
    elScoreState.textContent = 'ATTENTION';
    elFaceStatus.textContent = 'Face detected — Attention needed';
  } else {
    elCameraFrame.classList.add('state-unfocus');
    elScoreState.textContent = 'NOT FOCUSED';
    elFaceStatus.textContent = 'Face detected — Not focused';
  }
  
  elEAR.textContent = ear.toFixed(3);
  elHead.textContent = headPose.toFixed(3);
  elMouth.textContent = mouth.toFixed(3);
  
  elEARBar.style.width = Math.min(earNorm, 100) + '%';
  elHeadBar.style.width = Math.min(headNorm, 100) + '%';
  elMouthBar.style.width = Math.min(mouthNorm, 100) + '%';
  
  elEARBar.style.background = earNorm > 50 ? 'var(--accent)' : 'var(--accent2)';
  elHeadBar.style.background = headNorm > 50 ? 'var(--accent)' : 'var(--accent2)';
  elMouthBar.style.background = mouthNorm > 50 ? 'var(--accent)' : 'var(--accent3)';
}

function updateNoFace() {
  elScore.textContent = '0';
  elScoreBar.style.width = '0%';
  elScoreState.textContent = 'NO FACE';
  elFaceStatus.textContent = 'No face detected';
  elCameraFrame.classList.remove('state-focus', 'state-medium', 'state-unfocus');
  elEAR.textContent = '—';
  elHead.textContent = '—';
  elMouth.textContent = '—';
  lastScores = [];
}

// ─────────────────────────────────────────────
// Draw overlay landmarks
// ─────────────────────────────────────────────

function drawLandmarks(lm) {
  if (!ctx || !overlayCanvas) return;
  const canvas = overlayCanvas;
  const video = webcamEl;
  if (video.videoWidth === 0) return;
  
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  
  const w = canvas.width;
  const h = canvas.height;
  
  // Draw eye landmarks
  ctx.beginPath();
  ctx.strokeStyle = '#00ff88';
  ctx.lineWidth = 1;
  ctx.fillStyle = '#00ff88';
  
  [...LANDMARKS.LEFT_EYE, ...LANDMARKS.RIGHT_EYE].forEach(idx => {
    const point = lm[idx];
    if (point) {
      ctx.beginPath();
      ctx.arc(point.x * w, point.y * h, 1.5, 0, 2 * Math.PI);
      ctx.fill();
    }
  });
  
  // Draw face bounding box indicator
  const leftFace = lm[LANDMARKS.FACE_LEFT];
  const rightFace = lm[LANDMARKS.FACE_RIGHT];
  const topFace = lm[10];
  const bottomFace = lm[152];
  
  if (leftFace && rightFace && topFace && bottomFace) {
    ctx.beginPath();
    ctx.strokeStyle = 'rgba(0, 255, 136, 0.3)';
    ctx.lineWidth = 1;
    const x1 = leftFace.x * w;
    const x2 = rightFace.x * w;
    const y1 = topFace.y * h;
    const y2 = bottomFace.y * h;
    ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
  }
}

// ─────────────────────────────────────────────
// Stats update
// ─────────────────────────────────────────────

function updateStats() {
  if (!isRunning || !sessionStart) return;
  
  const elapsed = Math.floor((Date.now() - sessionStart) / 1000);
  const mm = String(Math.floor(elapsed / 60)).padStart(2, '0');
  const ss = String(elapsed % 60).padStart(2, '0');
  elDuration.textContent = `${mm}:${ss}`;
  
  if (sessionHistory.length > 0) {
    const avg = sessionHistory.reduce((a, b) => a + b, 0) / sessionHistory.length;
    elAvg.textContent = Math.round(avg);
    const focusCount = sessionHistory.filter(s => s >= CONFIG.ALERT_THRESHOLD).length;
    const pct = Math.round((focusCount / sessionHistory.length) * 100);
    elFocusTime.textContent = pct + '%';
  }
}

function addLog(msg, cls = '') {
  const now = new Date();
  const time = now.toTimeString().slice(0, 8);
  const el = document.createElement('div');
  el.className = `log-item ${cls}`;
  el.textContent = `[${time}] ${msg}`;
  elLogList.prepend(el);
  const items = elLogList.querySelectorAll('.log-item');
  if (items.length > 50) items[items.length - 1].remove();
}

// ─────────────────────────────────────────────
// MediaPipe FaceMesh setup
// ─────────────────────────────────────────────

const faceMesh = new FaceMesh({
  locateFile: (file) => `https://cdn.jsdelivr.net/npm/@mediapipe/face_mesh/${file}`
});

faceMesh.setOptions({
  maxNumFaces: 1,
  refineLandmarks: true,
  minDetectionConfidence: 0.5,
  minTrackingConfidence: 0.5,
});

faceMesh.onResults((results) => {
  if (!isRunning) return;
  
  frameCounter++;
  
  if (!results.multiFaceLandmarks || results.multiFaceLandmarks.length === 0) {
    updateNoFace();
    sessionHistory.push(0);
    checkAlert(0);
    if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
    return;
  }
  
  const lm = results.multiFaceLandmarks[0];
  drawLandmarks(lm);
  
  const earLeft = calcEAR(lm, LANDMARKS.LEFT_EYE);
  const earRight = calcEAR(lm, LANDMARKS.RIGHT_EYE);
  const ear = (earLeft + earRight) / 2;
  const headPose = calcHeadPose(lm);
  const mouth = calcMouth(lm);
  
  const { score, earNorm, headNorm, mouthNorm } = calculateFocusScoreWithEthics(ear, headPose, mouth);
  
  updateUI(score, earNorm, headNorm, mouthNorm, ear, headPose, mouth);
  sessionHistory.push(score);
  checkAlert(score);
});

// ─────────────────────────────────────────────
// Ethical Guardrails Initialization
// ─────────────────────────────────────────────

async function initEthicalGuardrails() {
  privacyGuard = new EthicalGuardrails.PrivacyGuard();
  ethicalCompliance.consentObtained = await privacyGuard.requestConsent();
  
  if (!ethicalCompliance.consentObtained) {
    addLog('❌ User denied privacy consent. Data will not be stored.', 'log-alert');
    elHeaderSt.textContent = 'PRIVACY MODE (Limited)';
    return false;
  }
  
  explainabilityEngine = new EthicalGuardrails.ExplainabilityEngine(MODEL.coef);
  biasMitigation = new EthicalGuardrails.BiasMitigation();
  safetyOverride = new EthicalGuardrails.SafetyOverride();
  
  addLog('✅ Ethical Guardrails active: Transparency + Bias Mitigation + Privacy', 'log-focus');
  addLog(`📊 Model coefficients: EAR=${MODEL.coef.ear}, Head=${MODEL.coef.head_pose}, Mouth=${MODEL.coef.mouth_ratio}`, 'log-system');
  
  addPrivacyButton();
  return true;
}

function addPrivacyButton() {
  if (document.querySelector('.privacy-btn')) return;
  const btn = document.createElement('button');
  btn.className = 'privacy-btn';
  btn.innerHTML = '🔒';
  btn.title = 'Privacy Control (GDPR)';
  btn.onclick = () => { if (privacyGuard) privacyGuard.showPrivacyPanel(); };
  document.body.appendChild(btn);
}

// ─────────────────────────────────────────────
// Start / Stop Session - ENHANCED RESOLUTION
// ─────────────────────────────────────────────

async function startSession() {
  try {
    elHeaderSt.textContent = 'REQUESTING CAMERA ACCESS...';
    
    const stream = await navigator.mediaDevices.getUserMedia({
      vide: {
        width: { ideal: 640, max: 854 },
        height: { ideal: 480, max: 480 },
        frameRate: { ideal: 24, max: 30 },
        facingMode: 'user'
      }
    });
    
    webcamEl.srcObject = stream;
    await webcamEl.play();
    
    if (overlayCanvas) {
      overlayCanvas.width = webcamEl.videoWidth || 640;
      overlayCanvas.height = webcamEl.videoHeight || 480;
    }
    
    camera = new Camera(webcamEl, {
      onFrame: async () => { await faceMesh.send({ image: webcamEl }); },
      width: 640,
      height: 480,
    });
    camera.start();
    
    isRunning = true;
    sessionStart = Date.now();
    sessionHistory = [];
    alertCount = 0;
    lowScoreCount = 0;
    lastAlertTime = -Infinity;
    lastScores = [];
    frameCounter = 0;
    
    elBtnStart.textContent = 'END SESSION';
    elBtnStart.classList.add('active');
    elHeaderSt.textContent = 'SESSION ACTIVE (Logistic Regression)';
    elAlerts.textContent = '0';
    elAvg.textContent = '--';
    elFocusTime.textContent = '--%';
    elDuration.textContent = '00:00';
    
    statsTimer = setInterval(updateStats, 1000);
    addLog('🎯 Session started — Using trained Logistic Regression model', 'log-focus');
    
  } catch (err) {
    elHeaderSt.textContent = 'CAMERA ERROR';
    addLog(`❌ Error: ${err.message}`, 'log-alert');
    alert('Cannot access camera. Please grant camera permission.');
  }
}

function stopSession() {
  isRunning = false;
  
  if (camera) {
    camera.stop();
    camera = null;
  }
  
  if (webcamEl.srcObject) {
    webcamEl.srcObject.getTracks().forEach(t => t.stop());
    webcamEl.srcObject = null;
  }
  
  clearInterval(statsTimer);
  
  if (ctx) ctx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);
  
  elBtnStart.textContent = 'START SESSION';
  elBtnStart.classList.remove('active');
  elHeaderSt.textContent = 'SESSION ENDED';
  elCameraFrame.classList.remove('state-focus', 'state-medium', 'state-unfocus');
  elScore.textContent = '--';
  elScoreBar.style.width = '0%';
  elScoreState.textContent = '—';
  elFaceStatus.textContent = 'Waiting for face...';
  lastScores = [];
  
  if (privacyGuard && ethicalCompliance.consentObtained && sessionHistory.length > 0) {
    const avg = sessionHistory.reduce((a, b) => a + b, 0) / sessionHistory.length;
    const focusCount = sessionHistory.filter(s => s >= CONFIG.ALERT_THRESHOLD).length;
    const pct = Math.round((focusCount / sessionHistory.length) * 100);
    const elapsed = sessionStart ? Math.floor((Date.now() - sessionStart) / 1000) : 0;
    
    privacyGuard.storeSessionData({
      scores: sessionHistory,
      avgScore: avg,
      focusPercentage: pct,
      alertCount: alertCount,
      duration: elapsed
    });
    addLog('🔒 Session data stored anonymously (local, auto-delete in 24 hours)', 'log-system');
  }
  
  if (sessionHistory.length > 0) {
    const avg = Math.round(sessionHistory.reduce((a, b) => a + b, 0) / sessionHistory.length);
    const pct = Math.round(sessionHistory.filter(s => s >= CONFIG.ALERT_THRESHOLD).length / sessionHistory.length * 100);
    addLog(`📊 Session ended — avg ${avg}, focus ${pct}%, ${alertCount} alerts`, 'log-system');
  } else {
    addLog('Session ended', 'log-system');
  }
}

// ─────────────────────────────────────────────
// Event Listener & Initialization
// ─────────────────────────────────────────────

elBtnStart.addEventListener('click', () => {
  if (isRunning) {
    stopSession();
  } else {
    startSession();
  }
});

// Initialize
elHeaderSt.textContent = 'READY — Logistic Regression Model';
addLog('🤖 Model loaded — Logistic Regression from focus_model.pkl', 'log-system');

// Load ethical guardrails
initEthicalGuardrails();