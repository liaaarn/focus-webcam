"""
FocusWebCam — Streamlit App
============================
Menggunakan streamlit-webrtc untuk akses kamera real-time,
MediaPipe FaceMesh untuk deteksi wajah, dan model Logistic
Regression yang sudah dilatih (focus_model.pkl).

Cara jalankan:
  pip install streamlit streamlit-webrtc av opencv-python-headless mediapipe scikit-learn
  streamlit run app.py
"""

import streamlit as st
import cv2
import numpy as np
import time
import queue
from collections import deque
from datetime import datetime

import av
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

# ─────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="FocusWebCam | Ethical AI Focus Detection",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="collapsed",
)

import mediapipe as mp
mp_face_mesh = mp.solutions.face_mesh

# ─────────────────────────────────────────────
# Landmark indices
# ─────────────────────────────────────────────
LEFT_EYE   = [362, 385, 387, 263, 373, 380]
RIGHT_EYE  = [33,  160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
NOSE_TIP, FACE_LEFT, FACE_RIGHT = 1, 234, 454

# ─────────────────────────────────────────────
# Model parameters
# ─────────────────────────────────────────────
MODEL_COEF = {"ear": 1.0494, "head_pose": -2.6625, "mouth_ratio": 2.0005}
MODEL_INTERCEPT = -0.5234
MODEL_SCALER = {
    "ear":         {"mean": 0.214, "std": 0.098},
    "head_pose":   {"mean": 0.178, "std": 0.245},
    "mouth_ratio": {"mean": 0.068, "std": 0.082},
}
ALERT_THRESHOLD  = 40
SMOOTHING_WINDOW = 3

# ─────────────────────────────────────────────
# Feature helpers
# ─────────────────────────────────────────────
MOUTH_MAX_REALISTIC = 0.12
def calc_ear(lm, indices, w, h):
    pts = [(lm[i].x * w, lm[i].y * h) for i in indices]
    A = np.hypot(pts[1][0]-pts[5][0], pts[1][1]-pts[5][1])
    B = np.hypot(pts[2][0]-pts[4][0], pts[2][1]-pts[4][1])
    C = np.hypot(pts[0][0]-pts[3][0], pts[0][1]-pts[3][1])
    return (A + B) / (2.0 * C) if C else 0.0

def calc_head_pose(lm, w, h):
    nose  = lm[NOSE_TIP]
    left  = lm[FACE_LEFT]
    right = lm[FACE_RIGHT]
    face_center = (left.x + right.x) / 2
    face_width  = abs(right.x - left.x)
    return abs(nose.x - face_center) / face_width if face_width else 0.0

def calc_mouth(lm, w, h):
    top    = lm[MOUTH_TOP]
    bottom = lm[MOUTH_BOTTOM]
    left   = lm[MOUTH_LEFT]
    right  = lm[MOUTH_RIGHT]
    vertical   = abs((top.y - bottom.y) * h)
    horizontal = abs((left.x - right.x) * w)
    ratio = vertical / horizontal if horizontal else 0.0
    return min(ratio, MOUTH_MAX_REALISTIC)

def predict_probability(ear, head_pose, mouth):
    mouth = min(mouth, MOUTH_MAX_REALISTIC)  # ← batasi mouth
    
    ear_s   = standardize(ear, **MODEL_SCALER["ear"])
    head_s  = standardize(head_pose, **MODEL_SCALER["head_pose"])
    mouth_s = standardize(mouth, **MODEL_SCALER["mouth_ratio"])
    
    # Clamp semua nilai
    ear_s = max(-3, min(3, ear_s))
    head_s = max(-3, min(3, head_s))
    mouth_s = max(-3, min(3, mouth_s))
    
    logit = (MODEL_COEF["ear"] * ear_s +
             MODEL_COEF["head_pose"] * head_s +
             MODEL_COEF["mouth_ratio"] * mouth_s +
             MODEL_INTERCEPT)
    return float(sigmoid(logit))

def standardize(v, mean, std):
    return (v - mean) / std if std else 0.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_probability(ear, head_pose, mouth):
    ear_s   = standardize(ear,       **MODEL_SCALER["ear"])
    head_s  = standardize(head_pose, **MODEL_SCALER["head_pose"])
    mouth_s = standardize(mouth,     **MODEL_SCALER["mouth_ratio"])
    logit   = (MODEL_COEF["ear"]        * ear_s  +
               MODEL_COEF["head_pose"]  * head_s +
               MODEL_COEF["mouth_ratio"]* mouth_s +
               MODEL_INTERCEPT)
    return float(sigmoid(logit))

def get_cv_color(score):
    if score >= 65:  return (58, 140, 82)
    if score >= 40:  return (20, 128, 232)
    return (43, 57, 192)

def explain_score(ear, head, mouth, score):
    neg = []
    if ear   < 0.20: neg.append("mata tertutup/berkedip")
    if head  > 0.15: neg.append("kepala menoleh")
    if mouth > 0.08: neg.append("mulut terbuka")
    if score >= 65:
        return f"Fokus baik ({score}/100)"
    elif score >= 40:
        isu = ", ".join(neg) if neg else "pertahankan kondisi"
        return f"Perhatian ({score}/100) — {isu}"
    else:
        isu = ", ".join(neg) if neg else "kondisi tidak optimal"
        return f"Tidak fokus ({score}/100) — {isu}"

# ─────────────────────────────────────────────
# Queue
# ─────────────────────────────────────────────
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=5)
result_queue: queue.Queue = st.session_state.result_queue

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "page":             "landing",   # landing | app
        "session_active":   False,
        "session_start":    None,
        "score_history":    [],
        "alert_count":      0,
        "low_score_count":  0,
        "last_alert_time":  0,
        "log_entries":      [("system", "— Sistem siap —")],
        "consent_given":    False,
        "consent_asked":    False,
        "show_session_complete": False,
        "session_summary":  {},
        "disp_score":       None,
        "disp_ear":         None,
        "disp_head":        None,
        "disp_mouth":       None,
        "disp_expl":        "",
        "disp_face":        False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# Video Processor
# ─────────────────────────────────────────────
class FocusVideoProcessor:
    def __init__(self):
        self.face_mesh = mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._smooth = deque(maxlen=SMOOTHING_WINDOW)

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res  = self.face_mesh.process(rgb)

        if res.multi_face_landmarks:
            lm    = res.multi_face_landmarks[0].landmark
            ear_l = calc_ear(lm, LEFT_EYE,  w, h)
            ear_r = calc_ear(lm, RIGHT_EYE, w, h)
            ear   = (ear_l + ear_r) / 2.0
            head  = calc_head_pose(lm, w, h)
            mouth = calc_mouth(lm, w, h)

            prob  = predict_probability(ear, head, mouth)
            self._smooth.append(prob * 100)
            score = int(np.clip(round(np.mean(self._smooth)), 0, 100))
            color = get_cv_color(score)
            expl  = explain_score(ear, head, mouth, score)

            data = {"face": True, "score": score,
                    "ear": round(ear,4), "head": round(head,4),
                    "mouth": round(mouth,4), "expl": expl}
            try:
                result_queue.put_nowait(data)
            except queue.Full:
                try:    result_queue.get_nowait()
                except: pass
                try:    result_queue.put_nowait(data)
                except: pass

            # Draw overlay on video
            for idx in LEFT_EYE + RIGHT_EYE:
                pt = lm[idx]
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 2, color, -1)
            fl = lm[FACE_LEFT]; fr = lm[FACE_RIGHT]
            ft = lm[10];        fb = lm[152]
            cv2.rectangle(img,
                (int(fl.x*w), int(ft.y*h)),
                (int(fr.x*w), int(fb.y*h)), color, 1)

            # HUD panel
            overlay = img.copy()
            cv2.rectangle(overlay, (10,10), (215,100), (240,245,250), -1)
            cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)
            cv2.putText(img, f"FOCUS: {score}", (18,38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(img, f"EAR:{ear:.3f}  HEAD:{head:.3f}", (18,58),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80,90,110), 1)
            cv2.putText(img, f"MOUTH:{mouth:.3f}", (18,74),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.36, (80,90,110), 1)
            bar_w = int((score/100)*188)
            cv2.rectangle(img,(18,82),(206,92),(200,215,225),-1)
            cv2.rectangle(img,(18,82),(18+bar_w,92),color,-1)

            # Corner markers
            sz, t = 14, 2
            pts = [(int(fl.x*w), int(ft.y*h)), (int(fr.x*w), int(ft.y*h)),
                   (int(fl.x*w), int(fb.y*h)), (int(fr.x*w), int(fb.y*h))]
            for i,(px,py) in enumerate(pts):
                dx = 1 if i in (0,2) else -1
                dy = 1 if i in (0,1) else -1
                cv2.line(img,(px,py),(px+dx*sz,py),color,t)
                cv2.line(img,(px,py),(px,py+dy*sz),color,t)
        else:
            cv2.putText(img, "Tidak ada wajah terdeteksi", (20,40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.62, (100,120,140), 1)
            data = {"face": False, "score": 0,
                    "ear": None, "head": None, "mouth": None, "expl": ""}
            try:
                result_queue.put_nowait(data)
            except queue.Full:
                try:    result_queue.get_nowait()
                except: pass
                try:    result_queue.put_nowait(data)
                except: pass

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ─────────────────────────────────────────────
# Drain queue
# ─────────────────────────────────────────────
def drain_queue():
    latest = None
    while True:
        try:
            latest = result_queue.get_nowait()
        except queue.Empty:
            break
    if latest is None:
        return False

    st.session_state.disp_score = latest["score"]
    st.session_state.disp_ear   = latest["ear"]
    st.session_state.disp_head  = latest["head"]
    st.session_state.disp_mouth = latest["mouth"]
    st.session_state.disp_expl  = latest["expl"]
    st.session_state.disp_face  = latest["face"]

    if st.session_state.session_active:
        score = latest["score"]
        st.session_state.score_history.append(score)
        if score < ALERT_THRESHOLD:
            st.session_state.low_score_count += 1
        else:
            st.session_state.low_score_count = 0
        now = time.time()
        if (st.session_state.low_score_count >= 5 and
                now - st.session_state.last_alert_time >= 30):
            st.session_state.alert_count += 1
            st.session_state.last_alert_time = now
            st.session_state.low_score_count = 0
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_entries.insert(
                0, ("alert", f"⚠ [{ts}] Alert #{st.session_state.alert_count} — skor {score}"))
    return True

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Lusitana:wght@400;700&family=Kameron:wght@400;600;700&family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

:root {
  --bg:         #cfdce8;
  --land-dark:  #1a2433;
  --land-text:  #1e2d40;
  --surface:    rgba(255,255,255,0.55);
  --border:     rgba(30,45,64,0.12);
  --green:      #3a8c52;
  --orange:     #e8a020;
  --red:        #c0392b;
  --text-dim:   #6a7e92;
  --text-mid:   #8a9eb0;
  --font-kame:  'Kameron', serif;
  --font-lusi:  'Lusitana', serif;
  --font-mono:  'Space Mono', monospace;
  --font-syne:  'Syne', sans-serif;
}

html, body,
[data-testid="stAppViewContainer"],
[data-testid="stMain"],
.stMainBlockContainer,
[data-testid="block-container"] {
  background: var(--bg) !important;
  font-family: var(--font-syne) !important;
}

[data-testid="stHeader"],
[data-testid="stToolbar"],
#MainMenu, footer,
[data-testid="stSidebar"] { display:none !important; }

/* Remove default padding */
.stMainBlockContainer { padding: 0 !important; max-width: 100% !important; }
[data-testid="block-container"] { padding: 0 !important; }

/* ── Buttons ── */
.stButton > button {
  font-family: var(--font-kame) !important;
  font-size: 1.1rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  border-radius: 6px !important;
  padding: 11px 0 !important;
  width: 100% !important;
  transition: all 0.22s ease !important;
}
/* Default = green start */
.stButton > button {
  background: transparent !important;
  border: 1.5px solid var(--green) !important;
  color: var(--green) !important;
}
.stButton > button:hover {
  background: var(--green) !important;
  color: #fff !important;
}
/* Stop button wrapper */
.focusbtn-stop .stButton > button {
  border-color: var(--red) !important;
  color: var(--red) !important;
}
.focusbtn-stop .stButton > button:hover {
  background: var(--red) !important;
  color: #fff !important;
}

/* ── Cards ── */
.fwc-card {
  background: var(--surface);
  border: 1px solid var(--border);
  border-radius: 6px;
  backdrop-filter: blur(6px);
  padding: 14px 16px;
  margin-bottom: 10px;
}

/* ── Score card ── */
.score-label {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-dim);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 6px;
}
.score-number {
  font-family: var(--font-kame);
  font-size: 3rem;
  font-weight: 700;
  line-height: 1;
  transition: color 0.4s;
}
.score-unit {
  font-family: var(--font-mono);
  font-size: 0.9rem;
  color: var(--text-mid);
}
.score-bar-track {
  height: 8px;
  background: #dce8f0;
  border-radius: 4px;
  overflow: hidden;
  margin: 8px 0 6px;
}
.score-bar-fill {
  height: 100%;
  border-radius: 4px;
  transition: width 0.4s ease, background 0.4s ease;
}
.score-state {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--text-dim);
  letter-spacing: 0.1em;
  text-transform: uppercase;
}

/* ── Feature cards ── */
.feat-grid { display:grid; grid-template-columns:repeat(3,1fr); gap:6px; margin-bottom:10px; }
.feat-card {
  background: rgba(255,255,255,0.5);
  border: 1px solid rgba(30,45,64,0.10);
  border-radius: 4px;
  padding: 10px 6px;
  text-align: center;
}
.feat-icon { font-size:1.1rem; margin-bottom:5px; }
.feat-name {
  font-family: var(--font-mono);
  font-size: 0.46rem;
  color: var(--text-dim);
  letter-spacing: 0.1em;
  text-transform: uppercase;
  margin-bottom: 4px;
  white-space: nowrap;
}
.feat-val {
  font-family: var(--font-kame);
  font-size: 0.9rem;
  font-weight: 600;
  color: var(--land-text);
  margin-bottom: 5px;
}
.feat-bar-track { height:3px; background:#dce8f0; border-radius:2px; overflow:hidden; }
.feat-bar-fill  { height:100%; border-radius:2px; transition:width 0.4s ease, background 0.4s; }

/* ── Stats card ── */
.stats-title, .log-title {
  font-family: var(--font-mono);
  font-size: 0.65rem;
  color: var(--text-dim);
  letter-spacing: 0.14em;
  text-transform: uppercase;
  margin-bottom: 10px;
}
.stats-grid { display:grid; grid-template-columns:repeat(4,1fr); gap:6px; text-align:center; }
.stat-val {
  font-family: var(--font-kame);
  font-size: 0.95rem;
  font-weight: 600;
  color: var(--red);
  margin-bottom: 2px;
}
.stat-lbl {
  font-family: var(--font-mono);
  font-size: 0.46rem;
  color: var(--text-mid);
  letter-spacing: 0.08em;
  text-transform: uppercase;
}

/* ── Log card ── */
.log-item {
  font-family: var(--font-mono);
  font-size: 0.56rem;
  color: var(--text-dim);
  padding: 3px 0;
  border-bottom: 1px solid rgba(30,45,64,0.07);
}
.log-alert { color: var(--red) !important; }
.log-focus { color: var(--green) !important; }
.log-system { color: var(--text-mid) !important; font-style: italic; }

/* ── Header ── */
.fwc-header {
  display:flex; align-items:center; justify-content:space-between;
  border-bottom: 1px solid rgba(30,45,64,0.15);
  padding: 14px 24px 10px;
  margin-bottom: 14px;
}
.fwc-logo { display:flex; align-items:center; gap:10px; }
.logo-dot {
  width:20px; height:20px;
  background: var(--green);
  border-radius:50%;
  box-shadow: 0 0 10px rgba(58,140,82,0.6);
  animation: ldpulse 2s infinite;
  flex-shrink:0;
}
@keyframes ldpulse {
  0%,100% { opacity:1; box-shadow:0 0 10px rgba(58,140,82,0.6); }
  50% { opacity:.4; box-shadow:0 0 4px rgba(58,140,82,0.2); }
}
.logo-text {
  font-family: var(--font-kame);
  font-size: 1.6rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--land-dark);
}
.hdr-status {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  letter-spacing: 0.12em;
}

/* ── Camera area ── */
.cam-hint {
  font-family: var(--font-mono);
  font-size: 0.72rem;
  color: var(--text-dim);
  text-align: center;
  padding: 6px 0 2px;
  letter-spacing: 0.06em;
}
/* WebRTC component */
div[data-testid="stVideo"] { border-radius: 6px !important; overflow:hidden; }

/* ── Privacy note ── */
.privacy-note {
  font-family: var(--font-mono);
  font-size: 0.48rem;
  color: var(--text-mid);
  text-align: center;
  padding: 8px 0 4px;
  letter-spacing: 0.06em;
}

/* ══════════════════
   LANDING PAGE
══════════════════ */
/* Full-screen background layer (behind Streamlit content) */
.landing-bg-layer {
  position: fixed; inset: 0; z-index: 0; pointer-events: none;
  background: linear-gradient(145deg, #dde8f0 0%, #c8d8e8 40%, #b8cfe0 100%);
  overflow: hidden;
}
.landing-bg-layer::before {
  content:'';
  position:absolute; top:-10%; right:-10%;
  width:55vw; height:55vw;
  background: radial-gradient(circle, #a8c8e0 0%, #8ab8d8 60%, transparent 100%);
  border-radius:50%; filter:blur(80px); opacity:0.7;
}
.landing-bg-layer::after {
  content:'';
  position:absolute; bottom:5%; right:15%;
  width:30vw; height:30vw;
  background: radial-gradient(circle, #c8d8e8 0%, #b0c8d8 60%, transparent 100%);
  border-radius:50%; filter:blur(70px); opacity:0.55;
}
.landing-text-block {
  position: relative; z-index: 1;
  padding: 10vh 4vw 0 8vw;
}
.landing-welcome {
  font-family: var(--font-lusi);
  font-size: clamp(2.5rem, 5vw, 5rem);
  font-weight: 400;
  color: var(--land-text);
  line-height: 1;
  margin-bottom: 10px;
}
.landing-brand {
  display: flex; align-items: center; gap: 20px;
  margin-bottom: 20px;
}
.landing-dot {
  width: 38px; height: 38px;
  background: var(--green);
  border-radius: 50%;
  box-shadow: 0 0 20px rgba(58,140,82,0.5);
  animation: ldpulse 2.2s ease-in-out infinite;
  flex-shrink: 0;
}
.landing-title {
  font-family: var(--font-kame);
  font-size: clamp(2.5rem, 5vw, 5rem);
  font-weight: 700;
  color: var(--land-dark);
  line-height: 1;
}
.landing-sub {
  font-family: var(--font-lusi);
  font-size: clamp(1rem, 1.8vw, 1.5rem);
  color: #4a6075;
}
/* CTA button — styled like Figma bottom-right CTA */.stButton > button {
  background: #3a8c52 !important;
  border: none !important;
  color: white !important;

  font-family: var(--font-kame) !important;
  font-size: 1.5rem !important;
  font-weight: 700 !important;

  padding: 24px 50px !important;
  min-width: 200px !important;
  min-height: 80px !important;

  border-radius: 60px !important;

  box-shadow: 0 10px 25px rgba(58,140,82,.25);
  transition: all .25s ease;
}

.stButton > button:hover {
  background: #2f7344 !important;
  color: white !important;
  transform: translateY(-3px);
}

/* ══════════════════
   POPUP / DIALOG
══════════════════ */
/* Style dialogs to look like the HTML design (dark card) */
[data-testid="stModal"] > div,
[role="dialog"] > div {
  background: transparent !important;
}
[data-testid="stModal"] [data-testid="stModalContent"],
[data-baseweb="modal"] > div > div {
  background: #1e2b3a !important;
  border-radius: 14px !important;
  border: none !important;
  box-shadow: 0 20px 60px rgba(0,0,0,0.5) !important;
  color: #dce8f0 !important;

  width: 850px !important;
  max-width: 85vw !important;

  padding: 20px 40px !important;
}
[role="dialog"] p, [role="dialog"] div {
  color: #dce8f0 !important;
}
[role="dialog"] h1 {
  display: none !important;
}
[role="dialog"] h1,[role="dialog"] h2,[role="dialog"] h3 {
  font-family: var(--font-kame) !important;
  color: #f0f6fc !important;
}
/* Allow/Deny button colors inside dialogs */
[role="dialog"] .stButton:first-child > button {
  background: var(--green) !important;
  border-color: var(--green) !important;
  color: #fff !important;
  border-radius: 8px !important;
}
[role="dialog"] .stButton:first-child > button:hover {
  background: #2d7040 !important;
}
[role="dialog"] .stButton:last-child > button {
  background: var(--red) !important;
  border-color: var(--red) !important;
  color: #fff !important;
  border-radius: 8px !important;
}
/* New session button = orange */
.btn-newsession .stButton > button {
  background: var(--orange) !important;
  border-color: var(--orange) !important;
  color: var(--land-dark) !important;
  font-weight: 700 !important;
  border-radius: 8px !important;
}
.btn-newsession .stButton > button:hover {
  background: #d49018 !important;
}

/* Popup text styling */
.popup-body {
  font-size: 0.85rem;
  line-height: 1.45;
  margin-bottom: 10px;
}

.popup-list li {
  margin-bottom: 5px;
  line-height: 1.35;
}

.popup-footer {
  margin-bottom: 12px;
}
.popup-list li {
  font-family: var(--font-lusi);
  font-size: 0.84rem;
  color: #b0c4d8;
  line-height: 1.5;
  margin-bottom: 9px;
  display: flex;
  gap: 8px;
}
.check { color: var(--green); font-weight: 700; flex-shrink:0; }
.popup-footer {
  font-family: var(--font-lusi);
  font-size: 0.84rem;
  color: #8ca0b4;
  margin-bottom: 22px;
}
.summary-box {
  background: rgba(255,255,255,0.06);
  border-radius: 8px;
  padding: 14px 16px;
  margin: 14px 0;
}
.summary-label {
  font-family: var(--font-kame);
  font-size: 0.84rem;
  color: #8ca0b4;
  margin-bottom: 8px;
  font-weight: 600;
}
.summary-li {
  font-family: var(--font-lusi);
  font-size: 0.84rem;
  color: #b0c4d8;
  margin-bottom: 4px;
}
.summary-li span { color:#f0f6fc; font-weight:700; }

.stButton > button {
    background: #3a8c52 !important;
    border: 1px solid #3a8c52 !important;
    color: white !important;
}

.allow-btn .stButton > button:hover {
    background: #2f7344 !important;
}

.deny-btn .stButton > button {
    background: #c0392b !important;
    border: 1px solid #c0392b !important;
    color: white !important;
}

.deny-btn .stButton > button:hover {
    background: #a93226 !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    # Full-screen background
    st.markdown('<div class="landing-bg-layer"></div>', unsafe_allow_html=True)

    # Main text content
    st.markdown("""
    <div class="landing-text-block">
      <p class="landing-welcome">Welcome to</p>
      <div class="landing-brand">
        <div class="landing-dot"></div>
        <span class="landing-title">FocusWebCam</span>
      </div>
      <p class="landing-sub">Your Personal AI Companion for Unstoppable Focus.</p>
    </div>
    """, unsafe_allow_html=True)

    # Spacer to push CTA toward bottom
    st.markdown('<div style="height:35vh;"></div>', unsafe_allow_html=True)

    # CTA row: spacer + button aligned right
    _sp, _btn_col = st.columns([3, 1])
    with _btn_col:
        st.markdown('<div class="cta-btn">', unsafe_allow_html=True)
        if st.button("Let's get started  \u2192", key="btn_landing"):
            st.session_state.page = "app"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.stop()

# ═══════════════════════════════════════════════════════════════
# PRIVACY AGREEMENT POPUP
# ═══════════════════════════════════════════════════════════════
if not st.session_state.consent_asked:
    @st.dialog("Privacy Agreement")
    def _consent():
        
        st.markdown("""
        <div style="display:flex; align-items:center; gap:14px; margin-bottom:18px;">
          <svg width="42" height="42" viewBox="0 0 36 36" fill="none">
    <path d="M18 3L4 9v9c0 8.28 5.92 16.02 14 18 8.08-1.98 14-9.72 14-18V9L18 3z"
      fill="#E8A020" opacity="0.18"/>
    <path d="M18 3L4 9v9c0 8.28 5.92 16.02 14 18 8.08-1.98 14-9.72 14-18V9L18 3z"
      stroke="#E8A020" stroke-width="2" fill="none"/>
    <text x="18" y="23" text-anchor="middle"
      font-size="14" fill="#E8A020" font-weight="bold">i</text>
  </svg>

  <div style="
      font-family: var(--font-kame);
      font-size: 1.7rem;
      font-weight: 700;
      color: #f0f6fc;
  ">
      Privacy Agreement
  </div>

</div>
        <p class="popup-body">
          To help you track your focus levels accurately, FocusWebCam needs to analyze your facial
          data through your camera. But don't worry, your privacy is our number one priority!
          Here is our safety guarantee to you:
        </p>
        <ul class="popup-list">
          <li><span class="check">✓</span>
            <span><strong>100% Local Processing:</strong> All facial analysis happens directly on
            your device right now. No data ever leaves your laptop or phone.</span></li>
          <li><span class="check">✓</span>
            <span><strong>No Video Streams Sent Anywhere:</strong> We absolutely do not stream,
            upload, or save your video recordings to any external servers. What happens in your
            room, stays on your device.</span></li>
          <li><span class="check">✓</span>
            <span><strong>Only Session Scores Saved:</strong> The system only records your aggregate
            focus scores (the final stats) during this active session for your own progress
            tracking.</span></li>
        </ul>
        <p class="popup-footer">Sounds good, right? Let's grant access and get started!</p>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="allow-btn">', unsafe_allow_html=True)
            if st.button("Allow", use_container_width=True, key="btn_allow"):
                st.session_state.consent_given = True
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, ("focus", "✓ Privacy consent granted."))
                st.rerun()
        with c2:
            st.markdown('<div class="deny-btn">', unsafe_allow_html=True)
            if st.button("Deny", use_container_width=True, key="btn_deny"):
                st.session_state.consent_given = False
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, ("alert", "✗ Privacy consent denied."))
                st.session_state.page = "landing"
                st.rerun()
    _consent()

# ═══════════════════════════════════════════════════════════════
# SESSION COMPLETE POPUP
# ═══════════════════════════════════════════════════════════════
if st.session_state.show_session_complete:
    @st.dialog("Session Complete!")
    def _session_complete():
        sm = st.session_state.session_summary
        st.markdown(f"""
        <div style="font-size:2rem;margin-bottom:8px;">🎉</div>
        <p class="popup-body">
          <strong style="color:#f0f6fc;">Amazing job!</strong> You made it to the end of your session.<br>
          Taking control of your time and focus isn't easy, but you just did it.
          Every minute you spent here is a step closer to your goals.
        </p>
        <div class="summary-box">
          <p class="summary-label">Your Session Summary:</p>
          <p class="summary-li">• Total Duration: <span>{sm.get('duration','—')}</span></p>
          <p class="summary-li">• Average Focus: <span>{sm.get('avg','—')}/100</span></p>
          <p class="summary-li">• Alerts Triggered: <span>{sm.get('alerts','0')} times</span></p>
        </div>
        <p class="popup-body" style="font-size:.82rem;color:#8ca0b4;">
          Proud of your progress today. Rest your eyes for a bit, grab a drink,
          and we'll see you in the next session!
        </p>
        """, unsafe_allow_html=True)
        st.markdown('<div class="btn-newsession">', unsafe_allow_html=True)
        if st.button("Start new session", use_container_width=True, key="btn_newsession"):
            st.session_state.show_session_complete = False
            st.session_state.session_active  = False
            st.session_state.score_history   = []
            st.session_state.alert_count     = 0
            st.session_state.low_score_count = 0
            st.session_state.last_alert_time = 0
            st.session_state.disp_score      = None
            st.session_state.log_entries     = [("system", "— Sistem siap —")]
            st.session_state.consent_asked   = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    _session_complete()

# ═══════════════════════════════════════════════════════════════
# APP PAGE
# ═══════════════════════════════════════════════════════════════

# Background blobs via HTML
st.markdown("""
<div style="position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;">
  <div style="position:absolute;inset:0;background:linear-gradient(145deg,#d8e6f0 0%,#c4d4e4 50%,#b8ccdc 100%);"></div>
  <div style="position:absolute;top:-15%;right:-8%;width:45vw;height:45vw;
    background:radial-gradient(circle,#9cbcd4 0%,#7aaac8 60%,transparent 100%);
    border-radius:50%;filter:blur(70px);opacity:0.55;"></div>
  <div style="position:absolute;bottom:10%;right:20%;width:30vw;height:30vw;
    background:radial-gradient(circle,#c8d8e8 0%,#b0c8d8 60%,transparent 100%);
    border-radius:50%;filter:blur(70px);opacity:0.55;"></div>
</div>
""", unsafe_allow_html=True)

# Header
if st.session_state.session_active:
    hdr_status = "SESSION ACTIVE"
elif st.session_state.consent_given:
    hdr_status = "READY — LR MODEL"
else:
    hdr_status = "LIMITED MODE"

st.markdown(f"""
<div class="fwc-header">
  <div class="fwc-logo">
    <div class="logo-dot"></div>
    <span class="logo-text">FocusWebCam</span>
  </div>
  <div class="hdr-status">{hdr_status}</div>
</div>
""", unsafe_allow_html=True)

# Drain queue
drain_queue()

# ── Layout: Camera | Info
cam_col, info_col = st.columns([3, 2], gap="medium")

with cam_col:
    # Start / Stop button
    if not st.session_state.session_active:
        if st.button("Start Session", key="btn_start"):
            st.session_state.session_active  = True
            st.session_state.session_start   = time.time()
            st.session_state.score_history   = []
            st.session_state.alert_count     = 0
            st.session_state.low_score_count = 0
            st.session_state.last_alert_time = 0
            st.session_state.disp_score      = None
            st.session_state.disp_ear        = None
            st.session_state.disp_head       = None
            st.session_state.disp_mouth      = None
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_entries.insert(0, ("focus", f"🎯 [{ts}] Session started"))
            st.rerun()
    else:
        st.markdown('<div class="focusbtn-stop">', unsafe_allow_html=True)
        if st.button("End Session", key="btn_stop"):
            hist = st.session_state.score_history
            avg  = round(sum(hist)/len(hist)) if hist else 0
            pct  = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100) if hist else 0
            elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
            mm2 = elapsed // 60; ss2 = elapsed % 60
            dur_str = f"{mm2} min {ss2} sec" if mm2 > 0 else f"{ss2} sec"
            ts  = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_entries.insert(
                0, ("focus", f"📊 [{ts}] Done — avg {avg}, focus {pct}%, {st.session_state.alert_count} alerts"))
            st.session_state.session_summary = {
                "duration": dur_str, "avg": avg,
                "focus_pct": pct, "alerts": st.session_state.alert_count
            }
            st.session_state.session_active     = False
            st.session_state.show_session_complete = True
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    # WebRTC
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    ctx = webrtc_streamer(
        key="focus-cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=FocusVideoProcessor,
        media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
        async_processing=True,
    )

    if not st.session_state.session_active:
        st.markdown('<p class="cam-hint">Press START SESSION button after the camera is on</p>',
                    unsafe_allow_html=True)

    # Back to landing
    st.markdown('<div class="focusbtn-stop" style="margin-top:6px;">', unsafe_allow_html=True)
    if st.button("← Back to Home", key="btn_back"):
        st.session_state.page = "landing"
        st.session_state.session_active = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

# ── Right panel
with info_col:
    score = st.session_state.disp_score
    ear   = st.session_state.disp_ear
    head  = st.session_state.disp_head
    mouth = st.session_state.disp_mouth
    expl  = st.session_state.disp_expl

    # Score values & colors
    if score is not None:
        if score >= 65:
            sc_color = "#2a7a48"; bar_color = "#3a8c52"; state_txt = "FOCUSED"
        elif score >= 40:
            sc_color = "#c08020"; bar_color = "#e8a020"; state_txt = "ATTENTION"
        else:
            sc_color = "#c0392b"; bar_color = "#e05252"; state_txt = "NOT FOCUSED"
        sc_disp = str(score)
        bar_w   = score
    else:
        sc_color = "#8a9eb0"; bar_color = "#dce8f0"; state_txt = "—"
        sc_disp = "--"; bar_w = 0; score = 0

    # Feature bar widths
    ear_w   = int(min((ear   / 0.4) * 100, 100)) if ear   is not None else 0
    head_w  = int(min((head  / 0.3) * 100, 100)) if head  is not None else 0
    mouth_w = int(min((mouth / 0.15)* 100, 100)) if mouth is not None else 0
    feat_color = bar_color

    ed = f"{ear:.3f}"   if ear   is not None else "—"
    hd = f"{head:.3f}"  if head  is not None else "—"
    md = f"{mouth:.3f}" if mouth is not None else "—"

    # ── Score card
    st.markdown(f"""
    <div class="fwc-card">
      <div class="score-label">Focus Score</div>
      <div style="display:flex;align-items:baseline;gap:5px;">
        <span class="score-number" style="color:{sc_color}">{sc_disp}</span>
        <span class="score-unit">/100</span>
      </div>
      <div class="score-bar-track">
        <div class="score-bar-fill" style="width:{bar_w}%;background:{bar_color};"></div>
      </div>
      <div class="score-state" style="color:{sc_color};">{state_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature cards
    st.markdown(f"""
    <div class="feat-grid">
      <div class="feat-card">
        <div class="feat-icon">👁</div>
        <div class="feat-name">Eye Aspect Ratio</div>
        <div class="feat-val">{ed}</div>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:{ear_w}%;background:{feat_color};"></div>
        </div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">↔</div>
        <div class="feat-name">Head Pose</div>
        <div class="feat-val">{hd}</div>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:{head_w}%;background:{feat_color};"></div>
        </div>
      </div>
      <div class="feat-card">
        <div class="feat-icon">💬</div>
        <div class="feat-name">Mouth Ratio</div>
        <div class="feat-val">{md}</div>
        <div class="feat-bar-track">
          <div class="feat-bar-fill" style="width:{mouth_w}%;background:{feat_color};"></div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Stats card
    hist = st.session_state.score_history
    avg_s = round(sum(hist)/len(hist)) if hist else 0
    fpct  = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100) if hist else 0
    elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
    hh = elapsed // 3600; mm = (elapsed % 3600) // 60; ss = elapsed % 60
    dur_disp = f"{str(hh).zfill(2)}:{str(mm).zfill(2)}:{str(ss).zfill(2)}"

    st.markdown(f"""
    <div class="fwc-card">
      <div class="stats-title">Current Session</div>
      <div class="stats-grid">
        <div>
          <div class="stat-val">{dur_disp}</div>
          <div class="stat-lbl">Duration</div>
        </div>
        <div>
          <div class="stat-val">{avg_s if hist else "--"}</div>
          <div class="stat-lbl">Average</div>
        </div>
        <div>
          <div class="stat-val">{(str(fpct)+"%") if hist else "--%"}</div>
          <div class="stat-lbl">Focus</div>
        </div>
        <div>
          <div class="stat-val">{st.session_state.alert_count}</div>
          <div class="stat-lbl">Alert</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Log card
    logs_html = ""
    for kind, entry in st.session_state.log_entries[:18]:
        css = "log-alert" if kind == "alert" else ("log-focus" if kind == "focus" else "log-system")
        logs_html += f'<div class="log-item {css}">{entry}</div>'

    st.markdown(f"""
    <div class="fwc-card">
      <div class="log-title">Log Activity</div>
      {logs_html}
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="privacy-note">🔒 Data processed locally — never sent to any server</div>',
                unsafe_allow_html=True)

# ── Auto-refresh while camera active
if "ctx" in dir() and ctx.state.playing:
    time.sleep(0.5)
    st.rerun()
