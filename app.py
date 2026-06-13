"""
FocusWebCam — Streamlit App 
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

def standardize(v, mean, std):
    return (v - mean) / std if std else 0.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_probability(ear, head_pose, mouth):
    mouth = min(mouth, MOUTH_MAX_REALISTIC)
    ear_s   = standardize(ear, **MODEL_SCALER["ear"])
    head_s  = standardize(head_pose, **MODEL_SCALER["head_pose"])
    mouth_s = standardize(mouth, **MODEL_SCALER["mouth_ratio"])
    ear_s = max(-3, min(3, ear_s))
    head_s = max(-3, min(3, head_s))
    mouth_s = max(-3, min(3, mouth_s))
    logit = (MODEL_COEF["ear"] * ear_s +
             MODEL_COEF["head_pose"] * head_s +
             MODEL_COEF["mouth_ratio"] * mouth_s +
             MODEL_INTERCEPT)
    return float(sigmoid(logit))

def get_cv_color(score):
    if score >= 65:  return (58, 140, 82)
    if score >= 40:  return (20, 128, 232)
    return (43, 57, 192)

def explain_score(ear, head, mouth, score):
    neg = []
    if ear   < 0.20: neg.append("eyes closed/blinking")
    if head  > 0.15: neg.append("head turned away")
    if mouth > 0.08: neg.append("mouth open")
    if score >= 65:
        return f"Good focus ({score}/100)"
    elif score >= 40:
        issue = ", ".join(neg) if neg else "maintain condition"
        return f"Attention ({score}/100) — {issue}"
    else:
        issue = ", ".join(neg) if neg else "suboptimal condition"
        return f"Not focused ({score}/100) — {issue}"

# ─────────────────────────────────────────────
# Queue Initialization
# ─────────────────────────────────────────────
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=10)
result_queue: queue.Queue = st.session_state.result_queue

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "page":             "landing",
        "session_active":   False,
        "session_start":    None,
        "score_history":    [],
        "alert_count":      0,
        "low_score_count":  0,
        "last_alert_time":  0,
        "log_entries":      [("system", "— System ready —")],
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

            for idx in LEFT_EYE + RIGHT_EYE:
                pt = lm[idx]
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 2, color, -1)
            fl = lm[FACE_LEFT]; fr = lm[FACE_RIGHT]
            ft = lm[10];        fb = lm[152]
            cv2.rectangle(img,
                (int(fl.x*w), int(ft.y*h)),
                (int(fr.x*w), int(fb.y*h)), color, 1)

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

            sz, t = 14, 2
            pts = [(int(fl.x*w), int(ft.y*h)), (int(fr.x*w), int(ft.y*h)),
                   (int(fl.x*w), int(fb.y*h)), (int(fr.x*w), int(fb.y*h))]
            for i,(px,py) in enumerate(pts):
                dx = 1 if i in (0,2) else -1
                dy = 1 if i in (0,1) else -1
                cv2.line(img,(px,py),(px+dx*sz,py),color,t)
                cv2.line(img,(px,py),(px,py+dy*sz),color,t)
        else:
            cv2.putText(img, "No face detected", (20,40),
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

        if score < ALERT_THRESHOLD and latest["face"]:
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
                0, ("alert", f"⚠️ [{ts}] Alert #{st.session_state.alert_count} — score {score}"))
            st.toast(f"⚠️ Focus decreasing! Your score: {score}", icon="🚨")
    return True

# ═══════════════════════════════════════════════════════════════
# GLOBAL CSS
# ═══════════════════════════════════════════════════════════════
eye_icon = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none">
<path d="M2 12C4.5 7.5 8 5 12 5C16 5 19.5 7.5 22 12C19.5 16.5 16 19 12 19C8 19 4.5 16.5 2 12Z"
stroke="#3A8C52" stroke-width="2"/>
<circle cx="12" cy="12" r="3" fill="#3A8C52"/>
</svg>
"""

head_icon = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none">
  <circle cx="12" cy="12" r="7"
          stroke="#3A8C52"
          stroke-width="2"/>
  <circle cx="9.5" cy="10" r="0.8"
          fill="#3A8C52"/>
  <circle cx="14.5" cy="10" r="0.8"
          fill="#3A8C52"/>
  <path d="M10 14C10.8 14.8 13.2 14.8 14 14"
        stroke="#3A8C52"
        stroke-width="1.5"
        stroke-linecap="round"/>
</svg>
"""

mouth_icon = """
<svg width="22" height="22" viewBox="0 0 24 24" fill="none">
<path d="M5 12C7 15 10 16 12 16C14 16 17 15 19 12"
stroke="#3A8C52" stroke-width="2.2" stroke-linecap="round"/>
<path d="M5 12C7 10 10 9 12 9C14 9 17 10 19 12"
stroke="#3A8C52" stroke-width="2.2" stroke-linecap="round"/>
</svg>
"""

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

.stMainBlockContainer,
[data-testid="block-container"],
[data-testid="stMainBlockContainer"] {
  padding-top: 0 !important;
  padding-bottom: 0 !important;
  padding-left: 0 !important;
  padding-right: 0 !important;
  max-width: 100% !important;
}

[data-testid="block-container"] > div:first-child {
  padding: 0 !important;
}

/* ── Header ── */
.fwc-header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  border-bottom: 1px solid rgba(30,45,64,0.15);
  padding: 16px 48px 12px 48px;
  margin-bottom: 0;
  background: transparent;
  width: 100%;
  box-sizing: border-box;
}
.fwc-logo { display:flex; align-items:center; gap:10px; }
.logo-dot {
  width: 18px; height: 18px;
  background: var(--green);
  border-radius: 50%;
  box-shadow: 0 0 10px rgba(58,140,82,0.6);
  animation: ldpulse 2s infinite;
  flex-shrink: 0;
}
@keyframes ldpulse {
  0%,100% { opacity:1; box-shadow:0 0 10px rgba(58,140,82,0.6); }
  50% { opacity:.4; box-shadow:0 0 4px rgba(58,140,82,0.2); }
}
.logo-text {
  font-family: var(--font-kame);
  font-size: 1.5rem;
  font-weight: 700;
  letter-spacing: 0.04em;
  color: var(--land-dark);
}
.hdr-status {
  font-family: var(--font-mono);
  font-size: 0.6rem;
  color: var(--text-dim);
  letter-spacing: 0.12em;
  text-transform: uppercase;
}

/* ── Layout ── */
[data-testid="stHorizontalBlock"] {
  padding-left: 48px !important;
  padding-right: 48px !important;
  padding-top: 24px !important;
  gap: 24px !important;
}

[data-testid="stColumn"] {
  padding: 0 !important;
}

/* ── Buttons ── */
.main-btn-container .stButton > button {
  font-family: var(--font-kame) !important;
  font-size: 1.05rem !important;
  font-weight: 600 !important;
  letter-spacing: 0.04em !important;
  border-radius: 5px !important;
  padding: 10px 0 !important;
  width: 100% !important;
  background: transparent !important;
  transition: all 0.22s ease !important;
}
.btn-start-box .stButton > button {
  border: 2px solid var(--green) !important;
  color: var(--green) !important;
}
.btn-start-box .stButton > button:hover {
  background: var(--green) !important;
  color: #fff !important;
}
.btn-stop-box .stButton > button {
  border: 2px solid var(--red) !important;
  color: var(--red) !important;
}
.btn-stop-box .stButton > button:hover {
  background: var(--red) !important;
  color: #fff !important;
}
.btn-back-box .stButton > button {
  border: 2px solid var(--land-dark) !important;
  color: var(--land-dark) !important;
}
.btn-back-box .stButton > button:hover {
  background: var(--land-dark) !important;
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
  font-size: 0.68rem;
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
  font-size: 0.62rem;
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
  font-size: 0.62rem;
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
  font-size: 0.44rem;
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

/* ── Camera area ── */
div[data-testid="stVideo"] { border-radius: 6px !important; overflow:hidden; }

/* ══════════════════
   LANDING PAGE
══════════════════ */
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
.landing-text-block {
  position: relative; z-index: 1;
  padding: 10vh 4vw 0 8vw;
}
.landing-welcome {
  font-family: var(--font-lusi);
  font-size: clamp(3.5rem, 6vw, 6rem);
  font-weight: 400;
  color: var(--land-text);
  line-height: 1;
  margin-bottom: 15px;
}
.landing-brand {
  display: flex; align-items: center; gap: 20px;
  margin-bottom: 25px;
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
  font-size: clamp(3rem, 5.5vw, 5.5rem);
  font-weight: 700;
  color: var(--land-dark);
  line-height: 1;
}
.landing-sub {
  font-family: var(--font-lusi);
  font-size: clamp(1.5rem, 2.5vw, 2.2rem);
  color: #4a6075;
}
.cta-btn .stButton > button {
  background: #3a8c52 !important;
  border: none !important;
  color: white !important;
  font-family: var(--font-kame) !important;
  font-size: 1.2rem !important;
  font-weight: 700 !important;
  padding: 14px 36px !important;
  min-width: 160px !important;
  min-height: 50px !important;
  border-radius: 30px !important;
  box-shadow: 0 6px 15px rgba(58,140,82,.25);
  transition: all .25s ease;
}
.cta-btn .stButton > button:hover {
  background: #2f7344 !important;
  transform: translateY(-2px);
}

/* ══════════════════
   POPUP / DIALOG
══════════════════ */
[data-testid="stModalContent"] {
  max-width: 520px !important;
  background: #1e2b3a !important;
  border-radius: 12px !important;
  padding: 20px 30px !important;
}
[data-testid="stModalHeader"] {
  display: none !important;
}
.popup-body {
  font-size: 0.88rem;
  line-height: 1.5;
  color: #dce8f0 !important;
  margin-bottom: 12px;
}
.popup-list { list-style: none; padding-left: 0; }
.popup-list li {
  font-family: var(--font-lusi);
  font-size: 0.84rem;
  color: #b0c4d8;
  line-height: 1.5;
  margin-bottom: 9px;
}
.check { color: var(--green); font-weight: 700; margin-right: 6px; }
.dialog-buttons .allow-btn .stButton > button {
  background: var(--green) !important;
  border: 1px solid var(--green) !important;
  color: white !important;
  border-radius: 6px !important;
  padding: 8px 0 !important;
}
.dialog-buttons .allow-btn .stButton > button:hover {
  background: #2f7344 !important;
}
.dialog-buttons .deny-btn .stButton > button {
  background: var(--red) !important;
  border: 1px solid var(--red) !important;
  color: white !important;
  border-radius: 6px !important;
  padding: 8px 0 !important;
}
.dialog-buttons .deny-btn .stButton > button:hover {
  background: #a93226 !important;
}
.btn-newsession .stButton > button {
  background: var(--orange) !important;
  border-color: var(--orange) !important;
  color: var(--land-dark) !important;
  border-radius: 6px !important;
}
.summary-box {
  background: rgba(255,255,255,0.06);
  border-radius: 8px;
  padding: 12px 14px;
  margin: 12px 0;
}

[data-testid="stHorizontalBlock"] {
  gap: 20px !important;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════
# LANDING PAGE
# ═══════════════════════════════════════════════════════════════
if st.session_state.page == "landing":
    st.markdown('<div class="landing-bg-layer"></div>', unsafe_allow_html=True)
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
    st.markdown('<div style="height:25vh;"></div>', unsafe_allow_html=True)
    _sp, _btn_col = st.columns([2.5, 1.5])
    with _btn_col:
        st.markdown('<div class="cta-btn">', unsafe_allow_html=True)
        if st.button("Let's get started  →", key="btn_landing"):
            st.session_state.page = "app"
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    st.stop()

# ═══════════════════════════════════════════════════════════════
# PRIVACY AGREEMENT POPUP
# ═══════════════════════════════════════════════════════════════
if not st.session_state.consent_asked:
    @st.dialog("Privacy")
    def _consent():
        st.markdown("""
        <div style="display:flex;align-items:center;gap:12px;margin-bottom:15px;">
          <span style="font-size:24px;">🛡️</span>
          <div style="font-family:var(--font-kame);font-size:1.4rem;font-weight:700;color:#f0f6fc;">
            Privacy Agreement
          </div>
        </div>
        <p class="popup-body">
          To help you track your focus levels accurately, FocusWebCam needs to analyze your facial
          data through your camera. Here is our safety guarantee to you:
        </p>
        <ul class="popup-list">
          <li><span class="check">✓</span><strong>100% Local Processing:</strong> Analysis happens directly on your device.</li>
          <li><span class="check">✓</span><strong>No Video Streams Sent:</strong> We absolutely do not upload or save your videos.</li>
          <li><span class="check">✓</span><strong>Only Session Scores Saved:</strong> Only statistical scores are kept for session history.</li>
        </ul>
        """, unsafe_allow_html=True)
        st.markdown('<div class="dialog-buttons">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown('<div class="allow-btn">', unsafe_allow_html=True)
            if st.button("Allow", use_container_width=True, key="btn_allow"):
                st.session_state.consent_given = True
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, ("focus", "✓ Privacy consent granted."))
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        with c2:
            st.markdown('<div class="deny-btn">', unsafe_allow_html=True)
            if st.button("Deny", use_container_width=True, key="btn_deny"):
                st.session_state.consent_given = False
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, ("alert", "✗ Privacy consent denied."))
                st.session_state.page = "landing"
                st.rerun()
            st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
    _consent()

# ═══════════════════════════════════════════════════════════════
# SESSION COMPLETE POPUP
# ═══════════════════════════════════════════════════════════════
if st.session_state.show_session_complete:
    @st.dialog("Complete")
    def _session_complete():
        sm = st.session_state.session_summary
        st.markdown(f"""
        <div style="font-size:1.8rem;margin-bottom:6px;">🎉</div>
        <div style="font-family:var(--font-kame);font-size:1.3rem;font-weight:700;color:#f0f6fc;margin-bottom:10px;">Session Complete!</div>
        <p class="popup-body">Every minute you spent here is a step closer to your goals.</p>
        <div class="summary-box">
          <p style="color:#b0c4d8;margin:4px 0;">• Total Duration: <span style="color:#f0f6fc;font-weight:700;">{sm.get('duration','—')}</span></p>
          <p style="color:#b0c4d8;margin:4px 0;">• Average Focus: <span style="color:#f0f6fc;font-weight:700;">{sm.get('avg','—')}/100</span></p>
          <p style="color:#b0c4d8;margin:4px 0;">• Alerts Triggered: <span style="color:#f0f6fc;font-weight:700;">{sm.get('alerts','0')} times</span></p>
        </div>
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
            st.session_state.log_entries     = [("system", "— System ready —")]
            st.session_state.consent_asked   = False
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    _session_complete()

# ═══════════════════════════════════════════════════════════════
# MAIN APP PAGE
# ═══════════════════════════════════════════════════════════════

# Background
st.markdown("""
<div style="position:fixed;inset:0;z-index:0;overflow:hidden;pointer-events:none;">
  <div style="position:absolute;inset:0;background:linear-gradient(145deg,#d8e6f0 0%,#c4d4e4 50%,#b8ccdc 100%);"></div>
</div>
""", unsafe_allow_html=True)

# Header status
hdr_status = "SESSION ACTIVE" if st.session_state.session_active else (
    "READY — LR MODEL" if st.session_state.consent_given else "LIMITED MODE"
)

# Drain queue
drain_queue()

# Header
st.markdown(f"""
<div class="fwc-header">
  <div class="fwc-logo">
    <div class="logo-dot"></div>
    <span class="logo-text">FocusWebCam</span>
  </div>
  <div class="hdr-status">{hdr_status}</div>
</div>
""", unsafe_allow_html=True)

# ── Layout Columns ──
cam_col, info_col = st.columns([3, 2], gap="medium")

with cam_col:
    st.markdown('<div class="main-btn-container">', unsafe_allow_html=True)

    # 1. Start / End Session button (top, full width)
    if not st.session_state.session_active:
        st.markdown('<div class="btn-start-box">', unsafe_allow_html=True)
        if st.button("▶  Start Session", key="btn_start", use_container_width=True):
            st.session_state.session_active  = True
            st.session_state.session_start   = time.time()
            st.session_state.score_history   = []
            st.session_state.alert_count     = 0
            st.session_state.low_score_count = 0
            st.session_state.last_alert_time = 0
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_entries.insert(0, ("focus", f"🎯 [{ts}] Session started"))
        st.markdown('</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="btn-stop-box">', unsafe_allow_html=True)
        if st.button("⏹  End Session", key="btn_stop", use_container_width=True):
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
            st.session_state.session_active        = False
            st.session_state.show_session_complete = True
        st.markdown('</div>', unsafe_allow_html=True)

    # 2. WebRTC camera (center) - ENHANCED RESOLUTION
    st.markdown('<div style="margin-top:10px;margin-bottom:8px;">', unsafe_allow_html=True)
    rtc_config = RTCConfiguration(
        {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
    )
    ctx = webrtc_streamer(
        key="focus-cam",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=rtc_config,
        video_processor_factory=FocusVideoProcessor,
        media_stream_constraints={
            "video": {
                "width": {"ideal": 1280, "max": 1920},
                "height": {"ideal": 720, "max": 1080},
                "frameRate": {"ideal": 30, "max": 60}
            }, 
            "audio": False
        },
        async_processing=True,
    )
    st.markdown('</div>', unsafe_allow_html=True)

    # 3. Back to Home button (bottom, full width)
    st.markdown('<div class="btn-back-box">', unsafe_allow_html=True)
    if st.button("← Back to Home", key="btn_back", use_container_width=True):
        st.session_state.page = "landing"
        st.session_state.session_active = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)  # close main-btn-container

# ── Right Info Panel ──
with info_col:
    score = st.session_state.disp_score
    ear   = st.session_state.disp_ear
    head  = st.session_state.disp_head
    mouth = st.session_state.disp_mouth
    expl  = st.session_state.disp_expl

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

    ear_w   = int(min((ear   / 0.4) * 100, 100)) if ear   is not None else 0
    head_w  = int(min((head  / 0.3) * 100, 100)) if head  is not None else 0
    mouth_w = int(min((mouth / 0.15)* 100, 100)) if mouth is not None else 0

    ed = f"{ear:.3f}"   if ear   is not None else "—"
    hd = f"{head:.3f}"  if head  is not None else "—"
    md = f"{mouth:.3f}" if mouth is not None else "—"

    # Score Box
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
      <div class="score-state" style="color:{sc_color};">
        {state_txt} <span style="color:#6a7e92;font-size:11px;">({expl})</span>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Grid
    st.markdown(f"""
<div class="feat-grid">

  <div class="feat-card">
    <div class="feat-icon">{eye_icon}</div>
    <div class="feat-name">EAR</div>
    <div class="feat-val">{ed}</div>
    <div class="feat-bar-track">
      <div class="feat-bar-fill"
           style="width:{ear_w}%;background:{bar_color};">
      </div>
    </div>
  </div>

  <div class="feat-card">
    <div class="feat-icon">{head_icon}</div>
    <div class="feat-name">Head Pose</div>
    <div class="feat-val">{hd}</div>
    <div class="feat-bar-track">
      <div class="feat-bar-fill"
           style="width:{head_w}%;background:{bar_color};">
      </div>
    </div>
  </div>

  <div class="feat-card">
    <div class="feat-icon">{mouth_icon}</div>
    <div class="feat-name">Mouth Ratio</div>
    <div class="feat-val">{md}</div>
    <div class="feat-bar-track">
      <div class="feat-bar-fill"
           style="width:{mouth_w}%;background:{bar_color};">
      </div>
    </div>
  </div>

</div>
""", unsafe_allow_html=True)

    # Statistics Card
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
        <div><div class="stat-val">{dur_disp}</div><div class="stat-lbl">Duration</div></div>
        <div><div class="stat-val">{avg_s if hist else "--"}</div><div class="stat-lbl">Average</div></div>
        <div><div class="stat-val">{(str(fpct)+"%") if hist else "--%"}</div><div class="stat-lbl">Focus</div></div>
        <div><div class="stat-val" style="color:#c0392b;">{st.session_state.alert_count}</div><div class="stat-lbl">Alert</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # Activity Log
    logs_html = ""
    for kind, entry in st.session_state.log_entries[:5]:
        css = "log-alert" if kind == "alert" else ("log-focus" if kind == "focus" else "log-system")
        logs_html += f'<div class="log-item {css}">{entry}</div>'

    st.markdown(f"""
    <div class="fwc-card">
      <div class="log-title">Log Activity</div>
      {logs_html}
    </div>
    """, unsafe_allow_html=True)

# ── Auto-rerun only when camera is active and session is running ──
if ctx is not None and ctx.state.playing:
    time.sleep(0.4)
    st.rerun()