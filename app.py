"""
FocusWebCam — Streamlit App
============================
Konversi dari versi HTML/JS ke Streamlit.
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
import pickle
import time
import threading
from pathlib import Path
from collections import deque
from datetime import datetime, timedelta

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

# ─────────────────────────────────────────────
# MediaPipe
# ─────────────────────────────────────────────
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

# ─────────────────────────────────────────────
# Landmark indices
# ─────────────────────────────────────────────
LEFT_EYE  = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33,  160, 158, 133, 153, 144]
MOUTH_TOP, MOUTH_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 78, 308
NOSE_TIP, FACE_LEFT, FACE_RIGHT = 1, 234, 454

# ─────────────────────────────────────────────
# Model parameters (dari training_report.txt)
# ─────────────────────────────────────────────
MODEL_COEF = {"ear": 1.0494, "head_pose": -2.6625, "mouth_ratio": 2.0005}
MODEL_INTERCEPT = -0.5234
MODEL_SCALER = {
    "ear":          {"mean": 0.214, "std": 0.098},
    "head_pose":    {"mean": 0.178, "std": 0.245},
    "mouth_ratio":  {"mean": 0.068, "std": 0.082},
}
ALERT_THRESHOLD = 40
EAR_OPEN  = 0.25
EAR_CLOSED = 0.15
SMOOTHING_WINDOW = 3

# ─────────────────────────────────────────────
# Warna tema
# ─────────────────────────────────────────────
COLOR_FOCUS   = (0, 255, 136)    # hijau — fokus
COLOR_MEDIUM  = (0, 204, 255)    # kuning — perhatian
COLOR_UNFOCUS = (68, 68, 255)    # merah — tidak fokus
COLOR_DIM     = (80, 80, 80)

# ─────────────────────────────────────────────
# Fitur
# ─────────────────────────────────────────────

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
    return vertical / horizontal if horizontal else 0.0

def standardize(v, mean, std):
    return (v - mean) / std if std else 0.0

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))

def predict_probability(ear, head_pose, mouth):
    ear_s  = standardize(ear,       **MODEL_SCALER["ear"])
    head_s = standardize(head_pose, **MODEL_SCALER["head_pose"])
    mouth_s = standardize(mouth,    **MODEL_SCALER["mouth_ratio"])
    logit = (MODEL_COEF["ear"] * ear_s +
             MODEL_COEF["head_pose"] * head_s +
             MODEL_COEF["mouth_ratio"] * mouth_s +
             MODEL_INTERCEPT)
    return float(sigmoid(logit))

def norm_ear(ear):
    clamped = max(EAR_CLOSED, min(EAR_OPEN, ear))
    return (clamped - EAR_CLOSED) / (EAR_OPEN - EAR_CLOSED)

def norm_head(head):
    return 1 - min(head, 0.3) / 0.3

def norm_mouth(mouth):
    return 1 - min(mouth, 0.5) / 0.5

def get_score_color(score):
    if score >= 65:
        return COLOR_FOCUS
    elif score >= 40:
        return COLOR_MEDIUM
    else:
        return COLOR_UNFOCUS

def explain_score(ear, head, mouth, score):
    neg = []
    if ear < 0.20:
        neg.append("mata tertutup/berkedip")
    if head > 0.15:
        neg.append("kepala menoleh")
    if mouth > 0.08:
        neg.append("mulut terbuka")
    if score >= 65:
        return f"✅ Fokus baik ({score}/100)"
    elif score >= 40:
        issues = ", ".join(neg) if neg else "pertahankan kondisi saat ini"
        return f"⚡ Perhatian ({score}/100) — {issues}"
    else:
        issues = ", ".join(neg) if neg else "kondisi tidak optimal"
        return f"⚠️ Tidak fokus ({score}/100) — {issues}"

# ─────────────────────────────────────────────
# Session state init
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "session_active": False,
        "session_start":  None,
        "score_history":  [],
        "alert_count":    0,
        "log_entries":    ["— Sistem siap —"],
        "last_score":     None,
        "last_ear":       None,
        "last_head":      None,
        "last_mouth":     None,
        "last_explanation": "",
        "smooth_scores":  deque(maxlen=SMOOTHING_WINDOW),
        "low_score_count": 0,
        "last_alert_time": 0,
        "consent_given":  False,
        "consent_asked":  False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

# ─────────────────────────────────────────────
# Shared state antara thread WebRTC & main thread
# ─────────────────────────────────────────────
class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self.score = None
        self.ear = None
        self.head = None
        self.mouth = None
        self.face_detected = False
        self.explanation = ""
        self.ear_norm = 0.0
        self.head_norm = 0.0
        self.mouth_norm = 0.0
        self.new_data = False

shared = SharedState()

# ─────────────────────────────────────────────
# Video processor (thread WebRTC)
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
        self._frame_count = 0

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        img = frame.to_ndarray(format="bgr24")
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)

        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark

            ear_l  = calc_ear(lm, LEFT_EYE,  w, h)
            ear_r  = calc_ear(lm, RIGHT_EYE, w, h)
            ear    = (ear_l + ear_r) / 2.0
            head   = calc_head_pose(lm, w, h)
            mouth  = calc_mouth(lm, w, h)

            prob  = predict_probability(ear, head, mouth)
            self._smooth.append(prob * 100)
            score = int(np.clip(round(np.mean(self._smooth)), 0, 100))

            color = get_score_color(score)
            expl  = explain_score(ear, head, mouth, score)

            # Draw eye landmarks
            for idx in LEFT_EYE + RIGHT_EYE:
                pt = lm[idx]
                cx, cy = int(pt.x * w), int(pt.y * h)
                cv2.circle(img, (cx, cy), 2, color, -1)

            # Draw face box
            fl = lm[FACE_LEFT]; fr = lm[FACE_RIGHT]
            ft = lm[10];        fb = lm[152]
            x1, y1 = int(fl.x * w), int(ft.y * h)
            x2, y2 = int(fr.x * w), int(fb.y * h)
            cv2.rectangle(img, (x1, y1), (x2, y2), (*color, 80), 1)

            # HUD overlay
            self._draw_hud(img, score, ear, head, mouth, color)

            # Update shared state
            with shared._lock:
                shared.score = score
                shared.ear   = round(ear,   4)
                shared.head  = round(head,  4)
                shared.mouth = round(mouth, 4)
                shared.face_detected  = True
                shared.explanation    = expl
                shared.ear_norm  = norm_ear(ear)
                shared.head_norm = norm_head(head)
                shared.mouth_norm = norm_mouth(mouth)
                shared.new_data  = True
        else:
            # No face
            cv2.putText(img, "Tidak ada wajah terdeteksi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (80, 80, 80), 1)
            with shared._lock:
                shared.face_detected = False
                shared.score = 0
                shared.new_data = True

        self._frame_count += 1
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def _draw_hud(self, img, score, ear, head, mouth, color):
        h_img, w_img = img.shape[:2]
        # Panel background
        overlay = img.copy()
        cv2.rectangle(overlay, (10, 10), (200, 100), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.5, img, 0.5, 0, img)

        # Score
        score_txt = f"FOCUS: {score}"
        cv2.putText(img, score_txt, (18, 38),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Feature values
        cv2.putText(img, f"EAR:{ear:.3f}  HEAD:{head:.3f}", (18, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)
        cv2.putText(img, f"MOUTH:{mouth:.3f}", (18, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180, 180, 180), 1)

        # Score bar
        bar_w = int((score / 100) * 180)
        cv2.rectangle(img, (18, 88), (18 + 180, 96), (40, 40, 40), -1)
        cv2.rectangle(img, (18, 88), (18 + bar_w, 96), color, -1)

# ─────────────────────────────────────────────
# CSS kustom
# ─────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');

  :root {
    --bg: #0a0a0a;
    --surface: #111111;
    --border: #2a2a2a;
    --accent: #00ff88;
    --accent2: #ff4444;
    --accent3: #ffcc00;
    --text: #e8e8e8;
    --text-dim: #555555;
    --text-mid: #888888;
  }

  html, body, [data-testid="stAppViewContainer"] {
    background: var(--bg) !important;
    font-family: 'Syne', sans-serif;
  }

  [data-testid="stHeader"] { display: none; }
  [data-testid="stSidebar"] { display: none; }
  [data-testid="stToolbar"] { display: none; }
  #MainMenu { display: none; }
  footer { display: none; }

  .app-title {
    font-family: 'Syne', sans-serif;
    font-weight: 800;
    font-size: 1.3rem;
    letter-spacing: 0.1em;
    color: var(--text);
  }

  .logo-dot {
    display: inline-block;
    width: 10px; height: 10px;
    background: #00ff88;
    border-radius: 50%;
    box-shadow: 0 0 12px #00ff88;
    margin-right: 10px;
    animation: pulse 2s infinite;
  }

  @keyframes pulse {
    0%, 100% { opacity: 1; box-shadow: 0 0 12px #00ff88; }
    50%       { opacity: 0.4; box-shadow: 0 0 4px #00ff88; }
  }

  .header-status {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    color: var(--text-dim);
    letter-spacing: 0.12em;
    text-align: right;
  }

  .score-card {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 16px;
    margin-bottom: 8px;
  }

  .score-label {
    font-family: 'Space Mono', monospace;
    font-size: 0.6rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    margin-bottom: 6px;
  }

  .score-big {
    font-family: 'Space Mono', monospace;
    font-size: 3.2rem;
    font-weight: 700;
    line-height: 1;
  }

  .score-state {
    font-family: 'Space Mono', monospace;
    font-size: 0.65rem;
    letter-spacing: 0.1em;
    margin-top: 4px;
  }

  .feature-card {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 10px;
    text-align: center;
  }

  .feature-name {
    font-family: 'Space Mono', monospace;
    font-size: 0.48rem;
    color: var(--text-dim);
    letter-spacing: 0.1em;
    margin-bottom: 4px;
  }

  .feature-value {
    font-family: 'Space Mono', monospace;
    font-size: 0.85rem;
    color: var(--text);
  }

  .stats-card {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 14px;
    margin-bottom: 8px;
  }

  .stats-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    margin-bottom: 10px;
  }

  .stat-val {
    font-family: 'Space Mono', monospace;
    font-size: 1rem;
    color: #00ff88;
    font-weight: 700;
  }

  .stat-lbl {
    font-family: 'Space Mono', monospace;
    font-size: 0.45rem;
    color: var(--text-dim);
    letter-spacing: 0.08em;
  }

  .log-card {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 14px;
  }

  .log-title {
    font-family: 'Space Mono', monospace;
    font-size: 0.58rem;
    color: var(--text-dim);
    letter-spacing: 0.15em;
    margin-bottom: 8px;
  }

  .log-entry {
    font-family: 'Space Mono', monospace;
    font-size: 0.55rem;
    color: #888;
    padding: 3px 0;
    border-bottom: 1px solid #1a1a1a;
  }

  .log-alert  { color: #ff4444 !important; }
  .log-focus  { color: #00ff88 !important; }
  .log-system { color: #555 !important; font-style: italic; }

  .expl-card {
    background: #111;
    border: 1px solid #2a2a2a;
    border-radius: 4px;
    padding: 12px;
    font-family: 'Space Mono', monospace;
    font-size: 0.62rem;
    color: #888;
    margin-bottom: 8px;
  }

  .stButton > button {
    background: transparent !important;
    border: 1px solid #00ff88 !important;
    color: #00ff88 !important;
    font-family: 'Space Mono', monospace !important;
    font-size: 0.75rem !important;
    letter-spacing: 0.15em !important;
    width: 100%;
    border-radius: 4px !important;
    transition: all 0.2s !important;
  }
  .stButton > button:hover {
    background: #00ff88 !important;
    color: #0a0a0a !important;
  }

  .stButton.stop > button {
    border-color: #ff4444 !important;
    color: #ff4444 !important;
  }

  div[data-testid="stProgress"] > div > div {
    background: linear-gradient(90deg, #00ff88, #00cc6a) !important;
  }

  .divider {
    border-top: 1px solid #2a2a2a;
    margin: 12px 0;
  }

  .privacy-note {
    font-family: 'Space Mono', monospace;
    font-size: 0.5rem;
    color: #333;
    text-align: center;
    margin-top: 6px;
  }

  [data-testid="stVideo"] video {
    border-radius: 4px;
    border: 1px solid #2a2a2a;
  }

  /* hide streamlit's own controls on webrtc */
  .stWebRtcStreamer { background: transparent !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Consent modal via st.dialog (if not consented)
# ─────────────────────────────────────────────
if not st.session_state.consent_given and not st.session_state.consent_asked:
    @st.dialog("📋 Persetujuan Privasi")
    def consent_dialog():
        st.markdown("""
        **FocusWebCam** memproses data wajah Anda untuk mendeteksi tingkat fokus.

        - ✅ Semua data diproses **lokal di perangkat Anda**
        - ✅ Video tidak pernah dikirim ke server manapun
        - ✅ Hanya skor agregat yang disimpan di session
        - ✅ Model AI berjalan sepenuhnya di browser/server lokal Anda
        """)
        col1, col2 = st.columns(2)
        with col1:
            if st.button("✅ Izinkan", use_container_width=True):
                st.session_state.consent_given = True
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, "✅ Persetujuan diberikan — guardrails aktif")
                st.rerun()
        with col2:
            if st.button("❌ Tolak", use_container_width=True):
                st.session_state.consent_given = False
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, "❌ Persetujuan ditolak — mode terbatas")
                st.rerun()
    consent_dialog()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
hcol1, hcol2 = st.columns([3, 1])
with hcol1:
    st.markdown('<div class="app-title"><span class="logo-dot"></span>FocusWebCam</div>', unsafe_allow_html=True)
with hcol2:
    status_txt = "SESI AKTIF" if st.session_state.session_active else ("SIAP — Model LR" if st.session_state.consent_given else "MODE TERBATAS")
    st.markdown(f'<div class="header-status">{status_txt}</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Layout utama
# ─────────────────────────────────────────────
cam_col, info_col = st.columns([3, 2])

with cam_col:
    # Tombol Start/Stop
    if not st.session_state.session_active:
        if st.button("▶  MULAI SESI", key="btn_start"):
            st.session_state.session_active = True
            st.session_state.session_start  = time.time()
            st.session_state.score_history  = []
            st.session_state.alert_count    = 0
            st.session_state.low_score_count = 0
            st.session_state.last_alert_time = 0
            st.session_state.smooth_scores   = deque(maxlen=SMOOTHING_WINDOW)
            st.session_state.log_entries.insert(0, f"🎯 [{datetime.now().strftime('%H:%M:%S')}] Sesi dimulai")
            st.rerun()
    else:
        if st.button("⏹  HENTIKAN SESI", key="btn_stop"):
            hist = st.session_state.score_history
            if hist:
                avg = round(sum(hist)/len(hist))
                pct = round(sum(1 for s in hist if s >= ALERT_THRESHOLD) / len(hist) * 100)
                st.session_state.log_entries.insert(0,
                    f"📊 Sesi selesai — avg {avg}, fokus {pct}%, {st.session_state.alert_count} alert")
            st.session_state.session_active = False
            st.rerun()

    # WebRTC streamer
    if st.session_state.session_active:
        rtc_config = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})
        ctx = webrtc_streamer(
            key="focus-cam",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=FocusVideoProcessor,
            media_stream_constraints={"video": {"width": 640, "height": 480}, "audio": False},
            async_processing=True,
        )

        # Pull data dari shared state
        if ctx.video_processor:
            st.write("FACE:", shared.face_detected)
            st.write("SCORE:", shared.score)
            st.write("EAR:", shared.ear)
            st.write("HEAD:", shared.head)
            st.write("MOUTH:", shared.mouth)
            
            with shared._lock:
                if shared.new_data:
                    st.session_state.last_score = shared.score
                    st.session_state.last_ear   = shared.ear
                    st.session_state.last_head  = shared.head
                    st.session_state.last_mouth = shared.mouth
                    st.session_state.last_explanation = shared.explanation
                    shared.new_data = False

                    score = shared.score or 0
                    st.session_state.score_history.append(score)

                    # Alert logic
                    if score < ALERT_THRESHOLD:
                        st.session_state.low_score_count += 1
                    else:
                        st.session_state.low_score_count = 0

                    now = time.time()
                    if (st.session_state.low_score_count >= 5 and
                            now - st.session_state.last_alert_time >= 30):
                        st.session_state.alert_count += 1
                        st.session_state.last_alert_time = now
                        st.session_state.log_entries.insert(0,
                            f"⚠️ [{datetime.now().strftime('%H:%M:%S')}] Alert #{st.session_state.alert_count} — skor {score}")
    else:
        st.info("Tekan **MULAI SESI** untuk mengaktifkan kamera dan memulai deteksi fokus.", icon="📷")

with info_col:
    # ── Score card ──
    score = st.session_state.last_score
    ear   = st.session_state.last_ear
    head  = st.session_state.last_head
    mouth = st.session_state.last_mouth
    expl  = st.session_state.last_explanation

    if score is not None:
        color_hex = "#00ff88" if score >= 65 else ("#ffcc00" if score >= 40 else "#ff4444")
        state_txt = "FOKUS" if score >= 65 else ("PERHATIAN" if score >= 40 else "TIDAK FOKUS")
    else:
        color_hex = "#555555"
        state_txt = "—"
        score = 0

    st.markdown(f"""
    <div class="score-card">
      <div class="score-label">FOCUS SCORE</div>
      <div class="score-big" style="color:{color_hex}">{score if st.session_state.last_score is not None else "--"}<span style="font-size:0.9rem;color:#555"> /100</span></div>
      <div class="score-state" style="color:{color_hex}">{state_txt}</div>
    </div>
    """, unsafe_allow_html=True)

    # Score bar
    st.progress(int(score) / 100)

    # ── Feature cards ──
    f1, f2, f3 = st.columns(3)
    ear_disp   = f"{ear:.3f}"   if ear   is not None else "—"
    head_disp  = f"{head:.3f}"  if head  is not None else "—"
    mouth_disp = f"{mouth:.3f}" if mouth is not None else "—"

    with f1:
        st.markdown(f"""
        <div class="feature-card">
          <div style="font-size:1rem">👁</div>
          <div class="feature-name">EAR (MATA)</div>
          <div class="feature-value">{ear_disp}</div>
        </div>""", unsafe_allow_html=True)
    with f2:
        st.markdown(f"""
        <div class="feature-card">
          <div style="font-size:1rem">↔</div>
          <div class="feature-name">HEAD POSE</div>
          <div class="feature-value">{head_disp}</div>
        </div>""", unsafe_allow_html=True)
    with f3:
        st.markdown(f"""
        <div class="feature-card">
          <div style="font-size:1rem">💬</div>
          <div class="feature-name">MOUTH RATIO</div>
          <div class="feature-value">{mouth_disp}</div>
        </div>""", unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Explanation ──
    if expl:
        st.markdown(f'<div class="expl-card">📊 {expl}</div>', unsafe_allow_html=True)

    # ── Session stats ──
    hist = st.session_state.score_history
    avg_score = round(sum(hist)/len(hist)) if hist else 0
    focus_pct = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100) if hist else 0

    elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
    mm = str(elapsed // 60).zfill(2)
    ss = str(elapsed  % 60).zfill(2)

    st.markdown(f"""
    <div class="stats-card">
      <div class="stats-title">SESI INI</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;text-align:center">
        <div><div class="stat-val">{mm}:{ss}</div><div class="stat-lbl">Durasi</div></div>
        <div><div class="stat-val">{avg_score if hist else "--"}</div><div class="stat-lbl">Rata-rata</div></div>
        <div><div class="stat-val">{focus_pct if hist else "--"}%</div><div class="stat-lbl">Fokus</div></div>
        <div><div class="stat-val">{st.session_state.alert_count}</div><div class="stat-lbl">Alert</div></div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Log ──
    st.markdown('<div class="log-card"><div class="log-title">LOG AKTIVITAS</div>', unsafe_allow_html=True)
    logs_html = ""
    for entry in st.session_state.log_entries[:20]:
        cls = "log-alert" if "⚠" in entry or "❌" in entry else \
              "log-focus"  if "✅" in entry or "🎯" in entry else "log-system"
        logs_html += f'<div class="log-entry {cls}">{entry}</div>'
    st.markdown(logs_html + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="privacy-note">🔒 Data diproses lokal — tidak dikirim ke server</div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Auto-refresh saat sesi aktif
# ─────────────────────────────────────────────
if st.session_state.session_active:
    time.sleep(1)
    st.rerun()
