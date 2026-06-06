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
import threading
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
# Model parameters (dari training_report.txt)
# ─────────────────────────────────────────────
MODEL_COEF = {"ear": 1.0494, "head_pose": -2.6625, "mouth_ratio": 2.0005}
MODEL_INTERCEPT = -0.5234
MODEL_SCALER = {
    "ear":         {"mean": 0.214, "std": 0.098},
    "head_pose":   {"mean": 0.178, "std": 0.245},
    "mouth_ratio": {"mean": 0.068, "std": 0.082},
}
ALERT_THRESHOLD  = 40
EAR_OPEN         = 0.25
EAR_CLOSED       = 0.15
SMOOTHING_WINDOW = 3

# ─────────────────────────────────────────────
# Fitur helpers
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
    return min(ratio, MOUTH_MAX_REALISTIC

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

def get_color(score):
    if score >= 65:  return (0, 255, 136)
    if score >= 40:  return (0, 200, 255)
    return (80, 80, 255)

def explain_score(ear, head, mouth, score):
    neg = []
    if ear   < 0.20: neg.append("mata tertutup/berkedip")
    if head  > 0.15: neg.append("kepala menoleh")
    if mouth > 0.08: neg.append("mulut terbuka")
    if score >= 65:
        return f"✅ Fokus baik ({score}/100)"
    elif score >= 40:
        isu = ", ".join(neg) if neg else "pertahankan kondisi"
        return f"⚡ Perhatian ({score}/100) — {isu}"
    else:
        isu = ", ".join(neg) if neg else "kondisi tidak optimal"
        return f"⚠️ Tidak fokus ({score}/100) — {isu}"

# ─────────────────────────────────────────────
# Queue untuk kirim data dari WebRTC → main thread
# ─────────────────────────────────────────────
# Pakai satu queue global yang persist di session_state
if "result_queue" not in st.session_state:
    st.session_state.result_queue = queue.Queue(maxsize=5)

result_queue: queue.Queue = st.session_state.result_queue

# ─────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────
def init_state():
    defaults = {
        "session_active":   False,
        "session_start":    None,
        "score_history":    [],
        "alert_count":      0,
        "low_score_count":  0,
        "last_alert_time":  0,
        "log_entries":      ["— Sistem siap —"],
        "consent_given":    False,
        "consent_asked":    False,
        # display values — diupdate dari queue
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
# Video Processor — kirim hasil via queue
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
            color = get_color(score)
            expl  = explain_score(ear, head, mouth, score)

            # Kirim ke main thread via queue (non-blocking)
            data = {
                "face":  True,
                "score": score,
                "ear":   round(ear,   4),
                "head":  round(head,  4),
                "mouth": round(mouth, 4),
                "expl":  expl,
            }
            try:
                result_queue.put_nowait(data)
            except queue.Full:
                try:    result_queue.get_nowait()
                except: pass
                try:    result_queue.put_nowait(data)
                except: pass

            # ── Draw overlay ──
            for idx in LEFT_EYE + RIGHT_EYE:
                pt = lm[idx]
                cv2.circle(img, (int(pt.x*w), int(pt.y*h)), 2, color, -1)

            fl = lm[FACE_LEFT]; fr = lm[FACE_RIGHT]
            ft = lm[10];        fb = lm[152]
            cv2.rectangle(img,
                (int(fl.x*w), int(ft.y*h)),
                (int(fr.x*w), int(fb.y*h)),
                color, 1)

            # HUD
            overlay = img.copy()
            cv2.rectangle(overlay, (10, 10), (210, 105), (0,0,0), -1)
            cv2.addWeighted(overlay, 0.55, img, 0.45, 0, img)
            cv2.putText(img, f"FOCUS: {score}", (18, 38),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)
            cv2.putText(img, f"EAR:{ear:.3f}  HEAD:{head:.3f}", (18, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
            cv2.putText(img, f"MOUTH:{mouth:.3f}", (18, 78),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,180), 1)
            bar_w = int((score/100)*182)
            cv2.rectangle(img, (18, 88), (200, 97), (40,40,40), -1)
            cv2.rectangle(img, (18, 88), (18+bar_w, 97), color, -1)

        else:
            cv2.putText(img, "Tidak ada wajah terdeteksi", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (80,80,80), 1)
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
# Drain queue → update session_state display values
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
            st.session_state.low_score_count = 0
            ts = datetime.now().strftime("%H:%M:%S")
            st.session_state.log_entries.insert(
                0, f"⚠️ [{ts}] Alert #{st.session_state.alert_count} — skor {score}")

    return True

# ─────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
:root {
  --bg:#0a0a0a; --surface:#111; --border:#2a2a2a;
  --accent:#00ff88; --accent2:#ff4444; --accent3:#ffcc00;
  --text:#e8e8e8; --text-dim:#555; --text-mid:#888;
}
html,body,[data-testid="stAppViewContainer"]{background:var(--bg)!important;font-family:'Syne',sans-serif;}
[data-testid="stHeader"],[data-testid="stToolbar"],#MainMenu,footer{display:none!important;}
[data-testid="stSidebar"]{display:none!important;}
.app-title{font-family:'Syne',sans-serif;font-weight:800;font-size:1.3rem;letter-spacing:.1em;color:#e8e8e8;}
.logo-dot{display:inline-block;width:10px;height:10px;background:#00ff88;border-radius:50%;
  box-shadow:0 0 12px #00ff88;margin-right:10px;animation:pulse 2s infinite;}
@keyframes pulse{0%,100%{opacity:1;box-shadow:0 0 12px #00ff88;}50%{opacity:.4;box-shadow:0 0 4px #00ff88;}}
.hstatus{font-family:'Space Mono',monospace;font-size:.65rem;color:#555;letter-spacing:.12em;text-align:right;}
.score-card{background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:16px;margin-bottom:8px;}
.score-label{font-family:'Space Mono',monospace;font-size:.58rem;color:#555;letter-spacing:.15em;margin-bottom:6px;}
.score-big{font-family:'Space Mono',monospace;font-size:3.2rem;font-weight:700;line-height:1;}
.score-state{font-family:'Space Mono',monospace;font-size:.65rem;letter-spacing:.1em;margin-top:4px;}
.feat-card{background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:10px;text-align:center;}
.feat-name{font-family:'Space Mono',monospace;font-size:.46rem;color:#555;letter-spacing:.1em;margin-bottom:4px;}
.feat-val{font-family:'Space Mono',monospace;font-size:.85rem;color:#e8e8e8;}
.feat-icon{font-size:.95rem;margin-bottom:3px;}
.stats-card{background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:14px;margin-bottom:8px;}
.stats-title{font-family:'Space Mono',monospace;font-size:.56rem;color:#555;letter-spacing:.15em;margin-bottom:10px;}
.stat-val{font-family:'Space Mono',monospace;font-size:1rem;color:#00ff88;font-weight:700;}
.stat-lbl{font-family:'Space Mono',monospace;font-size:.44rem;color:#555;letter-spacing:.08em;}
.log-card{background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:14px;}
.log-title{font-family:'Space Mono',monospace;font-size:.56rem;color:#555;letter-spacing:.15em;margin-bottom:8px;}
.log-entry{font-family:'Space Mono',monospace;font-size:.53rem;color:#888;padding:3px 0;border-bottom:1px solid #1a1a1a;}
.log-alert{color:#ff4444!important;} .log-focus{color:#00ff88!important;} .log-sys{color:#444!important;font-style:italic;}
.expl-card{background:#111;border:1px solid #2a2a2a;border-radius:4px;padding:12px;
  font-family:'Space Mono',monospace;font-size:.6rem;color:#888;margin-bottom:8px;}
.divider{border-top:1px solid #2a2a2a;margin:10px 0;}
.privacy-note{font-family:'Space Mono',monospace;font-size:.48rem;color:#2a2a2a;text-align:center;margin-top:8px;}
.stButton>button{background:transparent!important;border:1px solid #00ff88!important;color:#00ff88!important;
  font-family:'Space Mono',monospace!important;font-size:.75rem!important;letter-spacing:.15em!important;
  width:100%;border-radius:4px!important;}
.stButton>button:hover{background:#00ff88!important;color:#0a0a0a!important;}
div[data-testid="stProgress"]>div>div{background:linear-gradient(90deg,#00ff88,#00cc6a)!important;}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Consent dialog
# ─────────────────────────────────────────────
if not st.session_state.consent_asked:
    @st.dialog("📋 Persetujuan Privasi")
    def _consent():
        st.markdown("""
        **FocusWebCam** memproses data wajah Anda untuk mendeteksi tingkat fokus.
        - ✅ Semua data diproses **lokal di perangkat Anda**
        - ✅ Video tidak pernah dikirim ke server manapun
        - ✅ Hanya skor agregat yang disimpan di session
        """)
        c1, c2 = st.columns(2)
        with c1:
            if st.button("✅ Izinkan", use_container_width=True):
                st.session_state.consent_given = True
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, "✅ Persetujuan diberikan")
                st.rerun()
        with c2:
            if st.button("❌ Tolak", use_container_width=True):
                st.session_state.consent_given = False
                st.session_state.consent_asked = True
                st.session_state.log_entries.insert(0, "❌ Persetujuan ditolak")
                st.rerun()
    _consent()

# ─────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────
hc1, hc2 = st.columns([3, 1])
with hc1:
    st.markdown('<div class="app-title"><span class="logo-dot"></span>FocusWebCam</div>',
                unsafe_allow_html=True)
with hc2:
    if st.session_state.session_active:
        status = "SESI AKTIF"
    elif st.session_state.consent_given:
        status = "SIAP — Model LR"
    else:
        status = "MODE TERBATAS"
    st.markdown(f'<div class="hstatus">{status}</div>', unsafe_allow_html=True)

st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Drain queue setiap rerun
# ─────────────────────────────────────────────
drain_queue()

# ─────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────
cam_col, info_col = st.columns([3, 2])

with cam_col:
    # Tombol start/stop
    if not st.session_state.session_active:
        if st.button("▶  MULAI SESI", key="btn_start"):
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
            st.session_state.log_entries.insert(0, f"🎯 [{ts}] Sesi dimulai")
            st.rerun()
    else:
        if st.button("⏹  HENTIKAN SESI", key="btn_stop"):
            hist = st.session_state.score_history
            if hist:
                avg = round(sum(hist)/len(hist))
                pct = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100)
                ts  = datetime.now().strftime("%H:%M:%S")
                st.session_state.log_entries.insert(
                    0, f"📊 [{ts}] Selesai — avg {avg}, fokus {pct}%, {st.session_state.alert_count} alert")
            st.session_state.session_active = False
            st.rerun()

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
        st.info("Tekan **MULAI SESI** setelah kamera aktif untuk memulai deteksi.", icon="📷")

# ─────────────────────────────────────────────
# Panel kanan
# ─────────────────────────────────────────────
with info_col:
    score = st.session_state.disp_score
    ear   = st.session_state.disp_ear
    head  = st.session_state.disp_head
    mouth = st.session_state.disp_mouth
    expl  = st.session_state.disp_expl

    # Score card
    if score is not None:
        color_hex = "#00ff88" if score >= 65 else ("#ffcc00" if score >= 40 else "#ff4444")
        state_txt = "FOKUS" if score >= 65 else ("PERHATIAN" if score >= 40 else "TIDAK FOKUS")
        score_disp = str(score)
    else:
        color_hex, state_txt, score_disp = "#555555", "—", "--"
        score = 0

    st.markdown(f"""
    <div class="score-card">
      <div class="score-label">FOCUS SCORE</div>
      <div class="score-big" style="color:{color_hex}">{score_disp}
        <span style="font-size:.9rem;color:#555"> /100</span>
      </div>
      <div class="score-state" style="color:{color_hex}">{state_txt}</div>
    </div>""", unsafe_allow_html=True)

    st.progress(int(score) / 100)

    # Feature cards
    f1, f2, f3 = st.columns(3)
    ed = f"{ear:.3f}"   if ear   is not None else "—"
    hd = f"{head:.3f}"  if head  is not None else "—"
    md = f"{mouth:.3f}" if mouth is not None else "—"

    with f1:
        st.markdown(f'<div class="feat-card"><div class="feat-icon">👁</div>'
                    f'<div class="feat-name">EAR (MATA)</div>'
                    f'<div class="feat-val">{ed}</div></div>', unsafe_allow_html=True)
    with f2:
        st.markdown(f'<div class="feat-card"><div class="feat-icon">↔</div>'
                    f'<div class="feat-name">HEAD POSE</div>'
                    f'<div class="feat-val">{hd}</div></div>', unsafe_allow_html=True)
    with f3:
        st.markdown(f'<div class="feat-card"><div class="feat-icon">💬</div>'
                    f'<div class="feat-name">MOUTH RATIO</div>'
                    f'<div class="feat-val">{md}</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Explanation
    if expl:
        st.markdown(f'<div class="expl-card">📊 {expl}</div>', unsafe_allow_html=True)

    # Session stats
    hist = st.session_state.score_history
    avg_s = round(sum(hist)/len(hist)) if hist else 0
    fpct  = round(sum(1 for s in hist if s >= ALERT_THRESHOLD)/len(hist)*100) if hist else 0
    elapsed = int(time.time() - st.session_state.session_start) if st.session_state.session_start else 0
    mm = str(elapsed // 60).zfill(2)
    ss = str(elapsed  % 60).zfill(2)

    st.markdown(f"""
    <div class="stats-card">
      <div class="stats-title">SESI INI</div>
      <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:6px;text-align:center">
        <div><div class="stat-val">{mm}:{ss}</div><div class="stat-lbl">Durasi</div></div>
        <div><div class="stat-val">{avg_s if hist else "--"}</div><div class="stat-lbl">Rata-rata</div></div>
        <div><div class="stat-val">{fpct if hist else "--"}%</div><div class="stat-lbl">Fokus</div></div>
        <div><div class="stat-val">{st.session_state.alert_count}</div><div class="stat-lbl">Alert</div></div>
      </div>
    </div>""", unsafe_allow_html=True)

    # Log
    st.markdown('<div class="log-card"><div class="log-title">LOG AKTIVITAS</div>', unsafe_allow_html=True)
    logs_html = ""
    for entry in st.session_state.log_entries[:20]:
        cls = ("log-alert" if any(x in entry for x in ["⚠️","❌"])
               else "log-focus" if any(x in entry for x in ["✅","🎯","📊"])
               else "log-sys")
        logs_html += f'<div class="log-entry {cls}">{entry}</div>'
    st.markdown(logs_html + "</div>", unsafe_allow_html=True)

    st.markdown('<div class="privacy-note">🔒 Data diproses lokal — tidak dikirim ke server</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Auto-refresh — hanya saat kamera aktif
# ─────────────────────────────────────────────
if ctx.state.playing:
    time.sleep(0.5)
    st.rerun()
