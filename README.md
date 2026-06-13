# FocusWebCam — Streamlit Edition

Real-time AI-based focus detection using webcam, MediaPipe FaceMesh, and a trained Logistic Regression model.

## How to Run

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2.Ensure the model file exists

```
focuswebcam/
├── app.py
├── requirements.txt
└── focus_model.pkl   ← output from train_model.py
```

> If you don't have `focus_model.pkl` yet, run this first:
> ```bash
> python train_model.py --input features.csv
> ```

### 3. Run Streamlit

```bash
streamlit run app.py
```

This browser will automatically open at `http://localhost:8501`

---

## Streamlit Link 
Or run it directly via the link below:

[https://focus-webcam.streamlit.app/](url)

## Deploy to Streamlit Cloud

1. Push this folder to GitHub
2. Open [share.streamlit.io](https://share.streamlit.io)
3. Connect your repository and select `app.py` as the entry point
4. Click **Deploy**

> **Note:** For Streamlit Cloud, make sure `focus_model.pkl` is committed to the repository,
> or load the model from a URL (see comments in `app.py`).

---

## Features

| Feature | Description |
|-------|-----------|
| 🎯 Focus Score | 0–100 score from the Logistic Regression model |
| 👁 EAR | Eye Aspect Ratio — eye openness |
| ↔ Head Pose | Head orientation relative to the camera |
| 💬 Mouth Ratio | Yawning/open mouth detection |
| ⚠️ Alert | Notifications when focus is low for >5 seconds |
| 🔒 Privacy | All data is processed locally, no data is sent to servers |
| 📊 Explainability | Explanations of why the score goes up/down |

---

## Comparison vs HTML Version

| Aspek | HTML (localhost) | Streamlit |
|-------|-----------------|-----------|
| Deployment | Open file directly | `streamlit run` |
| Model inference | JS (hardcoded coefficients) | Python (sklearn pkl) |
| Real-time video | MediaPipe CDN | streamlit-webrtc + OpenCV |
| UI | Full custom CSS | Streamlit + CSS inject |
| Cloud deploy | Not straightforward | Streamlit Cloud / HuggingFace |
