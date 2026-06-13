# FocusWebCam — Streamlit Edition

Deteksi fokus real-time berbasis AI menggunakan webcam, MediaPipe FaceMesh,
dan model Logistic Regression terlatih.

## Cara Menjalankan

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Pastikan file model ada

```
focuswebcam/
├── app.py
├── requirements.txt
└── focus_model.pkl   ← hasil dari train_model.py
```

> Jika belum punya `focus_model.pkl`, jalankan dulu:
> ```bash
> python train_model.py --input features.csv
> ```

### 3. Jalankan Streamlit

```bash
streamlit run app.py
```

Browser akan terbuka otomatis di `http://localhost:8501`

---

### Link Streamlit 
'https://focus-webcam.streamlit.app/'

## Deploy ke Streamlit Cloud

1. Push folder ini ke GitHub
2. Buka [share.streamlit.io](https://share.streamlit.io)
3. Hubungkan repo, pilih `app.py` sebagai entry point
4. Klik **Deploy**

> **Catatan:** Untuk Streamlit Cloud, pastikan `focus_model.pkl` ikut di-commit ke repo,
> atau load model dari URL (lihat komentar di `app.py`).

---

## Fitur

| Fitur | Deskripsi |
|-------|-----------|
| 🎯 Focus Score | Skor 0–100 dari model Logistic Regression |
| 👁 EAR | Eye Aspect Ratio — keterbukaan mata |
| ↔ Head Pose | Orientasi kepala terhadap kamera |
| 💬 Mouth Ratio | Deteksi menguap/mulut terbuka |
| ⚠️ Alert | Notifikasi saat fokus rendah >5 detik |
| 🔒 Privacy | Semua data diproses lokal, tidak dikirim ke server |
| 📊 Explainability | Penjelasan mengapa skor naik/turun |

---

## Perbedaan vs Versi HTML

| Aspek | HTML (localhost) | Streamlit |
|-------|-----------------|-----------|
| Deployment | Buka file langsung | `streamlit run` |
| Model inference | JS (koefisien hardcode) | Python (sklearn pkl) |
| Real-time video | MediaPipe CDN | streamlit-webrtc + OpenCV |
| UI | CSS kustom penuh | Streamlit + CSS inject |
| Cloud deploy | Tidak langsung | Streamlit Cloud / HuggingFace |
