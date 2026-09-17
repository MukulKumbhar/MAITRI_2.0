# 🛰️ Project MAITRI 2.0

**Multimodal AI-based Astronaut Telemetry & Real-time Intelligence**

An offline AI system for monitoring astronaut psychological and physical well-being during long-duration space missions (Gaganyaan, ISS, Mars analog).

> Capstone Project — Minor in Aerospace Engineering  
> Supervisor: Dr. Sakthipriya Balu, Dept. of Aeronautical Engineering

---

## 👥 Team

| Name | Roll No | Branch |
|------|---------|--------|
| Kumbhar Mukul Ramesh | 23031012 | CSE |
| Kumbhar Aditya Ananda | 23031016 | CSE |
| Patil Aryan Amarsinh | 23101058 | CSE-IOT |
| Dalvi Shreedhar Suresh | 23041084 | EE |
| Mane Pranjal Sampatrao | 23101007 | CSE-IOT |

---

## 🧠 What MAITRI 2.0 Does

MAITRI 2.0 monitors astronaut well-being using **live multimodal streaming telemetry**:

| Module | Signal | Model/Method |
|--------|--------|-------------|
| M1 — Face Emotion | Live Webcam (WebRTC) | DeepFace VGG-Face (7 classes) |
| M2 — Eye Tracking | Live Webcam (WebRTC) | MediaPipe FaceMesh EAR & Blink Rate |
| M3 — Vitals | Sensor sliders | Physiological strain formula |
| M4 — Fusion | All above | EMA quality-aware weighted fusion |
| M5 — Alert Engine | Fused output | Rule-based CBT/grounding responses |
| M6 — Database | All events | SQLite local logging |
| M_Voice *(future)* | Microphone | SpeechBrain wav2vec2-IEMOCAP |

### Stress Level Classification

| Level | Stress Index | Action |
|-------|-------------|--------|
| ✅ NOMINAL | < 35% | Positive affirmation |
| 🔔 MILD | 35–55% | Box breathing (4-4-4-4) |
| ⚠️ ELEVATED | 55–75% | 4-7-8 breathing + rest cycle |
| 🚨 CRITICAL | > 75% | 5-4-3-2-1 grounding + alert log |

---

## 🗂️ Project Structure

```
MAITRI_2.0/
├── app.py                     # Main Streamlit dashboard (live WebRTC orchestrator)
├── requirements.txt
├── face_landmarker.task       # MediaPipe 478-point landmark model
├── maitri_logs.db             # SQLite telemetry database (auto-created)
└── modules/
    ├── __init__.py
    ├── face_module.py         # M1: DeepFace VGG-Face emotion analysis
    ├── eye_module.py          # M2: MediaPipe EAR blink/fatigue tracking
    ├── vitals_module.py       # M3: HR + Temp + SpO2 strain computation
    ├── fusion_module.py       # M4: EMA quality-aware multimodal fusion
    ├── alert_module.py        # M5: Rule-based psychological support engine
    ├── database_module.py     # M6: SQLite logging with schema migration
    └── live_state.py          # Shared thread-safe container for WebRTC streaming
```

---

## ⚙️ Setup & Run

### 1. Clone the Repository
```bash
git clone https://github.com/MukulKumbhar/MAITRI_2.0.git
cd MAITRI_2.0
```

### 2. Create Conda Environment (Python 3.11)
> MediaPipe requires Python < 3.13. Use the `maitri` conda env.

```bash
# One-time setup
conda create -n maitri python=3.11 -y
conda activate maitri

# Install core packages (fast — pre-compiled binaries)
conda install -c conda-forge numpy pandas streamlit opencv -y

# Install ML & WebRTC packages
pip install --prefer-binary deepface mediapipe streamlit-webrtc aiortc
```

### 3. Run the Dashboard
```bash
conda activate maitri
streamlit run app.py
```
Dashboard opens at **http://localhost:8501**

> **Without activating conda:**
> ```bash
> /home/mikey/anaconda3/envs/maitri/bin/python -m streamlit run app.py
> ```

---

## 🔬 Architecture

```
INPUT:  [Webcam snapshot] + [HR/Temp/SpO2 sliders]
           │
    ┌──────┴──────┐
    │  M1: Face   │  M2: Eye     M3: Vitals
    │  DeepFace   │  MediaPipe   Sliders
    │  7 emotions │  EAR/blink   HR+Temp+SpO2
    └──────┬──────┘      │            │
           └─────────────┼────────────┘
                         ▼
               M4: Fusion (EMA Quality Gate)
                         │
                ┌────────┴────────┐
            M5: Alert         M6: SQLite
         Rule-based CBT       Telemetry log
         + grounding
                         │
                    Dashboard
            (Streamlit real-time UI)
```

---

## 📊 Datasets Referenced

| Dataset | Use |
|---------|-----|
| FER-2013 | Facial emotion recognition benchmarks |
| RAVDESS | Speech emotion recognition benchmarks |
| IEMOCAP | SpeechBrain wav2vec2 training data (M_Voice) |

---

## 🔮 Roadmap

- [x] M1 — Face emotion (DeepFace)
- [x] M2 — Eye tracking (MediaPipe EAR)
- [x] M3 — Vitals strain (HR + Temp + SpO2)
- [x] M4 — Multimodal EMA fusion
- [x] M5 — Alert + CBT support engine
- [x] M6 — SQLite logging
- [ ] M_Voice — Speech emotion (SpeechBrain wav2vec2-IEMOCAP)
- [ ] Offline GPT-2 conversation engine
- [ ] PyInstaller packaging (single .exe)

