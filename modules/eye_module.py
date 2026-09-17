"""
MAITRI 2.0 — M2: Eye Tracking Module (Blink Rate + Fatigue)
Uses MediaPipe FaceLandmarker (Tasks API v1.0+) to extract 478 face landmarks
and compute Eye Aspect Ratio (EAR).
Tracks blink rate per-session via shared EyeSessionState.

IMPORTANT: Uses LAZY initialization to avoid TF + MediaPipe TFLite segfault.
The FaceLandmarker is created on the FIRST call to analyze_eyes(), NOT at import.
"""

import os
import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

# ── Model path — absolute project root ───────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODEL_PATH   = os.path.join(_PROJECT_ROOT, "face_landmarker.task")

# ── Lazy-init state (created on first call, not at import) ───────────────
_landmarker_lock = threading.Lock()
_landmarker      = None          # created lazily
_mp_available    = None          # None = not yet checked
_mp_err_msg      = ""

# Public alias for external checks (before first call, reports False)
_MP_AVAILABLE    = False         # updated after first successful lazy init


# ── Eye landmark indices (MediaPipe 478-point mesh) ──────────────────────
# Left eye:  [outer, top1, top2, inner, bot2, bot1]
_LEFT_EYE  = [362, 385, 387, 263, 373, 380]
# Right eye: [outer, top1, top2, inner, bot2, bot1]
_RIGHT_EYE = [33,  160, 158, 133, 153, 144]

# ── EAR thresholds ────────────────────────────────────────────────────────
_EAR_BLINK_THRESH   = 0.25   # below → blink
_EAR_FATIGUE_THRESH = 0.22   # sustained below → drowsy

# ── Blink rate thresholds (blinks / minute) ───────────────────────────────
_BLINK_HIGH_THRESH = 25    # stressed / dry eyes
_BLINK_LOW_THRESH  = 8     # hyperfocus / fatigue

# Rolling window for blink rate calculation
_BLINK_WINDOW_SEC = 60.0

# ── Per-session EAR calibration ───────────────────────────────────────────
# First N seconds are used to measure each subject's personal baseline EAR.
# Drowsy threshold = baseline × _EAR_DROWSY_RATIO (relative, not absolute).
_CALIBRATION_SEC    = 30.0   # duration of baseline measurement window
_EAR_DROWSY_RATIO   = 0.80   # drowsy if EAR < baseline × 0.80
_EAR_BLINK_RATIO    = 0.95   # blink if EAR < baseline × 0.95
_EAR_FLOOR          = 0.18   # hard floor — protects users with naturally narrow eyes


@dataclass
class EyeSessionState:
    """Persists across Streamlit reruns via st.session_state['eye_state']."""
    blink_timestamps:    List[float] = field(default_factory=list)
    prev_ear:            float = 0.30
    in_blink:            bool  = False
    # ── Per-session calibration ──────────────────────────────────────────
    calibration_samples: List[float] = field(default_factory=list)
    baseline_ear:        float = 0.0      # 0.0 = not yet calibrated
    is_calibrated:       bool  = False
    session_start_time:  float = field(default_factory=time.time)


@dataclass
class EyeResult:
    ear:             float   # current Eye Aspect Ratio (avg both eyes)
    blink_rate:      float   # blinks/minute over last 60 s
    fatigue_label:   str     # "Normal" / "Drowsy" / "Hyperfocused" / "Stressed Eyes"
    fatigue_strain:  float   # 0–1 (used by fusion module)
    eye_quality:     float   # 1 if data valid, 0 if not
    new_blink:       bool    # True if a new blink was registered this frame
    available:       bool    # False if MediaPipe not working


def _ear_from_landmarks(landmarks, indices, img_w: int, img_h: int) -> float:
    """Compute Eye Aspect Ratio for one eye from 6 landmark indices."""
    pts = np.array(
        [[landmarks[i].x * img_w, landmarks[i].y * img_h] for i in indices],
        dtype=np.float32,
    )
    A = np.linalg.norm(pts[1] - pts[5])   # vertical 1
    B = np.linalg.norm(pts[2] - pts[4])   # vertical 2
    C = np.linalg.norm(pts[0] - pts[3])   # horizontal
    return (A + B) / (2.0 * C) if C > 1e-6 else 0.0


def _get_landmarker():
    """
    Lazy singleton — creates FaceLandmarker on first call.
    Called AFTER TensorFlow has already loaded (no segfault).
    Thread-safe via _landmarker_lock.
    """
    global _landmarker, _mp_available, _mp_err_msg, _MP_AVAILABLE
    if _mp_available is not None:          # already attempted init
        return _landmarker

    with _landmarker_lock:
        if _mp_available is not None:      # double-check after acquiring lock
            return _landmarker
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision

            if not os.path.exists(_MODEL_PATH):
                raise FileNotFoundError(f"Model not found: {_MODEL_PATH}")

            opts = mp_vision.FaceLandmarkerOptions(
                base_options=mp_python.BaseOptions(model_asset_path=_MODEL_PATH),
                running_mode=mp_vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_tracking_confidence=0.5,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            _landmarker   = mp_vision.FaceLandmarker.create_from_options(opts)
            _mp_available = True
            _MP_AVAILABLE = True
        except Exception as e:
            _mp_available = False
            _mp_err_msg   = str(e)
        return _landmarker


def analyze_eyes(
    image_bgr: np.ndarray,
    session: EyeSessionState,
) -> EyeResult:
    """
    Run MediaPipe FaceLandmarker on the BGR image array (lazy-initialized).
    Detects blinks on rising EAR edge. Updates session state in-place.
    """
    import cv2
    import mediapipe as mp

    landmarker = _get_landmarker()
    if landmarker is None:
        return EyeResult(
            ear=0.30, blink_rate=0.0,
            fatigue_label=f"MediaPipe unavailable: {_mp_err_msg[:60]}",
            fatigue_strain=0.0, eye_quality=0.0,
            new_blink=False, available=False,
        )

    # ── Convert BGR → RGB → MediaPipe Image ──────────────────────────────
    img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    h, w = img_rgb.shape[:2]
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

    # ── Run landmarker ────────────────────────────────────────────────────
    try:
        result = landmarker.detect(mp_image)
    except Exception:
        return EyeResult(
            ear=0.30, blink_rate=0.0,
            fatigue_label="Detection error",
            fatigue_strain=0.0, eye_quality=0.0,
            new_blink=False, available=True,
        )

    if not result.face_landmarks:
        return EyeResult(
            ear=0.30, blink_rate=0.0,
            fatigue_label="No face detected",
            fatigue_strain=0.0, eye_quality=0.0,
            new_blink=False, available=True,
        )

    lm = result.face_landmarks[0]  # first face

    left_ear  = _ear_from_landmarks(lm, _LEFT_EYE,  w, h)
    right_ear = _ear_from_landmarks(lm, _RIGHT_EYE, w, h)
    ear = (left_ear + right_ear) / 2.0

    # ── Per-session EAR calibration ────────────────────────────────────────
    now = time.time()
    # For the first _CALIBRATION_SEC seconds, collect baseline EAR samples.
    # After calibration, use subject-specific relative thresholds.
    elapsed_session = now - session.session_start_time
    if not session.is_calibrated:
        # Only collect samples while eyes appear open (ear > hard floor)
        if ear > _EAR_FLOOR:
            session.calibration_samples.append(ear)
        if elapsed_session >= _CALIBRATION_SEC and len(session.calibration_samples) >= 10:
            session.baseline_ear   = float(np.mean(session.calibration_samples))
            session.is_calibrated  = True
        # During calibration: use global fallback thresholds
        effective_fatigue_thresh = _EAR_FATIGUE_THRESH
        effective_blink_thresh   = _EAR_BLINK_THRESH
        calibrating = True
    else:
        # Use relative thresholds anchored to this subject's baseline
        effective_fatigue_thresh = max(_EAR_FLOOR, session.baseline_ear * _EAR_DROWSY_RATIO)
        effective_blink_thresh   = session.baseline_ear * _EAR_BLINK_RATIO
        calibrating = False

    # ── Blink detection — rising edge ─────────────────────────────────────
    new_blink = False
    if session.prev_ear < effective_blink_thresh and ear >= effective_blink_thresh:
        if session.in_blink:
            session.blink_timestamps.append(now)
            new_blink = True
        session.in_blink = False
    elif ear < effective_blink_thresh:
        session.in_blink = True
    session.prev_ear = ear

    # ── Prune old timestamps ──────────────────────────────────────────────
    cutoff = now - _BLINK_WINDOW_SEC
    session.blink_timestamps = [t for t in session.blink_timestamps if t > cutoff]

    # ── Blink rate (per minute) ───────────────────────────────────────────
    if len(session.blink_timestamps) >= 2:
        elapsed = min(now - session.blink_timestamps[0], _BLINK_WINDOW_SEC)
        blink_rate = len(session.blink_timestamps) / max(elapsed, 1.0) * 60.0
    else:
        blink_rate = 0.0

    # ── Fatigue classification ────────────────────────────────────────────
    if calibrating:
        label, strain = "Calibrating…", 0.0
    elif ear < effective_fatigue_thresh:
        label, strain = "Drowsy", 0.80
    elif blink_rate > _BLINK_HIGH_THRESH:
        label, strain = "Stressed Eyes", 0.55
    elif 0 < blink_rate < _BLINK_LOW_THRESH:
        label, strain = "Hyperfocused", 0.40
    else:
        label, strain = "Normal", 0.10


    return EyeResult(
        ear=round(ear, 3),
        blink_rate=round(blink_rate, 1),
        fatigue_label=label,
        fatigue_strain=strain,
        eye_quality=1.0,
        new_blink=new_blink,
        available=True,
    )
