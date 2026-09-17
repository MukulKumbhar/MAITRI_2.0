"""
MAITRI 2.0 — M1: Face Emotion Analysis Module
Uses DeepFace (VGG-Face backend) for 7-class emotion recognition.

IMPORTANT: DeepFace / TensorFlow are LAZY-loaded on first analyze_frame() call.
No TF code runs at import time — prevents Streamlit startup segfault.
"""

import threading
from dataclasses import dataclass
from typing import Dict, Optional

import cv2
import numpy as np

# ── Constants ─────────────────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

_EMO_COLORS_BGR = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0),
    "fear":     (128, 0,   128),
    "happy":    (0,   215, 255),
    "neutral":  (180, 180, 180),
    "sad":      (220, 100, 50),
    "surprise": (0,   165, 255),
}

# ── Lazy-init state ────────────────────────────────────────────────────────
_df_lock      = threading.Lock()
_DeepFace     = None          # set on first call
_df_available = None          # None = not yet checked
_df_err_msg   = ""


@dataclass
class FaceResult:
    emotion_probs:    Dict[str, float]       # 0–100 scale (raw DeepFace)
    dominant_emotion: str
    face_confidence:  float                  # 0–1
    face_quality:     float                  # 0–1
    annotated_img:    Optional[np.ndarray]   # BGR with overlay
    error:            Optional[str]


def _uniform_probs() -> Dict[str, float]:
    return {e: 100.0 / len(EMOTIONS) for e in EMOTIONS}


def _get_deepface():
    """Lazy singleton — imports DeepFace (and TF) only on first call."""
    global _DeepFace, _df_available, _df_err_msg
    if _df_available is not None:
        return _DeepFace
    with _df_lock:
        if _df_available is not None:
            return _DeepFace
        try:
            from deepface import DeepFace as _DF
            _DeepFace     = _DF
            _df_available = True
        except Exception as e:
            _df_available = False
            _df_err_msg   = str(e)
    return _DeepFace


def _draw_overlay(
    img: np.ndarray,
    dominant: str,
    emotion_probs: Dict[str, float],
    region: Optional[dict],
) -> np.ndarray:
    out   = img.copy()
    color = _EMO_COLORS_BGR.get(dominant, (255, 255, 255))
    if region:
        x, y = region.get("x", 0), region.get("y", 0)
        w, h = region.get("w", 0), region.get("h", 0)
        if w > 0 and h > 0:
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)
            label = f"{dominant.upper()}  {emotion_probs.get(dominant, 0):.1f}%"
            cv2.putText(out, label, (x, max(y - 10, 20)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2, cv2.LINE_AA)
    return out


def analyze_frame(bgr_frame: np.ndarray) -> FaceResult:
    """
    Live-mode face analysis.  DeepFace/TF loaded on FIRST call only.
    Takes a BGR numpy array from the WebRTC frame directly.
    """
    DF = _get_deepface()
    if DF is None:
        return FaceResult(
            emotion_probs=_uniform_probs(), dominant_emotion="neutral",
            face_confidence=0.0, face_quality=0.0,
            annotated_img=bgr_frame,
            error=f"DeepFace unavailable: {_df_err_msg}",
        )

    if bgr_frame is None or bgr_frame.size == 0:
        return FaceResult(
            emotion_probs=_uniform_probs(), dominant_emotion="neutral",
            face_confidence=0.0, face_quality=0.0,
            annotated_img=bgr_frame, error="Empty frame",
        )

    try:
        results = DF.analyze(
            bgr_frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        result        = results[0]
        emotion_probs = result["emotion"]
        dominant      = result["dominant_emotion"]
        confidence    = emotion_probs.get(dominant, 0.0) / 100.0
        quality       = min(1.0, confidence * 1.2)
        annotated     = _draw_overlay(bgr_frame, dominant, emotion_probs, result.get("region"))
        return FaceResult(
            emotion_probs=emotion_probs, dominant_emotion=dominant,
            face_confidence=confidence, face_quality=quality,
            annotated_img=annotated, error=None,
        )
    except Exception as exc:
        return FaceResult(
            emotion_probs=_uniform_probs(), dominant_emotion="neutral",
            face_confidence=0.0, face_quality=0.0,
            annotated_img=bgr_frame, error=str(exc),
        )


def img_to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR → RGB for st.image()."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
