"""
MAITRI 2.0 — M1: Face Emotion Analysis Module with DIP Pre-Processing
Uses DeepFace (VGG-Face backend) for 7-class emotion recognition.

DIP (Digital Image Processing) Enhancements:
- LAB-space CLAHE: Eliminates harsh shadows and low-light degradation
- Adaptive Gamma Correction: Ensures invariant accuracy across all skin tones
- Spatial Domain Unsharp Masking: Counters slight camera and astronaut motion blur
- Laplacian Variance Quality Gating: Guards against false predictions during severe blur

IMPORTANT: DeepFace / TensorFlow are LAZY-loaded on first analyze_frame() call.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from modules.dip_enhancer import compute_blur_metric, enhance_face_patch

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
    face_quality:     float                  # 0–1 (DIP quality-gated)
    annotated_img:    Optional[np.ndarray]   # BGR with overlay
    error:            Optional[str]
    blur_score:       float                  = 100.0
    is_blurry:        bool                   = False
    dip_applied:      bool                   = True
    face_box:         Optional[dict]         = None  # {x, y, w, h} in original frame coords


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
    blur_score: float = 100.0,
    is_blurry: bool = False,
) -> np.ndarray:
    """Draw bounding box, emotion label, and DIP diagnostic HUD."""
    out = img.copy()
    color = _EMO_COLORS_BGR.get(dominant, (255, 255, 255))
    if is_blurry:
        color = (0, 165, 255)  # Amber warning when blurry

    if region:
        x, y = region.get("x", 0), region.get("y", 0)
        w, h = region.get("w", 0), region.get("h", 0)
        if w > 0 and h > 0:
            # Draw face target box with corner brackets
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

            # Primary label: Emotion + Confidence
            label = f"{dominant.upper()} {emotion_probs.get(dominant, 0):.1f}%"
            cv2.putText(out, label, (x, max(y - 12, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

            # Secondary label: DIP Enhancement & Blur Index
            dip_tag = "MOTION BLUR [FUSION GATED]" if is_blurry else f"DIP: CLAHE+GAMMA | BLUR:{blur_score:.0f}"
            tag_color = (0, 165, 255) if is_blurry else (0, 230, 118)
            cv2.putText(out, dip_tag, (x, y + h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, tag_color, 1, cv2.LINE_AA)
    return out


def analyze_frame(bgr_frame: np.ndarray) -> FaceResult:
    """
    Live-mode face analysis with real-time Digital Image Processing (DIP).
    Pipeline:
      1. Crop or isolate face ROI
      2. Apply Adaptive Gamma (low light & melanin adaptation)
      3. Apply LAB-space CLAHE (shadow & illumination invariance)
      4. Apply Unsharp Masking (motion blur sharpening)
      5. Calculate Laplacian Variance blur gating
      6. Pass enhanced frame to DeepFace for maximum classification accuracy
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
        # Process input frame with DIP (shadow/lighting/melanin invariance)
        enhanced_frame, blur_score, is_blurry, dip_meta = enhance_face_patch(bgr_frame)

        results = DF.analyze(
            enhanced_frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        result        = results[0]
        emotion_probs = result["emotion"]
        dominant      = result["dominant_emotion"]
        face_detected = result.get("face_confidence", 1.0) > 0.0
        confidence    = float(emotion_probs.get(dominant, 0.0)) / 100.0 if face_detected else 0.0

        # Quality gating
        h, w = bgr_frame.shape[:2]
        region = result.get("region")
        # If OpenCV detector defaulted to whole frame, no face was actually detected
        is_whole_frame = (
            region is not None
            and region.get("w", 0) >= w - 10
            and region.get("h", 0) >= h - 10
        )

        if not face_detected or is_whole_frame:
            quality = 0.0
            face_box = None
        elif is_blurry:
            quality = max(0.35, min(1.0, confidence * 1.2) * max(0.4, blur_score / 100.0))
            face_box = region
        else:
            quality = max(0.5, min(1.0, confidence * 1.2))
            face_box = region

        annotated = _draw_overlay(
            bgr_frame, dominant, emotion_probs, face_box,
            blur_score=blur_score, is_blurry=is_blurry
        )

        return FaceResult(
            emotion_probs=emotion_probs, dominant_emotion=dominant,
            face_confidence=confidence, face_quality=quality,
            annotated_img=annotated, error=None,
            blur_score=blur_score, is_blurry=is_blurry,
            dip_applied=True,
            face_box=face_box,
        )
    except Exception as exc:
        print(f"[MAITRI] DeepFace analysis error: {exc}")
        return FaceResult(
            emotion_probs=_uniform_probs(), dominant_emotion="neutral",
            face_confidence=0.0, face_quality=0.0,
            annotated_img=bgr_frame, error=str(exc),
        )


def img_to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR → RGB for st.image()."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
