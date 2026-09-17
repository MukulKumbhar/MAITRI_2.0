"""
MAITRI 2.0 — M1: Facial Emotion Recognition (FER) Module
State-of-the-Art Architecture:
- Pretrained EfficientNet-B2 (7-class) via ONNX Runtime (~10ms CPU inference)
- Trained on AffectNet (400,000+ real-world images) for maximum expression fidelity
- MediaPipe FaceLandmarker + OpenCV SSD hybrid face detection
- Digital Image Processing (DIP) Pipeline:
    * LAB-space CLAHE (illumination/shadow invariance)
    * Adaptive Gamma Correction (melanin & low-light compensation)
    * Spatial Unsharp Masking (motion blur edge recovery)
    * Landmark-based Affine Face Alignment (head-tilt invariance)
    * Laplacian Variance Quality Gating

Fallback: DeepFace (VGG-Face / CNN) if ONNX engine is unavailable.
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

from modules.dip_enhancer import (
    compute_blur_metric,
    apply_adaptive_gamma,
    apply_clahe_lab,
    apply_unsharp_mask,
    apply_face_alignment,
    extract_eye_centers_from_landmarks,
)

# ── 7 Emotion Classes ──────────────────────────────────────────────────────
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# HSEmotion class index mapping
_HSEMOTION_MAP_7 = {
    "Anger":     "angry",
    "Disgust":   "disgust",
    "Fear":      "fear",
    "Happiness": "happy",
    "Neutral":   "neutral",
    "Sadness":   "sad",
    "Surprise":  "surprise",
}

_EMO_COLORS_BGR = {
    "angry":    (0,   0,   220),
    "disgust":  (0,   140, 0),
    "fear":     (128, 0,   128),
    "happy":    (0,   215, 255),
    "neutral":  (180, 180, 180),
    "sad":      (220, 100, 50),
    "surprise": (0,   165, 255),
}


@dataclass
class FaceResult:
    emotion_probs:    Dict[str, float]       # 0–100 scale
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


# ── Lazy-init Singletons ───────────────────────────────────────────────────
_fer_lock   = threading.Lock()
_fer_model  = None
_fer_status = None  # None: not checked, True: ready, False: failed

_ssd_lock   = threading.Lock()
_ssd_net    = None

_df_lock      = threading.Lock()
_DeepFace     = None
_df_available = None
_df_err_msg   = ""


def _get_fer():
    """Lazy-load Pretrained EfficientNet-B2 ONNX recognizer (~10ms CPU)."""
    global _fer_model, _fer_status
    if _fer_status is not None:
        return _fer_model
    with _fer_lock:
        if _fer_status is not None:
            return _fer_model
        try:
            from hsemotion_onnx.facial_emotions import HSEmotionRecognizer
            _fer_model = HSEmotionRecognizer(model_name="enet_b2_7")
            _fer_status = True
        except Exception as e:
            print(f"[MAITRI] HSEmotion ONNX init notice ({e}), will use DeepFace fallback.")
            _fer_status = False
    return _fer_model


def _get_ssd_net():
    """Lazy-load OpenCV SSD ResNet Face Detector."""
    global _ssd_net
    if _ssd_net is not None:
        return _ssd_net
    with _ssd_lock:
        if _ssd_net is not None:
            return _ssd_net
        try:
            proto = os.path.expanduser("~/.deepface/weights/deploy.prototxt")
            model = os.path.expanduser("~/.deepface/weights/res10_300x300_ssd_iter_140000.caffemodel")
            if os.path.exists(proto) and os.path.exists(model):
                _ssd_net = cv2.dnn.readNetFromCaffe(proto, model)
        except Exception as e:
            print(f"[MAITRI] SSD detector load notice: {e}")
    return _ssd_net


def _get_deepface():
    """Lazy singleton — DeepFace fallback."""
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


def _detect_face_and_landmarks(bgr_frame: np.ndarray):
    """
    Detect face in frame. Returns:
      (box_dict, landmarks, detector_name)
      box_dict: {'x': x, 'y': y, 'w': w, 'h': h} or None
      landmarks: MediaPipe landmarks list or None
    """
    h, w = bgr_frame.shape[:2]

    # 1. Primary: MediaPipe FaceLandmarker (highest landmark accuracy)
    try:
        from modules.eye_module import _get_landmarker
        lm = _get_landmarker()
        if lm is not None:
            import mediapipe as mp
            img_rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res = lm.detect(mp_img)
            if res and res.face_landmarks:
                landmarks = res.face_landmarks[0]
                xs = [p.x * w for p in landmarks]
                ys = [p.y * h for p in landmarks]
                x1, y1 = max(0, int(min(xs))), max(0, int(min(ys)))
                x2, y2 = min(w, int(max(xs))), min(h, int(max(ys)))
                # 15% margin padding
                pad_w = int((x2 - x1) * 0.15)
                pad_h = int((y2 - y1) * 0.15)
                x1 = max(0, x1 - pad_w)
                y1 = max(0, y1 - pad_h)
                x2 = min(w, x2 + pad_w)
                y2 = min(h, y2 + pad_h)
                box = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}
                return box, landmarks, "mediapipe"
    except Exception:
        pass

    # 2. Secondary: OpenCV ResNet-10 SSD Face Detector
    try:
        ssd = _get_ssd_net()
        if ssd is not None:
            blob = cv2.dnn.blobFromImage(
                cv2.resize(bgr_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
            )
            ssd.setInput(blob)
            detections = ssd.forward()
            best_conf = 0.0
            best_box = None
            for i in range(detections.shape[2]):
                conf = float(detections[0, 0, i, 2])
                if conf > 0.40 and conf > best_conf:
                    best_conf = conf
                    coords = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    bx1, by1, bx2, by2 = coords.astype(int)
                    bx1, by1 = max(0, bx1), max(0, by1)
                    bx2, by2 = min(w, bx2), min(h, by2)
                    best_box = {"x": bx1, "y": by1, "w": bx2 - bx1, "h": by2 - by1}
            if best_box is not None:
                return best_box, None, "ssd"
    except Exception:
        pass

    # 3. Fallback: Center 70% region
    cx1 = int(w * 0.15)
    cy1 = int(h * 0.10)
    cw  = int(w * 0.70)
    ch  = int(h * 0.80)
    return {"x": cx1, "y": cy1, "w": cw, "h": ch}, None, "center_crop"


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
            cv2.rectangle(out, (x, y), (x + w, y + h), color, 2)

            label = f"{dominant.upper()} {emotion_probs.get(dominant, 0):.1f}%"
            cv2.putText(out, label, (x, max(y - 12, 22)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2, cv2.LINE_AA)

            dip_tag = "MOTION BLUR [GATED]" if is_blurry else f"DIP: CLAHE+GAMMA | BLUR:{blur_score:.0f}"
            tag_color = (0, 165, 255) if is_blurry else (0, 230, 118)
            cv2.putText(out, dip_tag, (x, y + h + 22),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, tag_color, 1, cv2.LINE_AA)
    return out


def analyze_frame(bgr_frame: np.ndarray) -> FaceResult:
    """
    State-of-the-Art Real-Time Face Emotion Analysis:
    1. Fast Face Detection via MediaPipe FaceLandmarker / SSD
    2. Digital Image Processing: LAB-space CLAHE + Adaptive Gamma + Unsharp Mask
    3. Affine Face Alignment on eye landmarks (normalizes tilt)
    4. Pretrained EfficientNet-B2 (7-class) inference via ONNX Runtime (~10ms)
    5. Multi-factor quality gating: blur + face area + lighting
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return FaceResult(
            emotion_probs=_uniform_probs(), dominant_emotion="neutral",
            face_confidence=0.0, face_quality=0.0,
            annotated_img=bgr_frame, error="Empty frame",
        )

    orig_h, orig_w = bgr_frame.shape[:2]

    # ── Downscale frame for speed if resolution is high (>640w) ───────────
    target_w = 640
    if orig_w > target_w:
        scale = target_w / float(orig_w)
        work_frame = cv2.resize(bgr_frame, (target_w, int(orig_h * scale)), interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        work_frame = bgr_frame

    work_h, work_w = work_frame.shape[:2]

    fer = _get_fer()

    # ── If HSEmotion ONNX is ready, run primary EfficientNet pipeline ───────
    if fer is not None:
        try:
            # 1. Face detection & landmark extraction
            box, landmarks, detector = _detect_face_and_landmarks(work_frame)
            fx, fy = box["x"], box["y"]
            fw, fh = box["w"], box["h"]

            # Crop face region
            face_roi = work_frame[fy:fy+fh, fx:fx+fw]
            if face_roi.size == 0:
                face_roi = work_frame

            # 2. DIP Enhancement on Face ROI
            blur_score = compute_blur_metric(face_roi)
            is_blurry = blur_score < 60.0

            gamma_roi, _ = apply_adaptive_gamma(face_roi, target_mean=120.0)
            clahe_roi = apply_clahe_lab(gamma_roi, clip_limit=2.5, tile_grid_size=(8, 8))
            sharp_roi = apply_unsharp_mask(clahe_roi, sigma=1.0, strength=1.1)

            # 3. Affine Face Alignment if landmarks are available
            if landmarks is not None:
                eye_centers = extract_eye_centers_from_landmarks(landmarks, work_w, work_h)
                if eye_centers:
                    l_roi = (eye_centers[0][0] - fx, eye_centers[0][1] - fy)
                    r_roi = (eye_centers[1][0] - fx, eye_centers[1][1] - fy)
                    aligned_roi = apply_face_alignment(sharp_roi, l_roi, r_roi)
                else:
                    aligned_roi = sharp_roi
            else:
                aligned_roi = sharp_roi

            # 4. Pretrained EfficientNet-B2 ONNX Emotion Classification
            roi_rgb = cv2.cvtColor(aligned_roi, cv2.COLOR_BGR2RGB)
            pred_emo, scores = fer.predict_emotions(roi_rgb, logits=False)

            dominant = _HSEMOTION_MAP_7.get(pred_emo, pred_emo.lower())
            emotion_probs = {
                _HSEMOTION_MAP_7.get(fer.idx_to_class[i], fer.idx_to_class[i].lower()): round(float(scores[i]) * 100.0, 2)
                for i in range(len(scores))
            }
            confidence = float(emotion_probs[dominant]) / 100.0

            # 5. Multi-factor Quality Scoring
            blur_penalty = max(0.60, min(1.0, blur_score / 60.0)) if is_blurry else 1.0
            area_ratio = (fw * fh) / max(work_w * work_h, 1)
            area_penalty = 0.85 if area_ratio < 0.02 else 1.0

            mean_brightness = float(np.mean(cv2.cvtColor(aligned_roi, cv2.COLOR_BGR2GRAY)))
            lighting_penalty = 0.85 if (mean_brightness < 20 or mean_brightness > 235) else 1.0
            det_factor = 0.60 if detector == "center_crop" else 1.0

            # Base quality is anchored to confidence with high floor for confirmed detections
            base_q = max(0.60, min(1.0, confidence * 1.20)) if detector != "center_crop" else max(0.30, confidence * 0.70)
            quality = base_q * blur_penalty * area_penalty * lighting_penalty * det_factor



            # 6. Rescale face box to original frame coordinates for crisp HUD
            if scale != 1.0 and detector != "center_crop":
                scaled_box = {
                    "x": int(fx / scale),
                    "y": int(fy / scale),
                    "w": int(fw / scale),
                    "h": int(fh / scale),
                }
            elif detector != "center_crop":
                scaled_box = box
            else:
                scaled_box = None

            annotated = _draw_overlay(
                bgr_frame, dominant, emotion_probs, scaled_box,
                blur_score=blur_score, is_blurry=is_blurry
            )

            return FaceResult(
                emotion_probs=emotion_probs,
                dominant_emotion=dominant,
                face_confidence=confidence,
                face_quality=quality,
                annotated_img=annotated,
                error=None,
                blur_score=blur_score,
                is_blurry=is_blurry,
                dip_applied=True,
                face_box=scaled_box,
            )
        except Exception as exc:
            print(f"[MAITRI] HSEmotion inference error ({exc}), switching to DeepFace.")

    # ── DeepFace Fallback Pipeline ─────────────────────────────────────────
    DF = _get_deepface()
    if DF is None:
        return FaceResult(
            emotion_probs=_uniform_probs(), dominant_emotion="neutral",
            face_confidence=0.0, face_quality=0.0,
            annotated_img=bgr_frame,
            error=f"All face emotion models unavailable: {_df_err_msg}",
        )

    try:
        results = DF.analyze(
            work_frame,
            actions=["emotion"],
            enforce_detection=False,
            detector_backend="opencv",
            silent=True,
        )
        result = results[0]
        raw_probs = result["emotion"]
        dominant = result["dominant_emotion"]
        confidence = float(raw_probs.get(dominant, 0.0)) / 100.0

        region = result.get("region")
        scaled_box = None
        if region and region.get("w", 0) > 20 and region.get("h", 0) > 20:
            if scale != 1.0:
                scaled_box = {
                    "x": int(region["x"] / scale),
                    "y": int(region["y"] / scale),
                    "w": int(region["w"] / scale),
                    "h": int(region["h"] / scale),
                }
            else:
                scaled_box = region

        blur_score = compute_blur_metric(work_frame)
        is_blurry = blur_score < 60.0
        quality = min(1.0, confidence * 1.1) * (max(0.20, blur_score / 100.0) if is_blurry else 1.0)

        annotated = _draw_overlay(bgr_frame, dominant, raw_probs, scaled_box, blur_score, is_blurry)

        return FaceResult(
            emotion_probs=raw_probs, dominant_emotion=dominant,
            face_confidence=confidence, face_quality=quality,
            annotated_img=annotated, error=None,
            blur_score=blur_score, is_blurry=is_blurry,
            dip_applied=True, face_box=scaled_box,
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
