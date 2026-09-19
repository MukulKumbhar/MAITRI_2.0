"""
MAITRI 2.0 — M1: Facial Emotion Recognition (FER) Module
State-of-the-Art Architecture:
- MediaPipe FaceLandmarker with FACS blendshapes (AU12 smile, AU6 cheek squint)
- Pretrained EfficientNet ONNX (~8-12ms CPU inference, AffectNet-trained)
- Isotropic square face cropping (1:1 aspect ratio, 30% margin padding, no face squashing)
- Natural, unmutated RGB inference crops (prevents DIP domain shift)
- Passive DIP quality estimation (Laplacian blur metric + brightness telemetry)
- Unified single-pass frame execution for both Face and Eye telemetry
"""

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

from modules.dip_enhancer import compute_blur_metric

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

_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
_MODELS_DIR   = os.path.join(_PROJECT_ROOT, "models")


@dataclass
class FaceResult:
    emotion_probs:    Dict[str, float]       # 0–100 scale
    dominant_emotion: str
    face_confidence:  float                  # 0–1
    face_quality:     float                  # 0–1 (quality-gated)
    annotated_img:    Optional[np.ndarray]   # BGR with overlay (or None)
    error:            Optional[str]          = None
    blur_score:       float                  = 100.0
    is_blurry:        bool                   = False
    dip_applied:      bool                   = False  # False = natural clean crops (no domain shift)
    face_box:         Optional[dict]         = None   # {"x": x, "y": y, "w": w, "h": h}
    smile_score:      float                  = 0.0    # AU12 score (0.0 to 1.0)
    method:           str                    = "none" # "facs_geometric_smile", "deep_semantic_onnx", etc.


def _uniform_probs() -> Dict[str, float]:
    val = round(100.0 / len(EMOTIONS), 2)
    return {e: val for e in EMOTIONS}


# ── Lazy Singletons ────────────────────────────────────────────────────────
_onnx_lock = threading.Lock()
_onnx_session = None
_onnx_model_path = None
_onnx_img_size = 260
_onnx_class_map = None

# Preallocated ImageNet normalization constants (avoids per-frame allocations)
_NORM_INV_STD = (1.0 / (255.0 * np.array([0.229, 0.224, 0.225], dtype=np.float32))).reshape(1, 1, 3)
_NORM_MEAN_DIV_STD = (np.array([0.485, 0.456, 0.406], dtype=np.float32) / np.array([0.229, 0.224, 0.225], dtype=np.float32)).reshape(1, 1, 3)
_onnx_init_attempted = False

_df_lock = threading.Lock()
_DeepFace = None
_df_available = None
_df_err_msg = ""

_ssd_lock = threading.Lock()
_ssd_net = None


def _get_onnx_session():
    """
    Lazy-load Pretrained EfficientNet ONNX session (~8-12ms CPU inference).
    Searches local models/ dir first, then ~/.hsemotion/.
    """
    global _onnx_session, _onnx_model_path, _onnx_img_size, _onnx_class_map, _onnx_init_attempted
    if _onnx_init_attempted:
        return _onnx_session

    with _onnx_lock:
        if _onnx_init_attempted:
            return _onnx_session
        _onnx_init_attempted = True

        candidate_paths = [
            os.path.join(_MODELS_DIR, "enet_b2_7.onnx"),
            os.path.join(_MODELS_DIR, "enet_b0_8_best_vgaf.onnx"),
            os.path.expanduser("~/.hsemotion/enet_b2_7.onnx"),
            os.path.expanduser("~/.hsemotion/enet_b0_8_best_vgaf.onnx"),
        ]

        found_path = None
        for p in candidate_paths:
            if os.path.isfile(p):
                found_path = p
                break

        if not found_path and os.path.isdir(_MODELS_DIR):
            for fname in os.listdir(_MODELS_DIR):
                if fname.endswith(".onnx"):
                    found_path = os.path.join(_MODELS_DIR, fname)
                    break

        if not found_path:
            return None

        try:
            import onnxruntime as ort
            sess_opts = ort.SessionOptions()
            sess_opts.intra_op_num_threads = min(6, os.cpu_count() or 6)
            sess_opts.inter_op_num_threads = 1
            sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

            session = ort.InferenceSession(found_path, sess_options=sess_opts, providers=["CPUExecutionProvider"])
            _onnx_session = session
            _onnx_model_path = found_path

            # Dynamically determine input image size from model input metadata
            inp_shape = session.get_inputs()[0].shape
            if len(inp_shape) >= 4 and isinstance(inp_shape[2], int) and inp_shape[2] > 0:
                _onnx_img_size = inp_shape[2]
            elif "_b0_" in found_path:
                _onnx_img_size = 224
            else:
                _onnx_img_size = 260

            # Dynamically determine number of classes and mapping
            out_shape = session.get_outputs()[0].shape
            num_classes = out_shape[1] if (len(out_shape) >= 2 and isinstance(out_shape[1], int)) else (7 if "_7" in found_path else 8)

            if num_classes == 7:
                # 7 classes: Anger, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
                _onnx_class_map = {0: "angry", 1: "disgust", 2: "fear", 3: "happy", 4: "neutral", 5: "sad", 6: "surprise"}
            else:
                # 8 classes: Anger, Contempt, Disgust, Fear, Happiness, Neutral, Sadness, Surprise
                _onnx_class_map = {0: "angry", 1: "disgust", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}

            print(f"[MAITRI] Loaded EfficientNet ONNX model from {found_path} ({_onnx_img_size}x{_onnx_img_size}, {num_classes} classes)")
        except Exception as exc:
            print(f"[MAITRI] ONNX session initialization error ({exc})")
            _onnx_session = None

    return _onnx_session


def _get_deepface():
    """Lazy singleton — DeepFace fallback if ONNX is unavailable."""
    global _DeepFace, _df_available, _df_err_msg
    if _df_available is not None:
        return _DeepFace
    with _df_lock:
        if _df_available is not None:
            return _DeepFace
        try:
            from deepface import DeepFace as _DF
            _DeepFace = _DF
            _df_available = True
        except Exception as e:
            _df_available = False
            _df_err_msg = str(e)
    return _DeepFace


def _get_ssd_net():
    """Lazy-load OpenCV SSD ResNet Face Detector as secondary fallback."""
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


# ── Geometric & Crop Helpers ───────────────────────────────────────────────
def extract_isotropic_square_face_box(
    landmarks, frame_w: int, frame_h: int, margin_ratio: float = 0.30
) -> Tuple[dict, Tuple[int, int, int, int]]:
    """
    Computes a centered isotropic (1:1 square) face bounding box from MediaPipe landmarks.
    Margin ratio: 30% padding to prevent 1:1 aspect ratio squashing and avoid clipping chin/mouth.
    """
    if not landmarks:
        cx, cy = frame_w // 2, frame_h // 2
        s = max(40, min(frame_w, frame_h) // 3)
        x1, y1 = max(0, cx - s // 2), max(0, cy - s // 2)
        x2, y2 = min(frame_w, x1 + s), min(frame_h, y1 + s)
        return {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1}, (x1, y1, x2, y2)

    xs = [p.x * frame_w for p in landmarks]
    ys = [p.y * frame_h for p in landmarks]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)

    raw_w = max_x - min_x
    raw_h = max_y - min_y

    # Ensure minimum span so single-point or tiny landmark sets produce a valid crop
    min_span = max(40.0, min(frame_w, frame_h) * 0.08)
    base_span = max(raw_w, raw_h)
    if base_span < min_span:
        base_span = min_span

    side = min(float(min(frame_w, frame_h)), base_span * (1.0 + margin_ratio))

    cx = (min_x + max_x) / 2.0
    cy = (min_y + max_y) / 2.0

    half = side / 2.0
    x1 = int(round(cx - half))
    y1 = int(round(cy - half))
    x2 = int(round(cx + half))
    y2 = int(round(cy + half))

    # Shift box if slightly outside boundaries to maintain square size
    if x1 < 0:
        x2 = min(frame_w, x2 - x1)
        x1 = 0
    if y1 < 0:
        y2 = min(frame_h, y2 - y1)
        y1 = 0
    if x2 > frame_w:
        x1 = max(0, x1 - (x2 - frame_w))
        x2 = frame_w
    if y2 > frame_h:
        y1 = max(0, y1 - (y2 - frame_h))
        y2 = frame_h

    # Enforce strict boundary clamping
    x1 = max(0, min(x1, frame_w))
    y1 = max(0, min(y1, frame_h))
    x2 = max(x1, min(x2, frame_w))
    y2 = max(y1, min(y2, frame_h))

    w = max(0, x2 - x1)
    h = max(0, y2 - y1)
    return {"x": x1, "y": y1, "w": w, "h": h}, (x1, y1, x2, y2)


def evaluate_facs_smile(blendshapes_dict: Dict[str, float]) -> Tuple[bool, float, float]:
    """
    MediaPipe FACS blendshapes:
    - AU12: mouthSmileLeft, mouthSmileRight (Lip Corner Puller)
    - AU6:  cheekSquintLeft, cheekSquintRight (Cheek Raiser - Duchenne marker)
    Returns: (is_smiling, smile_score, cheek_score)
    """
    if not blendshapes_dict:
        return False, 0.0, 0.0

    mouth_l = blendshapes_dict.get("mouthSmileLeft", 0.0)
    mouth_r = blendshapes_dict.get("mouthSmileRight", 0.0)
    cheek_l = blendshapes_dict.get("cheekSquintLeft", 0.0)
    cheek_r = blendshapes_dict.get("cheekSquintRight", 0.0)

    smile_score = float((mouth_l + mouth_r) / 2.0)
    cheek_score = float((cheek_l + cheek_r) / 2.0)

    # Active smile: strong AU12 (>= 0.40) or moderate AU12 (>= 0.28) + AU6 cheek squint (>= 0.20)
    is_smiling = (smile_score >= 0.40) or (smile_score >= 0.28 and cheek_score >= 0.20)
    return is_smiling, smile_score, cheek_score


def fuse_facs_with_deep_emotion(
    deep_dominant: str,
    deep_probs: Dict[str, float],
    deep_conf: float,
    smile_active: bool,
    smile_score: float,
    cheek_score: float,
) -> Tuple[str, Dict[str, float], float, str]:
    """
    Hybrid Geometric + Deep Semantic Emotion Fusion:
    - If smile AU is active, immediately registers 'happy' with high confidence (<1ms).
    - Otherwise, utilizes the full 7-class deep emotion distribution.
    """
    if smile_active:
        happy_pct = round(min(98.5, max(85.0, smile_score * 115.0)), 2)
        rem_pct = 100.0 - happy_pct

        fused_probs = {}
        sum_other = sum(deep_probs.get(e, 0.0) for e in EMOTIONS if e != "happy")
        for e in EMOTIONS:
            if e == "happy":
                fused_probs[e] = happy_pct
            else:
                if sum_other > 0:
                    fused_probs[e] = round((deep_probs.get(e, 0.0) / sum_other) * rem_pct, 2)
                else:
                    fused_probs[e] = round(rem_pct / (len(EMOTIONS) - 1), 2)
        dominant = "happy"
        conf = happy_pct / 100.0
        method = "facs_geometric_smile"
        return dominant, fused_probs, conf, method

    # Mild smile AU (0.20 <= smile_score < 0.28): gently boost happy probability
    if smile_score >= 0.20 and deep_probs:
        boost = (smile_score - 0.20) / 0.20 * 25.0
        fused_probs = dict(deep_probs)
        fused_probs["happy"] = fused_probs.get("happy", 0.0) + boost
        tot = sum(fused_probs.values())
        if tot > 0:
            fused_probs = {k: round(v / tot * 100.0, 2) for k, v in fused_probs.items()}
        dominant = max(fused_probs.items(), key=lambda x: x[1])[0]
        conf = float(fused_probs[dominant]) / 100.0
        method = "hybrid_onnx_facs"
        return dominant, fused_probs, conf, method

    return deep_dominant, deep_probs, deep_conf, "deep_semantic_onnx"


def compute_face_quality(
    confidence: float,
    blur_score: float,
    is_blurry: bool,
    fw: int,
    fh: int,
    frame_w: int,
    frame_h: int,
    mean_brightness: float,
) -> float:
    """Multi-factor passive quality score for fusion weighting."""
    if confidence <= 0.0:
        return 0.0

    blur_penalty = max(0.50, min(1.0, blur_score / 60.0)) if is_blurry else 1.0
    area_ratio = (fw * fh) / max(frame_w * frame_h, 1)
    area_penalty = 0.85 if area_ratio < 0.02 else 1.0
    lighting_penalty = 0.85 if (mean_brightness < 20.0 or mean_brightness > 235.0) else 1.0

    base_q = min(1.0, max(0.20, confidence * 1.20))
    return float(base_q * blur_penalty * area_penalty * lighting_penalty)


# ── Deep Semantic Inference ────────────────────────────────────────────────
def _predict_onnx(face_bgr: np.ndarray) -> Tuple[str, Dict[str, float], float]:
    """
    Pretrained EfficientNet ONNX inference on clean, unmutated face crop.
    ImageNet normalization (NO CLAHE, NO gamma, NO unsharp masking).
    """
    session = _get_onnx_session()
    if session is None or face_bgr is None or face_bgr.size == 0:
        return "neutral", _uniform_probs(), 0.0

    try:
        face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(face_rgb, (_onnx_img_size, _onnx_img_size), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        norm = resized * _NORM_INV_STD - _NORM_MEAN_DIV_STD
        tensor = np.ascontiguousarray(np.transpose(norm, (2, 0, 1))[np.newaxis, ...], dtype=np.float32)

        input_name = session.get_inputs()[0].name
        raw_scores = session.run(None, {input_name: tensor})[0][0]

        exp_s = np.exp(raw_scores - np.max(raw_scores))
        probs_raw = exp_s / np.sum(exp_s)

        cmap = _onnx_class_map or {0: "angry", 1: "disgust", 2: "disgust", 3: "fear", 4: "happy", 5: "neutral", 6: "sad", 7: "surprise"}
        probs_dict = {e: 0.0 for e in EMOTIONS}
        for idx, p in enumerate(probs_raw):
            emo = cmap.get(idx, "neutral")
            probs_dict[emo] = probs_dict.get(emo, 0.0) + float(p) * 100.0

        tot = sum(probs_dict.values())
        if tot > 0:
            probs_dict = {k: round(v / tot * 100.0, 2) for k, v in probs_dict.items()}

        dominant = max(probs_dict.items(), key=lambda x: x[1])[0]
        conf = float(probs_dict[dominant]) / 100.0
        return dominant, probs_dict, conf
    except Exception as exc:
        print(f"[MAITRI] ONNX predict error: {exc}")
        return "neutral", _uniform_probs(), 0.0


def _predict_fallback(face_bgr: np.ndarray) -> Tuple[str, Dict[str, float], float]:
    """Fallback classifier when ONNX model is unavailable."""
    DF = _get_deepface()
    if DF is not None and face_bgr is not None and face_bgr.size > 0:
        try:
            results = DF.analyze(
                face_bgr,
                actions=["emotion"],
                enforce_detection=False,
                detector_backend="skip",
                silent=True,
            )
            res = results[0]
            raw_probs = res["emotion"]
            dominant = res["dominant_emotion"]
            conf = float(raw_probs.get(dominant, 0.0)) / 100.0
            return dominant, raw_probs, conf
        except Exception:
            pass

    return "neutral", _uniform_probs(), 0.20


# ── Unified Pipeline (Face + Eye in Single Background Step) ───────────────
def process_unified_frame(bgr_frame: np.ndarray, eye_session) -> Tuple[FaceResult, Any]:
    """
    Unifies face emotion and eye tracking into a SINGLE pass.
    Eliminates redundant MediaPipe invocations and C++ XNNPACK thread contention.
    """
    from modules.eye_module import (
        EyeResult,
        _get_landmarker,
        extract_face_blendshapes,
        process_landmarks_for_eyes,
    )

    if bgr_frame is None or bgr_frame.size == 0:
        default_face = FaceResult(
            emotion_probs=_uniform_probs(), dominant_emotion="neutral",
            face_confidence=0.0, face_quality=0.0,
            annotated_img=bgr_frame, error="Empty frame",
        )
        default_eye = EyeResult(
            ear=0.30, blink_rate=0.0, fatigue_label="No frame",
            fatigue_strain=0.0, eye_quality=0.0, new_blink=False, available=False,
        )
        return default_face, default_eye

    orig_h, orig_w = bgr_frame.shape[:2]

    # Downscale for high-speed landmark detection if resolution > 480w
    target_w = 480
    if orig_w > target_w:
        scale = target_w / float(orig_w)
        work_frame = cv2.resize(bgr_frame, (target_w, int(orig_h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        work_frame = bgr_frame

    work_h, work_w = work_frame.shape[:2]

    landmarker = _get_landmarker()
    if landmarker is not None:
        try:
            import mediapipe as mp
            img_rgb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res = landmarker.detect(mp_img)

            if res and res.face_landmarks and len(res.face_landmarks) > 0:
                landmarks = res.face_landmarks[0]

                # 1. Eye tracking from landmarks
                eye_res = process_landmarks_for_eyes(landmarks, orig_w, orig_h, eye_session)

                # 2. FACS AU blendshapes
                bs_dict = extract_face_blendshapes(res)
                is_smiling, smile_score, cheek_score = evaluate_facs_smile(bs_dict)

                # 3. Isotropic square face cropping (30% margin) directly on original high-res frame
                box_dict, (x1, y1, x2, y2) = extract_isotropic_square_face_box(
                    landmarks, orig_w, orig_h, margin_ratio=0.30
                )
                face_roi = bgr_frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    face_roi = bgr_frame

                # 4. Passive DIP quality metrics (no pixel mutation)
                blur_score = compute_blur_metric(face_roi)
                is_blurry = blur_score < 20.0
                mean_brightness = float(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))) if face_roi.size > 0 else 120.0

                # 5. Deep semantic emotion classification on clean high-res crop
                onnx_sess = _get_onnx_session()
                if onnx_sess is not None:
                    deep_dom, deep_probs, deep_conf = _predict_onnx(face_roi)
                else:
                    deep_dom, deep_probs, deep_conf = _predict_fallback(face_roi)

                # 6. Hybrid Geometric + Deep Semantic Fusion
                dominant, fused_probs, conf, method = fuse_facs_with_deep_emotion(
                    deep_dom, deep_probs, deep_conf, is_smiling, smile_score, cheek_score
                )

                # 6b. Temporal Exponential Moving Average & Hysteresis Smoothing
                if eye_session is not None:
                    prev_probs = getattr(eye_session, "face_ema_probs", None)
                    prev_dom   = getattr(eye_session, "face_dominant", None)
                    alpha = 1.0 if is_smiling else 0.65

                    if prev_probs is not None:
                        ema_probs = {}
                        for e in EMOTIONS:
                            p = fused_probs.get(e, 0.0)
                            prev_p = prev_probs.get(e, p)
                            ema_probs[e] = alpha * p + (1.0 - alpha) * prev_p
                        tot = sum(ema_probs.values())
                        if tot > 0:
                            ema_probs = {k: round(v / tot * 100.0, 2) for k, v in ema_probs.items()}
                    else:
                        ema_probs = fused_probs

                    cand_dom = max(ema_probs, key=ema_probs.get)
                    cand_p   = ema_probs[cand_dom]
                    prev_p   = ema_probs.get(prev_dom, 0.0) if prev_dom else 0.0

                    if cand_dom == prev_dom or prev_dom is None or is_smiling:
                        final_dom = cand_dom
                    else:
                        # 6% hysteresis margin prevents frame-by-frame label flipping
                        if cand_p > prev_p + 6.0 or (cand_p >= 45.0 and prev_p < 25.0):
                            final_dom = cand_dom
                        else:
                            final_dom = prev_dom

                    final_conf = ema_probs[final_dom] / 100.0
                    eye_session.face_ema_probs = ema_probs
                    eye_session.face_dominant  = final_dom
                    eye_session.face_conf      = final_conf

                    dominant    = final_dom
                    fused_probs = ema_probs
                    conf        = final_conf

                # 7. Multi-factor quality score
                quality = compute_face_quality(
                    conf, blur_score, is_blurry, box_dict["w"], box_dict["h"],
                    orig_w, orig_h, mean_brightness
                )

                face_res = FaceResult(
                    emotion_probs=fused_probs,
                    dominant_emotion=dominant,
                    face_confidence=conf,
                    face_quality=quality,
                    annotated_img=None,
                    error=None,
                    blur_score=blur_score,
                    is_blurry=is_blurry,
                    dip_applied=False,
                    face_box=box_dict,
                    smile_score=smile_score,
                    method=method,
                )

                if eye_session is not None:
                    eye_session.consecutive_missing = 0
                    eye_session.last_face_box = box_dict
                    eye_session.last_face_res = face_res
                    eye_session.last_eye_res = eye_res

                return face_res, eye_res
        except Exception as exc:
            print(f"[MAITRI] Unified FaceLandmarker error: {exc}")

    # Transient motion persistence / coasting window (preserves smooth tracking across fast head movements)
    if eye_session is not None and getattr(eye_session, "last_face_box", None) is not None:
        eye_session.consecutive_missing = getattr(eye_session, "consecutive_missing", 0) + 1
        if eye_session.consecutive_missing <= 3:
            prev_f = eye_session.last_face_res
            prev_e = eye_session.last_eye_res
            coasted_box = dict(eye_session.last_face_box)
            coasted_conf = max(0.20, (prev_f.face_confidence if prev_f else 0.50) * 0.92)
            coasted_probs = dict(eye_session.face_ema_probs) if getattr(eye_session, "face_ema_probs", None) else _uniform_probs()
            coasted_dom = getattr(eye_session, "face_dominant", None) or (prev_f.dominant_emotion if prev_f else "neutral")

            face_res = FaceResult(
                emotion_probs=coasted_probs,
                dominant_emotion=coasted_dom,
                face_confidence=coasted_conf,
                face_quality=max(0.15, (prev_f.face_quality if prev_f else 0.40) * 0.90),
                annotated_img=None,
                error=None,
                blur_score=prev_f.blur_score if prev_f else 30.0,
                is_blurry=False,
                dip_applied=False,
                face_box=coasted_box,
                smile_score=0.0,
                method="motion_tracking_coasting",
            )
            eye_res = EyeResult(
                ear=prev_e.ear if prev_e else 0.30,
                blink_rate=prev_e.blink_rate if prev_e else 0.0,
                fatigue_label=prev_e.fatigue_label if prev_e else "Normal",
                fatigue_strain=prev_e.fatigue_strain if prev_e else 0.10,
                eye_quality=0.85,
                new_blink=False,
                available=True,
            ) if prev_e else EyeResult(
                ear=0.30,
                blink_rate=0.0,
                fatigue_label="Normal",
                fatigue_strain=0.10,
                eye_quality=0.85,
                new_blink=False,
                available=True,
            )
            return face_res, eye_res

    # MediaPipe did not detect landmarks for > 3 frames; check SSD detector fallback directly without re-running MediaPipe
    ssd_res = _detect_ssd_face(work_frame, orig_w, orig_h, scale)
    if ssd_res is not None:
        if eye_session is not None:
            eye_session.consecutive_missing = 0
            eye_session.last_face_box = ssd_res.face_box
            eye_session.last_face_res = ssd_res
        eye_res = EyeResult(
            ear=0.30, blink_rate=0.0,
            fatigue_label="Scanning...",
            fatigue_strain=0.0, eye_quality=0.0,
            new_blink=False, available=True,
        )
        return ssd_res, eye_res

    # No face detected in frame
    if eye_session is not None:
        eye_session.face_ema_probs = None
        eye_session.face_dominant  = None
        eye_session.last_face_box  = None
        eye_session.last_face_res  = None
        eye_session.consecutive_missing = 0
    blur_score = compute_blur_metric(work_frame)
    no_face_res = FaceResult(
        emotion_probs=_uniform_probs(),
        dominant_emotion="neutral",
        face_confidence=0.0,
        face_quality=0.0,
        annotated_img=None,
        error=None,
        blur_score=blur_score,
        is_blurry=blur_score < 20.0,
        dip_applied=False,
        face_box=None,
        smile_score=0.0,
        method="no_face",
    )
    eye_res = EyeResult(
        ear=0.30, blink_rate=0.0,
        fatigue_label="Scanning...",
        fatigue_strain=0.0, eye_quality=0.0,
        new_blink=False, available=True,
    )
    return no_face_res, eye_res


def _detect_ssd_face(
    work_frame: np.ndarray, orig_w: int, orig_h: int, scale: float
) -> Optional[FaceResult]:
    """Secondary fallback face detection using OpenCV SSD ResNet model."""
    try:
        ssd = _get_ssd_net()
        if ssd is None:
            return None

        work_h, work_w = work_frame.shape[:2]
        blob = cv2.dnn.blobFromImage(
            cv2.resize(work_frame, (300, 300)), 1.0, (300, 300), (104.0, 177.0, 123.0)
        )
        ssd.setInput(blob)
        detections = ssd.forward()
        best_conf = 0.0
        best_box = None
        for i in range(detections.shape[2]):
            c = float(detections[0, 0, i, 2])
            if c > 0.40 and c > best_conf:
                best_conf = c
                coords = detections[0, 0, i, 3:7] * np.array([work_w, work_h, work_w, work_h])
                bx1, by1, bx2, by2 = coords.astype(int)
                bx1, by1 = max(0, bx1), max(0, by1)
                bx2, by2 = min(work_w, bx2), min(work_h, by2)
                best_box = (bx1, by1, bx2, by2)

        if best_box is not None:
            bx1, by1, bx2, by2 = best_box
            face_roi = work_frame[by1:by2, bx1:bx2]
            blur_score = compute_blur_metric(face_roi)
            is_blurry = blur_score < 20.0
            mean_brightness = float(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))) if face_roi.size > 0 else 120.0

            onnx_sess = _get_onnx_session()
            if onnx_sess is not None:
                dominant, probs, conf = _predict_onnx(face_roi)
            else:
                dominant, probs, conf = _predict_fallback(face_roi)

            scaled_box = {
                "x": int(bx1 / scale), "y": int(by1 / scale),
                "w": int((bx2 - bx1) / scale), "h": int((by2 - by1) / scale),
            } if scale != 1.0 else {"x": bx1, "y": by1, "w": bx2 - bx1, "h": by2 - by1}

            quality = compute_face_quality(
                conf, blur_score, is_blurry, scaled_box["w"], scaled_box["h"],
                orig_w, orig_h, mean_brightness
            )

            return FaceResult(
                emotion_probs=probs,
                dominant_emotion=dominant,
                face_confidence=conf,
                face_quality=quality,
                annotated_img=None,
                error=None,
                blur_score=blur_score,
                is_blurry=is_blurry,
                dip_applied=False,
                face_box=scaled_box,
                smile_score=0.0,
                method="ssd_detection",
            )
    except Exception as exc:
        print(f"[MAITRI] SSD detector notice: {exc}")
    return None


def _draw_standalone_overlay(
    img: np.ndarray,
    box: Optional[dict],
    dominant: str,
    confidence: float,
    blur_score: float,
    is_blurry: bool,
    method: str = "none",
) -> np.ndarray:
    """Draw sleek aerospace corner-bracket reticle for standalone frame visualization."""
    if img is None:
        return img
    out = img.copy()
    if not box:
        return out
    bx, by = box.get("x", 0), box.get("y", 0)
    bw, bh = box.get("w", 0), box.get("h", 0)
    if bw <= 0 or bh <= 0:
        return out

    EMO_HUD_COLORS = {
        "happy":    (0,   215, 255),
        "neutral":  (210, 210, 210),
        "surprise": (0,   180, 255),
        "sad":      (255, 140, 50),
        "fear":     (210, 90,  210),
        "angry":    (50,  50,  240),
        "disgust":  (50,  200, 50),
    }
    col = (0, 165, 255) if is_blurry else EMO_HUD_COLORS.get(dominant.lower(), (0, 230, 118))
    c_len = max(14, min(bw, bh) // 5)
    thick = 2

    # Corner brackets
    cv2.line(out, (bx, by), (bx + c_len, by), col, thick)
    cv2.line(out, (bx, by), (bx, by + c_len), col, thick)
    cv2.line(out, (bx + bw, by), (bx + bw - c_len, by), col, thick)
    cv2.line(out, (bx + bw, by), (bx + bw, by + c_len), col, thick)
    cv2.line(out, (bx, by + bh), (bx + c_len, by + bh), col, thick)
    cv2.line(out, (bx, by + bh), (bx, by + bh - c_len), col, thick)
    cv2.line(out, (bx + bw, by + bh), (bx + bw - c_len, by + bh), col, thick)
    cv2.line(out, (bx + bw, by + bh), (bx + bw, by + bh - c_len), col, thick)

    # Center pip
    cx, cy = bx + bw // 2, by + bh // 2
    cv2.drawMarker(out, (cx, cy), col, cv2.MARKER_CROSS, 8, 1)

    # Reticle badge
    lbl = f"{dominant.upper()} {confidence*100:.0f}%"
    lbl_y = max(by - 8, 20)
    cv2.putText(out, lbl, (bx, lbl_y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, col, 2, cv2.LINE_AA)
    return out


# ── Standalone Face Analysis ───────────────────────────────────────────────
def analyze_frame(bgr_frame: np.ndarray) -> FaceResult:
    """
    Standalone face analysis function for single-frame calls.
    Uses isotropic square cropping, passive DIP quality, and hybrid emotion recognition.
    """
    if bgr_frame is None or bgr_frame.size == 0:
        return FaceResult(
            emotion_probs=_uniform_probs(), dominant_emotion="neutral",
            face_confidence=0.0, face_quality=0.0,
            annotated_img=bgr_frame, error="Empty frame",
        )

    orig_h, orig_w = bgr_frame.shape[:2]
    target_w = 480
    if orig_w > target_w:
        scale = target_w / float(orig_w)
        work_frame = cv2.resize(bgr_frame, (target_w, int(orig_h * scale)), interpolation=cv2.INTER_LINEAR)
    else:
        scale = 1.0
        work_frame = bgr_frame

    work_h, work_w = work_frame.shape[:2]

    # Try MediaPipe FaceLandmarker
    try:
        from modules.eye_module import _get_landmarker, extract_face_blendshapes
        lm = _get_landmarker()
        if lm is not None:
            import mediapipe as mp
            img_rgb = cv2.cvtColor(work_frame, cv2.COLOR_BGR2RGB)
            mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            res = lm.detect(mp_img)

            if res and res.face_landmarks and len(res.face_landmarks) > 0:
                landmarks = res.face_landmarks[0]
                bs_dict = extract_face_blendshapes(res)
                is_smiling, smile_score, cheek_score = evaluate_facs_smile(bs_dict)

                box_dict, (x1, y1, x2, y2) = extract_isotropic_square_face_box(
                    landmarks, orig_w, orig_h, margin_ratio=0.30
                )
                face_roi = bgr_frame[y1:y2, x1:x2]
                if face_roi.size == 0:
                    face_roi = bgr_frame

                blur_score = compute_blur_metric(face_roi)
                is_blurry = blur_score < 20.0
                mean_brightness = float(np.mean(cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY))) if face_roi.size > 0 else 120.0

                onnx_sess = _get_onnx_session()
                if onnx_sess is not None:
                    deep_dom, deep_probs, deep_conf = _predict_onnx(face_roi)
                else:
                    deep_dom, deep_probs, deep_conf = _predict_fallback(face_roi)

                dominant, fused_probs, conf, method = fuse_facs_with_deep_emotion(
                    deep_dom, deep_probs, deep_conf, is_smiling, smile_score, cheek_score
                )

                quality = compute_face_quality(
                    conf, blur_score, is_blurry, box_dict["w"], box_dict["h"],
                    orig_w, orig_h, mean_brightness
                )

                annotated = _draw_standalone_overlay(
                    bgr_frame, box_dict, dominant, conf, blur_score, is_blurry, method
                )

                return FaceResult(
                    emotion_probs=fused_probs,
                    dominant_emotion=dominant,
                    face_confidence=conf,
                    face_quality=quality,
                    annotated_img=annotated,
                    error=None,
                    blur_score=blur_score,
                    is_blurry=is_blurry,
                    dip_applied=False,
                    face_box=box_dict,
                    smile_score=smile_score,
                    method=method,
                )
    except Exception as exc:
        print(f"[MAITRI] MediaPipe in analyze_frame notice: {exc}")

    # Secondary: SSD detector
    ssd_res = _detect_ssd_face(work_frame, orig_w, orig_h, scale)
    if ssd_res is not None:
        annotated = _draw_standalone_overlay(
            bgr_frame, ssd_res.face_box, ssd_res.dominant_emotion,
            ssd_res.face_confidence, ssd_res.blur_score, ssd_res.is_blurry, ssd_res.method
        )
        ssd_res.annotated_img = annotated
        return ssd_res

    # No face detected
    blur_score = compute_blur_metric(work_frame)
    return FaceResult(
        emotion_probs=_uniform_probs(),
        dominant_emotion="neutral",
        face_confidence=0.0,
        face_quality=0.0,
        annotated_img=bgr_frame,
        error=None,
        blur_score=blur_score,
        is_blurry=blur_score < 20.0,
        dip_applied=False,
        face_box=None,
        smile_score=0.0,
        method="no_face",
    )


def img_to_rgb(img: np.ndarray) -> np.ndarray:
    """BGR → RGB helper for Streamlit display."""
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
