"""
MAITRI 2.0 — M4: Multimodal Fusion Module
Quality-aware weighted fusion of face emotion + vitals strain + eye fatigue.
Uses EMA temporal smoothing to prevent single-frame flickering.

No voice module yet (to be integrated as M_Voice in a future phase).
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

# 7 standard emotion classes (FER / DeepFace convention)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Base modality weights (must sum to 1.0)
_BASE_FACE_W    = 0.60
_BASE_VITALS_W  = 0.25
_BASE_EYE_W     = 0.15

# EMA alpha for temporal smoothing of fused probabilities
_EMA_ALPHA = 0.15

# EMA alpha for quality gate weight adjustment
_GATE_ALPHA = 0.05

# Weight bounds per modality (prevents a single modality dominating)
_W_MIN = 0.10
_W_MAX = 0.80


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _normalise(probs: Dict[str, float]) -> Dict[str, float]:
    """Ensure probabilities sum to 1; handle zero-sum edge case."""
    total = sum(probs.values())
    if total <= 0:
        return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    return {e: probs.get(e, 0.0) / total for e in EMOTIONS}


def _uniform() -> Dict[str, float]:
    return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}


@dataclass
class FusionState:
    """
    Persists EMA state across Streamlit reruns via st.session_state.
    Must be stored as a single object to survive reruns cleanly.
    """
    ema_probs:      Dict[str, float] = field(default_factory=_uniform)
    w_face:         float = _BASE_FACE_W
    w_vitals:       float = _BASE_VITALS_W
    w_eye:          float = _BASE_EYE_W
    last_emotion:   str   = "neutral"
    last_stress:    float = 0.0


@dataclass
class FusionResult:
    fused_probs:       Dict[str, float]  # smoothed 7-class distribution
    dominant_emotion:  str
    face_weight:       float
    vitals_weight:     float
    eye_weight:        float
    stress_pct:        float


def _vitals_to_probs(vitals_strain: float) -> Dict[str, float]:
    """
    Map a scalar vitals strain (0–1) to a pseudo-probability distribution
    over the 7 emotion classes. High strain → more sad/fear/angry signal.
    """
    # Low strain → neutral/happy; high strain → sad/fear/angry
    neutral_p = _clamp(1.0 - vitals_strain)
    stress_p  = _clamp(vitals_strain)
    return {
        "angry":    stress_p * 0.30,
        "disgust":  stress_p * 0.10,
        "fear":     stress_p * 0.30,
        "happy":    neutral_p * 0.40,
        "neutral":  neutral_p * 0.60,
        "sad":      stress_p * 0.30,
        "surprise": 0.0,
    }


def _eye_to_probs(fatigue_strain: float) -> Dict[str, float]:
    """
    Map eye fatigue strain (0–1) to emotion pseudo-probabilities.
    Fatigue → sad/neutral signal.
    """
    neutral_p = _clamp(1.0 - fatigue_strain * 0.8)
    sad_p     = _clamp(fatigue_strain * 0.6)
    return {
        "angry":    0.0,
        "disgust":  0.0,
        "fear":     fatigue_strain * 0.10,
        "happy":    neutral_p * 0.20,
        "neutral":  neutral_p * 0.80,
        "sad":      sad_p,
        "surprise": 0.0,
    }


def fuse(
    state:          FusionState,
    face_probs:     Optional[Dict[str, float]],
    face_quality:   float,
    vitals_strain:  float,
    fatigue_strain: float,   # 0=normal, 1=very fatigued
    eye_quality:    float,   # 1 if eye data available, 0 if not
) -> FusionResult:
    """
    Run one fusion step. Updates `state` in-place (EMA weights + smoothed probs).

    Args:
        state:          Persisted EMA state object
        face_probs:     DeepFace 7-class probability dict (None if no face detected)
        face_quality:   Confidence of face detection (0–1)
        vitals_strain:  Composite physiological strain (0–1)
        fatigue_strain: Eye fatigue level (0–1)
        eye_quality:    1 if MediaPipe data is available, 0 if not
    """
    # ── Build per-modality probability vectors ────────────────────────────
    f_probs = _normalise(face_probs) if face_probs else _uniform()
    v_probs = _normalise(_vitals_to_probs(vitals_strain))
    e_probs = _normalise(_eye_to_probs(fatigue_strain))

    # ── Quality-aware EMA weight update ──────────────────────────────────
    # If face is unavailable, shift weight toward vitals
    target_w_face   = _clamp(_BASE_FACE_W   * face_quality, _W_MIN, _W_MAX)
    target_w_eye    = _clamp(_BASE_EYE_W    * eye_quality,  _W_MIN, _W_MAX)

    state.w_face   = (1 - _GATE_ALPHA) * state.w_face   + _GATE_ALPHA * target_w_face
    state.w_eye    = (1 - _GATE_ALPHA) * state.w_eye    + _GATE_ALPHA * target_w_eye

    # Redistribute remaining weight to vitals
    used = state.w_face + state.w_eye
    state.w_vitals = _clamp(1.0 - used, _W_MIN, _W_MAX)

    # Renormalise so weights sum to 1
    total_w = state.w_face + state.w_vitals + state.w_eye
    w_f = state.w_face   / total_w
    w_v = state.w_vitals / total_w
    w_e = state.w_eye    / total_w

    # ── Weighted fusion ───────────────────────────────────────────────────
    raw_probs = {
        e: w_f * f_probs[e] + w_v * v_probs[e] + w_e * e_probs[e]
        for e in EMOTIONS
    }

    # ── EMA temporal smoothing ────────────────────────────────────────────
    smoothed = {
        e: (1 - _EMA_ALPHA) * state.ema_probs[e] + _EMA_ALPHA * raw_probs[e]
        for e in EMOTIONS
    }
    state.ema_probs = _normalise(smoothed)

    # ── Derive stress percentage ──────────────────────────────────────────
    # Stress = weighted sum of negative emotion probabilities
    neg_score = (
        state.ema_probs["sad"]     * 0.35
        + state.ema_probs["angry"] * 0.25
        + state.ema_probs["fear"]  * 0.25
        + state.ema_probs["disgust"] * 0.15
    )
    # Blend emotion stress signal with raw vitals strain for physiological grounding
    stress_raw = 0.65 * neg_score + 0.35 * vitals_strain
    stress_pct = round(_clamp(stress_raw) * 100, 2)

    dominant = max(state.ema_probs, key=state.ema_probs.get)
    state.last_emotion = dominant
    state.last_stress  = stress_pct

    return FusionResult(
        fused_probs=dict(state.ema_probs),
        dominant_emotion=dominant,
        face_weight=w_f,
        vitals_weight=w_v,
        eye_weight=w_e,
        stress_pct=stress_pct,
    )

