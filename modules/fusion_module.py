"""
MAITRI 2.0 — M4: Multimodal Fusion Module
Quality-aware weighted fusion of face emotion + voice emotion + vitals strain + eye fatigue.
Uses EMA temporal smoothing with adaptive reactivity to prevent flickering.

Modality Structure:
- Behavioral Channels: Face (visual) + Voice (audio)
  Initial nominal baseline ratio: 60% Face / 40% Voice.
- Physiological Channel: Vitals Strain (cardiac, thermal, SpO₂)
- Ocular Channel: Eye Fatigue & Blink Strain (MediaPipe EAR)

Dynamic Quality-Aware & Silence VAD Gating:
- When astronaut is silent (VAD=0, is_speaking=False), voice weight is strictly 0.0,
  preventing background cabin/microphone noise from interfering with telemetry.
- Face carries 100% of the behavioral signal during silence.
- When speech is detected, weights dynamically balance according to relative signal quality
  or respect manual slider overrides.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional

# 7 standard emotion classes (FER / DeepFace convention)
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Base modality weights (sum to 1.0)
# Behavioral share = 0.60 (Face=0.36, Voice=0.24 -> 60/40 ratio)
# Physiological share = 0.25
# Ocular share = 0.15
_BASE_BEHAVIORAL_W = 0.60
_BASE_FACE_RATIO   = 0.60   # 60% Face of behavioral share
_BASE_VOICE_RATIO  = 0.40   # 40% Voice of behavioral share

_BASE_FACE_W    = _BASE_BEHAVIORAL_W * _BASE_FACE_RATIO   # 0.36
_BASE_VOICE_W   = _BASE_BEHAVIORAL_W * _BASE_VOICE_RATIO  # 0.24
_BASE_VITALS_W  = 0.25
_BASE_EYE_W     = 0.15

# EMA alpha for temporal smoothing of fused probabilities
_EMA_ALPHA_SLOW = 0.15   # smooth, anti-flicker during steady state
_EMA_ALPHA_FAST = 0.40   # reactive, responds quickly to sudden stress spikes (>20% delta)
_EMA_SPIKE_THRESHOLD = 0.20

# EMA alpha for quality gate weight adjustment
_GATE_ALPHA = 0.12

# Weight bounds per modality
_W_MIN = 0.00
_W_MAX = 0.90


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def _normalise(probs: Dict[str, float]) -> Dict[str, float]:
    """Ensure probabilities sum to 1; handle zero-sum edge case."""
    total = sum(probs.values())
    if total <= 0:
        return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}
    return {e: probs.get(e, 0.0) / total for e in EMOTIONS}


def _neutral_probs() -> Dict[str, float]:
    return {
        "angry": 0.0, "disgust": 0.0, "fear": 0.0,
        "happy": 0.0, "neutral": 1.0, "sad": 0.0, "surprise": 0.0
    }


def _uniform() -> Dict[str, float]:
    return {e: 1.0 / len(EMOTIONS) for e in EMOTIONS}


@dataclass
class FusionState:
    """
    Persists EMA state across Streamlit reruns via st.session_state.
    Must be stored as a single object to survive reruns cleanly.
    """
    ema_probs:      Dict[str, float] = field(default_factory=_neutral_probs)
    w_face:         float = _BASE_FACE_W
    w_voice:        float = 0.0     # Starts at 0 until speech is verified
    w_vitals:       float = _BASE_VITALS_W
    w_eye:          float = _BASE_EYE_W
    last_emotion:   str   = "neutral"
    last_stress:    float = 0.0
    initialized:    bool  = False


@dataclass
class FusionResult:
    fused_probs:       Dict[str, float]  # smoothed 7-class distribution
    dominant_emotion:  str
    face_weight:       float
    voice_weight:      float
    vitals_weight:     float
    eye_weight:        float
    stress_pct:        float
    is_speaking:       bool = False


def _vitals_to_probs(vitals_strain: float) -> Dict[str, float]:
    """Map scalar vitals strain (0–1) to emotion pseudo-probabilities."""
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
    """Map eye fatigue strain (0–1) to emotion pseudo-probabilities."""
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
    fatigue_strain: float,
    eye_quality:    float,
    voice_probs:    Optional[Dict[str, float]] = None,
    voice_quality:  float = 0.0,
    is_speaking:    bool = False,
    manual_override: bool = False,
    manual_face_ratio: float = 0.60,
) -> FusionResult:
    """
    Run one multimodal fusion step. Updates `state` in-place.

    Args:
        state:             Persisted EMA state object
        face_probs:        7-class face emotion distribution (or None)
        face_quality:      Face clarity & confidence score (0–1)
        vitals_strain:     Composite physiological strain (0–1)
        fatigue_strain:    Eye fatigue level (0–1)
        eye_quality:       Eye tracking availability (0 or 1)
        voice_probs:       7-class voice emotion distribution (or None)
        voice_quality:     Microphone signal quality (0–1)
        is_speaking:       True if speech activity detected, False if silence
        manual_override:   If True, use user slider weights rather than dynamic quality
        manual_face_ratio: Behavioral ratio allocated to Face (e.g. 0.60 = 60% Face, 40% Voice)
    """
    # ── 1. Build per-modality probability vectors ─────────────────────────
    f_probs = _normalise(face_probs) if face_probs else _neutral_probs()
    v_probs = _normalise(_vitals_to_probs(vitals_strain))
    e_probs = _normalise(_eye_to_probs(fatigue_strain))
    vo_probs = _normalise(voice_probs) if (voice_probs and is_speaking) else _neutral_probs()

    # ── 2. Determine target weights ───────────────────────────────────────
    if manual_override:
        # User manual mode: slider specifies Face vs Voice ratio
        face_ratio = _clamp(manual_face_ratio, 0.0, 1.0)
        voice_ratio = 1.0 - face_ratio

        if not is_speaking:
            # Safety silence gating: if astronaut is silent, voice weight drops to 0
            target_w_face  = _BASE_BEHAVIORAL_W
            target_w_voice = 0.0
        else:
            target_w_face  = _BASE_BEHAVIORAL_W * face_ratio
            target_w_voice = _BASE_BEHAVIORAL_W * voice_ratio

        target_w_vitals = _BASE_VITALS_W
        target_w_eye    = _BASE_EYE_W * eye_quality
    else:
        # Automatic Dynamic Quality-Aware Mode
        q_f = _BASE_FACE_RATIO * face_quality
        q_v = _BASE_VOICE_RATIO * voice_quality if is_speaking else 0.0
        norm_q = q_f + q_v

        if norm_q > 0:
            target_w_face  = _BASE_BEHAVIORAL_W * (q_f / norm_q)
            target_w_voice = _BASE_BEHAVIORAL_W * (q_v / norm_q)
        else:
            # Both face and voice absent/silent: no behavioral signal
            target_w_face  = 0.0
            target_w_voice = 0.0

        target_w_eye = _BASE_EYE_W * eye_quality
        target_w_vitals = _BASE_VITALS_W

    # ── 3. Smooth weights via EMA ─────────────────────────────────────────
    # If not speaking, drop voice weight quickly to 0 to prevent noise leakage
    gate_alpha_voice = 0.35 if not is_speaking else _GATE_ALPHA

    state.w_face   = (1 - _GATE_ALPHA) * state.w_face   + _GATE_ALPHA * target_w_face
    state.w_voice  = (1 - gate_alpha_voice) * state.w_voice  + gate_alpha_voice * target_w_voice
    state.w_eye    = (1 - _GATE_ALPHA) * state.w_eye    + _GATE_ALPHA * target_w_eye
    state.w_vitals = (1 - _GATE_ALPHA) * state.w_vitals + _GATE_ALPHA * target_w_vitals

    # Renormalise so weights sum to exactly 1.0
    total_w = state.w_face + state.w_voice + state.w_vitals + state.w_eye
    if total_w <= 0:
        w_f, w_vo, w_v, w_e = _BASE_FACE_W, 0.0, _BASE_VITALS_W, _BASE_EYE_W
    else:
        w_f  = state.w_face   / total_w
        w_vo = state.w_voice  / total_w
        w_v  = state.w_vitals / total_w
        w_e  = state.w_eye    / total_w

    # ── 4. Weighted probability fusion ────────────────────────────────────
    raw_probs = {
        e: (w_f * f_probs[e] + w_vo * vo_probs[e] + w_v * v_probs[e] + w_e * e_probs[e])
        for e in EMOTIONS
    }

    # ── 5. Adaptive EMA temporal smoothing (anti-flicker) ─────────────────
    prev_stress_norm = state.last_stress / 100.0
    _raw_neg = _clamp(
        raw_probs.get("fear", 0.0) * 1.2
        + raw_probs.get("angry", 0.0) * 1.1
        + raw_probs.get("sad", 0.0) * 1.0
        + raw_probs.get("disgust", 0.0) * 0.7
    )
    _raw_stress_est = _clamp(
        w_f * _raw_neg + w_vo * _raw_neg + w_v * vitals_strain + w_e * fatigue_strain
    )
    stress_delta = abs(_raw_stress_est - prev_stress_norm)
    ema_alpha = _EMA_ALPHA_FAST if stress_delta >= _EMA_SPIKE_THRESHOLD else _EMA_ALPHA_SLOW

    if not state.initialized:
        state.ema_probs = _normalise(raw_probs)
        state.initialized = True
    else:
        smoothed = {
            e: (1 - ema_alpha) * state.ema_probs[e] + ema_alpha * raw_probs[e]
            for e in EMOTIONS
        }
        state.ema_probs = _normalise(smoothed)

    # ── 6. Composite stress calculation ───────────────────────────────────
    neg_score = _clamp(
        state.ema_probs.get("fear", 0.0) * 1.2
        + state.ema_probs.get("angry", 0.0) * 1.1
        + state.ema_probs.get("sad", 0.0) * 1.0
        + state.ema_probs.get("disgust", 0.0) * 0.7
    )

    composite_stress = (
        (w_f + w_vo) * neg_score
        + w_v * vitals_strain
        + w_e * fatigue_strain
    )

    stress_raw = max(
        composite_stress,
        0.80 * vitals_strain,
        0.75 * neg_score,
        0.70 * fatigue_strain if fatigue_strain >= 0.70 else 0.0,
    )
    stress_pct = round(_clamp(stress_raw) * 100, 2)

    dominant = max(state.ema_probs, key=state.ema_probs.get)
    state.last_emotion = dominant
    state.last_stress  = stress_pct

    return FusionResult(
        fused_probs=dict(state.ema_probs),
        dominant_emotion=dominant,
        face_weight=w_f,
        voice_weight=w_vo,
        vitals_weight=w_v,
        eye_weight=w_e,
        stress_pct=stress_pct,
        is_speaking=is_speaking,
    )
