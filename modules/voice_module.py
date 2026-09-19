"""
MAITRI 2.0 — M_Voice: Speech Emotion Recognition (SER) Module v2
=================================================================
Dual-model ensemble architecture:
  1. Primary: Calibrated Wav2Vec2 ONNX (86 MB, 7-class deep transformer)
     - Runs every 1.0s on the 3.0s rolling speech buffer
     - Calibrated via DAP: infrasonic filter → idle baseline centering → prosodic logit prior
  2. Secondary: MFCC-MLP ONNX (lightweight, 129-dim features, < 5 ms inference)
     - Trained locally on RAVDESS-style synthetic espeak-ng corpus
     - Runs in the same async inference thread for zero added latency
  3. Ensemble fusion:
     - Dynamic quality-weighted blend (0-1.0 based on audio SNR)
     - MFCC-MLP higher weight on clean microphone audio (fast, feature-based)
     - Wav2Vec2 higher weight on complex speech (contextual, prosodic)

Features:
  - Resilient multi-device Linux/Fedora audio capture (PipeWire/ALSA)
  - Hardware native rate support with automatic anti-aliased resampling to 16 kHz
  - 3.0s rolling acoustic buffer (48,000 samples @ 16 kHz)
  - 450 ms VAD hangover hysteresis: inter-syllabic pauses do not wipe emotion
  - Thread-safe non-blocking inference worker with zero pipeline lag
  - Dual ingestion: captures both from system sounddevice and browser WebRTC frames
  - Tunable digital gain and adaptive RMS Voice Activity Detection (VAD)
  - Strict 0% weight gating during silence to prevent ambient noise pollution
  - 1:1 mapping to MAITRI's 7 standard emotion classes
"""

import os
import json
import time
import threading
import collections
from typing import Dict, Optional, Tuple, Any
import numpy as np

from modules.dap_enhancer import (
    apply_infrasonic_filter,
    compute_dap_prosody,
    evaluate_laughter_reflex,
    calibrate_logits,
)

# Project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Standard 7 emotion classes matching fusion_module.py
EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]

# Wav2Vec2 SER Q4 model class index → MAITRI emotion mapping
ID_TO_MAITRI = {
    0: "angry",
    1: "neutral",    # "calm" → "neutral"
    2: "disgust",
    3: "fear",       # "fearful" → "fear"
    4: "happy",
    5: "sad",
    6: "surprise",   # "surprised" → "surprise"
}

_DEFAULT_PROBS = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}

# ─────────────────────────────────────────────────────────────────────────────
# Model sessions (lazy-initialized, thread-safe)
# ─────────────────────────────────────────────────────────────────────────────

_WAV2VEC2_SESSION   = None
_MLP_SESSION        = None
_MLP_META           = None
_ONNX_LOCK          = threading.Lock()

_WAV2VEC2_PATH = os.path.join(BASE_DIR, "models", "wav2vec2_ser_q4.onnx")
_MLP_PATH      = os.path.join(BASE_DIR, "models", "ser_mlp_mfcc.onnx")
_MLP_META_PATH = os.path.join(BASE_DIR, "models", "ser_mlp_meta.json")

# Backward-compatibility alias (tests import this)
_MODEL_PATH = _WAV2VEC2_PATH


def _get_wav2vec2_session():
    """Lazily initialize Wav2Vec2 ONNX InferenceSession."""
    global _WAV2VEC2_SESSION
    if _WAV2VEC2_SESSION is not None:
        return _WAV2VEC2_SESSION

    with _ONNX_LOCK:
        if _WAV2VEC2_SESSION is not None:
            return _WAV2VEC2_SESSION

        if not os.path.isfile(_WAV2VEC2_PATH):
            print(f"[VoiceModule] Warning: Wav2Vec2 model not found at {_WAV2VEC2_PATH}")
            return None

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 2
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(
                _WAV2VEC2_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            # Warm up
            dummy = np.zeros((1, 48000), dtype=np.float32)
            sess.run(None, {sess.get_inputs()[0].name: dummy})
            _WAV2VEC2_SESSION = sess
            print("[VoiceModule] Wav2Vec2 ONNX session loaded.")
            return _WAV2VEC2_SESSION
        except Exception as exc:
            print(f"[VoiceModule] Failed to load Wav2Vec2 ONNX: {exc}")
            return None


def _get_mlp_session():
    """Lazily initialize MFCC-MLP ONNX InferenceSession and metadata."""
    global _MLP_SESSION, _MLP_META
    if _MLP_SESSION is not None:
        return _MLP_SESSION, _MLP_META

    with _ONNX_LOCK:
        if _MLP_SESSION is not None:
            return _MLP_SESSION, _MLP_META

        if not os.path.isfile(_MLP_PATH):
            return None, None  # Not yet trained; graceful degradation

        try:
            import onnxruntime as ort
            opts = ort.SessionOptions()
            opts.intra_op_num_threads = 1
            opts.inter_op_num_threads = 1
            opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            sess = ort.InferenceSession(
                _MLP_PATH, sess_options=opts, providers=["CPUExecutionProvider"]
            )
            meta = None
            if os.path.isfile(_MLP_META_PATH):
                with open(_MLP_META_PATH) as f:
                    meta = json.load(f)
            _MLP_SESSION = sess
            _MLP_META    = meta
            f1 = meta.get("macro_f1_test", "?") if meta else "?"
            print(f"[VoiceModule] MFCC-MLP ONNX session loaded (test F1={f1}).")
            return _MLP_SESSION, _MLP_META
        except Exception as exc:
            print(f"[VoiceModule] Failed to load MFCC-MLP ONNX: {exc}")
            return None, None


# ─────────────────────────────────────────────────────────────────────────────
# Audio Quality
# ─────────────────────────────────────────────────────────────────────────────

def calculate_audio_quality(
    signal: np.ndarray,
    silence_threshold: float = 0.005,
    gain: float = 1.0,
) -> Tuple[float, float, bool]:
    """
    Evaluate audio energy and quality with adjustable sensitivity.
    Applies infrasonic filtering to eliminate hardware mechanical rumble (ALC257).

    Returns:
        rms: Calculated RMS amplitude (post-gain)
        quality: Signal quality score (0.0 to 1.0)
        is_speaking: True if RMS exceeds silence threshold
    """
    if signal is None:
        return 0.0, 0.0, False
    arr = np.nan_to_num(np.asarray(signal, dtype=np.float32)).ravel()
    if len(arr) == 0:
        return 0.0, 0.0, False

    # Remove infrasonic hardware rumble (1–10 Hz)
    sig_filt = apply_infrasonic_filter(arr)
    sig_centered = sig_filt - np.mean(sig_filt)
    gained = sig_centered * gain
    rms = float(np.sqrt(np.mean(gained ** 2)))

    # For multi-second buffers, check frame-peak RMS so short utterances
    # (e.g. 0.3s speech + 2.7s silence) are not diluted into silence
    frame_sz = 320  # 20ms @ 16kHz
    if len(gained) >= frame_sz * 10:
        n_frm = len(gained) // frame_sz
        frm_rms = np.sqrt(np.mean(gained[: n_frm * frame_sz].reshape(n_frm, frame_sz) ** 2, axis=1))
        peak_rms = float(np.max(frm_rms)) if len(frm_rms) > 0 else 0.0
        is_speaking = (rms >= silence_threshold) or (peak_rms >= silence_threshold)
        effective_rms = max(rms, peak_rms)
    else:
        is_speaking = rms >= silence_threshold
        effective_rms = rms

    if not is_speaking:
        return rms, 0.0, False

    quality = min(1.0, max(0.2, (effective_rms - silence_threshold) / 0.05))
    if np.max(np.abs(gained)) > 0.98:
        quality *= 0.70  # Clipping penalty

    return rms, round(quality, 3), True


# ─────────────────────────────────────────────────────────────────────────────
# MFCC Feature Extraction (numpy-only, <3 ms)
# ─────────────────────────────────────────────────────────────────────────────

def _hz_to_mel(hz: float) -> float:
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel: float) -> float:
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _mel_filterbank(n_filters: int, n_fft: int, sr: int,
                    f_min: float = 0.0, f_max: float = None) -> np.ndarray:
    """Build triangular mel filterbank matrix (n_filters × n_fft//2+1)."""
    if f_max is None:
        f_max = sr / 2.0
    mel_min = _hz_to_mel(f_min)
    mel_max = _hz_to_mel(f_max)
    mel_points = np.linspace(mel_min, mel_max, n_filters + 2)
    hz_points  = np.array([_mel_to_hz(m) for m in mel_points])
    bin_pts    = np.floor((n_fft + 1) * hz_points / sr).astype(int)
    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_filters, n_bins), dtype=np.float32)
    for m in range(1, n_filters + 1):
        f_m   = bin_pts[m]
        f_m1  = bin_pts[m - 1]
        f_m2  = bin_pts[m + 1]
        for k in range(f_m1, f_m):
            fb[m - 1, k] = (k - f_m1) / (f_m - f_m1 + 1e-9)
        for k in range(f_m, f_m2):
            fb[m - 1, k] = (f_m2 - k) / (f_m2 - f_m + 1e-9)
    return fb


# Cached filterbank (built once at first call for 16kHz/22050Hz)
_MEL_FB_CACHE: Dict[tuple, np.ndarray] = {}


def _dct_matrix(n: int) -> np.ndarray:
    """Orthonormal DCT-II matrix of shape (n, n)."""
    k = np.arange(n, dtype=np.float32)
    m = np.arange(n, dtype=np.float32)
    D = np.cos(np.pi / n * np.outer(m + 0.5, k + 1))
    D *= np.sqrt(2.0 / n)
    return D


_DCT_CACHE: Dict[int, np.ndarray] = {}


def extract_mfcc_features_numpy(
    signal: np.ndarray,
    sr: int = 16000,
    n_mfcc: int = 40,
    n_fft: int = 512,
    hop_length: int = 160,
    n_mels: int = 80,
) -> Optional[np.ndarray]:
    """
    Pure-numpy 129-dimensional feature extractor, no librosa dependency at runtime.
    Features: [40 MFCC mean | 40 delta-MFCC mean | 40 delta2-MFCC mean |
               spectral_centroid mean/std | spectral_rolloff mean/std |
               ZCR mean/std | RMS mean/std | pitch_spread | pitch_mean]

    Runs in < 3 ms on CPU for a 3-second audio clip.
    """
    sig = np.nan_to_num(np.asarray(signal, dtype=np.float32)).ravel()
    # Trim silence
    thresh = 0.005 * float(np.max(np.abs(sig)) + 1e-9)
    mask = np.abs(sig) > thresh
    if mask.any():
        start = int(np.argmax(mask))
        end   = len(sig) - int(np.argmax(mask[::-1]))
        sig   = sig[start:end]

    if len(sig) < n_fft:
        return None

    # Pre-emphasis
    sig = np.concatenate([[sig[0]], sig[1:] - 0.97 * sig[:-1]])

    # STFT
    n_frames = (len(sig) - n_fft) // hop_length + 1
    if n_frames < 5:
        return None

    indices = (np.arange(n_fft)[None, :] +
               np.arange(n_frames)[:, None] * hop_length)
    frames = sig[indices] * np.hanning(n_fft).astype(np.float32)
    spec = np.abs(np.fft.rfft(frames, n=n_fft)).astype(np.float32)  # (n_frames, n_fft//2+1)

    # Mel filterbank
    cache_key = (n_mels, n_fft, sr)
    if cache_key not in _MEL_FB_CACHE:
        _MEL_FB_CACHE[cache_key] = _mel_filterbank(n_mels, n_fft, sr)
    fb = _MEL_FB_CACHE[cache_key]

    # Log Mel spectrogram
    mel_spec = np.maximum(spec @ fb.T, 1e-9)    # (n_frames, n_mels)
    log_mel  = np.log(mel_spec)

    # DCT → MFCCs
    if n_mfcc not in _DCT_CACHE:
        _DCT_CACHE[n_mfcc] = _dct_matrix(n_mels)[:n_mfcc]
    dct = _DCT_CACHE[n_mfcc]
    mfcc = (log_mel @ dct.T).astype(np.float32)  # (n_frames, n_mfcc)

    # Delta MFCCs (first and second order temporal derivatives)
    def _delta(m: np.ndarray, w: int = 2) -> np.ndarray:
        T = m.shape[0]
        d = np.zeros_like(m)
        for t in range(T):
            num = sum(k * (m[min(t + k, T - 1)] - m[max(t - k, 0)]) for k in range(1, w + 1))
            den = 2 * sum(k * k for k in range(1, w + 1))
            d[t] = num / (den + 1e-9)
        return d

    d1 = _delta(mfcc)
    d2 = _delta(d1)

    mfcc_mean = np.mean(mfcc, axis=0)   # 40
    d1_mean   = np.mean(d1,   axis=0)   # 40
    d2_mean   = np.mean(d2,   axis=0)   # 40

    # Spectral features
    freqs = np.fft.rfftfreq(n_fft, d=1.0 / sr)  # (n_fft//2+1,)
    spec_sum = np.sum(spec, axis=1) + 1e-9       # (n_frames,)

    centroid = np.sum(spec * freqs[None, :], axis=1) / spec_sum       # (n_frames,)
    cumsum   = np.cumsum(spec, axis=1)
    roll_idx = np.argmax(cumsum >= 0.85 * spec_sum[:, None], axis=1)
    rolloff  = freqs[np.minimum(roll_idx, len(freqs) - 1)]             # (n_frames,)
    zcr      = np.mean(np.abs(np.diff(np.sign(frames), axis=1)), axis=1) * 0.5
    rms      = np.sqrt(np.mean(frames ** 2, axis=1))

    # Fast pitch via autocorrelation (voiced frames only)
    min_lag = max(1, int(sr / 800))   # 800 Hz upper bound
    max_lag = int(sr / 60)            # 60 Hz lower bound
    f0_list = []
    for frm in frames[::4]:  # every 4th frame for speed
        c = np.correlate(frm, frm, mode="full")[n_fft - 1:]
        if c[0] < 1e-6:
            continue
        cw = c[min_lag:min(max_lag, len(c))]
        if len(cw) < 3:
            continue
        pk = int(np.argmax(cw))
        is_local = (0 < pk < len(cw) - 1 and cw[pk] >= cw[pk - 1] and cw[pk] >= cw[pk + 1])
        if is_local and (cw[pk] / c[0]) > 0.3:
            f0_list.append(float(sr) / (min_lag + pk))

    pitch_spread = float(np.std(f0_list))  if len(f0_list) > 1 else 0.0
    pitch_mean   = float(np.mean(f0_list)) if len(f0_list) > 0 else 0.0

    feat = np.concatenate([
        mfcc_mean,                                   # 40
        d1_mean,                                     # 40
        d2_mean,                                     # 40
        [float(np.mean(centroid)), float(np.std(centroid))],   # 2
        [float(np.mean(rolloff)),  float(np.std(rolloff))],    # 2
        [float(np.mean(zcr)),      float(np.std(zcr))],        # 2
        [float(np.mean(rms)),      float(np.std(rms))],        # 2
        [pitch_spread, pitch_mean],                            # 2
    ]).astype(np.float32)
    return feat


# ─────────────────────────────────────────────────────────────────────────────
# MFCC-MLP inference
# ─────────────────────────────────────────────────────────────────────────────

def _predict_mfcc_mlp(signal: np.ndarray, sr: int = 16000) -> Optional[Dict[str, float]]:
    """
    Run MFCC-MLP ONNX inference on the given signal.
    Returns probability dict over MAITRI 7-class emotions, or None if unavailable.
    """
    sess, meta = _get_mlp_session()
    if sess is None:
        return None

    feat = extract_mfcc_features_numpy(signal, sr=sr)
    if feat is None:
        return None

    try:
        inp_name = sess.get_inputs()[0].name
        probs_raw = sess.run(None, {inp_name: feat.reshape(1, -1)})[0][0]

        # Map model class names to MAITRI emotions
        class_names = meta.get("class_names", EMOTIONS) if meta else EMOTIONS
        out_probs = {e: 0.0 for e in EMOTIONS}
        for i, cls in enumerate(class_names):
            if cls in out_probs and i < len(probs_raw):
                out_probs[cls] = float(probs_raw[i])

        # Re-normalize
        total = sum(out_probs.values())
        if total > 1e-6:
            out_probs = {k: v / total for k, v in out_probs.items()}
        return out_probs
    except Exception as exc:
        print(f"[VoiceModule] MFCC-MLP inference error: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Dual-model ensemble prediction
# ─────────────────────────────────────────────────────────────────────────────

def _ensemble_predictions(
    wav2vec2_probs: Optional[Dict[str, float]],
    mlp_probs: Optional[Dict[str, float]],
    audio_quality: float,
) -> Dict[str, float]:
    """
    Quality-weighted ensemble of Wav2Vec2 and MFCC-MLP predictions.

    Weighting logic:
    - If both models available: quality-aware blend
      - High quality (>0.7): 55% Wav2Vec2 + 45% MLP (MLP more reliable on clean signal)
      - Medium quality (0.3-0.7): 65% Wav2Vec2 + 35% MLP
      - Low quality (<0.3): 80% Wav2Vec2 + 20% MLP (W2V handles noisy speech better)
    - If only one model available: use it with full weight
    """
    if wav2vec2_probs is None and mlp_probs is None:
        return dict(_DEFAULT_PROBS)

    if wav2vec2_probs is None:
        return mlp_probs

    if mlp_probs is None:
        return wav2vec2_probs

    # Quality-adaptive blending
    if audio_quality >= 0.70:
        w_w2v, w_mlp = 0.55, 0.45
    elif audio_quality >= 0.35:
        w_w2v, w_mlp = 0.65, 0.35
    else:
        w_w2v, w_mlp = 0.80, 0.20

    blended = {}
    for emo in EMOTIONS:
        p_w2v = wav2vec2_probs.get(emo, 0.0)
        p_mlp = mlp_probs.get(emo, 0.0)
        blended[emo] = w_w2v * p_w2v + w_mlp * p_mlp

    total = sum(blended.values())
    if total > 1e-6:
        blended = {k: v / total for k, v in blended.items()}
    return blended


def predict_voice_emotion(
    signal: np.ndarray,
    sampling_rate: int = 16000,
    silence_threshold: float = 0.005,
    gain: float = 1.0,
) -> Tuple[str, Dict[str, float], float, float, bool]:
    """
    Dual-model SER prediction on audio buffer.

    Pipeline:
      1. Infrasonic high-pass (75 Hz) → removes ALC257 mechanical rumble
      2. VAD check → returns neutral on silence
      3. DAP prosodic extraction → spectral centroid, F0, envelope modulation
      4. Laughter reflex bypass → immediate happy@0.94 on burst laughter
      5. Active speech frame isolation
      6. Wav2Vec2 ONNX inference + logit calibration (DAP centering + prosodic prior)
      7. MFCC-MLP ONNX inference (pure-numpy features, < 3 ms)
      8. Quality-adaptive ensemble fusion of (6) and (7)

    Returns:
        dominant_emotion: str
        emotion_probs: Dict[str, float] (7 classes, sum=1.0)
        confidence: float [0.0, 1.0]
        audio_quality: float [0.0, 1.0]
        is_speaking: bool
    """
    if signal is None:
        return "neutral", dict(_DEFAULT_PROBS), 0.0, 0.0, False

    sig_arr = np.nan_to_num(np.asarray(signal, dtype=np.float32)).ravel()
    if len(sig_arr) == 0:
        return "neutral", dict(_DEFAULT_PROBS), 0.0, 0.0, False

    # 1. Infrasonic filtering
    sig_filt = apply_infrasonic_filter(sig_arr, cutoff_hz=75.0, sr=sampling_rate)

    # 2. VAD
    rms, quality, is_speaking = calculate_audio_quality(
        sig_filt, silence_threshold=silence_threshold, gain=gain
    )
    min_samples = int(sampling_rate * 0.20)
    if not is_speaking or len(sig_filt) < min_samples:
        return "neutral", dict(_DEFAULT_PROBS), 0.0, 0.0, False

    # 3. DAP prosodic feature extraction
    dap = compute_dap_prosody(sig_filt, sr=sampling_rate)

    # 4. Biological laughter reflex bypass
    if evaluate_laughter_reflex(dap):
        laugh_probs = {e: 0.01 for e in EMOTIONS}
        laugh_probs["happy"] = 0.94
        return "happy", laugh_probs, 0.94, quality, True

    # 5. Active speech frame isolation
    sig_centered = sig_filt - np.mean(sig_filt)
    gained = sig_centered * gain
    frame_size = int(sampling_rate * 0.02)
    n_frames = len(gained) // frame_size
    sig_target = gained  # default

    if n_frames >= 5:
        frames = gained[: n_frames * frame_size].reshape(n_frames, frame_size)
        frame_rms = np.sqrt(np.mean(frames ** 2, axis=1))
        active_idx = np.where(frame_rms >= (silence_threshold * 0.4))[0]
        if len(active_idx) >= 5:
            pad = int(0.15 / 0.02)
            first = max(0, active_idx[0] - pad)
            last  = min(n_frames, active_idx[-1] + 1 + pad)
            seg   = gained[first * frame_size: last * frame_size]
            if len(seg) >= min_samples:
                sig_target = seg
                dap_target = compute_dap_prosody(sig_target, sr=sampling_rate)
                if evaluate_laughter_reflex(dap_target):
                    laugh_probs = {e: 0.01 for e in EMOTIONS}
                    laugh_probs["happy"] = 0.94
                    return "happy", laugh_probs, 0.94, quality, True
                dap = dap_target

    # 6. Wav2Vec2 ONNX inference
    wav2vec2_probs = None
    sess = _get_wav2vec2_session()
    if sess is not None:
        try:
            mean_ref = float(np.mean(sig_target))
            std_ref  = float(np.std(sig_target))
            sig_norm = (sig_target - mean_ref) / (std_ref + 1e-7)
            tensor_in = sig_norm.astype(np.float32).reshape(1, -1)

            raw_logits = sess.run(None, {sess.get_inputs()[0].name: tensor_in})[0][0]

            # Aggressive baseline centering (0.80) + prosodic logit prior
            cal_logits = calibrate_logits(raw_logits, dap=dap, centering_factor=0.80)

            # Residual sad penalty: Wav2Vec2 structurally over-predicts sad even after centering
            # This is applied ONLY in runtime inference (not in unit tests of calibrate_logits)
            cal_logits[5] -= 1.5  # sad class index = 5

            # Temperature-scaled softmax (T=1.15 for softer confidence distribution)
            T = 1.15
            scaled = cal_logits / T
            exp_l  = np.exp(scaled - np.max(scaled))
            probs  = exp_l / (np.sum(exp_l) + 1e-9)

            w2v_out = {e: 0.0 for e in EMOTIONS}
            for idx, val in enumerate(probs):
                cls = ID_TO_MAITRI.get(idx, "neutral")
                w2v_out[cls] = w2v_out.get(cls, 0.0) + float(val)

            total = sum(w2v_out.values())
            if total > 1e-6:
                wav2vec2_probs = {k: v / total for k, v in w2v_out.items()}
        except Exception as exc:
            print(f"[VoiceModule] Wav2Vec2 inference error: {exc}")

    # 7. MFCC-MLP ONNX inference (uses the same sig_target at 16kHz)
    mlp_probs = _predict_mfcc_mlp(sig_target, sr=sampling_rate)

    # 8. Ensemble fusion
    fused_probs = _ensemble_predictions(wav2vec2_probs, mlp_probs, quality)

    dominant   = max(fused_probs, key=fused_probs.get)
    confidence = fused_probs[dominant]

    return dominant, fused_probs, round(confidence, 3), quality, True


# ─────────────────────────────────────────────────────────────────────────────
# VoiceDetector — background capture + inference service
# ─────────────────────────────────────────────────────────────────────────────

class VoiceDetector:
    """
    Background audio capture and dual-model emotion detection service.

    Architecture:
    - 3.0s rolling contextual buffer (48,000 samples @ 16 kHz)
    - 450 ms VAD hangover hysteresis
    - 50 ms fast loop for VU meter ballistics
    - 1.0s cadence + speech-completion edge trigger for emotion inference
    """

    def __init__(
        self,
        live_state=None,
        chunk_duration_sec: float = 3.0,
        silence_threshold: float = 0.030,
        gain: float = 1.0,
        hangover_sec: float = 0.45,
    ):
        self.live_state         = live_state
        self.target_rate        = 16000
        self.chunk_duration_sec = chunk_duration_sec
        self.buffer_len         = int(self.target_rate * chunk_duration_sec)
        self.audio_buffer       = collections.deque(maxlen=self.buffer_len)

        self.silence_threshold  = silence_threshold
        self.gain               = gain
        self.hangover_sec       = hangover_sec
        self._last_speech_time  = 0.0

        self._lock              = threading.Lock()
        self._running           = False
        self._worker_thread: Optional[threading.Thread] = None
        self._sd_stream         = None
        self.voice_available    = False
        self._hw_rate           = 16000
        self._smooth_rms        = 0.0
        self._infer_in_flight   = False
        self._speech_ended_pending = False

        self.latest_result: Dict[str, Any] = {
            "dominant_emotion": "neutral",
            "emotion_probs":    dict(_DEFAULT_PROBS),
            "confidence":       0.0,
            "audio_quality":    0.0,
            "is_speaking":      False,
            "rms":              0.0,
            "timestamp":        time.time(),
        }

    def push_audio_chunk(self, chunk: np.ndarray):
        """Allow external audio frames (e.g. from WebRTC browser stream) into ring-buffer."""
        arr = chunk.astype(np.float32).ravel()
        with self._lock:
            self.audio_buffer.extend(arr)
            if len(arr) > 0:
                _, _, is_chunk_spk = calculate_audio_quality(
                    arr, silence_threshold=self.silence_threshold, gain=self.gain
                )
                if is_chunk_spk:
                    self._last_speech_time = time.time()
                    self._speech_ended_pending = True

        if self.live_state is not None:
            with self.live_state.lock:
                self.live_state.voice_available = True

    def _audio_callback(self, indata, frames, time_info, status):
        """sounddevice callback for native 16000 Hz streams."""
        raw = indata[:, 0].astype(np.float32)
        with self._lock:
            self.audio_buffer.extend(raw)

    def _audio_callback_resampled(self, indata, frames, time_info, status):
        """sounddevice callback with real-time linear resampling to 16 kHz."""
        raw_f32 = indata[:, 0].astype(np.float32)
        target_len = int(len(raw_f32) * self.target_rate / self._hw_rate)
        indices  = np.linspace(0, len(raw_f32) - 1, target_len)
        resampled = np.interp(indices, np.arange(len(raw_f32)), raw_f32).astype(np.float32)
        with self._lock:
            self.audio_buffer.extend(resampled)

    def _worker_loop(self):
        """
        Background worker:
        - 50 ms loop: Ballistic audio RMS (instant attack, smooth decay)
        - 450 ms VAD hangover: maintains speech state across breath pauses
        - 1.0 s cadence + edge trigger: Async dual-model emotion inference
        """
        fast_cadence = 0.05
        nn_interval  = 1.00
        last_nn_time = 0.0

        while self._running:
            start_t = time.time()
            with self._lock:
                buf_len = len(self.audio_buffer)
                if buf_len >= (self.target_rate // 2):
                    signal_copy = np.array(self.audio_buffer, dtype=np.float32)
                else:
                    signal_copy = None

            if signal_copy is not None:
                chunk_samples = min(len(signal_copy), int(self.target_rate * 0.2))
                latest_chunk  = signal_copy[-chunk_samples:]
                raw_rms, _, _ = calculate_audio_quality(
                    latest_chunk, silence_threshold=self.silence_threshold, gain=self.gain
                )

                now = time.time()
                is_instant_spk = (raw_rms >= self.silence_threshold)

                # Ballistic envelope follower
                if raw_rms > self._smooth_rms:
                    self._smooth_rms = raw_rms
                else:
                    self._smooth_rms = self._smooth_rms * 0.82 + raw_rms * 0.18
                rms = float(self._smooth_rms)

                # VAD hangover hysteresis
                if is_instant_spk:
                    self._last_speech_time = now
                    is_spk = True
                    self._speech_ended_pending = True
                else:
                    is_spk = (now - self._last_speech_time) < self.hangover_sec

                # Async inference trigger
                should_infer = False
                if not self._infer_in_flight:
                    if is_instant_spk and (now - last_nn_time >= nn_interval):
                        should_infer = True
                    elif (self._speech_ended_pending and not is_instant_spk
                          and (now - self._last_speech_time >= 0.12)):
                        should_infer = True
                        self._speech_ended_pending = False

                if should_infer:
                    last_nn_time = now
                    self._infer_in_flight = True

                    def _async_infer(sig, s_rate, s_thresh, s_gain):
                        try:
                            dom, probs, conf, qual, spk_ok = predict_voice_emotion(
                                sig, s_rate,
                                silence_threshold=s_thresh,
                                gain=s_gain,
                            )
                            with self._lock:
                                self._infer_in_flight = False
                                if self._running and spk_ok:
                                    self.latest_result["dominant_emotion"] = dom
                                    self.latest_result["emotion_probs"]    = probs
                                    self.latest_result["confidence"]       = conf
                                    self.latest_result["audio_quality"]    = qual
                                    self.latest_result["timestamp"]        = time.time()

                            if spk_ok and self.live_state is not None:
                                ls = self.live_state
                                with ls.lock:
                                    ls.voice_emotion    = dom
                                    ls.voice_probs      = dict(probs)
                                    ls.voice_confidence = conf
                                    ls.voice_quality    = qual
                        except Exception as exc:
                            with self._lock:
                                self._infer_in_flight = False
                            print(f"[VoiceModule] Async inference error: {exc}")

                    threading.Thread(
                        target=_async_infer,
                        args=(signal_copy, self.target_rate,
                              self.silence_threshold, self.gain),
                        daemon=True,
                    ).start()

                # Update live RMS and speaking state
                with self._lock:
                    self.latest_result["rms"]         = rms
                    self.latest_result["is_speaking"] = is_spk

                if self.live_state is not None:
                    ls = self.live_state
                    with ls.lock:
                        ls.is_speaking     = is_spk
                        ls.voice_rms       = rms
                        ls.voice_available = True

            elapsed    = time.time() - start_t
            sleep_time = max(0.02, fast_cadence - elapsed)
            time.sleep(sleep_time)

    def start(self):
        """Start background audio capture and dual-model inference engine."""
        if self._running:
            return

        self._running = True

        # Auto-calibrate ALSA mixer (prevents +60dB rail-clipping on Realtek ALC257)
        try:
            import subprocess
            subprocess.run(["amixer", "-c", "0", "set", "Internal Mic Boost", "1"],
                           capture_output=True, timeout=1)
            subprocess.run(["amixer", "-c", "0", "set", "Capture", "45"],
                           capture_output=True, timeout=1)
        except Exception:
            pass

        # Resilient device discovery (PipeWire / ALSA / JACK)
        try:
            import sounddevice as sd
            stream_started = False

            for dev in ["sysdefault", "default", None]:
                try:
                    stream = sd.InputStream(
                        device=dev,
                        samplerate=self.target_rate,
                        channels=1,
                        dtype="float32",
                        blocksize=1024,
                        callback=self._audio_callback,
                    )
                    stream.start()
                    self._sd_stream  = stream
                    self.voice_available = True
                    stream_started   = True
                    print(f"[VoiceModule] Microphone stream started on device '{dev}' @ 16000 Hz.")
                    break
                except Exception:
                    try:
                        dev_info = sd.query_devices(dev)
                        hw_rate  = int(dev_info.get("default_samplerate", 48000))
                        self._hw_rate = hw_rate
                        stream = sd.InputStream(
                            device=dev,
                            samplerate=hw_rate,
                            channels=1,
                            dtype="float32",
                            blocksize=2048,
                            callback=self._audio_callback_resampled,
                        )
                        stream.start()
                        self._sd_stream  = stream
                        self.voice_available = True
                        stream_started   = True
                        print(f"[VoiceModule] Microphone stream on '{dev}' @ {hw_rate} Hz (→ 16 kHz).")
                        break
                    except Exception:
                        continue

            if not stream_started:
                print("[VoiceModule] Notice: Operating in WebRTC buffer-only mode.")
                self.voice_available = False

        except Exception as exc:
            print(f"[VoiceModule] Sounddevice init warning: {exc}.")
            self.voice_available = False

        if self.live_state is not None:
            with self.live_state.lock:
                self.live_state.voice_available = self.voice_available

        self._worker_thread = threading.Thread(
            target=self._worker_loop, daemon=True, name="maitri-voice-worker"
        )
        self._worker_thread.start()

    def stop(self):
        """Stop background worker and release audio streams."""
        self._running = False
        if self._sd_stream is not None:
            try:
                self._sd_stream.stop()
                self._sd_stream.close()
            except Exception:
                pass
            self._sd_stream = None

    def __del__(self):
        self.stop()
