"""
MAITRI 2.0 — Digital Audio Processing (DAP) Enhancement Pipeline
Real-time infrasonic filtering, prosodic contour analysis, laughter reflex bypass, and logit calibration.

Analogue to Digital Image Processing (DIP in dip_enhancer.py) for audio:
1. Infrasonic FFT High-Pass Filter (75 Hz cutoff) — drops ALC257 mechanical/fan rumble
   by >10x (RMS 0.099 -> <0.009), restoring true VAD silence gating.
2. Prosodic Feature Extraction (pure NumPy, <2.5ms latency):
   - Spectral Centroid (Hz)
   - Fundamental pitch F0 and pitch spread σ(F0) via normalized autocorrelation
   - Envelope modulation depth
   - Envelope autocorrelation periodicity R_env(τ) in the 3.8–7.0 Hz lag window
3. Laughter Reflex Bypass:
   - Immediate reflex detection (R_env ≥ 0.50, mod_depth ≥ 0.70, centroid ≥ 1400 Hz)
   - Bypasses transformer unvoiced sink state to output dominant='happy', confidence=0.94
   - Distinctly isolates laughter (R_env ~ 0.90) from angry yelling (R_env ~ 0.19) and speech.
4. Soft Baseline Centering & Prosodic Logit Prior:
   - Cancels the +8.79 logit unvoiced idle bias (z_cal = z - 0.60 * z0)
   - Injects prosodic priors boosting cheerful intonations and penalizing false sad sink states.
"""

from typing import Any, Dict, Optional, Tuple, Union
import numpy as np

# Infrasonic filter cutoffs
CUTOFF_HZ = 75.0
DEFAULT_SR = 16000

# Wav2Vec2 SER Q4 Idle Baseline Vector (empirical output on unvoiced/quiet audio)
# Classes: [0: angry, 1: neutral/calm, 2: disgust, 3: fear, 4: happy, 5: sad, 6: surprise]
# Note the +8.79 logit bias favoring sad (+5.487) over happy (-3.305).
Z_IDLE_BASELINE = np.array(
    [-3.2386, -0.3466, 0.4936, 1.4274, -3.3049, 5.4871, -1.2110],
    dtype=np.float32,
)


class DAPProsody(dict):
    """
    Lightweight prosodic features container supporting both dictionary
    and object attribute lookups with zero conversion overhead.
    """

    def __init__(
        self,
        spectral_centroid: float = 0.0,
        f0_mean: float = 0.0,
        pitch_spread: float = 0.0,
        modulation_depth: float = 0.0,
        r_env: float = 0.0,
        pitch_slope: float = 0.0,
    ):
        super().__init__(
            spectral_centroid=spectral_centroid,
            f0_mean=f0_mean,
            pitch_spread=pitch_spread,
            modulation_depth=modulation_depth,
            r_env=r_env,
            pitch_slope=pitch_slope,
        )
        self.spectral_centroid = float(spectral_centroid)
        self.f0_mean = float(f0_mean)
        self.pitch_spread = float(pitch_spread)
        self.modulation_depth = float(modulation_depth)
        self.r_env = float(r_env)
        self.pitch_slope = float(pitch_slope)


def apply_infrasonic_filter(
    signal: np.ndarray,
    cutoff_hz: float = CUTOFF_HZ,
    sr: int = DEFAULT_SR,
) -> np.ndarray:
    """
    FFT-based high-pass filter with a smooth cosine transition band to eliminate
    1–10 Hz mechanical/infrasonic rumble without introducing time-domain Gibbs ringing.

    Attenuates hardware rumble RMS from ~0.099 to < 0.009 (>10x noise reduction),
    restoring true VAD silence gating while preserving 100% of vocal formants (≥85 Hz).

    Args:
        signal: 1D or 2D numpy array of audio samples.
        cutoff_hz: Corner frequency in Hz (default 75 Hz).
        sr: Audio sampling rate in Hz (default 16000).

    Returns:
        Filtered 1D float32 numpy array.
    """
    if signal is None:
        return np.array([], dtype=np.float32)

    sig_f = np.nan_to_num(np.asarray(signal, dtype=np.float32)).ravel()
    n = len(sig_f)
    if n < 32:
        return sig_f

    # Reflection padding to eliminate circular convolution boundary wrap-around
    pad_len = min(n, 1024)
    padded = np.pad(sig_f, pad_len, mode="reflect")
    n_pad = len(padded)

    # Frequency grid
    freqs = np.fft.rfftfreq(n_pad, d=1.0 / sr)
    H = np.ones_like(freqs, dtype=np.float32)

    # Cosine taper from 0.5 * cutoff to cutoff
    f_stop = cutoff_hz * 0.5
    zero_mask = freqs <= f_stop
    ramp_mask = (freqs > f_stop) & (freqs < cutoff_hz)

    H[zero_mask] = 0.0
    ramp_arg = np.pi * (freqs[ramp_mask] - f_stop) / (cutoff_hz - f_stop)
    H[ramp_mask] = 0.5 * (1.0 - np.cos(ramp_arg))

    fft_vals = np.fft.rfft(padded)
    filt_padded = np.fft.irfft(fft_vals * H, n=n_pad).astype(np.float32)
    return filt_padded[pad_len : pad_len + n]


def compute_dap_prosody(
    signal: np.ndarray,
    sr: int = DEFAULT_SR,
) -> DAPProsody:
    """
    Extracts high-speed prosodic features in pure NumPy (< 2.5 ms execution time):
    1. Spectral Centroid (Hz) — acoustic brightness/sharpness.
    2. Fundamental Pitch F0 and Pitch Spread σ(F0) — pitch mean and intonation variability.
    3. Modulation Depth — dynamic contrast of the low-frequency envelope.
    4. Envelope Autocorrelation Periodicity R_env(τ) in 3.8–7.0 Hz lag window.

    Args:
        signal: 1D or 2D audio sample array.
        sr: Sampling rate (default 16000).

    Returns:
        DAPProsody object containing extracted acoustic metrics.
    """
    if signal is None:
        return DAPProsody()

    sig = np.nan_to_num(np.asarray(signal, dtype=np.float32)).ravel()
    n = len(sig)
    if n < 320:
        return DAPProsody()

    # ─────────────────────────────────────────────────────────────────────────
    # 1. Spectral Centroid
    # ─────────────────────────────────────────────────────────────────────────
    fft_vals = np.fft.rfft(sig)
    fft_mag = np.abs(fft_vals)
    freqs = np.fft.rfftfreq(n, d=1.0 / sr)
    mag_sum = float(np.sum(fft_mag))
    centroid = float(np.sum(freqs * fft_mag) / (mag_sum + 1e-9)) if mag_sum > 1e-6 else 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # 2. Envelope Modulation & R_env(τ) in [3.8, 7.0] Hz
    # ─────────────────────────────────────────────────────────────────────────
    hop = max(1, int(sr * 0.010))  # 10ms step -> fs_env = 100 Hz @ 16kHz
    win = max(2, int(sr * 0.025))  # 25ms RMS window
    n_frames = (n - win) // hop

    if n_frames >= 15:
        shape = (n_frames, win)
        strides = (sig.strides[0] * hop, sig.strides[0])
        frames = np.lib.stride_tricks.as_strided(sig, shape=shape, strides=strides)
        env = np.sqrt(np.sum(frames * frames, axis=1) / win + 1e-9)

        # Fast sort-based percentiles (<0.04ms vs 7ms for np.percentile)
        env_sorted = np.sort(env)
        p5 = float(env_sorted[int(len(env) * 0.05)])
        p95 = float(env_sorted[min(len(env) - 1, int(len(env) * 0.95))])
        mod_depth = float((p95 - p5) / (p95 + p5 + 1e-6))

        # Normalized autocorrelation of mean-subtracted envelope
        env_centered = env - np.mean(env)
        env_var = float(np.sum(env_centered * env_centered))

        if env_var > 1e-9:
            full_corr = np.correlate(env_centered, env_centered, mode="full")
            corr = full_corr[len(env_centered) - 1 :] / env_var
            fs_env = sr / hop
            tau_min = int(fs_env / 7.0)
            tau_max = int(fs_env / 3.8)
            window_corr = corr[tau_min : min(len(corr), tau_max + 1)]
            if len(window_corr) > 0:
                pk_w = int(np.argmax(window_corr))
                # True periodic rhythm requires an interior local peak, preventing
                # monotonically decaying non-laughing speech blocks from registering false rhythm
                is_local_pk = (pk_w > 0 and window_corr[pk_w] > window_corr[pk_w - 1])
                r_env = max(0.0, float(window_corr[pk_w])) if is_local_pk else 0.0
            else:
                r_env = 0.0
        else:
            r_env = 0.0
    else:
        mod_depth = 0.0
        r_env = 0.0

    # ─────────────────────────────────────────────────────────────────────────
    # 3. Fast Pitch (F0) Tracking & Pitch Spread σ(F0)
    # ─────────────────────────────────────────────────────────────────────────
    p_hop = max(1, int(sr * 0.040))  # 40ms hop
    p_win = max(2, int(sr * 0.050))  # 50ms frame
    n_p_frames = (n - p_win) // p_hop

    if n_p_frames >= 2:
        p_shape = (n_p_frames, p_win)
        p_strides = (sig.strides[0] * p_hop, sig.strides[0])
        p_frames = np.lib.stride_tricks.as_strided(sig, shape=p_shape, strides=p_strides)
        energies = np.sum(p_frames * p_frames, axis=1)
        thresh = float(np.mean(energies) * 0.35)
        active_idx = np.where(energies > thresh)[0]

        # Limit to representative voiced frames for sub-millisecond execution
        if len(active_idx) > 10:
            active_idx = active_idx[np.linspace(0, len(active_idx) - 1, 10, dtype=int)]

        min_lag = int(sr / 400)  # 400 Hz pitch limit
        max_lag = int(sr / 70)   # 70 Hz pitch limit
        f0_list = []

        for idx in active_idx:
            frm = p_frames[idx]
            c = np.correlate(frm, frm, mode="full")[p_win - 1 :]
            cw = c[min_lag:max_lag]
            if len(cw) < 3:
                continue
            pk = int(np.argmax(cw))
            # Require interior local peak to reject monotonically decaying rumble slopes
            is_local_pk = (0 < pk < len(cw) - 1 and cw[pk] >= cw[pk - 1] and cw[pk] >= cw[pk + 1])
            if is_local_pk and c[0] > 1e-6 and (cw[pk] / c[0]) > 0.35:
                f0_list.append(sr / (min_lag + pk))

        f0_mean = float(np.mean(f0_list)) if f0_list else 0.0
        pitch_spread = float(np.std(f0_list)) if len(f0_list) > 1 else 0.0
        pitch_slope = float(np.mean(np.diff(f0_list))) if len(f0_list) > 1 else 0.0
    else:
        f0_mean = 0.0
        pitch_spread = 0.0
        pitch_slope = 0.0

    return DAPProsody(
        spectral_centroid=centroid,
        f0_mean=f0_mean,
        pitch_spread=pitch_spread,
        modulation_depth=mod_depth,
        r_env=r_env,
        pitch_slope=pitch_slope,
    )


def evaluate_laughter_reflex(dap: Union[DAPProsody, Dict[str, Any], Any]) -> bool:
    """
    Biological laughter reflex detector.
    Laughter produces rhythmic staccato bursts (3.8–7.0 Hz modulation) accompanied
    by high dynamic contrast, bright acoustic energy, and flat or rising pitch contours.

    Dual-trigger architecture:
    1. Primary DAP trigger (bright staccato bursts):
       R_env(τ) ≥ 0.50 AND mod_depth ≥ 0.70 AND centroid ≥ 1400 Hz.
    2. Voiced vocal laughter trigger (natural vowel bursts "ha-ha", "ho-ho", "he-he"):
       Human vowel formants naturally place centroid in 300–1400 Hz range.
       R_env(τ) ≥ 0.60 AND mod_depth ≥ 0.75 AND centroid ≥ 300 Hz.

    Crying protection:
    Falling pitch contours (pitch_slope < -0.15) reject crying/sobbing from triggering laughter.
    """
    if dap is None:
        return False

    r_env = getattr(dap, "r_env", None)
    if r_env is None and isinstance(dap, dict):
        r_env = dap.get("r_env", 0.0)
    elif r_env is None:
        r_env = 0.0

    mod_depth = getattr(dap, "modulation_depth", None)
    if mod_depth is None and isinstance(dap, dict):
        mod_depth = dap.get("modulation_depth", 0.0)
    elif mod_depth is None:
        mod_depth = 0.0

    centroid = getattr(dap, "spectral_centroid", None)
    if centroid is None and isinstance(dap, dict):
        centroid = dap.get("spectral_centroid", 0.0)
    elif centroid is None:
        centroid = 0.0

    pitch_slope = getattr(dap, "pitch_slope", None)
    if pitch_slope is None and isinstance(dap, dict):
        pitch_slope = dap.get("pitch_slope", 0.0)
    elif pitch_slope is None:
        pitch_slope = 0.0

    # Crying protection: weeping/sobbing has distinctly falling pitch slopes
    if pitch_slope < -0.15:
        return False

    # 1. Primary DAP trigger: bright / breathy staccato bursts
    if r_env >= 0.50 and mod_depth >= 0.70 and centroid >= 1400.0:
        return True

    # 2. Voiced vocal laughter trigger: natural vowel bursts ("ha-ha", "ho-ho", "he-he")
    if r_env >= 0.60 and mod_depth >= 0.75 and centroid >= 300.0:
        return True

    return False


def evaluate_crying_reflex(dap: Union[DAPProsody, Dict[str, Any], Any]) -> bool:
    """
    Biological crying/sobbing/weeping reflex detector.
    Crying produces rhythmic sobbing spasms (3.5–7.0 Hz modulation) with
    characteristic falling pitch contours (pitch_slope < -0.15) or high-pitched
    whimpering distress.
    """
    if dap is None:
        return False

    r_env = getattr(dap, "r_env", None)
    if r_env is None and isinstance(dap, dict):
        r_env = dap.get("r_env", 0.0)
    elif r_env is None:
        r_env = 0.0

    mod_depth = getattr(dap, "modulation_depth", None)
    if mod_depth is None and isinstance(dap, dict):
        mod_depth = dap.get("modulation_depth", 0.0)
    elif mod_depth is None:
        mod_depth = 0.0

    centroid = getattr(dap, "spectral_centroid", None)
    if centroid is None and isinstance(dap, dict):
        centroid = dap.get("spectral_centroid", 0.0)
    elif centroid is None:
        centroid = 0.0

    f0_mean = getattr(dap, "f0_mean", None)
    if f0_mean is None and isinstance(dap, dict):
        f0_mean = dap.get("f0_mean", 0.0)
    elif f0_mean is None:
        f0_mean = 0.0

    pitch_slope = getattr(dap, "pitch_slope", None)
    if pitch_slope is None and isinstance(dap, dict):
        pitch_slope = dap.get("pitch_slope", 0.0)
    elif pitch_slope is None:
        pitch_slope = 0.0

    # 1. Rhythmic sobbing spasms: high modulation + falling pitch slope
    if r_env >= 0.40 and mod_depth >= 0.65 and pitch_slope < -0.15:
        return True

    # 2. High-pitched whimpering / distress weeping (F0 > 270 Hz with downward drift)
    if f0_mean > 270.0 and pitch_slope < -0.20 and centroid < 1000.0:
        return True

    return False



def compute_prosodic_logit_prior(dap: Union[DAPProsody, Dict[str, Any], Any]) -> np.ndarray:
    """
    Computes additive logit calibration priors matching model classes:
    [0: angry, 1: neutral, 2: disgust, 3: fear, 4: happy, 5: sad, 6: surprise]
    """
    prior = np.zeros(7, dtype=np.float32)
    if dap is None:
        return prior

    pitch_spread = getattr(dap, "pitch_spread", None)
    if pitch_spread is None and isinstance(dap, dict):
        pitch_spread = dap.get("pitch_spread", 0.0)
    elif pitch_spread is None:
        pitch_spread = 0.0

    centroid = getattr(dap, "spectral_centroid", None)
    if centroid is None and isinstance(dap, dict):
        centroid = dap.get("spectral_centroid", 0.0)
    elif centroid is None:
        centroid = 0.0

    f0_mean = getattr(dap, "f0_mean", None)
    if f0_mean is None and isinstance(dap, dict):
        f0_mean = dap.get("f0_mean", 0.0)
    elif f0_mean is None:
        f0_mean = 0.0

    r_env = getattr(dap, "r_env", None)
    if r_env is None and isinstance(dap, dict):
        r_env = dap.get("r_env", 0.0)
    elif r_env is None:
        r_env = 0.0

    mod_depth = getattr(dap, "modulation_depth", None)
    if mod_depth is None and isinstance(dap, dict):
        mod_depth = dap.get("modulation_depth", 0.0)
    elif mod_depth is None:
        mod_depth = 0.0

    # Laughter/chuckle cadence: rhythmic modulation
    if r_env >= 0.35 and mod_depth >= 0.40 and centroid >= 350.0:
        prior[4] += 3.0   # happy
        prior[5] -= 2.0   # sad

    # Cheerful voice: broad pitch variations and/or bright acoustic centroid
    if pitch_spread > 35.0 and centroid > 1800.0:
        boost = 3.5 + min(2.0, (pitch_spread - 35.0) / 15.0)
        prior[4] += boost   # happy
        prior[5] -= 2.0     # sad
    elif pitch_spread > 25.0 or (pitch_spread > 20.0 and centroid > 350.0):
        boost = 2.0 + min(2.0, (pitch_spread - 20.0) / 10.0)
        prior[4] += boost   # happy
        prior[5] -= 1.5     # sad
    elif pitch_spread > 30.0 or (pitch_spread > 25.0 and centroid > 1500.0):
        prior[4] += 2.0     # happy
        prior[5] -= 1.0     # sad

    # Somber voice: voiced low-frequency monotone, non-rhythmic (no bursts)
    if (
        60.0 <= f0_mean <= 170.0
        and pitch_spread < 15.0
        and centroid < 1200.0
        and r_env < 0.25
        and mod_depth < 0.40
    ):
        prior[5] += 2.0     # sad
        prior[4] -= 1.5     # happy

    return prior


def calibrate_logits(
    raw_logits: np.ndarray,
    dap: Optional[Union[DAPProsody, Dict[str, Any], Any]] = None,
    centering_factor: float = 0.60,
) -> np.ndarray:
    """
    Calibrates raw Wav2Vec2 logits:
    1. Soft baseline centering: z_cal = z - centering_factor * z0
       Cancels the artificial +8.79 logit unvoiced idle bias.
    2. Additive prosodic prior: z_cal += prior(dap)

    Args:
        raw_logits:       1D numpy array of 7 raw logits from Wav2Vec2 ONNX.
        dap:              Optional DAPProsody object with prosodic acoustic metrics.
        centering_factor: Soft subtraction coefficient (default 0.60).

    Returns:
        Calibrated 1D float32 logit array.
    """
    logits = np.asarray(raw_logits, dtype=np.float32)
    cal_logits = logits - (centering_factor * Z_IDLE_BASELINE)

    if dap is not None:
        prior = compute_prosodic_logit_prior(dap)
        cal_logits = cal_logits + prior

    return cal_logits.astype(np.float32)

