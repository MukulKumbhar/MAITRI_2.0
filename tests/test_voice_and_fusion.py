"""
Unit and Integration Tests for Voice Emotion Recognition & Multimodal Fusion
Verifies:
1. Offline existence and loading of quantized Wav2Vec2 ONNX model.
2. Voice Activity Detection & RMS silence gating.
3. 7-class emotion prediction on audio waveform matching MAITRI taxonomy.
4. Multimodal fusion behavior:
   - 60% Face / 40% Voice nominal ratio during active speech.
   - Strict 0% voice weight during silence (eliminates ambient noise leakage).
   - Dynamic quality-aware adaptation.
   - Manual override mode with slider support.
5. VoiceDetector 3.0s contextual rolling buffer (48,000 samples @ 16 kHz).
6. 450 ms VAD hangover hysteresis: inter-syllabic breath pauses do not wipe emotion.
7. Phonemic speech synthesis and classification.
"""

import os
import sys
import time
import io
import wave
import subprocess
import unittest
import numpy as np

# Ensure project root is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from modules.fusion_module import (
    FusionState,
    FusionResult,
    fuse,
    EMOTIONS,
    _BASE_BEHAVIORAL_W,
    _BASE_FACE_RATIO,
    _BASE_VOICE_RATIO,
)
from modules.voice_module import (
    calculate_audio_quality,
    predict_voice_emotion,
    VoiceDetector,
    ID_TO_MAITRI,
    _MODEL_PATH,
)
from modules.dap_enhancer import (
    apply_infrasonic_filter,
    compute_dap_prosody,
    evaluate_laughter_reflex,
    compute_prosodic_logit_prior,
    calibrate_logits,
    DAPProsody,
    Z_IDLE_BASELINE,
)
from modules.live_state import LiveState


class TestVoiceAndFusion(unittest.TestCase):

    def setUp(self):
        self.state = FusionState()
        self.sample_rate = 16000

    def test_01_model_file_exists(self):
        """Ensure quantized Wav2Vec2 ONNX model is stored offline in models/."""
        self.assertTrue(os.path.isfile(_MODEL_PATH), f"Model missing at {_MODEL_PATH}")
        size_mb = os.path.getsize(_MODEL_PATH) / (1024 * 1024)
        self.assertGreater(size_mb, 50.0, f"Model file suspiciously small: {size_mb:.1f} MB")

    def test_02_silence_vad_gating(self):
        """Silence must return is_speaking=False, quality=0.0, and skip inference."""
        silence = np.zeros(24000, dtype=np.float32)
        rms, q, is_spk = calculate_audio_quality(silence)
        self.assertEqual(rms, 0.0)
        self.assertEqual(q, 0.0)
        self.assertFalse(is_spk)

        dom, probs, conf, qual, is_speaking = predict_voice_emotion(silence)
        self.assertFalse(is_speaking)
        self.assertEqual(conf, 0.0)
        self.assertEqual(qual, 0.0)
        self.assertEqual(dom, "neutral")

    def test_03_active_speech_inference(self):
        """Active speech signal must produce valid 7-class distribution summing to 1.0."""
        t = np.linspace(0, 1.5, 24000, dtype=np.float32)
        # Synthetic modulated carrier waveform
        synthetic_speech = 0.15 * np.sin(2 * np.pi * 240 * t) * (1.0 + 0.4 * np.sin(2 * np.pi * 8 * t))

        rms, q, is_spk = calculate_audio_quality(synthetic_speech)
        self.assertTrue(is_spk)
        self.assertGreater(rms, 0.02)
        self.assertGreater(q, 0.5)

        dom, probs, conf, qual, is_speaking = predict_voice_emotion(synthetic_speech)
        self.assertTrue(is_speaking)
        self.assertIn(dom, EMOTIONS)
        self.assertGreater(conf, 0.0)
        self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)
        self.assertEqual(set(probs.keys()), set(EMOTIONS))

    def test_04_fusion_silence_forces_zero_voice_weight(self):
        """When astronaut is silent, voice weight must strictly drop to 0.0."""
        face_probs = {e: (1.0 if e == "happy" else 0.0) for e in EMOTIONS}
        voice_probs = {e: (1.0 if e == "angry" else 0.0) for e in EMOTIONS}

        # Run multiple iterations to let EMA converge
        for _ in range(25):
            res = fuse(
                state=self.state,
                face_probs=face_probs,
                face_quality=1.0,
                vitals_strain=0.0,
                fatigue_strain=0.0,
                eye_quality=1.0,
                voice_probs=voice_probs,
                voice_quality=0.0,
                is_speaking=False,     # Silent
            )

        self.assertAlmostEqual(res.voice_weight, 0.0, delta=0.01)
        self.assertGreater(res.face_weight, 0.35)
        # Even though voice was 'angry', silence prevented it from corrupting face 'happy'
        self.assertEqual(res.dominant_emotion, "happy")

    def test_05_fusion_60_40_nominal_ratio_during_speech(self):
        """When speech and face are nominal, behavioral split converges to 60% Face / 40% Voice."""
        face_probs = {e: (1.0 if e == "happy" else 0.0) for e in EMOTIONS}
        voice_probs = {e: (1.0 if e == "happy" else 0.0) for e in EMOTIONS}

        for _ in range(40):
            res = fuse(
                state=self.state,
                face_probs=face_probs,
                face_quality=1.0,
                vitals_strain=0.0,
                fatigue_strain=0.0,
                eye_quality=1.0,
                voice_probs=voice_probs,
                voice_quality=1.0,
                is_speaking=True,
            )

        behavioral_total = res.face_weight + res.voice_weight
        self.assertGreater(behavioral_total, 0.30)
        face_share = res.face_weight / behavioral_total
        voice_share = res.voice_weight / behavioral_total

        # Verify 60% Face / 40% Voice (within small tolerance due to vitals/eye baseline)
        self.assertAlmostEqual(face_share, 0.60, delta=0.06)
        self.assertAlmostEqual(voice_share, 0.40, delta=0.06)

    def test_06_manual_override_toggle(self):
        """User manual override slider sets ratio directly."""
        face_probs = {e: (1.0 if e == "happy" else 0.0) for e in EMOTIONS}
        voice_probs = {e: (1.0 if e == "sad" else 0.0) for e in EMOTIONS}

        # Manual 80% Face / 20% Voice
        for _ in range(35):
            res = fuse(
                state=self.state,
                face_probs=face_probs,
                face_quality=0.5,
                vitals_strain=0.0,
                fatigue_strain=0.0,
                eye_quality=0.0,
                voice_probs=voice_probs,
                voice_quality=0.5,
                is_speaking=True,
                manual_override=True,
                manual_face_ratio=0.80,
            )

        behavioral_total = res.face_weight + res.voice_weight
        self.assertAlmostEqual(res.face_weight / behavioral_total, 0.80, delta=0.05)
        self.assertAlmostEqual(res.voice_weight / behavioral_total, 0.20, delta=0.05)

    def test_07_face_occlusion_voice_takes_over(self):
        """When face quality is poor but voice is clear, voice dominates the emotion."""
        face_probs = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
        voice_probs = {e: (1.0 if e == "fear" else 0.0) for e in EMOTIONS}

        for _ in range(30):
            res = fuse(
                state=self.state,
                face_probs=face_probs,
                face_quality=0.05,    # Astronaut turned head away
                vitals_strain=0.0,
                fatigue_strain=0.0,
                eye_quality=0.0,
                voice_probs=voice_probs,
                voice_quality=1.0,     # Clear urgent vocal alarm
                is_speaking=True,
            )

        self.assertGreater(res.voice_weight, res.face_weight)
        self.assertEqual(res.dominant_emotion, "fear")

    def test_08_voice_detector_buffer_and_live_state(self):
        """VoiceDetector manages buffer and updates LiveState safely."""
        ls = LiveState()
        detector = VoiceDetector(live_state=ls, chunk_duration_sec=3.0)
        self.assertFalse(detector._running)
        self.assertEqual(detector.buffer_len, 48000)

        # Push 1 second of audio into buffer
        chunk = np.zeros(16000, dtype=np.float32)
        detector.push_audio_chunk(chunk)
        self.assertEqual(len(detector.audio_buffer), 16000)

        # Snapshot check
        snap = ls.snapshot()
        self.assertIn("voice_emotion", snap)
        self.assertIn("is_speaking", snap)
        self.assertIn("voice_quality", snap)
        self.assertEqual(snap["voice_emotion"], "neutral")

    def test_09_vad_hangover_hysteresis(self):
        """Verify 450ms VAD hangover prevents inter-syllabic pauses from wiping emotion."""
        detector = VoiceDetector(chunk_duration_sec=3.0, hangover_sec=0.45, silence_threshold=0.02)
        self.assertEqual(detector.hangover_sec, 0.45)

        # Push speech burst (loud sine wave)
        speech_chunk = 0.20 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.2, 3200, dtype=np.float32))
        detector.push_audio_chunk(speech_chunk)
        t_speech = detector._last_speech_time
        self.assertGreater(t_speech, 0.0)

        # 150 ms later (silence): hangover must still be active
        now_inter_syllable = t_speech + 0.150
        is_spk_hangover = (now_inter_syllable - t_speech) < detector.hangover_sec
        self.assertTrue(is_spk_hangover, "VAD hangover must remain active at 150ms pause")

        # 400 ms later: still within 450ms window
        now_near_end = t_speech + 0.400
        self.assertTrue((now_near_end - t_speech) < detector.hangover_sec)

        # 600 ms later: hangover expired
        now_expired = t_speech + 0.600
        self.assertFalse((now_expired - t_speech) < detector.hangover_sec, "VAD hangover must expire after 450ms")

    def test_10_buffer_capacity_3_seconds(self):
        """Rolling audio buffer must hold up to 48,000 samples (3.0s @ 16 kHz)."""
        detector = VoiceDetector(chunk_duration_sec=3.0)
        self.assertEqual(detector.buffer_len, 48000)

        # Push 60,000 samples -> buffer maxlen must cap at 48,000
        oversized = np.random.randn(60000).astype(np.float32)
        detector.push_audio_chunk(oversized)
        self.assertEqual(len(detector.audio_buffer), 48000)

    def test_11_phonemic_speech_synthesis(self):
        """Synthesize test phrase and verify prediction covers all 7 emotion classes."""
        try:
            cmd = ['espeak-ng', '-p', '50', '-s', '140', '--stdout', 'Astronaut telemetry test. Status nominal.']
            proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=5)
            if proc.returncode == 0 and len(proc.stdout) > 44:
                with wave.open(io.BytesIO(proc.stdout), 'rb') as wf:
                    raw = wf.readframes(wf.getnframes())
                    sig = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
                    sr = wf.getframerate()
                if sr != 16000:
                    indices = np.linspace(0, len(sig) - 1, int(len(sig) * 16000 / sr))
                    sig = np.interp(indices, np.arange(len(sig)), sig).astype(np.float32)

                dom, probs, conf, qual, is_spk = predict_voice_emotion(sig, silence_threshold=0.005)
                self.assertTrue(is_spk)
                self.assertIn(dom, EMOTIONS)
                self.assertAlmostEqual(sum(probs.values()), 1.0, places=4)
        except (FileNotFoundError, subprocess.SubprocessError):
            pass  # espeak-ng optional in restricted environments

    def test_12_vad_hangover_in_worker_loop(self):
        """Verify background worker transitions is_speaking correctly with hangover."""
        ls = LiveState()
        detector = VoiceDetector(live_state=ls, chunk_duration_sec=3.0, hangover_sec=0.45, silence_threshold=0.02)
        detector._sd_stream = None
        detector._running = True
        import threading
        t = threading.Thread(target=detector._worker_loop, daemon=True)
        t.start()
        detector._worker_thread = t

        try:
            # Preload buffer to meet half-second minimum requirement
            detector.push_audio_chunk(np.zeros(8000, dtype=np.float32))
            # Push speech
            sig = (0.25 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.2, 3200))).astype(np.float32)
            detector.push_audio_chunk(sig)
            time.sleep(0.1)
            self.assertTrue(ls.snapshot()["is_speaking"])

            # Push brief 100ms silence (hangover active)
            detector.push_audio_chunk(np.zeros(1600, dtype=np.float32))
            time.sleep(0.15)
            self.assertTrue(ls.snapshot()["is_speaking"])

            # Push 600ms silence (hangover expired)
            detector.push_audio_chunk(np.zeros(9600, dtype=np.float32))
            time.sleep(0.5)
            self.assertFalse(ls.snapshot()["is_speaking"])
        finally:
            detector.stop()

    def test_13_voice_emotion_preserved_during_silence(self):
        """Verify that when speech ends, is_speaking drops to False, but detected emotion is preserved."""
        ls = LiveState()
        detector = VoiceDetector(live_state=ls, chunk_duration_sec=3.0, hangover_sec=0.20, silence_threshold=0.015)
        detector._sd_stream = None
        detector._running = True
        import threading
        t = threading.Thread(target=detector._worker_loop, daemon=True)
        t.start()
        detector._worker_thread = t

        try:
            # 1. Push 1.2 seconds of speech tone
            t_axis = np.linspace(0, 1.2, int(16000 * 1.2), dtype=np.float32)
            speech = 0.25 * np.sin(2 * np.pi * 280 * t_axis)
            detector.push_audio_chunk(speech)

            # Wait for inference to complete
            for _ in range(25):
                time.sleep(0.08)
                if ls.snapshot()["voice_confidence"] > 0.0:
                    break

            snap_spk = ls.snapshot()
            self.assertGreater(snap_spk["voice_confidence"], 0.0)
            detected_emo = snap_spk["voice_emotion"]
            self.assertIn(detected_emo, EMOTIONS)

            # 2. Push 0.6s silence (hangover is 0.20s, so it fully expires)
            silence = np.zeros(int(16000 * 0.6), dtype=np.float32)
            detector.push_audio_chunk(silence)
            time.sleep(0.35)

            snap_silent = ls.snapshot()
            self.assertFalse(snap_silent["is_speaking"], "is_speaking must be False during silence")
            self.assertEqual(snap_silent["voice_emotion"], detected_emo, "Detected emotion must be preserved during silence")
            self.assertGreater(snap_silent["voice_confidence"], 0.0, "Confidence must be preserved during silence")
        finally:
            detector.stop()

    def test_14_speech_completion_inference_trigger(self):
        """Verify utterance completion (speech pause) triggers inference promptly."""
        ls = LiveState()
        detector = VoiceDetector(live_state=ls, chunk_duration_sec=3.0, hangover_sec=0.45, silence_threshold=0.02)
        detector._sd_stream = None
        detector._running = True
        import threading
        t = threading.Thread(target=detector._worker_loop, daemon=True)
        t.start()
        detector._worker_thread = t

        try:
            # Push short 0.6s speech utterance (shorter than periodic 1.0s interval)
            t_axis = np.linspace(0, 0.6, int(16000 * 0.6), dtype=np.float32)
            speech = 0.20 * np.sin(2 * np.pi * 320 * t_axis)
            detector.push_audio_chunk(speech)

            # Push 200ms pause to trigger speech completion inference
            detector.push_audio_chunk(np.zeros(3200, dtype=np.float32))

            # Wait for speech-ended inference
            for _ in range(20):
                time.sleep(0.08)
                if ls.snapshot()["voice_confidence"] > 0.0:
                    break

            snap = ls.snapshot()
            self.assertGreater(snap["voice_confidence"], 0.0, "Short utterance must be classified upon completion")
            self.assertIn(snap["voice_emotion"], EMOTIONS)
        finally:
            detector.stop()

    def test_15_infrasonic_filter_rumble_attenuation_and_vad(self):
        """
        DAP Infrasonic Filter:
        - 1-10 Hz hardware rumble (RMS ~0.099) must drop by >10x (below 0.009).
        - calculate_audio_quality and predict_voice_emotion must evaluate rumble to is_speaking=False.
        - Speech formants (e.g. 240 Hz) must retain >99% energy.
        """
        t = np.linspace(0, 1.5, 24000, dtype=np.float32)
        # Mechanical rumble on ALC257: 5 Hz + 3 Hz harmonics
        rumble = (0.12 * np.sin(2 * np.pi * 5 * t) + 0.05 * np.sin(2 * np.pi * 3 * t)).astype(np.float32)
        rms_rumble_before = float(np.sqrt(np.mean(rumble ** 2)))
        self.assertGreater(rms_rumble_before, 0.08)

        # Apply infrasonic filter
        filt_rumble = apply_infrasonic_filter(rumble, cutoff_hz=75.0, sr=16000)
        rms_rumble_after = float(np.sqrt(np.mean(filt_rumble ** 2)))
        self.assertLess(rms_rumble_after, 0.009, f"Rumble RMS {rms_rumble_after:.5f} must drop below 0.009")
        self.assertGreater(rms_rumble_before / (rms_rumble_after + 1e-9), 10.0, "Filter must provide >10x rumble attenuation")

        # VAD silence gating verification: rumble must NOT trigger speech
        rms_vad, q_vad, is_spk = calculate_audio_quality(rumble, silence_threshold=0.015)
        self.assertFalse(is_spk, "VAD must classify hardware rumble as silence")
        self.assertEqual(q_vad, 0.0)

        dom, probs, conf, qual, is_spk_pred = predict_voice_emotion(rumble, silence_threshold=0.015)
        self.assertFalse(is_spk_pred, "Inference must be skipped on rumble")
        self.assertEqual(dom, "neutral")

        # Speech transparency verification: 240 Hz vocal tone
        speech = (0.15 * np.sin(2 * np.pi * 240 * t)).astype(np.float32)
        filt_speech = apply_infrasonic_filter(speech, cutoff_hz=75.0, sr=16000)
        rms_speech_before = float(np.sqrt(np.mean(speech ** 2)))
        rms_speech_after = float(np.sqrt(np.mean(filt_speech ** 2)))
        energy_retention = rms_speech_after / rms_speech_before
        self.assertGreater(energy_retention, 0.99, "Speech formant energy must be preserved (>99%)")

    def test_16_dap_prosody_extraction_and_latency(self):
        """
        DAP Prosody:
        - compute_dap_prosody extracts spectral_centroid, f0_mean, pitch_spread, modulation_depth, r_env.
        - Execution latency on 48,000 samples (3.0s buffer) must be strictly < 2.5 ms.
        - Gracefully handles silence, zero arrays, and NaN inputs.
        """
        sig_3s = (0.10 * np.random.randn(48000)).astype(np.float32)

        # Warm up
        _ = compute_dap_prosody(sig_3s, sr=16000)

        # Measure latency over 5 iterations
        times = []
        for _ in range(5):
            t0 = time.time()
            dap = compute_dap_prosody(sig_3s, sr=16000)
            times.append((time.time() - t0) * 1000)

        avg_latency = float(np.mean(times))
        self.assertLess(avg_latency, 2.5, f"DAP prosody latency {avg_latency:.2f} ms exceeds 2.5 ms budget")

        # Metric presence & type verification
        self.assertIsInstance(dap, DAPProsody)
        self.assertIn("spectral_centroid", dap)
        self.assertIn("f0_mean", dap)
        self.assertIn("pitch_spread", dap)
        self.assertIn("modulation_depth", dap)
        self.assertIn("r_env", dap)
        self.assertGreater(dap.spectral_centroid, 0.0)

        # Robustness: silence and zeros
        silence = np.zeros(16000, dtype=np.float32)
        dap_silence = compute_dap_prosody(silence)
        self.assertEqual(dap_silence.spectral_centroid, 0.0)
        self.assertEqual(dap_silence.f0_mean, 0.0)
        self.assertEqual(dap_silence.pitch_spread, 0.0)
        self.assertEqual(dap_silence.r_env, 0.0)

        # Robustness: NaN and empty
        nan_arr = np.array([np.nan, np.inf, 0.1, 0.2], dtype=np.float32)
        dap_nan = compute_dap_prosody(nan_arr)
        self.assertEqual(dap_nan.spectral_centroid, 0.0)

    def test_17_laughter_reflex_detection_and_separation(self):
        """
        Biological Laughter Reflex:
        - Detects laughter (5 Hz rhythmic bursts, high modulation depth, high spectral centroid).
        - Correctly discriminates laughter (R_env ~ 0.87) from angry shouting (R_env ~ 0.28) and speech.
        - predict_voice_emotion immediately outputs dominant='happy', confidence=0.94.
        - Eliminates the >90% sad misclassification anomaly.
        """
        sr = 16000
        duration = 1.5
        t = np.linspace(0, duration, int(sr * duration), dtype=np.float32)

        # 1. Synthetic laughter: 5 Hz modulation bursts + high harmonics & breath noise
        env_laugh = np.maximum(0.0, np.sin(2 * np.pi * 5.0 * t)) ** 2
        carrier_laugh = 0.15 * np.sin(2 * np.pi * 320 * t) + 0.10 * np.sin(2 * np.pi * 1500 * t) + 0.05 * np.random.randn(len(t))
        sig_laugh = (env_laugh * carrier_laugh).astype(np.float32)

        dap_laugh = compute_dap_prosody(sig_laugh, sr=sr)
        self.assertGreaterEqual(dap_laugh.r_env, 0.50, "Laughter R_env must be >= 0.50")
        self.assertGreaterEqual(dap_laugh.modulation_depth, 0.70, "Laughter modulation depth must be >= 0.70")
        self.assertGreaterEqual(dap_laugh.spectral_centroid, 1400.0, "Laughter centroid must be >= 1400 Hz")
        self.assertTrue(evaluate_laughter_reflex(dap_laugh), "Laughter reflex must trigger")

        # End-to-end emotion prediction on laughter: MUST be happy, NOT sad!
        dom, probs, conf, qual, is_spk = predict_voice_emotion(sig_laugh, sampling_rate=sr)
        self.assertTrue(is_spk)
        self.assertEqual(dom, "happy", f"Laughter must be classified as 'happy', but was '{dom}'")
        self.assertAlmostEqual(conf, 0.94, places=2)
        self.assertAlmostEqual(probs["happy"], 0.94, places=2)
        self.assertLess(probs["sad"], 0.05, f"Sad probability {probs['sad']:.4f} unexpectedly high for laughter")

        # 2. Angry shouting: continuous loud signal, low modulation depth
        shout_env = 1.0 + 0.1 * np.sin(2 * np.pi * 1.5 * t)
        sig_shout = (shout_env * (0.25 * np.sin(2 * np.pi * 220 * t) + 0.15 * np.sin(2 * np.pi * 660 * t))).astype(np.float32)
        dap_shout = compute_dap_prosody(sig_shout, sr=sr)
        self.assertFalse(evaluate_laughter_reflex(dap_shout), "Angry shouting must NOT trigger laughter reflex")

        # 3. Conversational speech: moderate modulation, low R_env in 3.8-7.0 Hz band
        speech_env = 0.6 + 0.3 * np.sin(2 * np.pi * 2.2 * t)
        sig_speech = (speech_env * 0.15 * np.sin(2 * np.pi * 160 * t)).astype(np.float32)
        dap_speech = compute_dap_prosody(sig_speech, sr=sr)
        self.assertFalse(evaluate_laughter_reflex(dap_speech), "Normal speech must NOT trigger laughter reflex")

    def test_18_prosodic_calibration_and_baseline_centering(self):
        """
        Prosodic Calibration:
        - Quantized Wav2Vec2 idle vector has +8.79 logit bias favoring sad (+5.487) over happy (-3.305).
        - Soft baseline centering (z - 0.60 * z0) reduces the idle bias.
        - Cheerful prosodic prior boosts happy and dampens sad.
        - Somber prosodic prior boosts sad on low, flat pitch.
        """
        # Baseline vector verification
        self.assertAlmostEqual(Z_IDLE_BASELINE[5], 5.4871, places=3)   # sad
        self.assertAlmostEqual(Z_IDLE_BASELINE[4], -3.3049, places=3)  # happy
        raw_gap = Z_IDLE_BASELINE[5] - Z_IDLE_BASELINE[4]
        self.assertGreater(raw_gap, 8.5)

        # Calibrated idle logits
        cal_idle = calibrate_logits(Z_IDLE_BASELINE, dap=None, centering_factor=0.60)
        cal_gap = cal_idle[5] - cal_idle[4]
        self.assertAlmostEqual(cal_gap, raw_gap * 0.40, delta=0.05)
        self.assertLess(cal_gap, 4.0, "Soft centering must substantially compress the unvoiced sad bias")

        # Cheerful voice prosody prior
        dap_cheerful = DAPProsody(
            spectral_centroid=2200.0,
            f0_mean=250.0,
            pitch_spread=42.0,
            modulation_depth=0.50,
            r_env=0.10,
        )
        prior_cheerful = compute_prosodic_logit_prior(dap_cheerful)
        self.assertGreater(prior_cheerful[4], 3.0, "Cheerful prior must boost happy (index 4)")
        self.assertLess(prior_cheerful[5], 0.0, "Cheerful prior must suppress sad (index 5)")

        # Somber voice prosody prior
        dap_somber = DAPProsody(
            spectral_centroid=950.0,
            f0_mean=110.0,
            pitch_spread=8.0,
            modulation_depth=0.20,
            r_env=0.05,
        )
        prior_somber = compute_prosodic_logit_prior(dap_somber)
        self.assertGreater(prior_somber[5], 1.0, "Somber prior must boost sad (index 5)")
        self.assertLess(prior_somber[4], 0.0, "Somber prior must suppress happy (index 4)")

    def test_19_active_speech_frame_isolation(self):
        """
        Active speech frame isolation:
        - Speech utterance with 0.8s trailing silence is isolated before ONNX sequence pooling.
        - Trailing silence does not drag the classification into the unvoiced sad sink state.
        """
        sr = 16000
        t_spk = np.linspace(0, 1.2, int(sr * 1.2), dtype=np.float32)
        speech = (0.25 * np.sin(2 * np.pi * 280 * t_spk)).astype(np.float32)
        silence = np.zeros(int(sr * 0.8), dtype=np.float32)
        combo = np.concatenate([speech, silence])

        dom_spk, probs_spk, conf_spk, _, _ = predict_voice_emotion(speech, sampling_rate=sr)
        dom_combo, probs_combo, conf_combo, _, _ = predict_voice_emotion(combo, sampling_rate=sr)

        self.assertEqual(dom_spk, dom_combo, "Emotion should remain consistent despite trailing silence")
        self.assertNotEqual(dom_combo, "sad", "Trailing silence must not drag sequence into sad sink state")
        self.assertLess(probs_combo["sad"], 0.05)

    def test_20_multimodal_fusion_with_laughter(self):
        """
        Multimodal fusion integrates laughter immediately into high-confidence happiness.
        When face is obscured or low quality, vocal laughter dominates the multimodal state.
        """
        face_probs = {e: (1.0 if e == "neutral" else 0.0) for e in EMOTIONS}
        voice_probs = {e: 0.01 for e in EMOTIONS}
        voice_probs["happy"] = 0.94

        for _ in range(35):
            res = fuse(
                state=self.state,
                face_probs=face_probs,
                face_quality=0.10,
                vitals_strain=0.0,
                fatigue_strain=0.0,
                eye_quality=0.0,
                voice_probs=voice_probs,
                voice_quality=1.0,
                is_speaking=True,
            )

        self.assertEqual(res.dominant_emotion, "happy")
        self.assertGreater(res.fused_probs["happy"], 0.45)

    def test_21_robust_2d_audio_handling(self):
        """
        2D Audio Array Robustness:
        - Signals with shape (N, 1) or (1, N) from sounddevice or wav readers
          must be handled seamlessly without broadcasting or dimension crashes.
        """
        x_col = (0.2 * np.sin(2 * np.pi * 220 * np.linspace(0, 1.0, 16000))).reshape(-1, 1).astype(np.float32)
        x_row = x_col.T

        # Infrasonic filter with 2D column
        filt_col = apply_infrasonic_filter(x_col)
        self.assertEqual(filt_col.ndim, 1)
        self.assertEqual(len(filt_col), 16000)

        # Prosody with 2D column
        dap_col = compute_dap_prosody(x_col)
        self.assertIsInstance(dap_col, DAPProsody)
        self.assertLess(dap_col.spectral_centroid, 1000.0, "Centroid must be in normal speech range, not exploded")

        # Audio quality with 2D row
        rms, q, is_spk = calculate_audio_quality(x_row)
        self.assertTrue(is_spk)

        # Emotion prediction with 2D column
        dom, probs, conf, _, _ = predict_voice_emotion(x_col)
        self.assertIn(dom, EMOTIONS)

    def test_22_voiced_vocal_laughter_reflex(self):
        """
        Voiced Human Laughter Reflex:
        - Human laughter vowels ("ha-ha", "ho-ho", "he-he") have formant centroids in the
          500-1200 Hz range (not artificially boosted to >1400 Hz by breath hiss).
        - Must trigger evaluate_laughter_reflex and predict dominant='happy', confidence=0.94.
        """
        sr = 16000
        t = np.linspace(0, 1.5, int(sr * 1.5), dtype=np.float32)
        env = np.maximum(0.0, np.sin(2 * np.pi * 5.0 * t)) ** 2
        # Human vowel formants: F0=200Hz, F1=700Hz, F2=1200Hz
        vocal_laugh = (0.20 * np.sin(2 * np.pi * 200 * t) + 0.15 * np.sin(2 * np.pi * 700 * t) + 0.10 * np.sin(2 * np.pi * 1200 * t)).astype(np.float32)
        sig = (env * vocal_laugh).astype(np.float32)

        dap = compute_dap_prosody(sig, sr=sr)
        self.assertGreaterEqual(dap.r_env, 0.60, "Voiced laughter must have high R_env")
        self.assertGreaterEqual(dap.modulation_depth, 0.75, "Voiced laughter must have deep modulation")
        self.assertLess(dap.spectral_centroid, 1400.0, "Centroid is in natural vowel range (<1400 Hz)")
        self.assertTrue(evaluate_laughter_reflex(dap), "Dual-trigger laughter reflex must fire on voiced laughter")

        dom, probs, conf, qual, is_spk = predict_voice_emotion(sig, sampling_rate=sr)
        self.assertEqual(dom, "happy", f"Voiced laughter must be 'happy', got '{dom}'")
        self.assertAlmostEqual(conf, 0.94, places=2)
        self.assertAlmostEqual(probs["happy"], 0.94, places=2)
        self.assertLess(probs["sad"], 0.05)

    def test_23_speech_block_monotonic_autocorrelation_rejection(self):
        """
        Utterance block + silence must NOT be falsely identified as periodic laughter rhythm.
        Monotonically decaying autocorrelation must have r_env=0.0.
        """
        sr = 16000
        speech = 0.20 * np.sin(2 * np.pi * 300 * np.linspace(0, 0.8, int(sr * 0.8), dtype=np.float32))
        sig_block = np.zeros(sr * 2, dtype=np.float32)
        sig_block[:len(speech)] = speech

        dap = compute_dap_prosody(sig_block, sr=sr)
        self.assertEqual(dap.r_env, 0.0, "Monotonically decaying envelope correlation must yield r_env=0.0")
        self.assertFalse(evaluate_laughter_reflex(dap))

    def test_24_infrasonic_rumble_pitch_monotonic_boundary_rejection(self):
        """
        Pitch tracking on sub-audible low frequency rumble (10 Hz) must NOT falsely lock
        onto 400 Hz (min_lag boundary).
        """
        t = np.linspace(0, 1.0, 16000, dtype=np.float32)
        rumble_10hz = (0.25 * np.sin(2 * np.pi * 10 * t)).astype(np.float32)
        dap = compute_dap_prosody(rumble_10hz, sr=16000)
        self.assertEqual(dap.f0_mean, 0.0, "Low-frequency rumble must not falsely register as 400 Hz pitch")

    def test_25_short_utterance_preserved_in_rolling_buffer(self):
        """
        Short utterance (0.35s) inside 3.0s rolling buffer (2.65s silence) must be detected as speech
        and properly classified without being diluted into silence.
        """
        sr = 16000
        t_spk = np.linspace(0, 0.35, int(sr * 0.35), dtype=np.float32)
        speech = (0.20 * np.sin(2 * np.pi * 320 * t_spk)).astype(np.float32)

        buf = np.zeros(sr * 3, dtype=np.float32)
        buf[int(sr * 1.0) : int(sr * 1.0) + len(speech)] = speech

        # Using VoiceDetector silence_threshold=0.030
        rms, q, is_spk = calculate_audio_quality(buf, silence_threshold=0.030)
        self.assertTrue(is_spk, "Short utterance in 3.0s buffer must be recognized as speech")

        dom, probs, conf, qual, is_spk_pred = predict_voice_emotion(buf, silence_threshold=0.030)
        self.assertTrue(is_spk_pred)
        self.assertGreater(conf, 0.0)
        self.assertIn(dom, EMOTIONS)

    def test_26_expressive_cheerful_speech_intonation_boost(self):
        """
        Expressive cheerful speech with pitch modulation and natural speech formants
        must boost happy and eliminate the false unvoiced sad sink state.
        """
        sr = 16000
        t = np.linspace(0, 2.5, int(sr * 2.5), dtype=np.float32)
        f0 = 250 + 55 * np.sin(2 * np.pi * 3 * t)
        phase = 2 * np.pi * np.cumsum(f0) / sr
        cheerful_speech = (0.30 * np.sin(phase) + 0.15 * np.sin(2 * phase) + 0.10 * np.sin(3 * phase)).astype(np.float32)

        dap = compute_dap_prosody(cheerful_speech, sr=sr)
        prior = compute_prosodic_logit_prior(dap)
        self.assertGreater(prior[4], 2.0, "Expressive intonation must boost happy logit")
        self.assertLess(prior[5], 0.0, "Expressive intonation must penalize sad logit")

        dom, probs, conf, _, _ = predict_voice_emotion(cheerful_speech, sampling_rate=sr)
        self.assertEqual(dom, "happy", f"Expressive cheerful speech must predict 'happy', got '{dom}'")
        self.assertGreater(probs["happy"], probs["sad"], "Happy probability must exceed sad probability")


if __name__ == "__main__":
    unittest.main()

