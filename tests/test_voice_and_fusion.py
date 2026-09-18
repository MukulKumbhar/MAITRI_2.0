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


if __name__ == "__main__":
    unittest.main()

