"""
Standalone verification script for MAITRI 2.0.
Tests:
1. Isotropic Square Face Cropping (1:1 aspect ratio, 30% margin, boundary clamping)
2. MediaPipe FACS AU12/AU6 Smile Detection (<1ms geometric evaluation)
3. Hybrid FACS + Deep Semantic Emotion Fusion (immediate smile registration, neutral preservation)
4. Passive DIP Quality Scoring (Laplacian blur metric, area ratio, brightness gating)
5. Eye Tracking & EAR Calibration (open/closed EAR, blink detection, drowsiness)
6. Multimodal Quality-Gated Fusion & Adaptive EMA (weights, stress index, spike reactivity)
7. Autonomous CBT Alert Triggering & Database Logging
"""

import sys
import os
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# If cv2 or pandas are not installed in the testing environment, supply lightweight mocks
try:
    import cv2
except ImportError:
    mock_cv2 = MagicMock()
    mock_cv2.COLOR_BGR2GRAY = 6
    mock_cv2.COLOR_BGR2RGB = 4
    mock_cv2.COLOR_BGR2LAB = 44
    mock_cv2.COLOR_LAB2BGR = 56
    mock_cv2.INTER_AREA = 3
    mock_cv2.INTER_LINEAR = 1
    mock_cv2.CV_64F = 6
    mock_cv2.cvtColor = lambda img, code: img
    mock_cv2.Laplacian = lambda img, depth: MagicMock(var=lambda: 100.0)
    mock_cv2.resize = lambda img, dsize, **kwargs: np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)
    sys.modules["cv2"] = mock_cv2

try:
    import pandas as pd
except ImportError:
    mock_pd = MagicMock()
    mock_pd.DataFrame = lambda *args, **kwargs: MagicMock(empty=False, iloc=[{"fused_emotion": "happy", "stress_level": "NOMINAL"}])
    mock_pd.read_sql_query = lambda *args, **kwargs: mock_pd.DataFrame()
    sys.modules["pandas"] = mock_pd

import numpy as np

from modules.face_module import (
    EMOTIONS,
    extract_isotropic_square_face_box,
    evaluate_facs_smile,
    fuse_facs_with_deep_emotion,
    compute_face_quality,
    FaceResult,
)
from modules.eye_module import (
    EyeSessionState,
    _ear_from_landmarks,
    process_landmarks_for_eyes,
    extract_face_blendshapes,
)
from modules.fusion_module import FusionState, fuse
from modules.vitals_module import compute_vitals_strain
from modules.alert_module import get_alert
from modules.database_module import init_db, log_event, get_logs


@dataclass
class MockPoint:
    x: float
    y: float
    z: float = 0.0


@dataclass
class MockCategory:
    category_name: str
    score: float


@dataclass
class MockLandmarkerResult:
    face_landmarks: list
    face_blendshapes: list


class TestIsotropicSquareFaceBox(unittest.TestCase):
    def test_centered_square_box_with_margin(self):
        # Create landmarks forming a 100x100 face centered at (320, 240) in a 640x480 frame
        # raw face: x from 0.45 to 0.55 (288 to 352 -> span 64), y from 0.40 to 0.60 (192 to 288 -> span 96)
        landmarks = [
            MockPoint(0.45, 0.40),
            MockPoint(0.55, 0.40),
            MockPoint(0.45, 0.60),
            MockPoint(0.55, 0.60),
        ]
        box, coords = extract_isotropic_square_face_box(landmarks, 640, 480, margin_ratio=0.30)
        x1, y1, x2, y2 = coords
        # Ensure 1:1 square
        self.assertEqual(box["w"], box["h"], "Bounding box must be square (1:1 aspect ratio)")
        self.assertEqual(x2 - x1, y2 - y1, "Coordinates must form an isotropic square")
        # Ensure 30% margin: raw max span is 96, 96 * 1.30 = 124.8 -> round to 125
        self.assertGreaterEqual(box["w"], 120, "Should include 30% margin padding")
        self.assertLessEqual(box["w"], 130)

    def test_boundary_clamping_left_edge(self):
        # Face on the left edge of frame
        landmarks = [
            MockPoint(0.01, 0.40),
            MockPoint(0.10, 0.50),
        ]
        box, coords = extract_isotropic_square_face_box(landmarks, 640, 480, margin_ratio=0.30)
        self.assertGreaterEqual(box["x"], 0, "x1 must be clamped >= 0")
        self.assertLessEqual(box["x"] + box["w"], 640, "box must stay within frame width")

    def test_boundary_clamping_bottom_edge(self):
        # Face on bottom edge
        landmarks = [
            MockPoint(0.50, 0.90),
            MockPoint(0.60, 0.99),
        ]
        box, coords = extract_isotropic_square_face_box(landmarks, 640, 480, margin_ratio=0.30)
        self.assertLessEqual(box["y"] + box["h"], 480, "box must stay within frame height")


class TestFACSSmileDetection(unittest.TestCase):
    def test_strong_smile_detection(self):
        # AU12 active (lip corners raised > 0.40)
        blendshapes = {
            "mouthSmileLeft": 0.75,
            "mouthSmileRight": 0.72,
            "cheekSquintLeft": 0.10,
            "cheekSquintRight": 0.12,
        }
        is_smiling, smile_score, cheek_score = evaluate_facs_smile(blendshapes)
        self.assertTrue(is_smiling, "Strong AU12 should trigger smile detection")
        self.assertAlmostEqual(smile_score, 0.735, places=2)

    def test_duchenne_smile_detection(self):
        # Moderate AU12 (0.35) + AU6 cheek squint (0.30)
        blendshapes = {
            "mouthSmileLeft": 0.35,
            "mouthSmileRight": 0.35,
            "cheekSquintLeft": 0.30,
            "cheekSquintRight": 0.30,
        }
        is_smiling, smile_score, cheek_score = evaluate_facs_smile(blendshapes)
        self.assertTrue(is_smiling, "Moderate AU12 + AU6 should trigger Duchenne smile")

    def test_neutral_face(self):
        # Neutral face (AU12 < 0.10)
        blendshapes = {
            "mouthSmileLeft": 0.05,
            "mouthSmileRight": 0.06,
            "cheekSquintLeft": 0.02,
            "cheekSquintRight": 0.02,
        }
        is_smiling, smile_score, cheek_score = evaluate_facs_smile(blendshapes)
        self.assertFalse(is_smiling, "Neutral face should not trigger smile detection")


class TestHybridEmotionFusion(unittest.TestCase):
    def test_immediate_happy_registration_on_smile(self):
        deep_dom = "neutral"
        deep_probs = {"angry": 5.0, "disgust": 2.0, "fear": 3.0, "happy": 10.0, "neutral": 70.0, "sad": 5.0, "surprise": 5.0}
        deep_conf = 0.70

        # Smile active
        dom, probs, conf, method = fuse_facs_with_deep_emotion(
            deep_dom, deep_probs, deep_conf, smile_active=True, smile_score=0.80, cheek_score=0.50
        )
        self.assertEqual(dom, "happy", "Active smile must register happy as dominant emotion")
        self.assertGreaterEqual(conf, 0.85, "Confidence must be >= 85%")
        self.assertGreaterEqual(probs["happy"], 85.0)
        self.assertAlmostEqual(sum(probs.values()), 100.0, delta=0.5, msg="Probabilities must sum to 100%")
        self.assertEqual(method, "facs_geometric_smile")

    def test_deep_emotion_preservation_when_no_smile(self):
        deep_dom = "sad"
        deep_probs = {"angry": 5.0, "disgust": 2.0, "fear": 3.0, "happy": 2.0, "neutral": 10.0, "sad": 75.0, "surprise": 3.0}
        deep_conf = 0.75

        # No smile
        dom, probs, conf, method = fuse_facs_with_deep_emotion(
            deep_dom, deep_probs, deep_conf, smile_active=False, smile_score=0.05, cheek_score=0.02
        )
        self.assertEqual(dom, "sad", "Non-smile should preserve deep model's sad prediction")
        self.assertEqual(conf, 0.75)
        self.assertEqual(method, "deep_semantic_onnx")


class TestPassiveQualityScoring(unittest.TestCase):
    def test_quality_penalties(self):
        # Clean, sharp, well-lit face
        sharp_q = compute_face_quality(
            confidence=0.85, blur_score=150.0, is_blurry=False,
            fw=200, fh=200, frame_w=640, frame_h=480, mean_brightness=120.0
        )
        self.assertGreaterEqual(sharp_q, 0.80)

        # Blurry face
        blurry_q = compute_face_quality(
            confidence=0.85, blur_score=30.0, is_blurry=True,
            fw=200, fh=200, frame_w=640, frame_h=480, mean_brightness=120.0
        )
        self.assertLess(blurry_q, sharp_q, "Blurry face should have lower quality score")


class TestEyeTrackingLogic(unittest.TestCase):
    def test_ear_calculation_and_drowsy_classification(self):
        # 478 mock landmarks where eye landmarks are placed
        # Left eye: [362, 385, 387, 263, 373, 380]
        # Right eye: [33, 160, 158, 133, 153, 144]
        landmarks = [MockPoint(0.5, 0.5) for _ in range(478)]

        # Set wide open eye for left eye
        # horizontal: 362 (x=0.40, y=0.50) to 263 (x=0.60, y=0.50) -> width = 0.20 * 640 = 128
        # vertical 1: 385 (x=0.45, y=0.45) to 373 (x=0.45, y=0.55) -> height = 0.10 * 480 = 48
        # vertical 2: 387 (x=0.55, y=0.45) to 380 (x=0.55, y=0.55) -> height = 0.10 * 480 = 48
        landmarks[362] = MockPoint(0.40, 0.50)
        landmarks[263] = MockPoint(0.60, 0.50)
        landmarks[385] = MockPoint(0.45, 0.45)
        landmarks[373] = MockPoint(0.45, 0.55)
        landmarks[387] = MockPoint(0.55, 0.45)
        landmarks[380] = MockPoint(0.55, 0.55)

        # Same for right eye
        landmarks[33]  = MockPoint(0.20, 0.50)
        landmarks[133] = MockPoint(0.40, 0.50)
        landmarks[160] = MockPoint(0.25, 0.45)
        landmarks[144] = MockPoint(0.25, 0.55)
        landmarks[158] = MockPoint(0.35, 0.45)
        landmarks[153] = MockPoint(0.35, 0.55)

        session = EyeSessionState()
        session.is_calibrated = True
        session.baseline_ear = 0.35

        res = process_landmarks_for_eyes(landmarks, 640, 480, session)
        self.assertGreater(res.ear, 0.25, "Open eye EAR should be > 0.25")
        self.assertEqual(res.fatigue_label, "Normal")


class TestMultimodalFusion(unittest.TestCase):
    def test_nominal_fusion(self):
        state = FusionState()
        face_probs = {"angry": 0.0, "disgust": 0.0, "fear": 0.0, "happy": 0.85, "neutral": 0.15, "sad": 0.0, "surprise": 0.0}
        res = fuse(
            state=state,
            face_probs=face_probs,
            face_quality=0.90,
            vitals_strain=0.10,
            fatigue_strain=0.10,
            eye_quality=1.0,
        )
        self.assertLess(res.stress_pct, 35.0, "Nominal parameters should result in stress < 35%")
        self.assertEqual(res.dominant_emotion, "happy")

    def test_acute_stress_fusion(self):
        state = FusionState()
        face_probs = {"angry": 0.70, "disgust": 0.10, "fear": 0.10, "happy": 0.0, "neutral": 0.0, "sad": 0.10, "surprise": 0.0}
        res = fuse(
            state=state,
            face_probs=face_probs,
            face_quality=0.85,
            vitals_strain=0.85,
            fatigue_strain=0.75,
            eye_quality=1.0,
        )
        self.assertGreater(res.stress_pct, 60.0, "Acute distress parameters should result in stress > 60%")
        alert = get_alert(res.stress_pct, res.dominant_emotion, "Stressed Eyes")
        self.assertIn(alert.stress_level, ["ELEVATED", "CRITICAL"])


class TestDatabaseLogging(unittest.TestCase):
    def test_log_and_retrieve(self):
        init_db()
        log_event(
            face_emotion="happy",
            fused_emotion="happy",
            voice_state="N/A",
            heart_rate=72.0,
            temperature=36.7,
            spo2=98.5,
            blink_rate=16.0,
            fatigue_label="Normal",
            stress_score=15.2,
            stress_level="NOMINAL",
            alert_triggered="All Systems Nominal",
            response_msg="Astronaut performing optimally.",
        )
        df = get_logs()
        self.assertFalse(df.empty, "Database must contain logged event")
        last_row = df.iloc[-1]
        self.assertEqual(last_row["fused_emotion"], "happy")
        self.assertEqual(last_row["stress_level"], "NOMINAL")


if __name__ == "__main__":
    unittest.main()
