"""
Edge case and stress tests for MAITRI 2.0.
"""

import sys
import os
import unittest
from dataclasses import dataclass
from unittest.mock import MagicMock

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

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
    process_unified_frame,
    analyze_frame,
)
from modules.eye_module import EyeSessionState, extract_face_blendshapes
from modules.fusion_module import FusionState, fuse, _vitals_to_probs, _normalise


@dataclass
class MockPoint:
    x: float
    y: float
    z: float = 0.0


class TestEdgeCases(unittest.TestCase):
    def test_empty_and_single_point_landmarks(self):
        # Single point landmark
        landmarks = [MockPoint(0.5, 0.5)]
        box, coords = extract_isotropic_square_face_box(landmarks, 640, 480, margin_ratio=0.30)
        self.assertGreater(box["w"], 0, "Single point landmark should still produce valid box")
        self.assertEqual(box["w"], box["h"], "Box must remain square")

    def test_extreme_boundary_clamping(self):
        # Landmarks far outside the screen bounds [-10, 10]
        landmarks = [MockPoint(-2.0, -2.0), MockPoint(3.0, 3.0)]
        box, coords = extract_isotropic_square_face_box(landmarks, 640, 480, margin_ratio=0.30)
        self.assertGreaterEqual(box["x"], 0)
        self.assertGreaterEqual(box["y"], 0)
        self.assertLessEqual(box["x"] + box["w"], 640)
        self.assertLessEqual(box["y"] + box["h"], 480)

    def test_facs_smile_empty_and_none(self):
        self.assertEqual(evaluate_facs_smile({}), (False, 0.0, 0.0))
        self.assertEqual(evaluate_facs_smile(None), (False, 0.0, 0.0))

    def test_facs_smile_exact_thresholds(self):
        # Exact AU12 threshold 0.40
        res, smile, cheek = evaluate_facs_smile({"mouthSmileLeft": 0.40, "mouthSmileRight": 0.40})
        self.assertTrue(res)

        # AU12 just below threshold without cheek squint
        res, smile, cheek = evaluate_facs_smile({"mouthSmileLeft": 0.39, "mouthSmileRight": 0.39, "cheekSquintLeft": 0.10, "cheekSquintRight": 0.10})
        self.assertFalse(res)

        # AU12 0.28 with cheek squint 0.20 (exact Duchenne threshold)
        res, smile, cheek = evaluate_facs_smile({"mouthSmileLeft": 0.28, "mouthSmileRight": 0.28, "cheekSquintLeft": 0.20, "cheekSquintRight": 0.20})
        self.assertTrue(res)

    def test_fuse_facs_with_empty_or_zero_probs(self):
        dom, probs, conf, method = fuse_facs_with_deep_emotion(
            deep_dominant="neutral",
            deep_probs={},
            deep_conf=0.0,
            smile_active=True,
            smile_score=0.90,
            cheek_score=0.80,
        )
        self.assertEqual(dom, "happy")
        self.assertAlmostEqual(sum(probs.values()), 100.0, delta=0.5)

    def test_quality_zero_and_extreme_inputs(self):
        q = compute_face_quality(
            confidence=0.0, blur_score=0.0, is_blurry=True,
            fw=0, fh=0, frame_w=640, frame_h=480, mean_brightness=0.0
        )
        self.assertGreaterEqual(q, 0.0)
        self.assertLessEqual(q, 1.0)

    def test_process_unified_frame_none_and_empty(self):
        session = EyeSessionState()
        face, eye = process_unified_frame(None, session)
        self.assertEqual(face.dominant_emotion, "neutral")
        self.assertEqual(eye.ear, 0.30)
        self.assertFalse(eye.available)

        empty_img = np.zeros((0, 0, 3), dtype=np.uint8)
        face, eye = process_unified_frame(empty_img, session)
        self.assertEqual(face.dominant_emotion, "neutral")

    def test_vitals_to_probs_boundaries(self):
        # 0.0 strain
        p_low = _vitals_to_probs(0.0)
        self.assertAlmostEqual(sum(p_low.values()), 1.0, places=4)
        self.assertGreater(p_low["neutral"] + p_low["happy"], 0.70)

        # 1.0 strain
        p_high = _vitals_to_probs(1.0)
        self.assertAlmostEqual(sum(p_high.values()), 1.0, places=4)
        self.assertGreater(p_high["angry"] + p_high["fear"] + p_high["sad"], 0.80)

        # Clamped out-of-range strains
        p_neg = _vitals_to_probs(-0.5)
        self.assertAlmostEqual(sum(p_neg.values()), 1.0, places=4)

        p_huge = _vitals_to_probs(5.0)
        self.assertAlmostEqual(sum(p_huge.values()), 1.0, places=4)


if __name__ == "__main__":
    unittest.main()

