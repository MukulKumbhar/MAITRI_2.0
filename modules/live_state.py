"""
MAITRI 2.0 — Shared Live State
Thread-safe container for passing results from the WebRTC video thread
to the Streamlit main thread.
"""

import threading
from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class LiveState:
    """
    Shared state between the VideoTransformer (WebRTC thread) and
    the Streamlit main thread.  All access must go through the lock.
    """
    lock: threading.Lock = field(default_factory=threading.Lock)

    # Face
    face_emotion:    str                    = "neutral"
    emotion_probs:   Dict[str, float]       = field(default_factory=lambda: {
        "angry": 0.0, "disgust": 0.0, "fear": 0.0,
        "happy": 0.0, "neutral": 100.0, "sad": 0.0, "surprise": 0.0
    })
    face_confidence: float                  = 0.0
    face_quality:    float                  = 0.0
    face_error:      Optional[str]          = None

    # Eye
    ear:             float                  = 0.30
    blink_rate:      float                  = 0.0
    fatigue_label:   str                    = "Normal"
    fatigue_strain:  float                  = 0.0
    eye_quality:     float                  = 0.0
    eye_available:   bool                   = True

    # DIP & Video Quality Telemetry
    blur_score:      float                  = 100.0
    is_blurry:       bool                   = False
    dip_active:      bool                   = True

    # Frame counter (for throttling heavy models)
    frame_count:     int                    = 0

    def snapshot(self) -> dict:
        """Return a plain dict copy — safe to read from main thread."""
        with self.lock:
            return {
                "face_emotion":    self.face_emotion,
                "emotion_probs":   dict(self.emotion_probs),
                "face_confidence": self.face_confidence,
                "face_quality":    self.face_quality,
                "face_error":      self.face_error,
                "ear":             self.ear,
                "blink_rate":      self.blink_rate,
                "fatigue_label":   self.fatigue_label,
                "fatigue_strain":  self.fatigue_strain,
                "eye_quality":     self.eye_quality,
                "eye_available":   self.eye_available,
                "blur_score":      self.blur_score,
                "is_blurry":       self.is_blurry,
                "dip_active":      self.dip_active,
                "frame_count":     self.frame_count,
            }

