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
    face_lock: threading.Lock = field(default_factory=threading.Lock)
    voice_lock: threading.Lock = field(default_factory=threading.Lock)

    def __post_init__(self):
        # Alias lock to face_lock for backward compatibility
        self.lock = self.face_lock

    # Face
    face_emotion:    str                    = "neutral"
    emotion_probs:   Dict[str, float]       = field(default_factory=lambda: {
        "angry": 0.0, "disgust": 0.0, "fear": 0.0,
        "happy": 0.0, "neutral": 1.0, "sad": 0.0, "surprise": 0.0
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

    # Persistent face bounding box (last detected — survives frames between DeepFace runs)
    face_box:        Optional[Dict[str, int]] = None

    # Voice
    voice_emotion:   str                    = "neutral"
    voice_probs:     Dict[str, float]       = field(default_factory=lambda: {
        "angry": 0.0, "disgust": 0.0, "fear": 0.0,
        "happy": 0.0, "neutral": 1.0, "sad": 0.0, "surprise": 0.0
    })
    voice_confidence: float                 = 0.0
    voice_quality:   float                  = 0.0
    is_speaking:     bool                   = False
    voice_available: bool                   = False
    voice_rms:       float                  = 0.0

    # Frame counter (for throttling heavy models)
    frame_count:     int                    = 0

    def snapshot_face(self) -> dict:
        """Return face, eye, and video telemetry under face_lock (<0.05ms)."""
        with self.face_lock:
            return {
                "face_emotion":     self.face_emotion,
                "emotion_probs":    dict(self.emotion_probs),
                "face_confidence":  self.face_confidence,
                "face_quality":     self.face_quality,
                "face_error":       self.face_error,
                "ear":              self.ear,
                "blink_rate":       self.blink_rate,
                "fatigue_label":    self.fatigue_label,
                "fatigue_strain":   self.fatigue_strain,
                "eye_quality":      self.eye_quality,
                "eye_available":    self.eye_available,
                "blur_score":       self.blur_score,
                "is_blurry":        self.is_blurry,
                "dip_active":       self.dip_active,
                "face_box":         dict(self.face_box) if self.face_box else None,
                "frame_count":      self.frame_count,
            }

    def snapshot_voice(self) -> dict:
        """Return voice telemetry under voice_lock (<0.05ms)."""
        with self.voice_lock:
            return {
                "voice_emotion":    self.voice_emotion,
                "voice_probs":      dict(self.voice_probs),
                "voice_confidence": self.voice_confidence,
                "voice_quality":    self.voice_quality,
                "is_speaking":      self.is_speaking,
                "voice_available":  self.voice_available,
                "voice_rms":        self.voice_rms,
            }

    def snapshot(self) -> dict:
        """Return a plain dict copy of all telemetry — safe to read from main thread."""
        snap = self.snapshot_face()
        snap.update(self.snapshot_voice())
        return snap

