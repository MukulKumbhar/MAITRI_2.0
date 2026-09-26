"""
modules/asr_tap.py
Thread-safe ring buffer for ASR audio capture from the WebRTC pipeline.

Design:
- Module-level singleton `asr_buffer` is disabled by default → zero Phase 1 impact.
- ASRBuffer is push-only from the WebRTC thread and pop-only from the ASR thread.
- No dependencies on Phase 2 models — safe to import at Streamlit startup.
"""

from __future__ import annotations

import threading
from collections import deque
from typing import Optional

import numpy as np


# 3 seconds of audio at 16 kHz mono
_MIN_BATCH_SAMPLES: int = 48_000
_RING_MAX_SAMPLES: int = 160_000  # ~10 s — cap memory usage


class ASRBuffer:
    """
    Thread-safe ring buffer for audio chunks coming from WebRTC.

    Push audio from the WebRTC recv() thread.
    Call get_batch() from the ASR inference thread to drain ≥3 s of audio.
    """

    def __init__(
        self,
        min_batch_samples: int = _MIN_BATCH_SAMPLES,
        max_samples: int = _RING_MAX_SAMPLES,
    ) -> None:
        self._lock = threading.Lock()
        self._chunks: deque[np.ndarray] = deque()
        self._total_samples: int = 0
        self._min_batch_samples = min_batch_samples
        self._max_samples = max_samples
        self._enabled: bool = False

    # ── Control ───────────────────────────────────────────────────────────────

    @property
    def enabled(self) -> bool:
        return self._enabled

    def set_enabled(self, value: bool) -> None:
        """Enable or disable audio collection. Disabling also clears the buffer."""
        with self._lock:
            self._enabled = value
            if not value:
                self._chunks.clear()
                self._total_samples = 0

    # ── Producer API (WebRTC recv thread) ─────────────────────────────────────

    def push(self, audio: np.ndarray) -> None:
        """
        Push a float32 mono audio chunk.
        No-op if disabled. Silently drops oldest chunks when buffer is full.

        Args:
            audio: 1-D float32 numpy array at 16 kHz.
        """
        if not self._enabled:
            return
        chunk = np.asarray(audio, dtype=np.float32).ravel()
        if chunk.size == 0:
            return
        with self._lock:
            self._chunks.append(chunk)
            self._total_samples += chunk.size
            # Evict oldest chunks if we exceed the ring capacity
            while self._total_samples > self._max_samples and self._chunks:
                evicted = self._chunks.popleft()
                self._total_samples -= evicted.size

    # ── Consumer API (ASR inference thread) ───────────────────────────────────

    def get_batch(self) -> Optional[np.ndarray]:
        """
        Return a concatenated float32 array of all accumulated audio if
        ≥ min_batch_samples have been collected, then clear the buffer.
        Returns None if not enough audio is available yet.
        """
        with self._lock:
            if self._total_samples < self._min_batch_samples:
                return None
            batch = np.concatenate(list(self._chunks))
            self._chunks.clear()
            self._total_samples = 0
        return batch

    @property
    def pending_samples(self) -> int:
        """Number of samples currently queued (thread-safe approximate read)."""
        with self._lock:
            return self._total_samples


# ── Module-level singleton (disabled by default) ──────────────────────────────
asr_buffer: ASRBuffer = ASRBuffer()
