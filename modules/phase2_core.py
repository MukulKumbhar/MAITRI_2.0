"""
modules/phase2_core.py
Phase 2 conversational AI backend for MAITRI 2.0.

Completely self-contained: safe to import even when Phase 2 models are absent.
Phase 1 is never impacted — models are only loaded on explicit call to load_models().
"""

from __future__ import annotations

import io
import logging
import os
import threading
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

log = logging.getLogger(__name__)

# ── Default model paths relative to project root ─────────────────────────────
_HERE = Path(__file__).resolve().parent.parent  # project root
_MODELS = _HERE / "models"


@dataclass
class Phase2Config:
    """Paths to Phase 2 model files and feature flags."""
    llm_model_path: str = str(_MODELS / "phi3_mini_q4.gguf")
    whisper_model_dir: str = str(_MODELS / "whisper_tiny_en")
    piper_onnx_path: str = str(_MODELS / "piper_voice" / "en_US-lessac-medium.onnx")
    piper_json_path: str = str(_MODELS / "piper_voice" / "en_US-lessac-medium.onnx.json")

    llm_enabled: bool = True
    tts_enabled: bool = True
    asr_enabled: bool = True

    # LLM generation params
    llm_n_ctx: int = 2048
    llm_max_tokens: int = 256
    llm_temperature: float = 0.7

    # Extra fields for forward-compat
    extra: dict = field(default_factory=dict)


def _resolve_whisper_model_path(whisper_model_dir: str) -> str:
    """
    faster-whisper stores models inside an HF cache sub-structure:
      <download_root>/models--Systran--faster-whisper-tiny.en/snapshots/<hash>/
    Walk the tree to find model.bin and return that directory.
    Falls back to the dir itself if model.bin lives there directly.
    """
    base = Path(whisper_model_dir)
    # Direct layout
    if (base / "model.bin").exists():
        return str(base)
    # HF cache layout: walk up to 4 levels
    for model_bin in base.rglob("model.bin"):
        return str(model_bin.parent)
    # If nothing found return original dir (WhisperModel will raise a clear error)
    return str(base)


class Phase2Manager:
    """
    Lazy-loading manager for Phase 2 AI models (LLM, ASR, TTS).

    Usage:
        cfg = Phase2Config()
        mgr = Phase2Manager()
        mgr.load_models(cfg)           # blocks until all models ready
        text  = mgr.transcribe(audio)
        reply = mgr.generate_response(text, emotion="neutral", stress_pct=30.0)
        wav   = mgr.speak(reply)
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._llm = None
        self._whisper = None
        self._piper = None
        self._piper_config = None
        self._loaded = False
        self._config: Optional[Phase2Config] = None

    # ── Public API ────────────────────────────────────────────────────────────

    def load_models(self, config: Phase2Config) -> None:
        """
        Lazy-load LLM, Whisper and Piper models.
        Safe to call multiple times — subsequent calls are no-ops.
        Runs synchronously in the calling thread (call from a background thread
        if you want non-blocking startup).
        """
        with self._lock:
            if self._loaded:
                return
            self._config = config
            errors = []

            # ── 1. faster-whisper ─────────────────────────────────────────────
            if config.asr_enabled:
                try:
                    from faster_whisper import WhisperModel  # noqa: PLC0415
                    model_path = _resolve_whisper_model_path(config.whisper_model_dir)
                    self._whisper = WhisperModel(
                        model_path,
                        device="cpu",
                        compute_type="int8",
                    )
                    log.info("Phase2: Whisper tiny.en loaded from %s", model_path)
                except Exception as exc:
                    log.error("Phase2: Whisper load failed: %s", exc)
                    errors.append(f"Whisper: {exc}")

            # ── 2. llama-cpp-python (Phi-3-Mini) ──────────────────────────────
            if config.llm_enabled:
                try:
                    from llama_cpp import Llama  # noqa: PLC0415
                    self._llm = Llama(
                        model_path=config.llm_model_path,
                        n_ctx=config.llm_n_ctx,
                        n_threads=max(1, (os.cpu_count() or 4) - 1),
                        verbose=False,
                    )
                    log.info("Phase2: Phi-3-Mini loaded from %s", config.llm_model_path)
                except Exception as exc:
                    log.error("Phase2: LLM load failed: %s", exc)
                    errors.append(f"LLM: {exc}")

            # ── 3. Piper TTS ──────────────────────────────────────────────────
            if config.tts_enabled:
                try:
                    from piper import PiperVoice  # noqa: PLC0415
                    self._piper = PiperVoice.load(
                        model_path=config.piper_onnx_path,
                        config_path=config.piper_json_path,
                        use_cuda=False,
                    )
                    log.info("Phase2: Piper TTS loaded from %s", config.piper_onnx_path)
                except Exception as exc:
                    log.error("Phase2: Piper TTS load failed: %s", exc)
                    errors.append(f"Piper: {exc}")

            self._loaded = True
            if errors:
                log.warning("Phase2: loaded with errors: %s", "; ".join(errors))

    @property
    def is_loaded(self) -> bool:
        """True once load_models() has been called (even if some models failed)."""
        return self._loaded

    def transcribe(self, audio_np_array) -> str:
        """
        Run Whisper tiny.en on a float32 16 kHz mono numpy array.
        Returns the transcribed text string, or '' on failure / model not loaded.

        Args:
            audio_np_array: np.ndarray, dtype float32, shape (N,), 16 kHz mono.
        """
        import numpy as np  # noqa: PLC0415

        if self._whisper is None:
            log.debug("Phase2.transcribe: Whisper not loaded — returning ''")
            return ""

        try:
            audio = np.asarray(audio_np_array, dtype=np.float32)
            segments, _ = self._whisper.transcribe(
                audio,
                language="en",
                beam_size=5,
                vad_filter=False,
            )
            return " ".join(seg.text.strip() for seg in segments).strip()
        except Exception as exc:
            log.error("Phase2.transcribe error: %s", exc)
            return ""

    def generate_response(
        self,
        user_text: str,
        emotion: str = "neutral",
        stress_pct: float = 0.0,
    ) -> str:
        """
        Build a clinical system prompt and run Phi-3-Mini to produce a reply.

        Args:
            user_text:  What the astronaut said.
            emotion:    Current dominant emotion label (e.g. 'sad', 'anxious').
            stress_pct: Current stress percentage [0–100].

        Returns:
            Model reply string, or '' on failure / model not loaded.
        """
        if self._llm is None:
            log.debug("Phase2.generate_response: LLM not loaded — returning ''")
            return ""

        system_prompt = (
            "You are MAITRI, an AI psychological support companion for astronauts "
            "on long-duration space missions. Respond concisely (2–4 sentences), "
            "with empathy, calm, and clinical professionalism. Never alarm the crew. "
            f"Current detected emotion: {emotion}. "
            f"Current stress level: {stress_pct:.0f}%."
        )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_text},
        ]

        try:
            with self._lock:
                response = self._llm.create_chat_completion(
                    messages=messages,
                    max_tokens=self._config.llm_max_tokens if self._config else 256,
                    temperature=self._config.llm_temperature if self._config else 0.7,
                    stop=["<|end|>", "<|endoftext|>"],
                )
            return response["choices"][0]["message"]["content"].strip()
        except Exception as exc:
            log.error("Phase2.generate_response error: %s", exc)
            return ""

    def speak(self, text: str) -> bytes:
        """
        Synthesise text with Piper TTS and return raw WAV bytes.

        Returns:
            WAV bytes (16-bit PCM, mono, 22050 Hz for lessac-medium), or b'' on failure.
        """
        if self._piper is None:
            log.debug("Phase2.speak: Piper not loaded — returning b''")
            return b""

        try:
            buf = io.BytesIO()
            with wave.open(buf, "wb") as wav_file:
                self._piper.synthesize_wav(text, wav_file)
            return buf.getvalue()
        except Exception as exc:
            log.error("Phase2.speak error: %s", exc)
            return b""
