#!/usr/bin/env python3
"""
scripts/download_whisper.py
Download and cache the faster-whisper tiny.en model into models/whisper_tiny_en/.

Run from project root:
    /home/mikey/anaconda3/envs/maitri/bin/python scripts/download_whisper.py
"""

import sys
from pathlib import Path

# ── Resolve project root ─────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
WHISPER_DIR = ROOT / "models" / "whisper_tiny_en"
WHISPER_DIR.mkdir(parents=True, exist_ok=True)

print(f"Download target: {WHISPER_DIR}")

# ── Download model ────────────────────────────────────────────────────────────
try:
    from faster_whisper import WhisperModel
except ImportError:
    print("ERROR: faster-whisper not installed. Run:")
    print("  pip install faster-whisper")
    sys.exit(1)

print("Downloading Systran/faster-whisper-tiny.en …")
model = WhisperModel(
    "tiny.en",
    device="cpu",
    compute_type="int8",
    download_root=str(WHISPER_DIR),
)
print("Download complete.")

# ── Smoke test: transcribe a silent buffer ────────────────────────────────────
import numpy as np  # noqa: E402

print("Smoke-testing transcription on 1 s of silence …")
silence = np.zeros(16000, dtype=np.float32)
segments, info = model.transcribe(silence, language="en")
text = " ".join(seg.text.strip() for seg in segments).strip()
print(f"  Transcription result: {text!r}  (expected: empty or silence marker)")
print("Smoke test PASSED.")

# ── Verify files ──────────────────────────────────────────────────────────────
model_bins = list(WHISPER_DIR.rglob("model.bin"))
if model_bins:
    mb = model_bins[0].stat().st_size / 1024 / 1024
    print(f"model.bin found at {model_bins[0]} ({mb:.1f} MB)")
else:
    print("WARNING: model.bin not found under", WHISPER_DIR)
    sys.exit(1)

print("Done.")
