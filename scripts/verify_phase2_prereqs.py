#!/usr/bin/env python3
"""
scripts/verify_phase2_prereqs.py
Comprehensive PASS/FAIL check for all Phase 2 prerequisites.

Run from project root:
    /home/mikey/anaconda3/envs/maitri/bin/python scripts/verify_phase2_prereqs.py
"""

import sys
import os
from pathlib import Path

# Ensure project root is on sys.path so module imports work
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MODELS = ROOT / "models"

# ── Helpers ────────────────────────────────────────────────────────────────────
PASS_ICON = "✔"
FAIL_ICON = "✘"
results: list[tuple[str, bool, str]] = []


def check(label: str, passed: bool, detail: str = "") -> bool:
    results.append((label, passed, detail))
    icon = PASS_ICON if passed else FAIL_ICON
    suffix = f"  ({detail})" if detail else ""
    print(f"  {icon}  {label}{suffix}")
    return passed


# ── Section 1: Model files ─────────────────────────────────────────────────────
print("\n── Section 1: Model Files ───────────────────────────────────────────────")

phi3 = MODELS / "phi3_mini_q4.gguf"
check(
    "Phi-3-Mini GGUF exists (>1.5 GB)",
    phi3.exists() and phi3.stat().st_size > 1_500_000_000,
    f"{phi3.stat().st_size / 1e9:.2f} GB" if phi3.exists() else "MISSING",
)

whisper_dir = MODELS / "whisper_tiny_en"
model_bins = list(whisper_dir.rglob("model.bin")) if whisper_dir.exists() else []
whisper_ok = bool(model_bins) and model_bins[0].stat().st_size > 10_000_000
check(
    "Whisper tiny.en model.bin exists (>10 MB)",
    whisper_ok,
    f"{model_bins[0].stat().st_size / 1e6:.1f} MB at {model_bins[0]}" if model_bins else "NOT FOUND",
)

piper_onnx = MODELS / "piper_voice" / "en_US-lessac-medium.onnx"
check(
    "Piper ONNX exists (>60 MB)",
    piper_onnx.exists() and piper_onnx.stat().st_size > 60_000_000,
    f"{piper_onnx.stat().st_size / 1e6:.1f} MB" if piper_onnx.exists() else "MISSING",
)

piper_json = MODELS / "piper_voice" / "en_US-lessac-medium.onnx.json"
check(
    "Piper config JSON exists (>1 KB)",
    piper_json.exists() and piper_json.stat().st_size > 1000,
    f"{piper_json.stat().st_size} B" if piper_json.exists() else "MISSING",
)

# ── Section 2: Package imports ─────────────────────────────────────────────────
print("\n── Section 2: Package Imports ───────────────────────────────────────────")

try:
    import llama_cpp  # noqa: F401
    check("import llama_cpp", True, getattr(llama_cpp, "__version__", "ok"))
except Exception as e:
    check("import llama_cpp", False, str(e))

try:
    import faster_whisper  # noqa: F401
    check("import faster_whisper", True, getattr(faster_whisper, "__version__", "ok"))
except Exception as e:
    check("import faster_whisper", False, str(e))

try:
    import piper  # noqa: F401
    check("import piper", True, "ok")
except Exception as e:
    check("import piper", False, str(e))

# ── Section 3: Module imports ──────────────────────────────────────────────────
print("\n── Section 3: Module Imports ────────────────────────────────────────────")

try:
    from modules.phase2_core import Phase2Config, Phase2Manager
    check("import modules.phase2_core", True, "Phase2Config + Phase2Manager")
except Exception as e:
    check("import modules.phase2_core", False, str(e))
    Phase2Config = None
    Phase2Manager = None

try:
    from modules.asr_tap import ASRBuffer, asr_buffer
    check("import modules.asr_tap", True, "ASRBuffer + asr_buffer singleton")
except Exception as e:
    check("import modules.asr_tap", False, str(e))
    ASRBuffer = None
    asr_buffer = None

# ── Section 4: Whisper smoke test ──────────────────────────────────────────────
print("\n── Section 4: Whisper Transcription Smoke Test ──────────────────────────")

if Phase2Manager is not None and Phase2Config is not None and whisper_ok:
    try:
        import numpy as np
        mgr = Phase2Manager()
        cfg = Phase2Config(llm_enabled=False, tts_enabled=False, asr_enabled=True)
        mgr.load_models(cfg)
        silence = np.zeros(16000, dtype=np.float32)  # 1 s silence
        text = mgr.transcribe(silence)
        check(
            "Phase2Manager.transcribe(silence) returns str",
            isinstance(text, str),
            f"result={text!r}",
        )
    except Exception as e:
        check("Phase2Manager.transcribe(silence)", False, str(e))
else:
    check("Phase2Manager.transcribe(silence)", False, "skipped — module or model unavailable")

# ── Section 5: ASRBuffer ring buffer test ─────────────────────────────────────
print("\n── Section 5: ASRBuffer Ring Buffer Test ────────────────────────────────")

if ASRBuffer is not None:
    try:
        import numpy as np
        buf = ASRBuffer(min_batch_samples=4800)  # 0.3 s

        # Disabled by default — get_batch should return None
        buf.push(np.zeros(5000, dtype=np.float32))
        r = buf.get_batch()
        check("ASRBuffer disabled by default (push is no-op)", r is None, f"got {r!r}")

        # Enable and push enough audio
        buf.set_enabled(True)
        buf.push(np.zeros(4800, dtype=np.float32))
        batch = buf.get_batch()
        check("ASRBuffer returns batch when ≥ min samples", batch is not None and len(batch) >= 4800, f"shape={getattr(batch, 'shape', None)}")

        # After draining, get_batch should return None again
        r2 = buf.get_batch()
        check("ASRBuffer drained after get_batch", r2 is None, "")

        # Disable clears buffer
        buf.push(np.zeros(4800, dtype=np.float32))
        buf.set_enabled(False)
        r3 = buf.get_batch()
        check("ASRBuffer cleared on disable", r3 is None, "")

    except Exception as e:
        check("ASRBuffer ring buffer test", False, str(e))
else:
    check("ASRBuffer ring buffer test", False, "skipped — module not importable")

# ── Section 6: app.py injection checks ────────────────────────────────────────
print("\n── Section 6: app.py ASR Tap Injection ─────────────────────────────────")

app_py = ROOT / "app.py"
app_text = app_py.read_text(encoding="utf-8") if app_py.exists() else ""

check(
    "app.py: 'from modules.asr_tap import asr_buffer' present",
    "from modules.asr_tap import asr_buffer" in app_text,
)
check(
    "app.py: '_ASR_TAP_AVAILABLE' guard present",
    "_ASR_TAP_AVAILABLE" in app_text,
)
check(
    "app.py: '_asr_buffer.push(arr)' call present",
    "_asr_buffer.push(arr)" in app_text,
)
check(
    "app.py: original push_audio_chunk untouched",
    "self.voice_detector.push_audio_chunk(arr)" in app_text,
)

# ── Summary ────────────────────────────────────────────────────────────────────
print("\n─────────────────────────────────────────────────────────────────────────")
total = len(results)
passed = sum(1 for _, ok, _ in results if ok)
failed = total - passed

print(f"\nResult: {passed}/{total} checks passed")
if failed == 0:
    print("\n✔  ALL PASS — Phase 2 prerequisites are complete.\n")
    sys.exit(0)
else:
    print(f"\n✘  {failed} check(s) FAILED — see above.\n")
    sys.exit(1)
