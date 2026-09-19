#!/usr/bin/env python3
"""
MAITRI 2.0 — SER Model Evaluation & Wav2Vec2 Calibration Diagnostic
====================================================================
Run after training to:
1. Test the MFCC-MLP ONNX model on synthetic + synthetic-augmented samples
2. Diagnose Wav2Vec2 sad-bias on various audio types
3. Verify the dual-model ensemble behaves correctly on 7 emotions
4. Print a concise comparison table

Run:
    /home/mikey/anaconda3/envs/maitri/bin/python scripts/evaluate_ser_model.py
"""

import os, sys, time, json
import numpy as np
from pathlib import Path

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
SYNTH_DIR  = BASE_DIR / ".ser_cache" / "synthetic_audio"

sys.path.insert(0, str(BASE_DIR))

import librosa
from modules.voice_module import predict_voice_emotion, EMOTIONS, _MODEL_PATH, _MLP_PATH
from modules.dap_enhancer import (
    apply_infrasonic_filter, compute_dap_prosody,
    evaluate_laughter_reflex, calibrate_logits, Z_IDLE_BASELINE,
)

SR = 16000

print("\n" + "=" * 70, flush=True)
print("MAITRI 2.0 — SER Evaluation Diagnostic", flush=True)
print("=" * 70, flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Check what models are available
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Model Status]", flush=True)
print(f"  Wav2Vec2 ONNX : {'✓' if Path(_MODEL_PATH).exists() else '✗'} {_MODEL_PATH}", flush=True)
print(f"  MFCC-MLP ONNX : {'✓' if Path(_MLP_PATH).exists() else '?'} {_MLP_PATH}", flush=True)

mlp_meta_path = MODELS_DIR / "ser_mlp_meta.json"
if mlp_meta_path.exists():
    meta = json.loads(mlp_meta_path.read_text())
    print(f"  MLP Test F1   : {meta.get('macro_f1_test', '?')}", flush=True)
    print(f"  MLP Classes   : {meta.get('class_names', '?')}", flush=True)
else:
    print("  MLP metadata  : not found (model may not be trained yet)", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Wav2Vec2 idle bias diagnostic
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Wav2Vec2 Idle Bias Diagnostic]", flush=True)
print(f"  Z_IDLE raw sad[5]:   {Z_IDLE_BASELINE[5]:.4f}", flush=True)
print(f"  Z_IDLE raw happy[4]: {Z_IDLE_BASELINE[4]:.4f}", flush=True)
print(f"  Raw sad-happy gap:   {Z_IDLE_BASELINE[5] - Z_IDLE_BASELINE[4]:.4f}", flush=True)

cal_0 = calibrate_logits(Z_IDLE_BASELINE, dap=None, centering_factor=0.80)
print(f"  After 0.80 centering + -1.5 sad penalty:", flush=True)
# Note: runtime applies the extra -1.5 sad penalty after calibrate_logits
cal_0[5] -= 1.5
print(f"    sad[5]:   {cal_0[5]:.4f}", flush=True)
print(f"    happy[4]: {cal_0[4]:.4f}", flush=True)
print(f"    gap:      {cal_0[5] - cal_0[4]:.4f}  (was {Z_IDLE_BASELINE[5] - Z_IDLE_BASELINE[4]:.2f})", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Test on synthetic espeak-ng samples per emotion
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Per-Emotion Ensemble Predictions on Synthetic Speech]", flush=True)
print(f"  {'Emotion':10s}  {'Predicted':10s}  {'Correct':7s}  {'Conf':6s}  {'Quality':7s}", flush=True)
print("  " + "-" * 55, flush=True)

correct_total = 0
total_tested  = 0
emotion_results = {}

for emotion in EMOTIONS:
    emo_dir = SYNTH_DIR / emotion
    if not emo_dir.exists():
        print(f"  {emotion:10s}  [no synthetic data]", flush=True)
        continue

    wav_files = sorted(emo_dir.glob("*.wav"))[:10]  # test up to 10 per emotion
    if not wav_files:
        continue

    correct = 0
    predictions = []
    for wf in wav_files:
        try:
            y, _ = librosa.load(str(wf), sr=SR, mono=True)
        except Exception:
            continue
        dom, probs, conf, qual, is_spk = predict_voice_emotion(y, sampling_rate=SR)
        predictions.append(dom)
        if dom == emotion:
            correct += 1
        total_tested += 1

    correct_total += correct
    emotion_results[emotion] = {
        "tested": len(predictions),
        "correct": correct,
        "accuracy": correct / max(len(predictions), 1),
        "predictions": predictions[:5],
    }
    acc = correct / max(len(predictions), 1) * 100
    pred_sample = ", ".join(predictions[:3])
    mark = "✓" if acc >= 50 else "✗"
    print(f"  {emotion:10s}  {pred_sample:30s}  {mark} {acc:4.0f}%", flush=True)

if total_tested > 0:
    overall_acc = correct_total / total_tested * 100
    print(f"\n  Overall accuracy on synthetic samples: {overall_acc:.1f}%  ({correct_total}/{total_tested})", flush=True)

# ─────────────────────────────────────────────────────────────────────────────
# Edge case tests
# ─────────────────────────────────────────────────────────────────────────────
print("\n[Edge Case Tests]", flush=True)

# 1. Silence → should return neutral with is_speaking=False
silence = np.zeros(48000, dtype=np.float32)
dom, probs, conf, qual, is_spk = predict_voice_emotion(silence, sampling_rate=SR)
mark = "✓" if not is_spk else "✗"
print(f"  {mark} Silence        → dom='{dom}', is_speaking={is_spk} (expected False)", flush=True)

# 2. Laughter → should return happy
t = np.linspace(0, 2.0, SR * 2, dtype=np.float32)
env = np.maximum(0.0, np.sin(2 * np.pi * 5.0 * t)) ** 2
vocal_laugh = (0.30 * np.sin(2*np.pi*200*t) + 0.15 * np.sin(2*np.pi*700*t)).astype(np.float32)
laugh_sig = (env * vocal_laugh).astype(np.float32)
dom, probs, conf, qual, is_spk = predict_voice_emotion(laugh_sig, sampling_rate=SR)
mark = "✓" if dom == "happy" else "✗"
print(f"  {mark} Laughter       → dom='{dom}', conf={conf:.2f} (expected happy)", flush=True)

# 3. High-energy angry speech pattern
angry_t = np.linspace(0, 2.0, SR * 2, dtype=np.float32)
angry_sig = (0.40 * np.sin(2*np.pi*250*angry_t) + 0.20 * np.sin(2*np.pi*500*angry_t) +
             0.10 * np.random.randn(len(angry_t))).astype(np.float32)
dom, probs, conf, qual, is_spk = predict_voice_emotion(angry_sig, sampling_rate=SR)
print(f"  ? High-energy  → dom='{dom}', conf={conf:.2f} (expected angry/disgust/fear)", flush=True)

# 4. Low-energy monotone speech (sad pattern)
sad_t = np.linspace(0, 2.0, SR * 2, dtype=np.float32)
sad_sig = (0.08 * np.sin(2*np.pi*120*sad_t)).astype(np.float32)
dom, probs, conf, qual, is_spk = predict_voice_emotion(sad_sig, sampling_rate=SR)
print(f"  ? Monotone low → dom='{dom}', conf={conf:.2f} (expected sad/neutral)", flush=True)

# 5. Infrastructure noise check: should be gated out
noise = np.random.randn(48000).astype(np.float32) * 0.002
dom, probs, conf, qual, is_spk = predict_voice_emotion(noise, sampling_rate=SR)
mark = "✓" if not is_spk else "✗"
print(f"  {mark} Ambient noise  → dom='{dom}', is_speaking={is_spk} (expected False)", flush=True)

print("\n" + "=" * 70, flush=True)
print("Evaluation complete.", flush=True)
print("=" * 70, flush=True)

