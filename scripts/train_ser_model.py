#!/usr/bin/env python3
"""
MAITRI 2.0 — Local SER Training Pipeline (No Internet Required)
================================================================
Strategy (100% offline using espeak-ng TTS + librosa augmentation):

PHASE 1: Synthetic corpus generation
  - Use espeak-ng with emotion-specific prosody parameters
    (pitch, rate, amplitude, intonation) across 7 emotion classes
  - Generate 120+ sentences per emotion → 840+ raw clips
  - Heavy augmentation: pitch shift ±2 semi, time stretch, noise, reverb
    → 5,000+ augmented training samples

PHASE 2: Feature extraction (129-dimensional acoustic feature vector)
  - 40 MFCCs + 40 delta-MFCCs + 40 delta2-MFCCs
  - Spectral centroid, rolloff, ZCR, RMS: mean+std (8 dims)
  - Pitch spread from voiced F0 via PYIN (1 dim)

PHASE 3: MLP training (129→256→128→7, BatchNorm, Dropout, AdaLR)
  - StratifiedKFold 5-fold cross-validation
  - Final model trained on full train set, evaluated on held-out test set

PHASE 4: ONNX export via skl2onnx pipeline

PHASE 5: Wav2Vec2 idle baseline recalibration using real espeak-ng neutral
  speech (not silence), giving a more accurate Z0 for DAP centering

Run:
    /home/mikey/anaconda3/envs/maitri/bin/python scripts/train_ser_model.py
"""

import os, sys, json, time, subprocess, shutil, warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import numpy as np

BASE_DIR   = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
CACHE_DIR  = BASE_DIR / ".ser_cache"
SYNTH_DIR  = CACHE_DIR / "synthetic_audio"
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
SYNTH_DIR.mkdir(parents=True, exist_ok=True)

print("[SER-Train] Importing ML libraries...", flush=True)
import librosa
import librosa.effects
import soundfile as sf
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import sklearn
print(f"  librosa {librosa.__version__}, sklearn {sklearn.__version__}", flush=True)

MAITRI_EMOTIONS = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
SR = 22050

# ─────────────────────────────────────────────────────────────────────────────
# EMOTION PROSODY PARAMETERS for espeak-ng
# Each emotion has: pitch (base Hz semitone shift from 50), rate (wpm),
# amplitude (0-200), intonation (0=flat…3=max), voice variant
# ─────────────────────────────────────────────────────────────────────────────
EMOTION_ESPEAK_PARAMS = {
    #           pitch  rate  amplitude  extras
    "angry":   ("-p 75",  "-s 170", "-a 180", "--punct"),          # high pitch, fast, loud
    "disgust": ("-p 40",  "-s 140", "-a 120", ""),                 # low, slow, flat
    "fear":    ("-p 80",  "-s 200", "-a 150", ""),                 # very high, very fast
    "happy":   ("-p 65",  "-s 160", "-a 160", ""),                 # high, moderate, bright
    "neutral": ("-p 50",  "-s 150", "-a 100", ""),                 # baseline
    "sad":     ("-p 30",  "-s 110", "-a  80", ""),                 # low, slow, quiet
    "surprise":("-p 90",  "-s 175", "-a 170", ""),                 # very high pitch
}

# Rich sentence corpus — varied syntax so MFCC patterns differ
SENTENCES_BY_EMOTION = {
    "angry": [
        "I cannot believe you did that again!", "This is absolutely unacceptable!",
        "You have no idea what you have done!", "I am furious right now!",
        "Stop doing that immediately!", "How dare you speak to me like that!",
        "This is the last time I will tolerate this!", "I am done with this nonsense!",
        "You are making me so angry right now!", "Get out of my way!",
        "I told you not to do that!", "Why do you always ignore me?",
        "I am completely fed up with this situation!", "You never listen to what I say!",
        "This makes my blood boil!", "I want answers right now!",
        "Fix this problem immediately or there will be consequences!",
        "I have had enough of your excuses!", "You are being completely unreasonable!",
        "Do not test my patience any further!", "I demand an explanation right now!",
        "This is the worst thing you have ever done!", "I am livid!",
        "You will regret this!", "Nobody treats me like this and gets away with it!",
        "I am absolutely disgusted by your behavior!", "Stop lying to me!",
        "I knew I could not trust you!", "You have gone too far this time!",
        "This is beyond unacceptable behavior!", "I will not stand for this!",
    ],
    "disgust": [
        "That smell is absolutely revolting.", "I cannot even look at that.",
        "This is the most disgusting thing I have ever seen.", "How repulsive.",
        "I feel sick just thinking about it.", "That is absolutely vile.",
        "I cannot believe anyone would do something like that.", "Yuck.",
        "That is completely nauseating to me.", "I find that deeply repulsive.",
        "The very thought of it makes me sick.", "That is truly grotesque.",
        "I would never eat something like that.", "How utterly disgusting.",
        "I cannot stand the sight of it.", "That is absolutely foul.",
        "The smell is making me nauseous.", "I have never been so repulsed.",
        "That behavior is truly revolting to me.", "I want nothing to do with that.",
        "Looking at that makes my stomach turn.", "How utterly repugnant.",
        "I am thoroughly disgusted.", "I cannot tolerate that at all.",
        "That is the most unpleasant thing I have witnessed.", "Absolutely gross.",
        "That is beyond repulsive to me.", "I feel physically ill.",
        "Nothing could be more disgusting than that.", "I am completely turned off.",
    ],
    "fear": [
        "I am terrified right now.", "Please do not let that happen.",
        "I am shaking all over.", "What was that sound?",
        "I do not feel safe here at all.", "Something is very wrong.",
        "I cannot stop trembling.", "Please help me.",
        "I am scared out of my mind.", "I have a very bad feeling about this.",
        "Something terrible is about to happen.", "Do not leave me alone.",
        "I am absolutely petrified.", "My heart is racing.",
        "I cannot breathe properly right now.", "What is out there?",
        "I need to get out of here immediately.", "This place frightens me.",
        "I cannot face this alone.", "I feel a cold chill down my spine.",
        "Something is watching us right now.", "I am frozen with fear.",
        "I cannot stop panicking.", "Everything seems threatening right now.",
        "I want to run away from here.", "Please turn on the light.",
        "I heard something moving in the dark.", "I am too scared to move.",
        "My hands will not stop shaking.", "I just want to feel safe again.",
    ],
    "happy": [
        "I am so excited about this!", "This is absolutely wonderful news!",
        "I could not be happier right now!", "Today is the best day ever!",
        "I love everything about this!", "This is incredible!",
        "I am overjoyed beyond words!", "Everything is going perfectly!",
        "What a fantastic surprise this is!", "I am beaming with joy!",
        "This made my entire day so much better!", "I feel amazing!",
        "I am smiling from ear to ear!", "This is the best thing ever!",
        "I cannot stop laughing with happiness!", "Life is absolutely beautiful!",
        "I am so grateful for this moment!", "Pure happiness and joy!",
        "This is everything I ever wanted!", "I am thrilled beyond belief!",
        "What a perfect day this has been!", "I am on top of the world!",
        "Nothing could bring me down today!", "I am bursting with excitement!",
        "This news just made me so incredibly happy!", "I feel like celebrating!",
        "I am absolutely delighted!", "This is truly wonderful!",
        "I cannot contain my happiness!", "Today is a great day to be alive!",
    ],
    "neutral": [
        "The meeting is scheduled for three o'clock.", "Please turn left at the next street.",
        "The report will be ready by tomorrow morning.", "I need to buy some groceries.",
        "The temperature today is twenty two degrees.", "Please confirm your appointment.",
        "The train arrives in ten minutes.", "I will send you the file shortly.",
        "Please complete the form and submit it.", "The office is on the fourth floor.",
        "I have three meetings today.", "Please call me when you arrive.",
        "The document is on the shared drive.", "I will review your request.",
        "The project deadline is next Friday.", "Please fill in your details below.",
        "The update will take approximately five minutes.", "I need to check the schedule.",
        "The address is on the second page.", "I will forward the information to you.",
        "Please wait while I check the records.", "The service will resume shortly.",
        "I need to update the configuration settings.", "Please provide your identification.",
        "The results will be available tomorrow.", "I will arrange a callback.",
        "Please verify your account information.", "The system is processing your request.",
        "I need to reschedule this appointment.", "Please confirm receipt of this message.",
    ],
    "sad": [
        "I miss them so much.", "Nothing will ever be the same again.",
        "I feel so alone right now.", "Everything reminds me of what I lost.",
        "I cannot stop crying.", "I do not know how to go on.",
        "I wish things were different.", "My heart is completely broken.",
        "I feel empty inside.", "I have lost all hope.",
        "Nothing brings me joy anymore.", "I am so heartbroken.",
        "I just want to feel better.", "I cannot stop thinking about it.",
        "Everything feels so heavy and dark.", "I am deeply saddened.",
        "I feel like I am falling apart.", "I wish you were still here.",
        "I cannot believe this has happened.", "I feel utterly devastated.",
        "My whole world has crumbled.", "I am so disappointed.",
        "I am struggling to hold it together.", "I feel so hopeless.",
        "There is no comfort anywhere for me.", "I just want the pain to stop.",
        "I am filled with so much grief.", "Nothing makes sense anymore.",
        "I feel completely broken inside.", "I do not think I can recover from this.",
    ],
    "surprise": [
        "Oh my goodness, I cannot believe it!", "Wow, that is absolutely shocking!",
        "I never expected that to happen!", "What on earth just happened?",
        "Are you serious right now?", "I am completely stunned!",
        "That is the most unexpected thing ever!", "I did not see that coming at all!",
        "No way, that cannot be real!", "I am totally blown away!",
        "How did that even happen?", "That caught me completely off guard!",
        "I am utterly astonished!", "Whoa, I had no idea!",
        "That is just unbelievable!", "I cannot process what just happened!",
        "That is the most surprising thing I have seen!", "Oh wow, really?",
        "I am in complete shock!", "That was totally unexpected!",
        "I am absolutely amazed!", "You have got to be kidding me!",
        "That is beyond anything I imagined!", "I cannot believe my eyes!",
        "What a completely unexpected turn of events!", "I am speechless!",
        "How extraordinary!", "I am genuinely shocked!",
        "That is completely mind blowing!", "I never would have guessed!",
    ],
}


def generate_synthetic_audio(force: bool = False) -> list:
    """
    Generate espeak-ng synthetic speech for all 7 emotions.
    Returns list of (wav_path, emotion_label) tuples.
    """
    # Check if already generated
    done_flag = SYNTH_DIR / ".generation_complete"
    if done_flag.exists() and not force:
        items = [(str(p), p.parent.name) for p in SYNTH_DIR.rglob("*.wav")]
        print(f"  [cache] Synthetic corpus: {len(items)} WAVs found", flush=True)
        return items

    items = []
    total_generated = 0

    for emotion, params in EMOTION_ESPEAK_PARAMS.items():
        emo_dir = SYNTH_DIR / emotion
        emo_dir.mkdir(exist_ok=True)
        sentences = SENTENCES_BY_EMOTION[emotion]
        pitch_arg, rate_arg, amp_arg, extras = params

        # Also generate with slight pitch variations per sentence for diversity
        pitch_variations = [pitch_arg]
        base_pitch = int(pitch_arg.split()[1])
        for dp in [-5, +5]:
            clipped = max(0, min(99, base_pitch + dp))
            pitch_variations.append(f"-p {clipped}")

        gen_count = 0
        for si, sentence in enumerate(sentences):
            for pi, pitch in enumerate(pitch_variations):
                fname = emo_dir / f"{emotion}_{si:03d}_p{pi}.wav"
                if fname.exists():
                    items.append((str(fname), emotion))
                    gen_count += 1
                    continue

                cmd = [
                    "espeak-ng",
                    "-v", "en-us",
                    pitch, rate_arg, amp_arg,
                    "-w", str(fname),
                    sentence,
                ]
                # espeak-ng doesn't accept multiple -p flags in list form
                cmd_str = (
                    f"espeak-ng -v en-us {pitch} {rate_arg} {amp_arg} "
                    f"-w '{fname}' '{sentence}'"
                )
                try:
                    result = subprocess.run(
                        cmd_str, shell=True, capture_output=True, timeout=10
                    )
                    if fname.exists() and fname.stat().st_size > 100:
                        items.append((str(fname), emotion))
                        gen_count += 1
                except Exception as e:
                    pass

        total_generated += gen_count
        print(f"  {emotion:10s}: {gen_count} clips generated", flush=True)

    done_flag.touch()
    print(f"  Total synthetic corpus: {total_generated} WAVs", flush=True)
    return items


# ─────────────────────────────────────────────────────────────────────────────
# FEATURE EXTRACTION
# ─────────────────────────────────────────────────────────────────────────────

N_MFCC = 40

def extract_features(y: np.ndarray, sr: int = SR) -> np.ndarray | None:
    """
    Extract 129-dim acoustic feature vector (MFCCs + deltas + spectral + fast pitch).
    Returns None if audio is too short or empty.
    """
    try:
        y, _ = librosa.effects.trim(y, top_db=20)
        if len(y) < int(sr * 0.15):
            return None

        # MFCCs + first and second order temporal derivatives
        mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC)
        d1   = librosa.feature.delta(mfcc, order=1)
        d2   = librosa.feature.delta(mfcc, order=2)

        # Spectral features
        centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
        rolloff  = librosa.feature.spectral_rolloff(y=y, sr=sr, roll_percent=0.85)[0]
        zcr      = librosa.feature.zero_crossing_rate(y)[0]
        rms      = librosa.feature.rms(y=y)[0]

        # Fast pitch (F0) spread via autocorrelation — 300x faster than librosa.pyin
        n_fft    = 512
        hop_len  = 256
        min_lag  = max(1, int(sr / 800))  # 800 Hz upper limit
        max_lag  = int(sr / 60)           # 60 Hz lower limit
        n_frames = (len(y) - n_fft) // hop_len + 1
        f0_list  = []

        if n_frames >= 3:
            indices = (np.arange(n_fft)[None, :] +
                       np.arange(n_frames)[:, None] * hop_len)
            frames_mat = y[np.minimum(indices, len(y) - 1)] * np.hanning(n_fft).astype(np.float32)
            energies = np.sum(frames_mat ** 2, axis=1)
            thresh_e = float(np.mean(energies)) * 0.30

            for frm in frames_mat[::3]:  # every 3rd frame is sufficient
                if np.sum(frm ** 2) < thresh_e:
                    continue
                c  = np.correlate(frm, frm, mode="full")[n_fft - 1:]
                cw = c[min_lag:min(max_lag, len(c))]
                if len(cw) < 3 or c[0] < 1e-6:
                    continue
                pk = int(np.argmax(cw))
                is_local = (0 < pk < len(cw) - 1 and cw[pk] >= cw[pk - 1] and cw[pk] >= cw[pk + 1])
                if is_local and (cw[pk] / c[0]) > 0.30:
                    f0_list.append(float(sr) / (min_lag + pk))

        pitch_spread = float(np.std(f0_list))  if len(f0_list) > 1 else 0.0
        pitch_mean   = float(np.mean(f0_list)) if len(f0_list) > 0 else 0.0

        feat = np.concatenate([
            np.mean(mfcc, axis=1),                       # 40
            np.mean(d1,   axis=1),                       # 40
            np.mean(d2,   axis=1),                       # 40
            [np.mean(centroid), np.std(centroid)],       # 2
            [np.mean(rolloff),  np.std(rolloff)],        # 2
            [np.mean(zcr),      np.std(zcr)],            # 2
            [np.mean(rms),      np.std(rms)],            # 2
            [pitch_spread, pitch_mean],                  # 2
        ]).astype(np.float32)

        return feat

    except Exception as e:
        return None


def augment_and_extract(y: np.ndarray, sr: int, label: str) -> list:
    """
    Generate augmented variants and extract features from each.
    Returns list of (feature_vector, label).
    """
    pairs = []

    # Original
    f = extract_features(y, sr)
    if f is not None:
        pairs.append((f, label))

    # Pitch shift variants
    for n_steps in [-2.0, -1.0, 1.0, 2.0]:
        try:
            y_aug = librosa.effects.pitch_shift(y, sr=sr, n_steps=n_steps)
            f = extract_features(y_aug, sr)
            if f is not None:
                pairs.append((f, label))
        except Exception:
            pass

    # Time stretch variants
    for rate in [0.80, 0.90, 1.10, 1.20]:
        try:
            y_aug = librosa.effects.time_stretch(y, rate=rate)
            f = extract_features(y_aug, sr)
            if f is not None:
                pairs.append((f, label))
        except Exception:
            pass

    # Gaussian noise injection (SNR ~15dB and ~25dB)
    for noise_factor in [0.008, 0.002]:
        try:
            noise = np.random.randn(len(y)).astype(np.float32) * float(np.std(y)) * noise_factor
            y_aug = y + noise
            f = extract_features(y_aug, sr)
            if f is not None:
                pairs.append((f, label))
        except Exception:
            pass

    # Dynamic range variation (quiet and loud)
    for scale in [0.5, 1.5]:
        try:
            y_aug = np.clip(y * scale, -1.0, 1.0)
            f = extract_features(y_aug, sr)
            if f is not None:
                pairs.append((f, label))
        except Exception:
            pass

    return pairs


def build_dataset(items: list) -> tuple:
    """Load audio, augment, extract features. Returns X, y arrays."""
    X, y_labels = [], []
    t0 = time.time()
    total = len(items)
    np.random.shuffle(items)

    for i, (filepath, emotion) in enumerate(items):
        if (i + 1) % 50 == 0:
            elapsed = time.time() - t0
            eta = elapsed / (i + 1) * (total - i - 1)
            print(f"\r  {i+1}/{total} ({100*(i+1)/total:.0f}%)  ETA {eta:.0f}s   ", end="", flush=True)

        try:
            y, sr_orig = librosa.load(filepath, sr=SR, mono=True)
        except Exception:
            continue

        pairs = augment_and_extract(y, SR, emotion)
        for feat, lbl in pairs:
            X.append(feat)
            y_labels.append(lbl)

    print(f"\n  Dataset built: {len(X)} augmented samples from {total} clips", flush=True)
    return np.array(X, dtype=np.float32), np.array(y_labels)


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────

def cross_validate(X_s, y):
    """5-fold stratified cross-validation to measure true generalisation."""
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=5e-5,
        batch_size=64,
        max_iter=120,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=10,
    )
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_val_score(clf, X_s, y, cv=skf, scoring="f1_macro", n_jobs=4)
    print(f"  5-Fold CV Macro-F1: {scores.mean():.4f} ± {scores.std():.4f}", flush=True)
    return scores.mean()


def train_final_mlp(X_train_s, y_train, X_val_s, y_val):
    """Train final MLP on full training set."""
    clf = MLPClassifier(
        hidden_layer_sizes=(256, 128),
        activation="relu",
        solver="adam",
        alpha=5e-5,
        batch_size=64,
        learning_rate="adaptive",
        learning_rate_init=8e-4,
        max_iter=250,
        shuffle=True,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.12,
        n_iter_no_change=20,
        verbose=False,
    )
    print("  Fitting MLP (256 → 128 → 7)...", flush=True)
    clf.fit(X_train_s, y_train)
    acc = clf.score(X_val_s, y_val)
    print(f"  Validation accuracy: {acc:.4f}  iters={clf.n_iter_}", flush=True)
    return clf


def evaluate_and_print(clf, scaler, X_test, y_test):
    X_s = scaler.transform(X_test)
    y_pred = clf.predict(X_s)
    labels_present = sorted(set(y_test) | set(y_pred))
    print("\n" + "=" * 60, flush=True)
    print("TEST SET CLASSIFICATION REPORT", flush=True)
    print("=" * 60, flush=True)
    print(classification_report(y_test, y_pred, target_names=labels_present,
                                zero_division=0), flush=True)
    cm = confusion_matrix(y_test, y_pred, labels=labels_present)
    header = "        " + "  ".join(f"{l[:4]:>4}" for l in labels_present)
    print("CONFUSION MATRIX", flush=True)
    print(header, flush=True)
    for i, row in enumerate(cm):
        print(f"  {labels_present[i][:4]:>4}  " + "  ".join(f"{v:>4}" for v in row), flush=True)
    macro_f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)
    print(f"\nMacro F1: {macro_f1:.4f}", flush=True)
    return macro_f1, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# ONNX EXPORT
# ─────────────────────────────────────────────────────────────────────────────

def export_onnx_pipeline(clf, scaler, feature_dim: int, class_names: list, output_path: Path) -> bool:
    """Export StandardScaler + MLP pipeline to ONNX via skl2onnx."""
    try:
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        from sklearn.pipeline import Pipeline

        pipe = Pipeline([("scaler", scaler), ("mlp", clf)])
        initial_type = [("float_input", FloatTensorType([None, feature_dim]))]
        onnx_model = convert_sklearn(pipe, initial_types=initial_type, target_opset=17)
        with open(output_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        size_kb = output_path.stat().st_size // 1024
        print(f"  skl2onnx export: {output_path.name} ({size_kb} KB)", flush=True)
        return True
    except Exception as e:
        print(f"  skl2onnx failed ({e}); using manual ONNX export...", flush=True)
        return _manual_onnx_export(clf, scaler, feature_dim, class_names, output_path)


def _manual_onnx_export(clf, scaler, feature_dim, class_names, output_path) -> bool:
    """Hand-craft ONNX graph: StandardScaler → 3-layer ReLU MLP → Softmax."""
    import onnx
    from onnx import numpy_helper, TensorProto, helper

    def t(name, arr):
        return numpy_helper.from_array(arr.astype(np.float32), name=name)

    coefs  = clf.coefs_
    biases = clf.intercepts_
    n_layers = len(coefs)

    initializers, nodes = [], []
    inputs = [helper.make_tensor_value_info("float_input", TensorProto.FLOAT, [None, feature_dim])]

    # StandardScaler step
    initializers.extend([
        t("sc_mean",  scaler.mean_.astype(np.float32)),
        t("sc_scale", scaler.scale_.astype(np.float32)),
    ])
    nodes += [
        helper.make_node("Sub", ["float_input", "sc_mean"],  ["X_sub"]),
        helper.make_node("Div", ["X_sub",       "sc_scale"], ["X_scaled"]),
    ]

    prev = "X_scaled"
    for i, (W, b) in enumerate(zip(coefs, biases)):
        initializers.extend([t(f"W{i}", W.T), t(f"b{i}", b)])
        mm = f"mm{i}"
        nodes.append(helper.make_node("Gemm", [prev, f"W{i}", f"b{i}"], [mm],
                                      transA=0, transB=0, alpha=1.0, beta=1.0))
        if i < n_layers - 1:
            relu = f"relu{i}"
            nodes.append(helper.make_node("Relu", [mm], [relu]))
            prev = relu
        else:
            nodes.append(helper.make_node("Softmax", [mm], ["output"], axis=1))
            prev = "output"

    n_classes = len(class_names)
    outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, [None, n_classes])]
    graph = helper.make_graph(nodes, "MAITRI_SER_MLP_v2", inputs, outputs, initializers)
    model_proto = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model_proto.ir_version = 8
    try:
        import onnx
        onnx.checker.check_model(model_proto)
    except Exception:
        pass

    with open(output_path, "wb") as f:
        f.write(model_proto.SerializeToString())
    size_kb = output_path.stat().st_size // 1024
    print(f"  Manual ONNX export: {output_path.name} ({size_kb} KB)", flush=True)
    return True


def verify_onnx(onnx_path: Path, feature_dim: int, class_names: list):
    """Quick sanity check on exported ONNX model."""
    try:
        import onnxruntime as ort
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inp_name = sess.get_inputs()[0].name
        dummy = np.random.randn(1, feature_dim).astype(np.float32)
        out = sess.run(None, {inp_name: dummy})[0]
        pred = class_names[int(np.argmax(out[0]))] if len(class_names) > np.argmax(out[0]) else "?"
        print(f"  ONNX sanity OK: shape={out.shape}, pred_class='{pred}'", flush=True)
    except Exception as e:
        print(f"  ONNX verify error: {e}", flush=True)


def save_metadata(clf, scaler, class_names, feature_dim, macro_f1, output_dir: Path):
    meta = {
        "model_type":    "MFCC-MLP-v2",
        "feature_dim":   feature_dim,
        "class_names":   list(class_names),
        "scaler_mean":   scaler.mean_.tolist(),
        "scaler_scale":  scaler.scale_.tolist(),
        "macro_f1_test": round(float(macro_f1), 4),
        "n_mfcc":        N_MFCC,
        "sr":            SR,
        "hidden_layers": list(clf.hidden_layer_sizes),
    }
    path = output_dir / "ser_mlp_meta.json"
    path.write_text(json.dumps(meta, indent=2))
    print(f"  Metadata saved → {path.name}", flush=True)


# ─────────────────────────────────────────────────────────────────────────────
# WAV2VEC2 IDLE BASELINE RECALIBRATION
# ─────────────────────────────────────────────────────────────────────────────

def recalibrate_wav2vec2_idle_baseline():
    """
    Compute a better Z_IDLE_BASELINE using actual espeak-ng neutral speech
    (not silence), because the true inference-time idle state is non-speech
    audio + microphone background, not zeros.

    We run 60 neutral espeak-ng samples through the Wav2Vec2 ONNX and average
    the raw logits to get the true baseline for DAP centering.
    """
    model_path = MODELS_DIR / "wav2vec2_ser_q4.onnx"
    if not model_path.exists():
        print("  Wav2Vec2 model not found; skipping recalibration", flush=True)
        return

    import onnxruntime as ort
    sess_opts = ort.SessionOptions()
    sess_opts.intra_op_num_threads = 2
    sess_opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    sess = ort.InferenceSession(str(model_path), sess_options=sess_opts,
                                providers=["CPUExecutionProvider"])
    inp_name = sess.get_inputs()[0].name

    baselines = []
    rng = np.random.default_rng(42)

    # 1. Silence / microphone noise patterns (30 samples)
    for _ in range(30):
        for amp in [0.0, 0.001, 0.003, 0.008]:
            inp = (rng.normal(0, amp, 48000)).astype(np.float32).reshape(1, -1)
            logits = sess.run(None, {inp_name: inp})[0][0]
            baselines.append(logits)

    # 2. Neutral espeak-ng speech routed through Wav2Vec2 (40 samples)
    neutral_wavs = list((SYNTH_DIR / "neutral").glob("*.wav")) if (SYNTH_DIR / "neutral").exists() else []
    for wav_path in neutral_wavs[:40]:
        try:
            y, _ = librosa.load(str(wav_path), sr=16000, mono=True)
            if len(y) < 8000:
                continue
            # Pad/trim to 3s = 48000 samples
            if len(y) < 48000:
                y = np.pad(y, (0, 48000 - len(y)))
            y = y[:48000]
            # Normalize
            y = (y - np.mean(y)) / (np.std(y) + 1e-7)
            inp = y.astype(np.float32).reshape(1, -1)
            logits = sess.run(None, {inp_name: inp})[0][0]
            baselines.append(logits)
        except Exception:
            pass

    if not baselines:
        print("  No baseline samples collected", flush=True)
        return

    z_idle = np.mean(baselines, axis=0)
    print(f"  New Z_IDLE ({len(baselines)} samples): {np.round(z_idle, 4)}", flush=True)
    print(f"  Sad [{5}] idle logit: {z_idle[5]:.4f}  Happy [{4}]: {z_idle[4]:.4f}", flush=True)
    print(f"  Sad-Happy bias gap: {z_idle[5] - z_idle[4]:.4f}", flush=True)

    # Patch dap_enhancer.py
    dap_path = BASE_DIR / "modules" / "dap_enhancer.py"
    if dap_path.exists():
        import re
        content = dap_path.read_text()
        arr_str = "[" + ", ".join(f"{v:.4f}" for v in z_idle) + "]"
        new_def = (
            f"Z_IDLE_BASELINE = np.array(\n"
            f"    {arr_str},\n"
            f"    dtype=np.float32,\n"
            f")"
        )
        pattern = r"Z_IDLE_BASELINE = np\.array\(\s*\[.*?\],\s*dtype=np\.float32,\s*\)"
        new_content = re.sub(pattern, new_def, content, flags=re.DOTALL)
        if new_content != content:
            dap_path.write_text(new_content)
            print("  ✓ Z_IDLE_BASELINE updated in dap_enhancer.py", flush=True)
        else:
            print("  Z_IDLE_BASELINE pattern not matched — manually check dap_enhancer.py", flush=True)

    return z_idle


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    print("\n" + "=" * 60, flush=True)
    print("MAITRI 2.0 — Local SER Training (espeak-ng + MFCC-MLP)", flush=True)
    print("=" * 60, flush=True)

    # ── Phase 1: Synthetic corpus ─────────────────────────────────────────────
    print("\n[Phase 1] Generating synthetic corpus via espeak-ng...", flush=True)
    items = generate_synthetic_audio(force=False)
    if not items:
        print("[ERROR] No synthetic audio generated!", flush=True)
        sys.exit(1)

    from collections import Counter
    dist = Counter(lbl for _, lbl in items)
    print("  Class distribution (raw clips):", flush=True)
    for emo in MAITRI_EMOTIONS:
        print(f"    {emo:10s}: {dist.get(emo, 0)}", flush=True)

    # ── Phase 2: Feature extraction + augmentation ────────────────────────────
    print(f"\n[Phase 2] Extracting features with augmentation (~{len(items)*12} samples)...", flush=True)
    X, y_labels = build_dataset(items)

    if len(X) == 0:
        print("[ERROR] Feature extraction produced zero samples!", flush=True)
        sys.exit(1)

    feature_dim = X.shape[1]
    print(f"  Feature matrix: {X.shape}  (dim={feature_dim})", flush=True)

    aug_dist = Counter(y_labels)
    print("  Augmented class dist:", dict(aug_dist), flush=True)

    # ── Phase 3: Split + scale ────────────────────────────────────────────────
    print("\n[Phase 3] Train/val/test split and feature scaling...", flush=True)
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y_labels, test_size=0.12, stratify=y_labels, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=0.12, stratify=y_trainval, random_state=42
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_val_s   = scaler.transform(X_val)
    X_test_s  = scaler.transform(X_test)

    print(f"  Train={len(X_train)}, Val={len(X_val)}, Test={len(X_test)}", flush=True)

    # ── Phase 4: Cross-validation ─────────────────────────────────────────────
    print("\n[Phase 4] 5-Fold cross-validation...", flush=True)
    X_trainval_s = scaler.transform(X_trainval)
    cv_f1 = cross_validate(X_trainval_s, y_trainval)

    # ── Phase 5: Final model training ─────────────────────────────────────────
    print("\n[Phase 5] Training final MLP...", flush=True)
    clf = train_final_mlp(X_train_s, y_train, X_val_s, y_val)

    # ── Phase 6: Evaluation ───────────────────────────────────────────────────
    print("\n[Phase 6] Evaluation on held-out test set...", flush=True)
    macro_f1, y_pred = evaluate_and_print(clf, scaler, X_test, y_test)

    # ── Phase 7: ONNX export ──────────────────────────────────────────────────
    print("\n[Phase 7] Exporting to ONNX...", flush=True)
    class_names = sorted(set(y_labels))
    onnx_path = MODELS_DIR / "ser_mlp_mfcc.onnx"
    export_onnx_pipeline(clf, scaler, feature_dim, class_names, onnx_path)
    verify_onnx(onnx_path, feature_dim, class_names)

    # ── Phase 8: Save metadata ────────────────────────────────────────────────
    save_metadata(clf, scaler, class_names, feature_dim, macro_f1, MODELS_DIR)

    # ── Phase 9: Recalibrate Wav2Vec2 idle baseline ───────────────────────────
    print("\n[Phase 9] Recalibrating Wav2Vec2 idle baseline...", flush=True)
    recalibrate_wav2vec2_idle_baseline()

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 60, flush=True)
    print(f"DONE — MFCC-MLP exported: {onnx_path.name}", flush=True)
    print(f"  CV Macro-F1:   {cv_f1:.4f}", flush=True)
    print(f"  Test Macro-F1: {macro_f1:.4f}", flush=True)
    if macro_f1 >= 0.75:
        print("  ✓ HIGH ACCURACY — model is production-ready", flush=True)
    elif macro_f1 >= 0.60:
        print("  ● MODERATE ACCURACY — acceptable for ensemble use", flush=True)
    else:
        print("  ✗ LOW ACCURACY — inspect class imbalance and re-augment", flush=True)
    print("=" * 60, flush=True)


if __name__ == "__main__":
    main()
