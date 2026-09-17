import os

# ── TF / CUDA env flags (must be BEFORE any TF/ML import) ─────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

import queue
import threading
import time
import av
import cv2
import numpy as np
import streamlit as st

# ── Pure-Python project modules (safe to import at startup) ───────────────
from modules.alert_module    import get_alert
from modules.database_module import get_logs, get_recent_trend, init_db, log_event
from modules.fusion_module   import FusionState, fuse
from modules.live_state      import LiveState
from modules.vitals_module   import compute_vitals_strain
# NOTE: face_module and eye_module are imported INSIDE MAITRIVideoProcessor
# to prevent TF/MediaPipe from loading at Streamlit startup (segfault fix).

# ── WebRTC ────────────────────────────────────────────────────────────────
try:
    from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
    _WEBRTC_OK = True
except ImportError:
    _WEBRTC_OK = False

# ── DB init ───────────────────────────────────────────────────────────────
init_db()

# ── Page config ───────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Project MAITRI 2.0",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Global CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.4rem !important; }
.stress-badge {
    display: inline-block; padding: 5px 14px;
    border-radius: 20px; font-weight: bold;
    font-size: 1rem; color: white;
}
.emo-row { display:flex; align-items:center; gap:8px; margin:3px 0; }
.emo-label { width:72px; font-size:0.82rem; text-transform:capitalize; }
.emo-bar-bg { flex:1; background:#2d2d2d; border-radius:6px; height:16px; }
.emo-bar { border-radius:6px; height:16px; transition:width 0.5s; }
.emo-pct { width:44px; font-size:0.82rem; text-align:right; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────
if "live_state"   not in st.session_state:
    st.session_state.live_state   = LiveState()
if "fusion_state" not in st.session_state:
    st.session_state.fusion_state = FusionState()
if "eye_state"    not in st.session_state:
    from modules.eye_module import EyeSessionState
    st.session_state.eye_state    = EyeSessionState()

# ─────────────────────────────────────────────────────────────────────────
# WEBRTC VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────
_EYE_EVERY_N_FRAMES = 3   # kept for reference — eye worker throttles internally

class MAITRIVideoProcessor:
    """
    Processes each WebRTC video frame in a background thread.
    State is passed via constructor (NOT read from st.session_state,
    which is unavailable in WebRTC worker threads).

    Performance architecture (ZERO-BLOCKING recv()):
    - DeepFace runs in dedicated daemon thread — single-slot queue, frames dropped when busy.
    - MediaPipe eye tracking ALSO runs in its own daemon thread — single-slot queue.
    - recv() does NOTHING except: copy frame → try_enqueue both queues → draw HUD → return.
    - HUD uses only cached LiveState — never waits for inference.
    """

    def __init__(self, live_state: LiveState, eye_state):
        self.live_state = live_state
        self.eye_state  = eye_state

        # Single-slot queues — if worker busy, frame dropped (no backlog, no lag)
        self._face_queue: queue.Queue = queue.Queue(maxsize=1)
        self._eye_queue:  queue.Queue = queue.Queue(maxsize=1)

        # DeepFace worker
        self._face_worker = threading.Thread(
            target=self._face_inference_worker, daemon=True, name="deepface-worker"
        )
        self._face_worker.start()

        # MediaPipe eye worker
        self._eye_worker = threading.Thread(
            target=self._eye_inference_worker, daemon=True, name="eye-worker"
        )
        self._eye_worker.start()

    # ── Background DeepFace worker ────────────────────────────────────────
    def _face_inference_worker(self) -> None:
        from modules.face_module import analyze_frame
        while True:
            bgr = self._face_queue.get()
            if bgr is None:
                break
            try:
                face = analyze_frame(bgr)
                ls = self.live_state
                with ls.lock:
                    ls.face_emotion    = face.dominant_emotion
                    ls.emotion_probs   = {
                        k: float(v) / 100.0 for k, v in face.emotion_probs.items()
                    }
                    ls.face_confidence = face.face_confidence
                    ls.face_quality    = face.face_quality
                    ls.face_error      = face.error
                    ls.blur_score      = face.blur_score
                    ls.is_blurry       = face.is_blurry
                    ls.face_box        = face.face_box
            except Exception as exc:
                print(f"[MAITRI] Face worker error: {exc}")

    # ── Background Eye worker ─────────────────────────────────────────────
    def _eye_inference_worker(self) -> None:
        from modules.eye_module import analyze_eyes
        while True:
            bgr = self._eye_queue.get()
            if bgr is None:
                break
            try:
                eye = analyze_eyes(bgr, self.eye_state)
                ls  = self.live_state
                with ls.lock:
                    ls.ear            = eye.ear
                    ls.blink_rate     = eye.blink_rate
                    ls.fatigue_label  = eye.fatigue_label
                    ls.fatigue_strain = eye.fatigue_strain
                    ls.eye_quality    = eye.eye_quality
                    ls.eye_available  = eye.available
            except Exception as exc:
                print(f"[MAITRI] Eye worker error: {exc}")

    # ── Per-frame recv callback — ZERO blocking ───────────────────────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Called for EVERY incoming video frame by streamlit-webrtc.
        Returns immediately — all ML inference happens in background threads.
        """
        bgr = frame.to_ndarray(format="bgr24")
        ls  = self.live_state

        with ls.lock:
            ls.frame_count += 1

        # Non-blocking dispatch to background workers — passes independent copies to prevent HUD race
        try:
            self._face_queue.put_nowait(bgr.copy())
        except queue.Full:
            pass

        try:
            self._eye_queue.put_nowait(bgr.copy())
        except queue.Full:
            pass

        # Draw HUD using latest cached telemetry only
        bgr = _draw_hud(bgr, ls)
        return av.VideoFrame.from_ndarray(bgr, format="bgr24")


def _draw_hud(bgr: np.ndarray, ls: LiveState) -> np.ndarray:
    """Draw lightweight HUD and persistent face bounding box directly on the video frame."""
    snap = ls.snapshot()
    h, w = bgr.shape[:2]

    # Persistent face bounding box (updates in background from EfficientNet/MediaPipe worker)
    box = snap.get("face_box")
    if box:
        bx, by = box.get("x", 0), box.get("y", 0)
        bw, bh = box.get("w", 0), box.get("h", 0)
        if bw > 0 and bh > 0:
            box_color = (0, 165, 255) if snap["is_blurry"] else (0, 230, 118)
            cv2.rectangle(bgr, (bx, by), (bx + bw, by + bh), box_color, 2)

    # Emotion label top-left (with actual confidence and graceful searching state)
    emo   = snap["face_emotion"].upper()
    conf  = snap["face_confidence"] * 100
    if box is not None or snap["face_quality"] > 0.10:
        disp_text = f"{emo}  {conf:.0f}%"
        disp_color = (0, 230, 118) if emo in ["HAPPY", "NEUTRAL"] else (0, 165, 255)
    else:
        disp_text = "SCANNING FACE..."
        disp_color = (180, 180, 180)

    cv2.putText(bgr, disp_text, (12, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, disp_color, 2, cv2.LINE_AA)

    # Eye info bottom-left
    eye_label = f"EAR:{snap['ear']:.2f}  BLINK:{snap['blink_rate']:.0f}/min  {snap['fatigue_label']}"
    cv2.putText(bgr, eye_label, (12, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 200, 0), 1, cv2.LINE_AA)

    # DIP telemetry top-right
    dip_status = "BLUR GATED" if snap["is_blurry"] else "DIP: ENET-B2 + CLAHE"
    dip_color  = (0, 165, 255) if snap["is_blurry"] else (0, 230, 118)
    cv2.putText(bgr, f"{dip_status}  #{snap['frame_count']}", (w - 260, 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, dip_color, 1, cv2.LINE_AA)
    return bgr



# ─────────────────────────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────────────────────────
st.markdown(
    "<h1 style='text-align:center;margin-bottom:4px;'>🛰️ Project MAITRI 2.0</h1>"
    "<p style='text-align:center;color:#aaa;margin-top:0;font-size:0.9rem;'>"
    "Live Multimodal AI Astronaut Telemetry &amp; Psychological Monitoring</p>",
    unsafe_allow_html=True,
)
st.markdown("---")

# ─────────────────────────────────────────────────────────────────────────
# LIVE TELEMETRY FRAGMENTS (Smooth 1 Hz Refresh Without UI Lock)
# ─────────────────────────────────────────────────────────────────────────
def _render_dip_status(is_playing: bool):
    if is_playing:
        snap_v = st.session_state.live_state.snapshot()
        blur_status = (
            "<span style='color:#f39c12;'>⚠️ Motion Blur (Fusion Gated)</span>"
            if snap_v["is_blurry"]
            else f"<span style='color:#2ecc71;'>✅ Sharp ({snap_v['blur_score']:.0f})</span>"
        )
        st.markdown(
            f"<div style='background:#1b2838;padding:8px 12px;border-radius:6px;font-size:0.83rem;margin-top:6px;border:1px solid #2a475e;'>"
            f"🛡️ <b>DIP Pipeline:</b> CLAHE (LAB) + Adaptive Gamma (Melanin Invariant) + Unsharp Mask &nbsp;|&nbsp; "
            f"<b>Clarity:</b> {blur_status}"
            f"</div>",
            unsafe_allow_html=True,
        )


@st.fragment(run_every=1.0)
def _render_eye_panel(is_playing: bool):
    ls_snap = st.session_state.live_state.snapshot()
    if is_playing:
        e1, e2 = st.columns(2)
        e1.metric("EAR", f"{ls_snap['ear']:.3f}")
        e2.metric("Blink Rate", f"{ls_snap['blink_rate']:.1f} /min")
        eye_color = {
            "Normal": "#2ecc71", "Drowsy": "#e74c3c",
            "Stressed Eyes": "#e67e22", "Hyperfocused": "#f1c40f",
        }.get(ls_snap["fatigue_label"], "#aaa")
        st.markdown(
            f"<span class='stress-badge' style='background:{eye_color};'>"
            f"👁️ {ls_snap['fatigue_label']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(f"Frames processed: {ls_snap['frame_count']}")
    else:
        st.info("▶️ Start the camera stream to begin eye tracking.")


@st.fragment(run_every=1.0)
def _render_live_assessment(
    heart_rate: float,
    skin_temp: float,
    spo2: float,
    vitals_strain: float,
    is_playing: bool,
):
    ls_snap = st.session_state.live_state.snapshot()

    # Build fusion inputs from live state
    face_probs   = ls_snap["emotion_probs"] if is_playing else None
    face_quality = ls_snap["face_quality"]  if is_playing else 0.0
    face_emotion = ls_snap["face_emotion"]

    fusion = fuse(
        state          = st.session_state.fusion_state,
        face_probs     = face_probs,
        face_quality   = face_quality,
        vitals_strain  = vitals_strain,
        fatigue_strain = ls_snap["fatigue_strain"],
        eye_quality    = ls_snap["eye_quality"] if is_playing else 0.0,
    )

    # ── Metrics strip ─────────────────────────────────────────────────────
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Face Weight",    f"{fusion.face_weight*100:.0f}%")
    m2.metric("Vitals Weight",  f"{fusion.vitals_weight*100:.0f}%")
    m3.metric("Eye Weight",     f"{fusion.eye_weight*100:.0f}%")
    m4.metric("🎯 Stress Index", f"{fusion.stress_pct:.1f}%")

    emo_col, gauge_col = st.columns([1.2, 1.0], gap="large")

    EMO_COLORS = {
        "angry": "#e74c3c", "disgust": "#27ae60", "fear": "#9b59b6",
        "happy": "#f1c40f", "neutral": "#95a5a6", "sad": "#3498db",
        "surprise": "#e67e22",
    }

    with emo_col:
        st.markdown("**Fused Emotion Distribution**")
        sorted_emos = sorted(fusion.fused_probs.items(), key=lambda x: x[1], reverse=True)
        for emo, prob in sorted_emos:
            pct       = round(prob * 100, 1)
            bar_w     = max(pct, 1)
            bar_color = EMO_COLORS.get(emo, "#888")
            marker    = " ◀" if emo == fusion.dominant_emotion else ""
            st.markdown(
                f"<div class='emo-row'>"
                f"<span class='emo-label'>{emo}</span>"
                f"<div class='emo-bar-bg'>"
                f"<div class='emo-bar' style='width:{bar_w}%;background:{bar_color};'></div>"
                f"</div>"
                f"<span class='emo-pct'>{pct:.1f}%{marker}</span>"
                f"</div>",
                unsafe_allow_html=True,
            )

    with gauge_col:
        alert = get_alert(fusion.stress_pct, fusion.dominant_emotion, ls_snap.get("fatigue_label", "Normal"))
        st.markdown("**Stress Level**")
        st.markdown(
            f"<div style='text-align:center;padding:12px;border-radius:12px;"
            f"background:{alert.color}22;border:2px solid {alert.color};'>"
            f"<div style='font-size:2.6rem;'>{alert.icon}</div>"
            f"<div class='stress-badge' style='background:{alert.color};font-size:1.1rem;'>"
            f"{alert.stress_level}</div>"
            f"<div style='font-size:2rem;font-weight:bold;margin:6px 0;'>"
            f"{fusion.stress_pct:.1f}%</div>"
            f"<div style='font-size:0.8rem;color:#ccc;'>Combined Stress Index</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
        emo_color = EMO_COLORS.get(fusion.dominant_emotion, "#888")
        st.markdown(
            f"<div style='text-align:center;margin-top:8px;'>"
            f"<span class='stress-badge' style='background:{emo_color};'>"
            f"Emotion: {fusion.dominant_emotion.capitalize()}</span></div>",
            unsafe_allow_html=True,
        )

    # ── ROW 3: Support Response ───────────────────────────────────────────
    st.markdown("---")
    st.subheader("🤖 Autonomous Psychological Support")

    resp_col, cbt_col = st.columns([1.0, 1.2], gap="medium")
    alert_style = (
        f"background:{alert.color}18;border-left:4px solid {alert.color};"
        f"padding:14px;border-radius:8px;"
    )
    with resp_col:
        st.markdown(
            f"<div style='{alert_style}'><strong>{alert.icon} {alert.header}</strong>"
            f"<br><br>{alert.body}</div>",
            unsafe_allow_html=True,
        )

    with cbt_col:
        if alert.technique:
            st.markdown(
                f"<div style='background:#1a1a2e;padding:14px;border-radius:8px;"
                f"border:1px solid #444;'>"
                f"{alert.technique.replace(chr(10), '<br>')}</div>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<div style='background:#1a3a1a;padding:14px;border-radius:8px;"
                "border:1px solid #2ecc71;color:#2ecc71;'>"
                "✅ No intervention required. Astronaut is performing optimally.</div>",
                unsafe_allow_html=True,
            )

    # ── Log button ────────────────────────────────────────────────────────
    st.markdown("---")
    log_col, _ = st.columns([1, 3])
    with log_col:
        if st.button("💾 Log Reading to Mission Database", use_container_width=True):
            log_event(
                face_emotion    = face_emotion,
                fused_emotion   = fusion.dominant_emotion,
                voice_state     = "N/A",
                heart_rate      = float(heart_rate),
                temperature     = float(skin_temp),
                spo2            = float(spo2),
                blink_rate      = ls_snap["blink_rate"],
                fatigue_label   = ls_snap["fatigue_label"],
                stress_score    = fusion.stress_pct,
                stress_level    = alert.stress_level,
                alert_triggered = alert.alert_label,
                response_msg    = alert.body,
            )
            st.session_state["last_logged_time"] = time.time()

        if time.time() - st.session_state.get("last_logged_time", 0) < 4.0:
            st.success("✅ Mission telemetry logged to database.")

    if is_playing:
        st.caption("⚡ Real-time Telemetry: Live assessment & support engine updating dynamically (1 Hz).")


# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────
tab_live, tab_logs = st.tabs(["🔴 Live Monitoring", "📊 Mission Logs & Trends"])

# ═════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITORING
# ═════════════════════════════════════════════════════════════════════════
with tab_live:

    if not _WEBRTC_OK:
        st.error(
            "**streamlit-webrtc not installed.** Run:\n"
            "```\n/home/mikey/anaconda3/envs/maitri/bin/pip install streamlit-webrtc aiortc\n```"
        )
        st.stop()

    # ── ROW 1: Live Video + Vitals ────────────────────────────────────────
    col_video, col_vitals = st.columns([1.4, 1.0], gap="medium")

    with col_video:
        st.subheader("📹 Live Astronaut Feed")
        st.caption("Background ML pipeline · MediaPipe Eye Tracking (~10 FPS)")

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        # ── Capture state refs on MAIN thread before passing to factory ───
        _live_state = st.session_state.live_state
        _eye_state  = st.session_state.eye_state

        def _processor_factory():
            """Closure — captures state from main thread, safe for worker."""
            return MAITRIVideoProcessor(_live_state, _eye_state)

        ctx = webrtc_streamer(
            key="maitri-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=_processor_factory,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
            async_processing=True,
        )

        _render_dip_status(ctx.state.playing)

    with col_vitals:
        st.subheader("💓 Physiological Telemetry")
        st.caption("Adjust sliders to reflect sensor readings")

        heart_rate = st.slider("❤️  Heart Rate (BPM)",      50,  160, 75)
        skin_temp  = st.slider("🌡️  Skin Temperature (°C)", 35.0, 40.0, 36.6, step=0.1)
        spo2       = st.slider("🫁  Blood Oxygen (SpO₂ %)", 85,  100, 98)

        vitals = compute_vitals_strain(heart_rate, skin_temp, float(spo2))

        v1, v2, v3 = st.columns(3)
        v1.metric("HR Strain",   f"{vitals.hr_strain*100:.0f}%")
        v2.metric("Temp Strain", f"{vitals.temp_strain*100:.0f}%")
        v3.metric("SpO₂ Strain", f"{vitals.spo2_strain*100:.0f}%")

        vitals_color = {"Normal": "#2ecc71", "Elevated": "#e67e22", "Critical": "#e74c3c"}.get(
            vitals.status, "#aaa"
        )
        st.markdown(
            f"<span class='stress-badge' style='background:{vitals_color};'>"
            f"Vitals: {vitals.status}</span>",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("👁️ Eye Tracking (Live)")
        _render_eye_panel(ctx.state.playing)

    # ── ROW 2 & ROW 3: Live Assessment & Autonomous Psychological Support ──
    st.markdown("---")
    st.subheader("🧠 Live Multimodal Stress Assessment")
    _render_live_assessment(
        heart_rate=heart_rate,
        skin_temp=skin_temp,
        spo2=spo2,
        vitals_strain=vitals.vitals_strain,
        is_playing=ctx.state.playing,
    )

# ═════════════════════════════════════════════════════════════════════════
# TAB 2 — MISSION LOGS
# ═════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("📋 Mission Telemetry History")

    log_df = get_logs()
    if log_df.empty:
        st.info("No telemetry logs yet. Use '💾 Log Reading' in the Live tab.")
    else:
        trend_df = get_recent_trend(60)
        if not trend_df.empty and "stress_score" in trend_df.columns:
            st.markdown("**Stress Index Trend (last 60 readings)**")
            st.line_chart(trend_df.set_index("timestamp")["stress_score"], color="#e74c3c")

        st.markdown("---")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Readings", len(log_df))
        c2.metric("Avg Stress",     f"{log_df['stress_score'].mean():.1f}%")
        c3.metric("Peak Stress",    f"{log_df['stress_score'].max():.1f}%")
        c4.metric("Critical Alerts",
                  len(log_df[log_df["stress_level"] == "CRITICAL"])
                  if "stress_level" in log_df.columns else "–")

        st.markdown("---")
        st.markdown("**Full Telemetry Log**")
        st.dataframe(log_df, use_container_width=True, height=380)

        if "fused_emotion" in log_df.columns:
            st.markdown("**Emotion Distribution**")
            st.bar_chart(log_df["fused_emotion"].value_counts())