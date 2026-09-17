import os

# ── TF / CUDA env flags (must be BEFORE any TF/ML import) ─────────────────
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"]  = "3"
os.environ["CUDA_VISIBLE_DEVICES"]  = "-1"

import queue
import threading
import time
import datetime
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
from modules.watchdog_module import AstronautWatchdog, WatchdogState, WatchdogStatus
from modules.dtn_module      import DTNOutbox, DTNPacket
from modules.report_generator import generate_medical_dossier
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
    page_title="Project MAITRI 2.0 — ISRO Gaganyaan",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Aerospace Mission Profiles Catalog ────────────────────────────────────
MISSION_PROFILES = {
    "ISRO Gaganyaan (LEO Orbital)": {
        "latency_sec": 0.5,
        "tag": "S-Band Direct Link",
        "habitat": "Orbital Module Cabin M-02",
        "agency": "ISRO HSFC",
    },
    "Artemis Gateway (Lunar Orbit)": {
        "latency_sec": 2.4,
        "tag": "Lunar DSN Relay",
        "habitat": "HALO Habitat Module",
        "agency": "NASA / ESA",
    },
    "Mars Deep-Space Transit Habitat": {
        "latency_sec": 860.0,  # 14m 20s
        "tag": "DTN Active (RFC 5050)",
        "habitat": "Transit Hab Deep Space",
        "agency": "Interplanetary Consortium",
    },
    "Antarctic ICE Station Analog (Concordia)": {
        "latency_sec": 45.0,
        "tag": "Iridium Constellation",
        "habitat": "Dome C Isolated Station",
        "agency": "ESA / IPEV Analog",
    },
}

# ── Global CSS & Flight Deck Aesthetics ──────────────────────────────────
st.markdown("""
<style>
[data-testid="stMetricValue"] { font-size: 1.35rem !important; }
.stress-badge {
    display: inline-block; padding: 5px 14px;
    border-radius: 20px; font-weight: bold;
    font-size: 0.95rem; color: white;
}
.emo-row { display:flex; align-items:center; gap:8px; margin:3px 0; }
.emo-label { width:72px; font-size:0.82rem; text-transform:capitalize; }
.emo-bar-bg { flex:1; background:#1b2838; border-radius:6px; height:16px; border:1px solid #2a475e; }
.emo-bar { border-radius:6px; height:16px; transition:width 0.5s; }
.emo-pct { width:44px; font-size:0.82rem; text-align:right; font-family:monospace; }

@keyframes breathe_box {
    0%   { transform: scale(0.72); box-shadow: 0 0 10px rgba(72, 202, 228, 0.3); }
    25%  { transform: scale(1.18); box-shadow: 0 0 25px rgba(72, 202, 228, 0.9); }
    50%  { transform: scale(1.18); box-shadow: 0 0 25px rgba(72, 202, 228, 0.9); }
    75%  { transform: scale(0.72); box-shadow: 0 0 10px rgba(72, 202, 228, 0.3); }
    100% { transform: scale(0.72); box-shadow: 0 0 10px rgba(72, 202, 228, 0.3); }
}

@keyframes breathe_478 {
    0%   { transform: scale(0.70); box-shadow: 0 0 10px rgba(72, 202, 228, 0.3); }
    21%  { transform: scale(1.22); box-shadow: 0 0 30px rgba(72, 202, 228, 0.9); }
    58%  { transform: scale(1.22); box-shadow: 0 0 30px rgba(72, 202, 228, 0.9); }
    100% { transform: scale(0.70); box-shadow: 0 0 10px rgba(72, 202, 228, 0.3); }
}

@keyframes pulse_alert {
    0%   { opacity: 1.0; }
    50%  { opacity: 0.6; }
    100% { opacity: 1.0; }
}
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT
# ─────────────────────────────────────────────────────────────────────────
if "mission_profile" not in st.session_state:
    st.session_state.mission_profile = "ISRO Gaganyaan (LEO Orbital)"
if "mission_start_time" not in st.session_state:
    st.session_state.mission_start_time = time.time() - (4 * 86400 + 18 * 3600 + 22 * 60)
if "crew_member" not in st.session_state:
    st.session_state.crew_member = "Cdr. A. Sharma (Pilot / CMO)"
if "live_state" not in st.session_state:
    st.session_state.live_state = LiveState()
if "fusion_state" not in st.session_state:
    st.session_state.fusion_state = FusionState()
if "eye_state" not in st.session_state:
    from modules.eye_module import EyeSessionState
    st.session_state.eye_state = EyeSessionState()
if "watchdog" not in st.session_state:
    st.session_state.watchdog = AstronautWatchdog()
if "dtn_outbox" not in st.session_state:
    initial_delay = MISSION_PROFILES["ISRO Gaganyaan (LEO Orbital)"]["latency_sec"]
    st.session_state.dtn_outbox = DTNOutbox(
        ground_delay_seconds=initial_delay,
        mission_name="ISRO Gaganyaan (LEO Orbital)",
    )
if "last_dossier" not in st.session_state:
    st.session_state.last_dossier = None
if "slider_hr" not in st.session_state:
    st.session_state.slider_hr = 75
if "slider_temp" not in st.session_state:
    st.session_state.slider_temp = 36.6
if "slider_spo2" not in st.session_state:
    st.session_state.slider_spo2 = 98
if "hypoxia_dossier_sent" not in st.session_state:
    st.session_state.hypoxia_dossier_sent = False


def _format_met() -> str:
    elapsed = int(time.time() - st.session_state.mission_start_time)
    days = elapsed // 86400
    hours = (elapsed % 86400) // 3600
    mins = (elapsed % 3600) // 60
    secs = elapsed % 60
    return f"MET {days:02d}d:{hours:02d}h:{mins:02d}m:{secs:02d}s"


def _render_breathing_circle(technique_text: str) -> str:
    if "4-7-8" in technique_text:
        anim_name = "breathe_478"
        duration = "19s"
        caption = "4s Inhale ➔ 7s Hold ➔ 8s Exhale"
    else:
        anim_name = "breathe_box"
        duration = "16s"
        caption = "4s Inhale ➔ 4s Hold ➔ 4s Exhale ➔ 4s Hold"

    return f"""
    <div style="display:flex; flex-direction:column; align-items:center; justify-content:center; margin-top:12px; background:rgba(11,19,43,0.7); padding:14px; border-radius:8px; border:1px dashed #3a506b;">
      <div style="font-size:0.75rem; color:#48cae4; text-transform:uppercase; letter-spacing:1px; margin-bottom:8px; font-weight:bold;">
        🫁 Autonomous Breathing Pacer (Closed-Loop Somatic Regulation)
      </div>
      <div class="{anim_name}" style="width:64px; height:64px; border-radius:50%; background:radial-gradient(circle, #48cae4 0%, #0077b6 75%); animation:{anim_name} {duration} infinite ease-in-out; display:flex; align-items:center; justify-content:center; color:white; font-size:0.65rem; font-weight:bold; letter-spacing:1px;">
        PACE
      </div>
      <div style="font-size:0.75rem; color:#a0aec0; margin-top:10px; font-family:monospace;">
        {caption}
      </div>
    </div>
    """


# ─────────────────────────────────────────────────────────────────────────
# WEBRTC VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────
_EYE_EVERY_N_FRAMES = 2   # MediaPipe: runs at ~15 FPS, optimal for catching rapid blinks

class MAITRIVideoProcessor:
    """
    Processes each WebRTC video frame in a background thread.
    State is passed via constructor (NOT read from st.session_state,
    which is unavailable in WebRTC worker threads).
    ML libraries (TF, MediaPipe) only load when the first frame arrives.

    Performance architecture:
    - DeepFace (heavy, 200–500ms on CPU) runs in a separate daemon worker thread.
      recv() drops a frame into a single-slot queue (non-blocking put_nowait).
      If DeepFace is busy, the frame is silently dropped — the live video
      feed is NEVER delayed by inference.
    - MediaPipe eye tracking runs directly in recv() but only every 2nd frame,
      keeping landmark execution time to a fraction of the frame budget.
    - HUD is drawn on EVERY frame using the latest cached telemetry from LiveState.
    """

    def __init__(self, live_state: LiveState, eye_state):
        self.live_state = live_state
        self.eye_state  = eye_state

        # Single-slot queue — holds at most one pending frame for DeepFace
        self._face_queue: queue.Queue = queue.Queue(maxsize=1)

        # Daemon worker thread — runs for the lifetime of the processor
        self._worker = threading.Thread(
            target=self._inference_worker, daemon=True, name="deepface-worker"
        )
        self._worker.start()

    # ── Background DeepFace worker ────────────────────────────────────────
    def _inference_worker(self) -> None:
        """
        Runs in a dedicated daemon thread. Blocks on the queue, runs DeepFace,
        and writes results back to LiveState — never touching the video pipeline.
        """
        from modules.face_module import analyze_frame
        while True:
            bgr = self._face_queue.get()   # blocks until a frame is available
            if bgr is None:
                break                      # sentinel — graceful shutdown
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
                print(f"[MAITRI] Inference worker error: {exc}")

    # ── Per-frame recv callback ───────────────────────────────────────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Called for EVERY incoming video frame by streamlit-webrtc.
        Must return as fast as possible — no blocking ML calls here.
        """
        from modules.eye_module import analyze_eyes

        bgr = frame.to_ndarray(format="bgr24")
        ls  = self.live_state

        with ls.lock:
            ls.frame_count += 1
            fc = ls.frame_count

        # ── Eye tracking — every 2nd frame (~15 FPS) ──────────────────────
        if fc % _EYE_EVERY_N_FRAMES == 0:
            eye = analyze_eyes(bgr, self.eye_state)
            with ls.lock:
                ls.ear            = eye.ear
                ls.ear_baseline   = eye.ear_baseline
                ls.blink_rate     = eye.blink_rate
                ls.fatigue_label  = eye.fatigue_label
                ls.fatigue_strain = eye.fatigue_strain
                ls.eye_quality    = eye.eye_quality
                ls.eye_available  = eye.available

        # ── Non-blocking DeepFace dispatch ────────────────────────────────
        try:
            self._face_queue.put_nowait(bgr)
        except queue.Full:
            pass

        # ── Draw HUD on every frame using latest cached telemetry ─────────
        bgr = _draw_hud(bgr, ls)

        return av.VideoFrame.from_ndarray(bgr, format="bgr24")


def _draw_hud(bgr: np.ndarray, ls: LiveState) -> np.ndarray:
    """Draw lightweight HUD and persistent face bounding box directly on the video frame with consistent scaling."""
    snap = ls.snapshot()
    h, w = bgr.shape[:2]

    scale = max(0.55, min(w, h) / 480.0)
    font_main = 0.72 * scale
    font_sub  = 0.46 * scale
    thick_main = max(1, int(2 * scale))
    thick_sub  = max(1, int(1 * scale))
    pad = int(4 * scale)

    def _draw_pill(text, x, y, font_scale, font_thick, text_color, bg_color=(15, 20, 28)):
        (tw, th), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thick)
        x1 = max(0, x - pad)
        y1 = max(0, y - th - pad)
        x2 = min(w, x + tw + pad)
        y2 = min(h, y + baseline + pad)
        cv2.rectangle(bgr, (x1, y1), (x2, y2), bg_color, -1)
        cv2.putText(bgr, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_color, font_thick, cv2.LINE_AA)
        return tw, th

    box = snap.get("face_box")
    if box:
        bx, by = box.get("x", 0), box.get("y", 0)
        bw, bh = box.get("w", 0), box.get("h", 0)
        if bw > 0 and bh > 0 and bx < w and by < h:
            box_color = (0, 165, 255) if snap["is_blurry"] else (0, 230, 118)
            box_thick = max(1, int(2 * scale))
            cv2.rectangle(bgr, (bx, by), (min(w, bx + bw), min(h, by + bh)), box_color, box_thick)

    emo   = snap["face_emotion"].upper()
    conf  = snap["face_confidence"] * 100
    emo_text = f"{emo}  {conf:.0f}%"
    _draw_pill(emo_text, int(14 * scale), int(30 * scale), font_main, thick_main, (0, 230, 118))

    dip_status = "BLUR GATED" if snap["is_blurry"] else "DIP: CLAHE+GAMMA"
    dip_color  = (0, 165, 255) if snap["is_blurry"] else (0, 230, 118)
    dip_text   = f"{dip_status}  #{snap['frame_count']}"
    (dtw, _), _ = cv2.getTextSize(dip_text, cv2.FONT_HERSHEY_SIMPLEX, font_sub, thick_sub)
    dip_x = max(10, w - dtw - int(16 * scale))
    _draw_pill(dip_text, dip_x, int(28 * scale), font_sub, thick_sub, dip_color)

    eye_label = f"EAR: {snap['ear']:.2f} | BLINK: {snap['blink_rate']:.0f}/min | {snap['fatigue_label']}"
    _draw_pill(eye_label, int(14 * scale), h - int(14 * scale), font_sub, thick_sub, (0, 215, 255))

    return bgr


# ─────────────────────────────────────────────────────────────────────────
# TOP MISSION CONTROL SELECTOR & LIVE TICKING HEADER
# ─────────────────────────────────────────────────────────────────────────
col_ctrl_mission, col_ctrl_crew = st.columns([1.6, 1.2])

with col_ctrl_mission:
    profile_names = list(MISSION_PROFILES.keys())
    current_profile_idx = profile_names.index(st.session_state.mission_profile) if st.session_state.mission_profile in profile_names else 0
    selected_mission = st.selectbox(
        "🚀 Select Active Mission Profile",
        profile_names,
        index=current_profile_idx,
        key="mission_selector",
    )
    if selected_mission != st.session_state.mission_profile:
        st.session_state.mission_profile = selected_mission
        p_data = MISSION_PROFILES[selected_mission]
        st.session_state.dtn_outbox.set_mission_profile(selected_mission, p_data["latency_sec"])
        st.rerun()

with col_ctrl_crew:
    selected_crew = st.selectbox(
        "👨‍🚀 Designated Active Astronaut",
        [
            "Cdr. A. Sharma (Pilot / CMO)",
            "Lt. Cdr. P. Nair (Flight Engineer)",
            "Dr. V. Rao (Mission Specialist)",
        ],
        index=0,
        key="crew_selector",
    )
    st.session_state.crew_member = selected_crew


@st.fragment(run_every=1.0)
def _render_flight_deck_header():
    """Live flight deck banner updating clocks every 1 second without page reload."""
    utc_now = datetime.datetime.utcnow().strftime("%H:%M:%S UTC")
    met_now = _format_met()
    prof_key = st.session_state.mission_profile
    prof_info = MISSION_PROFILES.get(prof_key, list(MISSION_PROFILES.values())[0])
    latency_str = st.session_state.dtn_outbox.format_latency_str()

    st.markdown(f"""
    <div style="background: linear-gradient(90deg, #0b132b, #1c2541); padding: 12px 18px; border-radius: 10px; border: 1px solid #3a506b; margin-bottom: 12px; box-shadow: 0 4px 15px rgba(0,0,0,0.3);">
      <div style="display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;">
        <div>
          <div style="font-size: 1.18rem; font-weight: bold; color: #48cae4; letter-spacing: 0.5px;">
            🛰️ PROJECT MAITRI 2.0 &nbsp;|&nbsp; {prof_key.upper()}
          </div>
          <div style="font-size: 0.76rem; color: #a0aec0; margin-top: 2px;">
            {prof_info['habitat'].upper()} &nbsp;•&nbsp; {prof_info['agency'].upper()} &nbsp;•&nbsp; AUTONOMOUS MEDICAL LIFE SUPPORT SYSTEM (ExMS)
          </div>
        </div>
        <div style="display: flex; gap: 10px; align-items: center; flex-wrap: wrap;">
          <div style="background: #0d1b2a; padding: 4px 10px; border-radius: 6px; border: 1px solid #1e3a8a; text-align: center;">
            <div style="font-size: 0.60rem; color: #93c5fd; text-transform: uppercase;">Mission Elapsed Time</div>
            <div style="font-size: 0.86rem; font-weight: bold; color: #ffffff; font-family: monospace;">{met_now}</div>
          </div>
          <div style="background: #0d1b2a; padding: 4px 10px; border-radius: 6px; border: 1px solid #1e3a8a; text-align: center;">
            <div style="font-size: 0.60rem; color: #93c5fd; text-transform: uppercase;">Live UTC Clock</div>
            <div style="font-size: 0.86rem; font-weight: bold; color: #38bdf8; font-family: monospace;">{utc_now}</div>
          </div>
          <div style="background: #0d1b2a; padding: 4px 10px; border-radius: 6px; border: 1px solid #b45309; text-align: center;">
            <div style="font-size: 0.60rem; color: #fde047; text-transform: uppercase;">Earth Tele-Link Delay</div>
            <div style="font-size: 0.86rem; font-weight: bold; color: #f59e0b; font-family: monospace;">{latency_str}</div>
          </div>
          <div style="background: #0d1b2a; padding: 4px 10px; border-radius: 6px; border: 1px solid #047857; text-align: center;">
            <div style="font-size: 0.60rem; color: #6ee7b7; text-transform: uppercase;">Active Astronaut</div>
            <div style="font-size: 0.86rem; font-weight: bold; color: #10b981;">{st.session_state.crew_member}</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)


_render_flight_deck_header()

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
        e1, e2, e3 = st.columns(3)
        e1.metric("EAR", f"{ls_snap['ear']:.3f}")
        e2.metric("Baseline", f"{ls_snap.get('ear_baseline', 0.28):.3f}")
        e3.metric("Blink Rate", f"{ls_snap['blink_rate']:.1f} /min")
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
    vitals_status: str,
    hypoxia_risk: str,
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

    # ── Astronaut Multi-Factor Watchdog (NASA HRP Protocols) ───────────────
    watchdog = st.session_state.watchdog
    wd_status = watchdog.update(
        face_box         = ls_snap.get("face_box") if is_playing else None,
        ear              = ls_snap.get("ear", 0.28),
        blink_rate       = ls_snap.get("blink_rate", 16.0),
        dominant_emotion = fusion.dominant_emotion,
        spo2             = spo2,
        heart_rate       = heart_rate,
    )

    # ── Watchdog Alert Banners & Hands-Free Recovery ──────────────────────
    if wd_status.recovered_via_motion:
        st.markdown(
            "<div style='background:#064e3b;border:1px solid #10b981;padding:9px 14px;border-radius:6px;margin-bottom:10px;color:#d1fae5;font-size:0.86rem;'>"
            "✅ <b>Consciousness Verified via Movement (Hands-Free Recovery):</b> Astronaut motion confirmed after debounce safety window. Nominal status restored."
            "</div>",
            unsafe_allow_html=True,
        )

    if wd_status.state == WatchdogState.CHECK_IN:
        st.markdown(
            f"""
            <div style='background:#78350f;border:2px solid #f59e0b;padding:12px 16px;border-radius:8px;margin-bottom:12px;color:#fef3c7;'>
              <div style='display:flex;justify-content:space-between;align-items:center;'>
                <b>⚠️ MULTI-FACTOR CREW INACTIVITY DETECTED</b>
                <span style='background:#b45309;padding:2px 8px;border-radius:4px;font-family:monospace;font-size:0.85rem;'>Escalation in: {wd_status.countdown_remaining:.0f}s</span>
              </div>
              <div style='margin-top:6px;font-size:0.88rem;'>
                <b>Status:</b> {wd_status.alert_message}<br>
                <span style='color:#fde68a;'>Stillness: {wd_status.motionless_seconds:.0f}s &nbsp;|&nbsp; Affect Freeze: {wd_status.affect_frozen_seconds:.0f}s &nbsp;|&nbsp; Debounce: {wd_status.debounce_remaining:.0f}s</span>
              </div>
              <div style='font-size:0.8rem;color:#fed7aa;margin-top:6px;'>
                💡 <i>Hands-free recovery: Moving head or active blinking after 5s debounce will automatically restore nominal state.</i>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        c_ack, _ = st.columns([1.8, 3])
        with c_ack:
            if st.button("✅ I AM CONSCIOUS (MANUAL OVERRIDE)", key="btn_ack_conscious", type="primary", use_container_width=True):
                watchdog.acknowledge_conscious()
                st.rerun()

    elif wd_status.state == WatchdogState.INCAPACITATED:
        st.markdown(
            "<div style='background:#7f1d1d;border:2px solid #ef4444;padding:14px 18px;border-radius:8px;margin-bottom:12px;color:#fee2e2;animation:pulse_alert 1.5s infinite;'>"
            "<b>🚨 EMERGENCY ALERT: CREW INCAPACITATION CONFIRMED.</b><br>"
            "Multi-factor concurrence threshold breached. Medical Dossier auto-compiled & Priority-1 emergency packet queued for Ground Station."
            "</div>",
            unsafe_allow_html=True,
        )
        c_ack, _ = st.columns([1.8, 3])
        with c_ack:
            if st.button("🔄 RESET INCAPACITATION PROTOCOL", key="btn_reset_incap", type="secondary", use_container_width=True):
                watchdog.acknowledge_conscious()
                st.rerun()

    # ── Autonomous Emergency Dossier Compilation & DTN Dispatch ────────────
    if wd_status.incident_triggered:
        dossier = generate_medical_dossier(
            incident_type="CRITICAL_INCAPACITATION_ALERT",
            crew_member=st.session_state.crew_member,
            met_str=_format_met(),
            vitals_dict={
                "heart_rate": heart_rate,
                "temperature": skin_temp,
                "spo2": spo2,
                "psi_score": 6.8,
                "psi_category": "High Strain",
                "hypoxia_risk": hypoxia_risk,
            },
            eye_dict={
                "ear": ls_snap["ear"],
                "blink_rate": ls_snap["blink_rate"],
                "fatigue_label": ls_snap["fatigue_label"],
            },
            fusion_dict={
                "dominant_emotion": fusion.dominant_emotion,
                "stress_pct": fusion.stress_pct,
            },
            watchdog_status={
                "motionless_seconds": wd_status.motionless_seconds,
                "affect_frozen_seconds": wd_status.affect_frozen_seconds,
            },
            ground_delay_str=st.session_state.dtn_outbox.format_latency_str(),
            mission_name=st.session_state.mission_profile,
        )
        st.session_state.last_dossier = dossier
        st.session_state.dtn_outbox.dispatch_incident_packet(
            incident_type="INCAPACITATION_INCIDENT",
            met_str=_format_met(),
            summary=f"Unresponsive astronaut confirmed. Stillness: {wd_status.motionless_seconds:.0f}s. Emergency life support notified.",
            details={"stress_pct": fusion.stress_pct, "spo2": spo2, "heart_rate": heart_rate, "mission": st.session_state.mission_profile},
            priority="CRITICAL-1 (EMERGENCY)",
        )

    # Check for Acute Hypoxia Emergency Dispatch
    if spo2 < 89.0:
        if not st.session_state.get("hypoxia_dossier_sent", False):
            st.session_state.hypoxia_dossier_sent = True
            hypoxia_dossier = generate_medical_dossier(
                incident_type="CABIN_HYPOXIA_DECOMPRESSION",
                crew_member=st.session_state.crew_member,
                met_str=_format_met(),
                vitals_dict={
                    "heart_rate": heart_rate,
                    "temperature": skin_temp,
                    "spo2": spo2,
                    "psi_score": 5.4,
                    "psi_category": "Moderate Strain",
                    "hypoxia_risk": hypoxia_risk,
                },
                eye_dict={
                    "ear": ls_snap["ear"],
                    "blink_rate": ls_snap["blink_rate"],
                    "fatigue_label": ls_snap["fatigue_label"],
                },
                fusion_dict={
                    "dominant_emotion": fusion.dominant_emotion,
                    "stress_pct": fusion.stress_pct,
                },
                watchdog_status={
                    "motionless_seconds": wd_status.motionless_seconds,
                    "affect_frozen_seconds": wd_status.affect_frozen_seconds,
                },
                ground_delay_str=st.session_state.dtn_outbox.format_latency_str(),
                mission_name=st.session_state.mission_profile,
            )
            st.session_state.last_dossier = hypoxia_dossier
            st.session_state.dtn_outbox.dispatch_incident_packet(
                incident_type="HYPOXIA_EMERGENCY",
                met_str=_format_met(),
                summary=f"Acute blood oxygen desaturation (SpO2: {spo2:.1f}%). Automated life support mask advisory issued.",
                details={"spo2": spo2, "heart_rate": heart_rate, "mission": st.session_state.mission_profile},
                priority="CRITICAL-1 (EMERGENCY)",
            )
    elif spo2 >= 92.0:
        st.session_state.hypoxia_dossier_sent = False

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
        alert = get_alert(
            fusion.stress_pct,
            fusion.dominant_emotion,
            ls_snap.get("fatigue_label", "Normal"),
            vitals_status=vitals_status,
            hypoxia_risk=hypoxia_risk,
        )
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
            if any(k in alert.technique for k in ("Breathing", "4-7-8", "Box", "counts")):
                st.markdown(_render_breathing_circle(alert.technique), unsafe_allow_html=True)
        else:
            st.markdown(
                "<div style='background:#1a3a1a;padding:14px;border-radius:8px;"
                "border:1px solid #2ecc71;color:#2ecc71;'>"
                "✅ No intervention required. Astronaut is performing optimally.</div>",
                unsafe_allow_html=True,
            )

    # ── Local Onboard Logging (Bandwidth Conservation) ─────────────────────
    st.markdown("---")
    log_col, _ = st.columns([1, 3])
    with log_col:
        if st.button("💾 Log Reading to Spacecraft Database", use_container_width=True):
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
            st.success("✅ Logged locally to spacecraft database (maitri_logs.db).")

    if is_playing:
        st.caption("⚡ Real-time Telemetry: Live assessment & multi-factor watchdog updating dynamically (1 Hz).")


# ─────────────────────────────────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────────────────────────────────
tab_live, tab_logs, tab_dossier = st.tabs([
    "🔴 Live Monitoring & Simulator",
    "📊 Spacecraft Logs & Trends",
    "🩺 Flight Surgeon Dossier & Ground DTN",
])

# ═════════════════════════════════════════════════════════════════════════
# TAB 1 — LIVE MONITORING & SIMULATOR
# ═════════════════════════════════════════════════════════════════════════
with tab_live:

    # ── 1-Click Presentation Scenario Simulator Toolbar ────────────────────
    with st.expander("🧪 Mission Simulation & Presentation Presets (1-Click Demo Controller)", expanded=True):
        st.caption("Instantly inject aerospace physiological and operational edge cases for presentation demonstration.")
        sim_c1, sim_c2, sim_c3, sim_c4, sim_c5, sim_c6 = st.columns(6)
        with sim_c1:
            if st.button("🟢 Nominal Orbit", use_container_width=True, help="Resting nominal vitals: HR 72, SpO2 98%, 36.6°C"):
                st.session_state.slider_hr = 72
                st.session_state.slider_temp = 36.6
                st.session_state.slider_spo2 = 98
                st.session_state.watchdog.acknowledge_conscious()
                st.rerun()
        with sim_c2:
            if st.button("🏃 T2 EVA Exertion", use_container_width=True, help="Physical workout: HR 135, Temp 37.3°C, SpO2 97% (Strain without false panic)"):
                st.session_state.slider_hr = 135
                st.session_state.slider_temp = 37.3
                st.session_state.slider_spo2 = 97
                st.session_state.watchdog.acknowledge_conscious()
                st.rerun()
        with sim_c3:
            if st.button("😴 Microgravity Drowsiness", use_container_width=True, help="Hypo-vigilance: Low blinks, HR 60, SpO2 96%"):
                st.session_state.slider_hr = 60
                st.session_state.slider_temp = 36.4
                st.session_state.slider_spo2 = 96
                st.session_state.watchdog.acknowledge_conscious()
                st.rerun()
        with sim_c4:
            if st.button("🚨 Hypoxia (SpO2 87%)", use_container_width=True, help="Cabin decompression/CO2 pocket: SpO2 87%, HR 118 -> Auto-dossier + DTN dispatch"):
                st.session_state.slider_hr = 118
                st.session_state.slider_temp = 36.8
                st.session_state.slider_spo2 = 87
                st.rerun()
        with sim_c5:
            if st.button("⚡ Inactivity Check-In", use_container_width=True, help="Trigger watchdog check-in: Demonstrates 5s debounce and hands-free motion recovery"):
                st.session_state.watchdog.trigger_checkin_simulation()
                st.rerun()
        with sim_c6:
            if st.button("🔄 Reset to Live", use_container_width=True, help="Reset to standard baseline"):
                st.session_state.slider_hr = 75
                st.session_state.slider_temp = 36.6
                st.session_state.slider_spo2 = 98
                st.session_state.watchdog.acknowledge_conscious()
                st.rerun()

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
        st.caption("Background ML pipeline · MediaPipe Eye Tracking (~15 FPS)")

        rtc_config = RTCConfiguration(
            {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
        )

        _live_state = st.session_state.live_state
        _eye_state  = st.session_state.eye_state

        def _processor_factory():
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
        st.caption("Telemetry sensors or simulator controls")

        heart_rate = st.slider("❤️  Heart Rate (BPM)",      50,  160, key="slider_hr")
        skin_temp  = st.slider("🌡️  Skin Temperature (°C)", 35.0, 40.0, step=0.1, key="slider_temp")
        spo2       = st.slider("🫁  Blood Oxygen (SpO₂ %)", 85,  100, key="slider_spo2")

        vitals = compute_vitals_strain(float(heart_rate), float(skin_temp), float(spo2))

        v1, v2, v3, v4 = st.columns(4)
        v1.metric("HR Strain",   f"{vitals.hr_strain*100:.0f}%")
        v2.metric("Temp Strain", f"{vitals.temp_strain*100:.0f}%")
        v3.metric("SpO₂ Strain", f"{vitals.spo2_strain*100:.0f}%")
        v4.metric("Moran PSI",   f"{vitals.psi_score} / 10")

        vitals_color = {"Normal": "#2ecc71", "Elevated": "#e67e22", "Critical": "#e74c3c"}.get(
            vitals.status, "#aaa"
        )
        hypoxia_badge = f" &nbsp;|&nbsp; 🫁 <b style='color:#e74c3c;'>{vitals.hypoxia_risk}</b>" if vitals.hypoxia_risk != "Nominal" else ""
        st.markdown(
            f"<span class='stress-badge' style='background:{vitals_color};'>"
            f"Vitals: {vitals.status} ({vitals.psi_category})</span>{hypoxia_badge}",
            unsafe_allow_html=True,
        )

        st.markdown("---")
        st.subheader("👁️ Eye Tracking (Live)")
        _render_eye_panel(ctx.state.playing)

    # ── ROW 2 & ROW 3: Live Assessment & Autonomous Psychological Support ──
    st.markdown("---")
    st.subheader("🧠 Live Multimodal Stress Assessment")
    _render_live_assessment(
        heart_rate=float(heart_rate),
        skin_temp=float(skin_temp),
        spo2=float(spo2),
        vitals_strain=vitals.vitals_strain,
        vitals_status=vitals.status,
        hypoxia_risk=vitals.hypoxia_risk,
        is_playing=ctx.state.playing,
    )

# ═════════════════════════════════════════════════════════════════════════
# TAB 2 — SPACECRAFT LOGS & TRENDS
# ═════════════════════════════════════════════════════════════════════════
with tab_logs:
    st.subheader("📋 Onboard Spacecraft Telemetry Archive")
    st.caption("Local high-frequency telemetry stored safely on spacecraft storage (maitri_logs.db).")

    log_df = get_logs()
    if log_df.empty:
        st.info("No telemetry logs yet. Use '💾 Log Reading to Spacecraft Database' in the Live tab.")
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
        st.markdown("**Full Onboard Telemetry Log**")
        st.dataframe(log_df, use_container_width=True, height=380)

        if "fused_emotion" in log_df.columns:
            st.markdown("**Emotion Distribution**")
            st.bar_chart(log_df["fused_emotion"].value_counts())

# ═════════════════════════════════════════════════════════════════════════
# TAB 3 — FLIGHT SURGEON DOSSIER & GROUND DTN
# ═════════════════════════════════════════════════════════════════════════
with tab_dossier:
    st.subheader("🩺 Autonomous Flight Surgeon Dossier & Ground DTN Tele-Link")
    st.caption("Deep-space delay-tolerant medical incident compilation and ground station downlink telemetry.")

    col_dos_actions, col_dos_sim = st.columns([1.4, 1.0], gap="medium")
    with col_dos_actions:
        if st.button("📄 Compile Active Telemetry Incident Dossier", type="primary", use_container_width=True):
            ls_snap = st.session_state.live_state.snapshot()
            dossier_text = generate_medical_dossier(
                incident_type="AUTONOMOUS_CLINICAL_AUDIT",
                crew_member=st.session_state.crew_member,
                met_str=_format_met(),
                vitals_dict={
                    "heart_rate": float(st.session_state.slider_hr),
                    "temperature": float(st.session_state.slider_temp),
                    "spo2": float(st.session_state.slider_spo2),
                    "psi_score": 1.2,
                    "psi_category": "No Strain",
                    "hypoxia_risk": "Nominal",
                },
                eye_dict={
                    "ear": ls_snap["ear"],
                    "blink_rate": ls_snap["blink_rate"],
                    "fatigue_label": ls_snap["fatigue_label"],
                },
                fusion_dict={
                    "dominant_emotion": ls_snap["face_emotion"],
                    "stress_pct": 8.0,
                },
                watchdog_status={"motionless_seconds": 0.0, "affect_frozen_seconds": 0.0},
                ground_delay_str=st.session_state.dtn_outbox.format_latency_str(),
                mission_name=st.session_state.mission_profile,
            )
            st.session_state.last_dossier = dossier_text
            st.session_state.dtn_outbox.dispatch_incident_packet(
                incident_type="CLINICAL_AUDIT_DOSSIER",
                met_str=_format_met(),
                summary=f"Flight surgeon clinical audit dossier compiled for {st.session_state.crew_member}.",
                details={"crew_member": st.session_state.crew_member, "mission": st.session_state.mission_profile},
                priority="CLINICAL-AUDIT",
            )
            st.success("✅ Medical Dossier compiled & queued for Ground DTN transmission.")

    with col_dos_sim:
        if st.button("⚡ Simulate Watchdog Check-In (Demo Trigger)", use_container_width=True):
            st.session_state.watchdog.trigger_checkin_simulation()
            st.rerun()

    st.markdown("---")

    # Display compiled dossier if available
    if st.session_state.last_dossier:
        st.markdown("### 📄 Compiled Clinical Telemetry Dossier")
        st.download_button(
            label="📥 Download Flight Surgeon Incident Dossier (.md)",
            data=st.session_state.last_dossier,
            file_name=f"MAITRI_Medical_Dossier_{int(time.time())}.md",
            mime="text/markdown",
            use_container_width=True,
        )
        with st.expander("👁️ View Formatted NASA/ISRO Medical Dossier Preview", expanded=True):
            st.markdown(st.session_state.last_dossier)
    else:
        st.info("ℹ️ No incident dossier generated yet. Click '📄 Compile Active Telemetry Incident Dossier' or trigger an incident to generate one.")

    st.markdown("---")
    st.subheader("📡 Delay-Tolerant Network (DTN) Ground Telemetry Outbox")
    st.markdown(
        f"""
        <div style='background:#1e293b;padding:8px 12px;border-radius:6px;font-size:0.83rem;margin-bottom:12px;border-left:3px solid #38bdf8;'>
          🛰️ <b>Bandwidth Conservation Protocol Active:</b> Routine telemetry is stored locally on the spacecraft. 
          The ground outbox is strictly reserved for <b>Priority-1 Emergencies</b> and Flight Surgeon Clinical Audits. 
          Current Link Latency: <b>{st.session_state.dtn_outbox.format_latency_str()}</b>
        </div>
        """,
        unsafe_allow_html=True,
    )

    packets = st.session_state.dtn_outbox.get_packets()
    if not packets:
        st.info("No packets queued. DTN outbox transmits only upon confirmed emergencies or clinical audits.")
    else:
        for pkt in packets:
            border_col = "#ef4444" if "CRITICAL" in pkt.priority else "#3b82f6"
            st.markdown(
                f"<div style='background:#0f172a;padding:10px 14px;border-radius:8px;border-left:4px solid {border_col};margin-bottom:8px;border-top:1px solid #1e293b;border-right:1px solid #1e293b;border-bottom:1px solid #1e293b;'>"
                f"<div style='display:flex;justify-content:space-between;align-items:center;'>"
                f"<span style='font-family:monospace;font-weight:bold;color:#38bdf8;'>{pkt.packet_id} &nbsp;·&nbsp; {pkt.mission_profile}</span>"
                f"<span style='font-size:0.75rem;background:{'#7f1d1d' if 'CRITICAL' in pkt.priority else '#1e3a8a'};padding:2px 8px;border-radius:4px;color:white;'>{pkt.priority}</span>"
                f"<span style='font-family:monospace;font-size:0.8rem;color:#94a3b8;'>{pkt.met_timestamp}</span>"
                f"</div>"
                f"<div style='font-size:0.85rem;color:#e2e8f0;margin:6px 0;'><b>Type:</b> {pkt.packet_type} &nbsp;|&nbsp; <b>Status:</b> <span style='color:#34d399;'>{pkt.status}</span></div>"
                f"<div style='font-size:0.8rem;color:#94a3b8;'>{pkt.payload_summary}</div>"
                f"</div>",
                unsafe_allow_html=True,
            )