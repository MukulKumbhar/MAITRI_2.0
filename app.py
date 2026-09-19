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
from typing import Optional
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
    import streamlit_webrtc.component as _st_webrtc_comp
    _WEBRTC_OK = True

    # ── Rock-Solid WebRTC Lifecycle & Worker Hardening ─────────────────────────
    # 0. Suppress aioice STUN retry crashes on dead/closed UDP transports
    try:
        import aioice.ice
        import aioice.stun
        _orig_stun_retry = getattr(aioice.stun.Transaction, "_Transaction__retry", None)
        if _orig_stun_retry:
            def _safe_stun_retry(self):
                try:
                    fut = getattr(self, "_Transaction__future", None)
                    if fut and fut.done():
                        return
                    proto = getattr(self, "_Transaction__protocol", None)
                    if proto:
                        tr = getattr(proto, "transport", None)
                        if tr is None or getattr(tr, "_sock", None) is None or (hasattr(tr, "is_closing") and tr.is_closing()):
                            return
                    _orig_stun_retry(self)
                except Exception:
                    pass
            aioice.stun.Transaction._Transaction__retry = _safe_stun_retry

        _orig_ice_send_stun = getattr(aioice.ice.StunProtocol, "send_stun", None)
        if _orig_ice_send_stun:
            def _safe_ice_send_stun(self, message, addr):
                if self.transport is None or getattr(self.transport, "_sock", None) is None or (hasattr(self.transport, "is_closing") and self.transport.is_closing()):
                    return
                try:
                    _orig_ice_send_stun(self, message, addr)
                except Exception:
                    pass
            aioice.ice.StunProtocol.send_stun = _safe_ice_send_stun
    except Exception:
        pass

    # 1. Prevent aiortc from aborting stream on transient ICE "disconnected" / "failed" states
    #    (e.g., 5s STUN consent check timeouts under CPU load or firewall NAT).
    try:
        from aiortc import RTCPeerConnection
        _orig_listens_to = RTCPeerConnection.listens_to
        def _hardened_listens_to(self, event: str):
            decorator = _orig_listens_to(self, event)
            if event == "iceconnectionstatechange":
                def wrapped_decorator(handler):
                    async def wrapped_handler(*args, **kwargs):
                        state = getattr(self, "iceConnectionState", None)
                        if state in ("disconnected", "failed"):
                            return  # Suppress transient ICE disconnects; keep stream running
                        return await handler(*args, **kwargs)
                    return decorator(wrapped_handler)
                return wrapped_decorator
            return decorator
        RTCPeerConnection.listens_to = _hardened_listens_to
    except Exception:
        pass

    # 2. Prevent worker from being garbage-collected during active sessions
    _orig_set_worker = _st_webrtc_comp.WebRtcStreamerContext._set_worker
    def _hardened_set_worker(self, worker):
        self._strong_worker = worker  # Prevent GC reaping
        _orig_set_worker(self, worker)
    _st_webrtc_comp.WebRtcStreamerContext._set_worker = _hardened_set_worker

    # 3. Guard _reset_context from killing an active connected peer connection
    _orig_reset_context = _st_webrtc_comp._reset_context
    def _hardened_reset_context(context):
        worker = context._get_worker() if hasattr(context, "_get_worker") else None
        pc = getattr(worker, "pc", None)
        if (
            worker
            and pc
            and pc.connectionState == "connected"
            and not getattr(context, "_user_stopped", False)
        ):
            context._set_state(_st_webrtc_comp.WebRtcStreamerState(playing=True, signalling=False))
            return
        _orig_reset_context(context)
    _st_webrtc_comp._reset_context = _hardened_reset_context

    # 4. Prevent Streamlit reruns / fragments from wiping the WebRTC component value
    def _hardened_restore_snapshot(context, component_value):
        session_info = _st_webrtc_comp.get_this_session_info()
        run_count = _st_webrtc_comp.get_script_run_count(session_info) if session_info else 0
        worker = context._get_worker() if hasattr(context, "_get_worker") else None
        pc = getattr(worker, "pc", None)
        pc_connected = (
            worker is not None
            and pc is not None
            and getattr(pc, "connectionState", None) in ("connected", "connecting")
        )

        if component_value is not None:
            if component_value.get("playing") is False:
                context._user_stopped = True
                context._last_valid_component_value = None
            elif component_value.get("playing") is True:
                context._user_stopped = False
                context._last_valid_component_value = component_value

        if (component_value is None or not component_value.get("playing")) and not getattr(context, "_user_stopped", False):
            cached = getattr(context, "_last_valid_component_value", None)
            if cached is not None and (pc_connected or getattr(context.state, "playing", False)):
                component_value = cached

        if run_count is not None:
            context._component_value_snapshot = _st_webrtc_comp.ComponentValueSnapshot(
                component_value=component_value, run_count=run_count
            )
        return component_value
    _st_webrtc_comp._restore_snapshot_if_needed = _hardened_restore_snapshot

    # 5. Prevent orphan context cleanup from killing an active streaming session
    _orig_get_or_create_context = _st_webrtc_comp._get_or_create_context
    def _hardened_get_or_create_context(key: str):
        if key in st.session_state:
            ctx = st.session_state[key]
            worker = ctx._get_worker() if hasattr(ctx, "_get_worker") else None
            pc = getattr(worker, "pc", None)
            is_active = (
                (worker is not None and pc is not None and getattr(pc, "connectionState", None) not in ("closed", "failed"))
                or getattr(ctx.state, "playing", False)
                or getattr(ctx.state, "signalling", False)
            )
            if is_active:
                sinfo = _st_webrtc_comp.get_this_session_info()
                rc = _st_webrtc_comp.get_script_run_count(sinfo) if sinfo else None
                if rc is not None:
                    ctx._last_rendered_run_count = rc
        return _orig_get_or_create_context(key)
    _st_webrtc_comp._get_or_create_context = _hardened_get_or_create_context

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

# ── Standard Emotion Color Map (Matches HUD and Charts) ───────────────────
EMO_COLORS = {
    "angry": "#e74c3c", "disgust": "#27ae60", "fear": "#9b59b6",
    "happy": "#f1c40f", "neutral": "#95a5a6", "sad": "#3498db",
    "surprise": "#e67e22",
}

# ─────────────────────────────────────────────────────────────────────────
# SESSION STATE INIT & THREAD-SAFE REFERENCES
# ─────────────────────────────────────────────────────────────────────────
if "live_state"   not in st.session_state:
    st.session_state.live_state   = LiveState()
if "fusion_state" not in st.session_state:
    st.session_state.fusion_state = FusionState()
if "eye_state"    not in st.session_state:
    from modules.eye_module import EyeSessionState
    st.session_state.eye_state    = EyeSessionState()
if "voice_detector" not in st.session_state:
    from modules.voice_module import get_or_create_voice_detector
    st.session_state.voice_detector = get_or_create_voice_detector(st.session_state.live_state)

# Global references safe for background threads (aiortc workers cannot access st.session_state)
_ACTIVE_LIVE_STATE = st.session_state.live_state
_ACTIVE_EYE_STATE  = st.session_state.eye_state

# Eagerly pre-warm neural networks on page startup to eliminate the 1.7s initial video freeze
if "models_warmed" not in st.session_state:
    try:
        from modules.face_module import process_unified_frame, _get_onnx_session
        from modules.voice_module import _get_voice_onnx_session
        _get_onnx_session()
        _get_voice_onnx_session()
        _dummy = np.zeros((480, 640, 3), dtype=np.uint8)
        process_unified_frame(_dummy, st.session_state.eye_state)
        st.session_state.models_warmed = True
    except Exception:
        pass

# ─────────────────────────────────────────────────────────────────────────
# WEBRTC VIDEO PROCESSOR
# ─────────────────────────────────────────────────────────────────────────
_EYE_EVERY_N_FRAMES = 3   # kept for reference — eye worker throttles internally

class MAITRIVideoProcessor:
    """
    Processes WebRTC video frames with zero pipeline blocking.
    State is passed via constructor (safe for worker threads).

    High-Performance Unified Architecture (<1ms recv latency, ~30 FPS):
    - Unified background worker runs MediaPipe FaceLandmarker ONCE per cycle.
    - MediaPipe extracts both FACS blendshapes (AU12 smile, AU6 cheek squint)
      and 478 face landmarks for EAR and blink tracking simultaneously.
    - Deep EfficientNet ONNX inference runs on centered isotropic square face crops.
    - recv() NEVER blocks on ML: copy frame -> try_enqueue -> draw aerospace HUD -> return.
    - Single-slot queue drops stale frames under heavy system load to prevent any latency buildup.
    """

    def __init__(self, live_state: LiveState, eye_state):
        self.live_state = live_state
        self.eye_state  = eye_state
        self._running   = True

        # Single-slot queue — drops frames when background worker is busy (zero lag)
        self._inference_queue: queue.Queue = queue.Queue(maxsize=1)

        # Motion-aware adaptive reticle tracking state
        self._current_display_box: Optional[dict] = None
        self._last_raw_box: Optional[dict] = None
        self._vel_x: float = 0.0
        self._vel_y: float = 0.0
        self._vel_w: float = 0.0
        self._vel_h: float = 0.0

        # Unified background worker
        self._worker = threading.Thread(
            target=self._unified_inference_worker, daemon=True, name="maitri-unified-worker"
        )
        self._worker.start()

    def stop(self) -> None:
        self._running = False
        try:
            self._inference_queue.put_nowait(None)
        except Exception:
            pass

    def __del__(self) -> None:
        self.stop()

    # ── Background Unified Worker ─────────────────────────────────────────
    def _unified_inference_worker(self) -> None:
        from modules.face_module import process_unified_frame
        while self._running:
            try:
                bgr = self._inference_queue.get(timeout=0.5)
            except queue.Empty:
                continue

            if bgr is None or not self._running:
                break
            try:
                face_res, eye_res = process_unified_frame(bgr, self.eye_state)
                ls = self.live_state
                with ls.lock:
                    if face_res is not None:
                        ls.face_emotion    = face_res.dominant_emotion
                        ls.emotion_probs   = {
                            k: float(v) / 100.0 for k, v in face_res.emotion_probs.items()
                        }
                        ls.face_confidence = face_res.face_confidence
                        ls.face_quality    = face_res.face_quality
                        ls.face_error      = face_res.error
                        ls.blur_score      = face_res.blur_score
                        ls.is_blurry       = face_res.is_blurry
                        ls.face_box        = face_res.face_box

                    if eye_res is not None:
                        ls.ear            = eye_res.ear
                        ls.blink_rate     = eye_res.blink_rate
                        ls.fatigue_label  = eye_res.fatigue_label
                        ls.fatigue_strain = eye_res.fatigue_strain
                        ls.eye_quality    = eye_res.eye_quality
                        ls.eye_available  = eye_res.available
            except Exception as exc:
                print(f"[MAITRI] Unified worker error: {exc}")

    # ── Per-frame recv callback — ZERO blocking (<1ms execution) ──────────
    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        """
        Called for EVERY incoming video frame by streamlit-webrtc.
        Returns immediately — all ML inference runs asynchronously in background worker.
        """
        try:
            bgr = frame.to_ndarray(format="bgr24")
            ls  = self.live_state

            ls.frame_count += 1

            # Non-blocking single-copy dispatch to unified worker
            frame_copy = bgr.copy()
            try:
                self._inference_queue.put_nowait(frame_copy)
            except queue.Full:
                try:
                    self._inference_queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._inference_queue.put_nowait(frame_copy)
                except queue.Full:
                    pass

            # Motion-adaptive reticle tracking & lead-prediction
            snap = ls.snapshot_face_nonblocking() if hasattr(ls, "snapshot_face_nonblocking") else ls.snapshot_face()
            raw_box = snap.get("face_box")

            display_box = None
            if raw_box is not None:
                rx, ry = float(raw_box["x"]), float(raw_box["y"])
                rw, rh = float(raw_box["w"]), float(raw_box["h"])

                if self._current_display_box is None:
                    self._current_display_box = {
                        "x": rx, "y": ry, "w": rw, "h": rh,
                    }
                    self._vel_x = 0.0
                    self._vel_y = 0.0
                    self._vel_w = 0.0
                    self._vel_h = 0.0
                elif raw_box != self._last_raw_box:
                    # New detection from worker: calculate instantaneous velocity
                    lx, ly = float(self._last_raw_box["x"]), float(self._last_raw_box["y"])
                    lw, lh = float(self._last_raw_box["w"]), float(self._last_raw_box["h"])
                    inst_vx = rx - lx
                    inst_vy = ry - ly
                    inst_vw = rw - lw
                    inst_vh = rh - lh

                    # Velocity filter
                    self._vel_x = 0.60 * self._vel_x + 0.40 * inst_vx
                    self._vel_y = 0.60 * self._vel_y + 0.40 * inst_vy
                    self._vel_w = 0.60 * self._vel_w + 0.40 * inst_vw
                    self._vel_h = 0.60 * self._vel_h + 0.40 * inst_vh

                self._last_raw_box = raw_box

                # Dynamic responsiveness: fast tracking during motion, zero jitter when resting
                speed = abs(self._vel_x) + abs(self._vel_y)
                alpha = min(0.90, max(0.45, 0.45 + speed * 0.02))

                # Lead compensation for 1-frame asynchronous pipeline delay
                target_x = rx + self._vel_x * 0.45
                target_y = ry + self._vel_y * 0.45
                target_w = rw + self._vel_w * 0.20
                target_h = rh + self._vel_h * 0.20

                cb = self._current_display_box
                cb["x"] = alpha * target_x + (1.0 - alpha) * cb["x"]
                cb["y"] = alpha * target_y + (1.0 - alpha) * cb["y"]
                cb["w"] = alpha * target_w + (1.0 - alpha) * cb["w"]
                cb["h"] = alpha * target_h + (1.0 - alpha) * cb["h"]

                # Friction decay between background updates
                self._vel_x *= 0.88
                self._vel_y *= 0.88
                self._vel_w *= 0.88
                self._vel_h *= 0.88

                display_box = {
                    "x": int(round(cb["x"])),
                    "y": int(round(cb["y"])),
                    "w": int(round(cb["w"])),
                    "h": int(round(cb["h"])),
                }
            else:
                self._current_display_box = None
                self._last_raw_box = None

            # Draw sleek aerospace glass HUD using latest cached telemetry and smooth tracked box
            try:
                bgr = _draw_aerospace_hud(bgr, ls, display_box=display_box)
            except Exception:
                pass

            return av.VideoFrame.from_ndarray(bgr, format="bgr24")
        except Exception:
            return frame


def _make_video_processor() -> MAITRIVideoProcessor:
    """
    Thread-safe factory called by aiortc background worker threads.
    Uses module-level references because st.session_state is not accessible in worker threads.
    """
    global _ACTIVE_LIVE_STATE, _ACTIVE_EYE_STATE
    ls = _ACTIVE_LIVE_STATE
    es = _ACTIVE_EYE_STATE
    if ls is None:
        try:
            ls = st.session_state.live_state
        except Exception:
            ls = LiveState()
        _ACTIVE_LIVE_STATE = ls
    if es is None:
        try:
            es = st.session_state.eye_state
        except Exception:
            from modules.eye_module import EyeSessionState
            es = EyeSessionState()
        _ACTIVE_EYE_STATE = es
    return MAITRIVideoProcessor(ls, es)


_HUD_BG_CACHE = {}
def _get_hud_bg_slice(slice_h: int, slice_w: int) -> np.ndarray:
    key = (slice_h, slice_w)
    arr = _HUD_BG_CACHE.get(key)
    if arr is None or arr.shape != (slice_h, slice_w, 3):
        arr = np.full((slice_h, slice_w, 3), (12, 16, 24), dtype=np.uint8)
        _HUD_BG_CACHE[key] = arr
    return arr


def _draw_aerospace_hud(bgr: np.ndarray, ls: LiveState, display_box: Optional[dict] = None) -> np.ndarray:
    """
    Sleek, high-contrast semi-transparent aerospace glass HUD:
    - Slice-based in-place alpha blending (no full frame copy, 12x lower rendering overhead)
    - Top banner: System status, emotion badge with color indicator, confidence %, passive clarity
    - Bottom banner: Eye EAR, blink rate, fatigue state, mission status
    - Face reticle: Corner-bracket tactical targeting reticle with zero collision
    """
    snap = ls.snapshot_face_nonblocking() if hasattr(ls, "snapshot_face_nonblocking") else ls.snapshot_face()
    h, w = bgr.shape[:2]

    # 1. Semi-transparent top aerospace banner slice (42px)
    top_h = min(42, h)
    top_slice = bgr[0:top_h, 0:w]
    bg_top = _get_hud_bg_slice(top_h, w)
    cv2.addWeighted(bg_top, 0.65, top_slice, 0.35, 0, top_slice)
    cv2.line(bgr, (0, top_h), (w, top_h), (60, 80, 110), 1)

    # 2. Semi-transparent bottom aerospace banner slice (36px)
    bot_h = min(36, h)
    y_bot = max(0, h - bot_h)
    bot_slice = bgr[y_bot:h, 0:w]
    bg_bot = _get_hud_bg_slice(bot_h, w)
    cv2.addWeighted(bg_bot, 0.65, bot_slice, 0.35, 0, bot_slice)
    cv2.line(bgr, (0, y_bot), (w, y_bot), (60, 80, 110), 1)

    # 3. Corner-bracket face reticle
    box = display_box if display_box is not None else snap.get("face_box")
    emo = snap.get("face_emotion", "neutral").upper()
    conf = snap.get("face_confidence", 0.0) * 100.0
    is_blurry = snap.get("is_blurry", False)

    EMO_HUD_COLORS = {
        "HAPPY":     (0,   215, 255),  # Gold/Yellow
        "NEUTRAL":   (210, 210, 210),  # Crisp White/Silver
        "SURPRISE":  (0,   180, 255),  # Amber/Orange
        "SAD":       (255, 140, 50),   # Cyan/Blue
        "FEAR":      (210, 90,  210),  # Purple
        "ANGRY":     (50,  50,  240),  # Crimson Red
        "DISGUST":   (50,  200, 50),   # Emerald Green
    }
    theme_color = (0, 165, 255) if is_blurry else EMO_HUD_COLORS.get(emo, (0, 230, 118))

    if box:
        bx = max(0, min(box.get("x", 0), w - 10))
        by = max(0, min(box.get("y", 0), h - 10))
        bw = max(10, min(box.get("w", 0), w - bx))
        bh = max(10, min(box.get("h", 0), h - by))

        if bw > 30 and bh > 30:
            c_len = max(14, min(bw, bh) // 5)
            thick = 2

            # Top-left corner
            cv2.line(bgr, (bx, by), (bx + c_len, by), theme_color, thick)
            cv2.line(bgr, (bx, by), (bx, by + c_len), theme_color, thick)

            # Top-right corner
            cv2.line(bgr, (bx + bw, by), (bx + bw - c_len, by), theme_color, thick)
            cv2.line(bgr, (bx + bw, by), (bx + bw, by + c_len), theme_color, thick)

            # Bottom-left corner
            cv2.line(bgr, (bx, by + bh), (bx + c_len, by + bh), theme_color, thick)
            cv2.line(bgr, (bx, by + bh), (bx, by + bh - c_len), theme_color, thick)

            # Bottom-right corner
            cv2.line(bgr, (bx + bw, by + bh), (bx + bw - c_len, by + bh), theme_color, thick)
            cv2.line(bgr, (bx + bw, by + bh), (bx + bw, by + bh - c_len), theme_color, thick)

            # Center target crosshair pip
            cx, cy = bx + bw // 2, by + bh // 2
            cv2.drawMarker(bgr, (cx, cy), theme_color, cv2.MARKER_CROSS, 10, 1)

            # Reticle tag badge
            reticle_lbl = f"{emo} {conf:.0f}%"
            lbl_y = max(by - 8, top_h + 16)
            lbl_x = max(10, min(bx, w - 120))
            cv2.putText(bgr, reticle_lbl, (lbl_x, lbl_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, theme_color, 2, cv2.LINE_AA)

    # 4. Top Banner Content (Left: Emotion Status, Right: System Telemetry)
    if box is not None or snap.get("face_quality", 0.0) > 0.10:
        status_text = f"● {emo}  {conf:.0f}%"
        status_color = theme_color
    else:
        status_text = "◌ SCANNING ASTRONAUT..."
        status_color = (160, 175, 190)

    clarity_lbl = "BLUR GATED" if is_blurry else f"SHARP ({snap.get('blur_score', 100.0):.0f})"
    clarity_col = (0, 165, 255) if is_blurry else (0, 230, 118)
    telem_text = f"DIP: ISOTROPIC | {clarity_lbl} | #{snap.get('frame_count', 0)}"

    (tw_status, _), _ = cv2.getTextSize(status_text, cv2.FONT_HERSHEY_SIMPLEX, 0.62, 2)
    (tw_telem, _), _  = cv2.getTextSize(telem_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    x_telem = w - tw_telem - 14

    if 14 + tw_status + 16 > x_telem:
        # Compact telemetry label to guarantee zero overlap on small resolutions
        telem_text = f"{clarity_lbl} | #{snap.get('frame_count', 0)}"
        (tw_telem, _), _ = cv2.getTextSize(telem_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
        x_telem = w - tw_telem - 14

    cv2.putText(bgr, status_text, (14, 27),
                cv2.FONT_HERSHEY_SIMPLEX, 0.62, status_color, 2, cv2.LINE_AA)
    if 14 + tw_status + 10 <= x_telem:
        cv2.putText(bgr, telem_text, (x_telem, 26),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.44, clarity_col, 1, cv2.LINE_AA)

    # 5. Bottom Banner Content (Left: Eye Tracking, Right: Astronaut Vision Tag)
    fatigue = snap.get("fatigue_label", "Normal")
    fatigue_col = {
        "Normal":        (0, 230, 118),
        "Drowsy":        (0, 70, 240),
        "Stressed Eyes": (0, 165, 255),
        "Hyperfocused":  (0, 215, 255),
    }.get(fatigue, (180, 180, 180))

    eye_text = f"EAR: {snap.get('ear', 0.30):.2f}   BLINK: {snap.get('blink_rate', 0.0):.0f}/min   FATIGUE: {fatigue.upper()}"
    mission_tag = "MAITRI 2.0 // ASTRONAUT VISION"

    (tw_eye, _), _ = cv2.getTextSize(eye_text, cv2.FONT_HERSHEY_SIMPLEX, 0.44, 1)
    (tw_tag, _), _ = cv2.getTextSize(mission_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.40, 1)
    x_tag = w - tw_tag - 14

    if 14 + tw_eye + 16 > x_tag:
        mission_tag = "MAITRI 2.0"
        (tw_tag, _), _ = cv2.getTextSize(mission_tag, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        x_tag = w - tw_tag - 14

    cv2.putText(bgr, eye_text, (14, h - 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, fatigue_col, 1, cv2.LINE_AA)
    if 14 + tw_eye + 12 <= x_tag:
        cv2.putText(bgr, mission_tag, (x_tag, h - 14),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (150, 165, 180), 1, cv2.LINE_AA)

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
            f"🛡️ <b>Vision Pipeline:</b> Hybrid FACS (AU12/AU6) + EfficientNet ONNX (Isotropic Square Crop) &nbsp;|&nbsp; "
            f"<b>Clarity:</b> {blur_status}"
            f"</div>",
            unsafe_allow_html=True,
        )


@st.fragment(run_every=1.0)
def _render_live_voice_telemetry(is_playing: bool):
    ls = st.session_state.live_state
    ls_snap = ls.snapshot_voice() if hasattr(ls, "snapshot_voice") else ls.snapshot()
    rms_val = float(ls_snap.get("voice_rms", 0.0))
    is_spk  = bool(ls_snap.get("is_speaking", False))
    v_emo   = str(ls_snap.get("voice_emotion", "neutral")).lower()
    v_conf  = float(ls_snap.get("voice_confidence", 0.0))
    v_probs = dict(ls_snap.get("voice_probs", {}))

    # 1. Instant Voice Emotion Badge (matching face reticle aerospace style)
    if is_spk:
        theme_col = EMO_COLORS.get(v_emo, "#2ecc71")
        st.markdown(
            f"""
            <div style="background:{theme_col}18; border:2px solid {theme_col}; border-radius:10px; padding:10px 14px; display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
              <div>
                <div style="font-size:0.72rem; color:#8b9cb5; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Instant Voice Emotion</div>
                <div style="font-size:1.35rem; font-weight:bold; color:{theme_col}; margin-top:2px;">🎙️ {v_emo.upper()} ({v_conf*100:.0f}%)</div>
              </div>
              <div style="text-align:right;">
                <span style="display:inline-block; padding:4px 10px; border-radius:12px; background:#13381e; color:#00e676; font-size:0.75rem; font-weight:600; border:1px solid #00e67655; box-shadow:0 0 8px #00e67644;">
                  🔊 ACTIVE
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif v_conf > 0.0:
        # User spoke and paused: keep last detected emotion clearly displayed for side-by-side comparison
        theme_col = EMO_COLORS.get(v_emo, "#95a5a6")
        st.markdown(
            f"""
            <div style="background:{theme_col}12; border:2px solid {theme_col}aa; border-radius:10px; padding:10px 14px; display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
              <div>
                <div style="font-size:0.72rem; color:#8b9cb5; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Instant Voice Emotion (Last Captured)</div>
                <div style="font-size:1.35rem; font-weight:bold; color:{theme_col}; margin-top:2px;">🎙️ {v_emo.upper()} ({v_conf*100:.0f}%)</div>
              </div>
              <div style="text-align:right;">
                <span style="display:inline-block; padding:4px 10px; border-radius:12px; background:#1e2638; color:#8b9cb5; font-size:0.75rem; font-weight:600; border:1px solid #33425b;">
                  🔇 STANDBY (GATE: 0%)
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div style="background:#161d2b; border:2px solid #2a384c; border-radius:10px; padding:10px 14px; display:flex; justify-content:space-between; align-items:center; margin-bottom:10px;">
              <div>
                <div style="font-size:0.72rem; color:#60718b; font-weight:700; text-transform:uppercase; letter-spacing:1px;">Instant Voice Emotion</div>
                <div style="font-size:1.35rem; font-weight:bold; color:#8b9cb5; margin-top:2px;">🔇 STANDBY (AWAITING SPEECH)</div>
              </div>
              <div style="text-align:right;">
                <span style="display:inline-block; padding:4px 10px; border-radius:12px; background:#1e2638; color:#78889e; font-size:0.75rem; font-weight:600; border:1px solid #33425b;">
                  GATE: 0%
                </span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    # 2. Voice Emotion Distribution (7-Class SER)
    st.markdown("<div style='font-size:0.83rem;font-weight:600;margin-bottom:4px;'>Voice Emotion Distribution (7-Class SER)</div>", unsafe_allow_html=True)
    sorted_v = sorted(v_probs.items(), key=lambda x: x[1], reverse=True)
    for emo, prob in sorted_v:
        pct = round(prob * 100, 1)
        bar_w = max(pct, 1)
        bar_color = EMO_COLORS.get(emo, "#888")
        marker = " ◀" if (v_conf > 0.0 and emo == v_emo) else ""
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

    # 3. Live Hardware VU Meter
    if rms_val > 0.0005:
        eff_rms = min(1.0, max(0.0001, rms_val))
        db_val = max(-60.0, min(0.0, 20.0 * np.log10(eff_rms)))
        pct_val = int(max(2, min(100, ((db_val + 48.0) / 48.0) * 100)))
    else:
        db_val = -60.0
        pct_val = 2

    db_text = f"{db_val:.1f} dB" if db_val > -59.5 else "-∞ dB"

    if is_spk:
        badge_bg = "#13381e"
        badge_color = "#00e676"
        badge_dot = "#00e676"
        dot_glow = "0 0 8px #00e676"
        badge_label = "🔊 SPEECH ACTIVE (VAD OPEN)"
    else:
        badge_bg = "#161d2b"
        badge_color = "#8b9cb5"
        badge_dot = "#57677d"
        dot_glow = "none"
        badge_label = "🔇 SILENT / AMBIENT (0% WT)"

    st.progress(
        pct_val / 100.0,
        text=f"🎙️ Hardware Input Level: {db_text} ({pct_val}%)"
    )

    st.markdown(
        f"""
        <div style="display:flex; justify-content:space-between; align-items:center; margin-top:2px; font-size:0.75rem;">
          <span style="display:inline-flex; align-items:center; gap:6px; padding:3px 10px; border-radius:12px; background:{badge_bg}; color:{badge_color}; font-weight:600;">
            <span style="width:7px; height:7px; border-radius:50%; background:{badge_dot}; display:inline-block; box-shadow:{dot_glow};"></span>
            <span>{badge_label}</span>
          </span>
          <span style="color:#60718b; font-family:monospace; font-size:0.75rem;">
            RMS: {rms_val:.4f} | PipeWire 16kHz
          </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every=1.0)
def _render_eye_panel(is_playing: bool):
    if is_playing:
        ls_snap = st.session_state.live_state.snapshot()
        e1, e2 = st.columns(2)
        e1.metric("EAR", f"{ls_snap['ear']:.3f}")
        e2.metric("Blink Rate", f"{ls_snap['blink_rate']:.1f} /min")
        eye_color = {
            "Normal": "#2ecc71", "Drowsy": "#e74c3c",
            "Stressed Eyes": "#e67e22", "Hyperfocused": "#f1c40f",
        }.get(ls_snap["fatigue_label"], "#aaa")
        st.markdown(
            f"<div style='margin-top:6px;'>"
            f"<span class='stress-badge' style='background:{eye_color};'>"
            f"👁️ {ls_snap['fatigue_label']}</span></div>",
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

    weight_mode = st.session_state.get("fusion_weight_mode_toggle", "🤖 Dynamic Quality-Aware")
    manual_mode = (weight_mode == "🎛️ Manual Override")
    manual_face_pct = st.session_state.get("manual_face_split_slider", 60)
    manual_face_ratio = manual_face_pct / 100.0

    fusion = fuse(
        state             = st.session_state.fusion_state,
        face_probs        = face_probs,
        face_quality      = face_quality,
        vitals_strain     = vitals_strain,
        fatigue_strain    = ls_snap["fatigue_strain"],
        eye_quality       = ls_snap["eye_quality"] if is_playing else 0.0,
        voice_probs       = ls_snap["voice_probs"],
        voice_quality     = ls_snap["voice_quality"],
        is_speaking       = ls_snap["is_speaking"],
        manual_override   = manual_mode,
        manual_face_ratio = manual_face_ratio,
    )

    # ── Metrics strip (5 Modalities & Stress) ─────────────────────────────
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Face Weight",    f"{fusion.face_weight*100:.0f}%")
    m2.metric("Voice Weight",   f"{fusion.voice_weight*100:.0f}%")
    m3.metric("Vitals Weight",  f"{fusion.vitals_weight*100:.0f}%")
    m4.metric("Eye Weight",     f"{fusion.eye_weight*100:.0f}%")
    m5.metric("🎯 Stress Index", f"{fusion.stress_pct:.1f}%")

    emo_col, gauge_col = st.columns([1.2, 1.0], gap="large")

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

    # Cache latest fusion for mission logging
    st.session_state["latest_fusion"] = fusion
    st.session_state["latest_alert"] = alert
    st.session_state["latest_face_emotion"] = face_emotion

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

    # ── ROW 1: Side-by-Side Comparison (Live Video + Voice Telemetry) ────
    col_webcam, col_voice = st.columns([1.1, 1.0], gap="medium")

    with col_webcam:
        st.subheader("📹 Live Astronaut Feed & Face FER")
        st.caption("Background ML pipeline · MediaPipe Eye Tracking (~10 FPS)")

        rtc_config = RTCConfiguration(
            {
                "iceServers": [
                    {
                        "urls": [
                            "stun:stun.l.google.com:19302",
                            "stun:stun1.l.google.com:19302",
                            "stun:stun2.l.google.com:19302",
                        ]
                    }
                ]
            }
        )

        # ── Prevent Streamlit fragments from desyncing WebRTC run counters ──
        if "maitri-live" in st.session_state:
            _rtc_ctx = st.session_state["maitri-live"]
            try:
                from streamlit_webrtc.component import (
                    get_this_session_info,
                    get_script_run_count,
                    ComponentValueSnapshot,
                )
                _sinfo = get_this_session_info()
                if _sinfo:
                    _rc = get_script_run_count(_sinfo)
                    if _rc is not None:
                        if _rtc_ctx._last_rendered_run_count is not None:
                            _rtc_ctx._last_rendered_run_count = _rc - 1
                        if getattr(_rtc_ctx, "_component_value_snapshot", None) is not None:
                            _rtc_ctx._component_value_snapshot = ComponentValueSnapshot(
                                component_value=_rtc_ctx._component_value_snapshot.component_value,
                                run_count=_rc - 1,
                            )
            except Exception:
                pass

        # Refresh thread-safe references before WebRTC worker initialization
        _ACTIVE_LIVE_STATE = st.session_state.live_state
        _ACTIVE_EYE_STATE  = st.session_state.eye_state

        # Video-only WebRTC pipeline (<1ms recv latency, 30 FPS locked, zero audio sync delay)
        ctx = webrtc_streamer(
            key="maitri-live",
            mode=WebRtcMode.SENDRECV,
            rtc_configuration=rtc_config,
            video_processor_factory=_make_video_processor,
            media_stream_constraints={"video": {"width": {"ideal": 640}, "height": {"ideal": 480}}, "audio": False},
            async_processing=True,
        )

        _render_dip_status(ctx.state.playing)

    with col_voice:
        st.subheader("🎙️ Voice Emotion & Acoustic Telemetry")
        st.caption("Quantized Wav2Vec2 ONNX (16 kHz) · Hardware Ballistic VU")

        _render_live_voice_telemetry(ctx.state.playing)

        st.markdown("---")
        # Fusion Weighting Control toggle
        weight_mode = st.radio(
            "Fusion Weighting Control",
            ["🤖 Dynamic Quality-Aware", "🎛️ Manual Override"],
            horizontal=True,
            key="fusion_weight_mode_toggle",
        )
        if weight_mode == "🎛️ Manual Override":
            manual_face_pct = st.slider(
                "Behavioral Split (Face vs Voice)",
                min_value=0, max_value=100, value=60, step=5,
                format="%d%% Face",
                key="manual_face_split_slider",
                help="Adjust balance between visual facial expressions and vocal acoustics. Default: 60% Face / 40% Voice."
            )
            st.caption(f"Manual Ratio: **{manual_face_pct}% Face / {100 - manual_face_pct}% Voice** (VAD silence-gated)")
        else:
            st.caption("⚡ **Autonomous Quality-Aware Engine**: Auto-balances 60/40 nominal ratio based on real-time face lighting & microphone VAD.")

        # Microphone Sensitivity & Calibration expander
        with st.expander("🎛️ Microphone Sensitivity & Calibration", expanded=False):
            vd = st.session_state.get("voice_detector")
            current_gain = getattr(vd, "gain", 1.0) if vd else 1.0
            current_thresh = getattr(vd, "silence_threshold", 0.030) if vd else 0.030
            new_gain = st.slider("Mic Digital Pre-Gain", 0.5, 3.0, float(current_gain), 0.1, key="mic_gain_slider")
            new_thresh = st.slider("Silence Gate Threshold", 0.005, 0.080, float(current_thresh), 0.005, format="%.3f", key="mic_thresh_slider")
            if vd:
                vd.gain = new_gain
                vd.silence_threshold = new_thresh

    # ── ROW 2: Biological Telemetry & Eye Tracking ────────────────────────
    st.markdown("---")
    st.subheader("💓 Biological Telemetry & Eye Tracking")

    col_vitals, col_eye = st.columns([1.1, 1.0], gap="medium")
    with col_vitals:
        st.markdown("**Physiological Sensors**")
        st.caption("Adjust sliders to reflect sensor readings")

        heart_rate = st.slider("❤️  Heart Rate (BPM)",      50,  160, 75, key="live_hr_slider")
        skin_temp  = st.slider("🌡️  Skin Temperature (°C)", 35.0, 40.0, 36.6, step=0.1, key="live_temp_slider")
        spo2       = st.slider("🫁  Blood Oxygen (SpO₂ %)", 85,  100, 98, key="live_spo2_slider")

        vitals = compute_vitals_strain(heart_rate, skin_temp, float(spo2))

        v1, v2, v3 = st.columns(3)
        v1.metric("HR Strain",   f"{vitals.hr_strain*100:.0f}%")
        v2.metric("Temp Strain", f"{vitals.temp_strain*100:.0f}%")
        v3.metric("SpO₂ Strain", f"{vitals.spo2_strain*100:.0f}%")

        vitals_color = {"Normal": "#2ecc71", "Elevated": "#e67e22", "Critical": "#e74c3c"}.get(
            vitals.status, "#aaa"
        )
        st.markdown(
            f"<div style='margin-top:6px;'>"
            f"<span class='stress-badge' style='background:{vitals_color};'>"
            f"Vitals: {vitals.status}</span></div>",
            unsafe_allow_html=True,
        )

    with col_eye:
        st.markdown("**Eye Tracking & Fatigue**")
        st.caption("MediaPipe 478-Landmark Eye Geometry")
        _render_eye_panel(ctx.state.playing)

    # ── ROW 3: Live Multimodal Stress Assessment & Psychological Support ──
    st.markdown("---")
    st.subheader("🧠 Live Multimodal Stress Assessment & Psychological Support")

    _render_live_assessment(
        heart_rate=heart_rate,
        skin_temp=skin_temp,
        spo2=spo2,
        vitals_strain=vitals.vitals_strain,
        is_playing=ctx.state.playing,
    )

    # ── Mission Telemetry Logging Button (outside fragment for 100% stable execution) ──
    st.markdown("---")
    log_col, _ = st.columns([1, 3])
    with log_col:
        if st.button("💾 Log Reading to Mission Database", width="stretch"):
            snap_log = st.session_state.live_state.snapshot()
            latest_f = st.session_state.get("latest_fusion")
            latest_a = st.session_state.get("latest_alert")
            face_emo = st.session_state.get("latest_face_emotion", snap_log["face_emotion"])
            fused_emo = latest_f.dominant_emotion if latest_f else face_emo
            stress_sc = latest_f.stress_pct if latest_f else 0.0
            stress_lv = latest_a.stress_level if latest_a else "NORMAL"
            alert_lbl = latest_a.alert_label if latest_a else "NOMINAL"
            resp_body = latest_a.body if latest_a else "Telemetry nominal."

            log_event(
                face_emotion    = face_emo,
                fused_emotion   = fused_emo,
                voice_state     = f"{snap_log['voice_emotion'].capitalize()} ({'Speaking' if snap_log['is_speaking'] else 'Silent'})",
                heart_rate      = float(heart_rate),
                temperature     = float(skin_temp),
                spo2            = float(spo2),
                blink_rate      = snap_log["blink_rate"],
                fatigue_label   = snap_log["fatigue_label"],
                stress_score    = stress_sc,
                stress_level    = stress_lv,
                alert_triggered = alert_lbl,
                response_msg    = resp_body,
            )
            st.session_state["last_logged_time"] = time.time()

        if time.time() - st.session_state.get("last_logged_time", 0) < 4.0:
            st.success("✅ Mission telemetry logged to database.")

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
        st.dataframe(log_df, width="stretch", height=380)

        if "fused_emotion" in log_df.columns:
            st.markdown("**Emotion Distribution**")
            st.bar_chart(log_df["fused_emotion"].value_counts())