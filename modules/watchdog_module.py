"""
MAITRI 2.0 — M7: Astronaut Responsiveness & Inactivity Watchdog
Implements NASA Human Research Program (HRP) & ISS Crew Medical Protocols
for detecting astronaut immobility, catatonic stupor, silent hypoxia, or incapacitation.

Features:
- Multi-factor concurrence: Head stillness + facial affect freeze + ocular vigilance + vital telemetry.
- Prevents false alarms when astronauts sit motionless reading or piloting consoles.
- Hands-free consciousness recovery: Resuming movement or active blinking automatically
  clears check-in after a mandatory 5-second debounce safety window.
"""

import time
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple, Dict, Any


class WatchdogState(str, Enum):
    NORMAL = "NORMAL"
    CHECK_IN = "CHECK_IN"               # Multi-factor inactivity: In-cabin prompt
    INCAPACITATED = "INCAPACITATED"     # Timed out or acute collapse: Cabin emergency alert


@dataclass
class WatchdogStatus:
    state: WatchdogState
    motionless_seconds: float
    affect_frozen_seconds: float
    countdown_remaining: float
    debounce_remaining: float
    last_motion_timestamp: float
    alert_message: str
    incident_triggered: bool = False
    recovered_via_motion: bool = False


class AstronautWatchdog:
    """
    Intelligent Multi-Factor Astronaut Inactivity & Responsiveness Watchdog.
    Monitors head motion deltas, facial affect stability (expression freeze),
    ocular vigilance (EAR + blink frequency), and vital signs concurrence.
    """

    def __init__(
        self,
        inactivity_threshold_sec: float = 45.0,
        checkin_timeout_sec: float = 15.0,
        debounce_recovery_sec: float = 5.0,
    ):
        self.inactivity_threshold = inactivity_threshold_sec
        self.checkin_timeout = checkin_timeout_sec
        self.debounce_recovery_sec = debounce_recovery_sec

        self.last_motion_time: float = time.time()
        self.prev_face_center: Optional[Tuple[float, float]] = None
        self.checkin_start_time: Optional[float] = None

        self.state: WatchdogState = WatchdogState.NORMAL
        self.incident_logged: bool = False
        self.recovered_via_motion: bool = False

        # Affect freeze tracking
        self.last_dominant_emotion: Optional[str] = None
        self.emotion_freeze_start: float = time.time()

    def update(
        self,
        face_box: Optional[dict],
        ear: float,
        blink_rate: float,
        dominant_emotion: str = "neutral",
        spo2: float = 98.0,
        heart_rate: float = 75.0,
    ) -> WatchdogStatus:
        """
        Update watchdog state using multi-sensor concurrence.
        """
        now = time.time()
        head_motion_detected = False

        # ── 1. Head displacement tracking ──────────────────────────────────
        if face_box:
            cx = face_box.get("x", 0) + face_box.get("w", 0) / 2.0
            cy = face_box.get("y", 0) + face_box.get("h", 0) / 2.0

            if self.prev_face_center is not None:
                dx = abs(cx - self.prev_face_center[0])
                dy = abs(cy - self.prev_face_center[1])
                # Shift by at least 8.0 pixels counts as head movement
                if dx > 8.0 or dy > 8.0:
                    head_motion_detected = True
            else:
                # First observation: establish baseline position without registering motion
                head_motion_detected = False

            self.prev_face_center = (cx, cy)

        # ── 2. Facial affect freeze tracking ──────────────────────────────
        current_emo = (dominant_emotion or "neutral").lower()
        if self.last_dominant_emotion is None or current_emo != self.last_dominant_emotion:
            self.last_dominant_emotion = current_emo
            self.emotion_freeze_start = now
        affect_frozen_sec = max(0.0, now - self.emotion_freeze_start)
        is_affect_frozen = affect_frozen_sec >= self.inactivity_threshold

        # ── 3. Ocular & neuromotor activity ───────────────────────────────
        eyes_open_and_active = (ear >= 0.22 and blink_rate >= 6.0)
        ocular_stupor = (ear < 0.22 or blink_rate < 4.0)

        # ── 4. Vital telemetry distress checks ────────────────────────────
        vital_distress = (spo2 < 90.0) or (heart_rate < 50.0) or (heart_rate > 125.0)
        acute_vital_collapse = (spo2 < 88.0) or (heart_rate < 42.0) or (heart_rate > 145.0)

        # Update last motion time
        # Active blinking with open eyes also demonstrates neuromotor presence
        if head_motion_detected or (blink_rate >= 10.0 and ear >= 0.23):
            self.last_motion_time = now

        motionless_duration = max(0.0, now - self.last_motion_time)

        # ── 5. State Machine Logic ────────────────────────────────────────
        countdown = 0.0
        debounce_remaining = 0.0
        incident_triggered = False
        recovered_now = False

        if self.state == WatchdogState.NORMAL:
            self.recovered_via_motion = False
            trigger_checkin = False
            trigger_reason = ""

            # Check Concurrence Conditions:
            # Condition A: Acute physiological collapse (immediate check-in)
            if acute_vital_collapse:
                trigger_checkin = True
                trigger_reason = f"Acute physiological distress detected (SpO2: {spo2:.1f}%, HR: {heart_rate:.0f} BPM)."

            # Condition B: Motionless + Vital distress (rapid escalation)
            elif motionless_duration >= 20.0 and vital_distress:
                trigger_checkin = True
                trigger_reason = f"Prolonged immobility ({motionless_duration:.0f}s) combined with physiological strain (SpO2: {spo2:.1f}%)."

            # Condition C: Prolonged stillness + (Ocular stupor / closed eyes OR Frozen facial affect)
            elif motionless_duration >= self.inactivity_threshold:
                # If astronaut is sitting still, but eyes are open, blinking, and vitals are normal:
                # Astronaut is calmly reading / working at console -> DO NOT FALSE ALARM!
                if ocular_stupor or is_affect_frozen:
                    trigger_checkin = True
                    details = "closed/drowsy eyes" if ocular_stupor else "frozen facial affect"
                    trigger_reason = f"Immobility ({motionless_duration:.0f}s) concurrent with {details}."

            if trigger_checkin:
                self.state = WatchdogState.CHECK_IN
                self.checkin_start_time = now
                countdown = self.checkin_timeout
                debounce_remaining = self.debounce_recovery_sec
                msg = f"⚠️ Crew Status Inquiry: {trigger_reason} Audio/Visual Check-In Initiated."
            else:
                msg = "Nominal Alertness Verified. Multi-sensor concurrence nominal."

        elif self.state == WatchdogState.CHECK_IN:
            elapsed_in_checkin = now - (self.checkin_start_time or now)
            countdown = max(0.0, self.checkin_timeout - elapsed_in_checkin)
            debounce_remaining = max(0.0, self.debounce_recovery_sec - elapsed_in_checkin)

            # ── Hands-Free Automatic Recovery via Movement (after 5s debounce) ─
            if debounce_remaining <= 0.0:
                # Debounce window has passed! Check if astronaut moved or resumed active blinking
                if head_motion_detected or (ear >= 0.23 and blink_rate >= 8.0):
                    if not vital_distress:
                        self.state = WatchdogState.NORMAL
                        self.last_motion_time = now
                        self.checkin_start_time = None
                        self.recovered_via_motion = True
                        recovered_now = True
                        msg = "✅ Consciousness Verified via Astronaut Movement (Hands-Free Recovery)."
                        return WatchdogStatus(
                            state=self.state,
                            motionless_seconds=0.0,
                            affect_frozen_seconds=round(affect_frozen_sec, 1),
                            countdown_remaining=0.0,
                            debounce_remaining=0.0,
                            last_motion_timestamp=self.last_motion_time,
                            alert_message=msg,
                            incident_triggered=False,
                            recovered_via_motion=True,
                        )

            # Check if countdown expired or acute collapse occurred
            if countdown <= 0.0 or acute_vital_collapse:
                self.state = WatchdogState.INCAPACITATED
                msg = "🚨 CREW INCAPACITATION CONFIRMED. Emergency Life Support Protocol Engaged."
                incident_triggered = not self.incident_logged
                self.incident_logged = True
            else:
                if debounce_remaining > 0.0:
                    msg = f"⚠️ ASTRONAUT STATUS INQUIRY: Confirm conscious state within {countdown:.0f}s (Motion sensor armed in {debounce_remaining:.0f}s)."
                else:
                    msg = f"⚠️ ASTRONAUT STATUS INQUIRY: Confirm conscious state within {countdown:.0f}s (Move head or blink to auto-clear)."

        elif self.state == WatchdogState.INCAPACITATED:
            msg = "🚨 CREW INCAPACITATION CONFIRMED. Emergency Telemetry Dossier Queued for Ground Station."

        return WatchdogStatus(
            state=self.state,
            motionless_seconds=round(motionless_duration, 1),
            affect_frozen_seconds=round(affect_frozen_sec, 1),
            countdown_remaining=round(countdown, 1),
            debounce_remaining=round(debounce_remaining, 1),
            last_motion_timestamp=self.last_motion_time,
            alert_message=msg,
            incident_triggered=incident_triggered,
            recovered_via_motion=recovered_now or self.recovered_via_motion,
        )

    def acknowledge_conscious(self) -> None:
        """Manual button override to immediately dismiss check-in or reset alert."""
        self.state = WatchdogState.NORMAL
        self.last_motion_time = time.time()
        self.checkin_start_time = None
        self.incident_logged = False
        self.recovered_via_motion = False

    def trigger_checkin_simulation(self) -> None:
        """Trigger an instant check-in for presentation/demonstration purposes."""
        self.state = WatchdogState.CHECK_IN
        self.checkin_start_time = time.time()
        self.last_motion_time = time.time() - self.inactivity_threshold
        self.emotion_freeze_start = time.time() - self.inactivity_threshold
        self.incident_logged = False
        self.recovered_via_motion = False
