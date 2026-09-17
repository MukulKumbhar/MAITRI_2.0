"""
Comprehensive automated test suite for Project MAITRI 2.0 upgrades:
1. Multi-factor Watchdog & False Alarm Prevention
2. 5-Second Debounce Hands-Free Motion Consciousness Recovery
3. Mission Profile Dynamic Latency & Bandwidth-Conserving DTN Outbox
4. Medical Dossier Report Generation
"""

import time
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from modules.watchdog_module import AstronautWatchdog, WatchdogState
from modules.dtn_module import DTNOutbox
from modules.report_generator import generate_medical_dossier


def test_watchdog_no_false_alarm_while_reading():
    """Verify that an astronaut sitting motionless reading (eyes open, blinking, normal vitals) does NOT trigger alarm."""
    wd = AstronautWatchdog(inactivity_threshold_sec=45.0, debounce_recovery_sec=5.0)
    # Simulate sitting motionless for 50 seconds (reading console)
    wd.last_motion_time = time.time() - 50.0
    wd.emotion_freeze_start = time.time() - 20.0  # dynamic emotional response

    # Face box stays in place (motionless), but eyes open (EAR=0.28) and normal blink rate (16 bpm)
    status = wd.update(
        face_box={"x": 100, "y": 100, "w": 80, "h": 80},
        ear=0.28,
        blink_rate=16.0,
        dominant_emotion="neutral",
        spo2=98.0,
        heart_rate=72.0,
    )
    assert status.state == WatchdogState.NORMAL, f"Expected NORMAL state while reading, got {status.state}"
    print("✅ PASS: Sitting still while reading/working does NOT trigger false alarm.")


def test_watchdog_triggers_on_affect_freeze_and_stillness():
    """Verify that motionless + frozen affect for >= 45s triggers CHECK_IN."""
    wd = AstronautWatchdog(inactivity_threshold_sec=45.0, debounce_recovery_sec=5.0)
    # Both stillness and affect frozen for 50 seconds
    now = time.time()
    wd.last_motion_time = now - 50.0
    wd.last_dominant_emotion = "neutral"
    wd.emotion_freeze_start = now - 50.0

    # Motionless, low blinks, fixed expression
    status = wd.update(
        face_box={"x": 100, "y": 100, "w": 80, "h": 80},
        ear=0.20,
        blink_rate=3.0,
        dominant_emotion="neutral",
        spo2=98.0,
        heart_rate=65.0,
    )
    assert status.state == WatchdogState.CHECK_IN, f"Expected CHECK_IN, got {status.state}"
    print("✅ PASS: Watchdog triggers CHECK_IN on prolonged stillness + affect freeze / low blinks.")


def test_watchdog_debounce_and_motion_recovery():
    """Verify 5-second debounce window: movement within first 5s does not clear; after 5s clears automatically."""
    wd = AstronautWatchdog(inactivity_threshold_sec=45.0, debounce_recovery_sec=5.0)
    now = time.time()
    wd.state = WatchdogState.CHECK_IN
    wd.checkin_start_time = now - 2.0  # 2 seconds elapsed into check-in (within 5s debounce)
    wd.prev_face_center = (100, 100)

    # Astronaut moves head (dx=25), but debounce is still active (only 2s in)
    status1 = wd.update(
        face_box={"x": 125, "y": 100, "w": 80, "h": 80},
        ear=0.28,
        blink_rate=15.0,
        dominant_emotion="neutral",
        spo2=98.0,
        heart_rate=75.0,
    )
    assert status1.state == WatchdogState.CHECK_IN, "Debounce should prevent clearance within first 5s!"
    assert status1.debounce_remaining > 0.0, "Debounce remaining should be > 0"
    print("✅ PASS: 5-second safety debounce prevents premature auto-dismissal.")

    # Now advance checkin start time to 6 seconds ago (past 5s debounce)
    wd.checkin_start_time = now - 6.0
    wd.prev_face_center = (100, 100)

    # Astronaut moves head (dx=20)
    status2 = wd.update(
        face_box={"x": 120, "y": 100, "w": 80, "h": 80},
        ear=0.28,
        blink_rate=15.0,
        dominant_emotion="neutral",
        spo2=98.0,
        heart_rate=75.0,
    )
    assert status2.state == WatchdogState.NORMAL, f"Expected NORMAL after motion post-debounce, got {status2.state}"
    assert status2.recovered_via_motion is True, "Expected recovered_via_motion to be True"
    print("✅ PASS: Astronaut head motion after 5s debounce automatically restores NORMAL state (Hands-Free Recovery).")


def test_watchdog_acute_hypoxia_collapse():
    """Verify that SpO2 < 88% causes immediate check-in / escalation."""
    wd = AstronautWatchdog(inactivity_threshold_sec=45.0)
    status = wd.update(
        face_box={"x": 100, "y": 100, "w": 80, "h": 80},
        ear=0.28,
        blink_rate=15.0,
        dominant_emotion="neutral",
        spo2=87.0,  # Acute desaturation
        heart_rate=118.0,
    )
    assert status.state == WatchdogState.CHECK_IN, f"Expected CHECK_IN on acute hypoxia, got {status.state}"
    print("✅ PASS: Acute hypoxia desaturation immediately triggers status check-in.")


def test_dtn_dynamic_profiles_and_bandwidth():
    """Verify DTN supports dynamic mission latency and emergency-only queueing."""
    dtn = DTNOutbox(ground_delay_seconds=0.5, mission_name="ISRO Gaganyaan (LEO Orbital)")
    assert "0.5s" in dtn.format_latency_str()

    # Switch to Mars Deep Space
    dtn.set_mission_profile("Mars Deep-Space Transit Habitat", 860.0)
    assert "14m 20s" in dtn.format_latency_str()

    # Dispatch Priority-1 Incident Packet
    pkt = dtn.dispatch_incident_packet(
        incident_type="INCAPACITATION_INCIDENT",
        met_str="MET 04d:18h:22m:14s",
        summary="Crew member unresponsive. Life support alerted.",
        details={"spo2": 88.0, "hr": 120.0},
        priority="CRITICAL-1 (EMERGENCY)",
    )
    assert pkt.priority == "CRITICAL-1 (EMERGENCY)"
    assert pkt.mission_profile == "Mars Deep-Space Transit Habitat"
    assert "14m 20s" in pkt.status
    assert len(dtn.get_packets()) == 1
    print("✅ PASS: DTN dynamically updates mission latency and queues Priority-1 emergency packets.")


def test_medical_dossier_compilation():
    """Verify medical dossier report compiles all clinical sections with mission details."""
    dossier = generate_medical_dossier(
        incident_type="CRITICAL_INCAPACITATION_ALERT",
        crew_member="Cdr. A. Sharma (Pilot / CMO)",
        met_str="MET 04d:18h:25m:00s",
        vitals_dict={"heart_rate": 115.0, "temperature": 37.1, "spo2": 88.5, "psi_score": 6.8, "psi_category": "High Strain", "hypoxia_risk": "Severe Hypoxia Risk"},
        eye_dict={"ear": 0.19, "blink_rate": 2.5, "fatigue_label": "Drowsy"},
        fusion_dict={"dominant_emotion": "Neutral", "stress_pct": 78.5},
        watchdog_status={"motionless_seconds": 48.0, "affect_frozen_seconds": 48.0},
        ground_delay_str="14m 20s (DTN Active)",
        mission_name="Mars Deep-Space Transit Habitat",
    )
    assert "Mars Deep-Space Transit Habitat" in dossier
    assert "Cdr. A. Sharma" in dossier
    assert "CRITICAL / TIER-1 PRIORITY" in dossier
    assert "Severe Hypoxia Risk" in dossier
    assert "14m 20s (DTN Active)" in dossier
    assert "RECOMMENDED FLIGHT SURGEON ACTIONS" in dossier
    print("✅ PASS: Medical dossier formats complete NASA/ISRO clinical incident document.")


if __name__ == "__main__":
    test_watchdog_no_false_alarm_while_reading()
    test_watchdog_triggers_on_affect_freeze_and_stillness()
    test_watchdog_debounce_and_motion_recovery()
    test_watchdog_acute_hypoxia_collapse()
    test_dtn_dynamic_profiles_and_bandwidth()
    test_medical_dossier_compilation()
    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY! 100% VERIFICATION CONFIRMED.")
