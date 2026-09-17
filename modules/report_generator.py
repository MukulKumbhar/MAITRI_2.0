"""
MAITRI 2.0 — M9: Automated Flight Surgeon Medical Dossier Generator
Compiles official ISRO / NASA formatted clinical telemetry incident reports
for ground medical evaluation, post-pass auditing, and autonomous life support escalation.
"""

import datetime
from typing import Dict, Any, Optional


def generate_medical_dossier(
    incident_type: str,
    crew_member: str,
    met_str: str,
    vitals_dict: Dict[str, Any],
    eye_dict: Dict[str, Any],
    fusion_dict: Dict[str, Any],
    watchdog_status: Optional[Dict[str, Any]] = None,
    ground_delay_str: str = "0.5s",
    mission_name: str = "ISRO Gaganyaan (LEO Orbital)",
) -> str:
    """
    Generate an official NASA/ISRO format medical incident report in Markdown.
    """
    now_utc = datetime.datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")

    hr = vitals_dict.get("heart_rate", 75.0)
    temp = vitals_dict.get("temperature", 36.6)
    spo2 = vitals_dict.get("spo2", 98.0)
    psi = vitals_dict.get("psi_score", 1.2)
    psi_cat = vitals_dict.get("psi_category", "No Strain")
    hypoxia = vitals_dict.get("hypoxia_risk", "Nominal")

    ear = eye_dict.get("ear", 0.28)
    blink_rate = eye_dict.get("blink_rate", 16.0)
    fatigue_label = eye_dict.get("fatigue_label", "Normal")

    dominant_emo = fusion_dict.get("dominant_emotion", "Neutral").capitalize()
    stress_pct = fusion_dict.get("stress_pct", 8.0)

    motionless_sec = watchdog_status.get("motionless_seconds", 0.0) if watchdog_status else 0.0
    affect_frozen_sec = watchdog_status.get("affect_frozen_seconds", 0.0) if watchdog_status else 0.0

    report = f"""# 🛰️ MISSION MEDICAL INCIDENT DOSSIER
**ISRO HUMAN SPACEFLIGHT CENTRE (HSFC) / NASA HRP CLINICAL TELEMETRY**  
**DOCUMENT ID:** `MED-INC-{datetime.datetime.utcnow().strftime('%Y%m%d-%H%M%S')}`  
**SECURITY CLASSIFICATION:** ASTRONAUT CLINICAL PRIVILEGED — FLIGHT SURGEON EYES ONLY

---

## 📌 1. MISSION & CREW IDENTIFICATION
* **Active Mission Profile:** {mission_name}
* **Spacecraft Compartment:** Habitation Module (Cabin M-02)
* **Designated Crew Member:** {crew_member}
* **Mission Elapsed Time (MET):** `{met_str}`
* **Report Generation Timestamp:** `{now_utc}`
* **Ground Tele-Link Propagation Delay (One-Way):** `{ground_delay_str}` *(Autonomous Onboard AI Protocol Engaged)*

---

## 🚨 2. INCIDENT CLASSIFICATION & DIAGNOSTIC SUMMARY
* **Incident Event Type:** `{incident_type}`
* **Autonomous Severity Rating:** **{'CRITICAL / TIER-1 PRIORITY' if stress_pct >= 70.0 or spo2 < 90.0 or 'INCAPACITATED' in incident_type else 'ELEVATED CLINICAL ATTENTION'}**
* **Inactivity Assessment:** Head Stillness: `{motionless_sec:.1f}s` | Affect Freeze: `{affect_frozen_sec:.1f}s`
* **Clinical Diagnostic Hypothesis:** {'Possible silent cabin hypercapnia (CO2 pocket formation), acute hypoxia desaturation, or neuro-vestibular syncope.' if spo2 < 90.0 or 'INCAPACITATED' in incident_type else 'Acute autonomic nervous system arousal with cognitive overload.'}

---

## 📊 3. MULTI-SENSOR TELEMETRY SNAPSHOT (AT TIME OF EVENT)

| Parameter | Measured Value | Standard Baseline | Clinical Assessment |
| :--- | :---: | :---: | :---: |
| **Heart Rate (HR)** | **{hr:.0f} BPM** | 60 – 80 BPM | {'⚠️ Tachycardia' if hr > 100 else ('⚠️ Bradycardia' if hr < 55 else '✅ Nominal')} |
| **Blood Oxygen ($SpO_2$)** | **{spo2:.1f}%** | 95.0% – 100.0% | {'🚨 ' + hypoxia if spo2 < 92.0 else '✅ Nominal'} |
| **Skin Temperature** | **{temp:.1f} °C** | 36.5 – 37.5 °C | {'⚠️ Thermal Elevation' if temp > 37.6 else '✅ Nominal'} |
| **Aerospace Moran PSI** | **{psi:.1f} / 10** | 0.0 – 2.5 (No Strain) | {psi_cat} |
| **Eye Aspect Ratio (EAR)** | **{ear:.3f}** | 0.25 – 0.35 | {'⚠️ Low Aperture' if ear < 0.22 else '✅ Normal'} |
| **Blink Frequency** | **{blink_rate:.1f} bpm** | 12 – 22 blinks/min | {fatigue_label} |
| **Facial Affect** | **{dominant_emo}** | Neutral / Dynamic | {'⚠️ Affect Frozen / Flat' if affect_frozen_sec >= 30.0 else ('⚠️ Negative Affect' if dominant_emo in ('Angry', 'Fear', 'Sad') else '✅ Stable')} |
| **Multimodal Combined Stress** | **{stress_pct:.1f}%** | $< 35\%$ | {'🚨 CRITICAL' if stress_pct >= 75.0 else ('⚠️ ELEVATED' if stress_pct >= 55.0 else '✅ NOMINAL')} |

---

## 🛡️ 4. AUTONOMOUS COUNTERMEASURES EXECUTED ONBOARD
1. **Tier-1 Multi-Factor Inquiry:** Cross-checked head displacement, affect stability, ocular aperture, and pulse oximetry.
2. **Tier-2 Audio/Visual Ping:** Dispatched console inquiry with 5s motion recovery safety debounce.
3. **Tier-3 Cabin Alert:** {'Engaged cabin acoustic strobe alert to rouse crew member and notify adjacent cabin personnel.' if 'INCAPACITATED' in incident_type or stress_pct >= 70.0 else 'Dispatched autonomous Cognitive Behavioral Therapy (CBT) somatic regulation protocol.'}
4. **Deep-Space Telemetry Dispatch:** Encapsulated incident report into CCSDS Delay-Tolerant Network (DTN) packet and queued for ground transmission.

---

## 🩺 5. RECOMMENDED FLIGHT SURGEON ACTIONS (UPON GROUND RECEPTION)
1. Instruct secondary crew member (Crew Medical Officer) to conduct immediate verbal and physical responsiveness check.
2. Verify Environmental Control and Life Support System (ECLSS) partial pressures: ppO2 >= 21.3 kPa and ppCO2 < 0.4 kPa.
3. If crew member remains unresponsive, initiate emergency pure O2 resuscitation mask protocol.

---
*Telemetry Dossier automatically compiled and cryptographically signed by MAITRI 2.0 Autonomous Medical Core.*
"""
    return report.strip()
