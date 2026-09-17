"""
MAITRI 2.0 — M3: Vitals & Physiological Strain Module
Computes normalised physiological strain from HR, skin temperature, and SpO2.
All values come from sliders (simulated sensor telemetry).
"""

from dataclasses import dataclass


@dataclass
class VitalsResult:
    heart_rate:    float   # BPM
    temperature:   float   # °C
    spo2:          float   # % oxygen saturation
    hr_strain:     float   # 0–1
    temp_strain:   float   # 0–1
    spo2_strain:   float   # 0–1
    vitals_strain: float   # 0–1  (weighted composite)
    status:        str     # "Normal" / "Elevated" / "Critical"


def _clamp(val: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, val))


def compute_vitals_strain(
    heart_rate: float,
    temperature: float,
    spo2: float,
) -> VitalsResult:
    """
    Compute composite physiological strain.

    HR normalisation:
        ≤70 BPM → 0  (resting)
        130 BPM → 1  (high exertion / stress)

    Temperature normalisation:
        ≤37.0 °C → 0  (normal core)
        39.0 °C  → 1  (fever / hyperthermia)

    SpO2 normalisation:
        ≥98% → 0   (optimal)
        85%  → 1   (hypoxic — mission-critical concern)

    Composite weights: HR 50%, Temp 30%, SpO2 20%
    """
    hr_strain   = _clamp((heart_rate - 70) / 60.0)
    temp_strain = _clamp((temperature - 37.0) / 2.0)
    spo2_strain = _clamp((98.0 - spo2) / 13.0)

    vitals_strain = (
        0.50 * hr_strain
        + 0.30 * temp_strain
        + 0.20 * spo2_strain
    )
    vitals_strain = _clamp(vitals_strain)

    # Status label (used in dashboard badge)
    if vitals_strain >= 0.70:
        status = "Critical"
    elif vitals_strain >= 0.40:
        status = "Elevated"
    else:
        status = "Normal"

    return VitalsResult(
        heart_rate=heart_rate,
        temperature=temperature,
        spo2=spo2,
        hr_strain=hr_strain,
        temp_strain=temp_strain,
        spo2_strain=spo2_strain,
        vitals_strain=vitals_strain,
        status=status,
    )

