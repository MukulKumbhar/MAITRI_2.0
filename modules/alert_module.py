"""
MAITRI 2.0 — M5: Alert & Autonomous Support Engine
Rule-based offline psychological support responses.
Completely offline — no LLM, no cloud, no network.
"""

from dataclasses import dataclass
from typing import Optional

# ── Stress level thresholds ────────────────────────────────────────────────
NOMINAL_MAX   = 35.0
MILD_MAX      = 55.0
ELEVATED_MAX  = 75.0
# > 75% → CRITICAL

# ── Colour codes used in the dashboard ────────────────────────────────────
LEVEL_COLORS = {
    "NOMINAL":  "#2ecc71",
    "MILD":     "#f1c40f",
    "ELEVATED": "#e67e22",
    "CRITICAL": "#e74c3c",
}

LEVEL_ICONS = {
    "NOMINAL":  "✅",
    "MILD":     "🔔",
    "ELEVATED": "⚠️",
    "CRITICAL": "🚨",
}

# ── Psychological support messages ─────────────────────────────────────────
SUPPORT_MESSAGES = {
    "NOMINAL": {
        "header": "All Systems Nominal",
        "body": (
            "Vitals and emotional state are within optimal parameters. "
            "Continue mission tasks. You're performing excellently."
        ),
        "technique": None,
    },
    "MILD": {
        "header": "Mild Tension Detected",
        "body": (
            "Elevated fatigue or mild tension detected. Consider taking "
            "a 5-minute mindful pause. Rehydrate and focus on slow breathing."
        ),
        "technique": (
            "**Box Breathing (4-4-4-4):**\n"
            "Inhale for 4 counts → Hold for 4 → Exhale for 4 → Hold for 4.\n"
            "Repeat 3–4 times to reduce tension."
        ),
    },
    "ELEVATED": {
        "header": "Stress Elevation — Intervention Recommended",
        "body": (
            "Moderate-to-high stress detected. Initiating cabin stabilization "
            "protocol. Consider a 10-minute rest cycle before resuming tasks."
        ),
        "technique": (
            "**4-7-8 Breathing (Anxiety Relief):**\n"
            "Inhale through nose for **4 counts** →\n"
            "Hold breath for **7 counts** →\n"
            "Exhale completely through mouth for **8 counts**.\n"
            "Repeat 4 times. This activates the parasympathetic nervous system."
        ),
    },
    "CRITICAL": {
        "header": "⚠️ CRITICAL STRESS — Emergency Protocol Active",
        "body": (
            "Critical psychological stress threshold exceeded. "
            "Immediate intervention required. Logging alert to mission records. "
            "Suspend non-essential tasks now."
        ),
        "technique": (
            "**5-4-3-2-1 Grounding Technique:**\n"
            "Name **5 things** you can see →\n"
            "**4 things** you can physically feel →\n"
            "**3 things** you can hear →\n"
            "**2 things** you can smell →\n"
            "**1 thing** you can taste.\n\n"
            "This grounds you to the present moment and breaks the stress spiral."
        ),
    },
}

# ── Emotion-specific override messages ────────────────────────────────────
EMOTION_OVERRIDES = {
    "fear": {
        "header": "Fear Response Detected",
        "body": (
            "High fear response detected. This is normal in high-pressure "
            "environments. Use the grounding technique below."
        ),
        "technique": SUPPORT_MESSAGES["CRITICAL"]["technique"],
    },
    "angry": {
        "header": "Anger / Frustration Detected",
        "body": (
            "Frustration signals detected. Anger in isolation is common. "
            "Step back from the current task for 3 minutes before continuing."
        ),
        "technique": (
            "**De-escalation Breathing:**\n"
            "Take a slow, deliberate breath in for 5 counts.\n"
            "Hold for 2 counts. Exhale slowly for 7 counts.\n"
            "Say internally: 'I am calm. I am in control. I am capable.'"
        ),
    },
    "sad": {
        "header": "Low Mood Detected",
        "body": (
            "Sadness or low energy detected. Isolation-induced low mood is "
            "well-documented in long-duration missions. You are not alone."
        ),
        "technique": (
            "**Positive Anchoring:**\n"
            "Recall one specific, vivid positive memory — a success, "
            "a moment of joy, or a person you love. Hold it for 60 seconds.\n"
            "Your brain cannot simultaneously feel intense sadness and "
            "intense positive memory recall."
        ),
    },
    "disgust": {
        "header": "Discomfort / Aversion Detected",
        "body": (
            "Discomfort or aversion signals detected. "
            "Environmental factors may be contributing. "
            "Try changing your immediate environment or task."
        ),
        "technique": SUPPORT_MESSAGES["MILD"]["technique"],
    },
}


@dataclass
class AlertResult:
    stress_level:  str
    alert_label:   str
    color:         str
    icon:          str
    header:        str
    body:          str
    technique:     Optional[str]
    is_critical:   bool


def classify_stress(stress_pct: float) -> str:
    """Map a 0–100 stress percentage to a named stress level."""
    if stress_pct >= ELEVATED_MAX:
        return "CRITICAL"
    elif stress_pct >= MILD_MAX:
        return "ELEVATED"
    elif stress_pct >= NOMINAL_MAX:
        return "MILD"
    else:
        return "NOMINAL"


def get_alert(
    stress_pct: float,
    dominant_emotion: str,
    fatigue_label: str = "Normal",
) -> AlertResult:
    """
    Return the appropriate alert + support message.
    Emotion-specific overrides take priority whenever distress emotions are detected or stress is elevated.
    Fatigue guidance activates when eye drowsiness is detected.
    """
    level = classify_stress(stress_pct)
    emotion_key = dominant_emotion.lower()

    # Distress emotions: fear, anger, sadness
    is_distress_emotion = emotion_key in ("fear", "angry", "sad")

    # Apply emotion override if:
    # 1. Stress is MILD, ELEVATED, or CRITICAL and emotion has a specific countermeasure
    # 2. Dominant emotion is an acute distress emotion (even before vitals catch up)
    if (level in ("MILD", "ELEVATED", "CRITICAL") or is_distress_emotion) and emotion_key in EMOTION_OVERRIDES:
        override = EMOTION_OVERRIDES[emotion_key]
        # Promote effective level to at least MILD if displaying distress
        if level == "NOMINAL" and is_distress_emotion:
            level = "MILD"
        msg = {
            "header":    override["header"],
            "body":      override["body"],
            "technique": override["technique"],
        }
    elif fatigue_label == "Drowsy" and level in ("NOMINAL", "MILD", "ELEVATED"):
        level = "ELEVATED" if level == "ELEVATED" else "MILD"
        msg = {
            "header": "Fatigue Alert — Drowsiness Detected",
            "body": (
                "Sustained low Eye Aspect Ratio (EAR) indicates significant eyelid closure "
                "and micro-sleep hazard. Immediate rest recommended."
            ),
            "technique": (
                "**20-20-20 Eye Rest & Micro-Nap Protocol:**\n"
                "1. Look away from display screens at an object 20 feet away for 20 seconds.\n"
                "2. Blink deliberately 10 times.\n"
                "3. If operational schedule permits, initiate a 15-minute cabin rest cycle."
            ),
        }
    else:
        msg = SUPPORT_MESSAGES[level]

    return AlertResult(
        stress_level=level,
        alert_label=f"{LEVEL_ICONS[level]} {level}",
        color=LEVEL_COLORS[level],
        icon=LEVEL_ICONS[level],
        header=msg["header"],
        body=msg["body"],
        technique=msg["technique"],
        is_critical=(level == "CRITICAL"),
    )

