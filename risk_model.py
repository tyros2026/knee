"""
risk_model.py
-------------
Rule-based / weighted-score OA risk model (v1 prototype).

This is intentionally NOT a trained classifier — there is no labeled NER
dataset yet. Instead this is a transparent, literature-informed weighted
scoring system, positioned as a *screening/triage aid*, not a diagnosis.

Roadmap note (for pitch): once local/pilot data is collected, this module
is designed to be swapped for a trained gradient-boosting / logistic
regression model with the same feature schema, without changing the app.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

# High-load occupations relevant to NER (agriculture, terrace farming,
# load-carrying on hilly terrain) score higher on mechanical joint stress.
HIGH_LOAD_OCCUPATIONS = {
    "farmer / agricultural labor",
    "manual/load-carrying labor",
    "construction worker",
}
MODERATE_LOAD_OCCUPATIONS = {
    "homemaker (household physical work)",
    "driver",
}


@dataclass
class RiskResult:
    score: float           # 0-100
    band: str              # Low / Moderate / High
    contributing_factors: List[Tuple[str, float]]  # (factor label, points)


def _bmi_points(bmi: float) -> float:
    if bmi < 18.5:
        return 0
    if bmi < 25:
        return 2
    if bmi < 30:
        return 8
    return 14  # obese — strongest modifiable OA risk factor


def _age_points(age: int) -> float:
    if age < 40:
        return 0
    if age < 50:
        return 6
    if age < 60:
        return 12
    return 18


def _occupation_points(occupation: str) -> float:
    occ = occupation.lower()
    if occ in HIGH_LOAD_OCCUPATIONS:
        return 12
    if occ in MODERATE_LOAD_OCCUPATIONS:
        return 6
    return 2


def _pain_points(pain_months: float, pain_severity: int) -> float:
    # pain_severity: 0 (none) - 10 (severe), self-reported
    duration_pts = min(pain_months, 24) / 24 * 12  # cap contribution at 24 months
    severity_pts = pain_severity / 10 * 14
    return duration_pts + severity_pts


def _family_history_points(has_family_history: bool) -> float:
    return 6 if has_family_history else 0


def _gait_points(asymmetry_index: float, flexion_range_avg: float) -> float:
    pts = 0.0
    # Asymmetry between left/right knee flexion range
    if asymmetry_index > 25:
        pts += 10
    elif asymmetry_index > 12:
        pts += 5

    # Reduced overall flexion range (stiffness / guarding proxy).
    # Healthy walking gait typically shows ~50-65 deg of knee flexion range.
    if flexion_range_avg < 30:
        pts += 9
    elif flexion_range_avg < 40:
        pts += 4

    return pts


def _imu_points(peak_angular_velocity: float, jerk_rms: float, asymmetry: float) -> float:
    """
    Points from wearable IMU gait dynamics.

    - Reduced peak angular velocity at the knee during swing phase can
      indicate guarding/restricted motion (typical resting gait swings
      the shank through roughly 150-250 deg/s at the knee; well below
      that range is treated as a soft flag here).
    - High jerk (rate of change of acceleration) indicates less fluid,
      more abrupt movement — a proxy for pain-guarding or instability.
    - Left/right asymmetry in peak angular velocity mirrors the same
      asymmetry signal as the camera-based gait feature, from an
      independent sensor.
    """
    pts = 0.0

    if peak_angular_velocity < 90:
        pts += 8
    elif peak_angular_velocity < 130:
        pts += 4

    if jerk_rms > 25:
        pts += 6
    elif jerk_rms > 15:
        pts += 3

    if asymmetry is not None:
        if asymmetry > 25:
            pts += 8
        elif asymmetry > 12:
            pts += 4

    return pts


def compute_risk(
    age: int,
    bmi: float,
    occupation: str,
    pain_months: float,
    pain_severity: int,
    has_family_history: bool,
    gait_features: Dict = None,
    imu_features: Dict = None,
) -> RiskResult:
    factors: List[Tuple[str, float]] = []

    p = _age_points(age)
    factors.append((f"Age ({age})", p))

    p = _bmi_points(bmi)
    factors.append((f"BMI ({bmi:.1f})", p))

    p = _occupation_points(occupation)
    factors.append((f"Occupational load ({occupation})", p))

    p = _pain_points(pain_months, pain_severity)
    factors.append((f"Pain history ({pain_months:.0f} mo, severity {pain_severity}/10)", p))

    p = _family_history_points(has_family_history)
    factors.append(("Family history of OA/joint disease", p))

    if gait_features:
        avg_flex = (
            gait_features["knee_flexion_range_left"]
            + gait_features["knee_flexion_range_right"]
        ) / 2
        p = _gait_points(gait_features["asymmetry_index"], avg_flex)
        factors.append((
            f"Camera gait analysis (asymmetry {gait_features['asymmetry_index']}%, "
            f"avg flexion range {avg_flex:.0f}°)",
            p,
        ))

    if imu_features:
        p = _imu_points(
            imu_features["peak_angular_velocity"],
            imu_features["jerk_rms"],
            imu_features.get("left_right_asymmetry"),
        )
        asym_str = (
            f"{imu_features['left_right_asymmetry']}%"
            if imu_features.get("left_right_asymmetry") is not None
            else "n/a (single unit)"
        )
        factors.append((
            f"Wearable IMU analysis (peak angular velocity "
            f"{imu_features['peak_angular_velocity']}°/s, jerk {imu_features['jerk_rms']}, "
            f"asymmetry {asym_str})",
            p,
        ))

    total = sum(pts for _, pts in factors)
    total = min(total, 100)

    if total < 30:
        band = "Low"
    elif total < 55:
        band = "Moderate"
    else:
        band = "High"

    return RiskResult(score=round(total, 1), band=band, contributing_factors=factors)
