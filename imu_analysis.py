"""
imu_analysis.py
----------------
Extracts gait features from wearable IMU (MPU6050) CSV logs captured
via the ESP32 firmware (see firmware/oa_imu_firmware/).

Expected CSV columns: timestamp_ms,leg,ax,ay,az,gx,gy,gz
(ax/ay/az in m/s^2, gx/gy/gz in rad/s — Adafruit MPU6050 defaults)

Features extracted (v1 — deliberately small and interpretable,
mirroring the same design philosophy as gait_analysis.py):
    1. peak_angular_velocity   : max |gyro| magnitude over the trace (deg/s)
    2. cadence_imu             : estimated steps/sec from gyro peak counting
    3. jerk_rms                : RMS of acceleration derivative — a
                                  smoothness/guarding proxy (higher =
                                  jerkier, less fluid movement, which
                                  can indicate pain-guarding or stiffness)
    4. left_right_asymmetry    : % difference in peak angular velocity
                                  between left/right units, if both
                                  are present in the CSV (None otherwise)
"""

import csv
import math
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional


@dataclass
class ImuFeatures:
    peak_angular_velocity: float
    cadence_imu: float
    jerk_rms: float
    left_right_asymmetry: Optional[float]
    samples_left: int
    samples_right: int

    def as_dict(self) -> Dict:
        return asdict(self)


def _count_peaks(signal: List[float], min_distance: int = 5, threshold: float = 0.0) -> int:
    peaks = 0
    last_peak_idx = -min_distance
    for i in range(1, len(signal) - 1):
        if signal[i] > threshold and signal[i] > signal[i - 1] and signal[i] >= signal[i + 1]:
            if i - last_peak_idx >= min_distance:
                peaks += 1
                last_peak_idx = i
    return peaks


def _gyro_magnitude_deg(gx, gy, gz) -> float:
    # input in rad/s -> magnitude in deg/s
    mag_rad = math.sqrt(gx ** 2 + gy ** 2 + gz ** 2)
    return math.degrees(mag_rad)


def _process_leg_samples(rows: List[Dict]) -> Dict:
    """Compute per-leg raw metrics from a list of {t, ax, ay, az, gx, gy, gz}."""
    if len(rows) < 5:
        return {"peak_gyro": 0.0, "n_peaks": 0, "duration_s": 0.0, "jerk_rms": 0.0}

    gyro_mags = [_gyro_magnitude_deg(r["gx"], r["gy"], r["gz"]) for r in rows]
    peak_gyro = max(gyro_mags)

    # peak counting for cadence — threshold at 40% of peak to avoid noise
    thresh = 0.4 * peak_gyro
    n_peaks = _count_peaks(gyro_mags, min_distance=5, threshold=thresh)

    t0, t1 = rows[0]["t"], rows[-1]["t"]
    duration_s = max((t1 - t0) / 1000.0, 1e-6)

    # jerk = derivative of acceleration magnitude, RMS over trace
    accel_mags = [math.sqrt(r["ax"] ** 2 + r["ay"] ** 2 + r["az"] ** 2) for r in rows]
    jerks = []
    for i in range(1, len(accel_mags)):
        dt = max((rows[i]["t"] - rows[i - 1]["t"]) / 1000.0, 1e-3)
        jerks.append((accel_mags[i] - accel_mags[i - 1]) / dt)
    jerk_rms = math.sqrt(sum(j ** 2 for j in jerks) / len(jerks)) if jerks else 0.0

    return {
        "peak_gyro": peak_gyro,
        "n_peaks": n_peaks,
        "duration_s": duration_s,
        "jerk_rms": jerk_rms,
    }


def extract_imu_features(csv_path: str) -> ImuFeatures:
    left_rows: List[Dict] = []
    right_rows: List[Dict] = []

    with open(csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if len(row) < 8:
                continue
            if row[0].strip().lower().startswith("timestamp"):
                continue  # header line
            try:
                t = float(row[0])
                leg = row[1].strip().upper()
                ax, ay, az = float(row[2]), float(row[3]), float(row[4])
                gx, gy, gz = float(row[5]), float(row[6]), float(row[7])
            except (ValueError, IndexError):
                continue

            record = {"t": t, "ax": ax, "ay": ay, "az": az, "gx": gx, "gy": gy, "gz": gz}
            if leg == "L":
                left_rows.append(record)
            elif leg == "R":
                right_rows.append(record)
            else:
                left_rows.append(record)  # default to "left"/single-unit stream

    if not left_rows and not right_rows:
        raise ValueError(
            "No valid IMU rows found in CSV. Expected columns: "
            "timestamp_ms,leg,ax,ay,az,gx,gy,gz"
        )

    left_metrics = _process_leg_samples(left_rows)
    right_metrics = _process_leg_samples(right_rows) if right_rows else None

    # Combine cadence + peak gyro across available leg(s)
    if right_metrics:
        peak_gyro = max(left_metrics["peak_gyro"], right_metrics["peak_gyro"])
        total_peaks = left_metrics["n_peaks"] + right_metrics["n_peaks"]
        total_duration = max(left_metrics["duration_s"], right_metrics["duration_s"])
        jerk_rms = (left_metrics["jerk_rms"] + right_metrics["jerk_rms"]) / 2

        avg_peak = (left_metrics["peak_gyro"] + right_metrics["peak_gyro"]) / 2
        if avg_peak > 0:
            asymmetry = abs(left_metrics["peak_gyro"] - right_metrics["peak_gyro"]) / avg_peak * 100
        else:
            asymmetry = 0.0
    else:
        peak_gyro = left_metrics["peak_gyro"]
        total_peaks = left_metrics["n_peaks"]
        total_duration = left_metrics["duration_s"]
        jerk_rms = left_metrics["jerk_rms"]
        asymmetry = None

    cadence = (total_peaks / total_duration) if total_duration > 0 else 0.0

    return ImuFeatures(
        peak_angular_velocity=round(peak_gyro, 1),
        cadence_imu=round(cadence, 2),
        jerk_rms=round(jerk_rms, 2),
        left_right_asymmetry=round(asymmetry, 1) if asymmetry is not None else None,
        samples_left=len(left_rows),
        samples_right=len(right_rows),
    )
