"""
Streamlit demo — AI-Assisted Early Detection of OA Risk Markers (NER)
-----------------------------------------------------------------------
SIH26004 prototype: hardware-anchored, multi-modal screening/triage tool
(Tier 1 of the proposed 3-tier system).

Modalities fused into one risk score:
  1. Demographic / occupational / clinical inputs (form)
  2. Camera-based gait analysis (MediaPipe pose estimation on a phone video)
  3. Wearable IMU gait analysis (ESP32 + MPU6050 sensor, see firmware/)

NOT a diagnostic device — flags candidates for PHC-level follow-up.
"""

import os
import tempfile
from datetime import datetime

import pandas as pd
import streamlit as st

from gait_analysis import extract_gait_features
from imu_analysis import extract_imu_features
from risk_model import compute_risk

# ----------------------------------------------------------------------
# Local patient-record storage (offline-first: a simple CSV file sitting
# next to this script). Not a database — just enough for a prototype to
# keep a running log of screened patients that a PHC worker can review,
# export, or later migrate into a proper records system.
# ----------------------------------------------------------------------
RECORDS_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "patient_records.csv")


def save_patient_record(record: dict) -> None:
    df_new = pd.DataFrame([record])
    if os.path.exists(RECORDS_PATH):
        df_new.to_csv(RECORDS_PATH, mode="a", header=False, index=False)
    else:
        df_new.to_csv(RECORDS_PATH, mode="w", header=True, index=False)


def load_patient_records() -> pd.DataFrame:
    if os.path.exists(RECORDS_PATH):
        return pd.read_csv(RECORDS_PATH)
    return pd.DataFrame()

st.set_page_config(
    page_title="OA Risk Screening — NER Prototype",
    page_icon="🦵",
    layout="wide",
)

# ----------------------------------------------------------------------
# Minimal multilingual label set (PS requirement: "multilingual and
# easy-to-use interfaces suitable for rural healthcare settings in NER").
# v1 covers English + Assamese + Hindi as a demonstration of the pattern;
# the same dict structure extends to Bodo, Khasi, Mizo, Manipuri, etc.
# ----------------------------------------------------------------------
LABELS = {
    "English": {
        "title": "🦵 AI-Assisted Early OA Risk Screening",
        "subtitle": (
            "Prototype hardware-anchored community screening tool for the "
            "North Eastern Region. This is a **screening/triage aid**, not "
            "a diagnosis — flagged cases should be referred to a PHC / "
            "orthopedic professional."
        ),
        "section1": "1. Patient details",
        "section2": "2. Camera gait video",
        "section3": "3. Wearable IMU sensor data",
        "section4": "4. Risk assessment",
    },
    "অসমীয়া (Assamese)": {
        "title": "🦵 এআই-সহায়ক প্ৰাথমিক অষ্টিঅ'আৰ্থ্ৰাইটিছ স্ক্ৰীনিং",
        "subtitle": (
            "উত্তৰ পূৰ্বাঞ্চলৰ বাবে সমাজ পৰ্যায়ৰ স্ক্ৰীনিং প্ৰটোটাইপ। এইটো "
            "এটা **স্ক্ৰীনিং সহায়ক সঁজুলি**, ৰোগ নিৰ্ণয় নহয় — উচ্চ বিপদজনক "
            "গোট PHC/অস্থি বিশেষজ্ঞলৈ প্ৰেৰণ কৰিব লাগে।"
        ),
        "section1": "১. ৰোগীৰ বিৱৰণ",
        "section2": "২. কেমেৰা গেইট ভিডিঅ'",
        "section3": "৩. পিন্ধিব পৰা IMU ছেন্সৰ ডাটা",
        "section4": "৪. বিপদ মূল্যাংকন",
    },
    "हिन्दी (Hindi)": {
        "title": "🦵 एआई-सहायक प्रारंभिक ओए (ऑस्टियोआर्थराइटिस) जोखिम जांच",
        "subtitle": (
            "पूर्वोत्तर क्षेत्र के लिए हार्डवेयर-आधारित सामुदायिक जांच "
            "प्रोटोटाइप। यह एक **जांच/ट्राइएज सहायता** है, निदान नहीं — "
            "उच्च जोखिम वाले मामलों को PHC / हड्डी रोग विशेषज्ञ के पास "
            "भेजा जाना चाहिए।"
        ),
        "section1": "1. रोगी का विवरण",
        "section2": "2. कैमरा गेट वीडियो",
        "section3": "3. वियरेबल IMU सेंसर डेटा",
        "section4": "4. जोखिम मूल्यांकन",
    },
}

with st.sidebar:
    language = st.selectbox("Language / ভাষা / भाषा", list(LABELS.keys()))

    st.divider()
    st.subheader("🗂️ Patient record history")
    records_df = load_patient_records()
    if records_df.empty:
        st.caption("No records saved yet.")
    else:
        st.dataframe(
            records_df[["timestamp", "patient_name", "risk_band", "risk_score"]].iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = records_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download all records (CSV)",
            data=csv_bytes,
            file_name="patient_records.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption(
        "**Offline-first note:** the wearable IMU logs to onboard flash "
        "even with zero connectivity in the field. Sync the CSV to this "
        "app whenever you're back at a PHC or have a laptop connection — "
        "see `tools/log_serial_to_csv.py --dump`."
    )

L = LABELS[language]

st.title(L["title"])
st.caption(L["subtitle"])

if "gait_features" not in st.session_state:
    st.session_state.gait_features = None
if "imu_features" not in st.session_state:
    st.session_state.imu_features = None

col_left, col_right = st.columns([1, 1])

# ----------------------------------------------------------------------
# LEFT COLUMN — Patient / worker inputs + both sensing modalities
# ----------------------------------------------------------------------
with col_left:
    st.subheader(L["section1"])

    patient_name = st.text_input("Patient name")
    age = st.number_input("Age", min_value=10, max_value=100, value=45)
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=220, value=160)
    weight_kg = st.number_input("Weight (kg)", min_value=20, max_value=200, value=65)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.metric("Calculated BMI", f"{bmi:.1f}")

    occupation = st.selectbox(
        "Primary occupation",
        [
            "Farmer / agricultural labor",
            "Manual/load-carrying labor",
            "Construction worker",
            "Homemaker (household physical work)",
            "Driver",
            "Desk / office work",
            "Student",
            "Retired / not working",
            "Other",
        ],
    )

    pain_months = st.slider("Joint pain duration (months, 0 = none)", 0, 60, 0)
    pain_severity = st.slider("Pain severity (0 = none, 10 = severe)", 0, 10, 0)
    has_family_history = st.checkbox("Family history of OA / joint disease")

    # ---------------- Camera gait modality ----------------
    st.subheader(L["section2"])
    st.caption(
        "Upload a short (5-15s) side-on, full-body walking video "
        "(e.g. filmed walking across a room or corridor)."
    )
    video_file = st.file_uploader("Upload video", type=["mp4", "mov", "avi"], key="video_uploader")

    if video_file is not None:
        if st.button("Run camera gait analysis", type="primary"):
            with st.spinner("Running pose estimation on video..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(video_file.name)[1]
                ) as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                try:
                    features = extract_gait_features(tmp_path)
                    st.session_state.gait_features = features.as_dict()
                    st.success("Camera gait analysis complete.")
                except Exception as e:
                    st.error(f"Gait analysis failed: {e}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.gait_features:
        gf = st.session_state.gait_features
        st.write("**Extracted camera gait features:**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Left knee flexion range", f"{gf['knee_flexion_range_left']}°")
        c2.metric("Right knee flexion range", f"{gf['knee_flexion_range_right']}°")
        c3.metric("Asymmetry index", f"{gf['asymmetry_index']}%")
        st.caption(
            f"Cadence proxy: {gf['cadence_proxy']} steps/sec · "
            f"Pose detected in {gf['pose_detection_rate']}% of frames "
            f"({gf['frames_with_pose']}/{gf['frames_analyzed']})"
        )
        if st.button("Clear camera gait data"):
            st.session_state.gait_features = None
            st.rerun()

    # ---------------- Wearable IMU modality ----------------
    st.subheader(L["section3"])
    st.caption(
        "Upload the CSV log from the ESP32 + MPU6050 wearable sensor "
        "(captured live via `tools/log_serial_to_csv.py`, or dumped from "
        "onboard flash after offline field use). Columns: "
        "timestamp_ms,leg,ax,ay,az,gx,gy,gz"
    )
    imu_file = st.file_uploader("Upload IMU CSV", type=["csv"], key="imu_uploader")

    if imu_file is not None:
        if st.button("Run IMU gait analysis", type="primary"):
            with st.spinner("Processing IMU trace..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv", mode="wb"
                ) as tmp:
                    tmp.write(imu_file.read())
                    tmp_path = tmp.name
                try:
                    imu_feat = extract_imu_features(tmp_path)
                    st.session_state.imu_features = imu_feat.as_dict()
                    st.success("IMU gait analysis complete.")
                except Exception as e:
                    st.error(f"IMU analysis failed: {e}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.imu_features:
        imf = st.session_state.imu_features
        st.write("**Extracted IMU gait features:**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak angular velocity", f"{imf['peak_angular_velocity']}°/s")
        c2.metric("Jerk (RMS)", f"{imf['jerk_rms']}")
        asym_display = (
            f"{imf['left_right_asymmetry']}%"
            if imf["left_right_asymmetry"] is not None
            else "n/a (1 unit)"
        )
        c3.metric("L/R asymmetry", asym_display)
        st.caption(
            f"Cadence (IMU): {imf['cadence_imu']} steps/sec · "
            f"Samples — left: {imf['samples_left']}, right: {imf['samples_right']}"
        )
        if st.button("Clear IMU data"):
            st.session_state.imu_features = None
            st.rerun()

# ----------------------------------------------------------------------
# RIGHT COLUMN — Risk score output
# ----------------------------------------------------------------------
with col_right:
    st.subheader(L["section4"])

    if patient_name:
        st.caption(f"Patient: **{patient_name}**")

    result = compute_risk(
        age=age,
        bmi=bmi,
        occupation=occupation,
        pain_months=pain_months,
        pain_severity=pain_severity,
        has_family_history=has_family_history,
        gait_features=st.session_state.gait_features,
        imu_features=st.session_state.imu_features,
    )

    band_color = {"Low": "green", "Moderate": "orange", "High": "red"}[result.band]

    st.markdown(
        f"<h1 style='color:{band_color};'>{result.band} Risk</h1>",
        unsafe_allow_html=True,
    )
    st.progress(min(int(result.score), 100) / 100)
    st.metric("Composite risk score", f"{result.score} / 100")

    modalities_used = []
    if st.session_state.gait_features:
        modalities_used.append("camera gait")
    if st.session_state.imu_features:
        modalities_used.append("wearable IMU")

    if not modalities_used:
        st.info(
            "No sensor data yet — score is based on demographic/clinical "
            "inputs only. Add a gait video and/or IMU sensor log for a "
            "more complete, multi-modal assessment."
        )
    elif len(modalities_used) == 1:
        st.info(
            f"Score currently uses **{modalities_used[0]}** data only. "
            f"Add the other sensing modality for a stronger fused assessment."
        )
    else:
        st.success("Score fuses **both** camera gait and wearable IMU data.")

    st.write("**Contributing factors:**")
    for label, pts in result.contributing_factors:
        st.write(f"- {label}: **+{pts:.1f} pts**")

    st.divider()

    if result.band == "High":
        st.error(
            "Recommendation: Refer to PHC / orthopedic professional for "
            "clinical evaluation (X-ray / physical exam)."
        )
    elif result.band == "Moderate":
        st.warning(
            "Recommendation: Advise lifestyle modification (weight management, "
            "joint-friendly exercise) and re-screen in 6 months."
        )
    else:
        st.success("Recommendation: Routine monitoring — no immediate referral needed.")

    st.divider()

    # ---------------- Save patient record ----------------
    if st.button("💾 Save patient record", type="primary"):
        if not patient_name:
            st.warning("Please enter a patient name before saving.")
        else:
            record = {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "patient_name": patient_name,
                "age": age,
                "height_cm": height_cm,
                "weight_kg": weight_kg,
                "bmi": round(bmi, 1),
                "occupation": occupation,
                "pain_months": pain_months,
                "pain_severity": pain_severity,
                "has_family_history": has_family_history,
                "modalities_used": ", ".join(modalities_used) if modalities_used else "none",
                "risk_score": result.score,
                "risk_band": result.band,
            }
            save_patient_record(record)
            st.success(f"Record saved for **{patient_name}**.")

st.divider()
st.caption(
    "⚠️ Prototype for demonstration purposes only. Risk scoring uses a "
    "transparent, literature-informed weighted-points model (not a trained "
    "classifier) since no labeled NER-population dataset yet exists. "
    "Designed to be swapped for a trained ML model (e.g. on OAI/MOST + "
    "locally collected pilot data) without changing the interface. "
    "Digital patient records and secure sync/analytics dashboard are "
    "roadmap items beyond this prototype's scope."
)
