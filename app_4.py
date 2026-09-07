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
# UI styling — presentation only; screening logic and data flow are unchanged.
# ----------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* App background and typography */
    .stApp {
        background: linear-gradient(180deg, #f7fafc 0%, #eef5f3 100%);
    }
    .block-container {
        max-width: 1450px;
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid #e2e8f0;
    }
    section[data-testid="stSidebar"] .block-container {
        padding-top: 1.5rem;
    }

    /* Main hero */
    .oa-hero {
        background: linear-gradient(135deg, #0f766e 0%, #115e59 55%, #134e4a 100%);
        padding: 1.6rem 1.8rem;
        border-radius: 22px;
        margin-bottom: 1.25rem;
        box-shadow: 0 12px 30px rgba(15, 118, 110, 0.16);
    }
    .oa-hero h1 {
        color: white !important;
        margin: 0 0 .35rem 0;
        font-size: 2rem;
        letter-spacing: -0.02em;
    }
    .oa-hero p {
        color: rgba(255,255,255,.88);
        margin: 0;
        font-size: .98rem;
    }
    .oa-badge {
        display: inline-block;
        padding: .28rem .65rem;
        border-radius: 999px;
        background: rgba(255,255,255,.14);
        border: 1px solid rgba(255,255,255,.22);
        color: white;
        font-size: .78rem;
        font-weight: 700;
        margin-bottom: .7rem;
    }

    /* Section labels */
    div[data-testid="stVerticalBlock"] h2,
    div[data-testid="stVerticalBlock"] h3 {
        color: #123b3a;
    }

    /* Inputs / uploaders */
    div[data-baseweb="input"] > div,
    div[data-baseweb="select"] > div {
        border-radius: 10px;
        border-color: #d7e2e0;
        background: #fff;
    }
    div[data-testid="stFileUploaderDropzone"] {
        border: 1.5px dashed #9bbab5;
        border-radius: 14px;
        background: #fbfefd;
    }

    /* Buttons */
    .stButton > button,
    .stDownloadButton > button {
        border-radius: 10px;
        font-weight: 650;
        min-height: 2.55rem;
        transition: transform .12s ease, box-shadow .12s ease;
    }
    .stButton > button:hover,
    .stDownloadButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 16px rgba(15, 118, 110, .12);
    }

    /* Metrics */
    div[data-testid="stMetric"] {
        background: #ffffff;
        border: 1px solid #e1ebe9;
        border-radius: 14px;
        padding: .75rem .9rem;
        box-shadow: 0 5px 16px rgba(15, 23, 42, .04);
    }
    div[data-testid="stMetricLabel"] {
        color: #5b6b69;
    }

    /* Alerts / notices */
    div[data-testid="stAlert"] {
        border-radius: 12px;
    }

    /* Dataframe */
    div[data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
    }

    /* Small helper cards */
    .ui-card {
        background: rgba(255,255,255,.82);
        border: 1px solid #dfeae8;
        border-radius: 16px;
        padding: .85rem 1rem;
        margin: .35rem 0 .8rem 0;
        color: #36504d;
    }
    .ui-card strong { color: #123b3a; }
    .ui-kicker {
        font-size: .75rem;
        text-transform: uppercase;
        letter-spacing: .08em;
        font-weight: 800;
        color: #0f766e;
        margin-bottom: .2rem;
    }

    /* Footer */
    .oa-footer {
        text-align: center;
        color: #6b7c79;
        font-size: .78rem;
        padding: .8rem 0 0;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="oa-hero">
        <div class="oa-badge">NER • COMMUNITY SCREENING • PROTOTYPE</div>
        <h1>🦵 OA Risk Screening</h1>
        <p>AI-assisted, multimodal screening support for early osteoarthritis risk assessment.</p>
    </div>
    """,
    unsafe_allow_html=True,
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
        "ui": {
            "language": "Language / ভাষা / भाषा", "records": "🗂️ Patient record history",
            "no_records": "No records saved yet.", "download": t("download", "Download all records (CSV)"),
            "offline": "Offline-first wearable workflow. IMU logs can be captured locally and synced to this app later at a PHC.",
            "patient_profile": "Patient profile", "patient_help": "Enter basic clinical and occupational details",
            "patient_name": "Patient name", "age": "Age", "height": "Height (cm)", "weight": "Weight (kg)",
            "bmi": "Calculated BMI", "occupation": "Primary occupation",
            "pain_duration": "Joint pain duration (months, 0 = none)", "pain_severity": "Pain severity (0 = none, 10 = severe)",
            "family_history": "Family history of OA / joint disease", "computer_vision": "Computer vision",
            "camera_title": "Camera-based gait analysis", "camera_help": "Use a 5–15 second side-on, full-body walking video.",
            "camera_upload": "Upload a short (5-15s) side-on, full-body walking video (e.g. filmed walking across a room or corridor).",
            "upload_video": "Upload video", "run_camera": "Run camera gait analysis", "camera_processing": t("camera_processing", "Running pose estimation on video..."),
            "camera_done": "Camera gait analysis complete.", "camera_failed": "Gait analysis failed", "camera_features": "Extracted camera gait features:",
            "left_knee": "Left knee flexion range", "right_knee": "Right knee flexion range", "asymmetry": "Asymmetry index",
            "cadence_proxy": "Cadence proxy", "pose_detected": "Pose detected in", "clear_camera": "Clear camera gait data",
            "wearable": "Wearable sensing", "imu_title": "ESP32 + MPU6050 gait analysis",
            "imu_help": "Upload the recorded IMU CSV for sensor-based gait features.", "imu_upload_help": "Upload the CSV log from the ESP32 + MPU6050 wearable sensor. Columns: timestamp_ms,leg,ax,ay,az,gx,gy,gz",
            "upload_imu": "Upload IMU CSV", "run_imu": "Run IMU gait analysis", "imu_processing": t("imu_processing", "Processing IMU trace..."),
            "imu_done": "IMU gait analysis complete.", "imu_failed": "IMU analysis failed", "imu_features": "Extracted IMU gait features:",
            "peak_velocity": "Peak angular velocity", "jerk": "Jerk (RMS)", "lr_asymmetry": "L/R asymmetry",
            "cadence_imu": "Cadence (IMU)", "samples": "Samples", "clear_imu": "Clear IMU data",
            "clinical": "Clinical decision support", "risk_title": "Composite OA risk assessment",
            "risk_help": "Combines available patient, camera and wearable inputs. This is not a diagnosis.",
            "score": "Composite risk score", "no_sensor": "No sensor data yet — score is based on demographic/clinical inputs only. Add a gait video and/or IMU sensor log for a more complete, multi-modal assessment.",
            "one_sensor": "Score currently uses", "one_sensor_suffix": "data only. Add the other sensing modality for a stronger fused assessment.",
            "both_sensor": "Score fuses both camera gait and wearable IMU data.", "factors": "Contributing factors:",
            "high_rec": "Recommendation: Refer to PHC / orthopedic professional for clinical evaluation (X-ray / physical exam).",
            "moderate_rec": "Recommendation: Advise lifestyle modification (weight management, joint-friendly exercise) and re-screen in 6 months.",
            "low_rec": "Recommendation: Routine monitoring — no immediate referral needed.",
            "save": "💾 Save patient record", "name_required": "Please enter a patient name before saving.", "saved": "Record saved for",
            "contact": "Contact Us", "follow": "Follow Us", "prototype": "OA Risk Screening • SIH26004 prototype • Screening support, not diagnosis",
            "disclaimer": "⚠️ Prototype for demonstration purposes only. Risk scoring uses a transparent, literature-informed weighted-points model (not a trained classifier) since no labeled NER-population dataset yet exists. Designed to be swapped for a trained ML model without changing the interface.",
        },
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
        "ui": {},
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
        "ui": {
            "language": "भाषा", "records": "🗂️ मरीज के रिकॉर्ड", "no_records": "अभी कोई रिकॉर्ड सेव नहीं है।", "download": "सभी रिकॉर्ड डाउनलोड करें (CSV)",
            "offline": "ऑफलाइन-फर्स्ट वर्कफ़्लो: IMU लॉग स्थानीय रूप से रिकॉर्ड किए जा सकते हैं और बाद में PHC पर इस ऐप में सिंक किए जा सकते हैं।",
            "patient_profile": "मरीज की जानकारी", "patient_help": "बुनियादी स्वास्थ्य और काम से जुड़ी जानकारी दर्ज करें", "patient_name": "मरीज का नाम",
            "age": "उम्र", "height": "कद (cm)", "weight": "वजन (kg)", "bmi": "गणना किया गया BMI", "occupation": "मुख्य व्यवसाय",
            "pain_duration": "जोड़ों के दर्द की अवधि (महीने, 0 = नहीं)", "pain_severity": "दर्द की तीव्रता (0 = नहीं, 10 = बहुत तेज)", "family_history": "OA / जोड़ों की बीमारी का पारिवारिक इतिहास",
            "computer_vision": "कंप्यूटर विज़न", "camera_title": "कैमरा-आधारित चाल विश्लेषण", "camera_help": "5–15 सेकंड का साइड से, पूरे शरीर का चलते हुए वीडियो इस्तेमाल करें।",
            "camera_upload": "5–15 सेकंड का साइड से पूरे शरीर का walking video अपलोड करें।", "upload_video": "वीडियो अपलोड करें", "run_camera": "कैमरा चाल विश्लेषण चलाएँ",
            "camera_processing": "वीडियो पर pose estimation चल रहा है...", "camera_done": "कैमरा चाल विश्लेषण पूरा हुआ।", "camera_failed": "चाल विश्लेषण विफल हुआ", "camera_features": "कैमरा चाल से निकली जानकारी:",
            "left_knee": "बाएँ घुटने का flexion range", "right_knee": "दाएँ घुटने का flexion range", "asymmetry": "असमता सूचकांक", "cadence_proxy": "Cadence proxy", "pose_detected": "Pose पहचाना गया", "clear_camera": "कैमरा डेटा हटाएँ",
            "wearable": "वियरेबल सेंसर", "imu_title": "ESP32 + MPU6050 चाल विश्लेषण", "imu_help": "सेंसर-आधारित चाल की जानकारी के लिए रिकॉर्ड किया गया IMU CSV अपलोड करें।",
            "imu_upload_help": "ESP32 + MPU6050 वियरेबल सेंसर की CSV लॉग अपलोड करें। Columns: timestamp_ms,leg,ax,ay,az,gx,gy,gz", "upload_imu": "IMU CSV अपलोड करें",
            "run_imu": "IMU चाल विश्लेषण चलाएँ", "imu_processing": "IMU डेटा प्रोसेस हो रहा है...", "imu_done": "IMU चाल विश्लेषण पूरा हुआ।", "imu_failed": "IMU विश्लेषण विफल हुआ",
            "imu_features": "IMU चाल से निकली जानकारी:", "peak_velocity": "अधिकतम angular velocity", "jerk": "Jerk (RMS)", "lr_asymmetry": "बाएँ/दाएँ असमता",
            "cadence_imu": "Cadence (IMU)", "samples": "Samples", "clear_imu": "IMU डेटा हटाएँ", "clinical": "क्लिनिकल निर्णय सहायता", "risk_title": "OA जोखिम का संयुक्त आकलन",
            "risk_help": "उपलब्ध मरीज, कैमरा और वियरेबल इनपुट को मिलाता है। यह निदान नहीं है।", "score": "संयुक्त जोखिम स्कोर",
            "no_sensor": "अभी कोई सेंसर डेटा नहीं है — स्कोर केवल मरीज की जानकारी पर आधारित है। बेहतर multi-modal assessment के लिए gait video और/या IMU log जोड़ें।",
            "one_sensor": "स्कोर अभी", "one_sensor_suffix": "डेटा का उपयोग कर रहा है। बेहतर संयुक्त आकलन के लिए दूसरी sensing modality जोड़ें।", "both_sensor": "स्कोर कैमरा gait और wearable IMU दोनों डेटा को मिलाता है।",
            "factors": "जोखिम में योगदान देने वाले कारक:", "high_rec": "सुझाव: PHC / हड्डी रोग विशेषज्ञ से clinical evaluation (X-ray / physical exam) के लिए रेफर करें।",
            "moderate_rec": "सुझाव: वजन प्रबंधन और joint-friendly exercise जैसी lifestyle modification करें और 6 महीने में दोबारा जांच करें।", "low_rec": "सुझाव: सामान्य निगरानी रखें — अभी तत्काल referral की जरूरत नहीं है।",
            "save": "💾 मरीज का रिकॉर्ड सेव करें", "name_required": "रिकॉर्ड सेव करने से पहले मरीज का नाम दर्ज करें।", "saved": "रिकॉर्ड सेव हो गया:", "contact": "संपर्क करें", "follow": "हमें फॉलो करें",
            "prototype": "OA Risk Screening • SIH26004 prototype • Screening support, diagnosis नहीं", "disclaimer": "⚠️ यह केवल demonstration prototype है। जोखिम स्कोर transparent, literature-informed weighted-points model पर आधारित है; यह trained classifier नहीं है।",
        },
    },
}

with st.sidebar:
    language = st.selectbox(t("language", "Language / ভাষা / भाषा"), list(LABELS.keys()))

    st.divider()
    st.subheader(t("records", "🗂️ Patient record history"))
    records_df = load_patient_records()
    if records_df.empty:
        st.caption(t("no_records", "No records saved yet."))
    else:
        st.dataframe(
            records_df[["timestamp", "patient_name", "risk_band", "risk_score"]].iloc[::-1],
            use_container_width=True,
            hide_index=True,
        )
        csv_bytes = records_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            t("download", "Download all records (CSV)"),
            data=csv_bytes,
            file_name="patient_records.csv",
            mime="text/csv",
        )

    st.divider()
    st.caption(t("offline", "Offline-first wearable workflow. IMU logs can be captured locally and synced to this app later at a PHC."))

L = LABELS[language]
UI = L.get("ui", {})

def t(key, default):
    return UI.get(key, default)


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
    st.markdown(f'<div class="ui-card"><div class="ui-kicker">{t("patient_profile", "Patient profile")}</div><strong>{t("patient_help", "Enter basic clinical and occupational details")}</strong><br><span>These inputs contribute to the transparent screening score.</span></div>', unsafe_allow_html=True)

    patient_name = st.text_input(t("patient_name", "Patient name"))
    age = st.number_input(t("age", "Age"), min_value=10, max_value=100, value=45)
    height_cm = st.number_input(t("height", "Height (cm)"), min_value=100, max_value=220, value=160)
    weight_kg = st.number_input(t("weight", "Weight (kg)"), min_value=20, max_value=200, value=65)
    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.metric(t("bmi", "Calculated BMI"), f"{bmi:.1f}")

    occupation = st.selectbox(
        t("occupation", "Primary occupation"),
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

    pain_months = st.slider(t("pain_duration", "Joint pain duration (months, 0 = none)"), 0, 60, 0)
    pain_severity = st.slider(t("pain_severity", "Pain severity (0 = none, 10 = severe)"), 0, 10, 0)
    has_family_history = st.checkbox(t("family_history", "Family history of OA / joint disease"))

    # ---------------- Camera gait modality ----------------
    st.subheader(L["section2"])
    st.markdown(f'<div class="ui-card"><div class="ui-kicker">{t("computer_vision", "Computer vision")}</div><strong>{t("camera_title", "Camera-based gait analysis")}</strong><br><span>{t("camera_help", "Use a 5–15 second side-on, full-body walking video.")}</span></div>', unsafe_allow_html=True)
    st.caption(
        t("camera_upload", "Upload a short (5-15s) side-on, full-body walking video (e.g. filmed walking across a room or corridor).")
    )
    video_file = st.file_uploader(t("upload_video", "Upload video"), type=["mp4", "mov", "avi"], key="video_uploader")

    if video_file is not None:
        if st.button(t("run_camera", "Run camera gait analysis"), type="primary"):
            with st.spinner(t("camera_processing", "Running pose estimation on video...")):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=os.path.splitext(video_file.name)[1]
                ) as tmp:
                    tmp.write(video_file.read())
                    tmp_path = tmp.name
                try:
                    features = extract_gait_features(tmp_path)
                    st.session_state.gait_features = features.as_dict()
                    st.success(t("camera_done", "Camera gait analysis complete."))
                except Exception as e:
                    st.error(f"{t("camera_failed", "Gait analysis failed")}: {e}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.gait_features:
        gf = st.session_state.gait_features
        st.write(f"**{t("camera_features", "Extracted camera gait features:")}**")
        c1, c2, c3 = st.columns(3)
        c1.metric("Left knee flexion range", f"{gf['knee_flexion_range_left']}°")
        c2.metric("Right knee flexion range", f"{gf['knee_flexion_range_right']}°")
        c3.metric("Asymmetry index", f"{gf['asymmetry_index']}%")
        st.caption(
            f"Cadence proxy: {gf['cadence_proxy']} steps/sec · "
            f"Pose detected in {gf['pose_detection_rate']}% of frames "
            f"({gf['frames_with_pose']}/{gf['frames_analyzed']})"
        )
        if st.button(t("clear_camera", "Clear camera gait data")):
            st.session_state.gait_features = None
            st.rerun()

    # ---------------- Wearable IMU modality ----------------
    st.subheader(L["section3"])
    st.markdown(f'<div class="ui-card"><div class="ui-kicker">{t("wearable", "Wearable sensing")}</div><strong>{t("imu_title", "ESP32 + MPU6050 gait analysis")}</strong><br><span>{t("imu_help", "Upload the recorded IMU CSV for sensor-based gait features.")}</span></div>', unsafe_allow_html=True)
    st.caption(t("imu_upload_help", "Upload the CSV log from the ESP32 + MPU6050 wearable sensor. Columns: timestamp_ms,leg,ax,ay,az,gx,gy,gz"))
    imu_file = st.file_uploader(t("upload_imu", "Upload IMU CSV"), type=["csv"], key="imu_uploader")

    if imu_file is not None:
        if st.button(t("run_imu", "Run IMU gait analysis"), type="primary"):
            with st.spinner(t("imu_processing", "Processing IMU trace...")):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv", mode="wb"
                ) as tmp:
                    tmp.write(imu_file.read())
                    tmp_path = tmp.name
                try:
                    imu_feat = extract_imu_features(tmp_path)
                    st.session_state.imu_features = imu_feat.as_dict()
                    st.success(t("imu_done", "IMU gait analysis complete."))
                except Exception as e:
                    st.error(f"{t("imu_failed", "IMU analysis failed")}: {e}")
                finally:
                    os.unlink(tmp_path)

    if st.session_state.imu_features:
        imf = st.session_state.imu_features
        st.write(f"**{t("imu_features", "Extracted IMU gait features:")}**")
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
        if st.button(t("clear_imu", "Clear IMU data")):
            st.session_state.imu_features = None
            st.rerun()

# ----------------------------------------------------------------------
# RIGHT COLUMN — Risk score output
# ----------------------------------------------------------------------
with col_right:
    st.subheader(L["section4"])
    st.markdown(f'<div class="ui-card"><div class="ui-kicker">{t("clinical", "Clinical decision support")}</div><strong>{t("risk_title", "Composite OA risk assessment")}</strong><br><span>{t("risk_help", "Combines available patient, camera and wearable inputs. This is not a diagnosis.")}</span></div>', unsafe_allow_html=True)

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

    risk_band_display = {
        "English": {"Low": "Low Risk", "Moderate": "Moderate Risk", "High": "High Risk"},
        "हिन्दी (Hindi)": {"Low": "कम जोखिम", "Moderate": "मध्यम जोखिम", "High": "उच्च जोखिम"},
        "অসমীয়া (Assamese)": {"Low": "কম বিপদ", "Moderate": "মধ্যম বিপদ", "High": "উচ্চ বিপদ"},
    }.get(language, {}).get(result.band, result.band)
    st.markdown(
        f"<h1 style='color:{band_color};'>{risk_band_display}</h1>",
        unsafe_allow_html=True,
    )
    st.progress(min(int(result.score), 100) / 100)
    st.metric(t("score", "Composite risk score"), f"{result.score} / 100")

    modalities_used = []
    if st.session_state.gait_features:
        modalities_used.append("camera gait")
    if st.session_state.imu_features:
        modalities_used.append("wearable IMU")

    if not modalities_used:
        st.info(t("no_sensor", "No sensor data yet — score is based on demographic/clinical inputs only. Add a gait video and/or IMU sensor log for a more complete, multi-modal assessment."))
    elif len(modalities_used) == 1:
        st.info(f"{t("one_sensor", "Score currently uses")} **{modalities_used[0]}** {t("one_sensor_suffix", "data only. Add the other sensing modality for a stronger fused assessment.")}")
    else:
        st.success(t("both_sensor", "Score fuses **both** camera gait and wearable IMU data."))

    st.write(f"**{t("factors", "Contributing factors:")}**")
    for label, pts in result.contributing_factors:
        st.write(f"- {label}: **+{pts:.1f} pts**")

    st.divider()

    if result.band == "High":
        st.error(t("high_rec", "Recommendation: Refer to PHC / orthopedic professional for clinical evaluation (X-ray / physical exam)."))
    elif result.band == "Moderate":
        st.warning(t("moderate_rec", "Recommendation: Advise lifestyle modification (weight management, joint-friendly exercise) and re-screen in 6 months."))
    else:
        st.success(t("low_rec", "Recommendation: Routine monitoring — no immediate referral needed."))

    st.divider()

    # ---------------- Save patient record ----------------
    if st.button(t("save", "💾 Save patient record"), type="primary"):
        if not patient_name:
            st.warning(t("name_required", "Please enter a patient name before saving."))
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
            st.success(f"{t("saved", "Record saved for") } **{patient_name}**.")

st.markdown(f"""
<div class="contact-wrap" id="contact">
  <h2>{t("contact", "Contact Us")}</h2>
  <div class="contact-info">✉️ tyros2026@gmail.com &nbsp; • &nbsp; ☎️ 7979824251 &nbsp; • &nbsp; 📍 GEC Vaishali</div>
  <h3>{t("follow", "Follow Us")}</h3>
  <div class="social-row">
    <span class="social-icon" title="Instagram">◎</span>
    <span class="social-icon" title="X / Twitter">𝕏</span>
    <span class="social-icon" title="Facebook">f</span>
  </div>
</div>
""", unsafe_allow_html=True)
st.divider()
st.markdown(f'<div class="oa-footer">🦵 {t("prototype", "OA Risk Screening • SIH26004 prototype • Screening support, not diagnosis")}</div>', unsafe_allow_html=True)
st.caption(t("disclaimer", "⚠️ Prototype for demonstration purposes only."))
