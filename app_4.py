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
    page_title="OA Risk Screening — Tyros",
    page_icon="🦵",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');

:root {
  --ink:#12365a;
  --blue:#147be8;
  --blue2:#54a7ff;
  --green:#1f9d78;
  --purple:#8064d9;
  --bg:#f5f9ff;
  --card:rgba(255,255,255,.86);
  --line:#dce9f6;
}

html { scroll-behavior:smooth; }
.stApp {
  background:
    radial-gradient(circle at 12% 8%, rgba(86,167,255,.15), transparent 28%),
    radial-gradient(circle at 90% 30%, rgba(71,192,154,.10), transparent 25%),
    linear-gradient(180deg,#f7fbff 0%,#edf6ff 50%,#f8fbff 100%);
  color:var(--ink);
  font-family:'DM Sans',sans-serif;
}
.block-container { max-width:1320px; padding:1rem 2rem 3rem; }
h1,h2,h3,h4 { font-family:'Space Grotesk',sans-serif !important; color:var(--ink)!important; }
h2 { font-size:2rem!important; letter-spacing:-.04em; }
p { color:#60758d; }

section[data-testid="stSidebar"] {
  background:rgba(255,255,255,.94);
  border-right:1px solid var(--line);
}
section[data-testid="stSidebar"] .block-container { padding:1.3rem 1rem; }

.side-brand { display:flex; gap:.65rem; align-items:center; padding:.4rem .2rem 1rem; }
.brand-mark { width:42px;height:42px;border-radius:14px;background:#e4f7ee;display:grid;place-items:center;font-size:23px; }
.side-brand b { display:block;font-size:1.25rem;color:var(--ink);font-family:'Space Grotesk'; }
.side-brand small { color:#7590a6; }

.top-nav {
  position:sticky; top:.4rem; z-index:50;
  display:flex; align-items:center; justify-content:space-between; gap:1rem;
  padding:.65rem .9rem; margin-bottom:1rem;
  background:rgba(255,255,255,.78); backdrop-filter:blur(18px);
  border:1px solid rgba(205,224,241,.9); border-radius:18px;
  box-shadow:0 8px 28px rgba(26,72,112,.06);
}
.nav-logo { font-family:'Space Grotesk';font-size:1.15rem;font-weight:700;color:var(--ink);white-space:nowrap; }
.nav-logo span { margin-right:.25rem; }
.nav-links { display:flex; gap:1.3rem; }
.nav-links a { text-decoration:none;color:#58718a;font-size:.86rem;font-weight:600; }
.nav-links a:hover { color:var(--blue); }
.nav-cta,.primary-link {
  text-decoration:none; color:white!important; background:linear-gradient(135deg,#1179e8,#1266c4);
  border-radius:12px; padding:.62rem 1rem; font-weight:700; box-shadow:0 8px 18px rgba(20,123,232,.18);
}
.secondary-link { text-decoration:none;color:var(--ink)!important;background:#fff;border:1px solid var(--line);border-radius:12px;padding:.62rem 1rem;font-weight:700; }

.hero-section {
  min-height:510px; display:grid; grid-template-columns:1.02fr .98fr; overflow:hidden;
  border-radius:30px; border:1px solid #d8e8f7;
  background:linear-gradient(135deg,rgba(255,255,255,.96),rgba(225,241,255,.86));
  box-shadow:0 22px 60px rgba(39,92,139,.12); position:relative;
}
.hero-section:before { content:"";position:absolute;inset:auto auto -100px -80px;width:320px;height:320px;border-radius:50%;background:rgba(60,166,255,.12);filter:blur(2px); }
.hero-copy { padding:5rem 2.8rem 3rem; position:relative;z-index:2; }
.eyebrow { display:inline-block;font-size:.72rem;font-weight:800;letter-spacing:.12em;color:#1778d7;background:#eaf5ff;padding:.38rem .65rem;border-radius:999px;margin-bottom:.8rem; }
.hero-copy h1 { font-size:4.2rem!important; line-height:.95;margin:.15rem 0 .75rem; }
.hero-copy h1 span { color:#1979df; }
.hero-copy h3 { font-size:1.28rem!important;margin:.2rem 0 1rem;color:#1e5685!important; }
.hero-copy p { max-width:600px;font-size:1rem;line-height:1.7; }
.hero-actions { display:flex;gap:.7rem;margin:1.5rem 0; }
.hero-pills { display:grid;grid-template-columns:repeat(3,1fr);gap:.6rem;margin-top:1.7rem; }
.hero-pills div { background:rgba(255,255,255,.72);border:1px solid #dceafa;border-radius:14px;padding:.7rem;font-size:.85rem; }
.hero-pills b,.hero-pills small { display:block;margin-left:1.5rem; }
.hero-pills small { color:#7890a6;font-size:.72rem; }
.hero-visual { position:relative;min-height:510px;overflow:hidden;background:linear-gradient(145deg,#e9f7ff,#cbe7ff 48%,#eefcf8); }
.hero-visual:after { content:"";position:absolute;inset:0;background:linear-gradient(90deg,rgba(230,245,255,.7),transparent 38%); }
.person-art { position:absolute;right:22%;bottom:14%;font-size:10rem;filter:drop-shadow(0 20px 25px rgba(21,74,110,.18));animation:float 3.8s ease-in-out infinite;z-index:2; }
.knee-pulse { position:absolute;right:41%;bottom:40%;font-size:2rem;color:#ff5964;text-shadow:0 0 24px #ff7b84;animation:pulse 1.4s ease-in-out infinite;z-index:3; }
.orbit { position:absolute;border:1px solid rgba(21,126,223,.35);border-radius:50%;z-index:1; }
.orbit-a { width:310px;height:310px;right:10%;top:10%;animation:spin 16s linear infinite; }
.orbit-b { width:220px;height:220px;right:19%;top:20%;border-style:dashed;animation:spin 10s linear reverse infinite; }
.scan-line { position:absolute;left:18%;right:10%;height:2px;top:20%;background:#57a9ff;box-shadow:0 0 18px #57a9ff;z-index:4;animation:scan 3.6s ease-in-out infinite; }
.visual-label { position:absolute;right:7%;bottom:9%;background:rgba(255,255,255,.86);border:1px solid #d8e8f7;border-radius:15px;padding:.75rem 1rem;font-weight:800;color:#21567f;z-index:5;box-shadow:0 12px 25px rgba(30,85,125,.10); }
.visual-label small { font-weight:500;color:#7891a7; }

.overview-section,.how-section { padding:5rem .5rem 2rem; }
.section-heading { text-align:center;margin-bottom:2rem; }
.section-heading.left { text-align:left;margin-top:3.5rem; }
.section-heading h2 { margin:.2rem 0 .35rem; }
.section-heading p { margin:0; }
.feature-grid { display:grid;grid-template-columns:repeat(4,1fr);gap:1rem; }
.feature-card,.glass-section,.risk-dashboard {
  background:var(--card); border:1px solid var(--line); border-radius:20px;
  box-shadow:0 14px 38px rgba(40,91,132,.07); backdrop-filter:blur(12px);
}
.feature-card { padding:1.25rem;min-height:165px;transition:.25s ease; }
.feature-card:hover { transform:translateY(-5px);box-shadow:0 20px 45px rgba(30,101,159,.12); }
.feature-card h3 { font-size:1.05rem!important;margin:.8rem 0 .35rem; }
.feature-card p { font-size:.84rem;line-height:1.55; }
.feature-icon { width:43px;height:43px;border-radius:13px;display:grid;place-items:center;font-size:1.2rem;font-weight:800; }
.blue{background:#e6f3ff;color:#1677dc}.green{background:#e5f8f0;color:#16966f}.purple{background:#eeeaff;color:#775bd0}.teal{background:#e2f8f8;color:#178d91}

.workflow { display:grid;grid-template-columns:1fr auto 1fr auto 1fr auto 1fr;align-items:center;gap:.7rem;background:rgba(255,255,255,.6);border:1px solid var(--line);padding:1.7rem;border-radius:24px; }
.workflow-item { text-align:center; }
.workflow-circle { width:58px;height:58px;border-radius:50%;background:#e8f4ff;border:1px solid #d4e8fa;display:grid;place-items:center;margin:0 auto .65rem;font-size:1.4rem; }
.workflow-item b,.workflow-item span { display:block; }
.workflow-item span { font-size:.72rem;color:#7b91a6;margin-top:.2rem; }
.workflow-arrow { font-size:1.8rem;color:#197be0; }

.glass-section { padding:1.35rem;margin-bottom:1rem; }
.section-title { font-family:'Space Grotesk';font-weight:700;font-size:1.15rem;color:var(--ink);margin-bottom:1rem; }
.section-title small { display:block;font-family:'DM Sans';font-weight:500;color:#8195a8;font-size:.76rem;margin-top:.2rem; }
.sensor-card { min-height:410px; }
div[data-baseweb="input"]>div,div[data-baseweb="select"]>div,div[data-baseweb="textarea"]>div {
  border-radius:11px!important;border-color:#d9e7f3!important;background:rgba(255,255,255,.8)!important;
}
div[data-testid="stFileUploaderDropzone"] { border:1.5px dashed #a8c9e6!important;border-radius:15px!important;background:#fafdff!important; }
.stButton>button,.stDownloadButton>button { border-radius:11px!important;font-weight:700!important;min-height:2.65rem;transition:.2s ease!important; }
.stButton>button:hover,.stDownloadButton>button:hover { transform:translateY(-2px);box-shadow:0 8px 20px rgba(20,123,232,.16); }
div[data-testid="stMetric"] { background:rgba(248,252,255,.9);border:1px solid #e1edf7;border-radius:13px;padding:.65rem .75rem; }
div[data-testid="stMetricValue"] { color:var(--ink);font-family:'Space Grotesk'; }
.mini-result { margin-top:1rem;padding:1rem;background:#f7fbff;border:1px solid #e1edf7;border-radius:15px; }
.signal-box { margin-top:.7rem;padding:.65rem .8rem;border-radius:12px;background:#eef8ff; }
.signal-head { display:flex;justify-content:space-between;font-size:.72rem;color:#52728e; }
.signal-head span { color:#16a16e; }
.signal-wave { color:#2c91ed;font-size:1.25rem;letter-spacing:.08rem;overflow:hidden;white-space:nowrap;animation:wave 2.2s linear infinite; }
.imu-visual { display:flex;align-items:center;gap:1rem;padding:1.3rem;margin:.7rem 0;background:linear-gradient(135deg,#eff9ff,#eefbf6);border-radius:16px;border:1px solid #dcecf6; }
.imu-chip { padding:.7rem .9rem;border-radius:12px;background:#173e63;color:white;font-weight:800;font-size:.78rem; }
.imu-signal { flex:1;display:flex;gap:.3rem;align-items:center;justify-content:center; }
.imu-signal span { font-size:1.4rem;color:#238be6;animation:bob 1s ease-in-out infinite alternate; }
.imu-signal span:nth-child(2n) { color:#24a77b;animation-delay:.2s; }
.imu-visual small { color:#71889e; }

.results-heading { margin-bottom:1.1rem; }
.risk-dashboard { display:grid;grid-template-columns:.85fr 1.15fr 1fr;gap:1rem;padding:1.3rem;margin-bottom:1rem; }
.risk-main { text-align:center;padding:1rem;border-right:1px solid #e2edf6; }
.risk-gauge { width:190px;height:190px;border-radius:50%;margin:0 auto 1rem;background:conic-gradient(var(--risk) var(--score),#e2eaf2 0);display:grid;place-items:center;position:relative; }
.risk-gauge:before { content:"";position:absolute;inset:12px;border-radius:50%;background:white; }
.gauge-inner { position:relative;z-index:1;text-align:center; }
.gauge-inner strong { display:block;font:700 3.2rem/1 'Space Grotesk';color:var(--ink); }
.gauge-inner span { color:#8294a5;font-size:.8rem; }
.risk-label { font:700 1.25rem 'Space Grotesk';margin-bottom:.45rem; }
.risk-main p { font-size:.82rem;line-height:1.55;max-width:270px;margin:auto; }
.factor-panel,.recommend-panel { padding:.6rem .8rem; }
.factor-panel h3,.recommend-panel h3 { font-size:1rem!important;margin:.2rem 0 1rem; }
.factor-row { margin:.85rem 0; }
.factor-row>div:first-child { display:flex;justify-content:space-between;font-size:.8rem;color:#58718a; }
.factor-row b { color:var(--ink); }
.factor-track { height:8px;background:#e9f0f6;border-radius:99px;overflow:hidden;margin-top:.35rem; }
.factor-track i { display:block;height:100%;border-radius:99px;background:linear-gradient(90deg,#2587e8,#69b3ff); }
.recommend-panel { background:#effaf5;border-radius:16px;border:1px solid #d7eee3; }
.recommend-item { padding:.55rem 0;border-bottom:1px solid rgba(32,137,97,.1);font-size:.82rem;color:#2f705c; }
.recommend-item:last-child { border-bottom:0; }
.prototype-note { padding:1rem 1.1rem;border-radius:14px;background:#fff9e9;border:1px solid #f5e2aa;color:#7d6630;font-size:.78rem;line-height:1.55; }

.journey-banner { margin:3.5rem 0 1.5rem;padding:2.2rem 1.3rem;border-radius:25px;border:1px solid #cfe4f6;background:linear-gradient(120deg,#edf8ff,#f2fbf7);text-align:center;overflow:hidden;position:relative; }
.journey-banner:before { content:"";position:absolute;width:380px;height:180px;left:-100px;bottom:-120px;background:#cde8ff;border-radius:50%; }
.journey-title { font:700 1.45rem 'Space Grotesk';color:var(--ink); }
.journey-sub { color:#718ba1;font-size:.8rem;margin:.2rem 0 1.5rem; }
.journey-row { display:flex;justify-content:center;align-items:center;gap:1.2rem;position:relative;z-index:2; }
.journey-row div { min-width:125px; }
.journey-row span { width:48px;height:48px;border-radius:50%;display:grid;place-items:center;margin:0 auto .45rem;background:#fff;border:1px solid #d8e9f6;font-size:1.2rem; }
.journey-row b,.journey-row small { display:block; }
.journey-row b { font-size:.78rem; }
.journey-row small { color:#8297a9;font-size:.66rem; }
.journey-row i { font-style:normal;font-size:1.5rem;color:#2086e4; }

.footer-wrap { margin-top:3rem;padding:2.1rem 1.6rem;background:#102f4d;border-radius:24px;color:#dcebf7;display:grid;grid-template-columns:1.3fr 1.3fr .8fr 1fr;gap:2rem;align-items:center; }
.footer-logo { font:700 1.3rem 'Space Grotesk';color:#fff; }
.footer-brand small,.footer-wrap h4 { color:#91aec4;font-size:.72rem; }
.footer-wrap h4 { margin:0 0 .6rem;text-transform:uppercase;letter-spacing:.08em; }
.footer-contact div { font-size:.78rem;margin:.35rem 0; }
.social-row { display:flex;gap:.5rem; }
.social-row span { width:34px;height:34px;border-radius:10px;background:#193e5e;border:1px solid #2a5373;display:grid;place-items:center;font-weight:800;font-size:1.1rem;color:white; }
.footer-tag { text-align:right;color:#a9c0d1;font-size:.76rem;line-height:1.5; }
.copyright { text-align:center;color:#8ca1b2;font-size:.7rem;padding:1rem 0 .2rem; }

.reveal { animation:fadeUp .75s ease both; }
.feature-card:nth-child(2),.workflow-item:nth-child(3) { animation-delay:.08s; }
.feature-card:nth-child(3),.workflow-item:nth-child(5) { animation-delay:.16s; }
.feature-card:nth-child(4),.workflow-item:nth-child(7) { animation-delay:.24s; }

@keyframes fadeUp { from{opacity:0;transform:translateY(24px)} to{opacity:1;transform:translateY(0)} }
@keyframes float { 0%,100%{transform:translateY(0)} 50%{transform:translateY(-12px)} }
@keyframes pulse { 0%,100%{transform:scale(.8);opacity:.55} 50%{transform:scale(1.5);opacity:1} }
@keyframes spin { to{transform:rotate(360deg)} }
@keyframes scan { 0%,100%{top:18%;opacity:.25} 50%{top:75%;opacity:1} }
@keyframes wave { from{transform:translateX(0)} to{transform:translateX(-25px)} }
@keyframes bob { from{transform:translateY(-3px)} to{transform:translateY(3px)} }

@media(max-width:950px){
  .nav-links{display:none}.hero-section{grid-template-columns:1fr}.hero-visual{min-height:360px}.hero-copy{padding:3rem 1.5rem}.hero-copy h1{font-size:3.2rem!important}.feature-grid{grid-template-columns:repeat(2,1fr)}.risk-dashboard{grid-template-columns:1fr}.risk-main{border-right:0;border-bottom:1px solid #e2edf6;padding-bottom:1.5rem}.journey-row{flex-wrap:wrap}.footer-wrap{grid-template-columns:1fr 1fr}.footer-tag{text-align:left}
}
@media(max-width:600px){
  .block-container{padding:.5rem .75rem 2rem}.top-nav{position:relative}.hero-pills{grid-template-columns:1fr}.feature-grid{grid-template-columns:1fr}.workflow{grid-template-columns:1fr}.workflow-arrow{transform:rotate(90deg)}.footer-wrap{grid-template-columns:1fr}.hero-visual{min-height:300px}.person-art{font-size:7rem}
}
</style>
""",
    unsafe_allow_html=True,
)

# ----------------------------------------------------------------------
# Existing multilingual labels are retained below; all scoring and analysis
# functions remain unchanged.
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
    st.markdown("""
    <div class="side-brand">
        <div class="brand-mark">🌿</div>
        <div><b>Tyros</b><small>AI for a Healthier Northeast</small></div>
    </div>
    """, unsafe_allow_html=True)

    language = st.selectbox("Language / ভাষা / भाषा", list(LABELS.keys()))

    st.markdown("---")
    st.markdown("### 🗂️ Saved records")
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
            "⬇️ Download records (CSV)",
            data=csv_bytes,
            file_name="patient_records.csv",
            mime="text/csv",
        )

    st.markdown("---")
    st.caption(
        "Offline-first wearable workflow. IMU logs can be captured locally "
        "and synced to this app later at a PHC."
    )

L = LABELS[language]

if "gait_features" not in st.session_state:
    st.session_state.gait_features = None
if "imu_features" not in st.session_state:
    st.session_state.imu_features = None

# ----------------------------------------------------------------------
# HERO
# ----------------------------------------------------------------------
st.markdown(f"""
<div class="top-nav">
  <div class="nav-logo"><span>🌿</span> Tyros</div>
  <div class="nav-links">
    <a href="#overview">Home</a>
    <a href="#how-it-works">How it works</a>
    <a href="#screening">Screening</a>
    <a href="#results">Results</a>
    <a href="#contact">Contact</a>
  </div>
  <a class="nav-cta" href="#screening">Start Screening →</a>
</div>

<section class="hero-section" id="overview">
  <div class="hero-copy">
    <div class="eyebrow">✦ AI FOR A HEALTHIER NORTHEAST</div>
    <h1>{L["title"]}</h1>
    <h3>Early Detection. Stronger Generations.</h3>
    <p>{L["subtitle"]}</p>
    <div class="hero-actions">
      <a class="primary-link" href="#screening">Get Started →</a>
      <a class="secondary-link" href="#how-it-works">▶ Learn More</a>
    </div>
    <div class="hero-pills">
      <div>🧠 <b>AI Powered</b><small>Computer vision</small></div>
      <div>📡 <b>Wearable IMU</b><small>Motion signals</small></div>
      <div>📊 <b>Multimodal</b><small>Clinical + gait + IMU</small></div>
    </div>
  </div>
  <div class="hero-visual">
    <div class="orbit orbit-a"></div>
    <div class="orbit orbit-b"></div>
    <div class="scan-line"></div>
    <div class="person-art">🚶</div>
    <div class="knee-pulse">●</div>
    <div class="visual-label">AI ANALYSIS<br><small>Detecting movement patterns...</small></div>
  </div>
</section>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# OVERVIEW / FEATURE CARDS
# ----------------------------------------------------------------------
st.markdown("""
<section class="overview-section">
  <div class="section-heading">
    <div class="eyebrow">THE PLATFORM</div>
    <h2>A Complete Risk Assessment Solution</h2>
    <p>Combining computer vision, wearable sensors and clinical data for better insights.</p>
  </div>
  <div class="feature-grid">
    <div class="feature-card reveal"><div class="feature-icon blue">◉</div><h3>Video Gait Analysis</h3><p>AI pose estimation from a short side-on walking video.</p></div>
    <div class="feature-card reveal"><div class="feature-icon green">⌁</div><h3>Wearable IMU Analysis</h3><p>Motion-signal processing from ESP32 + MPU6050 data.</p></div>
    <div class="feature-card reveal"><div class="feature-icon purple">▥</div><h3>Multimodal Assessment</h3><p>Clinical + gait + IMU information in one screening score.</p></div>
    <div class="feature-card reveal"><div class="feature-icon teal">✓</div><h3>Community Healthcare</h3><p>Simple workflow designed for practical field screening.</p></div>
  </div>
</section>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# HOW IT WORKS
# ----------------------------------------------------------------------
st.markdown("""
<section class="how-section" id="how-it-works">
  <div class="section-heading">
    <div class="eyebrow">SIMPLE WORKFLOW</div>
    <h2>How It Works</h2>
    <p>From movement data to meaningful screening insights.</p>
  </div>
  <div class="workflow">
    <div class="workflow-item reveal"><div class="workflow-circle">👤</div><b>Patient Details</b><span>Basic clinical information</span></div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-item reveal"><div class="workflow-circle">📷</div><b>Upload & Analyse</b><span>Video and IMU data</span></div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-item reveal"><div class="workflow-circle">⚙️</div><b>AI Processing</b><span>Extract gait features</span></div>
    <div class="workflow-arrow">→</div>
    <div class="workflow-item reveal"><div class="workflow-circle">📈</div><b>Risk Assessment</b><span>Personalised result</span></div>
  </div>
</section>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# SCREENING INPUTS
# ----------------------------------------------------------------------
st.markdown("""
<section id="screening">
  <div class="section-heading left">
    <div class="eyebrow">START SCREENING</div>
    <h2>Patient & Movement Data</h2>
    <p>Enter the patient profile, then add camera and/or wearable sensor data.</p>
  </div>
</section>
""", unsafe_allow_html=True)

st.markdown(f'<div class="glass-section reveal"><div class="section-title">👤 {L["section1"]} <small>Let’s start with some basic information</small></div>', unsafe_allow_html=True)

p1, p2, p3 = st.columns([1.6, .8, .9])
with p1:
    patient_name = st.text_input("Patient name", placeholder="Enter patient name")
with p2:
    age = st.number_input("Age", min_value=10, max_value=100, value=45)
with p3:
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])

p4, p5, p6, p7 = st.columns(4)
with p4:
    height_cm = st.number_input("Height (cm)", min_value=100, max_value=220, value=160)
with p5:
    weight_kg = st.number_input("Weight (kg)", min_value=20, max_value=200, value=65)
with p6:
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
with p7:
    bmi = weight_kg / ((height_cm / 100) ** 2)
    st.metric("Calculated BMI", f"{bmi:.1f}")

p8, p9 = st.columns([1, 1])
with p8:
    pain_months = st.slider("Joint pain duration (months, 0 = none)", 0, 60, 0)
with p9:
    pain_severity = st.slider("Pain severity (0 = none, 10 = severe)", 0, 10, 0)
has_family_history = st.checkbox("Family history of OA / joint disease")

st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# VIDEO + IMU
# ----------------------------------------------------------------------
vcol, icol = st.columns(2)

with vcol:
    st.markdown(f'<div class="glass-section sensor-card reveal"><div class="section-title">📷 {L["section2"]} <small>AI pose estimation</small></div>', unsafe_allow_html=True)
    st.caption("Upload a 5–15 second side-on, full-body walking video.")
    video_file = st.file_uploader(
        "Upload walking video",
        type=["mp4", "mov", "avi"],
        key="video_uploader",
    )

    if video_file is not None:
        st.video(video_file)
        st.caption("Preview • Side-on walking video")

        if st.button("🏃 Run camera gait analysis", type="primary", use_container_width=True):
            with st.spinner("AI is analysing movement patterns..."):
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
        st.markdown('<div class="mini-result">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Left knee range", f"{gf['knee_flexion_range_left']}°")
        c2.metric("Right knee range", f"{gf['knee_flexion_range_right']}°")
        c3.metric("Asymmetry", f"{gf['asymmetry_index']}%")
        st.markdown(
            f"""<div class="signal-box">
              <div class="signal-head"><b>Movement signal</b><span>● analysed</span></div>
              <div class="signal-wave">〰〰〰〰〰〰〰〰〰〰〰〰〰</div>
            </div>""",
            unsafe_allow_html=True,
        )
        st.caption(
            f"Cadence proxy: {gf['cadence_proxy']} steps/sec · "
            f"Pose detected in {gf['pose_detection_rate']}% of frames"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

with icol:
    st.markdown(f'<div class="glass-section sensor-card reveal"><div class="section-title">📡 {L["section3"]} <small>ESP32 + MPU6050</small></div>', unsafe_allow_html=True)
    st.caption("Upload the recorded IMU CSV for sensor-based gait features.")
    imu_file = st.file_uploader(
        "Upload IMU data (CSV)",
        type=["csv"],
        key="imu_uploader",
    )

    if imu_file is not None:
        if st.button("📊 Run IMU gait analysis", type="primary", use_container_width=True):
            with st.spinner("Processing wearable motion signals..."):
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".csv", mode="wb"
                ) as tmp:
                    tmp.write(imu_file.read())
                    tmp_path = tmp.name
                try:
                    imu_feat = extract_imu_features(tmp_path)
                    st.session_state.imu_features = imu_feat.as_dict()
                    st.success("IMU analysis complete.")
                except Exception as e:
                    st.error(f"IMU analysis failed: {e}")
                finally:
                    os.unlink(tmp_path)

    st.markdown("""
    <div class="imu-visual">
      <div class="imu-chip">MPU6050</div>
      <div class="imu-signal">
        <span>↗</span><span>↘</span><span>↗</span><span>↘</span><span>↗</span><span>↘</span>
      </div>
      <small>3-axis motion signal • acceleration • gyroscope</small>
    </div>
    """, unsafe_allow_html=True)

    if st.session_state.imu_features:
        imf = st.session_state.imu_features
        st.markdown('<div class="mini-result">', unsafe_allow_html=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Peak angular velocity", f"{imf['peak_angular_velocity']}°/s")
        c2.metric("Jerk RMS", f"{imf['jerk_rms']}")
        asym_display = (
            f"{imf['left_right_asymmetry']}%"
            if imf["left_right_asymmetry"] is not None
            else "n/a"
        )
        c3.metric("L/R asymmetry", asym_display)
        st.caption(
            f"Cadence: {imf['cadence_imu']} steps/sec · "
            f"Samples — left: {imf['samples_left']}, right: {imf['samples_right']}"
        )
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('</div>', unsafe_allow_html=True)

# ----------------------------------------------------------------------
# RISK RESULT
# ----------------------------------------------------------------------
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

modalities_used = []
if st.session_state.gait_features:
    modalities_used.append("camera gait")
if st.session_state.imu_features:
    modalities_used.append("wearable IMU")

band_meta = {
    "Low": ("#16a34a", "Low Risk", "Routine monitoring — no immediate referral needed.", "✓"),
    "Moderate": ("#f59e0b", "Moderate Risk", "Consider lifestyle modification and re-screening / clinical follow-up.", "!"),
    "High": ("#ef4444", "High Risk", "Refer to PHC / orthopedic professional for clinical evaluation.", "!"),
}
risk_color, risk_title, risk_text, risk_icon = band_meta[result.band]

st.markdown('<div id="results"></div>', unsafe_allow_html=True)
st.markdown(f"""
<div class="section-heading left results-heading">
  <div class="eyebrow">AI SCREENING OUTPUT</div>
  <h2>{L["section4"]}</h2>
  <p>Your multimodal screening result is shown below.</p>
</div>
""", unsafe_allow_html=True)

score = max(0, min(float(result.score), 100.0))
st.markdown(f"""
<div class="risk-dashboard reveal">
  <div class="risk-main">
    <div class="risk-gauge" style="--score:{score}%; --risk:{risk_color};">
      <div class="gauge-inner">
        <strong>{score:.0f}</strong>
        <span>/ 100</span>
      </div>
    </div>
    <div class="risk-label" style="color:{risk_color};">{risk_icon} {risk_title}</div>
    <p>{risk_text}</p>
  </div>
  <div class="factor-panel">
    <h3>Contributing Factors</h3>
""", unsafe_allow_html=True)

for label, pts in result.contributing_factors:
    pct = max(4, min(100, float(pts) * 5))
    st.markdown(
        f"""<div class="factor-row">
          <div><span>{label}</span><b>+{pts:.1f}</b></div>
          <div class="factor-track"><i style="width:{pct}%;"></i></div>
        </div>""",
        unsafe_allow_html=True,
    )

st.markdown("</div><div class=\"recommend-panel\"><h3>💡 Recommendations</h3>", unsafe_allow_html=True)

if result.band == "High":
    recommendations = [
        "PHC / orthopedic clinical evaluation",
        "Physical examination and imaging if advised",
        "Follow local referral pathway",
    ]
elif result.band == "Moderate":
    recommendations = [
        "Maintain a healthy body weight",
        "Regular low-impact exercise",
        "Consult a healthcare professional if symptoms persist",
        "Periodic re-screening",
    ]
else:
    recommendations = [
        "Continue routine activity",
        "Maintain healthy weight",
        "Monitor symptoms",
        "Periodic re-screening",
    ]

for item in recommendations:
    st.markdown(f'<div class="recommend-item">✓ {item}</div>', unsafe_allow_html=True)

st.markdown("</div></div>", unsafe_allow_html=True)

if not modalities_used:
    st.info("Add camera gait and/or wearable IMU data for a more complete multimodal assessment.")
elif len(modalities_used) == 1:
    st.warning(f"Current score uses **{modalities_used[0]}** plus clinical inputs. Add the other modality for a fuller assessment.")
else:
    st.success("✓ Score currently fuses **camera gait + wearable IMU + clinical inputs**.")

# ----------------------------------------------------------------------
# SAVE RECORD
# ----------------------------------------------------------------------
save_col, note_col = st.columns([1, 2])
with save_col:
    if st.button("💾 Save patient record", type="primary", use_container_width=True):
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
            st.success(f"Record saved for {patient_name}.")

with note_col:
    st.markdown("""
    <div class="prototype-note">
      <b>⚠️ Screening support, not diagnosis</b><br>
      This prototype flags candidates for PHC-level follow-up. The current
      score is a transparent weighted-points model and should not replace
      clinical judgement.
    </div>
    """, unsafe_allow_html=True)

# ----------------------------------------------------------------------
# PROCESS BANNER
# ----------------------------------------------------------------------
st.markdown("""
<div class="journey-banner reveal">
  <div class="journey-title">From Movement Data to Better Lives</div>
  <div class="journey-sub">Early detection • stronger generations</div>
  <div class="journey-row">
    <div><span>👤</span><b>Patient</b><small>Community screening</small></div>
    <i>→</i>
    <div><span>📷</span><b>Video Analysis</b><small>AI pose estimation</small></div>
    <i>→</i>
    <div><span>〰</span><b>IMU Analysis</b><small>Motion signals</small></div>
    <i>→</i>
    <div><span>🧠</span><b>Fusion Engine</b><small>Multimodal AI</small></div>
    <i>→</i>
    <div><span>📈</span><b>Risk Assessment</b><small>Personalised result</small></div>
  </div>
</div>
""", unsafe_allow_html=True)

# ----------------------------------------------------------------------
# FOOTER
# ----------------------------------------------------------------------
st.markdown("""
<div id="contact" class="footer-wrap">
  <div class="footer-brand">
    <div class="footer-logo">🌿 <b>Tyros</b></div>
    <small>AI for a Healthier Northeast</small>
  </div>
  <div class="footer-contact">
    <h4>Contact Us</h4>
    <div>✉️ tyros2026@gmail.com</div>
    <div>☎️ 7979824251</div>
    <div>📍 GEC Vaishali</div>
  </div>
  <div class="footer-social">
    <h4>Follow Us</h4>
    <div class="social-row">
      <span title="Instagram">◎</span>
      <span title="X / Twitter">𝕏</span>
      <span title="Facebook">f</span>
    </div>
  </div>
  <div class="footer-tag">Made with ❤️<br>for a Healthier Tomorrow</div>
</div>
<div class="copyright">© 2026 Tyros. All rights reserved. • Prototype for demonstration</div>
""", unsafe_allow_html=True)
