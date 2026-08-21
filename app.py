import os
import logging

# Suppress HuggingFace transformers internal docstring warnings & hub warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

import tempfile
import base64
import io
from pathlib import Path
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.agent.router import AgenticRouter
from src.video.processor import VideoProcessor

# Page Configuration
st.set_page_config(
    page_title="Video Analytics & Intelligence Platform",
    page_icon="🎥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise Styling with High Text Contrast Rules
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background-color: #f8fafc;
    }
    
    /* Header Container */
    .header-container {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        padding: 2rem 2.5rem;
        border-radius: 12px;
        color: #ffffff !important;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(15, 23, 42, 0.08);
    }
    
    .header-title {
        font-size: 2.1rem;
        font-weight: 700;
        letter-spacing: -0.02em;
        margin: 0;
        color: #ffffff !important;
    }
    
    .header-subtitle {
        font-size: 1.0rem;
        color: #94a3b8 !important;
        margin-top: 0.4rem;
        font-weight: 400;
    }

    /* Metric Cards */
    .metric-card {
        background: #ffffff !important;
        border: 1px solid #cbd5e1;
        border-radius: 8px;
        padding: 1rem 1.25rem;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0,0,0,0.04);
    }
    .metric-val {
        font-size: 1.6rem;
        font-weight: 700;
        color: #0f172a !important;
    }
    .metric-lbl {
        font-size: 0.825rem;
        color: #475569 !important;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 0.2rem;
    }

    /* Timestamp Tag */
    .ts-tag {
        background-color: #e2e8f0 !important;
        color: #0f172a !important;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 3px 8px;
        border-radius: 4px;
        border: 1px solid #94a3b8;
        font-family: monospace;
    }

    /* Clean High-Contrast Cards */
    .content-card {
        background-color: #ffffff !important;
        color: #0f172a !important;
        border: 1px solid #cbd5e1;
        border-radius: 10px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
        font-size: 1.0rem;
        line-height: 1.6;
    }
    .content-card p, .content-card div, .content-card span, .content-card li {
        color: #0f172a !important;
    }

    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        border-bottom: 2px solid #e2e8f0;
    }
    .stTabs [data-baseweb="tab"] {
        height: 48px;
        white-space: pre;
        border-radius: 6px 6px 0 0;
        font-weight: 600;
        font-size: 0.95rem;
        color: #475569 !important;
    }
    .stTabs [aria-selected="true"] {
        color: #2563eb !important;
        border-bottom: 3px solid #2563eb !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Banner Header
st.markdown("""
    <div class="header-container">
        <div class="header-title">Video Analytics & Intelligence Platform</div>
        <div class="header-subtitle">Automated Scene Indexing, Multimodal Search & Natural Language Reasoning Engine</div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration
st.sidebar.markdown("### ⚙️ Workspace Control")

api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("API Access Key:", type="password")

st.sidebar.markdown("---")
st.sidebar.markdown("### 📹 Media Input")
input_option = st.sidebar.radio("Input Selection Method:", ["Upload File", "System Path"])

temp_video_path = None
if input_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Select Video File:", type=["mp4", "mkv", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
else:
    path_input = st.sidebar.text_input("Enter Video Path:", value="sample.mp4")
    if path_input and Path(path_input).exists():
        temp_video_path = path_input
    elif path_input:
        st.sidebar.error(f"Path invalid or file missing: {path_input}")

# Main Content Layout
if temp_video_path and Path(temp_video_path).exists():
    router = AgenticRouter(api_key=api_key if api_key else None)
    path_obj = Path(temp_video_path)
    video_id = path_obj.stem

    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        st.subheader("Media Player")
        st.video(temp_video_path)

    with col_right:
        st.subheader("Index Overview")
        
        # Check database stats if indexed
        shot_count = 0
        kf_count = 0
        with router.indexer._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT shot_count FROM videos WHERE video_id = ?", (video_id,))
            row = cursor.fetchone()
            if row:
                shot_count = row["shot_count"]
                cursor.execute(
                    "SELECT COUNT(*) as kfc FROM keyframes k JOIN shots s ON k.shot_id = s.shot_id WHERE s.video_id = ?",
                    (video_id,)
                )
                kf_row = cursor.fetchone()
                kf_count = kf_row["kfc"] if kf_row else 0

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{shot_count}</div><div class="metric-lbl">Shots</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{kf_count}</div><div class="metric-lbl">Keyframes</div></div>', unsafe_allow_html=True)
        with m3:
            status_text = "Ready" if shot_count > 0 else "Pending"
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{"#16a34a" if shot_count > 0 else "#ea580c"}">{status_text}</div><div class="metric-lbl">Status</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("⚡ Process & Index Media", use_container_width=True, type="secondary"):
            with st.spinner("Processing video shots, extracting keyframes & indexing visual vectors..."):
                def log_status(msg):
                    st.toast(msg, icon="ℹ️")
                router.ensure_indexed(temp_video_path, verbose_callback=log_status)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # Workspace Tabs
    tab1, tab2, tab3 = st.tabs(["💬 Search & Analytics", "🧩 Scene & Storyboard Explorer", "📄 Executive Summary"])

    # TAB 1: Search & Q&A
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        q_col1, q_col2 = st.columns([3.5, 1])
        with q_col1:
            user_query = st.text_input("Natural Language Query:", placeholder="e.g. Describe what happened between 5 seconds to 15 seconds...", label_visibility="collapsed")
        with q_col2:
            run_btn = st.button("Run Analytics", use_container_width=True, type="primary")

        if run_btn and user_query.strip():
            with st.spinner("Executing multimodal vector search & synthesizing intelligence..."):
                result = router.answer_query(temp_video_path, user_query)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Analysis Result")
            ans_content = result.get("answer", "No analysis output generated.")
            st.markdown(f'<div class="content-card">{ans_content}</div>', unsafe_allow_html=True)

            if "observations" in result:
                with st.expander("Detailed Grounded Evidence", expanded=True):
                    events = result["observations"].get("events", [])
                    for ev in events:
                        ts = f"<span class='ts-tag'>{ev.get('start_time')} - {ev.get('end_time')}</span>"
                        objs = ", ".join(ev.get("visible_objects", []))
                        st.markdown(f"{ts} &nbsp; **{ev.get('description')}**", unsafe_allow_html=True)
                        if objs:
                            st.caption(f"Detected Attributes: {objs}")
                        st.markdown("---")

    # TAB 2: Storyboard Explorer
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        with router.indexer._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM shots WHERE video_id = ? ORDER BY shot_index ASC", (video_id,))
            shots_rows = cursor.fetchall()

        if not shots_rows:
            st.info("Media file has not been processed yet. Click 'Process & Index Media' above.")
        else:
            for row in shots_rows:
                with st.container():
                    st.markdown(f"##### Shot #{row['shot_index'] + 1} &nbsp; <span class='ts-tag'>{row['start_ts']} ➔ {row['end_ts']}</span>", unsafe_allow_html=True)
                    tags = row['tags_json']
                    if tags and tags != "[]":
                        st.caption(f"Detected Object Tags: `{tags}`")

                    b64_str = row['storyboard_b64']
                    if b64_str:
                        try:
                            header, data = b64_str.split(",")
                            img_bytes = base64.b64decode(data)
                            img = Image.open(io.BytesIO(img_bytes))
                            st.image(img, use_container_width=True)
                        except Exception as e:
                            st.error(f"Error rendering storyboard: {e}")
                    st.markdown("<br>", unsafe_allow_html=True)

    # TAB 3: Summary
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Narrative Summary", type="primary"):
            with st.spinner("Synthesizing full video narrative..."):
                summary_text = router.summarize_video(temp_video_path)
                st.markdown(f'<div class="content-card">{summary_text}</div>', unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #64748b;">
            <h4>No Media Selected</h4>
            <p>Please select a video file or path from the control panel in the left sidebar.</p>
        </div>
    """, unsafe_allow_html=True)
