import os
import logging

# Suppress HuggingFace transformers internal docstring warnings & hub warnings
os.environ["TRANSFORMERS_VERBOSITY"] = "error"
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
logging.getLogger("transformers").setLevel(logging.ERROR)

import tempfile
import base64
import io
import json
import sqlite3
from pathlib import Path
import streamlit as st
from PIL import Image
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

import importlib
import sys
import src.agent.router
importlib.reload(src.agent.router)
from src.agent.router import AgenticRouter
from src.video.processor import VideoProcessor

# Page Configuration
st.set_page_config(
    page_title="AgenticVDO — Matrix Terminal Video Intelligence",
    layout="wide",
    initial_sidebar_state="expanded"
)

# VIGIL Black-Green Matrix Design System & JetBrains Mono Typography
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:ital,wght@0,300..800;1,300..800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'JetBrains Mono', monospace !important;
    }
    
    .stApp {
        background-color: #050505 !important;
        color: #e0ffe0 !important;
    }

    /* Sidebar Styling */
    section[data-testid="stSidebar"] {
        background-color: #0d0d0d !important;
        border-right: 1px solid rgba(0, 255, 65, 0.18) !important;
    }
    section[data-testid="stSidebar"] * {
        color: #e0ffe0 !important;
    }
    
    /* Header Container */
    .header-container {
        background: #0d0d0d !important;
        padding: 2.2rem 2.5rem;
        border-radius: 14px;
        border: 1px solid rgba(0, 255, 65, 0.25) !important;
        box-shadow: 0 0 25px rgba(0, 255, 65, 0.15), 0 8px 32px rgba(0, 0, 0, 0.8) !important;
        margin-bottom: 2rem;
    }
    
    .header-title-row {
        display: flex;
        align-items: center;
        justify-content: space-between;
        flex-wrap: wrap;
        gap: 1rem;
    }

    .header-title {
        font-size: 2.2rem;
        font-weight: 800;
        letter-spacing: -0.03em;
        margin: 0;
        background: linear-gradient(135deg, #b9f6ca 0%, #00ff41 50%, #69f0ae 100%) !important;
        -webkit-background-clip: text !important;
        -webkit-text-fill-color: transparent !important;
    }
    
    .header-subtitle {
        font-size: 1.0rem;
        color: #7ca97c !important;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    .badge-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(0, 255, 65, 0.12) !important;
        border: 1px solid rgba(0, 255, 65, 0.35) !important;
        color: #00ff41 !important;
        font-size: 0.825rem;
        font-weight: 700;
        padding: 5px 14px;
        border-radius: 9999px;
        box-shadow: 0 0 12px rgba(0, 255, 65, 0.2) !important;
    }

    /* Metric Cards */
    .metric-card {
        background: #0d0d0d !important;
        border: 1px solid rgba(0, 255, 65, 0.22) !important;
        border-radius: 12px;
        padding: 1.2rem 1.4rem;
        text-align: center;
        box-shadow: 0 4px 16px rgba(0, 0, 0, 0.6);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        border-color: #00ff41 !important;
        box-shadow: 0 0 18px rgba(0, 255, 65, 0.3) !important;
        transform: translateY(-2px);
    }
    .metric-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00ff41 !important;
        text-shadow: 0 0 10px rgba(0, 255, 65, 0.4);
    }
    .metric-lbl {
        font-size: 0.75rem;
        color: #7ca97c !important;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        font-weight: 600;
        margin-top: 0.3rem;
    }

    /* Timestamp Tag */
    .ts-tag {
        background-color: rgba(0, 255, 65, 0.12) !important;
        color: #00ff41 !important;
        font-weight: 700;
        font-size: 0.85rem;
        padding: 4px 10px;
        border-radius: 6px;
        border: 1px solid rgba(0, 255, 65, 0.35) !important;
    }

    /* Content Cards & Expanders */
    .content-card {
        background-color: #0d0d0d !important;
        color: #e0ffe0 !important;
        border: 1px solid rgba(0, 255, 65, 0.22) !important;
        border-radius: 12px;
        padding: 1.6rem;
        margin-bottom: 1.2rem;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.6);
        font-size: 1.05rem;
        line-height: 1.7;
    }
    .content-card p, .content-card div, .content-card span, .content-card li {
        color: #e0ffe0 !important;
    }

    /* Streamlit Buttons - VIGIL High-Contrast Matrix Green */
    .stButton > button {
        background: linear-gradient(135deg, #00ff41 0%, #00cc33 100%) !important;
        color: #000000 !important;
        font-weight: 800 !important;
        border: none !important;
        border-radius: 8px !important;
        box-shadow: 0 2px 14px rgba(0, 255, 65, 0.4) !important;
        transition: all 0.2s ease !important;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #33ff66 0%, #00ff41 100%) !important;
        box-shadow: 0 4px 22px rgba(0, 255, 65, 0.6) !important;
        transform: translateY(-1px) !important;
        color: #000000 !important;
    }

    /* Tabs Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
        border-bottom: 2px solid rgba(0, 255, 65, 0.18) !important;
        padding-bottom: 4px;
    }
    .stTabs [data-baseweb="tab"], button[data-baseweb="tab"], div[data-testid="stTab"] {
        height: 50px;
        white-space: pre;
        border-radius: 8px 8px 0 0;
        font-weight: 700 !important;
        font-size: 0.95rem;
        color: #7ca97c !important;
        background-color: transparent;
        border: none;
    }
    .stTabs [data-baseweb="tab"] *, button[data-baseweb="tab"] *, div[data-testid="stTab"] * {
        color: #7ca97c !important;
        font-weight: 700 !important;
    }
    .stTabs [aria-selected="true"], button[data-baseweb="tab"][aria-selected="true"], div[data-testid="stTab"][data-selected="true"] {
        color: #00ff41 !important;
        border-bottom: 3px solid #00ff41 !important;
        background-color: #0d0d0d !important;
        border-radius: 8px 8px 0 0;
    }
    .stTabs [aria-selected="true"] *, div[data-testid="stTab"][data-selected="true"] * {
        color: #00ff41 !important;
    }

    /* Expander Styling */
    div[data-testid="stExpander"] {
        background-color: #0d0d0d !important;
        border: 1px solid rgba(0, 255, 65, 0.22) !important;
        border-radius: 10px !important;
    }
    div[data-testid="stExpander"] summary, div[data-testid="stExpander"] summary * {
        color: #00ff41 !important;
        font-weight: 700 !important;
    }
    div[data-testid="stExpander"] div[data-testid="stMarkdownContainer"] *,
    div[data-testid="stExpander"] div[data-testid="stCaptionContainer"] *,
    div[data-testid="stExpander"] p,
    div[data-testid="stExpander"] strong,
    div[data-testid="stExpander"] span {
        color: #e0ffe0 !important;
    }
    div[data-testid="stCaptionContainer"], div[data-testid="stCaptionContainer"] p {
        color: #7ca97c !important;
        font-weight: 600 !important;
    }

    /* Input & Select Box Dark Base */
    input, select, textarea {
        background-color: #111111 !important;
        color: #e0ffe0 !important;
        border: 1px solid rgba(0, 255, 65, 0.22) !important;
    }
    input:focus, select:focus, textarea:focus {
        border-color: #00ff41 !important;
        box-shadow: 0 0 0 3px rgba(0, 255, 65, 0.3) !important;
    }
    </style>
""", unsafe_allow_html=True)

# Main Hero Banner Header
st.markdown("""
    <div class="header-container">
        <div class="header-title-row">
            <div>
                <div class="header-title">AgenticVDO — Matrix Terminal</div>
                <div class="header-subtitle">Autonomous Multimodal Scene Indexing, Zero-Shot Search & Temporal Reasoning Engine</div>
            </div>
            <div>
                <span class="badge-pill">Zero-LLM Mode ($0 Cost)</span>
                <span class="badge-pill">Latency: &lt; 50ms</span>
            </div>
        </div>
    </div>
""", unsafe_allow_html=True)

# Sidebar Configuration & Controls
st.sidebar.markdown("## TERMINAL CONTROLS")

api_key = os.getenv("GROQ_API_KEY", "")
if not api_key:
    api_key = st.sidebar.text_input("Groq API Access Key:", type="password", help="Required only for Groq Cloud VLM Mode")

st.sidebar.markdown("---")
st.sidebar.markdown("### MEDIA SOURCE")
input_option = st.sidebar.radio("Selection Method:", ["Upload File", "System Path"], index=1)

temp_video_path = None
if input_option == "Upload File":
    uploaded_file = st.sidebar.file_uploader("Select Video File:", type=["mp4", "mkv", "avi", "mov"])
    if uploaded_file is not None:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix)
        tfile.write(uploaded_file.read())
        temp_video_path = tfile.name
else:
    path_input = st.sidebar.text_input("Enter System Video Path:", value="input_vdo/Stealing095_x264.mp4")
    if path_input and Path(path_input).exists():
        temp_video_path = path_input
    elif path_input:
        st.sidebar.error(f"Path invalid or file missing: {path_input}")

st.sidebar.markdown("---")
st.sidebar.markdown("### INTELLIGENCE ENGINE MODE")
engine_mode = st.sidebar.radio(
    "Engine Selection:",
    ["Zero-LLM Mode ($0 Cost, Instant CPU)", "Groq VLM Cloud Mode"],
    index=0
)

st.sidebar.markdown("---")
if st.sidebar.button("Purge & Reset Database Cache", use_container_width=True):
    conn = sqlite3.connect("video_index.db")
    conn.execute("DELETE FROM keyframes")
    conn.execute("DELETE FROM shots")
    conn.execute("DELETE FROM videos")
    conn.execute("DELETE FROM vlm_observations")
    conn.commit()
    conn.close()
    st.sidebar.success("Database cache completely purged!")
    st.rerun()

# Main Workspace Content
if temp_video_path and Path(temp_video_path).exists():
    router = AgenticRouter(api_key=api_key if api_key else None)
    path_obj = Path(temp_video_path)
    video_id = path_obj.stem

    col_left, col_right = st.columns([1.1, 1.0])

    with col_left:
        st.subheader("Media Viewport")
        st.video(temp_video_path)

    with col_right:
        st.subheader("Indexing & Scene Overview")
        
        # Check database stats
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
            st.markdown(f'<div class="metric-card"><div class="metric-val">{shot_count}</div><div class="metric-lbl">Shots / Windows</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="metric-card"><div class="metric-val">{kf_count}</div><div class="metric-lbl">Visual Keyframes</div></div>', unsafe_allow_html=True)
        with m3:
            status_text = "READY" if shot_count > 0 else "PENDING"
            color_code = "#00ff41" if shot_count > 0 else "#f59e0b"
            st.markdown(f'<div class="metric-card"><div class="metric-val" style="color:{color_code}">{status_text}</div><div class="metric-lbl">Index Status</div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Process & Ingest Video Vectors", use_container_width=True):
            with st.spinner("Executing PySceneDetect segmentation, YOLOv8 object tag extraction & OpenCLIP vector indexing..."):
                def log_status(msg):
                    st.toast(msg)
                router.ensure_indexed(temp_video_path, verbose_callback=log_status)
                st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    # 4 Enterprise Workspace Tabs
    tab1, tab2, tab3, tab4 = st.tabs([
        "Search & Analytics",
        "Storyboard & Keyframe Explorer",
        "Executive Narrative & Event Log",
        "Engine Architecture"
    ])

    # TAB 1: Search & Analytics
    with tab1:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Sample Demo Queries")
        
        demo_cols = st.columns(4)
        sample_query = ""
        with demo_cols[0]:
            if st.button("Vehicle Loitering & Theft", use_container_width=True):
                sample_query = "is anyone stealing or tampering with a vehicle?"
        with demo_cols[1]:
            if st.button("People & Objects Scene", use_container_width=True):
                sample_query = "describe all people, cars, and motorcycles visible"
        with demo_cols[2]:
            if st.button("Timestamp 10s - 20s", use_container_width=True):
                sample_query = "what happens between 10 seconds to 20 seconds?"
        with demo_cols[3]:
            if st.button("Romance / Casual Walk", use_container_width=True):
                sample_query = "do you see any romantic activity or people walking together?"

        q_col1, q_col2 = st.columns([3.5, 1])
        with q_col1:
            initial_val = sample_query if sample_query else ""
            user_query = st.text_input("Natural Language Query:", value=initial_val, placeholder="e.g. Is anyone stealing or loitering near vehicles?", label_visibility="collapsed")
        with q_col2:
            run_btn = st.button("Run Analytics", use_container_width=True)

        if (run_btn or sample_query) and user_query.strip():
            with st.spinner("Executing OpenCLIP vector alignment & synthesizing intelligence..."):
                selected_mode = "zero_llm" if "Zero-LLM" in engine_mode else "groq_vlm"
                result = router.answer_query(temp_video_path, user_query, mode=selected_mode)

            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("##### Grounded Analysis Result")
            ans_content = result.get("answer", "No analysis output generated.")
            st.markdown(f'<div class="content-card">{ans_content}</div>', unsafe_allow_html=True)

            if "observations" in result:
                with st.expander("Grounded Visual & Temporal Evidence", expanded=True):
                    events = result["observations"].get("events", [])
                    for ev in events:
                        ts = f"<span class='ts-tag'>{ev.get('start_time')} - {ev.get('end_time')}</span>"
                        desc = ev.get('description') or f"Video segment from {ev.get('start_time')} to {ev.get('end_time')}."
                        st.markdown(f"{ts} &nbsp; **{desc}**", unsafe_allow_html=True)

                        objs = ", ".join(ev.get("visible_objects", []))
                        if objs:
                            st.caption(f"Detected Objects: {objs}")

                        interactions = ", ".join(ev.get("physical_interactions", []))
                        if interactions:
                            st.caption(f"Physical Interactions: {interactions}")

                        rating = ev.get("suspicion_rating")
                        if rating and rating.upper() in ["HIGH", "MEDIUM"]:
                            st.markdown(f"**Suspicion Rating**: `{rating}` &nbsp; *{ev.get('suspicion_reason', '')}*")

                        rec = ev.get("recommendation")
                        if rec:
                            st.warning(f"{rec}")

                        st.markdown("---")

    # TAB 2: Storyboard Explorer
    with tab2:
        st.markdown("<br>", unsafe_allow_html=True)
        with router.indexer._get_conn() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM shots WHERE video_id = ? ORDER BY shot_index ASC", (video_id,))
            shots_rows = cursor.fetchall()

        if not shots_rows:
            st.info("Media file has not been processed yet. Click 'Process & Ingest Video Vectors' above.")
        else:
            for row in shots_rows:
                with st.container():
                    st.markdown(f"##### Shot #{row['shot_index'] + 1} &nbsp; <span class='ts-tag'>{row['start_ts']} ➔ {row['end_ts']}</span>", unsafe_allow_html=True)
                    tags = row['tags_json']
                    if tags and tags != "[]":
                        st.caption(f"YOLOv8 Detected Tags: `{tags}`")

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

    # TAB 3: Summary & Event Log
    with tab3:
        st.markdown("<br>", unsafe_allow_html=True)
        if st.button("Generate Executive Narrative Summary & Event Log"):
            with st.spinner("Synthesizing full video narrative & chronological window log..."):
                selected_mode = "zero_llm" if "Zero-LLM" in engine_mode else "groq_vlm"
                res = router.generate_full_video_log(temp_video_path, mode=selected_mode)

                st.markdown("##### Executive Narrative Overview")
                st.markdown(f'<div class="content-card">{res["summary"]}</div>', unsafe_allow_html=True)

                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("##### Chronological Window-by-Window Event Log ('Time Talks' Analytics)")

                if res.get("csv_payload"):
                    st.download_button(
                        label="Download Event Log (.CSV / Excel)",
                        data=res["csv_payload"],
                        file_name=f"{video_id}_event_log.csv",
                        mime="text/csv"
                    )

                if res.get("events_log"):
                    st.dataframe(res["events_log"], use_container_width=True)

    # TAB 4: Engine Architecture & Metrics
    with tab4:
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("##### Agentic System Architecture & Vector Indexing Specifications")
        
        st.markdown("""
            <div class="content-card">
                <h4>Agentic Engine Specifications (VIGIL Design Standard)</h4>
                <ul>
                    <li><b>Orchestrator Agent</b>: <code>AgenticRouter</code> with autonomous query intent parsing & mode dispatch.</li>
                    <li><b>Zero-Shot Visual Vector Space</b>: <code>OpenCLIP (openai/clip-vit-base-patch32)</code> with 512-dimensional joint text-image embedding space.</li>
                    <li><b>Object Perception Sensor</b>: <code>YOLOv8</code> object detector for frame-level tag extraction.</li>
                    <li><b>Temporal Segmentation</b>: <code>PySceneDetect</code> content detector for adaptive shot window boundary splitting.</li>
                    <li><b>Vector Indexing & Storage</b>: SQLite local persistent database (<code>video_index.db</code>) with BLOB vector embeddings.</li>
                    <li><b>Execution Profile</b>: 100% Local CPU Execution, <b>$0 API Cost</b>, <b>&lt; 50ms Search Latency</b>.</li>
                </ul>
            </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("""
        <div style="text-align: center; padding: 5rem 2rem; color: #7ca97c;">
            <h3 style="color: #00ff41;">No Media Selected</h3>
            <p style="color: #7ca97c;">Please upload a video file or specify a system path from the terminal control panel in the left sidebar to launch analytics.</p>
        </div>
    """, unsafe_allow_html=True)
