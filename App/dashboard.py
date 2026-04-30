from __future__ import annotations

import logging
import tempfile
from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

from analyzer import (
    POSES_DIR,
    analyze_pose_file,
    analyze_swing,
    list_pose_files,
    load_template,
)
from event_detector import GolfDBEventDetector, convert_app_events_to_analysis_indices
from feedback_api import generate_ai_feedback
from video_pipeline import process_video_to_model_input

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="Virtual Golf Coach",
    page_icon="⛳",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design tokens ─────────────────────────────────────────────────────────────
ACCENT   = "#c8f135"
GREEN    = "#4caf72"
AMBER    = "#f5a623"
RED      = "#ff4d4d"
BG       = "#0a0c0f"
SURFACE  = "#111417"
SURFACE2 = "#181b1f"
BORDER   = "#1e2226"
TEXT     = "#e8eaed"
TEXT2    = "#8a93a0"
MUTED    = "#4a5260"

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [data-testid="stAppViewContainer"], [data-testid="stApp"] {{
    background: {BG} !important;
    color: {TEXT} !important;
    font-family: 'DM Sans', sans-serif !important;
}}
[data-testid="stHeader"] {{ background: {SURFACE} !important; border-bottom: 1px solid {BORDER}; }}
[data-testid="stSidebar"] {{ background: {SURFACE} !important; border-right: 1px solid {BORDER} !important; }}
[data-testid="stSidebar"] * {{ color: {TEXT} !important; }}
section.main > div {{ padding-top: 1.5rem !important; }}

h1, h2, h3 {{ font-family: 'Bebas Neue', sans-serif !important; letter-spacing: 2px !important; color: {TEXT} !important; }}
h1 {{ font-size: 2.2rem !important; }}
p, li {{ color: {TEXT2} !important; font-size: 0.875rem !important; }}

.stButton > button {{
    background: {ACCENT} !important; color: {BG} !important; border: none !important;
    font-family: 'DM Sans', sans-serif !important; font-weight: 500 !important;
    font-size: 0.85rem !important; border-radius: 6px !important;
    padding: 0.5rem 1.5rem !important; letter-spacing: 0.5px !important;
}}
.stButton > button:hover {{ opacity: 0.88 !important; }}

[data-testid="stFileUploader"] {{
    background: {SURFACE2} !important; border: 1px dashed {BORDER} !important;
    border-radius: 10px !important; padding: 1rem !important;
}}
[data-testid="stSelectbox"] > div {{
    background: {SURFACE2} !important; border: 1px solid {BORDER} !important;
    border-radius: 8px !important; color: {TEXT} !important;
}}
[data-testid="stMetric"] {{
    background: {SURFACE} !important; border: 1px solid {BORDER} !important;
    border-radius: 10px !important; padding: 0.8rem 1rem !important;
}}
[data-testid="stMetricLabel"] {{
    color: {TEXT2} !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important; letter-spacing: 1px !important; text-transform: uppercase !important;
}}
[data-testid="stMetricValue"] {{
    color: {TEXT} !important; font-family: 'Bebas Neue', sans-serif !important; font-size: 2rem !important;
}}
hr {{ border-color: {BORDER} !important; margin: 0.8rem 0 !important; }}
[data-testid="stCaptionContainer"] {{
    color: {MUTED} !important; font-family: 'DM Mono', monospace !important; font-size: 0.7rem !important;
}}
[data-testid="stExpander"] {{
    background: {SURFACE} !important; border: 1px solid {BORDER} !important; border-radius: 8px !important;
}}
[data-testid="stExpander"] summary {{
    color: {TEXT2} !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important; letter-spacing: 1px !important;
}}
[data-baseweb="tab-list"] {{
    gap: 4px;
}}
button[role="tab"] {{
    background: {SURFACE2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 8px !important;
    color: {TEXT2} !important;
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 1px !important;
}}
button[aria-selected="true"][role="tab"] {{
    color: {ACCENT} !important;
    border-color: {ACCENT}55 !important;
}}
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: {BG}; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 2px; }}
@keyframes pulse {{ 0%,100% {{ opacity:1 }} 50% {{ opacity:0.3 }} }}
</style>
""", unsafe_allow_html=True)

# ── Cached resources ──────────────────────────────────────────────────────────

@st.cache_resource
def cached_load_template():
    return load_template()

@st.cache_resource
def cached_golfdb_detector():
    return GolfDBEventDetector()

@st.cache_data
def cached_list_pose_files():
    return list_pose_files(POSES_DIR)

# ── Helpers ───────────────────────────────────────────────────────────────────

GOLFDB_EVENT_LABELS = {
    "address": "Address",
    "toe_up": "Toe-up",
    "mid_backswing": "Mid-backswing",
    "top_backswing": "Top",
    "mid_downswing": "Mid-downswing",
    "impact": "Impact",
    "mid_follow_through": "Mid-follow-through",
    "finish": "Finish",
}

def score_color(score: float) -> str:
    return GREEN if score >= 85 else (AMBER if score >= 70 else RED)

def score_label(score: float) -> str:
    return "Good" if score >= 85 else ("Fair" if score >= 70 else "Weak")

def deviation_color(sigma: float) -> str:
    return GREEN if sigma <= 1.0 else (AMBER if sigma <= 2.0 else RED)

def _extract_frame(video_path: Path, frame_idx: int):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total <= 0:
        cap.release()
        return None
    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, min(frame_idx, total - 1)))
    ret, frame = cap.read()
    cap.release()
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if ret else None

def _analysis_to_raw(idx: int, n_analysis: int, n_raw: int) -> int:
    if n_analysis <= 1 or n_raw <= 1:
        return 0
    return int((idx / (n_analysis - 1)) * (n_raw - 1))

# ── HTML components ───────────────────────────────────────────────────────────

def render_score_hero(result: dict, source_label: str):
    score  = result["overall_score"]
    grade  = result["grade"]
    circ   = 226
    offset = circ * (1 - score / 100)
    color  = score_color(score)
    events = result.get("events", {})

    event_html = "".join(f"""
        <div style="text-align:center">
          <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:{TEXT};line-height:1">{events.get(k, "—")}</div>
          <div style="font-family:'DM Mono',monospace;font-size:9px;color:{MUTED};letter-spacing:1px;margin-top:2px">{label}</div>
        </div>
    """ for k, label in [("top_backswing","TOP"),("impact","IMPACT"),("finish","FINISH")])

    html = f"""
    <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:12px;
         padding:20px 28px;display:flex;align-items:center;gap:28px;
         position:relative;overflow:hidden;margin-bottom:4px;
         box-sizing:border-box;width:100%;">
      <div style="position:absolute;inset:0;background:radial-gradient(ellipse at 0% 50%,{color}08 0%,transparent 65%);pointer-events:none"></div>

      <div style="position:relative;width:88px;height:88px;flex-shrink:0">
        <svg width="88" height="88" viewBox="0 0 88 88" style="transform:rotate(-90deg)">
          <circle cx="44" cy="44" r="36" fill="none" stroke="{BORDER}" stroke-width="6"/>
          <circle cx="44" cy="44" r="36" fill="none" stroke="{color}" stroke-width="6"
            stroke-linecap="round" stroke-dasharray="{circ}" stroke-dashoffset="{offset:.1f}"/>
        </svg>
        <div style="position:absolute;inset:0;display:flex;flex-direction:column;align-items:center;justify-content:center">
          <span style="font-family:'Bebas Neue',sans-serif;font-size:26px;color:{color};line-height:1">{score:.0f}</span>
          <span style="font-family:'DM Mono',monospace;font-size:9px;color:{MUTED};letter-spacing:1px">/100</span>
        </div>
      </div>

      <div style="flex:1">
        <div style="font-family:'Bebas Neue',sans-serif;font-size:52px;line-height:1;color:{color};letter-spacing:3px">{grade}</div>
        <div style="font-family:'DM Mono',monospace;font-size:11px;color:{TEXT2};margin-top:4px">{source_label}</div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};margin-top:2px">
          {result['num_frames']} frames · worst at frame {result['worst_frame']}
        </div>
      </div>

      <div style="display:flex;gap:24px;padding-left:20px;border-left:1px solid {BORDER}">
        {event_html}
      </div>
    </div>
    """

    components.html(html, height=150, scrolling=False)

def render_phase_cards(result: dict, video_path=None, num_raw_frames=None, raw_phase_frames=None):
    phases    = result["phases"]
    best_idx  = max(range(len(phases)), key=lambda i: phases[i]["score"])
    worst_idx = min(range(len(phases)), key=lambda i: phases[i]["score"])
    cols = st.columns(4)

    for i, (col, phase) in enumerate(zip(cols, phases)):
        score  = phase["score"]
        color  = score_color(score)
        border = f"1px solid {color}55" if i in (best_idx, worst_idx) else f"1px solid {BORDER}"
        tag    = "BEST" if i == best_idx else ("FOCUS" if i == worst_idx else "")

        with col:
            if video_path and num_raw_frames:
                if raw_phase_frames and phase["name"] in raw_phase_frames:
                    raw_idx = raw_phase_frames[phase["name"]]
                else:
                    raw_idx = _analysis_to_raw(phase["frame_index"], result["num_frames"], num_raw_frames)

                frame = _extract_frame(video_path, raw_idx)
                if frame is not None:
                    st.image(frame, use_container_width=True)

            tag_html = f'<span style="font-family:DM Mono,monospace;font-size:9px;padding:2px 7px;border-radius:4px;background:{color}22;color:{color};letter-spacing:1px">{tag}</span>' if tag else ""
            st.markdown(f"""
            <div style="background:{SURFACE};border:{border};border-radius:10px;padding:14px 16px;margin-bottom:4px">
              <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:8px">
                <span style="font-family:'DM Mono',monospace;font-size:10px;color:{TEXT2};letter-spacing:1px;text-transform:uppercase">{phase['name']}</span>
                {tag_html}
              </div>
              <div style="font-family:'Bebas Neue',sans-serif;font-size:40px;line-height:1;color:{color}">{score:.1f}</div>
              <div style="height:3px;background:{BORDER};border-radius:2px;margin:10px 0 8px;overflow:hidden">
                <div style="height:100%;width:{score:.0f}%;background:{color};border-radius:2px"></div>
              </div>
              <div style="font-size:11px;color:{TEXT2};line-height:1.5">{phase['feedback']}</div>
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};margin-top:6px">Frame {phase['frame_index']}</div>
            </div>
            """, unsafe_allow_html=True)

def render_event_timeline(raw_golfdb_events: dict[str, int] | None, video_path: Path | None = None):
    if not raw_golfdb_events:
        st.info("No event timeline available.")
        return

    ordered_keys = [
        "address",
        "toe_up",
        "mid_backswing",
        "top_backswing",
        "mid_downswing",
        "impact",
        "mid_follow_through",
        "finish",
    ]

    cols = st.columns(4)
    for i, key in enumerate(ordered_keys):
        col = cols[i % 4]
        with col:
            frame_idx = raw_golfdb_events.get(key)
            frame = _extract_frame(video_path, frame_idx) if video_path is not None and frame_idx is not None else None
            if frame is not None:
                st.image(frame, use_container_width=True)

            st.markdown(f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:14px 16px;margin-bottom:12px">
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:{TEXT2};letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">{GOLFDB_EVENT_LABELS[key]}</div>
              <div style="font-family:'Bebas Neue',sans-serif;font-size:30px;line-height:1;color:{ACCENT}">{frame_idx if frame_idx is not None else "—"}</div>
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};margin-top:6px">Raw video frame</div>
            </div>
            """, unsafe_allow_html=True)

def render_metrics(metrics: dict):
    items = [
        ("Head stability", metrics["head_stability"]),
        ("Spine posture",  metrics["spine_posture"]),
        ("Hip rotation",   metrics["hip_rotation"]),
        ("Balance",        metrics["balance"]),
    ]
    cols = st.columns(4)
    for col, (name, val) in zip(cols, items):
        color = score_color(val)
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:14px 16px">
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};letter-spacing:1px;text-transform:uppercase;margin-bottom:6px">{name}</div>
              <div style="font-family:'Bebas Neue',sans-serif;font-size:36px;color:{color};line-height:1">{val:.0f}</div>
              <div style="height:3px;background:{BORDER};border-radius:2px;margin-top:10px;overflow:hidden">
                <div style="height:100%;width:{val:.0f}%;background:{color};border-radius:2px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

def render_worst_joints(worst_joints: list[dict]):
    if not worst_joints:
        return
    st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px">Most Deviant Joints</div>', unsafe_allow_html=True)
    cols = st.columns(len(worst_joints))
    for col, joint in zip(cols, worst_joints):
        dev   = joint["deviation"]
        color = deviation_color(dev)
        pct   = min(dev / 3.0, 1.0) * 100
        with col:
            st.markdown(f"""
            <div style="background:{SURFACE};border:1px solid {BORDER};border-radius:10px;padding:14px 16px">
              <div style="font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};letter-spacing:1px;margin-bottom:4px">#{joint['joint_index']:02d}</div>
              <div style="font-size:13px;font-weight:500;color:{TEXT};margin-bottom:4px">{joint['joint_name'].title()}</div>
              <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:{color};line-height:1">{dev:.2f}<span style="font-size:16px">σ</span></div>
              <div style="height:3px;background:{BORDER};border-radius:2px;margin-top:10px;overflow:hidden">
                <div style="height:100%;width:{pct:.0f}%;background:{color};border-radius:2px"></div>
              </div>
            </div>
            """, unsafe_allow_html=True)

def render_ai_feedback(text: str):
    st.markdown(f"""
    <div style="background:{SURFACE2};border:1px solid {BORDER};border-radius:10px;padding:16px 20px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:10px">
        <div style="width:7px;height:7px;border-radius:50%;background:{ACCENT};animation:pulse 2s ease-in-out infinite"></div>
        <span style="font-family:'DM Mono',monospace;font-size:10px;color:{ACCENT};letter-spacing:1.5px">AI ANALYSIS</span>
      </div>
      <div style="font-size:13px;color:{TEXT2};line-height:1.7">{text}</div>
    </div>
    """, unsafe_allow_html=True)

def make_deviation_chart(per_frame_error: list[float], events: dict) -> plt.Figure:
    fig, ax = plt.subplots(figsize=(12, 2.8))
    fig.patch.set_facecolor(SURFACE)
    ax.set_facecolor(SURFACE)

    frames = list(range(len(per_frame_error)))
    ax.fill_between(frames, per_frame_error, alpha=0.12, color=ACCENT)
    ax.plot(frames, per_frame_error, color=ACCENT, linewidth=1.5, solid_capstyle="round")

    for key, color, label in [
        ("top_backswing", AMBER, "TOP"),
        ("impact",        RED,   "IMPACT"),
        ("finish",        GREEN, "FINISH"),
    ]:
        idx = events.get(key)
        if idx is not None and 0 <= idx < len(per_frame_error):
            ax.axvline(idx, color=color, linewidth=1.0, linestyle="--", alpha=0.7)
            ax.text(idx + 0.5, max(per_frame_error) * 0.85, label, fontsize=7, color=color, fontfamily="monospace")

    worst = int(np.argmax(per_frame_error))
    ax.scatter([worst], [per_frame_error[worst]], color=RED, s=24, zorder=5)

    ax.set_xlabel("Frame", fontsize=8, color=MUTED, labelpad=4)
    ax.set_ylabel("Avg |z|", fontsize=8, color=MUTED, labelpad=4)
    ax.tick_params(colors=MUTED, labelsize=7)
    for spine in ax.spines.values():
        spine.set_edgecolor(BORDER)
    ax.grid(True, alpha=0.08, color=TEXT2)
    fig.tight_layout(pad=0.8)
    return fig

# ── Sidebar ───────────────────────────────────────────────────────────────────

def render_sidebar() -> str:
    with st.sidebar:
        st.markdown(f"""
        <div style="font-family:'Bebas Neue',sans-serif;font-size:26px;letter-spacing:3px;color:{ACCENT};margin-bottom:4px">Virtual Golf Coach</div>
        <div style="font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};letter-spacing:1px;margin-bottom:20px">Golf Biomechanics Analyser</div>
        """, unsafe_allow_html=True)

        mode = st.radio("Input mode", ["Upload video", "Pose file"], label_visibility="collapsed")

        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:1.5px;text-transform:uppercase;margin:20px 0 8px">Session History</div>', unsafe_allow_html=True)

        history = st.session_state.get("history", [])
        if history:
            for entry in reversed(history[-6:]):
                c = score_color(entry["score"])
                st.markdown(f"""
                <div style="padding:10px 12px;border-radius:8px;border-left:2px solid {c};background:{SURFACE2};margin-bottom:6px">
                  <div style="display:flex;justify-content:space-between;align-items:center">
                    <span style="font-size:12px;font-weight:500;color:{TEXT}">{entry['name'][:22]}</span>
                    <span style="font-family:'Bebas Neue',sans-serif;font-size:18px;color:{c}">{entry['grade']}</span>
                  </div>
                  <div style="font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};margin-top:2px">{entry['score']}/100</div>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown(f'<div style="font-size:12px;color:{MUTED}">No sessions yet.</div>', unsafe_allow_html=True)

    return mode

# ── Result renderer ───────────────────────────────────────────────────────────

def render_result(
    result: dict,
    source_label: str,
    ai_feedback: str | None = None,
    video_path: Path | None = None,
    num_raw_frames: int | None = None,
    raw_phase_frames: dict[str, int] | None = None,
    raw_events: dict[str, int] | None = None,
    analysis_events: dict[str, int] | None = None,
    raw_golfdb_events: dict[str, int] | None = None,
):
    st.markdown("<br>", unsafe_allow_html=True)
    render_score_hero(result, source_label)

    st.markdown("<br>", unsafe_allow_html=True)
    tabs = st.tabs(["Phases", "Event Timeline", "Metrics", "AI Feedback", "Technical"])

    with tabs[0]:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px">Phase Breakdown</div>', unsafe_allow_html=True)
        render_phase_cards(
            result,
            video_path=video_path,
            num_raw_frames=num_raw_frames,
            raw_phase_frames=raw_phase_frames,
        )

    with tabs[1]:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px">Detected Swing Events</div>', unsafe_allow_html=True)
        render_event_timeline(raw_golfdb_events=raw_golfdb_events, video_path=video_path)

    with tabs[2]:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px">Body Region Metrics</div>', unsafe_allow_html=True)
        render_metrics(result["metrics"])

        st.markdown("<br>", unsafe_allow_html=True)
        render_worst_joints(result.get("worst_joints", []))

    with tabs[3]:
        if ai_feedback:
            render_ai_feedback(ai_feedback)
        else:
            st.info("No AI feedback available.")

    with tabs[4]:
        st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:10px;color:{MUTED};letter-spacing:1.5px;text-transform:uppercase;margin-bottom:10px">Frame Deviation</div>', unsafe_allow_html=True)
        fig = make_deviation_chart(result["per_frame_error"], result.get("events", {}))
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

        with st.expander("Detailed coaching notes"):
            for phase in result["phases"]:
                color = score_color(phase["score"])
                st.markdown(f"""
                <div style="padding:10px 0;border-bottom:1px solid {BORDER}">
                  <span style="font-family:'DM Mono',monospace;font-size:11px;color:{TEXT2};text-transform:uppercase;letter-spacing:1px">{phase['name']}</span>
                  <span style="font-family:'Bebas Neue',sans-serif;font-size:18px;color:{color};float:right">{phase['score']}</span>
                  <div style="font-size:12px;color:{TEXT2};margin-top:4px">{phase['feedback']}</div>
                  <div style="font-family:'DM Mono',monospace;font-size:10px;color:{MUTED};margin-top:2px">Key frame: {phase['frame_index']}</div>
                </div>
                """, unsafe_allow_html=True)

        if result.get("summary"):
            with st.expander("Session summary"):
                for line in result["summary"]:
                    st.markdown(f'<div style="font-size:12px;color:{TEXT2};padding:4px 0">· {line}</div>', unsafe_allow_html=True)

        with st.expander("Debug event data"):
            if raw_events is not None:
                st.write("Raw GolfDB app events", raw_events)
            if raw_golfdb_events is not None:
                st.write("All raw GolfDB events", raw_golfdb_events)
            if analysis_events is not None:
                st.write("Mapped analysis events", analysis_events)
            if raw_phase_frames is not None:
                st.write("Raw phase frames", raw_phase_frames)

# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if "history" not in st.session_state:
        st.session_state.history = []

    mode = render_sidebar()

    st.markdown(f"""
    <div style="display:flex;align-items:baseline;gap:14px;margin-bottom:4px">
      <h1 style="margin:0">Virtual Golf Coach</h1>
      <span style="font-family:'DM Mono',monospace;font-size:11px;color:{MUTED};letter-spacing:1px">Golf Biomechanics Analyser</span>
    </div>
    <hr>
    """, unsafe_allow_html=True)

    # ── Upload mode ──
    if mode == "Upload video":
        uploaded = st.file_uploader(
            "Drop your swing video here",
            type=["mp4", "mov", "avi"],
            help="Max 30 MB · Under 60 s · MP4 recommended",
            label_visibility="collapsed",
        )

        if uploaded is None:
            st.markdown(f"""
            <div style="text-align:center;padding:60px 20px;background:{SURFACE};border:1px dashed {BORDER};border-radius:12px;margin-top:20px">
              <div style="font-family:'Bebas Neue',sans-serif;font-size:28px;color:{MUTED};letter-spacing:3px">DROP A SWING TO BEGIN</div>
              <div style="font-family:'DM Mono',monospace;font-size:11px;color:{MUTED};margin-top:8px">MP4 · MOV · AVI · max 30 MB</div>
            </div>
            """, unsafe_allow_html=True)
            return

        # THIS IS THE CHANGED LINE: Adjusting columns to make the video smaller.
        col_v, col_info = st.columns([1, 1]) 
        with col_v:
            st.video(uploaded)
        with col_info:
            st.markdown(f'<div style="font-family:DM Mono,monospace;font-size:11px;color:{TEXT2};margin-bottom:12px">{uploaded.name}</div>', unsafe_allow_html=True)
            analyse = st.button("Analyse Swing", type="primary", use_container_width=True)

        if not analyse:
            return

        suffix   = Path(uploaded.name).suffix
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
                tmp.write(uploaded.read())
                tmp_path = Path(tmp.name)

            with st.spinner("Extracting pose data…"):
                try:
                    X64, info = process_video_to_model_input(tmp_path)
                except ValueError as e:
                    st.error(str(e))
                    return

            with st.spinner("Detecting swing events…"):
                detector = cached_golfdb_detector()

                raw_golfdb_events = detector.predict_raw_events(tmp_path)
                raw_events = {
                    "address": raw_golfdb_events["address"],
                    "top_backswing": raw_golfdb_events["top_backswing"],
                    "impact": raw_golfdb_events["impact"],
                    "finish": raw_golfdb_events["finish"],
                }

                analysis_events = convert_app_events_to_analysis_indices(
                    raw_events,
                    num_raw_frames=info["num_raw_frames"],
                    num_analysis_frames=X64.shape[0],
                )

            with st.spinner("Comparing against pro template…"):
                mean, std = cached_load_template()
                result = analyze_swing(X64, mean, std, external_events=analysis_events)

            raw_phase_frames = {
                "Backswing": raw_events["top_backswing"],
                "Downswing": (raw_events["top_backswing"] + raw_events["impact"]) // 2,
                "Impact": raw_events["impact"],
                "Follow-through": raw_events["finish"],
            }

            with st.spinner("Generating AI coaching feedback…"):
                ai_feedback = generate_ai_feedback(result)

            st.success(
                f"Analysis complete · {info['num_raw_frames']} raw frames · "
                f"{info['valid_ratio']*100:.1f}% valid poses · {info['fps']:.1f} fps"
            )

            st.session_state.history.append({
                "name": uploaded.name, "score": result["overall_score"], "grade": result["grade"],
            })

            render_result(
                result,
                uploaded.name,
                ai_feedback=ai_feedback,
                video_path=tmp_path,
                num_raw_frames=info["num_raw_frames"],
                raw_phase_frames=raw_phase_frames,
                raw_events=raw_events,
                analysis_events=analysis_events,
                raw_golfdb_events=raw_golfdb_events,
            )

        finally:
            if tmp_path and tmp_path.exists():
                tmp_path.unlink(missing_ok=True)

    # ── Pose file mode ──
    else:
        try:
            pose_files = cached_list_pose_files()
        except Exception as e:
            st.error(str(e))
            return

        selected_name = st.selectbox("Choose a pose file", [p.name for p in pose_files], label_visibility="collapsed")
        selected_path = next(p for p in pose_files if p.name == selected_name)

        if not st.button("Analyse Pose File", type="primary"):
            st.markdown(f'<div style="font-size:12px;color:{MUTED};margin-top:8px">Select a file above and click Analyse.</div>', unsafe_allow_html=True)
            return

        with st.spinner("Running analysis…"):
            try:
                result = analyze_pose_file(selected_path)
                ai_feedback = generate_ai_feedback(result)
            except Exception as e:
                st.error(str(e))
                return

        st.session_state.history.append({
            "name": result["source_file"], "score": result["overall_score"], "grade": result["grade"],
        })

        render_result(result, result["source_file"], ai_feedback=ai_feedback)

if __name__ == "__main__":
    main()