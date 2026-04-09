import streamlit as st
import pandas as pd
import time
import datetime
import random

# -----------------------------------------------------------------
# PAGE CONFIG
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI Career Coach",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# -----------------------------------------------------------------
# CUSTOM CSS — Premium Dark Theme
# -----------------------------------------------------------------
def load_css():
    st.markdown("""
    <style>
    /* ——— Google Font ——— */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

    /* ——— Root variables ——— */
    :root {
        --bg: #0f172a;
        --card: #111827;
        --card-hover: #1e293b;
        --accent: #6366f1;
        --accent-light: #818cf8;
        --text: #f1f5f9;
        --text-dim: #94a3b8;
        --success: #22c55e;
        --warning: #f59e0b;
        --danger: #ef4444;
        --radius: 16px;
        --radius-sm: 10px;
    }

    /* ——— Global ——— */
    html, body, [class*="css"] {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
    }

    .block-container {
        padding-top: 1.5rem !important;
        padding-bottom: 2rem !important;
        max-width: 1200px !important;
    }

    /* ——— Hide Streamlit branding ——— */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}

    /* ——— Tab styling ——— */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background: var(--card);
        border-radius: var(--radius);
        padding: 6px;
        border: 1px solid rgba(255,255,255,0.06);
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: var(--radius-sm);
        padding: 10px 18px;
        font-weight: 500;
        font-size: 0.85rem;
        color: var(--text-dim);
        background: transparent;
        border: none;
        transition: all 0.2s ease;
    }
    .stTabs [data-baseweb="tab"]:hover {
        color: var(--text);
        background: rgba(99,102,241,0.1);
    }
    .stTabs [aria-selected="true"] {
        background: var(--accent) !important;
        color: white !important;
        font-weight: 600;
        box-shadow: 0 4px 12px rgba(99,102,241,0.35);
    }
    .stTabs [data-baseweb="tab-highlight"] {
        display: none;
    }
    .stTabs [data-baseweb="tab-border"] {
        display: none;
    }

    /* ——— Card component ——— */
    .card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 24px;
        margin-bottom: 16px;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .card:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    }

    /* ——— Metric card ——— */
    .metric-card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 20px 24px;
        text-align: center;
        box-shadow: 0 4px 24px rgba(0,0,0,0.2);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        border-color: var(--accent);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 800;
        background: linear-gradient(135deg, var(--accent), var(--accent-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 8px 0 4px;
    }
    .metric-label {
        font-size: 0.8rem;
        color: var(--text-dim);
        text-transform: uppercase;
        letter-spacing: 1.2px;
        font-weight: 600;
    }
    .metric-delta {
        font-size: 0.78rem;
        color: var(--success);
        margin-top: 4px;
    }

    /* ——— AI Insight box ——— */
    .ai-insight {
        background: linear-gradient(135deg, rgba(99,102,241,0.15), rgba(139,92,246,0.1));
        border: 1px solid rgba(99,102,241,0.25);
        border-left: 4px solid var(--accent);
        border-radius: var(--radius);
        padding: 24px 28px;
        margin: 20px 0;
        box-shadow: 0 4px 20px rgba(99,102,241,0.1);
    }
    .ai-insight h4 {
        margin: 0 0 8px 0;
        color: var(--accent-light);
        font-size: 0.9rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .ai-insight p {
        margin: 0;
        font-size: 1.05rem;
        color: var(--text);
        line-height: 1.6;
    }

    /* ——— Smart alert ——— */
    .smart-alert {
        background: linear-gradient(135deg, rgba(245,158,11,0.12), rgba(239,68,68,0.08));
        border: 1px solid rgba(245,158,11,0.25);
        border-left: 4px solid var(--warning);
        border-radius: var(--radius);
        padding: 16px 22px;
        margin-bottom: 20px;
        font-size: 0.95rem;
        color: var(--text);
    }

    /* ——— Hero section ——— */
    .hero {
        text-align: center;
        padding: 36px 20px 28px;
    }
    .hero h1 {
        font-size: 3rem;
        font-weight: 900;
        margin: 0;
        background: linear-gradient(135deg, #fff, var(--accent-light));
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        letter-spacing: -1px;
    }
    .hero p {
        font-size: 1.15rem;
        color: var(--text-dim);
        margin: 12px 0 0;
        font-weight: 400;
    }

    /* ——— Skill tag ——— */
    .skill-tag {
        display: inline-block;
        background: linear-gradient(135deg, rgba(99,102,241,0.2), rgba(99,102,241,0.08));
        border: 1px solid rgba(99,102,241,0.3);
        color: var(--accent-light);
        padding: 6px 14px;
        border-radius: 20px;
        margin: 4px;
        font-size: 0.82rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        transition: all 0.2s ease;
    }
    .skill-tag:hover {
        background: rgba(99,102,241,0.3);
        transform: scale(1.05);
    }

    /* ——— Day card for roadmap ——— */
    .day-card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 20px 24px;
        margin-bottom: 12px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
        transition: border-color 0.2s ease;
    }
    .day-card:hover {
        border-color: var(--accent);
    }
    .day-card .day-title {
        font-size: 1rem;
        font-weight: 700;
        color: var(--text);
        margin-bottom: 4px;
    }
    .day-card .day-focus {
        font-size: 0.85rem;
        color: var(--text-dim);
        margin-bottom: 10px;
    }

    /* Difficulty badges */
    .badge-easy {
        background: rgba(34,197,94,0.15);
        color: #22c55e;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-medium {
        background: rgba(245,158,11,0.15);
        color: #f59e0b;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .badge-hard {
        background: rgba(239,68,68,0.15);
        color: #ef4444;
        padding: 3px 10px;
        border-radius: 8px;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }

    /* ——— Chat bubbles ——— */
    .chat-user {
        background: var(--accent);
        color: white;
        padding: 14px 18px;
        border-radius: 18px 18px 4px 18px;
        margin: 8px 0;
        max-width: 80%;
        margin-left: auto;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 12px rgba(99,102,241,0.25);
    }
    .chat-ai {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.08);
        color: var(--text);
        padding: 14px 18px;
        border-radius: 18px 18px 18px 4px;
        margin: 8px 0;
        max-width: 80%;
        font-size: 0.92rem;
        line-height: 1.5;
        box-shadow: 0 2px 12px rgba(0,0,0,0.15);
    }

    /* ——— Eval card ——— */
    .eval-card {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: var(--radius);
        padding: 24px;
        margin-top: 16px;
        box-shadow: 0 4px 20px rgba(0,0,0,0.2);
    }

    /* ——— Section heading ——— */
    .section-heading {
        font-size: 1.6rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 6px;
        letter-spacing: -0.5px;
    }
    .section-sub {
        font-size: 0.95rem;
        color: var(--text-dim);
        margin-bottom: 24px;
    }

    /* ——— Progress bar override ——— */
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, var(--accent), var(--accent-light)) !important;
        border-radius: 10px;
    }
    .stProgress > div > div > div {
        background: rgba(255,255,255,0.06) !important;
        border-radius: 10px;
    }

    /* ——— Streamlit native metric override ——— */
    div[data-testid="stMetric"] {
        background: var(--card);
        border: 1px solid rgba(255,255,255,0.06);
        padding: 16px;
        border-radius: var(--radius);
        box-shadow: 0 4px 16px rgba(0,0,0,0.15);
    }

    /* ——— Big score ——— */
    .big-score {
        font-size: 5rem;
        font-weight: 900;
        background: linear-gradient(135deg, var(--accent), var(--accent-light), #a78bfa);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin: 16px 0 8px;
        line-height: 1;
    }
    .big-score-label {
        text-align: center;
        color: var(--text-dim);
        font-size: 1rem;
        font-weight: 500;
        margin-bottom: 24px;
    }

    /* ——— Strength / Weakness cards ——— */
    .strength-card {
        background: linear-gradient(135deg, rgba(34,197,94,0.1), rgba(34,197,94,0.03));
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: var(--radius);
        padding: 20px;
    }
    .weakness-card {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.03));
        border: 1px solid rgba(239,68,68,0.2);
        border-radius: var(--radius);
        padding: 20px;
    }

    /* ——— Button override ——— */
    .stButton > button[kind="primary"],
    .stButton > button[data-testid="stBaseButton-primary"] {
        background: linear-gradient(135deg, var(--accent), #7c3aed) !important;
        border: none !important;
        border-radius: var(--radius-sm) !important;
        padding: 10px 28px !important;
        font-weight: 600 !important;
        font-size: 0.9rem !important;
        letter-spacing: 0.3px !important;
        box-shadow: 0 4px 14px rgba(99,102,241,0.35) !important;
        transition: all 0.25s ease !important;
    }
    .stButton > button[kind="primary"]:hover,
    .stButton > button[data-testid="stBaseButton-primary"]:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(99,102,241,0.5) !important;
    }

    /* ——— File uploader ——— */
    [data-testid="stFileUploader"] {
        background: var(--card);
        border: 2px dashed rgba(99,102,241,0.3);
        border-radius: var(--radius);
        padding: 24px;
    }

    /* ——— Text area ——— */
    .stTextArea textarea {
        background: var(--card) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: var(--radius-sm) !important;
        color: var(--text) !important;
        font-family: 'Inter', sans-serif !important;
    }
    .stTextArea textarea:focus {
        border-color: var(--accent) !important;
        box-shadow: 0 0 0 2px rgba(99,102,241,0.2) !important;
    }

    /* ——— Checkbox override ——— */
    .stCheckbox label span {
        font-size: 0.9rem !important;
    }

    /* ——— Weak area highlight ——— */
    .weak-area {
        background: linear-gradient(135deg, rgba(239,68,68,0.1), rgba(239,68,68,0.03));
        border: 1px solid rgba(239,68,68,0.15);
        border-left: 3px solid var(--danger);
        border-radius: var(--radius-sm);
        padding: 14px 18px;
        margin: 6px 0;
        font-size: 0.9rem;
        color: var(--text);
    }

    /* ——— Fix expander ——— */
    .streamlit-expanderHeader {
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }

    /* ——— Divider ——— */
    .divider {
        height: 1px;
        background: rgba(255,255,255,0.06);
        margin: 24px 0;
    }
    </style>
    """, unsafe_allow_html=True)


# -----------------------------------------------------------------
# SESSION STATE INIT
# -----------------------------------------------------------------
def init_state():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = [
            {
                "role": "ai",
                "content": "Hello! I'm your AI Career Coach. I can help with interview prep, resume advice, DSA strategies, and career planning. What would you like to work on today?",
            }
        ]
    if "interview_started" not in st.session_state:
        st.session_state.interview_started = False
    if "interview_feedback" not in st.session_state:
        st.session_state.interview_feedback = None
    if "resume_analyzed" not in st.session_state:
        st.session_state.resume_analyzed = False


# -----------------------------------------------------------------
# HELPER — render a styled card via HTML
# -----------------------------------------------------------------
def card(content: str, extra_class: str = ""):
    st.markdown(
        f'<div class="card {extra_class}">{content}</div>',
        unsafe_allow_html=True,
    )


def metric_card(label: str, value: str, delta: str = ""):
    delta_html = f'<div class="metric-delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


# -----------------------------------------------------------------
# 1. 🏠 DASHBOARD
# -----------------------------------------------------------------
def render_dashboard():
    # Hero
    st.markdown(
        """
        <div class="hero">
            <h1>Your AI Career Coach</h1>
            <p>Personalized guidance. Real results.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Smart alert at top
    st.markdown(
        """
        <div class="smart-alert">
            ⚠️ <strong>You missed 3 tasks.</strong> Stay consistent to hit your 80% placement target.
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        metric_card("Placement Score", "62%", "↑ 5% this week")
    with col2:
        metric_card("DSA Level", "Intermediate", "↑ 1 level")
    with col3:
        metric_card("Aptitude", "75%", "↑ 5%")
    with col4:
        metric_card("Consistency", "12 Days", "🔥 Streak")

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Placement progress + AI Insight
    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown('<div class="section-heading">Placement Readiness</div>', unsafe_allow_html=True)
        st.markdown('<div class="section-sub">Your progress toward placement-ready status</div>', unsafe_allow_html=True)
        st.progress(0.62)
        st.caption("62 / 100  —  Target: 80%")

    with col_right:
        st.markdown(
            """
            <div class="ai-insight">
                <h4>🧠 AI Insight</h4>
                <p>Your placement probability is <strong>62%</strong>. Focus on DSA Hard problems and take 2 mock interviews this week to push toward <strong>80%</strong>.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------
# 2. 📄 RESUME ANALYZER
# -----------------------------------------------------------------
def render_resume_analyzer():
    st.markdown('<div class="section-heading">Resume Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Upload your resume for instant AI-driven feedback</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader("Upload Resume (PDF / TXT)", type=["pdf", "txt"])

    if st.button("🚀  Analyze Resume", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Analyzing resume against industry standards…"):
                time.sleep(2)

            st.session_state.resume_analyzed = True

    if st.session_state.resume_analyzed:
        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        # Skills
        st.markdown('<div class="section-heading" style="font-size:1.2rem">Extracted Skills</div>', unsafe_allow_html=True)
        skills = ["Python", "JavaScript", "React", "Node.js", "HTML/CSS", "SQL", "Git", "REST APIs"]
        tags = "".join([f'<span class="skill-tag">{s}</span>' for s in skills])
        st.markdown(tags, unsafe_allow_html=True)

        st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(
                """
                <div class="strength-card">
                    <h4 style="color:#22c55e; margin-top:0;">✅ Strengths</h4>
                    <ul style="margin:0; padding-left:18px; color:#cbd5e1; line-height:1.8;">
                        <li>Strong web development foundation</li>
                        <li>Multiple personal projects</li>
                        <li>Clean formatting & readability</li>
                        <li>Relevant tech stack for SDE roles</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            st.markdown(
                """
                <div class="weakness-card">
                    <h4 style="color:#ef4444; margin-top:0;">⚠️ Weaknesses</h4>
                    <ul style="margin:0; padding-left:18px; color:#cbd5e1; line-height:1.8;">
                        <li>No cloud technologies (AWS/GCP)</li>
                        <li>Missing quantifiable metrics</li>
                        <li>No Docker / containerization</li>
                        <li>System Design not mentioned</li>
                    </ul>
                </div>
                """,
                unsafe_allow_html=True,
            )

        st.markdown("")
        st.markdown(
            """
            <div class="ai-insight">
                <h4>🚨 Missing Skills for SDE Goal</h4>
                <p>Add <strong>AWS/GCP</strong>, <strong>Docker</strong>, and <strong>System Design</strong> experience to significantly boost your resume score.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------
# 3. 📊 PLACEMENT PREDICTOR
# -----------------------------------------------------------------
def render_placement_predictor():
    st.markdown('<div class="section-heading">Placement Predictor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">AI-powered prediction of your placement readiness</div>', unsafe_allow_html=True)

    with st.spinner("Calculating placement score…"):
        time.sleep(1)

    st.markdown('<div class="big-score">62%</div>', unsafe_allow_html=True)
    st.markdown('<div class="big-score-label">Placement Readiness Score</div>', unsafe_allow_html=True)

    st.progress(0.62)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    with col1:
        card(
            """
            <h4 style="color:#f1f5f9; margin-top:0;">📋 Score Breakdown</h4>
            <ul style="margin:0; padding-left:18px; color:#94a3b8; line-height:2;">
                <li><strong style="color:#22c55e;">High Aptitude</strong> — Logical reasoning boosts your score</li>
                <li><strong style="color:#f59e0b;">Low Mock Interviews</strong> — Behavioral confidence needs work</li>
                <li><strong style="color:#ef4444;">Resume Gap</strong> — Missing cloud & DevOps experience</li>
            </ul>
            """
        )
    with col2:
        card(
            """
            <h4 style="color:#f1f5f9; margin-top:0;">💡 Improvement Suggestions</h4>
            <ul style="margin:0; padding-left:18px; color:#94a3b8; line-height:2;">
                <li>Solve 5 Hard-level DSA problems this week <span class="badge-hard">+4%</span></li>
                <li>Take 2 mock interviews <span class="badge-medium">+3%</span></li>
                <li>Add System Design basics to resume <span class="badge-easy">+2%</span></li>
                <li>Dockerize one project <span class="badge-medium">+2%</span></li>
            </ul>
            """
        )


# -----------------------------------------------------------------
# 4. 🧠 ADAPTIVE LEARNING PATH
# -----------------------------------------------------------------
def render_adaptive_learning():
    st.markdown('<div class="section-heading">Adaptive Learning Path</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Your personalized 7-day sprint to patch weak areas</div>', unsafe_allow_html=True)

    days = [
        {"day": "Day 1 — Today", "focus": "System Design Basics", "tasks": ["Read 'Grokking System Design' Ch 1", "Draw a URL Shortener architecture"], "diff": "Medium"},
        {"day": "Day 2", "focus": "DSA — Trees", "tasks": ["LC #104  Maximum Depth of Binary Tree", "LC #236  LCA of a Binary Tree"], "diff": "Hard"},
        {"day": "Day 3", "focus": "Mock Interview Prep", "tasks": ["Record a 2-min self introduction", "Review STAR method for behavioral Q's"], "diff": "Easy"},
        {"day": "Day 4", "focus": "Cloud Deployment", "tasks": ["Dockerize your React app", "Deploy to AWS EC2 or Vercel"], "diff": "Medium"},
        {"day": "Day 5", "focus": "DSA — Graphs", "tasks": ["LC #200  Number of Islands", "Review BFS/DFS templates"], "diff": "Hard"},
        {"day": "Day 6", "focus": "Resume Update", "tasks": ["Add quantifiable metrics to bullet points", "Re-run Resume Analyzer"], "diff": "Easy"},
        {"day": "Day 7", "focus": "Rest & Review", "tasks": ["Review all mistakes from Day 1–6", "Plan next week's sprint"], "diff": "Easy"},
    ]

    for i, d in enumerate(days):
        diff = d["diff"]
        badge_class = {"Easy": "badge-easy", "Medium": "badge-medium", "Hard": "badge-hard"}[diff]

        st.markdown(
            f"""
            <div class="day-card">
                <div style="display:flex; justify-content:space-between; align-items:center;">
                    <div class="day-title">{d['day']}</div>
                    <span class="{badge_class}">{diff}</span>
                </div>
                <div class="day-focus">{d['focus']}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        for task in d["tasks"]:
            st.checkbox(task, key=f"learn_{i}_{task}")


# -----------------------------------------------------------------
# 5. 🤖 AI MENTOR CHAT
# -----------------------------------------------------------------
def render_ai_mentor():
    st.markdown('<div class="section-heading">AI Mentor</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Chat with your career coach for advice, tips, and strategies</div>', unsafe_allow_html=True)

    # Render chat history
    for msg in st.session_state.chat_history:
        if msg["role"] == "user":
            st.markdown(f'<div class="chat-user">{msg["content"]}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-ai">🤖  {msg["content"]}</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Input
    if prompt := st.chat_input("Ask me anything — interview tips, DSA help, career advice…"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})

        # Mock AI responses
        responses = [
            "Great question! For optimizing search algorithms, consider using a Hash Map for O(1) lookups, or a Binary Search Tree for O(log N) ordered operations. Would you like me to walk through a specific example?",
            "I recommend focusing on the STAR method for behavioral interviews: Situation, Task, Action, Result. This framework helps structure your answers clearly.",
            "For system design interviews, start with requirements clarification, then high-level design, then deep dive. Practice with classic problems like 'Design a URL Shortener' or 'Design Twitter'.",
            "To improve your DSA skills, I suggest the 'Blind 75' list. Start with Easy problems, then progress to Medium. Focus on understanding patterns, not memorizing solutions.",
            "For your resume, quantify your achievements. Instead of 'Improved performance', write 'Reduced API latency by 40% through query optimization'. Numbers make a huge difference.",
        ]
        reply = random.choice(responses)
        st.session_state.chat_history.append({"role": "ai", "content": reply})
        st.rerun()


# -----------------------------------------------------------------
# 6. 🎤 MOCK INTERVIEW
# -----------------------------------------------------------------
def render_mock_interview():
    st.markdown('<div class="section-heading">Mock Interview</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Practice with AI-generated interview questions</div>', unsafe_allow_html=True)

    if not st.session_state.interview_started:
        card(
            """
            <div style="text-align:center; padding:20px 0;">
                <div style="font-size:3rem; margin-bottom:12px;">🎤</div>
                <h3 style="color:#f1f5f9; margin:0 0 8px;">Ready to practice?</h3>
                <p style="color:#94a3b8; margin:0;">The AI will ask you a technical question and evaluate your answer on correctness, clarity, and depth.</p>
            </div>
            """
        )
        if st.button("▶️  Start Interview", type="primary"):
            st.session_state.interview_started = True
            st.session_state.interview_feedback = None
            st.rerun()
    else:
        st.markdown(
            """
            <div class="card">
                <div style="display:flex; align-items:center; gap:10px; margin-bottom:12px;">
                    <span style="font-size:1.2rem;">🤖</span>
                    <span style="color:#94a3b8; font-size:0.85rem; font-weight:600; text-transform:uppercase; letter-spacing:0.5px;">Interviewer</span>
                </div>
                <p style="color:#f1f5f9; font-size:1.05rem; margin:0; line-height:1.7;">
                    Explain the difference between <code style="background:rgba(99,102,241,0.2); padding:2px 6px; border-radius:4px;">let</code>,
                    <code style="background:rgba(99,102,241,0.2); padding:2px 6px; border-radius:4px;">var</code>, and
                    <code style="background:rgba(99,102,241,0.2); padding:2px 6px; border-radius:4px;">const</code> in JavaScript. When would you use each?
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        answer = st.text_area("Your Answer:", height=150, placeholder="Type your response here…")

        col1, col2, _ = st.columns([1, 1, 3])
        with col1:
            if st.button("Submit Answer", type="primary"):
                if answer.strip():
                    with st.spinner("Evaluating your response…"):
                        time.sleep(1.5)
                    st.session_state.interview_feedback = {
                        "correctness": "85%",
                        "clarity": "70%",
                        "depth": "75%",
                        "feedback": "Good understanding of block vs function scope. However, you didn't mention that `const` arrays and objects can still be mutated (only the reference is immutable). Default to `const` unless reassignment is needed.",
                    }
                    st.rerun()
                else:
                    st.warning("Please write an answer before submitting.")
        with col2:
            if st.button("End Interview"):
                st.session_state.interview_started = False
                st.session_state.interview_feedback = None
                st.rerun()

        if st.session_state.interview_feedback:
            fb = st.session_state.interview_feedback
            st.markdown('<div class="divider"></div>', unsafe_allow_html=True)
            st.markdown(
                f"""
                <div class="eval-card">
                    <h4 style="color:#f1f5f9; margin-top:0; margin-bottom:16px;">📊 Evaluation</h4>
                    <div style="display:flex; gap:20px; margin-bottom:16px;">
                        <div class="metric-card" style="flex:1;">
                            <div class="metric-label">Correctness</div>
                            <div class="metric-value">{fb['correctness']}</div>
                        </div>
                        <div class="metric-card" style="flex:1;">
                            <div class="metric-label">Clarity</div>
                            <div class="metric-value">{fb['clarity']}</div>
                        </div>
                        <div class="metric-card" style="flex:1;">
                            <div class="metric-label">Depth</div>
                            <div class="metric-value">{fb['depth']}</div>
                        </div>
                    </div>
                    <div style="background:rgba(99,102,241,0.08); border:1px solid rgba(99,102,241,0.2); border-radius:10px; padding:16px;">
                        <strong style="color:#818cf8;">💬 Feedback:</strong>
                        <p style="color:#cbd5e1; margin:8px 0 0; line-height:1.6;">{fb['feedback']}</p>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )


# -----------------------------------------------------------------
# 7. 📉 PROGRESS TRACKING
# -----------------------------------------------------------------
def render_progress_tracking():
    st.markdown('<div class="section-heading">Progress Tracking</div>', unsafe_allow_html=True)
    st.markdown('<div class="section-sub">Monitor your skill growth and task completion</div>', unsafe_allow_html=True)

    # Skill growth line chart
    st.markdown("#### 📈 Skill Growth — Last 30 Days")
    dates = pd.date_range(end=datetime.date.today(), periods=30)
    chart_data = pd.DataFrame(
        {
            "DSA": [40 + (i * 1.2) + (i % 3) for i in range(30)],
            "Aptitude": [50 + (i * 0.8) - (i % 2) for i in range(30)],
        },
        index=dates,
    )
    st.line_chart(chart_data, color=["#6366f1", "#22c55e"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Task completion bar chart
    st.markdown("#### 📊 Weekly Task Completion")
    bar_data = pd.DataFrame(
        {
            "Completed": [12, 15, 10, 18],
            "Missed": [3, 1, 5, 2],
        },
        index=["Week 1", "Week 2", "Week 3", "Week 4"],
    )
    st.bar_chart(bar_data, color=["#6366f1", "#ef4444"])

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # Weak areas
    st.markdown("#### ⚠️ Areas Needing Attention")
    weak = [
        ("Mock Interviews", "Only 2 completed — target is 6 per month"),
        ("System Design", "No study sessions logged in 2 weeks"),
        ("Consistency", "3 days missed in the last sprint"),
    ]
    for area, detail in weak:
        st.markdown(
            f"""
            <div class="weak-area">
                <strong>{area}:</strong> {detail}
            </div>
            """,
            unsafe_allow_html=True,
        )


# -----------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------
def main():
    load_css()
    init_state()

    # Tab navigation
    tabs = st.tabs(
        [
            "🏠 Dashboard",
            "📄 Resume",
            "📊 Predictor",
            "🧠 Learning",
            "🤖 Mentor",
            "🎤 Interview",
            "📈 Progress",
        ]
    )

    with tabs[0]:
        render_dashboard()
    with tabs[1]:
        render_resume_analyzer()
    with tabs[2]:
        render_placement_predictor()
    with tabs[3]:
        render_adaptive_learning()
    with tabs[4]:
        render_ai_mentor()
    with tabs[5]:
        render_mock_interview()
    with tabs[6]:
        render_progress_tracking()


if __name__ == "__main__":
    main()
