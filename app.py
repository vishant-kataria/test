import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import time
import datetime

# -------------------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------------------
st.set_page_config(
    page_title="Agentic AI Career Coach",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -------------------------------------------------------------
# CUSTOM CSS (Premium Startup Vibe)
# -------------------------------------------------------------
def load_css():
    st.markdown("""
    <style>
    /* Main container block padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    /* Sleek card styling */
    .st-emotion-cache-1r6slb0, .st-emotion-cache-16txtl3 {
        background: linear-gradient(145deg, var(--background-color), var(--secondary-background-color)) !important;
        border-radius: 10px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    /* AI Insight gradient box */
    .ai-insight-box {
        background: linear-gradient(90deg, rgba(79,70,229,0.2) 0%, rgba(147,51,234,0.2) 100%);
        border-left: 4px solid #4F46E5;
        padding: 1.5rem;
        border-radius: 8px;
        margin-bottom: 1.5rem;
        color: var(--text-color);
        box-shadow: 0 2px 10px rgba(79, 70, 229, 0.15);
    }
    
    /* Smart Alert box */
    .smart-alert-box {
        background: linear-gradient(90deg, rgba(239,68,68,0.2) 0%, rgba(249,115,22,0.2) 100%);
        border-left: 4px solid #EF4444;
        padding: 1rem;
        border-radius: 8px;
        margin-bottom: 1rem;
        color: var(--text-color);
    }

    /* Metric numbers */
    div[data-testid="metric-container"] {
        background-color: var(--secondary-background-color);
        padding: 1rem;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.05);
    }
    </style>
    """, unsafe_allow_html=True)

# -------------------------------------------------------------
# SESSION STATE INIT
# -------------------------------------------------------------
def init_state():
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = [
            {"role": "assistant", "content": "Hello! I am your AI Career Coach. How can I help you level up your career today?"}
        ]
    if 'interview_started' not in st.session_state:
        st.session_state.interview_started = False
    if 'interview_feedback' not in st.session_state:
        st.session_state.interview_feedback = None

# -------------------------------------------------------------
# 1. 🏠 DASHBOARD
# -------------------------------------------------------------
def render_dashboard():
    st.title("🏠 Dashboard")
    st.markdown("Welcome back, **Vishant**! Here is an overview of your career progress.")
    
    st.markdown(f'''
    <div class="ai-insight-box">
        <h4>🤖 AI Insight</h4>
        <p style="font-size: 1.1rem; margin: 0;">Your placement probability is <b>62%</b>. Improve your DSA consistency and start solving Hard-level questions to reach 80%.</p>
    </div>
    ''', unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.subheader("Student Summary")
        st.markdown("**Goal:** SDE Role at MAANG")
        st.markdown("**Placement Target:** October 2026")
        
        st.write("Placement Score Progress (62%)")
        st.progress(0.62)
        
    with col2:
        st.subheader("Key Stats")
        mcol1, mcol2, mcol3, mcol4 = st.columns(4)
        mcol1.metric("DSA Level", "Intermediate", delta="1 Level Up")
        mcol2.metric("Aptitude", "75%", delta="5%")
        mcol3.metric("Projects", "3 Fullstack", delta=None)
        mcol4.metric("Consistency", "12 Day Streak", delta="🔥")
        
# -------------------------------------------------------------
# 2. 📄 RESUME ANALYZER
# -------------------------------------------------------------
def render_resume_analyzer():
    st.title("📄 AI Resume Analyzer")
    st.write("Upload your resume to get instant AI-driven feedback against your target SDE role.")
    
    uploaded_file = st.file_uploader("Upload Resume (PDF/TXT)", type=['pdf', 'txt'])
    
    if st.button("🚀 Analyze Resume", type="primary"):
        if uploaded_file is None:
            st.warning("Please upload a file first.")
        else:
            with st.spinner("Analyzing resume against industry standards..."):
                time.sleep(2) # Simulate processing
                
                st.success("Analysis Complete!")
                
                st.subheader("Extracted Data")
                st.write("**Extracted Skills:**")
                # Using Streamlit pills or basic tags via html
                skills = ["Python", "JavaScript", "React", "HTML/CSS", "SQL", "Git"]
                skill_tags = "".join([f"<span style='background:#1E293B; padding:5px 10px; border-radius:15px; margin-right:5px; border: 1px solid #4F46E5;'>{s}</span>" for s in skills])
                st.markdown(skill_tags, unsafe_allow_html=True)
                
                st.markdown("---")
                col1, col2 = st.columns(2)
                with col1:
                    st.info("👍 **Strengths**\n- Strong foundation in web development.\n- Multiple personal projects listed.\n- Good formatting and readability.")
                with col2:
                    st.error("📉 **Weaknesses**\n- No cloud technologies mentioned.\n- Lack of quantifiable metrics in project bullets (e.g., 'improved speed by X%').")
                    
                st.warning("🚨 **Missing Skills for SDE Goal:** AWS/GCP, Docker, System Design basics.")

# -------------------------------------------------------------
# 3. 📊 PLACEMENT PREDICTION
# -------------------------------------------------------------
def render_placement_prediction():
    st.title("📊 Placement Prediction Engine")
    
    fig = go.Figure(go.Indicator(
        mode = "gauge+number+delta",
        value = 62,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Placement Readiness Score", 'font': {'size': 24}},
        delta = {'reference': 80, 'increasing': {'color': "green"}},
        gauge = {
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "white"},
            'bar': {'color': "#4F46E5"},
            'bgcolor': "rgba(0,0,0,0)",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 40], 'color': "rgba(239,68,68,0.3)"},
                {'range': [40, 75], 'color': "rgba(245,158,11,0.3)"},
                {'range': [75, 100], 'color': "rgba(16,185,129,0.3)"}],
            'threshold': {
                'line': {'color': "white", 'width': 4},
                'thickness': 0.75,
                'value': 80}
        }
    ))
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", font={'color': "white"}, height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("### Why is your score 62%?")
    st.markdown("- **High Aptitude**: Your cognitive and logical score pulls the metric up.\n- **Low Mock Interviews**: You haven't taken enough mock interviews, dropping your behavioral confidence score.\n- **Resume Gap**: Missing cloud deployment experience.")
    
    st.info("💡 **Dynamic Strategy:** Improve your System Design basics and take 2 Mock Interviews this week to increase your score by **+8%**.")

# -------------------------------------------------------------
# 4. 🧠 ADAPTIVE LEARNING PLAN
# -------------------------------------------------------------
def render_adaptive_learning():
    st.title("🧠 Adaptive Learning Roadmap")
    st.write("Your personalized 7-Day sprint, tailored to patch your weak areas.")
    
    days = [
        {"day": "Day 1 (Today)", "focus": "System Design Basics", "tasks": ["Read 'Grokking' Ch 1", "Draw a URL Shortener Architecture"], "diff": "Medium"},
        {"day": "Day 2", "focus": "DSA - Trees", "tasks": ["LC #104 Maximum Depth", "LC #236 LCA of Binary Tree"], "diff": "Hard"},
        {"day": "Day 3", "focus": "Mock Interview Prep", "tasks": ["Record a 2-min self intro", "Review STAR method"], "diff": "Easy"},
        {"day": "Day 4", "focus": "Cloud Deployment", "tasks": ["Dockerize your React App", "Deploy to AWS EC2/Vercel"], "diff": "Medium"},
        {"day": "Day 5", "focus": "DSA - Graphs", "tasks": ["LC #200 Number of Islands", "Review BFS/DFS templates"], "diff": "Hard"},
        {"day": "Day 6", "focus": "Resume Update", "tasks": ["Add quantifiable metrics to bullet points", "Run Resume Analyzer again"], "diff": "Easy"},
        {"day": "Day 7", "focus": "Rest & Review", "tasks": ["Review all mistakes from Day 1-6", "Plan next week"], "diff": "Easy"}
    ]
    
    for i, d in enumerate(days):
        with st.expander(f"📅 {d['day']} - **{d['focus']}** | Difficulty: {d['diff']}"):
            for task in d['tasks']:
                st.checkbox(task, key=f"task_{i}_{task}")

# -------------------------------------------------------------
# 5. 🤖 AI CHAT MENTOR
# -------------------------------------------------------------
def render_ai_mentor():
    st.title("🤖 AI Coaching Mentor")
    st.write("Chat with your specialized AI agent to get interview tips, debugging help, or career advice.")
    
    chat_container = st.container()
    
    with chat_container:
        for msg in st.session_state.chat_history:
            avatar = "🤖" if msg["role"] == "assistant" else "🧑‍💻"
            with st.chat_message(msg["role"], avatar=avatar):
                st.write(msg["content"])
                
    if prompt := st.chat_input("Ask for advice, e.g., 'How do I optimize a search algorithm?'"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        
        # Display new user message
        with st.chat_message("user", avatar="🧑‍💻"):
            st.write(prompt)
            
        # Mock AI Response
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("AI is thinking..."):
                time.sleep(1)
                reply = "That's a great question! For a standard search, a Hash Map offers O(1) lookups, but if you need an ordered set, a Binary Search tree yields O(log N). Do you have a specific use case?"
                st.write(reply)
        st.session_state.chat_history.append({"role": "assistant", "content": reply})

# -------------------------------------------------------------
# 6. 🎤 MOCK INTERVIEW
# -------------------------------------------------------------
def render_mock_interview():
    st.title("🎤 AI Mock Interview")
    
    if not st.session_state.interview_started:
        st.write("Ready to test your nerves? The AI will ask you a technical question and evaluate your answer.")
        if st.button("▶️ Start Interview", type="primary"):
            st.session_state.interview_started = True
            st.session_state.interview_feedback = None
            st.rerun()
    else:
        st.subheader("Question 1 / 1")
        st.markdown("**Interviewer (AI):** Explain the difference between `let`, `var`, and `const` in JavaScript, and when you would use each.")
        
        answer = st.text_area("Your Answer:", height=150)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Submit Answer", type="primary"):
                with st.spinner("Evaluating..."):
                    time.sleep(1.5)
                    st.session_state.interview_feedback = {
                        "correctness": "85%",
                        "clarity": "70%",
                        "feedback": "Good understanding of block vs function scope. However, you forgot to mention that `const` arrays/objects can still have their properties mutated. You should use `const` by default unless you expect reassignment."
                    }
        with col2:
            if st.button("End Interview"):
                st.session_state.interview_started = False
                st.session_state.interview_feedback = None
                st.rerun()
                
        if st.session_state.interview_feedback:
            st.markdown("---")
            st.subheader("📊 Evaluation")
            fcol1, fcol2 = st.columns(2)
            fcol1.metric("Correctness", st.session_state.interview_feedback["correctness"])
            fcol2.metric("Clarity", st.session_state.interview_feedback["clarity"])
            st.info(f"**Detailed Feedback:** {st.session_state.interview_feedback['feedback']}")

# -------------------------------------------------------------
# 7. 📉 PROGRESS TRACKING
# -------------------------------------------------------------
def render_progress_tracking():
    st.title("📉 Progress Tracking")
    
    # Mock data
    dates = pd.date_range(end=datetime.date.today(), periods=30)
    data = pd.DataFrame({
        "Date": dates,
        "DSA Skill Level": [40 + (i*1.2) + (i%3) for i in range(30)],
        "Aptitude Score": [50 + (i*0.8) - (i%2) for i in range(30)]
    })
    
    st.subheader("Skill Improvement Over 30 Days")
    fig = px.line(data, x="Date", y=["DSA Skill Level", "Aptitude Score"], 
                  color_discrete_sequence=["#4F46E5", "#10B981"],
                  template="plotly_dark")
    fig.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
                     legend_title_text="Skill Domain")
    st.plotly_chart(fig, use_container_width=True)
    
    st.subheader("Task Completion")
    bar_data = pd.DataFrame({
        "Week": ["Week 1", "Week 2", "Week 3", "Week 4"],
        "Completed": [12, 15, 10, 18],
        "Missed": [3, 1, 5, 2]
    })
    fig_bar = px.bar(bar_data, x="Week", y=["Completed", "Missed"],
                     barmode="group",
                     color_discrete_map={"Completed": "#4F46E5", "Missed": "#EF4444"},
                     template="plotly_dark")
    fig_bar.update_layout(paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_bar, use_container_width=True)

# -------------------------------------------------------------
# 8. 🔔 SMART ALERTS
# -------------------------------------------------------------
def render_smart_alerts():
    st.title("🔔 Smart Alerts")
    
    st.markdown('''
    <div class="smart-alert-box">
        <strong>⚠️ Task Warning:</strong> You missed 3 tasks from yesterday's Adaptive Learning Plan. Complete them today to stay on track for your 80% placement readiness goal.
    </div>
    ''', unsafe_allow_html=True)
    
    st.warning("📅 Upcoming Mock Interview scheduled for tomorrow at 5:00 PM.")
    st.info("💡 A new System Design course module was dynamically added to your roadmap based on your latest weaknesses.")

# -------------------------------------------------------------
# MAIN APP ROUTING
# -------------------------------------------------------------
def main():
    load_css()
    init_state()
    
    st.sidebar.title("🤖 AI Career Coach")
    st.sidebar.markdown("---")
    
    menu = [
        "🏠 Dashboard", 
        "📄 Resume Analyzer", 
        "📊 Placement Prediction",
        "🧠 Adaptive Learning",
        "🤖 AI Chat Mentor",
        "🎤 Mock Interview",
        "📉 Progress Tracking",
        "🔔 Smart Alerts"
    ]
    
    choice = st.sidebar.radio("Navigation", menu)
    
    st.sidebar.markdown("---")
    st.sidebar.caption("© 2026 Team The Renegades")
    st.sidebar.caption("Hack AI 2.0 Project")
    
    if choice == "🏠 Dashboard":
        render_dashboard()
    elif choice == "📄 Resume Analyzer":
        render_resume_analyzer()
    elif choice == "📊 Placement Prediction":
        render_placement_prediction()
    elif choice == "🧠 Adaptive Learning":
        render_adaptive_learning()
    elif choice == "🤖 AI Chat Mentor":
        render_ai_mentor()
    elif choice == "🎤 Mock Interview":
        render_mock_interview()
    elif choice == "📉 Progress Tracking":
        render_progress_tracking()
    elif choice == "🔔 Smart Alerts":
        render_smart_alerts()

if __name__ == "__main__":
    main()
