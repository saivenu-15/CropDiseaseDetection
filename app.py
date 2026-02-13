import streamlit as st
from auth import init_db, add_user, verify_user
from model_predict import predict
from disease_advice import DISEASE_ADVICE
from PIL import Image
import sqlite3
import time
from datetime import datetime
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from io import BytesIO

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="AI Crop Disease Detection",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------
# CUSTOM CSS STYLING
# -----------------------------
st.markdown("""
<style>
    /* Main styling */
    .main {
        padding: 2rem;
    }
    
    /* Gradient background */
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        background-attachment: fixed;
    }
    
    /* Card styling */
    .custom-card {
        background: white;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        margin: 1rem 0;
    }
    
    /* Login/Signup card */
    .auth-card {
        max-width: 450px;
        margin: 3rem auto;
        background: white;
        padding: 3rem;
        border-radius: 20px;
        box-shadow: 0 15px 35px rgba(0,0,0,0.3);
    }
    
    /* Title styling */
    h1 {
        color: #2d3748;
        font-weight: 700;
        text-align: center;
        margin-bottom: 2rem;
    }
    
    h2 {
        color: #4a5568;
        font-weight: 600;
        margin-top: 1.5rem;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        font-weight: 600;
        transition: all 0.3s;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6);
    }
    
    /* Success/Info boxes */
    .stSuccess, .stInfo {
        border-radius: 10px;
        padding: 1rem;
    }
    
    /* Metrics styling */
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 15px;
        text-align: center;
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    /* Chat bubble styling */
    .chat-user {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 5px 20px;
        margin: 0.5rem 0;
        margin-left: 20%;
        box-shadow: 0 3px 10px rgba(0,0,0,0.2);
    }
    
    .chat-bot {
        background: #f7fafc;
        color: #2d3748;
        padding: 1rem 1.5rem;
        border-radius: 20px 20px 20px 5px;
        margin: 0.5rem 0;
        margin-right: 20%;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 10px rgba(0,0,0,0.1);
    }
    
    /* Image container */
    .image-container {
        border-radius: 15px;
        overflow: hidden;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        background: white;
        padding: 1rem;
    }
    
    /* Sidebar styling - Force visibility with multiple selectors - STRONGER */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div,
    div[data-testid="stSidebar"],
    .css-1d391kg,
    [data-testid="stSidebar"],
    [data-testid="stSidebar"] > *,
    [data-baseweb="drawer"],
    [class*="sidebar"],
    [class*="Sidebar"] {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Override Streamlit's sidebar hiding */

    
    /* Ensure sidebar content is visible */
    section[data-testid="stSidebar"] [data-testid="stSidebarContent"],
    section[data-testid="stSidebar"] [data-testid="stSidebarNav"] {
        background: transparent !important;
        display: block !important;
        visibility: visible !important;
    }
    
    /* Override any hiding */

    
    /* Sidebar text color - force white */
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] div,
    section[data-testid="stSidebar"] span {
        color: white !important;
    }
    
    /* Sidebar markdown headings */
    section[data-testid="stSidebar"] .stMarkdown h3 {
        color: white !important;
    }
    
    /* Remove white spaces - minimize padding */
    .main .block-container {
        padding-top: 1rem !important;
        padding-bottom: 1rem !important;
    }
    
    /* Fix empty divs causing white space */
    div[data-testid="stVerticalBlock"] > div:empty {
        display: none !important;
    }
    
    /* Remove extra spacing between elements */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Reduce markdown margins */
    .stMarkdown {
        margin-bottom: 0.5rem !important;
    }
    
    /* Fix white space in main content */
    .main .block-container > div {
        padding-top: 0 !important;
    }
    
    /* Progress bar */
    .progress-container {
        background: #e2e8f0;
        border-radius: 10px;
        height: 25px;
        overflow: hidden;
        margin: 1rem 0;
    }
    
    .progress-fill {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
        display: flex;
        align-items: center;
        justify-content: center;
        color: white;
        font-weight: 600;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        font-size: 0.9rem;
        margin: 0.25rem;
    }
    
    .badge-success {
        background: #48bb78;
        color: white;
    }
    
    .badge-warning {
        background: #ed8936;
        color: white;
    }
    
    .badge-info {
        background: #4299e1;
        color: white;
    }
    
    /* Hide default Streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Ensure sidebar is visible */

    
    /* Fix main content padding */

    
    /* Remove extra white space */
    .element-container {
        margin-bottom: 0.5rem !important;
    }
    
    /* Minimize spacing in vertical blocks */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem !important;
    }
    
    /* Remove extra padding from Streamlit components */
    .stButton {
        margin-bottom: 0.5rem !important;
    }
    
    .stFileUploader {
        margin-bottom: 1rem !important;
    }
    
    /* Remove padding from empty blocks */
    div[data-testid="stVerticalBlock"] {
        gap: 0.5rem;
    }
    
    /* Fix white background issues */
    .stApp > header {
        background-color: transparent;
    }
    
    /* Main content area */
    .main .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
    
    /* Remove unnecessary margins */
    .stMarkdown {
        margin-bottom: 0.5rem;
    }
    
    /* Sidebar button styling */
    section[data-testid="stSidebar"] .stButton > button {
        background: rgba(255, 255, 255, 0.15) !important;
        color: white !important;
        border: 2px solid rgba(255, 255, 255, 0.25) !important;
        font-weight: 600 !important;
        margin-bottom: 0.5rem !important;
        width: 100% !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.3) !important;
        border-color: rgba(255, 255, 255, 0.5) !important;
        transform: translateX(5px);
        transition: all 0.3s;
        color: white !important;
    }
    
    /* Logout button special styling - VERY PROMINENT */
    section[data-testid="stSidebar"] .stButton > button[kind="primary"],
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"] {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
        border: 2px solid #ef4444 !important;
        font-size: 1.15rem !important;
        padding: 1rem 0.75rem !important;
        font-weight: 800 !important;
        color: white !important;
        box-shadow: 0 4px 12px rgba(239, 68, 68, 0.4) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
    }
    
    section[data-testid="stSidebar"] .stButton > button[kind="primary"]:hover,
    section[data-testid="stSidebar"] button[data-testid="baseButton-primary"]:hover {
        background: linear-gradient(135deg, #dc2626 0%, #b91c1c 100%) !important;
        transform: scale(1.05) !important;
        box-shadow: 0 6px 20px rgba(239, 68, 68, 0.6) !important;
        color: white !important;
    }
    
    /* Section anchors for scrolling */
    .section-anchor {
        scroll-margin-top: 2rem;
    }
    
    /* Smooth scrolling */
    html {
        scroll-behavior: smooth;
    }
    
    /* Navigation active state */
    .nav-active {
        background: rgba(255, 255, 255, 0.3) !important;
        border-left: 4px solid white !important;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# INIT
# -----------------------------
init_db()

st.session_state.setdefault("logged_in", False)
st.session_state.setdefault("page", "login")
st.session_state.setdefault("chat_history", [])

# -----------------------------
# LOGIN PAGE
# -----------------------------
def login_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # st.markdown("""
        # <div class="auth-card">
        # """, unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: #667eea; margin-bottom: 0.5rem;'>🌾</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'> AI Crop Disease Detection </h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #4a5568; font-size: 1.5rem; margin-bottom: 2rem;'>Welcome Back</h2>", unsafe_allow_html=True)

        username = st.text_input("👤 Username", placeholder="Enter your username")
        password = st.text_input("🔒 Password", type="password", placeholder="Enter your password")

        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("🚀 Login", use_container_width=True):
                if username and password:
                    if verify_user(username, password):
                        st.session_state.logged_in = True
                        st.session_state.page = "dashboard"
                        st.success("✅ Login successful!")
                        time.sleep(0.5)
                        st.rerun()
                    else:
                        st.error("❌ Invalid username or password")
                else:
                    st.warning("⚠️ Please enter both username and password")

        with col_btn2:
            if st.button("📝 Sign Up", use_container_width=True):
                st.session_state.page = "signup"
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# SIGNUP PAGE
# -----------------------------
def signup_page():
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        st.markdown("""
        <div class="auth-card">
        """, unsafe_allow_html=True)
        
        st.markdown("<h1 style='text-align: center; color: #667eea; margin-bottom: 0.5rem;'>🌱</h1>", unsafe_allow_html=True)
        st.markdown("<h1 style='text-align: center; margin-bottom: 2rem;'>Create Account</h1>", unsafe_allow_html=True)
        st.markdown("<h2 style='text-align: center; color: #4a5568; font-size: 1.2rem; margin-bottom: 2rem;'>Join Crop Health AI Today</h2>", unsafe_allow_html=True)

        username = st.text_input("👤 Choose Username", placeholder="Enter a unique username")
        password = st.text_input("🔒 Create Password", type="password", placeholder="Create a strong password")

        col_btn1, col_btn2 = st.columns(2)
        
        with col_btn1:
            if st.button("✨ Register", use_container_width=True):
                if username and password:
                    try:
                        add_user(username, password)
                        st.success("🎉 Account created successfully!")
                        time.sleep(1)
                        st.session_state.page = "login"
                        st.rerun()
                    except sqlite3.IntegrityError:
                        st.error("❌ Username already exists. Please choose another.")
                else:
                    st.warning("⚠️ Please fill in all fields")

        with col_btn2:
            if st.button("← Back to Login", use_container_width=True):
                st.session_state.page = "login"
                st.rerun()
        
        st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# PDF GENERATOR
# -----------------------------
def generate_pdf(crop, disease, confidence, advice):
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50

    # Title
    pdf.setFont("Helvetica-Bold", 18)
    pdf.drawString(50, y, "Crop Disease Detection Report")
    y -= 40

    # Basic Information
    pdf.setFont("Helvetica-Bold", 12)
    pdf.drawString(50, y, "Detection Information:")
    y -= 20
    pdf.setFont("Helvetica", 11)
    pdf.drawString(50, y, f"Crop Type: {crop}")
    y -= 18
    pdf.drawString(50, y, f"Disease: {disease}")
    y -= 18
    pdf.drawString(50, y, f"Confidence: {confidence*100:.2f}%")
    y -= 30

    # Cause
    if "cause" in advice:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Cause:")
        y -= 18
        pdf.setFont("Helvetica", 10)
        # Wrap text if too long
        cause_text = advice["cause"]
        lines = []
        words = cause_text.split()
        current_line = ""
        for word in words:
            if len(current_line + word) < 90:
                current_line += word + " "
            else:
                lines.append(current_line)
                current_line = word + " "
        if current_line:
            lines.append(current_line)
        
        for line in lines:
            pdf.drawString(50, y, line.strip())
            y -= 16
        y -= 10

    # Symptoms
    if "symptoms" in advice and advice["symptoms"]:
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Symptoms:")
        y -= 18
        pdf.setFont("Helvetica", 10)
        for symptom in advice["symptoms"]:
            if y < 50:
                pdf.showPage()
                y = height - 50
            pdf.drawString(60, y, f"• {symptom}")
            y -= 16
        y -= 10

    # Treatment Steps
    if "treatment" in advice and advice["treatment"]:
        if y < 100:
            pdf.showPage()
            y = height - 50
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Treatment Steps:")
        y -= 18
        pdf.setFont("Helvetica", 10)
        for i, step in enumerate(advice["treatment"], 1):
            if y < 50:
                pdf.showPage()
                y = height - 50
            # Wrap long lines
            step_text = f"{i}. {step}"
            words = step_text.split()
            current_line = ""
            for word in words:
                if len(current_line + word) < 85:
                    current_line += word + " "
                else:
                    pdf.drawString(60, y, current_line.strip())
                    y -= 16
                    current_line = word + " "
            if current_line:
                pdf.drawString(60, y, current_line.strip())
                y -= 16
        y -= 10

    # Prevention
    if "prevention" in advice and advice["prevention"]:
        if y < 100:
            pdf.showPage()
            y = height - 50
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Prevention Measures:")
        y -= 18
        pdf.setFont("Helvetica", 10)
        for tip in advice["prevention"]:
            if y < 50:
                pdf.showPage()
                y = height - 50
            pdf.drawString(60, y, f"• {tip}")
            y -= 16
        y -= 10

    # Requirements
    if "requirements" in advice and advice["requirements"]:
        if y < 100:
            pdf.showPage()
            y = height - 50
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Required Materials/Equipment:")
        y -= 18
        pdf.setFont("Helvetica", 10)
        for req in advice["requirements"]:
            if y < 50:
                pdf.showPage()
                y = height - 50
            pdf.drawString(60, y, f"• {req}")
            y -= 16
        y -= 10

    # Best Suggestion
    if "best_suggestion" in advice:
        if y < 100:
            pdf.showPage()
            y = height - 50
        pdf.setFont("Helvetica-Bold", 12)
        pdf.drawString(50, y, "Best Recommendation:")
        y -= 18
        pdf.setFont("Helvetica", 10)
        # Wrap text
        suggestion = advice["best_suggestion"]
        words = suggestion.split()
        current_line = ""
        for word in words:
            if len(current_line + word) < 90:
                current_line += word + " "
            else:
                if y < 50:
                    pdf.showPage()
                    y = height - 50
                pdf.drawString(50, y, current_line.strip())
                y -= 16
                current_line = word + " "
        if current_line:
            if y < 50:
                pdf.showPage()
                y = height - 50
            pdf.drawString(50, y, current_line.strip())

    # Footer
    pdf.setFont("Helvetica", 8)
    pdf.drawString(50, 30, f"Report generated on {datetime.now().strftime('%d-%m-%Y %H:%M:%S')}")
    pdf.drawString(50, 20, "Generated by Crop Health AI - Disease Detection System")

    pdf.save()
    buffer.seek(0)
    return buffer

# -----------------------------
# INTELLIGENT CHATBOT LOGIC
# -----------------------------
def chatbot_response(query: str) -> str:
    """
    Answer user questions about the detected disease using DISEASE_ADVICE
    and basic project information.
    """
    q = query.lower().strip()

    # Friendly greetings / small-talk
    if any(word in q for word in ["hi", "hello", "hey"]):
        return "Hello! 👋 I am your Crop Health AI assistant. Ask me about the disease, its cause, symptoms, treatment, prevention, or requirements."

    if "project" in q or "about you" in q:
        return "I am a Crop Health AI assistant. I analyze rice and pulse leaf images, detect diseases using deep learning, and then show you clear steps for treatment, prevention, and future requirements."

    if "who made" in q or "developer" in q or "created" in q:
        return "This application was created as a crop disease detection project using deep learning models trained on rice and pulse leaf images."

    # If no prediction yet, guide the user
    if "prediction_result" not in st.session_state or not st.session_state.prediction_result:
        return "Please upload a crop leaf image first. After prediction, I can give you detailed information about the disease, treatment, and prevention."

    # Fetch current disease advice
    pred = st.session_state.prediction_result
    crop = pred.get("crop")
    disease = pred.get("disease")
    key = f"{crop} - {disease}"
    advice = DISEASE_ADVICE.get(key)

    if not advice:
        return f"I detected **{disease}** in **{crop}**, but there is no detailed advice stored for this combination. Please consult a local agricultural expert for exact recommendations."

    # Cause of disease
    if any(word in q for word in ["cause", "reason", "why this", "why my"]):
        cause_text = advice.get("cause", "Cause information is not available.")
        return f"Cause of **{disease}** in **{crop}**:\n\n{cause_text}"

    # Symptoms
    if "symptom" in q or "sign" in q or "how to identify" in q:
        symptoms = advice.get("symptoms", [])
        if not symptoms:
            return f"Detailed symptom information for **{disease}** is not available, but you can usually see spots, color changes, or growth problems on the leaves."
        bullet_points = "\n".join([f"- {s}" for s in symptoms])
        return f"Key symptoms of **{disease}** in **{crop}** are:\n\n{bullet_points}"

    # Treatment steps / how to cure
    if any(word in q for word in ["treat", "cure", "control", "solution", "medicine", "spray"]):
        steps = advice.get("treatment", [])
        if not steps:
            return f"For **{disease}**, no specific treatment steps are stored. Please consult an agriculture officer for suitable chemicals and doses."
        bullet_points = "\n".join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        return f"Recommended treatment steps for **{disease}** in **{crop}**:\n\n{bullet_points}"

    # Prevention
    if any(word in q for word in ["prevent", "avoid", "future", "next time", "precaution"]):
        prev = advice.get("prevention", [])
        if not prev:
            return f"Prevention details for **{disease}** are not stored, but regular field monitoring, balanced fertilizer, and good drainage usually help reduce many diseases."
        bullet_points = "\n".join([f"- {p}" for p in prev])
        return f"To prevent **{disease}** in **{crop}** in the future, follow these points:\n\n{bullet_points}"

    # Requirements (materials / equipment)
    if any(word in q for word in ["requirement", "material", "equipment", "what do i need", "what i need"]):
        reqs = advice.get("requirements", [])
        if not reqs:
            return f"There is no specific list of materials stored for **{disease}**, but you will generally need proper fungicides/insecticides, a sprayer, safety gear, and good quality seeds."
        bullet_points = "\n".join([f"- {r}" for r in reqs])
        return f"For managing **{disease}** in **{crop}**, you will typically need:\n\n{bullet_points}"

    # Best suggestion / summary
    if any(word in q for word in ["best suggestion", "best advice", "what should i do", "summary", "overall"]):
        suggestion = advice.get("best_suggestion")
        if suggestion:
            return f"Best overall suggestion for **{disease}** in **{crop}**:\n\n{suggestion}"
        return f"For **{disease}**, follow the treatment and prevention steps shown above and keep monitoring your field closely."

    # Generic disease explanation
    if "disease" in q or "what is this" in q:
        base = f"The detected disease is **{disease}** in **{crop}**."
        cause = advice.get("cause")
        if cause:
            return f"{base}\n\nShort overview:\n{cause}"
        return base

    # Fallback help
    return (
        "I can help you with:\n"
        "- cause of the disease (ask: 'what is the cause?')\n"
        "- symptoms (ask: 'what are the symptoms?')\n"
        "- treatment (ask: 'how to treat or cure this disease?')\n"
        "- prevention (ask: 'how to prevent this in future?')\n"
        "- requirements (ask: 'what materials are required?')\n"
        "You can also ask about the project itself."
    )

# -----------------------------
# DASHBOARD
# -----------------------------
def dashboard():
    # Initialize navigation state
    if "current_section" not in st.session_state:
        st.session_state.current_section = "upload"
    
    # Sidebar - Enhanced Navigation Panel - ALWAYS VISIBLE
    with st.sidebar:
        # Force sidebar visibility with inline style

        
        st.markdown("""
        <div style='text-align: center; padding: 1.5rem 0; background: rgba(255,255,255,0.1); border-radius: 10px; margin-bottom: 1rem;'>
            <h1 style='color: white; font-size: 2.5rem; margin-bottom: 0.5rem; margin-top: 0;'>🌾</h1>
            <h2 style='color: white; font-size: 1.3rem; margin-bottom: 0; font-weight: 700;'>Crop Health AI</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Check if prediction exists
        has_prediction = "prediction_result" in st.session_state and st.session_state.prediction_result is not None
        
        # Navigation Menu - Only show if prediction exists (except Upload)
        if has_prediction:
            st.markdown("### 🧭 Quick Navigation")

            # Use HTML anchor links instead of buttons to avoid reruns
            nav_options = [
                ("📤 Upload Image", "upload"),
                ("📊 Detection Results", "results"),
                ("📋 Disease Information", "disease_info"),
                ("✅ Treatment Steps", "treatment"),
                ("🛡️ Prevention", "prevention"),
                ("📦 Requirements", "requirements"),
                ("💡 Best Suggestion", "suggestion"),
                ("🤖 AI Assistant", "assistant"),
            ]

            nav_html = """
            <style>
                .sidebar-nav-link {
                    display: block;
                    width: 100%;
                    padding: 0.6rem 0.9rem;
                    margin-bottom: 0.35rem;
                    border-radius: 6px;
                    color: #f7fafc;
                    text-decoration: none;
                    font-size: 0.95rem;
                    font-weight: 500;
                    background: rgba(15, 23, 42, 0.55);
                    border: 1px solid rgba(148, 163, 184, 0.4);
                    transition: all 0.15s ease;
                }
                .sidebar-nav-link:hover {
                    background: rgba(59, 130, 246, 0.85);
                    border-color: rgba(191, 219, 254, 0.8);
                    transform: translateX(2px);
                }
            </style>
            <div>
            """

            for label, section_id in nav_options:
                nav_html += f"<a class='sidebar-nav-link' href='#{section_id}'>{label}</a>"

            nav_html += "</div>"

            st.markdown(nav_html, unsafe_allow_html=True)
            
        else:
            # Before prediction - only show Upload option
            st.markdown("### 🧭 Quick Navigation")
            if st.button("📤 Upload Image", use_container_width=True, key="nav_upload_only"):
                st.session_state.current_section = "upload"
                st.markdown("""
                <script>
                    window.scrollTo({ top: 0, behavior: 'smooth' });
                </script>
                """, unsafe_allow_html=True)
            
            st.info("💡 Upload an image to see navigation options")
        
        st.markdown("---")
        
        # Quick Actions - Only show if prediction exists
        if has_prediction:
            st.markdown("### ⚡ Quick Actions")
            if st.button("🔄 Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()
            
            if st.button("📋 Clear Prediction", use_container_width=True):
                if "prediction_result" in st.session_state:
                    del st.session_state.prediction_result
                if "uploaded_file_key" in st.session_state:
                    del st.session_state.uploaded_file_key
                st.session_state.current_section = "upload"
                st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Logout Section - VERY PROMINENT at bottom - ALWAYS VISIBLE
        st.markdown("### 🚪 Account Management")
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.25); padding: 1.2rem; border-radius: 10px; border: 3px solid rgba(239, 68, 68, 0.6); margin-bottom: 1rem; box-shadow: 0 4px 12px rgba(239, 68, 68, 0.3);'>
            <p style='color: white; font-size: 0.95rem; margin: 0; text-align: center; font-weight: 600;'>Click below to securely log out</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Make logout button VERY prominent
        logout_clicked = st.button(
            "🚪 **LOGOUT** 🚪", 
            use_container_width=True, 
            type="primary", 
            key="logout_btn_main",
            help="Click to log out of your account"
        )
        
        if logout_clicked:
            st.session_state.logged_in = False
            st.session_state.page = "login"
            st.session_state.chat_history = []
            if "prediction_result" in st.session_state:
                del st.session_state.prediction_result
            if "uploaded_file_key" in st.session_state:
                del st.session_state.uploaded_file_key
            st.session_state.current_section = "upload"
            st.success("✅ Logged out successfully!")
            time.sleep(0.3)
            st.rerun()

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # About Section
        st.markdown("### ℹ️ About")
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; color: white; font-size: 0.85rem; line-height: 1.5;'>
            <p style='margin: 0;'>Upload an image of your crop leaf to detect diseases using AI-powered deep learning models.</p>
        </div>
        """, unsafe_allow_html=True)

    # Force sidebar visibility with JavaScript - AGGRESSIVE checks
    st.markdown("""
    <script>
        // Function to force sidebar visibility - ENHANCED
        function forceSidebarVisible() {
            // Try multiple selectors with more aggressive approach
            const selectors = [
                'section[data-testid="stSidebar"]',
                'div[data-testid="stSidebar"]',
                '[data-baseweb="drawer"]',
                '.css-1d391kg',
                '[class*="sidebar"]',
                '[class*="Sidebar"]'
            ];
            
            selectors.forEach(selector => {
                try {
                    const sidebar = document.querySelector(selector);
                    if (sidebar) {
                        sidebar.style.setProperty('display', 'block', 'important');
                        sidebar.style.setProperty('visibility', 'visible', 'important');
                        sidebar.style.setProperty('width', '300px', 'important');
                        sidebar.style.setProperty('min-width', '300px', 'important');
                        sidebar.style.setProperty('max-width', '300px', 'important');
                        sidebar.style.setProperty('opacity', '1', 'important');
                        sidebar.style.setProperty('transform', 'translateX(0)', 'important');
                        sidebar.style.setProperty('position', 'relative', 'important');
                        sidebar.setAttribute('aria-hidden', 'false');
                        sidebar.setAttribute('aria-expanded', 'true');
                    }
                } catch(e) {
                    console.log('Sidebar selector error:', selector);
                }
            });
            
            // Also check all sidebar-related elements
            try {
                const allSidebars = document.querySelectorAll('[data-testid*="Sidebar"], [class*="sidebar"], [class*="Sidebar"]');
                allSidebars.forEach(el => {
                    el.style.setProperty('display', 'block', 'important');
                    el.style.setProperty('visibility', 'visible', 'important');
                });
            } catch(e) {
                console.log('Sidebar query error');
            }
        }
        
        // Run immediately
        forceSidebarVisible();
        
        // Run on load
        window.onload = function() {
            forceSidebarVisible();
            setTimeout(forceSidebarVisible, 50);
            setTimeout(forceSidebarVisible, 100);
            setTimeout(forceSidebarVisible, 300);
            setTimeout(forceSidebarVisible, 500);
            setTimeout(forceSidebarVisible, 1000);
        };
        
        // Also run immediately if DOM is ready
        if (document.readyState === 'loading') {
            document.addEventListener('DOMContentLoaded', function() {
                forceSidebarVisible();
                setTimeout(forceSidebarVisible, 100);
            });
        } else {
            forceSidebarVisible();
            setTimeout(forceSidebarVisible, 100);
        }
        
        // Continuous check - more frequent
        setInterval(forceSidebarVisible, 1000);
        
        // Also use MutationObserver to catch dynamic changes
        const observer = new MutationObserver(function(mutations) {
            forceSidebarVisible();
        });
        
        observer.observe(document.body, {
            childList: true,
            subtree: true,
            attributes: true,
            attributeFilter: ['style', 'class', 'aria-hidden']
        });
        
        // Enhanced scroll function - MORE RELIABLE (REPLACES OLD ONE)
        function enhancedScrollToSection(sectionId) {{
            if (!sectionId || sectionId === 'upload' || sectionId === 'null' || sectionId === 'None' || sectionId === '') {{
                if (sectionId === 'upload') {{
                    window.scrollTo({{ top: 0, behavior: 'smooth' }});
                }}
                return;
            }}
            
            // Try multiple times with increasing delays
            const delays = [50, 150, 300, 500, 800, 1200, 2000, 3000];
            let executed = false;
            
            delays.forEach((delay) => {{
                setTimeout(function() {{
                    if (executed) return;
                    
                    const element = document.getElementById(sectionId);
                    if (element) {{
                        executed = true;
                        const rect = element.getBoundingClientRect();
                        const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
                        const offsetTop = rect.top + scrollTop - 120;
                        
                        window.scrollTo({{ top: offsetTop, behavior: 'smooth' }});
                        element.scrollIntoView({{ behavior: 'smooth', block: 'start', inline: 'nearest' }});
                        
                        // Double-check after a short delay
                        setTimeout(() => {{
                            const el = document.getElementById(sectionId);
                            if (el) {{
                                window.scrollTo({{ top: el.getBoundingClientRect().top + window.pageYOffset - 120, behavior: 'smooth' }});
                            }}
                        }}, 200);
                    }}
                }}, delay);
            }});
        }}
        
        // Get scroll target from session state
        const scrollTarget = '""" + str(st.session_state.get("scroll_target", "")) + """';
        const currentSection = '""" + str(st.session_state.get("current_section", "upload")) + """';
        
        // Determine which section to scroll to
        let sectionToScroll = '';
        if (scrollTarget && scrollTarget !== '' && scrollTarget !== 'None' && scrollTarget !== 'null') {{
            sectionToScroll = scrollTarget;
        }} else if (currentSection && currentSection !== '' && currentSection !== 'None' && currentSection !== 'null') {{
            sectionToScroll = currentSection;
        }}
        
        // Scroll after page loads - MULTIPLE ATTEMPTS
        if (sectionToScroll && sectionToScroll !== 'upload') {{
            // Immediate attempts
            setTimeout(() => enhancedScrollToSection(sectionToScroll), 100);
            setTimeout(() => enhancedScrollToSection(sectionToScroll), 300);
            
            // On DOM ready
            if (document.readyState === 'loading') {{
                document.addEventListener('DOMContentLoaded', function() {{
                    setTimeout(() => enhancedScrollToSection(sectionToScroll), 200);
                    setTimeout(() => enhancedScrollToSection(sectionToScroll), 500);
                }});
            }} else {{
                setTimeout(() => enhancedScrollToSection(sectionToScroll), 200);
                setTimeout(() => enhancedScrollToSection(sectionToScroll), 500);
            }}
            
            // On window load
            window.addEventListener('load', function() {{
                setTimeout(() => enhancedScrollToSection(sectionToScroll), 300);
                setTimeout(() => enhancedScrollToSection(sectionToScroll), 800);
            }});
            
            // Additional delayed attempts
            setTimeout(() => enhancedScrollToSection(sectionToScroll), 1000);
            setTimeout(() => enhancedScrollToSection(sectionToScroll), 2000);
        }} else if (sectionToScroll === 'upload') {{
            window.scrollTo({{ top: 0, behavior: 'smooth' }});
        }}
        
        // Clear scroll_target after use (prevents repeated scrolling)
        if (scrollTarget && scrollTarget !== '' && scrollTarget !== 'None') {{
            setTimeout(function() {{
                // Will be cleared on next Python rerun
            }}, 3500);
        }}
    </script>
    """, unsafe_allow_html=True)
    
    st.markdown("<h1 style='text-align: center; color: white; margin-bottom: 2rem;'>🌾 Crop Disease Detection Dashboard</h1>", unsafe_allow_html=True)
    
    # Upload section with anchor and styling
    st.markdown("""
    <div id="upload" class="section-anchor" style='background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin-bottom: 2rem;'>
    """, unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader(
        "📤 Upload Crop Image", 
        type=["jpg", "jpeg", "png"],
        help="Upload a clear image of a crop leaf for disease detection",
        key="crop_image_uploader"
    )
    
    # Store upload state in session state for navigation
    if uploaded_file:
        st.session_state.uploaded_file_key = uploaded_file.name
    else:
        if "uploaded_file_key" in st.session_state:
            del st.session_state.uploaded_file_key
    
    st.markdown("</div>", unsafe_allow_html=True)

    if uploaded_file:
        # Always open the image for display
        img = Image.open(uploaded_file)
        
        # Check if we already have a prediction for this file
        predict_again = True
        if "prediction_result" in st.session_state:
            if st.session_state.prediction_result and st.session_state.prediction_result.get("image") == uploaded_file.name:
                predict_again = False
                crop = st.session_state.prediction_result["crop"]
                disease = st.session_state.prediction_result["disease"]
                confidence = st.session_state.prediction_result["confidence"]
        
        if predict_again:
            # Show loading
            with st.spinner("🔍 Analyzing image..."):
                crop, disease, confidence = predict(img)
                # Store prediction result in session state
                st.session_state.prediction_result = {
                    "crop": crop,
                    "disease": disease,
                    "confidence": confidence,
                    "image": uploaded_file.name
                }
                time.sleep(0.5)  # Simulate processing
        
        # Results section with anchor - reduced padding
        st.markdown("""
        <div id="results" class="section-anchor" style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 1.5rem 0;'>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns([1, 1.2])

        with col1:
            st.markdown("### 📷 Uploaded Image")
            st.markdown("""
            <div class="image-container">
            """, unsafe_allow_html=True)
            st.image(img, use_container_width=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            st.markdown("### 🔬 Detection Results")
            
            # Metrics with better styling
            col_metric1, col_metric2, col_metric3 = st.columns(3)
            
            with col_metric1:
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🌾</div>
                    <div style='font-size: 1.2rem; font-weight: 700;'>{crop}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>Crop Type</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric2:
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>🦠</div>
                    <div style='font-size: 1rem; font-weight: 700;'>{disease}</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>Disease</div>
                </div>
                """, unsafe_allow_html=True)
            
            with col_metric3:
                confidence_pct = confidence * 100
                badge_color = "badge-success" if confidence_pct >= 80 else "badge-warning" if confidence_pct >= 60 else "badge-info"
                st.markdown(f"""
                <div class="metric-card">
                    <div style='font-size: 2rem; margin-bottom: 0.5rem;'>📊</div>
                    <div style='font-size: 1.5rem; font-weight: 700;'>{confidence_pct:.1f}%</div>
                    <div style='font-size: 0.9rem; opacity: 0.9;'>Confidence</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Progress bar for confidence
            st.markdown(f"""
            <div style='margin-top: 1.5rem;'>
                <div style='display: flex; justify-content: space-between; margin-bottom: 0.5rem;'>
                    <span style='font-weight: 600; color: #4a5568;'>Confidence Level</span>
                    <span style='font-weight: 600; color: #667eea;'>{confidence_pct:.1f}%</span>
                </div>
                <div class="progress-container">
                    <div class="progress-fill" style='width: {confidence_pct}%;'>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("</div>", unsafe_allow_html=True)

        key = f"{crop} - {disease}"

        if key in DISEASE_ADVICE:
            advice = DISEASE_ADVICE[key]

            # Disease Information Section - All Details with anchors - reduced padding
            st.markdown("""
            <div id="disease_info" class="section-anchor" style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 1.5rem 0;'>
            """, unsafe_allow_html=True)
            
            # Cause Section
            if "cause" in advice:
                st.markdown("### 📋 Disease Cause")
                st.markdown(f"""
                <div style='background: #f0f4ff; padding: 1.5rem; border-radius: 10px; margin: 1rem 0; border-left: 4px solid #667eea;'>
                    <p style='color: #2d3748; font-size: 1rem; line-height: 1.6; margin: 0;'>{advice["cause"]}</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Symptoms Section
            if "symptoms" in advice and advice["symptoms"]:
                st.markdown("---")
                st.markdown('<div id="symptoms" class="section-anchor"></div>', unsafe_allow_html=True)
                st.markdown("### 🔍 Disease Symptoms")
                for i, symptom in enumerate(advice["symptoms"], 1):
                    st.markdown(f"""
                    <div style='background: #fff5f5; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #fc8181;'>
                        <div style='display: flex; align-items: start;'>
                            <span style='background: #fc8181; color: white; width: 25px; height: 25px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; margin-right: 1rem; flex-shrink: 0;'>{i}</span>
                            <span style='color: #2d3748; font-size: 1rem;'>{symptom}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Treatment Steps Section with anchor
            if "treatment" in advice and advice["treatment"]:
                st.markdown("---")
                st.markdown('<div id="treatment" class="section-anchor"></div>', unsafe_allow_html=True)
                st.markdown("### ✅ Recommended Treatment Steps")
                for i, step in enumerate(advice["treatment"], 1):
                    st.markdown(f"""
                    <div style='background: #f0fdf4; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #48bb78;'>
                        <div style='display: flex; align-items: start;'>
                            <span style='background: #48bb78; color: white; width: 30px; height: 30px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; margin-right: 1rem; flex-shrink: 0;'>{i}</span>
                            <span style='color: #2d3748; font-size: 1rem; line-height: 1.5;'>{step}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Prevention Section with anchor
            if "prevention" in advice and advice["prevention"]:
                st.markdown("---")
                st.markdown('<div id="prevention" class="section-anchor"></div>', unsafe_allow_html=True)
                st.markdown("### 🛡️ Prevention Measures")
                for i, tip in enumerate(advice["prevention"], 1):
                    st.markdown(f"""
                    <div style='background: #fef3c7; padding: 1rem; border-radius: 10px; margin: 0.5rem 0; border-left: 4px solid #f59e0b;'>
                        <div style='display: flex; align-items: start;'>
                            <span style='background: #f59e0b; color: white; width: 25px; height: 25px; border-radius: 50%; display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.9rem; margin-right: 1rem; flex-shrink: 0;'>{i}</span>
                            <span style='color: #2d3748; font-size: 1rem; line-height: 1.5;'>{tip}</span>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
            
            # Requirements Section with anchor
            if "requirements" in advice and advice["requirements"]:
                st.markdown("---")
                st.markdown('<div id="requirements" class="section-anchor"></div>', unsafe_allow_html=True)
                st.markdown("### 📦 Required Materials & Equipment")
                st.markdown("""
                <div style='background: #f7fafc; padding: 1rem; border-radius: 10px; margin: 1rem 0;'>
                """, unsafe_allow_html=True)
                for req in advice["requirements"]:
                    st.markdown(f"""
                    <div style='padding: 0.75rem; margin: 0.5rem 0; background: white; border-radius: 8px; border-left: 3px solid #4299e1;'>
                        <span style='color: #2d3748; font-size: 1rem;'>📌 {req}</span>
                    </div>
                    """, unsafe_allow_html=True)
                st.markdown("</div>", unsafe_allow_html=True)
            
            # Best Suggestion Section with anchor
            if "best_suggestion" in advice:
                st.markdown("---")
                st.markdown('<div id="suggestion" class="section-anchor"></div>', unsafe_allow_html=True)
                st.markdown("### 💡 Best Recommendation")
                st.markdown(f"""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); padding: 2rem; border-radius: 15px; margin: 1rem 0; box-shadow: 0 5px 15px rgba(102, 126, 234, 0.3);'>
                    <div style='color: white; font-size: 1.1rem; line-height: 1.8; font-weight: 500;'>
                        {advice["best_suggestion"]}
                    </div>
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

            # PDF Download
            pdf = generate_pdf(crop, disease, confidence, advice)
            st.download_button(
                "📄 Download Full Disease Report (PDF)",
                pdf,
                file_name=f"disease_report_{crop}_{disease.replace(' ', '_')}.pdf",
                mime="application/pdf",
                use_container_width=True
            )

            # Chatbot section with anchor - reduced padding
            st.markdown("""
            <div id="assistant" class="section-anchor" style='background: white; padding: 1.5rem; border-radius: 15px; box-shadow: 0 10px 30px rgba(0,0,0,0.2); margin: 1.5rem 0;'>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🤖 AI Assistant")
            st.markdown("Ask me anything about crop diseases, treatment, or prevention!")
            
            user_q = st.text_input("💬 Your question", placeholder="e.g., How to prevent this disease?")
            
            col_chat1, col_chat2 = st.columns([4, 1])
            with col_chat1:
                pass
            with col_chat2:
                ask_clicked = st.button("Send ➤", use_container_width=True)
            
            if ask_clicked and user_q.strip():
                reply = chatbot_response(user_q)
                st.session_state.chat_history.append(("You", user_q))
                st.session_state.chat_history.append(("Bot", reply))
                st.rerun()

            # Chat history display
            if st.session_state.chat_history:
                st.markdown("---")
                st.markdown("### 💭 Conversation History")
            for sender, msg in st.session_state.chat_history[-6:]:
                    if sender == "You":
                        st.markdown(f"""
                        <div class="chat-user">
                            <strong>You:</strong> {msg}
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown(f"""
                        <div class="chat-bot">
                            <strong>🤖 Bot:</strong> {msg}
                        </div>
                        """, unsafe_allow_html=True)
            
            st.markdown("</div>", unsafe_allow_html=True)

        else:
            st.warning("⚠️ No detailed advice available for this disease combination. Please consult with an agricultural expert.")

# -----------------------------
# ROUTER
# -----------------------------
if st.session_state.logged_in:
    dashboard()
else:
    if st.session_state.page == "login":
        login_page()
    else:
        signup_page()
        
