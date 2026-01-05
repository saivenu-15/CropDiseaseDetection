import streamlit as st
from auth import init_db, add_user, verify_user
from model_predict import predict
from PIL import Image
import time
import sqlite3
from chatbot import ProjectChatBot   # 👈 chatbot import

# -------------------------
# INIT
# -------------------------
init_db()
bot = ProjectChatBot()  # 👈 chatbot object

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if "page" not in st.session_state:
    st.session_state.page = "login"

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# -------------------------
# LOGIN PAGE
# -------------------------
def login_page():
    st.title("🔐 Login")

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if verify_user(username, password):
            st.session_state.logged_in = True
            st.session_state.page = "dashboard"
            st.success("Login successful!")
            st.rerun()
        else:
            st.error("Invalid username or password")

    if st.button("Create New Account"):
        st.session_state.page = "signup"
        st.rerun()

# -------------------------
# SIGNUP PAGE
# -------------------------
def signup_page():
    st.title("🆕 Create Account")

    username = st.text_input("New Username")
    password = st.text_input("New Password", type="password")

    if st.button("Register"):
        try:
            add_user(username, password)
            st.success("Account created successfully!")
            time.sleep(1)
            st.session_state.page = "login"
            st.rerun()
        except sqlite3.IntegrityError:
            st.error("Username already exists")

    if st.button("Back to Login"):
        st.session_state.page = "login"
        st.rerun()

# -------------------------
# DASHBOARD
# -------------------------
def dashboard():
    st.title("🌾 Crop Disease Prediction")

    # ---------- Image Upload ----------
    uploaded_file = st.file_uploader(
        "Upload an image", type=["jpg", "jpeg", "png"]
    )

    if uploaded_file:
        img = Image.open(uploaded_file)
        st.image(img, caption="Uploaded Image", use_column_width=True)

        result = predict(img)
        st.success(f"Prediction: {result}")

    st.markdown("---")

    # ---------- CHATBOT ----------
    st.subheader("💬 🤖 Project Assistant Bot")

    user_msg = st.text_input("Ask something about the project")

    if st.button("Send"):
        if user_msg.strip() != "":
            reply = bot.get_response(user_msg)
            st.session_state.chat_history.append(("You", user_msg))
            st.session_state.chat_history.append(("Bot", reply))
            st.rerun()

    for sender, msg in st.session_state.chat_history:
        if sender == "You":
            st.markdown(f"**🧑 You:** {msg}")
        else:
            st.markdown(f"**🤖 Bot:** {msg}")

    st.markdown("---")

    # ---------- LOGOUT ----------
    if st.button("Logout"):
        st.session_state.logged_in = False
        st.session_state.page = "login"
        st.session_state.chat_history = []
        st.rerun()

# -------------------------
# ROUTER
# -------------------------
if st.session_state.logged_in:
    dashboard()
else:
    if st.session_state.page == "login":
        login_page()
    elif st.session_state.page == "signup":
        signup_page()
