import streamlit as st

# Page config must be first
st.set_page_config(page_title="Service Variations Chatbot", layout="wide")

def login_screen():
    st.title("Login")
    from helper_functions.utility import check_password
    check_password()

# Check authentication state
if not st.session_state.get("password_correct", False):
    # Show only the login page
    pg = st.navigation([st.Page(login_screen, title="Login")])
    pg.run()
else:
    # Show the full application
    pg = st.navigation([
        st.Page("pages/1_Chatbot_Interface.py", title="Chatbot", icon="🚌"),
        st.Page("pages/2_Upload_New_Data.py", title="Upload Data", icon="📤"),
        st.Page("pages/3_Methodology.py", title="Methodology", icon="🧠"),
        st.Page("pages/4_About.py", title="About", icon="ℹ️"),
    ])
    pg.run()