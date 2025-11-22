import streamlit as st
import pandas as pd

from logics.query_handler import process_user_message
from logics.vector_db import initialize_vector_db

# Note: Page config is handled by the main entry point (Chatbot.py)

@st.cache_resource
def setup_vector_db():
    return initialize_vector_db()


loading_status = st.status("🔄 Loading knowledge base...", expanded=True)

try:
    with loading_status:
        loading_status.write("Connecting to vector store…")
        vector_store = setup_vector_db()

    loading_status.update(
        label="✅ Knowledge base ready.",
        state="complete",
        expanded=False,
    )
    loading_status.empty()  # clears the dropdown entirely
except Exception as exc:
    loading_status.update(
        label=f"⚠️ Database not found or empty. Please upload data.",
        state="error",
        expanded=True,
    )
    # Do not stop execution, allow UI to load so user can navigate to Upload page
    # st.stop() 


if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "current_interaction" not in st.session_state:
    st.session_state.current_interaction = None

if "processing" not in st.session_state:
    st.session_state.processing = False


def build_context_prompt(history: list[dict], current_prompt: str) -> str:
    raw_prompt = current_prompt.strip()
    if not history:
        return raw_prompt

    fragments = []
    for turn in history[-3:]:
        if turn.get("user"):
            fragments.append(f"User: {turn['user']}")
        if turn.get("assistant"):
            fragments.append(f"Assistant: {turn['assistant']}")
    history_text = "\n".join(fragments)

    return (
        "Previous conversation:\n"
        f"{history_text}\n\n"
        f"Latest question:\n{raw_prompt}"
    )


st.title("Bus Service Variation Chatbot")
st.caption("Submit a question, see results, and keep prior answers in the history panel.")

# IMPORTANT NOTICE: proof-of-concept disclaimer
with st.expander("IMPORTANT NOTICE: Prototype Disclaimer", expanded=False):
    st.markdown(
        """
        **IMPORTANT NOTICE:** This web application is developed as a proof-of-concept prototype. The information provided here is **NOT** intended for actual usage and should not be relied upon for making any decisions, especially those related to financial, legal, or healthcare matters.

        Furthermore, please be aware that the LLM may generate inaccurate or incorrect information. You assume full responsibility for how you use any generated output.

        Always consult with qualified professionals for accurate and personalized advice.
        """
    )

# Use a form to prevent double-click issues and enable "Enter to submit"
with st.form(key="query_form"):
    user_prompt = st.text_area("Enter your question", height=100, placeholder="e.g. Show improvements for service 154 in 2024")
    submit_button = st.form_submit_button("Submit", type="primary", width='content', disabled=st.session_state.processing)

if submit_button:
    question = user_prompt.strip()

    if not question:
        st.warning("⚠️ Please enter a question.")
    else:
        if st.session_state.current_interaction:
            st.session_state.chat_history.append(st.session_state.current_interaction)
            st.session_state.current_interaction = None

        st.session_state.processing = True
        try:
            # Pass raw question to router, not history-augmented prompt
            # The router needs the clean intent (e.g. "Show svc 154")
            with st.spinner("Generating answer..."):
                response, service_details, extracted_filters = process_user_message(
                    question,  # Pass raw question for better routing
                    vector_store,
                )

            st.session_state.current_interaction = {
                "user": question,
                "assistant": response,
                "results": service_details or [],
                "filters": extracted_filters or {},
            }
        except Exception as exc:
            st.error(f"❌ Error processing query: {exc}")
        finally:
            st.session_state.processing = False

st.divider()

current = st.session_state.current_interaction
if current:
    st.subheader("Current Answer")
    with st.chat_message("user", avatar="🙋"):
        st.markdown(current["user"])
    with st.chat_message("assistant", avatar="🤖"):
        st.markdown(current["assistant"])

    results = current.get("results") or []
    if results:
        st.write(f"📊 Showing {min(len(results), 10)} of {len(results)} results")
        df = pd.DataFrame(results)
        visible_cols = [col for col in df.columns if not col.startswith("_")]
        st.dataframe(df[visible_cols].head(10), width='stretch', hide_index=True)
    else:
        st.info("No matching service improvements were found.")

    filters = current.get("filters") or {}
    if filters:
        with st.expander("🔧 Extracted filters"):
            for key, value in filters.items():
                # Clean up date range display
                if key == "Implementation Date" and isinstance(value, dict):
                    date_str = []
                    if "$gte" in value: date_str.append(f"From {value['$gte']}")
                    if "$lte" in value: date_str.append(f"To {value['$lte']}")
                    st.write(f"- **Date Range:** {' and '.join(date_str)}")
                else:
                    st.write(f"- **{key}:** {value}")

st.divider()

history = st.session_state.chat_history
if history:
    st.subheader("Conversation History")
    for idx, turn in enumerate(reversed(history[-10:]), start=1):
        st.markdown(f"### Interaction #{len(history) - (idx - 1)}")
        with st.chat_message("user", avatar="🙋"):
            st.markdown(turn["user"])
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(turn["assistant"])

        results = turn.get("results") or []
        if results:
            with st.expander("📊 View prior results"):
                df = pd.DataFrame(results)
                visible_cols = [col for col in df.columns if not col.startswith("_")]
                st.dataframe(df[visible_cols].head(10), width='stretch', hide_index=True)

st.caption("🔒 Queries are processed securely and not stored permanently.")
