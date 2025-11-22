import streamlit as st

from logics.vector_db import get_collection_stats

st.title("ℹ️ Project Overview")

st.markdown("""
### 1. Project Scope
The **Bus Service Variation Chatbot** is an AI-powered analytical tool designed to assist transport planners and analysts in querying, tracking, and understanding bus service variations. 

The scope covers:
*   **Historical Analysis**: Retrieving past service changes (e.g., route amendments, capacity adjustments).
*   **Trend Identification**: Summarizing improvements by operator, region (BCM Package), or improvement type.
*   **Data Privacy**: Demonstrating secure AI adoption by masking sensitive operational data before processing.

### 2. Objectives
*   **Efficiency**: Reduce time spent manually searching through Excel logs for service history.
*   **Accessibility**: Allow non-technical users to query complex datasets using natural language.
*   **Security**: Prove that Large Language Models (LLMs) can be safely used with sensitive transport data via a "Data Firewall" architecture.
*   **Insight**: Go beyond simple keyword search to understand the *context* of changes (e.g., "Show me all reliability improvements").

### 3. Key Features
*   **Natural Language Querying**: Ask questions like *"What changes happened to Service 123 in 2024?"* or *"List all double-decker deployments."*
*   **Hybrid Search Engine**: Automatically switches between **Strict Filtering** (for specific facts) and **Semantic Search** (for broad topics).
*   **Privacy-First Architecture**: Sensitive entities (Service Nos, Dates, Operators) are masked locally before reaching the AI model.
*   **Smart Acronym Recognition**: Automatically detects and explains transport acronyms (e.g., BCM, BSEP, PTO).
*   **Dynamic Data Ingestion**: Supports uploading new datasets which are instantly indexed for search.

### 4. Data Sources & Synthetic Data
This application is currently running on a **Synthetic Dataset** designed to mirror the structure of real operational logs.

*   **Source Format**: Structured Excel logs containing Service No, Operator, Implementation Date, Improvement Type, and Free-text Details.
*   **Synthetic Nature**: All service numbers, dates, and specific details in this demo are generated for illustration. They do not reflect actual LTA/PTO operations.
*   **Data Masking**: Even though the data is synthetic, the system treats it as "Confidential" to demonstrate the privacy masking capabilities.
""")

st.divider()

st.header("Database Status")
st.write("Current volume of indexed service improvement records:")
    
# Database stats
stats = get_collection_stats()
if stats:
    st.metric("📊 Total Records", stats['total_documents'])
    st.caption(f"Model: {stats['embedding_model']}")