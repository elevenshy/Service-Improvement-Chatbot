import streamlit as st
import pandas as pd

st.title("System Methodology & Architecture")
st.markdown("### How the Service Variations Chatbot Works")

st.info(
    """
    This chatbot uses a **Retrieval-Augmented Generation (RAG)** architecture, enhanced with a **Hybrid Search Strategy** 
    that combines semantic understanding with strict database filtering. This ensures high accuracy for specific data lookups 
    while maintaining the ability to answer broad, conceptual questions.
    """
)

# --- Section 1: Architecture Overview ---
st.header("1. System Architecture")
st.markdown("""
The system is composed of four main intelligence layers:
1.  **The Ingestion Layer**: Converts raw Excel data into "Semantic Mini-Stories" with **Privacy Masking**.
2.  **The Routing Layer**: Analyzes user intent to decide *how* to search.
3.  **The Retrieval Layer**: Executes either a "Strict Deterministic Fetch" or a "Semantic Vector Search".
4.  **The Privacy Layer**: Ensures sensitive data is tokenized before leaving the local environment and detokenized upon return.
""")

# --- Section 2: Use Case Process Flows ---
st.header("2. Use Case Process Flows")
st.markdown("Visualizing how the system handles different types of user interactions.")

tab_flow1, tab_flow2 = st.tabs(["Use Case A: Intelligent Search (Specific)", "Use Case B: Chat / Semantic Discovery (Broad)"])

with tab_flow1:
    st.subheader("Use Case A: Intelligent Search (Specific)")
    st.markdown("Triggered when the user asks for specific facts (e.g., *'Show changes for Svc 123'*).")
    st.graphviz_chart('''
    digraph {
        rankdir=LR;
        node [shape=box, style=rounded];
        U [label="User Query"];
        R [label="Router Analysis", shape=diamond, style=filled, fillcolor=lightblue];
        F [label="Extract Filters\\n(Svc: 123)", shape=note];
        DB [label="Strict DB Query", shape=cylinder, style=filled, fillcolor=gold];
        REL [label="Find Related\\n(Swaps)", shape=cylinder, style=filled, fillcolor=gold];
        M [label="Merge Results"];
        LLM [label="LLM Synthesis", shape=component, style=filled, fillcolor=lightgreen];
        A [label="Final Answer"];

        U -> R;
        R -> F [label="Intent: SPECIFIC"];
        F -> DB;
        F -> REL;
        DB -> M;
        REL -> M;
        M -> LLM;
        LLM -> A;
    }
    ''')

with tab_flow2:
    st.subheader("Use Case B: Chat / Semantic Discovery (Broad)")
    st.markdown("Triggered when the user asks conceptual questions (e.g., *'How is reliability improved?'*).")
    st.graphviz_chart('''
    digraph {
        rankdir=LR;
        node [shape=box, style=rounded];
        U [label="User Query"];
        R [label="Router Analysis", shape=diamond, style=filled, fillcolor=lightblue];
        E [label="Generate Embedding", shape=ellipse];
        V [label="Vector Search\\n(Cosine Similarity)", shape=cylinder, style=filled, fillcolor=violet];
        LLM [label="LLM Synthesis", shape=component, style=filled, fillcolor=lightgreen];
        A [label="Final Answer"];

        U -> R;
        R -> E [label="Intent: BROAD"];
        E -> V;
        V -> LLM;
        LLM -> A;
    }
    ''')

# --- Section 3: Data Privacy & Security ---
st.header("3. Data Privacy & Security: The 'Data Firewall'")
st.info(" **Zero-Leakage Architecture**: No raw sensitive data is ever sent to the AI Provider.")

col_p1, col_p2 = st.columns([1, 1])

with col_p1:
    st.markdown("""
    **The Challenge:**  
    The dataset contains classified information (e.g., future plans, specific trip counts, internal remarks) that cannot be exposed to external AI models (OpenAI/Azure).
    
    **The Solution: Tokenization Strategy**  
    We implement a strict **Data Firewall** that intercepts all data before it leaves the local server.
    

    **How it works:**
    1.  **Local Masking**: Before embedding or querying, sensitive values are replaced with generic tokens.
        *   *Service 123* → `<<SVC_1>>`
        *   *Operator XYZ* → `<<PTO_1>>`
        *   *2025-01-01* → `<<DATE_1>>`
    2.  **Selective Transparency**: While "Who" (Service) and "When" (Date) are masked, the "What" (Metrics like "Added 2 buses") remains unmasked. This allows the AI to reason about the *nature* of the change (e.g., "Capacity increase") without knowing *which* service it applies to.
    3.  **Blind AI Processing**: The AI model processes these tokens. It understands the *structure* ("`<<SVC_1>>` had 2 buses added") but not the *identity*.
    4.  **Local Detokenization**: When the AI response returns, the system locally swaps the tokens back to their original values before displaying the answer to you.
    """)

with col_p2:
    st.caption("Visualizing the Data Flow:")
    st.code("""
    [Local Server]              [AI Provider]
    Raw Data: "Svc 123"   -->   Masked: "<<SVC_1>>"
    Metric: "Add 2 Bus"   -->   Raw:    "Add 2 Bus"
                                      |
                                      v
    User View: "Svc 123"  <--   Response: "<<SVC_1>>"
    """, language="text")

# --- Section 4: Deployment & Infrastructure Security ---
st.header("4. Deployment & Infrastructure Security")
st.warning("⚠️ **Deployment Strategy that keep data security in mind**")

st.markdown("""
**Current Demo Deployment (Streamlit Cloud):**
*   **Data Source**: The data currently loaded in this public demo is **100% Synthetic**. It is generated for illustration purposes to demonstrate the system's capabilities without exposing real operational data.
*   **Security**: Since the data is synthetic, strict infrastructure security is not required for this specific instance.

**Production Deployment (Airbase):**
*   **Containerization**: For actual operational use, the application is containerized (using Docker).
*   **Platform**: The container is deployed to **Airbase**, a secure internal platform.
*   **Data Security**: In the Airbase environment, the application connects to secure, persistent storage for real operational data, ensuring that sensitive information remains within the organization's trust boundary.
""")

# --- Section 5: Data Processing ---
st.header("5. Data Processing: The 'Mini-Story' Approach")
col1, col2 = st.columns([1, 1])

with col1:
    st.markdown("""
    **The Challenge:**  
    Raw Excel rows (e.g., `Svc: 123, Type: Imp, Date: 2024-01-01`) are hard for AI models to "understand" semantically.
    
    **The Solution:**  
    We convert every row into a natural language **"Mini-Story"**.
    
    **Key Features:**
    *   **Narrative Generation**: Converts codes into sentences (e.g., "Service 123 operated by SBS Transit underwent a Route Extension...").
    *   **Relationship Extraction**: We use Regex to detect "Swaps" in the free-text details.
        *   *Example:* If the text says "Swap in from Svc 154", the system tags this record as related to **Service 154** as well.
    """)

with col2:
    st.caption("Example of a Semantic Document:")
    st.code("""
    "Service Improvement Record for Bus Service 123. 
    In Q1 2024, on 2024-01-15, Service 123 operated by SBS Transit 
    underwent a Route Amendment. 
    Details: Route extended to new interchange. 
    This event involves an interaction with Service(s): 154.
    Key statistics: 0 trips changed, 2 additional buses added."
    """, language="text")

# --- Section 6: The 'Brain': Query Router & Intent Analysis ---
st.header("6. The 'Brain': Query Router & Intent Analysis")
st.markdown("""
Before searching, the system analyzes the user's question to determine the best strategy.
""")

with st.expander("See how the Router analyzes a query", expanded=True):
    st.markdown("#### Input: *'Show me all changes for Service 123 since Jan 2024'*")
    
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("**1. Intent Classification**")
        st.success("SPECIFIC")
        st.caption("User wants a list/facts, not a general explanation.")
    
    with c2:
        st.markdown("**2. Filter Extraction**")
        st.code("""
{
  "Svc No": "123",
  "Date": {"$gte": "2024-01-01"}
}
        """, language="json")
    
    with c3:
        st.markdown("**3. Strategy Selection**")
        st.warning("STRICT MODE")
        st.caption("Disable fuzzy search. Use Database Filters.")

# --- Section 7: Hybrid Search Strategy ---
st.header("7. Hybrid Search Engine")
st.markdown("The system dynamically switches between two search modes based on the Router's decision.")

tab1, tab2 = st.tabs(["⚡ Strict Mode (Deterministic)", "🌊 Semantic Mode (Vector)"])

with tab1:
    st.markdown("### Used for: Specific Lookups (e.g., 'Svc 123', 'Q1 Changes')")
    st.markdown("""
    When specific filters (Service No, Date, Operator) are detected, we **bypass** the vector similarity search.
    
    **Logic:**
    1.  **Primary Fetch**: Query the database directly for `Svc No == 123`.
    2.  **Secondary 'Swap' Fetch**: 
        *   The system knows that Service 123 might be mentioned in *other* services' records (e.g., "Swap with Svc 123").
        *   It runs a parallel query for records where `related_svc` contains "123".
    3.  **Merge**: Combines both sets of results to give a complete picture.
    """)

with tab2:
    st.markdown("### Used for: Broad Questions (e.g., 'How is reliability improved?')")
    st.markdown("""
    When no specific filters are found, we use **Cosine Similarity Search**.
    
    **Logic:**
    1.  Convert user question into a vector embedding (using OpenAI `text-embedding-3-small`).
    2.  Find the nearest "Mini-Stories" in the vector space.
    3.  Retrieve the top 20-30 most semantically similar records.
    """)

# --- Section 8: Answer Synthesis ---
st.header("8. Final Answer Generation")
st.markdown("""
Once the relevant records are retrieved, they are passed to **GPT-4o-mini** with a specialized system prompt.

**The Prompt Instructions:**
*   **Be Honest**: If the data isn't there, say so.
*   **Cite Evidence**: Always mention specific dates and service numbers.
*   **Context Aware**: If the user asked for a "List", format as a list. If they asked for a "Summary", write a paragraph.
""")

# --- Section 9: Data Ingestion & Updates ---
st.header("9. Data Ingestion & Updates")
st.markdown("""
The system supports dynamic knowledge updates without requiring code changes.

**The Upload Process:**
1.  **File Upload**: Users submit an Excel file via the **Upload New Data** page.
2.  **Schema Validation**: The system automatically checks for required columns (e.g., `Svc No`, `Implementation Date`, `Details`) to ensure the data structure matches the AI's expectations.
3.  **Processing & Embedding**:
    *   Valid rows are converted into "Mini-Stories" (as described in Section 4).
    *   New embeddings are generated using OpenAI's model.
4.  **Incremental Indexing**: The new vectors are added to the existing ChromaDB collection. The chatbot becomes aware of the new information immediately after the upload completes.
""")

st.divider()

st.subheader("View Raw Data")
with st.expander("Preview Dataset (Last 10 Records)"):
    # Load Excel file
    filepath = './data/data.xlsx'
    try:
        df = pd.read_excel(filepath, sheet_name=0)
        
        # Ensure Svc No is string to avoid PyArrow errors
        if "Svc No" in df.columns:
            df["Svc No"] = df["Svc No"].astype(str)
            
        # Get last 10 rows
        df_last_10 = df.tail(10)
        st.dataframe(df_last_10)
    except Exception as e:
        st.error(f"Could not load data: {e}")