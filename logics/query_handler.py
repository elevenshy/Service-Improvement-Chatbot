import streamlit as st
import pandas as pd
from helper_functions import llm
from logics.vector_db import search_similar_improvements
import json
from datetime import datetime
import re

@st.cache_data
def load_acronym_knowledge():
    """Load acronyms into a dictionary for efficient lookup"""
    acronyms = pd.read_excel('./data/acronym.xlsx')
    acronym_dict = {}
    for _, row in acronyms.iterrows():
        acronym_dict[row['Acronym'].upper().strip()] = row['Meaning'].strip()
    return acronym_dict

def detect_acronyms_in_text(text, acronym_dict):
    """Detect all acronyms mentioned in the given text"""
    if not text:
        return {}
    
    text_upper = str(text).upper()
    detected_acronyms = {}
    
    for acronym, meaning in acronym_dict.items():
        # Use word boundaries to match whole words only
        if re.search(r'\b' + re.escape(acronym) + r'\b', text_upper):
            detected_acronyms[acronym] = meaning
    
    return detected_acronyms

def detect_acronyms_in_results(si_details, acronym_dict):
    """Detect acronyms mentioned in search results"""
    detected_acronyms = {}
    
    for result in si_details:
        for key, value in result.items():
            if key != "embedding" and value:
                result_acronyms = detect_acronyms_in_text(str(value), acronym_dict)
                detected_acronyms.update(result_acronyms)
    
    return detected_acronyms

def build_smart_acronym_context(query_acronyms, result_acronyms):
    """Build context with only relevant acronyms, prioritizing query acronyms"""
    all_acronyms = {}
    
    # Priority 1: Acronyms from user query
    all_acronyms.update(query_acronyms)
    
    # Priority 2: Acronyms from results (but limit to avoid token bloat)
    result_count = 0
    max_result_acronyms = 5  # Limit result acronyms to save tokens
    
    for acronym, meaning in result_acronyms.items():
        if acronym not in all_acronyms and result_count < max_result_acronyms:
            all_acronyms[acronym] = meaning
            result_count += 1
    
    if not all_acronyms:
        return ""
    
    # Build compact context
    context = "Relevant acronyms:\n"
    for acronym, meaning in sorted(all_acronyms.items()):
        context += f"• {acronym}: {meaning}\n"
    
    return context + "\n"

def expand_query_with_acronyms(user_message, query_acronyms):
    """Expand query with acronym meanings for better semantic search"""
    if not query_acronyms:
        return user_message
    
    expanded_query = user_message
    
    # Add meanings to help semantic search
    for acronym, meaning in query_acronyms.items():
        if len(meaning.split()) <= 4:  # Only add short meanings to avoid query bloat
            expanded_query += f" {meaning}"
    
    return expanded_query

# --- DATA MASKING / TOKENIZATION LOGIC ---
class DataMasker:
    """
    Handles tokenization of sensitive data before sending to LLM,
    and detokenization of the LLM's response.
    """
    def __init__(self):
        self.token_map = {} # Token -> Original Value
        self.value_map = {} # (Prefix, Original Value) -> Token
        self.counters = {}  # Prefix -> Count

    def mask(self, text, prefix="DATA"):
        """Replaces text with a token like <<PREFIX_1>>"""
        if not text or str(text).lower() == 'nan' or str(text).strip() == "":
            return text
            
        val_str = str(text).strip()
        
        # Scope caching by prefix to ensure different field types don't share tokens
        # (e.g. Service "2" and 2 Trips should be different tokens)
        # But Svc No and Related Svc (both "SVC") WILL share tokens.
        cache_key = (prefix, val_str)
        
        if cache_key in self.value_map:
            return self.value_map[cache_key]
            
        # Create new token
        if prefix not in self.counters:
            self.counters[prefix] = 0
        self.counters[prefix] += 1
        
        token = f"<<{prefix}_{self.counters[prefix]}>>"
        
        self.token_map[token] = val_str
        self.value_map[cache_key] = token
        return token

    def unmask(self, text):
        """Replaces all tokens in text with their original values"""
        if not text: return ""
        
        result = text
        # Sort by length descending to avoid partial replacements
        for token, value in sorted(self.token_map.items(), key=lambda x: len(x[0]), reverse=True):
            result = result.replace(token, value)
            
        return result

    def apply_mask_to_query(self, query_text):
        """
        Replaces known values in the query with their tokens so the LLM can match them to the context.
        """
        if not query_text: return ""
        
        masked_query = query_text
        
        # Collect all replacements from value_map
        replacements = []
        seen_values = set()
        
        # Prioritize SVC and PTO prefixes
        priority_prefixes = ["SVC", "PTO", "BCM", "DATE"]
        
        # First pass: Priority prefixes
        for (prefix, val), token in self.value_map.items():
            if prefix in priority_prefixes:
                if val not in seen_values:
                    replacements.append((val, token))
                    seen_values.add(val)
        
        # Second pass: Others
        for (prefix, val), token in self.value_map.items():
            if val not in seen_values:
                replacements.append((val, token))
                seen_values.add(val)
                
        # Sort by length of original value (descending) to prevent partial replacements
        replacements.sort(key=lambda x: len(x[0]), reverse=True)
        
        for val, token in replacements:
            # Use word boundary for safety
            pattern = r'\b' + re.escape(val) + r'\b'
            masked_query = re.sub(pattern, token, masked_query, flags=re.IGNORECASE)
            
        return masked_query

def analyze_query_requirements(user_message):
    """
    Combined Router & Filter Extractor:
    1. Extracts key filters (Svc No, PTO, Date Ranges, etc.)
    2. Determines Search Intent (SPECIFIC vs BROAD)
    """
    
    analysis_prompt = f"""Analyze this bus service variation query.
    
    User Query: "{user_message}"
    
    Task 1: Extract Filters (JSON)
    - **Svc No**: e.g., "123", "45A"
    - **PTO**: e.g., "SBST", "SMRT", "TTS", "GAS"
    - **Improvement / Degrade**: Extract one of these EXACT values if relevant:
      "Adjustment", "Degradation", "Improvement", "New Route", "No change", "Non-AI Improvement", "Offpeak Improvement", "Route amendment", "Route extension", "Withdrawn".
    - **Type of Improvement / Degrade**: Extract one of these EXACT values if relevant: "Additional bus stops", "Additional trips", "Adjust Trip Timing", "Adjustment", "Advance first trip", "Bring forward 1st bus", "BSS change", "Change trip block", "Cut operating hours", "Degradation", "Degrade headways", "Extend last bus timing", "Extend operating hours", "Extend SWT/HWT route", "Improvement", "Maintain service level", "New Route", "No change", "PTO change", "Remove bus stops", "Route Amendment", "Route extension", "Route Truncation", "Shift bus stop", "Shift operating base", "Swap hi-cap buses", "Withdrawn".
    - **BCM Package**: e.g., "Bukit Merah", "Clementi", "Sengkang-Hougang", etc.
    - **Day Type**: Extract relevant day types. Preferred terms: "Weekday", "Weekend", "Daily", "Public Holiday", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun". 
      **IMPORTANT**: Do NOT convert specific days (e.g. "Sat") to categories (e.g. "Weekend"). If user says "Saturday", output "Sat".
    - **Ops Relay or BCEP**: Extract "Ops Relay" or "BCEP" if mentioned.
    - **improvement (swap dd bus)**: Set to "Y" if user asks about double decker swaps.
    - **improvement (addtional trip during peak)**: Set to "Y" if user asks about peak hour trips.
    - **Date Range**: If user mentions specific dates, months, or QUARTERS (e.g. "since Jan 2024", "in 2023", "Q1 2024").
      - Convert Quarters to dates (e.g. Q1 2024 -> 2024-01-01 to 2024-03-31).
      - Format: "start_date": "YYYY-MM-DD", "end_date": "YYYY-MM-DD"
    
    Task 2: Determine Intent
    - **SPECIFIC**: User wants a LIST, COUNT, or to IDENTIFY specific services/events based on a description. (e.g. "Which services had trip cuts?", "List all changes involving double deckers", "How many improvements were made?"). -> Needs HIGH RECALL (more results).
    - **BROAD**: User wants an EXPLANATION, SUMMARY, or EXAMPLES of a topic. (e.g. "How do they improve reliability?", "What are the trends?", "Tell me about bus bunching"). -> Needs HIGH PRECISION (fewer, highly relevant results).
    
    Return JSON format:
    {{
        "filters": {{
            "Svc No": "...",
            "PTO": "...",
            "Improvement / Degrade": "...",
            "Type of Improvement / Degrade": "...",
            "BCM Package": "...",
            "Day Type": "...",
            "Ops Relay or BCEP": "...",
            "improvement (swap dd bus)": "...",
            "improvement (addtional trip during peak)": "...",
            "start_date": "YYYY-MM-DD",
            "end_date": "YYYY-MM-DD"
        }},
        "intent": "SPECIFIC" or "BROAD"
    }}
    """

    messages = [
        {"role": "system", "content": "You are a query router. Output valid JSON only."},
        {"role": "user", "content": analysis_prompt}
    ]
    
    try:
        response = llm.get_completion_by_messages(messages, model="gpt-4o-mini", temperature=0.0, max_tokens=300)
        
        # Extract JSON
        json_match = re.search(r'\{.*\}', response, re.DOTALL)
        if json_match:
            response = json_match.group()
            
        result = json.loads(response)
        filters = result.get("filters", {})
        intent = result.get("intent", "BROAD")
        
        # Clean empty filters
        filters = {k: v for k, v in filters.items() if v is not None and v != ""}
        
        # --- Normalization Logic ---
        if "Svc No" in filters:
            val = filters["Svc No"]
            if isinstance(val, list):
                filters["Svc No"] = [str(x).upper().strip() for x in val]
            else:
                filters["Svc No"] = str(val).upper().strip()

        # PTO Handling: Support both Acronyms and Full Names by expanding to all known aliases
        if "PTO" in filters:
            val = filters["PTO"]
            aliases = set()
            
            def get_pto_aliases(v):
                v_lower = str(v).lower().strip()
                if "sbs" in v_lower: return ["SBS Transit", "SBST", "SBS"]
                if "smrt" in v_lower: return ["SMRT Buses", "SMRT"]
                if "tower" in v_lower or "tts" in v_lower: return ["Tower Transit", "TTS"]
                if ("go" in v_lower and "ahead" in v_lower) or "gas" in v_lower: return ["Go-Ahead Singapore", "GAS", "Go-Ahead"]
                return [v] # Return original if no match

            if isinstance(val, list):
                for item in val:
                    aliases.update(get_pto_aliases(item))
            else:
                aliases.update(get_pto_aliases(val))
            
            filters["PTO"] = list(aliases)

        
        if "Improvement / Degrade" in filters:
            val = str(filters["Improvement / Degrade"]).lower()
            if "new" in val: filters["Improvement / Degrade"] = "New Route"
            elif "exten" in val: filters["Improvement / Degrade"] = "Route extension"
            elif "amend" in val: filters["Improvement / Degrade"] = "Route amendment"
            elif "withdraw" in val: filters["Improvement / Degrade"] = "Withdrawn"
            elif "adjust" in val: filters["Improvement / Degrade"] = "Adjustment"
            elif "offpeak" in val: filters["Improvement / Degrade"] = "Offpeak Improvement"
            elif "non-ai" in val: filters["Improvement / Degrade"] = "Non-AI Improvement"
            elif "degrad" in val: filters["Improvement / Degrade"] = "Degradation"
            elif "no change" in val: filters["Improvement / Degrade"] = "No change"
            elif "imp" in val: filters["Improvement / Degrade"] = "Improvement"
        
        # Normalize Day Type
        if "Day Type" in filters:
            val = str(filters["Day Type"]).lower()
            if "every" in val or "all day" in val or "daily" in val: filters["Day Type"] = "Daily"
            elif "work" in val or "school" in val: filters["Day Type"] = "Weekday"
            elif "public" in val or "holiday" in val: filters["Day Type"] = "PH"

        # Normalize Boolean/Flag fields
        if "improvement (swap dd bus)" in filters:
            val = str(filters["improvement (swap dd bus)"]).upper()
            if "Y" in val or "TRUE" in val: filters["improvement (swap dd bus)"] = "Y"
            
        if "improvement (addtional trip during peak)" in filters:
            val = str(filters["improvement (addtional trip during peak)"]).upper()
            if "Y" in val or "TRUE" in val: filters["improvement (addtional trip during peak)"] = "Y"
        
        # --- Date Range Logic ---
        # Convert start/end dates into ChromaDB operator syntax
        date_filter = {}
        if "start_date" in filters:
            date_filter["$gte"] = filters.pop("start_date")
        if "end_date" in filters:
            date_filter["$lte"] = filters.pop("end_date")
            
        if date_filter:
            filters["Implementation Date"] = date_filter
        
        return filters, intent
        
    except Exception as e:
        print(f"Router Error: {e}")
        return {}, "BROAD"

def process_user_message(user_message, vector_store): 
    """
    Improved RAG with Query Routing and Dynamic Retrieval
    """
    # Load acronym dictionary
    acronym_dict = load_acronym_knowledge()
    
    # Step 1: Detect acronyms
    query_acronyms = detect_acronyms_in_text(user_message, acronym_dict)
    
    # Step 2: Analyze Query (Routing + Filtering)
    with st.spinner("Routing query..."):
        key_filters, search_intent = analyze_query_requirements(user_message)
    
    # Display Routing Info
    if key_filters:
        display_parts = []
        for k, v in key_filters.items():
            if k == "Implementation Date" and isinstance(v, dict):
                dates = []
                if "$gte" in v: dates.append(f"From {v['$gte']}")
                if "$lte" in v: dates.append(f"To {v['$lte']}")
                display_parts.append(f"**Date Range:** {' '.join(dates)}")
            elif isinstance(v, list):
                display_parts.append(f"**{k}:** {', '.join(map(str, v))}")
            else:
                display_parts.append(f"**{k}:** {v}")
        
        st.info(f"**Active Filters:** {'  •  '.join(display_parts)}")
    
    # Step 3: Dynamic Retrieval Strategy
    # If filters are present, we enforce STRICT mode (no fallback to generic semantic search)
    # This ensures we only get records that actually match the user's criteria.
    allow_fallback = True
    
    if key_filters:
        top_k = 100  # Fetch all potential matches
        allow_fallback = False # Disable fallback to ensure strict filtering
        st.caption(f"⚡ Mode: **Strict Filter Search** (Retrieving matches)")
    elif search_intent == "SPECIFIC":
        top_k = 60  # Fetch more history for specific entities
        st.caption(f"⚡ Mode: **Deep Semantic Search** (High Recall - Top {top_k})")
    else:
        top_k = 20  # Standard semantic search
        st.caption(f"⚡ Mode: **Focused Semantic Search** (High Precision - Top {top_k})")

    # Step 4: Expand query
    expanded_query = expand_query_with_acronyms(user_message, query_acronyms)
    
    # Step 5: Search
    with st.spinner("🔎 Searching database..."):
        si_details = search_similar_improvements(
            expanded_query, 
            vector_store, 
            llm, 
            top_k=top_k,
            filters=key_filters if key_filters else None,
            allow_fallback=allow_fallback
        )
    
    # Step 5b: Sort results chronologically
    # This ensures that when we mask dates, DATE_1 comes before DATE_2, 
    # allowing the LLM to understand the timeline implicitly without seeing real dates.
    if si_details:
        def get_date_key(item):
            # Try sanitized key first, then original
            val = item.get('Implementation_Date', item.get('Implementation Date', ''))
            return str(val)
            
        si_details.sort(key=get_date_key)
    
    # Step 6: Detect acronyms in results
    result_acronyms = {}
    if si_details:
        result_acronyms = detect_acronyms_in_results(si_details[:10], acronym_dict)
    
    smart_acronym_context = build_smart_acronym_context(query_acronyms, result_acronyms)
    
    if query_acronyms:
        acronym_list = ", ".join(query_acronyms.keys())
        st.info(f"**Detected acronyms:** {acronym_list}")
    
    # Step 7: Process results for Context Window
    masker = DataMasker()
    
    if si_details:
        results_summary = []
        # If specific, we want to show as much history as possible within token limits
        limit = 50 if search_intent == "SPECIFIC" else 20
        
        for i, imp in enumerate(si_details[:limit]):
            # Helper to get value with fuzzy key matching (sanitized vs original)
            def get_val(keys, default=""):
                if isinstance(keys, str): keys = [keys]
                for k in keys:
                    # Try exact match
                    if k in imp: return imp[k]
                    # Try sanitized match (spaces to underscores)
                    sanitized = re.sub(r'[^a-zA-Z0-9]', '_', k).strip('_')
                    if sanitized in imp: return imp[sanitized]
                return default

            # --- DATA MASKING ---
            # 1. Mask Service Numbers (Main + Related)
            svc_raw = str(get_val(["Svc No", "Svc_No"], "Unknown")).strip()
            svc_token = masker.mask(svc_raw, "SVC")
            
            related_raw = str(get_val("related_svc", "")).strip()
            related_token = ""
            if related_raw:
                # Mask each related service individually
                masked_rels = []
                for r in related_raw.split(','):
                    r = r.strip()
                    if r: masked_rels.append(masker.mask(r, "SVC"))
                related_token = ", ".join(masked_rels)

            # 2. Mask Operator / BCM Package
            pto_token = masker.mask(get_val("PTO", ""), "PTO")
            bcm_token = masker.mask(get_val("BCM Package", ""), "BCM")
            
            # 3. Mask Date
            date_val = get_val(["Implementation Date"], "")
            # Convert timestamp back to readable date if needed
            if isinstance(date_val, (int, float)) and date_val > 0:
                try:
                    date_val = datetime.fromtimestamp(date_val).strftime('%Y-%m-%d')
                except:
                    pass
            
            date_token = masker.mask(date_val, "DATE")
            
            # 3b. Get Improvement Types (Unmasked)
            imp_type = str(get_val(["Improvement / Degrade", "Improvement_Degrade"], "Change")).strip()
            imp_category = str(get_val(["Type of Improvement / Degrade", "Type_of_Improvement_Degrade"], "General")).strip()
            
            # 4. Interpret Metrics (Natural Language for LLM Context)
            metrics_text = []
            
            def interpret_numeric(val, label):
                try:
                    num = float(val)
                    if num > 0: return f"added {int(num) if num.is_integer() else num} {label}"
                    if num < 0: return f"reduced {int(abs(num)) if num.is_integer() else abs(num)} {label}"
                except:
                    pass
                return ""

            # Trips & Buses
            trips_txt = interpret_numeric(get_val("Trip count change", 0), "trips")
            if trips_txt: metrics_text.append(trips_txt)
            
            buses_txt = interpret_numeric(get_val("Total additional buses (Exclude spares)", 0), "buses")
            if buses_txt: metrics_text.append(buses_txt)
            
            # Fleet Changes
            hicap_txt = interpret_numeric(get_val(["hi-cap bus change", "hi_cap_bus_change"], 0), "high-capacity buses")
            if hicap_txt: metrics_text.append(hicap_txt)
            
            sd_txt = interpret_numeric(get_val("SD", 0), "single-deck buses")
            if sd_txt: metrics_text.append(sd_txt)
            
            ddbd_txt = interpret_numeric(get_val(["DD/BD", "DD_BD"], 0), "double-deck/bendy buses")
            if ddbd_txt: metrics_text.append(ddbd_txt)
            
            # Flags
            swap_val = str(get_val(["improvement (swap dd bus)", "improvement_swap_dd_bus"], "")).strip().upper()
            if swap_val == 'Y': metrics_text.append("swapped in double-deck buses")
            
            peak_val = str(get_val(["improvement (addtional trip during peak)", "improvement_addtional_trip_during_peak"], "")).strip().upper()
            if peak_val == 'Y': metrics_text.append("added trips during peak hours")
            
            # 5. Mask Details (Replace Service Numbers)
            details_raw = str(get_val(["Details (free text)", "Details"], "")).strip()
            details_masked = details_raw
            
            # Replace the main service number in details
            if svc_raw and svc_raw.lower() != "unknown":
                details_masked = re.sub(r'\b' + re.escape(svc_raw) + r'\b', svc_token, details_masked, flags=re.IGNORECASE)
            
            # Replace related service numbers in details
            if related_raw:
                for r in related_raw.split(','):
                    r = r.strip()
                    if r:
                        r_token = masker.mask(r, "SVC") # Get existing token
                        details_masked = re.sub(r'\b' + re.escape(r) + r'\b', r_token, details_masked, flags=re.IGNORECASE)

            # 6. Construct the Masked Narrative
            parts = []
            parts.append(f"Service {svc_token} ({pto_token}, Package: {bcm_token}) on {date_token}")
            parts.append(f"Action: {imp_type} (Category: {imp_category})")
            
            if related_token:
                parts.append(f"Related to: {related_token}")
                
            if details_masked and details_masked.lower() != "nan":
                parts.append(f"Details: {details_masked}")
            
            if metrics_text:
                parts.append("Stats: " + ", ".join(metrics_text))
            
            sanitized_content = ". ".join(parts) + "."
            results_summary.append(f"[{i+1}] {sanitized_content}")
        
        results_text = "\n\n".join(results_summary)
        
        # Mask the query so it matches the tokens in the context
        masked_user_message = masker.apply_mask_to_query(user_message)
        
        context = f"""Found {len(si_details)} relevant records.
Here are the details (Anonymized):

{results_text}

Based on the user's query: "{masked_user_message}", analyze these results."""
        
    else:
        context = "No service improvements found matching the criteria."
        masked_user_message = user_message
    
    # Step 8: LLM Synthesis
    system_message = f"""You are an expert Bus Service Variation Analyst.
{smart_acronym_context}
**Context:**
The user is asking about bus service variations.
Search Mode: {search_intent} ({"Prioritize completeness/chronology" if search_intent == "SPECIFIC" else "Prioritize relevance/examples"})

**Data Privacy Note:**
The data provided uses anonymized tokens for sensitive fields (Service, Operator, Date), but other metrics are real.
The user's query has ALSO been masked to match these tokens (e.g. "Service 123" -> "Service <<SVC_1>>").
- **DO NOT** try to guess what the tokens mean.
- **DO NOT** complain about them.
- **USE THEM EXACTLY** as they appear in your response.
- Example: "Service <<Svc_1>> had a <<TYPE_1>> involving <<DETAILS_2>>."

**Search Results:**
{context}

**Instructions:**
1. **Direct Answer**: Start with a direct answer. ({" State find how many related results" if search_intent == "SPECIFIC" else "Provide a general summary"})
2. **Evidence**: Cite specific details using the tokens provided.
3. **Synthesis**: Summarize the events using the tokens.
4. **Day Type Logic**: If the user asks for a specific day (e.g., "Sat"), and the results show "Weekend" or "all day types", treat them as MATCHES. "Weekend" includes Saturday and Sunday. "all day types" includes all days. Do NOT say "No Saturday records found" if the day type matches like this. If the query is about holiday, all day types also matches.

Answer based ONLY on the provided results. if there is no relevant information, state that clearly."""
    
    messages = [
        {"role": "system", "content": system_message},
        {"role": "user", "content": masked_user_message}
    ]
    
    with st.spinner("💬 Analyzing results..."):
        response = llm.get_completion_by_messages(
            messages, 
            model="gpt-4o-mini", 
            temperature=0.3,
            max_tokens=2000
        )
        
    # --- DETOKENIZATION ---
    # Transform the LLM's response back to original values
    response = masker.unmask(response)
    
    # Add strict mode summary line if filters were applied
    if key_filters and si_details:
        filter_desc = ", ".join([f"{k}: {v}" for k, v in key_filters.items()])
        summary_line = f"\n\n*Found {len(si_details)} related variations based on filters: {filter_desc}.*"
        response += summary_line
    
    return response, si_details, key_filters