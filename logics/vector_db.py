import pandas as pd
import numpy as np
import re
import streamlit as st
from datetime import datetime
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
import os
from dotenv import load_dotenv

load_dotenv('.env')

@st.cache_resource
def get_embedding_model():
    """Use LangChain OpenAI embeddings with 384 dimensions"""
    return OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=384,  # ✅ Specify here for OpenAI API
        openai_api_key=os.getenv('OPENAI_API_KEY')
    )

@st.cache_resource
def get_vector_store():
    """Initialize LangChain Chroma vector store"""
    embeddings = get_embedding_model()
    
    # ✅ No need to specify dimensions here - LangChain handles it automatically
    vector_store = Chroma(
        collection_name="service_improvements",
        embedding_function=embeddings,
        persist_directory="./data/chroma_db"
    )
    
    return vector_store

def sanitize_field_name(field_name):
    """Clean field names to be ChromaDB-compatible"""
    # Replace non-alphanumeric characters with underscores
    clean_name = re.sub(r'[^a-zA-Z0-9]', '_', str(field_name))
    # Remove leading/trailing underscores and multiple underscores
    clean_name = re.sub(r'_+', '_', clean_name).strip('_')
    return clean_name

def sanitize_metadata(metadata):
    """Convert all metadata values to LangChain-compatible types"""
    sanitized = {}
    for key, value in metadata.items():
        clean_key = sanitize_field_name(key)
        
        if isinstance(value, (np.integer, np.int64, np.int32)):
            sanitized[clean_key] = int(value)
        elif isinstance(value, (np.floating, np.float64, np.float32)):
            if pd.isna(value):
                sanitized[clean_key] = ""
            else:
                sanitized[clean_key] = float(value)
        elif isinstance(value, (np.bool_, bool)):
            sanitized[clean_key] = bool(value)
        elif pd.isna(value) or value is None or str(value).lower() == 'nan':
            sanitized[clean_key] = ""
        elif isinstance(value, (pd.Timestamp, datetime)):
            # Convert to Unix timestamp (integer) for robust filtering
            sanitized[clean_key] = int(value.timestamp())
        else:
            sanitized[clean_key] = str(value).strip()
    
    return sanitized

def initialize_vector_db():
    """Initialize and return the vector database collection"""
    try:
        # embedding model that we will use for the session
        embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
        
        # llm to be used in RAG pipelines
        llm = ChatOpenAI(model='gpt-4o-mini', temperature=0, seed=42)
        
        # Updated vector store initialization
        db_path = "./data/chroma_db"
        
        # Create vector store with explicit embedding function
        vector_store = Chroma(
            persist_directory=db_path,
            embedding_function=embeddings_model,
            collection_name="service_improvements"  # Add explicit collection name
        )
        
        print(f"✓ Vector store initialized successfully")
        print(f"✓ Database path: {db_path}")
        
        return vector_store
        
    except Exception as e:
        print(f"❌ Error initializing vector store: {e}")
        raise e

# --- DATA MASKING LOGIC (Mirrors query_handler.py) ---
class DataMasker:
    """
    Handles tokenization of sensitive data.
    Used here to anonymize text before embedding, ensuring privacy.
    """
    def __init__(self):
        self.token_map = {} 
        self.value_map = {} 
        self.counters = {} 

    def mask(self, text, prefix="DATA"):
        if not text or str(text).lower() == 'nan' or str(text).strip() == "":
            return text
            
        val_str = str(text).strip()
        cache_key = (prefix, val_str)
        
        if cache_key in self.value_map:
            return self.value_map[cache_key]
            
        if prefix not in self.counters:
            self.counters[prefix] = 0
        self.counters[prefix] += 1
        
        token = f"<<{prefix}_{self.counters[prefix]}>>"
        
        self.token_map[token] = val_str
        self.value_map[cache_key] = token
        return token

def create_semantic_document(row):
    """
    Convert Excel row to semantic-rich text (Mini-Story format).
    
    PRIVACY UPDATE: 
    We use the DataMasker to replace real values with tokens (<<SVC_1>>, <<DATE_1>>).
    This preserves the grammatical structure for the Embedding Model (better accuracy)
    while ensuring NO raw data is sent to the AI Provider.
    """
    # Instantiate a fresh masker for each row.
    # This ensures every row uses standard tokens (e.g. the main service is always <<SVC_1>>).
    # This consistency helps the AI model recognize patterns better.
    masker = DataMasker()

    # 1. Categories (Safe to embed)
    imp_type = str(row.get('Improvement / Degrade', 'Change')).strip()
    category = str(row.get('Type of Improvement / Degrade', 'General')).strip()
    
    # 2. Mask Sensitive Identifiers
    # We mask them so the AI sees "Service <<SVC_1>>" instead of "Service 123"
    svc_token = masker.mask(str(row.get('Svc No', 'Unknown')), "SVC")
    pto_token = masker.mask(str(row.get('PTO', 'Unknown')), "PTO")
    bcm_token = masker.mask(str(row.get('BCM Package', 'Unknown')), "BCM")
    date_token = masker.mask(str(row.get('Implementation Date', 'Unknown')), "DATE")
    
    # 3. Interpret Metrics (Natural Language for Embedding)
    # We convert raw numbers/flags into semantic text so the embedding model understands "increase" vs "decrease"
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
    trips_txt = interpret_numeric(row.get('Trip count change', 0), "trips")
    if trips_txt: metrics_text.append(trips_txt)
    
    buses_txt = interpret_numeric(row.get('Total additional buses (Exclude spares)', 0), "buses")
    if buses_txt: metrics_text.append(buses_txt)
    
    # Fleet Changes
    hicap_txt = interpret_numeric(row.get('hi-cap bus change', 0), "high-capacity buses")
    if hicap_txt: metrics_text.append(hicap_txt)
    
    sd_txt = interpret_numeric(row.get('SD', 0), "single-deck buses")
    if sd_txt: metrics_text.append(sd_txt)
    
    ddbd_txt = interpret_numeric(row.get('DD/BD', 0), "double-deck/bendy buses")
    if ddbd_txt: metrics_text.append(ddbd_txt)
    
    # Flags
    swap_val = str(row.get('improvement (swap dd bus)', '')).strip().upper()
    if swap_val == 'Y': metrics_text.append("swapped in double-deck buses")
    
    peak_val = str(row.get('improvement (addtional trip during peak)', '')).strip().upper()
    if peak_val == 'Y': metrics_text.append("added trips during peak hours")

    # 4. Details (Sanitized with Tokens)
    details = str(row.get('Details (free text)', 'No details provided')).strip()
    if details.lower() == 'nan': details = "No details provided"
    
    # Replace the main service number in details with its token
    svc_raw = str(row.get('Svc No', '')).strip()
    if svc_raw:
        details = re.sub(r'\b' + re.escape(svc_raw) + r'\b', svc_token, details, flags=re.IGNORECASE)
    
    # Replace other service numbers (Related Services) with generic tokens
    def replace_related(match):
        return f"Service {masker.mask(match.group(1), 'SVC')}"
        
    details_safe = re.sub(r'(?:Svc|Service)\s+([0-9]+[A-Za-z]?)', replace_related, details, flags=re.IGNORECASE)

    # 5. Construct Anonymized Narrative
    # The AI sees: "Service <<SVC_1>> operated by <<PTO_1>> (<<BCM_1>>)..."
    content = (
        f"Service Improvement Record for Service {svc_token}. "
        f"In {date_token}, Service {svc_token} operated by {pto_token} (Package: {bcm_token}) "
        f"underwent a {imp_type} categorized as '{category}'. "
        f"Details: {details_safe}. "
    )
    
    if metrics_text:
        content += " Changes involved: " + ", ".join(metrics_text) + "."
    
    return content

def add_improvements_to_db(df, vector_store):
    """Add service improvements using LangChain"""
    documents = []
    
    print("Converting Excel rows to LangChain Documents...")
    for idx, row in df.iterrows():
        try:
            # Normalize critical fields for better filtering
            if 'Svc No' in row and pd.notna(row['Svc No']):
                row['Svc No'] = str(row['Svc No']).strip().upper()
            
            # PTO Normalization removed to preserve acronyms (SBST, SMRT, etc.)
            # if 'PTO' in row and pd.notna(row['PTO']):
            #     pto = str(row['PTO']).lower()
            #     if 'sbs' in pto: row['PTO'] = 'SBS Transit'
            #     elif 'smrt' in pto: row['PTO'] = 'SMRT Buses'
            #     elif 'tower' in pto: row['PTO'] = 'Tower Transit'
            #     elif 'go' in pto and 'ahead' in pto: row['PTO'] = 'Go-Ahead Singapore'

            doc_text = create_semantic_document(row)
            
            # Build metadata
            metadata = {}
            for col in df.columns:
                metadata[col] = row[col]
            
            # Extract related services for metadata indexing
            details = str(row.get('Details (free text)', '')).strip()
            related_svcs = re.findall(r'(?:Svc|Service)\s+([0-9]+[A-Za-z]?)', details, re.IGNORECASE)
            if related_svcs:
                # Store as a comma-separated string for simple retrieval checks
                # (Note: Chroma filtering on this is limited, but useful for display/logic)
                clean_related = [s.upper() for s in related_svcs if s.upper() != str(row.get('Svc No', '')).upper()]
                if clean_related:
                    metadata['related_svc'] = ", ".join(sorted(list(set(clean_related))))

            # Sanitize metadata
            sanitized = sanitize_metadata(metadata)
            sanitized['row_id'] = f"si_{idx}"
            
            # Create LangChain Document
            doc = Document(
                page_content=doc_text,
                metadata=sanitized
            )
            documents.append(doc)
            
        except Exception as e:
            print(f"⚠️ Warning: Skipping row {idx} due to error: {e}")
            continue
    
    if not documents:
        print("❌ No valid documents to add!")
        return
    
    print(f"\nAdding {len(documents)} documents to vector store...")
    print("(LangChain will handle embeddings automatically)")
    
    # Add in batches
    batch_size = 50
    total_batches = (len(documents) + batch_size - 1) // batch_size
    
    added_count = 0
    for i in range(0, len(documents), batch_size):
        batch_num = i // batch_size + 1
        batch_end = min(i + batch_size, len(documents))
        
        print(f"  Batch {batch_num}/{total_batches} ({i+1}-{batch_end})...", end=" ", flush=True)
        
        try:
            vector_store.add_documents(documents[i:batch_end])
            added_count += (batch_end - i)
            print(f"✓")
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
    
    print(f"\n✅ Complete! Added {added_count}/{len(documents)} documents")

def build_chroma_filters(filters_dict):
    """Helper to build ChromaDB compatible filters"""
    if not filters_dict:
        return None
        
    filter_conditions = []
    for k, v in filters_dict.items():
        if v:
            clean_key = sanitize_field_name(k)
            
            # Convert date strings to timestamps if this is a date field
            # We assume keys containing 'Date' are date fields
            if "Date" in k or "date" in k:
                def convert_val(val):
                    try:
                        # Try YYYY-MM-DD
                        dt = datetime.strptime(str(val), "%Y-%m-%d")
                        return int(dt.timestamp())
                    except:
                        return val
                
                if isinstance(v, dict):
                    new_v = {}
                    for op, val in v.items():
                        new_v[op] = convert_val(val)
                    v = new_v
                elif isinstance(v, list):
                    v = [convert_val(x) for x in v]
                else:
                    v = convert_val(v)

            if isinstance(v, list):
                filter_conditions.append({clean_key: {"$in": v}})
            elif isinstance(v, dict):
                # Handle multiple operators (e.g. date range)
                # ChromaDB requires explicit $and for multiple operators on same field
                # e.g. {"$and": [{"date": {"$gte": "..."}}, {"date": {"$lte": "..."}}]}
                if len(v) > 1:
                    for op, val in v.items():
                        filter_conditions.append({clean_key: {op: val}})
                else:
                    # Single operator
                    filter_conditions.append({clean_key: v})
            else:
                val = v if isinstance(v, (int, float)) else str(v).strip()
                filter_conditions.append({clean_key: val})
    
    if not filter_conditions:
        return None
        
    if len(filter_conditions) > 1:
        return {"$and": filter_conditions}
    else:
        return filter_conditions[0]

def get_covered_days(day_str):
    """Parse a day string into a set of covered days (mon, tue, etc.)"""
    s = str(day_str).lower()
    days = set()
    
    # Universal
    if "daily" in s or "all day" in s:
        return {"mon", "tue", "wed", "thu", "fri", "sat", "sun", "ph"}
    
    # Categories
    if "weekday" in s:
        days.update(["mon", "tue", "wed", "thu", "fri"])
    if "weekend" in s:
        days.update(["sat", "sun"])
        
    # Specifics (3-letter codes cover full names too)
    if "mon" in s: days.add("mon")
    if "tue" in s: days.add("tue")
    if "wed" in s: days.add("wed")
    if "thu" in s: days.add("thu")
    if "fri" in s: days.add("fri")
    if "sat" in s: days.add("sat")
    if "sun" in s: days.add("sun")
    if "ph" in s or "public" in s: days.add("ph")
    
    return days

def search_similar_improvements(query, vector_store, llm_model, top_k=30, filters=None, allow_fallback=True):
    """Search using LangChain similarity search with robust filtering"""
    try:
        print(f"🔍 Searching for: '{query}'")
        
        # Handle "Day Type" specially (Partial Match / Contains logic)
        # ChromaDB metadata filters are exact match only, but "Day Type" data might be "Weekday / Sat"
        # So we remove it from DB filters and apply it in Python post-processing
        day_type_filter = None
        chroma_filters = filters.copy() if filters else {}
        
        # Check for Day Type keys (normalized or original)
        for k in list(chroma_filters.keys()):
            # Make check case-insensitive to catch 'day type', 'Day Type', 'DAY_TYPE' etc.
            if sanitize_field_name(k).lower() == 'day_type':
                day_type_filter = chroma_filters.pop(k)
                print(f"ℹ️ Handling Day Type filter in Python: '{day_type_filter}'")

        # Prepare filters if any
        formatted_filters = build_chroma_filters(chroma_filters)
        if formatted_filters:
            print(f"🔧 Filters: {formatted_filters}")

        # OPTIMIZATION: Deterministic Search for Strict Mode
        # If we have filters (including Day Type) and fallback is disabled, use direct DB fetch instead of vector search
        # This ensures we get ALL matching records regardless of semantic similarity
        if (formatted_filters or day_type_filter) and not allow_fallback:
            try:
                print("⚡ Mode: Deterministic Metadata Fetch (Skipping Vector Search)")
                # Access raw ChromaDB collection
                collection = vector_store._collection
                
                # 1. Primary Fetch: Strict match on all filters
                # If formatted_filters is None (only Day Type filter exists), fetch all (up to limit)
                # Note: We must pass None, not {}, if we want no filters.
                
                # INCREASED LIMIT: When doing post-processing (Day Type) or strict filtering,
                # we need to fetch enough candidates to ensure we don't miss matches due to pagination.
                # 5000 should cover most datasets.
                response = collection.get(
                    where=formatted_filters,
                    limit=5000,
                    include=["metadatas", "documents"]
                )
                
                results = []
                metadatas = response.get("metadatas", [])
                documents = response.get("documents", [])
                
                if metadatas and documents:
                    for i in range(len(metadatas)):
                        # Combine metadata and content
                        item = metadatas[i].copy()
                        item['_content'] = documents[i]
                        results.append(item)
                
                # 2. Secondary Fetch: Check for "Related Services" (Swaps/Interactions)
                # If 'Svc No' is in filters, we also want to find rows where this service is mentioned in 'related_svc'
                # even if the main 'Svc No' is different.
                
                if chroma_filters and 'Svc No' in chroma_filters:
                    target_svcs = set()
                    svc_val = chroma_filters['Svc No']
                    if isinstance(svc_val, list):
                        target_svcs.update([str(s).upper() for s in svc_val])
                    else:
                        target_svcs.add(str(svc_val).upper())
                    
                    # Construct filters for the secondary query (exclude Svc No)
                    sec_filters_dict = {k: v for k, v in chroma_filters.items() if k != 'Svc No'}
                    secondary_where = build_chroma_filters(sec_filters_dict)
                    
                    # Fetch potential related rows (must have interaction text)
                    # We use where_document to narrow down to rows with interactions
                    print(f"  ↳ Checking for related services: {target_svcs}")
                    
                    related_response = collection.get(
                        where=secondary_where,
                        where_document={"$contains": "interaction with Service(s):"},
                        limit=5000,
                        include=["metadatas", "documents"]
                    )
                    
                    rel_metadatas = related_response.get("metadatas", [])
                    rel_documents = related_response.get("documents", [])
                    
                    if rel_metadatas:
                        for i in range(len(rel_metadatas)):
                            meta = rel_metadatas[i]
                            # Check if any target service is in the 'related_svc' field
                            # related_svc is comma-separated string e.g. "154, 174"
                            row_related = str(meta.get('related_svc', '')).upper().split(', ')
                            # Check for exact match in the list
                            if any(t in row_related for t in target_svcs):
                                # Avoid duplicates
                                if meta.get('row_id') not in [r.get('row_id') for r in results]:
                                    item = meta.copy()
                                    item['_content'] = rel_documents[i]
                                    results.append(item)
                                    print(f"    + Found related record: {meta.get('row_id')}")

                # --- POST-PROCESSING: Day Type Filter ---
                if day_type_filter:
                    filtered_results = []
                    # Parse the query filter into a set of target days
                    target_days = get_covered_days(day_type_filter)
                    print(f"  ↳ Applying Day Type filter: '{day_type_filter}' -> {target_days}")
                    
                    for res in results:
                        # Get the day type from result metadata
                        val = str(res.get('Day Type', res.get('Day_Type', ''))).lower()
                        res_days = get_covered_days(val)
                        
                        # Check for intersection (if they share ANY day)
                        if not target_days.isdisjoint(res_days):
                            filtered_results.append(res)
                            
                    print(f"    - Filtered from {len(results)} to {len(filtered_results)} results")
                    results = filtered_results

                # Slice to requested top_k after all filtering is done
                results = results[:top_k]
                print(f"✅ Found {len(results)} deterministic matches (including related)")
                return results
                
            except Exception as e:
                print(f"⚠️ Deterministic fetch failed: {e}, falling back to vector search")

        # Execute search
        results = []
        if formatted_filters:
            try:
                # Try strict filtering first
                results = vector_store.similarity_search(query, k=top_k, filter=formatted_filters)
                print(f"✅ Found {len(results)} results with filters")
            except Exception as e:
                print(f"⚠️ Filter search failed: {e}")
        
        # Fallback or supplement if few results AND fallback is allowed
        if allow_fallback and len(results) < 5:
            print("🔄 Supplementing with semantic search...")
            fallback_results = vector_store.similarity_search(query, k=top_k)
            
            # Deduplicate
            seen_ids = {doc.metadata.get('row_id') for doc in results}
            for doc in fallback_results:
                if doc.metadata.get('row_id') not in seen_ids:
                    results.append(doc)
                    seen_ids.add(doc.metadata.get('row_id'))
            
            results = results[:top_k]

        # Format output
        final_results = [{**doc.metadata, '_content': doc.page_content} for doc in results]
        
        # Apply Day Type filter to vector search results too if needed
        if day_type_filter and (allow_fallback or not formatted_filters):
             filtered_final = []
             target_days = get_covered_days(day_type_filter)
             for res in final_results:
                 val = str(res.get('Day Type', res.get('Day_Type', ''))).lower()
                 res_days = get_covered_days(val)
                 if not target_days.isdisjoint(res_days):
                     filtered_final.append(res)
             final_results = filtered_final

        return final_results
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return []

def get_collection_stats():
    """Get statistics about the collection"""
    try:
        embeddings_model = OpenAIEmbeddings(model='text-embedding-3-small')
        
        vector_store = Chroma(
            persist_directory="./data/chroma_db",
            embedding_function=embeddings_model,
            collection_name="service_improvements"
        )
        
        # Get collection info
        collection = vector_store._collection
        count = collection.count()
        
        return {
            'total_documents': count,
            'embedding_model': 'text-embedding-3-small',
            'collection_name': 'service_improvements'
        }
    except Exception as e:
        print(f"❌ Error getting collection stats: {e}")
        return None