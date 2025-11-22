import streamlit as st
import pandas as pd
from logics.vector_db import initialize_vector_db, get_embedding_model, add_improvements_to_db, get_collection_stats
from helper_functions.utility import check_password_admin
import io

# Password check
if not check_password_admin():
    st.stop()

st.title("Upload New Data")
st.markdown("Upload an Excel file to add new service variation records to the database.")

# --- 1. Download Template Section ---
with st.expander("Need a template?"):
    st.write("Download this template to ensure your data is in the correct format.")
    
    # Create a sample template dataframe
    template_cols = [
        "Qtr of improvement", "BCM Package", "Svc No", "PTO", "Implementation Date",
        "Improvement / Degrade", "Improvement (Swap DD Bus)", 
        "Improvement (Additional Peak Period Trips)", "Day Type",
        "Type of Improvement / Degrade", "Trip count change", "Hi-cap bus change",
        "Ops Relay or BCEP", "SD", "DD/BD", "Total additional buses (Exclude spares)",
        "Details (free text)"
    ]
    df_template = pd.DataFrame(columns=template_cols)
    
    # Convert to Excel in memory
    buffer = io.BytesIO()
    with pd.ExcelWriter(buffer, engine='xlsxwriter') as writer:
        df_template.to_excel(writer, index=False, sheet_name='Sheet1')
    
    st.download_button(
        label="⬇️ Download Excel Template",
        data=buffer.getvalue(),
        file_name="service_improvement_template.xlsx",
        mime="application/vnd.ms-excel"
    )

# --- 2. File Upload Section ---
uploaded_file = st.file_uploader("Choose an Excel file", type=['xlsx', 'xls'])

if uploaded_file:
    try:
        df_new = pd.read_excel(uploaded_file)
        
        # Basic Validation
        required_cols = ["Svc No", "Implementation Date", "Details (free text)", "Improvement / Degrade"]
        missing_cols = [col for col in required_cols if col not in df_new.columns]
        
        if missing_cols:
            st.error(f"❌ Missing required columns: {', '.join(missing_cols)}")
        else:
            st.success("✅ File format looks good!")
            
            # Ensure Svc No is string to avoid PyArrow errors
            if "Svc No" in df_new.columns:
                df_new["Svc No"] = df_new["Svc No"].astype(str).str.strip()

            # --- Duplicate Check ---
            try:
                existing_df = pd.read_excel('./data/data.xlsx')
                if "Svc No" in existing_df.columns:
                    existing_df["Svc No"] = existing_df["Svc No"].astype(str).str.strip()
                
                # Define columns to check for duplicates
                check_cols = ["Svc No", "Implementation Date", "Details (free text)"]
                available_check_cols = [c for c in check_cols if c in df_new.columns and c in existing_df.columns]
                
                if available_check_cols:
                    # Create signatures for comparison
                    def create_signature(df, cols):
                        return df[cols].astype(str).apply(lambda x: x.str.strip().str.lower()).apply(lambda x: '_'.join(x), axis=1)

                    existing_sigs = set(create_signature(existing_df, available_check_cols))
                    new_sigs = create_signature(df_new, available_check_cols)
                    
                    is_duplicate = new_sigs.isin(existing_sigs)
                    num_duplicates = is_duplicate.sum()
                    
                    if num_duplicates > 0:
                        st.warning(f"⚠️ Found {num_duplicates} duplicate rows that already exist in the database.")
                        df_upload = df_new[~is_duplicate]
                    else:
                        st.info("✅ No duplicates found.")
                        df_upload = df_new
                else:
                    df_upload = df_new
            except Exception as e:
                st.warning(f"Could not check for duplicates (Database might be empty or inaccessible): {e}")
                df_upload = df_new

            # Preview
            st.subheader("Preview (New Records Only)")
            if len(df_upload) > 0:
                st.dataframe(df_upload.head(3), width='stretch')
                st.caption(f"Ready to upload: {len(df_upload)} rows")

                # Upload Button
                if st.button("Process & Upload to Database", type="primary", width='stretch'):
                    with st.status("Processing...", expanded=True) as status:
                        
                        # 1. Prepare Data
                        st.write("Preparing data...")
                        # (Svc No is already string/stripped)
                        
                        # 2. Update Vector DB
                        st.write("Updating AI Knowledge Base...")
                        vector_store = initialize_vector_db()
                        add_improvements_to_db(df_upload, vector_store)
                        
                        # 3. Update Excel File
                        st.write("Saving to Master Record...")
                        try:
                            current_db = pd.read_excel('./data/data.xlsx')
                            combined_df = pd.concat([current_db, df_upload], ignore_index=True)
                            combined_df.to_excel('./data/data.xlsx', index=False)
                        except Exception as e:
                            st.warning(f"Could not update local Excel file: {e}")

                        # 4. Clear Cache
                        st.cache_data.clear()
                        st.cache_resource.clear()
                        
                        status.update(label="✅ Upload Complete!", state="complete", expanded=False)
                    
                    st.success(f"Successfully added {len(df_upload)} records!")
                    st.button("Refresh Page")
            else:
                st.warning("⚠️ No new records to upload (all rows are duplicates).")

    except Exception as e:
        st.error(f"Error reading file: {e}")
