import shutil
import pandas as pd
import numpy as np
import sys
from datetime import datetime
import os
from dotenv import load_dotenv
from logics.vector_db import add_improvements_to_db, initialize_vector_db

load_dotenv('.env')

# Set this BEFORE importing Chroma to avoid loading default embedding function
os.environ["TOKENIZERS_PARALLELISM"] = "false"

db_path = "./data/chroma_db"

# Delete existing database
print("\nDeleting existing database...")

if os.path.exists(db_path):
    if '-y' in sys.argv or '--yes' in sys.argv:
        confirm = 'yes'
    else:
        confirm = input("⚠️ Found existing database. Delete and rebuild? (yes/no): ")
    
    if confirm.lower() != 'yes':
        print("❌ Cancelled")
        exit()

    try:
        shutil.rmtree(db_path)
        print("✓ Deleted old database")
    except Exception as e:
        print(f"⚠️ Warning: Could not delete: {e}")
        print("  Trying to continue anyway...")
else:
    print("✓ No existing database found")


# Create new vector store
print("\nCreating new vector store...")
try:
    # Use the centralized initialization function
    vector_store = initialize_vector_db()
    print("✓ Created (using LangChain + OpenAI embeddings)")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit()

# Load data
print("\nLoading data from Excel...")
try:
    df = pd.read_excel('./data/data.xlsx', sheet_name=0)
    
    # Pre-process dataframe to ensure compatibility
    if "Svc No" in df.columns:
        df["Svc No"] = df["Svc No"].astype("string").fillna("").str.strip()
        
    # Ensure Implementation Date is parsed as datetime for correct filtering
    if "Implementation Date" in df.columns:
        print("  - Parsing 'Implementation Date' column...")
        df["Implementation Date"] = pd.to_datetime(df["Implementation Date"], dayfirst=True, errors='coerce')
        
    print(f"✓ Loaded {len(df)} rows")
except Exception as e:
    print(f"❌ Error: {e}")
    exit()

# Add data
print("\nAdding data to vector store...")
print("(This will call OpenAI API for embeddings)")
try:
    start = datetime.now()
    add_improvements_to_db(df, vector_store)
    duration = (datetime.now() - start).total_seconds()
    print(f"✓ Time: {duration:.2f}s")
except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()
    exit()

    # Verify
print("\nVerifying...")
try:
    # Access the underlying collection to get count
    # Note: Depending on LangChain version, this might be accessed differently
    if hasattr(vector_store, '_collection'):
        count = vector_store._collection.count()
        print(f"✓ Database has {count} documents")
    
    # Test a simple query
    print("\nTesting a sample query...")
    results = vector_store.similarity_search("bus service 123", k=1)
    if results:
        print("✓ Query test successful!")
    else:
        print("⚠️ Query returned no results (this may be normal if no matching data)")
            
except Exception as e:
    print(f"⚠️ Could not verify: {e}")
    
print("\n" + "=" * 60)
print("DATABASE INITIALIZATION COMPLETE!")
print("=" * 60)
print("\nSummary:")
print("   • Embedding model: OpenAI text-embedding-3-small")
print("   • Vector store: LangChain + ChromaDB")
print("   • Location: ./data/chroma_db")
print("\nYou can now run: streamlit run Chatbot.py")
exit()