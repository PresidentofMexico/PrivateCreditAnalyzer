# --- 1. SETUP & IMPORTS ---
import streamlit as st
import os
import shutil
import time
import warnings
import uuid

# CRITICAL FIX: Disable ChromaDB Telemetry
os.environ["ANONYMIZED_TELEMETRY"] = "False"

# CRITICAL FIX: Swap SQLite for Streamlit Cloud Compatibility
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

# Silence Warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# CONFIG
st.set_page_config(page_title="Eldridge CLO Stip Analyzer", layout="wide", page_icon="🛡️")

# --- HELPER: GET API KEY ---
def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    elif os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    return None

# --- 2. CSS STYLING ---
st.markdown("""
    <style>
    .stApp { background-color: #0e1117; color: #FAFAFA; }
    .stButton>button {
        width: 100%; border-radius: 4px; height: 3.5em;
        background-color: #1f2937; color: white; border: 1px solid #374151;
    }
    .stButton>button:hover { background-color: #374151; border-color: #6b7280; }
    .stTextInput>div>div>input { background-color: #1f2937; color: white; border: 1px solid #374151; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKEND LOGIC ---
def get_session_db_path():
    """
    Generates or retrieves a unique DB path for this specific session.
    This prevents 'Tenant' errors by ensuring every upload gets a fresh folder.
    """
    if 'db_path' not in st.session_state:
        # Create a unique folder name
        st.session_state['db_path'] = f"./chroma_db_{uuid.uuid4().hex}"
    return st.session_state['db_path']

def process_document(uploaded_file):
    api_key = get_api_key()
    if not api_key:
        st.error("❌ API Key missing.")
        return False

    # 1. Save temp PDF
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"📖 Reading {uploaded_file.name}...")
    
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # 2. Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # 3. Clean up OLD DB if it exists (To save space)
        # We perform a "Soft Reset" by generating a BRAND NEW path
        if 'db_path' in st.session_state:
            old_path = st.session_state['db_path']
            if os.path.exists(old_path):
                try:
                    shutil.rmtree(old_path)
                except:
                    pass # If locked, we just ignore it and move to a new folder
        
        # Generate FRESH path for this new document
        current_db_path = f"./chroma_db_{uuid.uuid4().hex}"
        st.session_state['db_path'] = current_db_path
        
        # 4. Embed (Batched)
        batch_size = 50
        progress_bar = st.progress(0)
        
        # Init DB
        vectorstore = Chroma.from_documents(
            documents=splits[:batch_size], 
            embedding=OpenAIEmbeddings(api_key=api_key), 
            persist_directory=current_db_path
        )
        
        total_batches = (len(splits) // batch_size) + 1
        for i in range(batch_size, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            vectorstore.add_documents(batch)
            progress = min((i / len(splits)), 1.0)
            progress_bar.progress(progress, text=f"Embedding batch {i // batch_size}/{total_batches}")
            time.sleep(0.1)
            
        progress_bar.empty()
        os.remove(temp_path)
        return True
        
    except Exception as e:
        st.error(f"Error during ingestion: {e}")
        if os.path.exists(temp_path):
            os.remove(temp_path)
        return False

def run_query(question):
    api_key = get_api_key()
    if not api_key:
        return "Error: API Key missing."
    
    # Check if we have a DB path
    if 'db_path' not in st.session_state:
        return "⚠️ No document loaded. Please upload one first."
    
    current_db_path = st.session_state['db_path']

    try:
        # Re-initialize vector store pointing to the SESSION SPECIFIC path
        vectorstore = Chroma(persist_directory=current_db_path, embedding_function=OpenAIEmbeddings(api_key=api_key))
        
        if vectorstore._collection.count() == 0:
            return "⚠️ Database is empty."

        retriever = vectorstore.as_retriever()
        
        template = """You are a senior private credit analyst. 
        Use the following pieces of context from the loan agreement to answer the question.
        If you don't know the answer, just say that you don't know, don't make up terms.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=api_key)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(question)
    except Exception as e:
        return f"Error processing query: {e}"

# --- 4. FRONTEND UI ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2704/2704022.png", width=50)
    st.title("Deal Room")
    
    if get_api_key():
        st.caption("✅ API Key Active")
    else:
        st.error("❌ API Key Missing")

    uploaded_file = st.file_uploader("Upload Indenture (PDF)", type=['pdf'])
    
    if uploaded_file and st.button("🚀 Ingest Document"):
        with st.spinner("Processing..."):
            success = process_document(uploaded_file)
            if success:
                st.success("Ingestion Complete!")
                st.session_state['doc_ready'] = True
    
    # Simple Reset
    if st.button("⚠️ Clear Memory"):
        st.session_state['doc_ready'] = False
        if 'db_path' in st.session_state:
            del st.session_state['db_path']
        st.warning("Memory Cleared.")

    # Eldridge Stips Cheat Sheet
    with st.expander("📋 Eldridge Stips Checklist", expanded=True):
        st.markdown("""
        **1. Concentration Limits**
        - Caa/CCC Limit: **7.5%**
        - Top 5 Obligors: **2.5%** (1.5% non-senior)
        - Cov-lite: **60%**
        - Small Obligors ($150-250M): **5%**
        - Long Dated: **0%**
        - Bridge Loans: **2.5%**
        - Fixed Rate: **5%**
        - Senior Secured: **>90%**
        - DIP: **7.5%**
        - Industry Cap: **10%** (Exceptions: 2x12%, 1x15%)
        
        **2. Reinvestment**
        - Post-Reinv Maturity: **<= Sold Asset**
        - O/C Test: **Must Satisfy**
        - Proceeds: Reinvest w/in 45 days or 2nd determination date
        
        **3. Definitions**
        - **CCC Excess:** NO carveouts
        - **Discount Obligation:** NO carveouts
        - **Small Obligor:** Min $150M Indebtedness
        
        **4. Other Req.**
        - Distressed Exchange: **5% (20% cum)**
        - FLLO = **Second Lien**
        - Min Price: **50%** (5% allow for 50-60%)
        - Trading Plan: **5%** (No Credit Risk carveout)
        
        **5. Workouts**
        - Sale Proceeds -> Principal (Cap at default bal)
        - Interest use strictly limited
        """)

st.title("🛡️ CLO Indenture vs. Stip Analyzer")

if 'doc_ready' not in st.session_state:
    st.info("👈 Please upload a document to begin.")
else:
    # Row 1: Concentration
    st.subheader("📊 1. Concentration Checks")
    c1, c2, c3 = st.columns(3)
    with c1:
        if st.button("Check Caa/CCC & Top 5"):
            with st.spinner("Checking Caa/CCC and Top 5 limits..."):
                st.success(run_query("Does the indenture limit Moody’s Caa and S&P CCC obligations to 7.5%? Does it limit Top 5 Obligors to 2.5%?"))
    with c2:
        if st.button("Check Cov-Lite & Long Dated"):
            with st.spinner("Checking Cov-Lite and Long Dated..."):
                st.success(run_query("Is there a 60% limit for Cov-lite loans? Is there a 0% limit for Long Dated Obligations?"))
    with c3:
        if st.button("Check Industry Caps"):
            with st.spinner("Checking Industry Caps..."):
                st.success(run_query("Verify Industry Concentration limits. Is it 10% standard? Are exceptions 12% (up to 2) and 15% (up to 1)?"))

    # Row 2: Definitions & Reinvestment
    st.subheader("⚖️ 2. Definitions & Reinvestment")
    d1, d2, d3 = st.columns(3)
    with d1:
        if st.button("Check CCC/Discount Defs"):
            with st.spinner("Checking Definitions..."):
                st.success(run_query("Are there any carveouts within the definition of CCC Excess or Discount Obligations? (Stip requires NO carveouts)."))
    with d2:
        if st.button("Check Post-Reinv Maturity"):
            with st.spinner("Checking Maturity Rules..."):
                st.success(run_query("Does the indenture require post-reinvestment purchases to have a maturity equal to or shorter than the prepaid/sold obligation?"))
    with d3:
        if st.button("Check Small Obligors"):
             with st.spinner("Checking Small Obligor limits..."):
                st.success(run_query("What is the minimum total indebtedness for Small Obligors? (Expect $150M). Is there a 5% concentration limit for them?"))

    # Row 3: Other Requirements
    st.subheader("🚨 3. Other Requirements")
    o1, o2, o3 = st.columns(3)
    with o1:
        if st.button("Check Distressed Exchange"):
            with st.spinner("Checking Distressed Exchange..."):
                st.success(run_query("What are the limits for Distressed Exchanges? (Expect 5% point-in-time, 20% cumulative)."))
    with o2:
        if st.button("Check Trading Plan"):
            with st.spinner("Checking Trading Plan..."):
                st.success(run_query("What is the Trading Plan allowance? (Expect 5%). Does it have a carveout for Credit Risk sales? (Expect NO)."))
    with o3:
        if st.button("Check Workouts"):
            with st.spinner("Checking Workout treatment..."):
                st.success(run_query("How are sale proceeds from Workout Assets treated? Must they be counted as principal up to the defaulted balance?"))

    st.divider()
    user_input = st.chat_input("Ask a custom question about the indenture...")
    if user_input:
        with st.chat_message("assistant"):
            st.write(run_query(user_input))
