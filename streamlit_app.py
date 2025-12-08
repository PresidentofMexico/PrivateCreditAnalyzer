# --- 1. SETUP & IMPORTS ---
import streamlit as st
import os
import shutil
import time
import warnings
import uuid
import pandas as pd

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

# --- HARDCODED ELDRIDGE STIPS ---
ELDRIDGE_STIPS = [
    {
        "Category": "Concentration",
        "Rule": "Moody’s Caa / S&P CCC Limit",
        "Threshold": "Max 7.5% (Check for Excess Caa/CCC definitions)"
    },
    {
        "Category": "Concentration",
        "Rule": "Top 5 Obligors",
        "Threshold": "Max 2.5% each (1.5% if non-senior secured)"
    },
    {
        "Category": "Concentration",
        "Rule": "Cov-Lite Loans",
        "Threshold": "Max 60%"
    },
    {
        "Category": "Concentration",
        "Rule": "Long Dated Obligations",
        "Threshold": "0% allowed (Strict prohibition)"
    },
    {
        "Category": "Concentration",
        "Rule": "Industry Concentration",
        "Threshold": "Max 10% (Exceptions: 2 at 12%, 1 at 15%)"
    },
    {
        "Category": "Reinvestment",
        "Rule": "Post-Reinvestment Maturity",
        "Threshold": "Purchases must have maturity <= Prepaid/Sold Asset maturity"
    },
    {
        "Category": "Definitions",
        "Rule": "CCC Excess Definition",
        "Threshold": "Must NOT have carveouts (e.g. excluding CCCs trading > par)"
    },
    {
        "Category": "Definitions",
        "Rule": "Discount Obligation Definition",
        "Threshold": "Must NOT have carveouts for CCC Collateral Obligations"
    },
    {
        "Category": "Other",
        "Rule": "Distressed Exchange",
        "Threshold": "Max 5% point-in-time, Max 20% cumulative"
    },
    {
        "Category": "Other",
        "Rule": "Trading Plans",
        "Threshold": "Max 5%. NO carveouts for Credit Risk sales."
    }
]

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
        width: 100%; border-radius: 4px; height: 3em;
        background-color: #1f2937; color: white; border: 1px solid #374151;
    }
    .stButton>button:hover { background-color: #374151; border-color: #6b7280; }
    .stTextInput>div>div>input { background-color: #1f2937; color: white; border: 1px solid #374151; }
    /* Table Styling */
    div[data-testid="stDataFrame"] { width: 100%; }
    </style>
    """, unsafe_allow_html=True)

# --- 3. BACKEND LOGIC ---
def get_session_db_path():
    if 'db_path' not in st.session_state:
        st.session_state['db_path'] = f"./chroma_db_{uuid.uuid4().hex}"
    return st.session_state['db_path']

def process_document(uploaded_file):
    api_key = get_api_key()
    if not api_key:
        st.error("❌ API Key missing.")
        return False

    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"📖 Reading {uploaded_file.name}...")
    
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Soft Reset DB
        if 'db_path' in st.session_state:
            old_path = st.session_state['db_path']
            if os.path.exists(old_path):
                try: shutil.rmtree(old_path)
                except: pass
        
        current_db_path = f"./chroma_db_{uuid.uuid4().hex}"
        st.session_state['db_path'] = current_db_path
        
        batch_size = 50
        progress_bar = st.progress(0)
        
        vectorstore = Chroma.from_documents(
            documents=splits[:batch_size], 
            embedding=OpenAIEmbeddings(api_key=api_key), 
            persist_directory=current_db_path
        )
        
        for i in range(batch_size, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            vectorstore.add_documents(batch)
            progress = min((i / len(splits)), 1.0)
            progress_bar.progress(progress, text="Indexing Document...")
            time.sleep(0.1)
            
        progress_bar.empty()
        os.remove(temp_path)
        return True
        
    except Exception as e:
        st.error(f"Error: {e}")
        if os.path.exists(temp_path): os.remove(temp_path)
        return False

def run_compliance_check(stip_rule):
    """
    Specific Agent logic to compare a Stip against the Doc.
    """
    api_key = get_api_key()
    if not api_key or 'db_path' not in st.session_state:
        return "Error", "No DB"

    current_db_path = st.session_state['db_path']
    
    try:
        vectorstore = Chroma(persist_directory=current_db_path, embedding_function=OpenAIEmbeddings(api_key=api_key))
        retriever = vectorstore.as_retriever()
        
        # Focused Prompt for Compliance
        template = """You are a strict Private Credit Compliance Officer.
        
        YOUR TASK: Compare the 'Eldridge Requirement' against the 'Document Language'.
        
        Eldridge Requirement: {question}
        
        Context from Document: {context}
        
        OUTPUT FORMAT:
        Provide a concise response starting with one of these tags:
        [MATCH] - If the document strictly meets or is better than the requirement.
        [DISCREPANCY] - If the document is looser, missing, or contradicts the requirement.
        
        After the tag, quote the specific language from the document that proves your decision. 
        If there is a discrepancy, explain exactly what the difference is.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=api_key)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        response = rag_chain.invoke(stip_rule)
        return response
    except Exception as e:
        return f"Error: {str(e)}"

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
    
    if st.button("⚠️ Clear Memory"):
        st.session_state['doc_ready'] = False
        if 'db_path' in st.session_state:
            del st.session_state['db_path']
        st.warning("Memory Cleared.")

st.title("🛡️ CLO Indenture vs. Stip Analyzer")

if 'doc_ready' not in st.session_state:
    st.info("👈 Please upload a document to begin.")
else:
    # --- TABBED INTERFACE ---
    tab1, tab2 = st.tabs(["📋 Compliance Audit", "💬 Deal Chat"])
    
    with tab1:
        st.subheader("Eldridge Compliance Matrix")
        st.caption("Automated check of Key Stipulations against the uploaded Indenture.")
        
        if st.button("RUN FULL AUDIT"):
            results = []
            progress_bar = st.progress(0)
            
            for idx, stip in enumerate(ELDRIDGE_STIPS):
                # Update Progress
                progress_bar.progress((idx + 1) / len(ELDRIDGE_STIPS), text=f"Checking: {stip['Rule']}...")
                
                # Run AI Check
                query = f"Check if the document complies with this rule: {stip['Rule']} which requires {stip['Threshold']}"
                ai_response = run_compliance_check(query)
                
                # Parse Result (Basic parsing for UI color)
                status = "❓ Review"
                if "[MATCH]" in ai_response:
                    status = "✅ MATCH"
                    ai_response = ai_response.replace("[MATCH]", "").strip()
                elif "[DISCREPANCY]" in ai_response:
                    status = "❌ DISCREPANCY"
                    ai_response = ai_response.replace("[DISCREPANCY]", "").strip()
                
                results.append({
                    "Stipulation": stip['Rule'],
                    "Eldridge Requirement": stip['Threshold'],
                    "Document Language & Analysis": ai_response,
                    "Status": status
                })
            
            progress_bar.empty()
            
            # Display as Dataframe
            df = pd.DataFrame(results)
            st.dataframe(
                df, 
                column_config={
                    "Status": st.column_config.TextColumn(
                        "Status",
                        help="Match vs Discrepancy",
                        width="medium"
                    ),
                    "Document Language & Analysis": st.column_config.TextColumn(
                        "Analysis",
                        width="large"
                    )
                },
                hide_index=True
            )

    with tab2:
        st.subheader("Deep Dive Query")
        user_input = st.chat_input("Ask a custom question about the indenture...")
        
        # Chat History/Session
        if "messages" not in st.session_state:
            st.session_state.messages = []

        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        if user_input:
            with st.chat_message("user"):
                st.markdown(user_input)
            st.session_state.messages.append({"role": "user", "content": user_input})

            with st.chat_message("assistant"):
                response = run_compliance_check(user_input) # Re-using the check function as a general query
                st.markdown(response)
            st.session_state.messages.append({"role": "assistant", "content": response})
