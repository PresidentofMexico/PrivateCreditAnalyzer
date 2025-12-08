# --- 1. SETUP & IMPORTS ---
import streamlit as st
import os
import shutil
import time

# FIX: ChromaDB requires new SQLite, which Streamlit Cloud doesn't have by default.
# We swap it for a binary version here.
try:
    __import__('pysqlite3')
    import sys
    sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
except ImportError:
    pass

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# CONFIG
st.set_page_config(page_title="Eldridge CLO Stip Analyzer", layout="wide", page_icon="🛡️")
PERSIST_DIRECTORY = "./db_storage_streamlit"

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

# --- 3. BACKEND LOGIC (The "Brain") ---
def process_document(uploaded_file):
    """
    Ingest the document directly within the Streamlit session.
    """
    # Save temp file
    temp_path = f"temp_{uploaded_file.name}"
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    
    st.info(f"📖 Reading {uploaded_file.name}...")
    
    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()
        
        # Split
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        splits = text_splitter.split_documents(docs)
        
        # Embed (Batched for safety)
        batch_size = 50
        progress_bar = st.progress(0)
        
        # Initialize DB with first batch
        if os.path.exists(PERSIST_DIRECTORY):
            shutil.rmtree(PERSIST_DIRECTORY)
            
        vectorstore = Chroma.from_documents(
            documents=splits[:batch_size], 
            embedding=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"]), 
            persist_directory=PERSIST_DIRECTORY
        )
        
        # Add rest
        total_batches = (len(splits) // batch_size) + 1
        for i in range(batch_size, len(splits), batch_size):
            batch = splits[i:i + batch_size]
            vectorstore.add_documents(batch)
            progress = min((i / len(splits)), 1.0)
            progress_bar.progress(progress, text=f"Embedding batch {i // batch_size}/{total_batches}")
            time.sleep(0.5) # Safety pause
            
        progress_bar.empty()
        os.remove(temp_path)
        return True
        
    except Exception as e:
        st.error(f"Error: {e}")
        return False

def run_query(question):
    """
    Run the RAG chain against the local vector store
    """
    try:
        vectorstore = Chroma(persist_directory=PERSIST_DIRECTORY, embedding_function=OpenAIEmbeddings(api_key=st.secrets["OPENAI_API_KEY"]))
        retriever = vectorstore.as_retriever()
        
        template = """You are a senior private credit analyst. 
        Use the following pieces of context from the loan agreement to answer the question.
        If you don't know the answer, just say that you don't know, don't make up terms.
        
        Context: {context}
        
        Question: {question}
        
        Answer:"""
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=st.secrets["OPENAI_API_KEY"])
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )
        
        return rag_chain.invoke(question)
    except Exception as e:
        return f"Error: {e}"

# --- 4. FRONTEND UI (The "Face") ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2704/2704022.png", width=50)
    st.title("Deal Room")
    uploaded_file = st.file_uploader("Upload Indenture (PDF)", type=['pdf'])
    
    if uploaded_file and st.button("🚀 Ingest Document"):
        with st.spinner("Processing..."):
            success = process_document(uploaded_file)
            if success:
                st.success("Ingestion Complete!")
                st.session_state['doc_ready'] = True

    # Eldridge Stips Cheat Sheet
    with st.expander("📋 Eldridge Stips", expanded=True):
        st.markdown("""
        **Concentration Limits**
        - Caa/CCC Limit: **7.5%**
        - Cov-lite: **60%**
        - Industry Cap: **10%**
        **Reinvestment**
        - Post-Reinv Maturity: **<= Sold Asset**
        """)

st.title("🛡️ CLO Indenture vs. Stip Analyzer")

if 'doc_ready' not in st.session_state:
    st.info("👈 Please upload a document to begin.")
else:
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 📊 Concentration Tests")
        if st.button("Check Caa/CCC Limits"):
            with st.spinner("Analyzing..."):
                st.success(run_query("Does the indenture limit Moody’s Caa and S&P CCC obligations to 7.5%? Are there carveouts?"))
        
        if st.button("Check Cov-Lite & Long Dated"):
            with st.spinner("Analyzing..."):
                st.success(run_query("Is there a 60% concentration limit for Cov-lite loans? Is there a 0% limit for Long Dated Obligations?"))

    with col2:
        st.markdown("### ⚖️ Reinvestment & Workouts")
        if st.button("Check Post-Reinvestment Maturity"):
            with st.spinner("Analyzing..."):
                st.success(run_query("Does the indenture require post-reinvestment purchases to have a maturity equal to or shorter than the prepaid/sold obligation?"))

    st.divider()
    user_input = st.chat_input("Ask a custom question about the indenture...")
    if user_input:
        with st.chat_message("assistant"):
            st.write(run_query(user_input))
