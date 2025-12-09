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

# --- HELPER: GET API KEY ---
def get_api_key():
    if "OPENAI_API_KEY" in st.secrets:
        return st.secrets["OPENAI_API_KEY"]
    elif os.getenv("OPENAI_API_KEY"):
        return os.getenv("OPENAI_API_KEY")
    return None

# --- STATE MANAGEMENT (FULL 30 ELDRIDGE STIPS) ---
# UPDATED: Switched 'SearchQuery' to "Mock Legal Clauses" for ALL Sections (A-F)
if 'stips' not in st.session_state:
    st.session_state['stips'] = [
        # A. Concentration Limitations (1-16)
        {
            "Category": "Concentration", 
            "Rule": "Moody’s Caa / S&P CCC Limit", 
            "Threshold": "Max 7.5% (Check Excess Caa/CCC)",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Caa Collateral Obligations and CCC Collateral Obligations"
        },
        {
            "Category": "Concentration", 
            "Rule": "Top 5 Obligors", 
            "Threshold": "Max 2.5% each (1.5% if non-senior secured)",
            "SearchQuery": "Concentration Limitations single Obligor issued by up to five such Obligors" 
        },
        {
            "Category": "Concentration", 
            "Rule": "Cov-Lite Loans", 
            "Threshold": "Max 60%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Covenant Lite Loans"
        },
        {
            "Category": "Concentration", 
            "Rule": "Small Obligors ($150M-$250M)", 
            "Threshold": "Max 5% for obligors with debt $150M-$250M",
            "SearchQuery": "Concentration Limitations Domiciled Obligors Indebtedness of less than"
        },
        {
            "Category": "Concentration", 
            "Rule": "Long Dated Obligations", 
            "Threshold": "0% allowed (Strict prohibition)",
            "SearchQuery": "Concentration Limitations Long Dated Obligations maturity date"
        },
        {
            "Category": "Concentration", 
            "Rule": "Bridge Loans", 
            "Threshold": "Max 2.5%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Bridge Loans"
        },
        {
            "Category": "Concentration", 
            "Rule": "Fixed Rate / Non-Loan Assets", 
            "Threshold": "Max 5%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Fixed Rate Obligations"
        },
        {
            "Category": "Concentration", 
            "Rule": "Delayed Drawdown / Revolving", 
            "Threshold": "Max 10%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Revolving Collateral Obligations and Delayed Drawdown Collateral Obligations"
        },
        {
            "Category": "Concentration", 
            "Rule": "Senior Secured Loans", 
            "Threshold": "Min 90% of Collateral Principal Amount",
            "SearchQuery": "Concentration Limitations not less than % of the Collateral Principal Amount may consist of Senior Secured Loans"
        },
        {
            "Category": "Concentration", 
            "Rule": "Participation Interests", 
            "Threshold": "Max 10%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Participation Interests"
        },
        {
            "Category": "Concentration", 
            "Rule": "Deferrable Obligations", 
            "Threshold": "Max 5%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Deferrable Obligations"
        },
        {
            "Category": "Concentration", 
            "Rule": "DIP Obligations", 
            "Threshold": "Max 7.5%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of DIP Collateral Obligations"
        },
        {
            "Category": "Concentration", 
            "Rule": "Industry Concentration", 
            "Threshold": "Max 10% (Exceptions: 2 at 12%, 1 at 15%)",
            "SearchQuery": "Concentration Limitations S&P Industry Classification single industry"
        },
        {
            "Category": "Concentration", 
            "Rule": "Current Pay Obligations", 
            "Threshold": "Max 5%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Current Pay Obligations"
        },
        {
            "Category": "Concentration", 
            "Rule": "Payment Frequency < Quarterly", 
            "Threshold": "Max 5%",
            "SearchQuery": "Concentration Limitations pay interest less frequently than quarterly"
        },
        {
            "Category": "Concentration", 
            "Rule": "Discount Obligations", 
            "Threshold": "Max 20%",
            "SearchQuery": "Concentration Limitations not more than % of the Collateral Principal Amount may consist of Discount Obligations"
        },

        # B. Reinvestment (17-19)
        {
            "Category": "Reinvestment", 
            "Rule": "Post-Reinvestment Maturity", 
            "Threshold": "Maturity must be <= Prepaid/Sold Asset",
            "SearchQuery": "Reinvestment Period Substitute Obligations shall have a Stated Maturity not later than the Stated Maturity of the Reinvested Asset"
        },
        {
            "Category": "Reinvestment", 
            "Rule": "O/C Test Compliance", 
            "Threshold": "Must satisfy O/C test after reinvestment",
            "SearchQuery": "Reinvestment Period satisfied the Overcollateralization Ratio Test"
        },
        {
            "Category": "Reinvestment", 
            "Rule": "Proceeds Reinvestment Timing", 
            "Threshold": "Later of 45 days or 2nd determination date",
            "SearchQuery": "Reinvestment Period Sale Proceeds may be reinvested within Business Days"
        },

        # C. Supplemental Indenture (20)
        {
            "Category": "Structural", 
            "Rule": "Supplemental Indenture Consent", 
            "Threshold": "Majority of Controlling Class required to change Tests/Limits/Defs",
            "SearchQuery": "Supplemental Indenture with the consent of the Majority of the Controlling Class"
        },

        # D. Required Definitions (21-23)
        {
            "Category": "Definitions", 
            "Rule": "CCC Excess Definition", 
            "Threshold": "NO carveouts allowed (e.g. excluding CCCs > par)",
            "SearchQuery": "definition Excess CCC/Caa Adjustment Amount means"
        },
        {
            "Category": "Definitions", 
            "Rule": "Discount Obligation Definition", 
            "Threshold": "NO carveouts for CCC Collateral Obligations",
            "SearchQuery": "definition Discount Obligation means any Collateral Obligation"
        },
        {
            "Category": "Definitions", 
            "Rule": "Small Obligor Definition", 
            "Threshold": "Min Indebtedness $150M. NO allowance for <$150M.",
            "SearchQuery": "definition Small Obligor means any Obligor with total Indebtedness"
        },

        # E. Other Requirements (24-28)
        {
            "Category": "Other", 
            "Rule": "Distressed Exchange", 
            "Threshold": "Max 5% point-in-time, 20% cumulative",
            "SearchQuery": "Distressed Exchange Offer means an offer by the Issuer"
        },
        {
            "Category": "Other", 
            "Rule": "FLLO Treatment", 
            "Threshold": "Must be treated as Second Lien Loans",
            "SearchQuery": "First Lien Last Out Loan shall be treated as a Second Lien Loan"
        },
        {
            "Category": "Other", 
            "Rule": "Minimum Purchase Price", 
            "Threshold": "50% floor (5% allowance for 50-60%)",
            "SearchQuery": "Collateral Obligation acquired for a purchase price of less than %"
        },
        {
            "Category": "Other", 
            "Rule": "Trading Plan Allowance", 
            "Threshold": "Max 5%. NO Credit Risk sales carveout.",
            "SearchQuery": "Trading Plan provided that the aggregate Principal Balance"
        },
        {
            "Category": "Other", 
            "Rule": "Trading Plan Maturity", 
            "Threshold": "Min 6 months. Max 3 years avg life diff.",
            "SearchQuery": "Trading Plan Average Life difference"
        },

        # F. Workouts (29-30)
        {
            "Category": "Workouts", 
            "Rule": "Workout Sale Proceeds", 
            "Threshold": "Treat as Principal up to default balance (no distinction)",
            "SearchQuery": "Workout Loan Sale Proceeds shall be treated as Principal Proceeds"
        },
        {
            "Category": "Workouts", 
            "Rule": "Workout Purchase w/ Interest", 
            "Threshold": "Only if all notes interest is paid/sufficient",
            "SearchQuery": "Workout Loan interest may be purchased using Interest Proceeds only if"
        },
    ]

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
    /* Dataframe styling */
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
        
        # UPDATED: Increased Chunk Size to 2000 to capture full Context Lists
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=400)
        splits = text_splitter.split_documents(docs)
        
        # Soft Reset DB if it exists
        if 'db_path' in st.session_state:
            old_path = st.session_state['db_path']
            if os.path.exists(old_path):
                try: shutil.rmtree(old_path)
                except: pass
        
        # Create fresh path
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

def run_compliance_check(stip_rule, stip_threshold, user_tweaks="", search_override=None):
    """
    Specific Agent logic to compare a Stip against the Doc using Prompt Engineering.
    """
    api_key = get_api_key()
    if not api_key or 'db_path' not in st.session_state:
        return "Error: No DB"

    current_db_path = st.session_state['db_path']
    
    try:
        vectorstore = Chroma(persist_directory=current_db_path, embedding_function=OpenAIEmbeddings(api_key=api_key))
        
        # UPDATED: Increased k to 15 to ensure we catch distributed lists
        retriever = vectorstore.as_retriever(search_kwargs={"k": 15})
        
        # UPDATED: Added [STRICTER] tag to the Prompt Template
        template = """You are a strict Private Credit Compliance Officer.
        
        YOUR TASK: Compare the 'Eldridge Requirement' against the 'Document Language'.
        
        Eldridge Requirement: {rule} which requires {threshold}
        
        User Guidance (Important Context): {tweaks}
        
        Context from Document: {context}
        
        OUTPUT FORMAT:
        Provide a concise response starting with one of these tags:
        [MATCH] - If the document meets the requirement exactly.
        [STRICTER] - If the document is MORE conservative/restrictive than the requirement (this is GOOD/ACCEPTABLE).
        [DISCREPANCY] - If the document is looser, missing, or contradicts the requirement (this is BAD).
        
        After the tag, quote the specific language from the document (with Section # if available) that proves your decision. 
        If there is a discrepancy, explain exactly what the difference is.
        """
        
        prompt = ChatPromptTemplate.from_template(template)
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=api_key)
        
        rag_chain = (
            {"context": retriever, "question": RunnablePassthrough()}
            | prompt.partial(rule=stip_rule, threshold=stip_threshold, tweaks=user_tweaks)
            | llm
            | StrOutputParser()
        )
        
        # UPDATED: Use specific Search Terms if available, else fall back to Rich Query
        if search_override:
            search_query = search_override
        else:
            search_query = f"Find language regarding {stip_rule} limit of {stip_threshold} definitions concentration limitations"
        
        response = rag_chain.invoke(search_query)
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

    st.divider()
    st.subheader("⚙️ Audit Settings")
    user_guidance = st.text_area("Global Instructions (Optional)", 
        placeholder="e.g., 'Ignore the Preliminary OM section, focus on Article 12 definitions.'",
        help="These instructions will be added to every check the agent runs."
    )

st.title("🛡️ CLO Indenture vs. Stip Analyzer")

if 'doc_ready' not in st.session_state:
    st.info("👈 Please upload a document to begin.")
else:
    # --- TABBED INTERFACE ---
    tab1, tab2, tab3 = st.tabs(["📋 Audit Table", "💬 Composer (Chat)", "✏️ Edit Stips"])
    
    # TAB 1: THE AUDIT TABLE
    with tab1:
        st.subheader("Eldridge Compliance Matrix")
        st.caption("Automated check of Key Stipulations against the uploaded Indenture.")
        
        if st.button("RUN FULL AUDIT", type="primary"):
            results = []
            progress_bar = st.progress(0)
            stips_list = st.session_state['stips'] 
            
            for idx, stip in enumerate(stips_list):
                progress_bar.progress((idx + 1) / len(stips_list), text=f"Checking: {stip['Rule']}...")
                
                # UPDATED: Pass SearchQuery if available
                query_override = stip.get("SearchQuery", None)
                ai_response = run_compliance_check(stip['Rule'], stip['Threshold'], user_guidance, query_override)
                
                status = "❓ Review"
                clean_response = ai_response
                # UPDATED: New Status Logic for STRICTER
                if "[MATCH]" in ai_response:
                    status = "✅ MATCH"
                    clean_response = ai_response.replace("[MATCH]", "").strip()
                elif "[STRICTER]" in ai_response:
                    status = "✅ STRICTER"
                    clean_response = ai_response.replace("[STRICTER]", "").strip()
                elif "[DISCREPANCY]" in ai_response:
                    status = "❌ DISCREPANCY"
                    clean_response = ai_response.replace("[DISCREPANCY]", "").strip()
                
                results.append({
                    "Category": stip.get("Category", "General"),
                    "Stipulation": stip['Rule'],
                    "Requirement": stip['Threshold'],
                    "Analysis & Language": clean_response,
                    "Status": status
                })
            
            progress_bar.empty()
            
            # Display as Interactive Dataframe
            df = pd.DataFrame(results)
            st.dataframe(
                df, 
                column_config={
                    "Status": st.column_config.TextColumn("Status", width="small"),
                    "Analysis & Language": st.column_config.TextColumn("Analysis", width="large"),
                    "Stipulation": st.column_config.TextColumn("Rule", width="medium"),
                    "Requirement": st.column_config.TextColumn("Threshold", width="medium"),
                    "Category": st.column_config.TextColumn("Category", width="small"),
                },
                hide_index=True,
                use_container_width=True
            )

    # TAB 2: FREE CHAT
    with tab2:
        st.subheader("Deal Chat Composer")
        user_input = st.chat_input("Ask a custom question about the indenture...")
        
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
                response = run_compliance_check(user_input, "N/A (Custom Query)", user_guidance)
                # UPDATED: Clean new tag from Chat
                clean_response = response.replace("[MATCH]", "").replace("[DISCREPANCY]", "").replace("[STRICTER]", "")
                st.markdown(clean_response)
            st.session_state.messages.append({"role": "assistant", "content": clean_response})

    # TAB 3: EDIT RULES
    with tab3:
        st.subheader("Edit Standard Stipulations")
        st.info("Modify these rules for this specific deal. Changes apply when you run the Audit.")
        
        with st.form("edit_stips_form"):
            current_stips = st.session_state['stips']
            updated_stips = []
            
            for i, stip in enumerate(current_stips):
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.text_input(f"Rule", value=stip['Rule'], key=f"rule_name_{i}", disabled=True)
                with col2:
                    new_thresh = st.text_input(f"Threshold", value=stip['Threshold'], key=f"thresh_{i}")
                    # Preserve Category and SearchQuery
                    updated_stips.append({
                        "Rule": stip['Rule'], 
                        "Threshold": new_thresh, 
                        "Category": stip.get("Category", "General"),
                        "SearchQuery": stip.get("SearchQuery", None)
                    })
            
            if st.form_submit_button("💾 Save Rule Changes"):
                st.session_state['stips'] = updated_stips
                st.success("Rules updated! Go back to the 'Audit Table' tab to run the check.")
