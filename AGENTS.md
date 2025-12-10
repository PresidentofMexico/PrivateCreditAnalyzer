Private Credit CLO Analyzer (Eldridge Deal Room)

### Dev environment tips

Setup: Create a fresh virtual environment using python -m venv venv and activate it.

Dependencies: Run pip install -r requirements.txt to install the pinned versions (Streamlit 1.41.0, LangChain 0.3.7, etc.).

Local Run: Use streamlit run streamlit_app.py to launch the monolith locally.

Secrets: You must create a .streamlit/secrets.toml file containing OPENAI_API_KEY = "sk-..." for the local app to function.

Database: The app creates local ./chroma_db_{uuid} folders. Add chroma_db_* to your .gitignore to prevent committing massive vector stores.

### Testing instructions

UAT Protocol: Navigate to the "Audit Table" tab and click "RUN FULL AUDIT".

Logic Verification:

Verify that conservative deviations are flagged as ✅ STRICTER (e.g., limit is 1.0% vs required 1.5%).

Verify that riskier deviations are flagged as ❌ DISCREPANCY.

Rate Limit Test: Rapidly trigger the audit. Verify that the app catches 429 errors, shows a "Cooling Down" toast notification, and retries automatically without crashing.

Snowflake Migration: If testing the decoupled architecture, run snowflake_setup.sql in a Snowflake Worksheet first to establish the CLO_DEALS_VECTORS table.

### PR instructions

Title format: [Feature/Fix] <Short Description> (e.g., [Fix] Update Rate Limit Backoff).

Dependency Check: If you import a new library, run pip freeze > requirements.txt and verify it doesn't conflict with pysqlite3-binary.

Logic Updates: If you modify the Compliance Agent's logic (e.g., changing the "Mock Clause" for a Stip), you must update the "Agent Logic" section below to reflect the new search strategy.

Monolith Integrity: Ensure streamlit_app.py remains self-contained until the full Snowflake migration is approved.

### Agent Logic & Architecture Reference

Core Identity

The Eldridge Compliance Agent is a Legal RAG system auditing Private Credit CLO Indentures against 30 Investor Stipulations.

Retrieval Strategy (HyDE)

We use Mock Legal Clauses instead of keywords.

Old: Concentration Limitations Top 5 Obligors

Current: "Concentration Limitations single Obligor issued by up to five such Obligors"

Reason: Captures the exact legal phrasing of the limit, avoiding generic definitions.

Decision Engine States

[MATCH]: Exact alignment.

[STRICTER]: Document is safer/more conservative than required (PASS).

[DISCREPANCY]: Document is riskier than required (FAIL).

Infrastructure (Current vs. Target)

Current: Streamlit Cloud + ChromaDB (Local). Uses pysqlite3-binary fix.

Target: Snowflake Native App.

Write: Background Task chunks/embeds PDFs into CLO_DEALS_VECTORS.

Read: Streamlit in Snowflake queries vectors via SQL (VECTOR_COSINE_SIMILARITY).
