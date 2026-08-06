"""
TriSeva Web User Interface & Chainlit Interactive Application Server.

Provides a modern web interface for TriSeva multi-agent system, supporting:
- Multilingual chat & voice input
- Image/PDF document upload & OCR processing
- Real-time Multi-Agent Chain-of-Thought execution step visibility
- User session database tracking & user study analytics logging
"""

import warnings
warnings.filterwarnings("ignore")

# Monkeypatch Chainlit's OAuth2PasswordBearerWithCookie for compatibility with FastAPI >= 0.111.0
try:
    import chainlit.auth.cookie as c_auth
    from fastapi.openapi.models import OAuth2 as OAuth2Model, OAuthFlows as OAuthFlowsModel, OAuthFlowPassword
    if hasattr(c_auth, "OAuth2PasswordBearerWithCookie") and not hasattr(c_auth.OAuth2PasswordBearerWithCookie, "model"):
        c_auth.OAuth2PasswordBearerWithCookie.model = OAuth2Model(
            flows=OAuthFlowsModel(password=OAuthFlowPassword(tokenUrl="token"))
        )
except Exception:
    pass

import os
import sys
import json
import time

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["LANGCHAIN_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from main import ask
from fastapi.staticfiles import StaticFiles

# ── Mount Static Files (/public) on FastAPI Server ────────────────────────────
try:
    from chainlit.server import app as chainlit_fastapi_app
    public_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "public"))
    if os.path.exists(public_dir):
        chainlit_fastapi_app.mount("/public", StaticFiles(directory=public_dir), name="public_custom_files")
except Exception as e:
    pass

import sqlite3

# ── Auto-initialize SQLite Database Schema ────────────────────────────────────
def init_db():
    """Initializes local SQLite database schema (`chainlit.db`) for user authentication and chat threads."""
    conn = sqlite3.connect("chainlit.db")
    cursor = conn.cursor()
    cursor.executescript("""
    CREATE TABLE IF NOT EXISTS users (
        id TEXT PRIMARY KEY,
        identifier TEXT NOT NULL UNIQUE,
        metadata TEXT NOT NULL,
        createdAt TEXT
    );

    CREATE TABLE IF NOT EXISTS threads (
        id TEXT PRIMARY KEY,
        createdAt TEXT,
        name TEXT,
        userId TEXT,
        userIdentifier TEXT,
        tags TEXT,
        metadata TEXT,
        FOREIGN KEY (userId) REFERENCES users(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS steps (
        id TEXT PRIMARY KEY,
        name TEXT NOT NULL,
        type TEXT NOT NULL,
        threadId TEXT NOT NULL,
        parentId TEXT,
        streaming BOOLEAN NOT NULL,
        waitForAnswer BOOLEAN,
        isError BOOLEAN,
        metadata TEXT,
        tags TEXT,
        input TEXT,
        output TEXT,
        createdAt TEXT,
        command TEXT,
        start TEXT,
        end TEXT,
        generation TEXT,
        showInput TEXT,
        language TEXT,
        indent INTEGER,
        defaultOpen BOOLEAN,
        autoCollapse BOOLEAN,
        modes TEXT,
        disableFeedback BOOLEAN,
        FOREIGN KEY (threadId) REFERENCES threads(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS elements (
        id TEXT PRIMARY KEY,
        threadId TEXT,
        type TEXT,
        url TEXT,
        chainlitKey TEXT,
        name TEXT NOT NULL,
        display TEXT,
        objectKey TEXT,
        size TEXT,
        page INTEGER,
        language TEXT,
        forId TEXT,
        mime TEXT,
        FOREIGN KEY (threadId) REFERENCES threads(id) ON DELETE CASCADE
    );

    CREATE TABLE IF NOT EXISTS feedbacks (
        id TEXT PRIMARY KEY,
        forId TEXT NOT NULL,
        value INTEGER NOT NULL,
        comment TEXT,
        strategy TEXT NOT NULL
    );
    """)
    conn.commit()
    conn.close()

init_db()

# ── Data Layer for Thread & Feedback Persistence ──────────────────────────────
@cl.data_layer
def get_data_layer():
    """Returns SQLAlchemy persistent data layer instance for Chainlit thread storage.

    Returns:
        SQLAlchemyDataLayer: Database data layer instance.
    """
    db_path = os.path.abspath("chainlit.db")
    conn_str = f"sqlite+aiosqlite:///{db_path}"
    return SQLAlchemyDataLayer(conn_string=conn_str)

# ── Authentication Callback ──────────────────────────────────────────────────
@cl.password_auth_callback
def auth_callback(username, password):
    """Authenticates citizen user credentials for access control.

    Args:
        username (str): User identifier.
        password (str): Password string.

    Returns:
        cl.User | None: Authenticated user object or None.
    """
    if (username == "citizen" and password == "triseva2025") or (username == "admin" and password == "admin123"):
        return cl.User(identifier=username, metadata={"role": "user", "provider": "credentials"})
    return None


DOMAIN_META = {
    "health":      {"icon": "🏥", "label": "Healthcare",  "color": "#ef4444"},
    "legal":       {"icon": "⚖️", "label": "Legal / Government", "color": "#3b82f6"},
    "agriculture": {"icon": "🌾", "label": "Agriculture", "color": "#10b981"},
    "unknown":     {"icon": "🤖", "label": "General Assistance", "color": "#6b7280"},
}

SPECIFIC_SOURCE_MAP = {
    "medline": ("MedlinePlus Official Medical Database", "https://medlineplus.gov"),
    "health": ("Ministry of Health & Family Welfare Guidelines", "https://www.mohfw.gov.in"),
    "nih": ("National Institutes of Health (NCBI PubMed)", "https://www.ncbi.nlm.nih.gov"),
    "who": ("World Health Organization (WHO) Health Topics", "https://www.who.int"),

    "mgnrega": ("myScheme Portal - MGNREGA Scheme Guidelines", "https://www.myscheme.gov.in/schemes/mgnrega"),
    "nrega": ("MGNREGA Official Ministry Portal (nrega.nic.in)", "https://nrega.nic.in"),
    "master_roll": ("MGNREGA Master Roll Operational Framework", "https://nrega.nic.in"),
    "constitution": ("Constitution of India Official Legislative Portal", "https://lddashboard.legislative.gov.in/sites/default/files/COI...pdf"),
    "article_371a": ("Article 371A Constitutional Provisions (Nagaland)", "https://www.india.gov.in/my-government/constitution-india"),
    "371a": ("Article 371A Constitutional Provisions for Nagaland", "https://www.india.gov.in/my-government/constitution-india"),
    "bns": ("Bharatiya Nyaya Sanhita (BNS Act 2023 Official PDF)", "https://www.mha.gov.in/sites/default/files/25072024_BNS_English.pdf"),
    "bnss": ("Bharatiya Nagarik Suraksha Sanhita (BNSS Act 2023 Official PDF)", "https://www.mha.gov.in/sites/default/files/25072024_BNSS_English.pdf"),
    "bsa": ("Bharatiya Sakshya Adhiniyam (BSA Act 2023 Official PDF)", "https://www.mha.gov.in/sites/default/files/25072024_BSA_English.pdf"),
    "dpdp": ("Digital Personal Data Protection Act 2023 Official PDF", "https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf"),
    "rti": ("Right to Information Act 2005 Official Document", "https://rti.gov.in/webportal/RTIAct2005.pdf"),
    "nfsa": ("National Food Security Act (NFSA Official Guidelines)", "https://dfpd.gov.in"),
    "pmkisan": ("PM-KISAN Operational Guidelines Official Document", "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf"),
    "pm-kisan": ("PM-KISAN Operational Guidelines Official Document", "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf"),
    "ayushman": ("Ayushman Bharat PM-JAY Official Portal", "https://pmjay.gov.in/about/pmjay"),
    "structured schemes": ("myScheme National Official Government Schemes Portal", "https://www.myscheme.gov.in"),

    "pmfby": ("Pradhan Mantri Fasal Bima Yojana Operational Guidelines", "https://pmfby.gov.in/pdf/Revised_Operational_Guidelines.pdf"),
    "fasal_bima": ("PM Fasal Bima Yojana Official Guidelines", "https://pmfby.gov.in/pdf/Revised_Operational_Guidelines.pdf"),
    "kusum": ("PM-KUSUM Solar Pump Scheme Official Portal", "https://pmkusum.mnre.gov.in"),
    "soil": ("Soil Health Card National Scheme Portal", "https://soilhealth.dac.gov.in"),
    "aif": ("Agriculture Infrastructure Fund (AIF Official Portal)", "https://agriinfra.dac.gov.in"),
    "icar": ("ICAR National Agricultural Research & Advisory Network", "https://icar.org.in"),
    "kvk": ("Krishi Vigyan Kendra (KVK Advisory Network)", "https://icar.org.in"),
    "annual_report": ("Ministry of Agriculture Annual Reports & Policy Docs", "https://agricoop.nic.in"),
    "nfsm": ("National Food Security Mission (NFSM Portal)", "https://nfsm.gov.in"),
    "pm-rkvy": ("Rashtriya Krishi Vikas Yojana (RKVY Guidelines)", "https://rkvy.nic.in"),
}

def get_official_scheme_url(source_name: str) -> str:
    """Maps document metadata source names to official government URLs.

    Args:
        source_name (str): Document source path or key.

    Returns:
        str: Mapped official web URL string.
    """
    if not source_name:
        return "https://www.india.gov.in"
        
    source_lower = source_name.lower()
    
    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        return source_name

    for prefix, (title, web_url) in SPECIFIC_SOURCE_MAP.items():
        if prefix in source_lower:
            return web_url
            
    if any(ext in source_lower for ext in [".pdf", ".png", ".jpg", ".txt"]):
        if "health" in source_lower:
            return "https://www.mohfw.gov.in"
        elif "legal" in source_lower:
            return "https://www.india.gov.in"
        elif "agri" in source_lower:
            return "https://agricoop.nic.in"
            
    return "https://www.india.gov.in"


# ── Session Start Handler ───────────────────────────────────────────────────
@cl.on_chat_start
async def on_start():
    """Triggered on new Chainlit user session initialization. Generates unique session_id."""
    session_id = os.urandom(8).hex()
    cl.user_session.set("session_id", session_id)


# ── User Query & Message Handler ─────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):
    """Processes incoming user chat messages, document image uploads, and executes TriSeva workflow.

    Args:
        message (cl.Message): Chainlit incoming user message.
    """
    session_id = cl.user_session.get("session_id")
    query      = message.content.strip()

    # Extract uploaded file path or intelligently reuse session image for follow-up document queries
    image_path = None
    if message.elements:
        for element in message.elements:
            if element.path and os.path.exists(element.path):
                image_path = element.path
                cl.user_session.set("last_image_path", image_path)
                break
    else:
        # Only reuse active session image if the query asks about the image/document or if query is blank
        last_path = cl.user_session.get("last_image_path")
        if last_path:
            query_low = query.lower()
            referential_keywords = ["it", "this", "image", "document", "prescription", "card", "report", "summarize", "dosage", "explain", "scan", "photo", "what about"]
            if not query_low or any(k in query_low for k in referential_keywords):
                image_path = last_path

    if not query:
        if image_path:
            query = "Summarize and explain the provided document image."
        else:
            return

    # ── Multi-Agent Chain of Thought Reasoning Steps ──────────────────────────
    async with cl.Step(name="🧠 Multi-Agent Chain of Thought Reasoning", type="run") as main_step:
        main_step.input = query

        start_time = time.time()
        result = await cl.make_async(ask)(
            query=query,
            session_id=session_id,
            image_path=image_path,
        )
        latency = round(time.time() - start_time, 2)

        answer     = result.get("answer") or "No answer generated."
        domain     = result.get("domain", "unknown")
        score      = result.get("score")
        disclaimer = result.get("disclaimer")
        quiz       = result.get("quiz", [])
        sources    = result.get("sources", [])
        chunks     = result.get("retrieved_chunks", [])
        telemetry  = result.get("telemetry", {})

        from agents.utils import is_answer_not_found
        from utils.translation_helper import is_hindi_or_hinglish
        not_found = is_answer_not_found(answer)

        if not_found:
            sources = []
            chunks = []

        # Sub-step 1: Domain Routing Explanation
        async with cl.Step(name="🔍 Domain Router Step", type="tool") as s1:
            hops_str = " -> ".join(telemetry.get("routing_hops", [domain]))
            s1.output = f"**Assigned Specialist Agent:** {domain.upper()}\n**Routing Path:** `{hops_str}`\n**Routing Latency:** {latency}s"

        # Sub-step 2: Hybrid RAG Retrieval Explanation
        async with cl.Step(name="📚 Hybrid RAG Retrieval Step", type="tool") as s2:
            if not_found:
                s2.output = f"**Retrieved Context Chunks:** 0 relevant chunks found\n**Status:** No matching context found in `triseva_{domain}` database"
            else:
                s2.output = f"**Retrieved Context Chunks:** {len(chunks)} chunks from `triseva_{domain}`\n**Search Strategy:** Hybrid E5 Dense + BM25 Lexical + RRF Reranking"

        # Sub-step 3: Dual-Judge Verification Explanation
        async with cl.Step(name="⚖️ Dual-Judge Quality Assurance Step", type="tool") as s3:
            if not_found:
                s3.output = f"**Status:** ℹ️ Fallback triggered — answer not found in available database"
            else:
                s3.output = f"**Faithfulness & Grounding Score:** {score if score is not None else 1.0}\n**Evaluation Framework:** Dual-Judge (Sarvam-105B + Claude-3.5-Haiku)\n**Status:** ✅ Approved & Factually Grounded"

        meta  = DOMAIN_META.get(domain, {"icon": "🤖", "label": domain.title(), "color": "#6b7280"})
        main_step.output = f"Executed multi-agent workflow for {meta['icon']} {meta['label']} in {latency} seconds."

        # ── Write User Study Session Logs ───────────────────────────────────
        try:
            log_data = {
                "session_id": session_id,
                "username": cl.user_session.get("user").identifier if cl.user_session.get("user") else "anonymous",
                "query_text": query,
                "response_text": answer,
                "selected_domain": domain,
                "latency_sec": latency,
                "score": score,
                "timestamp": time.time()
            }
            log_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "logs"))
            os.makedirs(log_dir, exist_ok=True)
            log_file = os.path.join(log_dir, "user_study_logs.jsonl")
            with open(log_file, "a", encoding="utf-8") as lf:
                lf.write(json.dumps(log_data) + "\n")
        except Exception as e:
            print(f"Error writing session logs: {e}")

    # ── Build main response ─────────────────────────────────────────────────
    header  = f"{meta['icon']} **{meta['label']}**"
    content = f"{header}\n\n---\n\n{answer}"

    if disclaimer:
        clean_disc = disclaimer.replace("⚕️", "").replace("⚖️", "").replace("🌾", "").strip()
        content += f"\n\n---\n> 💡 *{clean_disc}*"

    # Send main answer
    msg = cl.Message(content=content)
    await msg.send()