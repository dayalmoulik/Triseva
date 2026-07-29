import warnings
warnings.filterwarnings("ignore")

# Monkeypatch FastAPI openapi security scheme to prevent AttributeError in Chainlit with FastAPI>=0.115
try:
    import fastapi.openapi.utils as fou
    _orig_get_openapi_security_definitions = fou.get_openapi_security_definitions
    def _patched_get_openapi_security_definitions(flat_dependant, security_definitions):
        try:
            return _orig_get_openapi_security_definitions(flat_dependant, security_definitions)
        except AttributeError as e:
            if "model" in str(e):
                return {}, []
            raise e
    fou.get_openapi_security_definitions = _patched_get_openapi_security_definitions
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

import sqlite3

# ── Auto-initialize SQLite Database Schema ────────────────────────────────────
def init_db():
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
        props TEXT
    );

    CREATE TABLE IF NOT EXISTS feedbacks (
        id TEXT PRIMARY KEY,
        forId TEXT NOT NULL,
        value INTEGER NOT NULL,
        comment TEXT
    );
    """)
    # Programmatic migration to add autoCollapse to existing steps table if missing
    try:
        cursor.execute("ALTER TABLE steps ADD COLUMN autoCollapse BOOLEAN;")
    except sqlite3.OperationalError:
        pass
    # Programmatic migration to add mime and props columns to elements table if missing
    try:
        cursor.execute("ALTER TABLE elements ADD COLUMN mime TEXT;")
    except sqlite3.OperationalError:
        pass
    try:
        cursor.execute("ALTER TABLE elements ADD COLUMN props TEXT;")
    except sqlite3.OperationalError:
        pass
    conn.commit()
    conn.close()

init_db()


# ── Data Persistence Layer ──────────────────────────────────────────────────
@cl.data_layer
def get_data_layer():
    return SQLAlchemyDataLayer(conninfo="sqlite+aiosqlite:///chainlit.db")


# ── User Authentication ──────────────────────────────────────────────────────
@cl.password_auth_callback
def auth_callback(username: str, password: str):
    import re
    user_clean = username.strip().lower()
    
    # 1. Verify Admin Account
    if user_clean == "admin":
        admin_password = os.getenv("ADMIN_PASSWORD") or "triseva@admin"
        if password == admin_password:
            return cl.User(identifier=user_clean, role="admin")
        return None
        
    # 2. Verify regular study participants
    required_password = os.getenv("STUDY_PASSWORD") or "triseva@study"
    if password != required_password:
        return None
        
    match = re.match(r"^study_(0[1-9]|[1-2][0-9]|30)$", user_clean)
    if not match:
        return None
        
    return cl.User(identifier=user_clean, role="user")


# ── Domain styling ─────────────────────────────────────────────────────────
DOMAIN_META = {
    "health":    {"icon": "🏥", "label": "Healthcare",         "color": "#be123c"},
    "legal":     {"icon": "⚖️",  "label": "Legal / Government", "color": "#0369a1"},
    "agriculture": {"icon": "🌾", "label": "Agriculture",       "color": "#0f766e"},
}

# ── Local Document to Official Web Portal Mapping ───────────────────────────
LOCAL_TO_WEB_MAP = {
    # Healthcare
    "medline": "https://medlineplus.gov",
    "health": "https://www.mohfw.gov.in",
    "nih": "https://www.ncbi.nlm.nih.gov",
    "who": "https://www.who.int",
    # Legal & Government
    "bnss": "https://www.mha.gov.in",
    "bns": "https://www.mha.gov.in",
    "bsa": "https://www.mha.gov.in",
    "dpdp": "https://www.meity.gov.in",
    "rti": "https://rti.gov.in",
    "nfsa": "https://dfpd.gov.in",
    "pmkisan": "https://pmkisan.gov.in",
    "pm-kisan": "https://pmkisan.gov.in",
    "ayushman": "https://pmjay.gov.in",
    "legal": "https://www.india.gov.in",
    # Agriculture
    "annual_report": "https://agricoop.nic.in",
    "nfsm": "https://nfsm.gov.in",
    "pm-rkvy": "https://rkvy.nic.in",
    "pdmc": "https://pmksy.gov.in",
    "midh": "https://midh.gov.in",
    "atma": "https://agricoop.nic.in",
    "fpo": "https://sfacindia.com",
    "aif": "https://agriinfra.dac.gov.in",
    "soil": "https://soilhealth.dac.gov.in",
    "pm-aasha": "https://pmaasha.nic.in",
    "agriculture": "https://agricoop.nic.in",
    "agri": "https://agricoop.nic.in"
}

def resolve_web_url(source_name: str) -> str:
    """Returns a valid web URL for a source, mapping local docs to official portals, or None if unmapped."""
    if not source_name:
        return None
    source_lower = source_name.strip().lower()
    
    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        return source_name
        
    for prefix, web_url in LOCAL_TO_WEB_MAP.items():
        if prefix in source_lower:
            return web_url
            
    if any(ext in source_lower for ext in [".pdf", ".png", ".jpg", ".txt"]):
        if "health" in source_lower:
            return "https://www.mohfw.gov.in"
        elif "legal" in source_lower:
            return "https://www.india.gov.in"
        elif "agri" in source_lower:
            return "https://agricoop.nic.in"
            
    return None



# ── Session start ───────────────────────────────────────────────────────────
@cl.on_chat_start
async def on_start():
    session_id = os.urandom(8).hex()
    cl.user_session.set("session_id", session_id)


# ── Message handler ─────────────────────────────────────────────────────────
@cl.on_message
async def on_message(message: cl.Message):
    session_id = cl.user_session.get("session_id")
    query      = message.content.strip()

    # Extract uploaded file path
    image_path = None
    if message.elements:
        for element in message.elements:
            if element.path and os.path.exists(element.path):
                image_path = element.path
                break

    if not query:
        if image_path:
            query = "Describe this document."
        else:
            return

    # Thinking indicator
    async with cl.Step(name="TriSeva", type="llm") as step:
        step.input = query

        start_time = time.time()
        # Run in thread to avoid blocking the event loop
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

        meta  = DOMAIN_META.get(domain, {"icon": "🤖", "label": domain.title(), "color": "#6b7280"})
        step.output = f"Domain: {meta['icon']} {meta['label']}"

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
    elements = []

    # Main message content
    header  = f"{meta['icon']} **{meta['label']}**"
    content = f"{header}\n\n---\n\n{answer}"

    if disclaimer:
        clean_disc = disclaimer.replace("⚕️", "").replace("⚖️", "").replace("🌾", "").strip()
        content += f"\n\n---\n> 💡 *{clean_disc}*"

    # Send main answer
    msg = cl.Message(content=content)
    await msg.send()



    # ── Sources as separate message (Web Links Only) ─────────────────────────
    if sources:
        web_urls = set()
        for s in sources:
            name = s.get("source")
            url = resolve_web_url(name)
            if url:
                web_urls.add(url)

        if web_urls:
            src_content = "### 🔗 Reference Web Links\n\n"
            for url in sorted(web_urls):
                src_content += f"- [{url}]({url})\n"

            await cl.Message(
                content=src_content,
                parent_id=msg.id,
            ).send()