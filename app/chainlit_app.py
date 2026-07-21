import warnings
warnings.filterwarnings("ignore")

import os
import sys

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

        # Run in thread to avoid blocking the event loop
        result = await cl.make_async(ask)(
            query=query,
            session_id=session_id,
            image_path=image_path,
        )

        answer     = result.get("answer") or "No answer generated."
        domain     = result.get("domain", "unknown")
        score      = result.get("score")
        disclaimer = result.get("disclaimer")
        quiz       = result.get("quiz", [])
        sources    = result.get("sources", [])

        meta  = DOMAIN_META.get(domain, {"icon": "🤖", "label": domain.title(), "color": "#6b7280"})
        step.output = f"Domain: {meta['icon']} {meta['label']} | Faithfulness: {score:.0%}" if score else f"Domain: {meta['icon']} {meta['label']}"

    # ── Build main response ─────────────────────────────────────────────────
    elements = []

    # Score bar as text element
    if score is not None:
        pct        = int(score * 100)
        bar_filled = "█" * (pct // 10)
        bar_empty  = "░" * (10 - pct // 10)
        score_text = f"**Faithfulness** `{bar_filled}{bar_empty}` **{pct}%**"
    else:
        score_text = ""

    # Main message content
    header  = f"{meta['icon']} **{meta['label']}**"
    content = f"{header}\n{score_text}\n\n---\n\n{answer}"

    if disclaimer:
        content += f"\n\n---\n> {disclaimer}"

    # Send main answer
    msg = cl.Message(content=content)
    await msg.send()



    # ── Sources as separate message ─────────────────────────────────────────
    if sources:
        # Deduplicate sources by filename, keeping the highest relevance score
        deduped = {}
        for s in sources:
            name = s.get("source")
            score = s.get("score", 0.0)
            if name:
                if name not in deduped or score > deduped[name]:
                    deduped[name] = score

        src_content = "### 📎 Retrieved Sources\n\n"
        # Sort sources by score descending
        for name, score in sorted(deduped.items(), key=lambda x: x[1], reverse=True):
            pct = int(score * 100)
            src_content += f"- `{name}` — **{pct}%** relevance\n"

        await cl.Message(
            content=src_content,
            parent_id=msg.id,
        ).send()