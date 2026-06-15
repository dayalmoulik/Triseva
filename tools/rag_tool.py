import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or os.getenv("HF_Token") or ""

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from typing import List

# ── ChromaDB connection ───────────────────────────────────────────────────────
CHROMA_PATH = "data/chromadb"
EMBED_MODEL  = "intfloat/multilingual-e5-base"

_client = None
_ef     = None

def _get_client():
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=CHROMA_PATH)
    return _client

def _get_ef():
    global _ef
    if _ef is None:
        _ef = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name=EMBED_MODEL
        )
    return _ef


# ── Core retrieval ────────────────────────────────────────────────────────────
def retrieve(query: str, domain: str, n_results: int = 5) -> List[dict]:
    """Retrieve top-n relevant chunks — with e5 query prefix for cross-lingual."""
    client = _get_client()
    ef     = _get_ef()

    collection = client.get_or_create_collection(
        name=f"triseva_{domain}",
        embedding_function=ef,
    )

    if collection.count() == 0:
        print(f"  [RAG] Warning: collection 'triseva_{domain}' is empty")
        return []

    # multilingual-e5 requires "query: " prefix at retrieval time
    prefixed_query = f"query: {query}"

    results = collection.query(
        query_texts=[prefixed_query],
        n_results=min(n_results, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        # Strip "passage: " prefix before returning to agents
        text = doc
        if text.startswith("passage: "):
            text = text[len("passage: "):]
        chunks.append({
            "text":   text,
            "source": results["metadatas"][0][i].get("source", "unknown"),
            "domain": domain,
            "score":  round(1 - results["distances"][0][i], 4),
        })

    return chunks


# ── LangChain tool wrappers ───────────────────────────────────────────────────
@tool
def health_rag_tool(query: str) -> str:
    """Search the healthcare knowledge base for medical conditions,
    lab results, medications, symptoms, and treatments."""
    chunks = retrieve(query, domain="health", n_results=5)
    if not chunks:
        return "No relevant health information found in the knowledge base."
    result = ""
    for i, chunk in enumerate(chunks, 1):
        result += f"[Source {i}: {chunk['source']} | Relevance: {chunk['score']}]\n"
        result += f"{chunk['text']}\n\n"
    return result.strip()


@tool
def legal_rag_tool(query: str) -> str:
    """Search the legal and government schemes knowledge base for
    welfare schemes, eligibility criteria, RTI, and legal rights."""
    chunks = retrieve(query, domain="legal", n_results=5)
    if not chunks:
        return "No relevant legal information found in the knowledge base."
    result = ""
    for i, chunk in enumerate(chunks, 1):
        result += f"[Source {i}: {chunk['source']} | Relevance: {chunk['score']}]\n"
        result += f"{chunk['text']}\n\n"
    return result.strip()


@tool
def agriculture_rag_tool(query: str) -> str:
    """Search the agriculture knowledge base for Indian farming schemes,
    crop advisories, irrigation, and farm support programs."""
    chunks = retrieve(query, domain="agriculture", n_results=5)
    if not chunks:
        return "No relevant agriculture information found in the knowledge base."
    result = ""
    for i, chunk in enumerate(chunks, 1):
        result += f"[Source {i}: {chunk['source']} | Relevance: {chunk['score']}]\n"
        result += f"{chunk['text']}\n\n"
    return result.strip()