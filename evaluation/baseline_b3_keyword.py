import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["LANGCHAIN_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

from dotenv import load_dotenv
load_dotenv()

import chromadb
from rank_bm25 import BM25Okapi
from agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate

# Initialize same LLM
llm = get_llm(temperature=0.2, max_tokens=1024)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document QA assistant. Answer the user's question using the provided context.
If the answer is not in the context, say "I cannot find the answer in the provided documents." and try to answer with general knowledge while clearly stating that it is general knowledge."""),
    ("human", """Context:
{context}

Question: {query}"""),
])

rag_chain = RAG_PROMPT | llm

# Cached indices
_bm25_indices = {}
_corpus_chunks = {}

def get_bm25_index(domain: str):
    """Retrieve all chunks from ChromaDB for a domain and build/cache a BM25 index."""
    if domain not in _bm25_indices:
        client = chromadb.PersistentClient(path="data/chromadb")
        try:
            collection = client.get_collection(name=f"triseva_{domain}")
            data = collection.get(include=["documents", "metadatas"])
        except Exception as e:
            print(f"  [B3 Error] Failed to load collection for {domain}: {e}")
            return None, []

        documents = data.get("documents", [])
        metadatas = data.get("metadatas", [])
        ids = data.get("ids", [])

        chunks = []
        tokenized_corpus = []
        for doc, meta, cid in zip(documents, metadatas, ids):
            text = doc
            if text.startswith("passage: "):
                text = text[len("passage: "):]

            chunks.append({
                "text": text,
                "source": meta.get("source", "unknown"),
                "domain": domain,
                "chunk_id": cid
            })
            # Tokenize document (simple whitespace split + lowercase)
            tokenized_corpus.append(text.lower().split())

        if not chunks:
            return None, []

        bm25 = BM25Okapi(tokenized_corpus)
        _bm25_indices[domain] = bm25
        _corpus_chunks[domain] = chunks

    return _bm25_indices[domain], _corpus_chunks[domain]

def retrieve_bm25(query: str, domain: str, n_results: int = 3) -> list:
    """Query BM25 index for the top-n results."""
    bm25, chunks = get_bm25_index(domain)
    if not bm25 or not chunks:
        return []

    tokenized_query = query.lower().split()
    scores = bm25.get_scores(tokenized_query)
    
    # Pair scores with chunks and sort descending
    paired = list(zip(scores, chunks))
    paired.sort(key=lambda x: x[0], reverse=True)

    retrieved = []
    for score, chunk in paired[:n_results]:
        retrieved.append({
            "text": chunk["text"],
            "source": chunk["source"],
            "domain": chunk["domain"],
            "score": round(float(score), 4)
        })
    return retrieved

def ask_b3_keyword(query: str, domain: str = "health") -> dict:
    """B3 Baseline: Retrieve via BM25 keyword search, then generate answer."""
    try:
        chunks = retrieve_bm25(query, domain=domain, n_results=3)
        
        context = ""
        if chunks:
            for idx, c in enumerate(chunks, 1):
                context += f"[Doc {idx} - Source: {c['source']}]: {c['text']}\n\n"
        else:
            context = "No relevant context found."

        response = rag_chain.invoke({
            "context": context,
            "query": query,
        })
        
        answer = response.content
        return {
            "answer": answer,
            "retrieved_chunks": [c["text"] for c in chunks] if chunks else [],
            "sources": [{"source": c["source"], "score": c["score"]} for c in chunks] if chunks else []
        }
    except Exception as e:
        print(f"  [B3 Run Error]: {e}")
        return {
            "answer": "An error occurred during B3 BM25 RAG processing.",
            "retrieved_chunks": [],
            "sources": []
        }

if __name__ == "__main__":
    test_query = "What is the normal haemoglobin level for adult men?"
    print(f"Testing B3 Baseline (Keyword Search via BM25) on: '{test_query}'")
    res = ask_b3_keyword(test_query, domain="health")
    print(f"Answer:\n{res['answer']}")
