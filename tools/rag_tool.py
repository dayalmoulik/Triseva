import warnings
warnings.filterwarnings("ignore")

import os
import string
from dotenv import load_dotenv
load_dotenv()

os.environ["HF_TOKEN"] = os.getenv("HF_TOKEN") or os.getenv("HF_Token") or ""

import chromadb
from chromadb.utils import embedding_functions
from langchain_core.tools import tool
from typing import List
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# ── ChromaDB connection ───────────────────────────────────────────────────────
CHROMA_PATH = "data/chromadb"
EMBED_MODEL  = "intfloat/multilingual-e5-base"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

_client = None
_ef     = None
_reranker = None

# In-memory BM25 caches: domain -> BM25Okapi / domain -> list of chunk dicts
_bm25_indices = {}
_corpus_chunks = {}

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

def _get_reranker():
    global _reranker
    if _reranker is None:
        print(f"  [RERANKER] Initializing CrossEncoder model: {RERANK_MODEL}...")
        _reranker = CrossEncoder(RERANK_MODEL)
    return _reranker

def _tokenize(text: str) -> List[str]:
    """Helper to tokenize text for BM25 with subword/character n-grams for Indic & Hinglish scripts."""
    words = text.lower().translate(str.maketrans("", "", string.punctuation)).split()
    tokens = list(words)
    # Add subword 3-grams and 4-grams for words >= 4 chars to improve Indic stem matching
    for w in words:
        if len(w) >= 4:
            tokens.extend([w[i:i+3] for i in range(len(w) - 2)])
            tokens.extend([w[i:i+4] for i in range(len(w) - 3)])
    return tokens

def _get_bm25_index(domain: str):
    """Lazy initialize and cache the BM25 index for a domain."""
    global _bm25_indices, _corpus_chunks
    if domain not in _bm25_indices:
        print(f"  [BM25] Building index for 'triseva_{domain}'...")
        client = _get_client()
        collection = client.get_or_create_collection(name=f"triseva_{domain}")
        count = collection.count()
        
        if count == 0:
            print(f"  [BM25] Warning: collection 'triseva_{domain}' is empty.")
            return None, []
            
        # Fetch all records
        res = collection.get(include=["documents", "metadatas"])
        chunks = []
        tokenized_corpus = []
        
        for i, doc in enumerate(res["documents"]):
            text = doc
            if text.startswith("passage: "):
                text = text[len("passage: "):]
                
            chunk = {
                "text": text,
                "source": res["metadatas"][i].get("source", "unknown"),
                "domain": domain,
                "id": res["ids"][i]
            }
            chunks.append(chunk)
            tokenized_corpus.append(_tokenize(text))
            
        _corpus_chunks[domain] = chunks
        _bm25_indices[domain] = BM25Okapi(tokenized_corpus)
        print(f"  [BM25] Completed index for 'triseva_{domain}' with {len(chunks)} chunks.")
        
    return _bm25_indices[domain], _corpus_chunks[domain]


# ── Core hybrid retrieval with RRF and Re-ranking ────────────────────────────────
_expansion_cache = {}

# ── Core hybrid retrieval with RRF and Re-ranking ────────────────────────────────
def expand_query(query: str) -> List[str]:
    """Generate 3 alternative search queries in English to expand context recall."""
    import re
    global _expansion_cache
    
    cleaned_query = query.strip()
    if cleaned_query in _expansion_cache:
        return _expansion_cache[cleaned_query]
        
    try:
        from agents.llm_factory import get_llm
        # Using a low temperature for stable search query generation
        llm = get_llm(temperature=0.0, max_tokens=150)
        if not llm:
            return [query]
            
        prompt = f"""You are a search query expansion assistant. Your task is to generate exactly 3 alternative search queries in English to help retrieve relevant documents from a vector database for the following user question.
        
User Question: {query}

Provide exactly 3 queries, one per line. Do not number them or include any other text."""
        res = llm.invoke(prompt)
        lines = [line.strip() for line in res.content.strip().split("\n") if line.strip()]
        
        # Clean up any bullet points or numbering that the LLM might have introduced
        cleaned_lines = []
        for line in lines:
            line_clean = re.sub(r"^\d+\.\s*|-\s*", "", line).strip()
            if line_clean:
                cleaned_lines.append(line_clean)
                
        if not cleaned_lines:
            return [query]
            
        result = cleaned_lines[:3]
        _expansion_cache[cleaned_query] = result
        return result
    except Exception as e:
        print(f"  [Query Expansion] Failed: {e}. Using original query.")
        return [query]


def expand_context_with_neighbors(collection, chunk_id: str, current_text: str) -> str:
    """Expand retrieved chunk text by joining it with adjacent sequential chunks."""
    try:
        parts = chunk_id.rsplit("_", 1)
        if len(parts) != 2:
            return current_text
        prefix, num_str = parts[0], parts[1]
        num = int(num_str)
        
        neighbor_ids = [
            f"{prefix}_{num-1:04d}",
            f"{prefix}_{num+1:04d}"
        ]
        
        # Query adjacent chunks
        res = collection.get(ids=neighbor_ids)
        if not res or not res.get("documents"):
            return current_text
            
        # Map documents to their ids
        doc_map = dict(zip(res["ids"], res["documents"]))
        
        expanded_docs = []
        # Previous chunk
        prev_id = neighbor_ids[0]
        if prev_id in doc_map:
            doc_text = doc_map[prev_id]
            if doc_text.startswith("passage: "):
                doc_text = doc_text[9:]
            expanded_docs.append(doc_text)
            
        # Current chunk
        curr_text = current_text
        if curr_text.startswith("passage: "):
            curr_text = curr_text[9:]
        expanded_docs.append(curr_text)
        
        # Next chunk
        next_id = neighbor_ids[1]
        if next_id in doc_map:
            doc_text = doc_map[next_id]
            if doc_text.startswith("passage: "):
                doc_text = doc_text[9:]
            expanded_docs.append(doc_text)
            
        return "\n\n".join(expanded_docs)
    except Exception as e:
        # Fallback to current text on any errors
        return current_text


# ── Core hybrid retrieval with RRF and Re-ranking ────────────────────────────────
def retrieve(query: str, domain: str, n_results: int = 7, native_query: str = None) -> List[dict]:
    """Retrieve top-n relevant chunks using Query Expansion and Hybrid Search (E5 + BM25 + RRF + Re-ranking)."""
    import re
    from concurrent.futures import ThreadPoolExecutor
    
    client = _get_client()
    ef     = _get_ef()

    collection = client.get_or_create_collection(
        name=f"triseva_{domain}",
        embedding_function=ef,
    )

    if collection.count() == 0:
        print(f"  [RAG] Collection 'triseva_{domain}' is empty. Auto-building starter knowledge base...")
        try:
            import sys
            root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            if root_dir not in sys.path:
                sys.path.insert(0, root_dir)
            from knowledge_base.build_kb import build_knowledge_base
            build_knowledge_base()
            collection = client.get_or_create_collection(
                name=f"triseva_{domain}",
                embedding_function=ef,
            )
        except Exception as e:
            print(f"  [RAG] Auto-build KB failed: {e}")
            return []

    # Generate expanded queries (3 variations + original)
    expanded_queries = expand_query(query)
    if query not in expanded_queries:
        expanded_queries.insert(0, query)

    # Dual-query enhancement: Include original native Hindi/Hinglish query if provided
    if native_query and native_query.strip() and native_query.strip() != query.strip():
        if native_query.strip() not in expanded_queries:
            expanded_queries.insert(0, native_query.strip())
            print(f"  [RAG Multilingual] Added native query to parallel retrieval: '{native_query.strip()}'")
        
    print(f"  [RAG] Query expansion generated: {expanded_queries}")

    candidate_limit = n_results * 3
    rrf_scores = {} # text_content_lower -> (chunk_dict, combined_rrf_score)

    def get_key(chunk):
        return chunk["text"].strip().lower()

    # Worker function for parallel dense retrievals
    def fetch_dense_for_query(eq):
        prefixed_query = f"query: {eq}"
        try:
            dense_results = collection.query(
                query_texts=[prefixed_query],
                n_results=min(candidate_limit, collection.count()),
                include=["documents", "metadatas", "distances"],
            )
            chunks = []
            if dense_results and "documents" in dense_results and dense_results["documents"]:
                for rank, doc in enumerate(dense_results["documents"][0], 1):
                    text = doc
                    if text.startswith("passage: "):
                        text = text[len("passage: "):]
                    
                    chunks.append({
                        "text": text,
                        "source": dense_results["metadatas"][0][rank-1].get("source", "unknown"),
                        "domain": domain,
                        "score": round(1.0 - float(dense_results["distances"][0][rank-1]), 4),
                        "rank": rank,
                        "id": dense_results["ids"][0][rank-1] if "ids" in dense_results and dense_results["ids"] else None
                    })
            return chunks
        except Exception as e:
            print(f"  [RAG] Dense retrieval error for '{eq}': {e}")
            return []

    # Worker function for parallel sparse retrievals
    bm25_index, corpus_chunks = _get_bm25_index(domain)
    def fetch_sparse_for_query(eq):
        if not bm25_index:
            return []
        try:
            tokenized_query = _tokenize(eq)
            scores = bm25_index.get_scores(tokenized_query)
            top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:candidate_limit]
            
            chunks = []
            rank = 1
            for idx in top_indices:
                if scores[idx] > 0:
                    c = corpus_chunks[idx]
                    chunks.append({
                        "text": c["text"],
                        "source": c["source"],
                        "domain": domain,
                        "score": round(float(scores[idx]), 4),
                        "rank": rank,
                        "id": c.get("id")
                    })
                    rank += 1
            return chunks
        except Exception as e:
            print(f"  [RAG] BM25 retrieval error for '{eq}': {e}")
            return []

    # Run dense and sparse searches in parallel
    dense_futures = []
    sparse_futures = []
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        for eq in expanded_queries:
            dense_futures.append(executor.submit(fetch_dense_for_query, eq))
            if bm25_index:
                sparse_futures.append(executor.submit(fetch_sparse_for_query, eq))
                
        # Wait for all dense searches and compile RRF
        for future in dense_futures:
            chunks = future.result()
            for chunk in chunks:
                key = get_key(chunk)
                score_contrib = 1.0 / (60.0 + chunk["rank"])
                if key in rrf_scores:
                    prev_chunk, prev_score = rrf_scores[key]
                    rrf_scores[key] = (prev_chunk, prev_score + score_contrib)
                else:
                    rrf_scores[key] = (chunk, score_contrib)
                    
        # Wait for all sparse searches and compile RRF
        for future in sparse_futures:
            chunks = future.result()
            for chunk in chunks:
                key = get_key(chunk)
                score_contrib = 1.0 / (60.0 + chunk["rank"])
                if key in rrf_scores:
                    prev_chunk, prev_score = rrf_scores[key]
                    rrf_scores[key] = (prev_chunk, prev_score + score_contrib)
                else:
                    rrf_scores[key] = (chunk, score_contrib)

    # Sort candidates by combined RRF score descending
    sorted_candidates = sorted(rrf_scores.values(), key=lambda item: item[1], reverse=True)
    # Select top candidates to pass to the local Re-Ranker
    rerank_candidates = [item[0] for item in sorted_candidates[:n_results * 2]]

    if not rerank_candidates:
        return []

    # 3. Cross-Encoder Re-ranking (evaluated against the ORIGINAL user query)
    try:
        reranker = _get_reranker()
        pairs = [[query, c["text"]] for c in rerank_candidates]
        rerank_scores = reranker.predict(pairs)
        
        # Update scores with re-ranker outputs
        for idx, score in enumerate(rerank_scores):
            rerank_candidates[idx]["score"] = round(float(score), 4)
            
        # Sort candidates by re-ranker score descending
        final_chunks = sorted(rerank_candidates, key=lambda c: c["score"], reverse=True)[:n_results]
        
        # Expand context with sequential neighbors
        for chunk in final_chunks:
            if chunk.get("id"):
                chunk["text"] = expand_context_with_neighbors(collection, chunk["id"], chunk["text"])
                
        print(f"  [RAG] Hybrid retrieval with Query Expansion and Context Neighbor Expansion succeeded. Returning {len(final_chunks)} chunks for domain '{domain}'")
        return final_chunks
    except Exception as e:
        print(f"  [RAG] Error during re-ranking: {e}. Falling back to RRF rankings.")
        fallback_chunks = rerank_candidates[:n_results]
        for chunk in fallback_chunks:
            if chunk.get("id"):
                chunk["text"] = expand_context_with_neighbors(collection, chunk["id"], chunk["text"])
        return fallback_chunks


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