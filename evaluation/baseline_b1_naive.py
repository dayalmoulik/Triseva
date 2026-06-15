import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["LANGCHAIN_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from tools.rag_tool import retrieve

# Use same LLM configuration as specialist agents for fair comparison
llm = get_llm(temperature=0.2, max_tokens=1024)

RAG_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a document QA assistant. Answer the user's question using the provided context.
If the answer is not in the context, say "I cannot find the answer in the provided documents." and try to answer with general knowledge while clearly stating that it is general knowledge."""),
    ("human", """Context:
{context}

Question: {query}"""),
])

rag_chain = RAG_PROMPT | llm

def retrieve_cross_domain(query: str, n_results: int = 3) -> list:
    """Retrieve top-n relevant chunks from ALL collections merged and sorted by distance score."""
    all_chunks = []
    for domain in ["health", "legal", "agriculture"]:
        try:
            chunks = retrieve(query, domain=domain, n_results=n_results)
            all_chunks.extend(chunks)
        except Exception as e:
            print(f"  [B1 Error] Failed to retrieve from {domain}: {e}")
            
    # Sort all retrieved chunks by distance score (relevance) descending
    all_chunks.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return all_chunks[:n_results]

def ask_b1_naive(query: str) -> dict:
    """B1 Baseline: Query all collections, merge and pick top chunks, execute direct generation."""
    try:
        chunks = retrieve_cross_domain(query, n_results=3)
        
        context = ""
        if chunks:
            for idx, c in enumerate(chunks, 1):
                context += f"[Doc {idx} - Domain: {c['domain']} - Source: {c['source']}]: {c['text']}\n\n"
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
            "sources": [{"source": c["source"], "score": c["score"], "domain": c["domain"]} for c in chunks] if chunks else []
        }
    except Exception as e:
        print(f"  [B1 Run Error]: {e}")
        return {
            "answer": "An error occurred during B1 Baseline RAG processing.",
            "retrieved_chunks": [],
            "sources": []
        }

if __name__ == "__main__":
    test_query = "What is the normal haemoglobin level for adult men?"
    print(f"Testing B1 Baseline (Naive RAG - Cross Domain) on: '{test_query}'")
    res = ask_b1_naive(test_query)
    print(f"\nRetrieved Domains: {[s['domain'] for s in res['sources']]}")
    print(f"Answer:\n{res['answer']}")
