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

def ask_naive_rag(query: str, domain: str = "health") -> dict:
    """Standard single-agent Naive RAG pipeline."""
    try:
        # Retrieve chunks (top-3 like specialist agents)
        chunks = retrieve(query, domain=domain, n_results=3)
        
        context = ""
        if chunks:
            for idx, c in enumerate(chunks, 1):
                context += f"[Doc {idx}]: {c['text']}\n\n"
        else:
            context = "No relevant context found."

        response = rag_chain.invoke({
            "context": context,
            "query": query,
        })
        
        answer = response.content
        if isinstance(answer, list):
            answer = " ".join(
                b.get("text", "") for b in answer
                if isinstance(b, dict) and b.get("type") == "text"
            )

        return {
            "answer": answer,
            "retrieved_chunks": [c["text"] for c in chunks] if chunks else [],
            "sources": [{"source": c["source"], "score": c["score"]} for c in chunks] if chunks else []
        }
    except Exception as e:
        print(f"  [Naive RAG Error]: {e}")
        return {
            "answer": "An error occurred during Naive RAG processing.",
            "retrieved_chunks": [],
            "sources": []
        }

if __name__ == "__main__":
    test_query = "What is the normal haemoglobin level for adult men?"
    print(f"Testing Naive RAG on query: '{test_query}'")
    res = ask_naive_rag(test_query, domain="health")
    print(f"\nAnswer:\n{res['answer']}")
