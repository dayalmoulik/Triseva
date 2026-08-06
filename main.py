"""
TriSeva Multi-Agent Orchestration Main Pipeline.

This module builds, compiles, and exposes the primary StateGraph for the TriSeva multi-agent AI system.
It routes user queries through memory retrieval, multimodal document processing, query translation,
intent classification & domain routing, specialist execution (Healthcare, Legal, Agriculture),
critic evaluation, back-translation, and persistent state logging.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Configure UTF-8 standard output encoding for cross-platform log consistency
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

os.environ["LANGCHAIN_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from agents.state import TriSevaState
from agents.memory import memory_read_node, memory_write_node
from agents.orchestrator import orchestrator_node, route_to_agent
from agents.health_agent import health_agent_node
from agents.legal_agent import legal_agent_node
from agents.agri_agent import agri_agent_node
from agents.critic import critic_node, should_retry
from agents.multimodal import image_processing_node
from agents.translation_nodes import translation_pre_node, translation_post_node

import signal
import threading


def build_graph():
    """Constructs and compiles the TriSeva LangGraph multi-agent execution pipeline.

    Registers pipeline nodes (memory, OCR, translation, routing, domain specialists, critic),
    defines static edges, registers conditional routing branches, and attaches memory checkpointing.

    Returns:
        CompiledStateGraph: The compiled state graph instance ready for invocation.
    """
    graph = StateGraph(TriSevaState)

    # ── Register Pipeline Nodes ────────────────────────────────────────────
    graph.add_node("memory_read",      memory_read_node)
    graph.add_node("image_processing", image_processing_node)
    graph.add_node("orchestrator",     orchestrator_node)
    graph.add_node("health_agent",     health_agent_node)
    graph.add_node("legal_agent",      legal_agent_node)
    graph.add_node("agri_agent",       agri_agent_node)
    graph.add_node("critic",           critic_node)
    graph.add_node("translation_pre",  translation_pre_node)
    graph.add_node("translation_post", translation_post_node)
    graph.add_node("memory_write",     memory_write_node)

    # ── Pipeline Entry Point ───────────────────────────────────────────────
    graph.set_entry_point("memory_read")

    # ── Fixed Flow Edges ───────────────────────────────────────────────────
    graph.add_edge("memory_read",      "image_processing")
    graph.add_edge("image_processing", "translation_pre")
    graph.add_edge("translation_pre",  "orchestrator")
    graph.add_edge("health_agent",     "critic")
    graph.add_edge("legal_agent",      "critic")
    graph.add_edge("agri_agent",       "critic")
    graph.add_edge("translation_post", "memory_write")
    graph.add_edge("memory_write",     END)

    # ── Conditional Edge 1: Orchestrator → Domain Specialist ─────────────
    graph.add_conditional_edges(
        "orchestrator",
        route_to_agent,
        {
            "health":      "health_agent",
            "legal":       "legal_agent",
            "agriculture": "agri_agent",
        }
    )

    # ── Conditional Edge 2: Critic → Retry Loop or Approval ───────────────
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry":   "orchestrator",
            "approve": "translation_post",
        }
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


# Compile single shared graph instance at module load
triseva = build_graph()


from langchain_core.callbacks import BaseCallbackHandler

class RunIDCallbackHandler(BaseCallbackHandler):
    """LangChain callback handler to capture execution run IDs for telemetry logging."""

    def __init__(self):
        super().__init__()
        self.run_id = None
        
    def on_chain_start(self, serialized, inputs, *, run_id, **kwargs):
        """Captures the root execution run ID when graph execution starts."""
        if self.run_id is None:
            self.run_id = str(run_id)


def ask(query: str, session_id: str = "default", image_path: str = None, domain_override: str = None) -> dict:
    """Primary execution entry point to process citizen queries through TriSeva.

    Args:
        query (str): Natural language citizen question (in English, Hindi, or Hinglish).
        session_id (str, optional): Unique session tracking identifier for memory persistence. Defaults to "default".
        image_path (str, optional): File path to an uploaded document image (prescription, land record, soil card). Defaults to None.
        domain_override (str, optional): Explicit user domain override ("health", "legal", "agriculture"). Defaults to None.

    Returns:
        dict: Response dictionary containing 'answer', 'domain', 'score', 'disclaimer', 'quiz', 'sources', and 'telemetry'.
    """
    import time
    start_time = time.time()
    
    handler = RunIDCallbackHandler()
    config = {
        "configurable": {"thread_id": session_id},
        "callbacks": [handler]
    }

    result = [None]
    error  = [None]

    def run():
        try:
            result[0] = triseva.invoke(
                {
                    "user_query":         query,
                    "session_id":         session_id,
                    "image_path":         image_path,
                    "domain_override":    domain_override,
                    "retry_count":        0,
                    "chat_history":       [],
                    "domain":             None,
                    "domain_confidence":  None,
                    "retrieved_chunks":   [],
                    "sources":            [],
                    "image_text":         None,
                    "draft_answer":       None,
                    "final_answer":       None,
                    "faithfulness_score": None,
                    "relevancy_score":    None,
                    "domain_disclaimer":  None,
                    "quiz":               None,
                    "telemetry": {
                        "routing_hops": [],
                        "retries": 0,
                        "latency": 0.0,
                        "is_fallback_routing": False,
                        "fallback_routing_method": None,
                        "is_fallback_critic": False,
                        "is_fallback_retrieval": False,
                    },
                },
                config=config,
            )
        except Exception as e:
            error[0] = e

    # Launch graph invocation inside a threaded timeout container (300 seconds limit)
    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=300)

    if thread.is_alive():
        return {
            "answer":     "Request timed out. Please try again.",
            "domain":     "unknown",
            "score":      None,
            "disclaimer": None,
            "quiz":       [],
            "sources":    [],
            "telemetry":  {},
            "run_id":     None,
        }

    if error[0]:
        raise error[0]

    r = result[0]
    latency = time.time() - start_time
    telemetry = r.get("telemetry") or {}
    telemetry["latency"] = round(latency, 2)
    telemetry["retries"] = r.get("retry_count", 0)

    raw_sources = r.get("sources", [])
    answer_text = r.get("final_answer") or r.get("draft_answer") or ""

    from agents.utils import append_source_links, filter_representative_sources, is_answer_not_found

    # Suppress sources if the query resulted in a 'not found' response
    if is_answer_not_found(answer_text):
        sources = []
    else:
        sources = filter_representative_sources(answer_text, raw_sources, r.get("retrieved_chunks"))

    answer_text = append_source_links(answer_text, sources)
    disclaimer_text = r.get("domain_disclaimer")

    return {
        "answer":           answer_text,
        "domain":           r.get("domain"),
        "score":            r.get("faithfulness_score"),
        "disclaimer":       disclaimer_text,
        "quiz":             r.get("quiz"),
        "sources":          sources,
        "retrieved_chunks": r.get("retrieved_chunks", []),
        "telemetry":        telemetry,
        "run_id":           handler.run_id,
        "original_query":   r.get("original_query", query),
        "english_query":    r.get("user_query", query),
        "draft_answer":     r.get("draft_answer", ""),
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRISEVA — FULL END-TO-END TEST")
    print("=" * 60)

    test_queries = [
        ("health",      "My haemoglobin is 10.2 g/dL. Is this normal?"),
        ("legal",       "Am I eligible for PM Kisan if I own 1.5 hectares of land?"),
        ("agriculture", "What support is available for small farmers during crop loss?"),
    ]

    for expected_domain, query in test_queries:
        print(f"\n{'-' * 60}")
        print(f"Query: {query}")
        print(f"{'-' * 60}")

        result = ask(query, session_id=f"test-{expected_domain}")

        print(f"\nDomain:      {result['domain']}")
        print(f"Faith score: {result['score']}")
        print(f"\nAnswer:\n{result['answer']}")

        if result.get("disclaimer"):
            print(f"\n{result['disclaimer']}")

        if result.get("quiz"):
            print(f"\n[QUIZ] Quiz: {len(result['quiz'])} questions generated")

    print("\n" + "=" * 60)
    print("[SUCCESS] Full end-to-end test complete")