import warnings
warnings.filterwarnings("ignore")

import os
import sys

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

import signal
import threading


def build_graph():
    graph = StateGraph(TriSevaState)

    # ── Register nodes ────────────────────────────────────────────
    graph.add_node("memory_read",  memory_read_node)
    graph.add_node("image_processing", image_processing_node)
    graph.add_node("orchestrator", orchestrator_node)
    graph.add_node("health_agent", health_agent_node)
    graph.add_node("legal_agent",  legal_agent_node)
    graph.add_node("agri_agent",   agri_agent_node)
    graph.add_node("critic",       critic_node)
    graph.add_node("memory_write", memory_write_node)

    # ── Entry point ───────────────────────────────────────────────
    graph.set_entry_point("memory_read")

    # ── Fixed edges ───────────────────────────────────────────────
    graph.add_edge("memory_read",  "image_processing")
    graph.add_edge("image_processing", "orchestrator")
    graph.add_edge("health_agent", "critic")
    graph.add_edge("legal_agent",  "critic")
    graph.add_edge("agri_agent",   "critic")
    graph.add_edge("memory_write", END)

    # ── Conditional: orchestrator → domain agent ──────────────────
    graph.add_conditional_edges(
        "orchestrator",
        route_to_agent,
        {
            "health":    "health_agent",
            "legal":     "legal_agent",
            "agriculture": "agri_agent",
        }
    )

    # ── Conditional: critic → retry or approve ────────────────────
    graph.add_conditional_edges(
        "critic",
        should_retry,
        {
            "retry":   "orchestrator",
            "approve": "memory_write",
        }
    )

    memory = MemorySaver()
    return graph.compile(checkpointer=memory)


triseva = build_graph()


def ask(query: str, session_id: str = "default", image_path: str = None, domain_override: str = None):
    """Main entry point to query TriSeva."""
    import time
    start_time = time.time()
    config = {"configurable": {"thread_id": session_id}}

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

    thread = threading.Thread(target=run)
    thread.start()
    thread.join(timeout=300)  # 300 second timeout

    if thread.is_alive():
        return {
            "answer":     "Request timed out. Please try again.",
            "domain":     "unknown",
            "score":      None,
            "disclaimer": None,
            "quiz":       [],
            "sources":    [],
            "telemetry":  {},
        }

    if error[0]:
        raise error[0]

    r = result[0]
    latency = time.time() - start_time
    telemetry = r.get("telemetry") or {}
    telemetry["latency"] = round(latency, 2)
    telemetry["retries"] = r.get("retry_count", 0)

    return {
        "answer":      r.get("final_answer") or r.get("draft_answer"),
        "domain":      r.get("domain"),
        "score":       r.get("faithfulness_score"),
        "disclaimer":  r.get("domain_disclaimer"),
        "quiz":        r.get("quiz"),
        "sources":     r.get("sources", []),
        "telemetry":   telemetry,
    }


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("TRISEVA — FULL END-TO-END TEST")
    print("=" * 60)

    test_queries = [
        ("health",    "My haemoglobin is 10.2 g/dL. Is this normal?"),
        ("legal",     "Am I eligible for PM Kisan if I own 1.5 hectares of land?"),
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