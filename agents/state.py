from typing import TypedDict, Optional, List, Annotated
from langgraph.graph.message import add_messages


class TriSevaState(TypedDict):
    # ── User input ────────────────────────────────────────────────
    user_query: str                        # raw query from user
    session_id: str                        # unique session identifier
    image_path: Optional[str]              # path to uploaded document image

    # ── Routing ───────────────────────────────────────────────────
    domain: Optional[str]                  # health / legal / agriculture / multi
    domain_confidence: Optional[float]     # router confidence score

    # ── Retrieval ─────────────────────────────────────────────────
    retrieved_chunks: Optional[List[str]]  # top-k chunks from ChromaDB
    sources: Optional[List[dict]]          # chunk metadata (source, page, score)
    image_text: Optional[str]             # extracted text from document image

    # ── Generation ────────────────────────────────────────────────
    draft_answer: Optional[str]            # answer from specialist agent
    final_answer: Optional[str]            # answer approved by critic

    # ── Critic ────────────────────────────────────────────────────
    faithfulness_score: Optional[float]    # 0.0 to 1.0
    relevancy_score: Optional[float]       # 0.0 to 1.0
    retry_count: int                       # tracks re-query loops (max 2)

    # ── Memory ────────────────────────────────────────────────────
    chat_history: Annotated[list, add_messages]  # full conversation history

    # ── Domain quiz ───────────────────────────────────────────────
    quiz: Optional[List[dict]]             # [{question, options, answer}]

    # ── Metadata ──────────────────────────────────────────────────
    domain_disclaimer: Optional[str]       # safety note based on domain
    domain_override: Optional[str]         # user override choice for routing
    telemetry: Optional[dict]              # dict containing metrics telemetry
    critic_feedback: Optional[str]         # feedback reason from the critic on failure