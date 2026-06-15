import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

from agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from agents.state import TriSevaState

FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7
MAX_RETRIES = 2

llm = None

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a reflective quality assurance critic for TriSeva, an Indian document QA system.
Your job is to evaluate if a generated answer is:
1. Faithful: Grounded entirely and strictly in the provided context. If the answer contains ANY facts, numbers, clinical explanations, timelines, eligibility details, or recommendations that are NOT explicitly written in the provided context (even if they are correct in the real world), it is UNFAITHFUL and must be flagged.
2. Relevant: Directly and completely answers the user's question without avoiding the core question or adding irrelevant padding.

You must perform a step-by-step assessment:
- Extract all factual assertions and explanations made in the answer.
- Cross-reference each assertion with the provided context.
- If any assertion, definition, or background explanation is not explicitly supported by the context, mark faithfulness below 0.7.
- If the answer includes formatting sections (such as eligibility, symptoms, how to apply) that are not mentioned in the context, mark faithfulness below 0.7.

Respond with ONLY a JSON object in this exact format:
{{
  "faithfulness": 0.0 to 1.0,
  "relevancy": 0.0 to 1.0,
  "reason": "Specify exactly which statements/claims are not supported by context, or why the relevancy is lacking."
}}

Score guidelines:
- 0.9-1.0: Fully grounded and highly relevant; zero unsupported details.
- 0.7-0.89: Satisfied with minor semantic paraphrasing, but no factual extrapolation.
- 0.5-0.69: Noticeable unsupported facts, general background explanations, or ungrounded warning details.
- 0.0-0.49: Major hallucinations, off-topic, or completely ungrounded assertions."""),
    ("human", """Question: {query}

Context retrieved:
{context}

Answer to evaluate:
{answer}

Evaluate faithfulness and relevancy:"""),
])

critic_chain = None


def fallback_evaluator(query: str, context: str, answer: str) -> dict:
    """Rule-based local fallback calculation for faithfulness and relevancy."""
    import re
    def get_clean_words(text: str) -> set:
        words = re.findall(r'\b\w+\b', text.lower())
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'to', 'of', 'in', 'on', 'at', 'by',
            'for', 'with', 'about', 'against', 'between', 'into', 'through', 'during', 'before', 'after', 'above',
            'below', 'from', 'up', 'down', 'in', 'out', 'off', 'over', 'under', 'again', 'further', 'then', 'once',
            'this', 'that', 'these', 'those', 'it', 'its', 'they', 'them', 'their', 'our', 'we', 'you', 'your',
            'he', 'him', 'his', 'she', 'her', 'i', 'me', 'my', 'myself'
        }
        return {w for w in words if w not in stop_words}

    query_words = get_clean_words(query)
    context_words = get_clean_words(context)
    answer_words = get_clean_words(answer)

    # 1. Faithfulness: what fraction of content words in answer are backed by the context
    if answer_words:
        faithfulness = len(answer_words & context_words) / len(answer_words)
    else:
        faithfulness = 1.0

    # 2. Relevancy: what fraction of query content words are matched in the answer
    if query_words:
        relevancy = len(query_words & answer_words) / len(query_words)
        relevancy = min(relevancy * 1.5, 1.0)  # scale up slightly for synonym gap
    else:
        relevancy = 1.0

    return {
        "faithfulness": round(faithfulness, 2),
        "relevancy": round(relevancy, 2),
        "reason": "Lexical overlap local fallback calculation."
    }


def critic_node(state: TriSevaState) -> dict:
    """Check faithfulness and relevancy of draft answer against retrieved chunks."""
    print(f"  [Critic] Evaluating answer faithfulness and relevancy...")

    draft    = state.get("draft_answer", "")
    chunks   = state.get("retrieved_chunks", [])
    query    = state.get("user_query", "")

    # Load or initialize telemetry
    telemetry = state.get("telemetry") or {
        "routing_hops": [],
        "retries": 0,
        "latency": 0.0,
        "is_fallback_routing": False,
        "fallback_routing_method": None,
        "is_fallback_critic": False,
        "is_fallback_retrieval": False,
    }

    # If no chunks retrieved, skip evaluation
    if not chunks or not draft:
        print(f"  [Critic] No chunks or draft to evaluate — auto-approving")
        return {
            "faithfulness_score": 0.8,
            "relevancy_score": 0.8,
            "final_answer": draft,
            "telemetry": telemetry,
        }

    context = "\n\n".join(chunks[:3])  # use top 3 chunks

    try:
        global llm, critic_chain
        if llm is None:
            llm = get_llm(temperature=0.0, max_tokens=256)
            critic_chain = CRITIC_PROMPT | llm

        response = critic_chain.invoke({
            "query":   query,
            "context": context[:2000],
            "answer":  draft[:1500],
        })

        content = response.content
        if isinstance(content, list):
            content = " ".join(
                b.get("text", "") for b in content
                if isinstance(b, dict) and b.get("type") == "text"
            )

        # Parse JSON response
        import json
        content = content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        evaluation = json.loads(content)
        
        # Support both new and old response format schemas
        f_score = float(evaluation.get("faithfulness", evaluation.get("score", 0.8)))
        r_score = float(evaluation.get("relevancy", 0.8))
        reason  = evaluation.get("reason", "")

        print(f"  [Critic] Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f} | {reason}")

        approved = (f_score >= FAITHFULNESS_THRESHOLD and r_score >= RELEVANCY_THRESHOLD)
        return {
            "faithfulness_score": f_score,
            "relevancy_score": r_score,
            "final_answer": draft if approved else "",
            "telemetry": telemetry,
            "critic_feedback": None if approved else reason,
        }

    except Exception as e:
        print(f"  [Critic] Evaluation error: {str(e)[:80]} — falling back to local validator")
        telemetry["is_fallback_critic"] = True
        fb = fallback_evaluator(query, context, draft)
        f_score = fb["faithfulness"]
        r_score = fb["relevancy"]
        print(f"  [Critic Local Fallback] Faithfulness: {f_score:.2f} | Relevancy: {r_score:.2f}")
        
        reason = fb.get("reason", "Lexical overlap local fallback calculation.")
        approved = (f_score >= FAITHFULNESS_THRESHOLD and r_score >= RELEVANCY_THRESHOLD)
        return {
            "faithfulness_score": f_score,
            "relevancy_score": r_score,
            "final_answer": draft if approved else "",
            "telemetry": telemetry,
            "critic_feedback": None if approved else reason,
        }


def should_retry(state: TriSevaState) -> str:
    """Decide whether to re-query or approve the answer."""
    f_score  = state.get("faithfulness_score", 1.0)
    r_score  = state.get("relevancy_score", 1.0)
    retries  = state.get("retry_count", 0)

    if (f_score < FAITHFULNESS_THRESHOLD or r_score < RELEVANCY_THRESHOLD) and retries < MAX_RETRIES:
        print(f"  [Critic] Score(s) below threshold (Faithfulness: {f_score:.2f}, Relevancy: {r_score:.2f}) — triggering retry #{retries + 1}")
        return "retry"

    print(f"  [Critic] ✅ Answer approved (Faithfulness: {f_score:.2f}, Relevancy: {r_score:.2f})")
    return "approve"