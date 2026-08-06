"""
TriSeva Natural Language Inference (NLI) Quality Assurance Critic Node.

Implements a gated multi-stage quality assurance verification pipeline evaluating:
1. Faithfulness Score (0.0 to 1.0): Factual grounding against retrieved document passages.
2. Relevancy Score (0.0 to 1.0): Direct query alignment and helpfulness.

Uses NLI lexical overlap pre-filtering, primary Indic LLM judging (Sarvam 105B),
secondary agreement verification for borderline scores (Claude Haiku), and rule-based fallback.
"""

import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
import json
from dotenv import load_dotenv
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

from agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate
from agents.state import TriSevaState
from agents.utils import initialize_telemetry

FAITHFULNESS_THRESHOLD = 0.7
RELEVANCY_THRESHOLD = 0.7
MAX_RETRIES = 2

llm = None

CRITIC_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are an expert quality assurance critic for TriSeva, an Indian multilingual document QA system.
Your job is to evaluate if a generated answer is:
1. Faithfulness (0.0 to 1.0): Are all key factual claims, numbers, eligibility criteria, and details in the answer semantically grounded in the provided context?
   - MULTILINGUAL RULE: The context may be in English while the question/answer is in Hindi, Devanagari, or Hinglish. Translate and evaluate semantic factual equivalence. Accurately translated or synthesized facts are FAITHFUL (0.9 - 1.0).
   - Only penalize (mark < 0.5) if the answer makes direct factual contradictions or fabricates specific numbers/rules not in the context.
2. Answer Relevancy (0.0 to 1.0): Does the answer directly and helpful answer the user's question?

Respond ONLY with a JSON object in this format:
{{
  "faithfulness": 0.0 to 1.0,
  "relevancy": 0.0 to 1.0,
  "reason": "Brief summary of factual grounding and relevancy alignment."
}}

CRITICAL: Output ONLY valid raw JSON with no backticks, no markdown, and no extra text."""),
    ("human", """Question: {query}

Context retrieved:
{context}

Answer to evaluate:
{answer}

Evaluate faithfulness and relevancy:"""),
])

critic_chain = None


def fallback_evaluator(query: str, context: str, answer: str) -> dict:
    """Rule-based lexical overlap fallback calculation when LLM judges are unreachable.

    Args:
        query (str): Citizen input query.
        context (str): Combined retrieved context text.
        answer (str): Generated draft response text.

    Returns:
        dict: Dictionary with 'faithfulness' (float), 'relevancy' (float), and 'reason' (str).
    """
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

    # 1. Faithfulness: fraction of content words in answer backed by context
    if answer_words:
        faithfulness = len(answer_words & context_words) / len(answer_words)
    else:
        faithfulness = 1.0

    # 2. Relevancy: fraction of query content words matched in answer
    if query_words:
        relevancy = len(query_words & answer_words) / len(query_words)
        relevancy = min(relevancy * 1.5, 1.0)
    else:
        relevancy = 1.0

    return {
        "faithfulness": round(faithfulness, 2),
        "relevancy": round(relevancy, 2),
        "reason": "Lexical overlap local fallback calculation."
    }


def get_judge_llm(provider: str):
    """Instantiates a specific LLM judge model instance.

    Args:
        provider (str): Judge provider key ('sarvam' or 'haiku').

    Returns:
        BaseChatModel: Configured LLM model instance or None.
    """
    import os
    if provider == "haiku":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        return ChatAnthropic(
            model="claude-haiku-4-5-20251001",
            api_key=api_key,
            temperature=0.0,
            max_tokens=256,
            timeout=30,
        )
    elif provider == "sarvam":
        from langchain_openai import ChatOpenAI
        api_key = os.getenv("SARVAM_API_KEY")
        return ChatOpenAI(
            model="sarvam-105b",
            openai_api_key=api_key,
            openai_api_base="https://api.sarvam.ai/v1",
            temperature=0.0,
            max_tokens=256,
            timeout=30,
        )
    return None


def calculate_nli_overlap(context: str, answer: str) -> float:
    """Calculates rule-based lexical overlap representing factual NLI entailment.

    Args:
        context (str): Retrieved context passages.
        answer (str): Draft answer text.

    Returns:
        float: Lexical overlap score (0.0 to 1.0).
    """
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
        
    context_words = get_clean_words(context)
    answer_words = get_clean_words(answer)
    
    if not answer_words:
        return 1.0

    # Pass-through Indic non-ASCII text to LLM judge (return 0.50 neutral)
    if any(ord(c) > 127 for c in answer):
        return 0.50
        
    overlap = len(answer_words & context_words) / len(answer_words)
    return overlap


def _parse_critic_json(content) -> dict:
    """Parses JSON output from LLM judges robustly.

    Args:
        content (str | list): Raw response content from LLM judge.

    Returns:
        dict: Parsed evaluation dictionary with 'faithfulness', 'relevancy', and 'reason'.
    """
    import json
    if isinstance(content, list):
        content = " ".join(
            b.get("text", "") for b in content
            if isinstance(b, dict) and b.get("type") == "text"
        )
    content = content.strip()
    if "```" in content:
        content = content.split("```")[1]
        if content.startswith("json"):
            content = content[4:]
    return json.loads(content)


def critic_node(state: TriSevaState) -> dict:
    """Evaluates draft answer faithfulness and relevancy using a gated multi-stage Critic pipeline.

    Passes draft responses through:
    1. Direct document summary auto-approval (0.95 confidence).
    2. NLI Lexical Overlap pre-filter (auto-approves >=0.92, escalates <=0.05).
    3. Primary Indic LLM Judge (Sarvam-105B).
    4. Borderline score secondary panel verification (Claude Haiku).
    5. Rule-based lexical fallback on model errors.

    Args:
        state (TriSevaState): Pipeline state containing 'draft_answer', 'retrieved_chunks', and 'user_query'.

    Returns:
        dict: Updated state dictionary containing 'faithfulness_score', 'relevancy_score', 'final_answer', 'telemetry', and 'critic_feedback'.
    """
    print(f"  [Critic] Evaluating answer faithfulness and relevancy...")

    draft    = state.get("draft_answer", "")
    chunks   = state.get("retrieved_chunks", [])
    query    = state.get("user_query", "")

    # Load or initialize telemetry
    telemetry = initialize_telemetry(state)

    eval_chunks = []
    if state.get("image_text"):
        eval_chunks.append(f"[Document Context]\n{state['image_text']}")
    if chunks:
        eval_chunks.extend(chunks)

    # If direct document summary (from uploaded image/file), auto-approve with high confidence
    if state.get("image_text") and draft:
        print(f"  [Critic Direct Document] Direct document summary present — auto-approving (Faithfulness: 0.95, Relevancy: 0.95)")
        return {
            "faithfulness_score": 0.95,
            "relevancy_score": 0.95,
            "final_answer": draft,
            "telemetry": telemetry,
            "critic_feedback": None,
        }

    # If no chunks retrieved/evaluable, skip evaluation
    if not eval_chunks or not draft:
        print(f"  [Critic] No chunks or draft to evaluate — auto-approving")
        return {
            "faithfulness_score": 0.8,
            "relevancy_score": 0.8,
            "final_answer": draft,
            "telemetry": telemetry,
        }

    context = "\n\n".join(eval_chunks)

    # 1. Calculate NLI Lexical Overlap Pre-filter
    nli_score = calculate_nli_overlap(context, draft)
    telemetry["nli_prefilter_score"] = nli_score

    if nli_score < 0.05:
        print(f"  [Critic NLI Pre-filter] Low lexical overlap detected (overlap: {nli_score:.2f}). Escalating to Dual-Judge LLM evaluation...")
        telemetry["nli_prefilter_action"] = "escalate_to_judge"
    elif nli_score > 0.92:
        print(f"  [Critic NLI Pre-filter] High entailment detected (overlap: {nli_score:.2f}). Auto-approving.")
        telemetry["nli_prefilter_action"] = "auto_approve"
        telemetry["dual_judge_triggered"] = False
        telemetry["judge_disagreement"] = 0.0
        return {
            "faithfulness_score": 0.95,
            "relevancy_score": 0.95,
            "final_answer": draft,
            "telemetry": telemetry,
            "critic_feedback": None,
        }

    # Proceed to LLM Judges
    telemetry["nli_prefilter_action"] = "llm_judging"
    dual_judge_triggered = False
    judge_disagreement = 0.0

    try:
        # Run Judge 1: Sarvam-105B (Primary Indic Judge)
        sarvam_llm = get_judge_llm("sarvam")
        sarvam_chain = CRITIC_PROMPT | sarvam_llm
        
        response = sarvam_chain.invoke({
            "query":   query,
            "context": context[:20000],
            "answer":  draft[:2000],
        })

        evaluation = _parse_critic_json(response.content)
        f_sarvam = float(evaluation.get("faithfulness", evaluation.get("score", 0.8)))
        r_sarvam = float(evaluation.get("relevancy", 0.8))
        reason = evaluation.get("reason", "")
        
        print(f"  [Critic Judge 1 (Sarvam-105B)] Faithfulness: {f_sarvam:.2f} | Relevancy: {r_sarvam:.2f} | {reason}")

        # Check if score falls in borderline band
        if 0.5 <= f_sarvam < 0.85:
            dual_judge_triggered = True
            print(f"  [Critic] Borderline score detected ({f_sarvam:.2f}). Triggering Judge 2 (Claude Haiku)...")
            
            haiku_llm = get_judge_llm("haiku")
            haiku_chain = CRITIC_PROMPT | haiku_llm
            
            response_haiku = haiku_chain.invoke({
                "query":   query,
                "context": context[:20000],
                "answer":  draft[:2000],
            })
            
            eval_haiku = _parse_critic_json(response_haiku.content)
            f_haiku = float(eval_haiku.get("faithfulness", eval_haiku.get("score", 0.8)))
            r_haiku = float(eval_haiku.get("relevancy", 0.8))
            reason_haiku = eval_haiku.get("reason", "")
            
            print(f"  [Critic Judge 2 (Haiku)] Faithfulness: {f_haiku:.2f} | Relevancy: {r_haiku:.2f} | {reason_haiku}")
            
            # Resolve scores conservatively (minimum)
            f_score = min(f_sarvam, f_haiku)
            r_score = min(r_sarvam, r_haiku)
            judge_disagreement = abs(f_sarvam - f_haiku)
            reason = f"Sarvam: {reason} | Haiku: {reason_haiku}"
        else:
            # High confidence score: accept Sarvam's judgment
            f_score = f_sarvam
            r_score = r_sarvam
            
        telemetry["dual_judge_triggered"] = dual_judge_triggered
        telemetry["judge_disagreement"] = judge_disagreement

        approved = (f_score >= FAITHFULNESS_THRESHOLD and r_score >= RELEVANCY_THRESHOLD)
        return {
            "faithfulness_score": f_score,
            "relevancy_score": r_score,
            "final_answer": draft if approved else "",
            "telemetry": telemetry,
            "critic_feedback": None if approved else reason,
        }

    except Exception as e:
        print(f"  [Critic] LLM Judge Error: {str(e)[:80]} — falling back to local validator")
        telemetry["is_fallback_critic"] = True
        telemetry["dual_judge_triggered"] = False
        telemetry["judge_disagreement"] = 0.0
        
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
    """Determines whether state graph should route to Orchestrator retry loop or approve answer.

    Args:
        state (TriSevaState): Pipeline state containing faithfulness and relevancy scores and retry count.

    Returns:
        str: Edge name ('retry' or 'approve').
    """
    f_score  = state.get("faithfulness_score", 1.0)
    r_score  = state.get("relevancy_score", 1.0)
    retries  = state.get("retry_count", 0)

    if (f_score < FAITHFULNESS_THRESHOLD or r_score < RELEVANCY_THRESHOLD) and retries < MAX_RETRIES:
        print(f"  [Critic] Score(s) below threshold (Faithfulness: {f_score:.2f}, Relevancy: {r_score:.2f}) — triggering retry #{retries + 1}")
        return "retry"

    print(f"  [Critic] ✅ Answer approved (Faithfulness: {f_score:.2f}, Relevancy: {r_score:.2f})")
    return "approve"