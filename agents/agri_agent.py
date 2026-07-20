import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from agents.llm_factory import get_llm
from agents.state import TriSevaState
from agents.utils import parse_agent_json, initialize_telemetry, get_document_context
from tools.rag_tool import retrieve
from tools.search_tool import web_search_tool

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = None

# ── System Prompt ─────────────────────────────────────────────────────────────
AGRI_SYSTEM_PROMPT = """You are TriSeva's Agriculture Assistant for India.
Help farmers with practical, safe, and policy-aware guidance using the provided context.

Your task is to answer the farmer's question.
You MUST use your agriculture database tool to retrieve grounded facts to answer the question.
If local retrieval returns no results or insufficient information, you may search the web.

Guidelines:
- Keep recommendations practical and easy to act on.
- Do NOT include general definitions, background explanations, or advice unless explicitly written in the retrieved context.
- Use Indian context: Kharif/Rabi seasons, MSP, mandi, PM-KISAN, KVK, ICAR advisories.
- If policy eligibility is uncertain, clearly say users should verify on official portals.
- Avoid unsafe chemical advice; recommend label and local agriculture officer guidance.
- CRITICAL PARAMETRIC GUARDRAIL: You are strictly forbidden from answering using your own pre-trained external knowledge. If the local database context does not contain the specific facts needed to answer the user's query, you MUST use the `agriculture_web_search` tool to search the web for the necessary facts. Only if BOTH the local database and the web search fail to find the answer should you return a JSON object where "factual_response" is exactly "I cannot find the answer to this in the available database." and "caution_note" contains your standard disclaimer. Do not extrapolate, guess, or synthesize any answer.
- CRITICAL: Do NOT introduce crop timelines, advisory details, or fertilizer recommendations that are not explicitly present in the retrieved context (from either database or web search). Every statement you make must be directly backed by the retrieved context.
- Only include sections (like action steps, scheme/advisory pointers) if the retrieved context explicitly contains that information. If not, omit those sections.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": The direct agricultural factual answer directly grounded in the context (including context-supported action steps or advisory pointers if present, formatted cleanly with markdown bullet points if appropriate). Do not include any caution or safety warning here.
2. "caution_note": A safety note/caution note (e.g., 'Always verify schemes on official government portals and follow local agriculture officer guidance.').

Example output format:
{
  "factual_response": "Under the PM-KISAN scheme, eligible farmers receive a financial benefit of Rs. 6,000 per year in three equal installments of Rs. 2,000.",
  "caution_note": "Always verify schemes on official government portals and follow local agriculture officer guidance."
}

Do not include any text outside the JSON object."""


def agri_agent_node(state: TriSevaState) -> dict:
    """Agriculture Specialist Node running a dynamic ReAct agent loop."""
    print(f"  [Agri Agent] Processing: '{state['user_query'][:60]}'")

    # Load or initialize telemetry
    telemetry = initialize_telemetry(state)

    retrieved_chunks_list = []
    retrieved_sources_list = []

    # ── Define Tools ──────────────────────────────────────────────────────────
    @tool
    def agriculture_knowledge_base_retrieval(query: str) -> str:
        """
        Query the agriculture database for practical farming guidelines, MSP, and crop advisory.
        Use this as your primary tool to retrieve grounded facts.
        """
        print(f"    [Agri Agent Tool] Querying local KB: '{query}'")
        chunks = retrieve(query, domain="agriculture", n_results=5)
        
        context = ""
        if chunks:
            for idx, c in enumerate(chunks, 1):
                retrieved_chunks_list.append(c["text"])
                retrieved_sources_list.append({"source": c["source"], "score": c["score"]})
                context += f"[Source {idx}: {c['source']}]\n{c['text']}\n\n"
        return context if context else "No relevant context found in agriculture database."

    @tool
    def agriculture_web_search(query: str) -> str:
        """
        Search the web for current farming advisories, MSP, mandi prices, or schemes.
        Use this ONLY when the agriculture database does not contain the answer.
        """
        print(f"    [Agri Agent Tool] Searching web: '{query}'")
        telemetry["is_fallback_retrieval"] = True
        res = web_search_tool.invoke(query)
        retrieved_sources_list.append({"source": "Web Search", "score": 1.0})
        return res

    try:
        global llm
        if llm is None:
            llm = get_llm(temperature=0.3, max_tokens=1024)

        doc_context = get_document_context(state)
        if doc_context:
            # Direct LLM call to prevent tool-binding and avoid 403/Forbidden issues on model endpoints
            prompt = f"""You are TriSeva's Agriculture Assistant for India.
Analyze the provided document context and answer the user's query.

[Document Context]
{doc_context}

User Query: {state['user_query']}

Guidelines:
- Ground your answer strictly in the provided [Document Context].
- CRITICAL PARAMETRIC GUARDRAIL: You are strictly forbidden from answering using your own pre-trained external knowledge. If the provided [Document Context] does not contain the specific facts needed to answer the query, you MUST return a JSON object where "factual_response" is exactly "I cannot find the answer to this in the available database." and "caution_note" contains your standard disclaimer. Do not extrapolate or guess.
- Avoid unsafe chemical advice; recommend label and local agriculture officer guidance.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": The direct agricultural factual answer directly grounded in the document context. Do not include any caution or safety warning here.
2. "caution_note": A safety note/caution note (e.g., 'Always verify schemes on official government portals and follow local agriculture officer guidance.').

Do not include any text outside the JSON object."""
            
            response = llm.invoke(prompt)
            raw_answer = response.content
            print(f"  [Agri Agent Direct] Done ({len(raw_answer)} chars)")
            
            default_disclaimer = "🌾 Always verify schemes and advisories on official government portals (e.g., pmkisan.gov.in) and follow local agriculture officer guidance."
            factual, caution = parse_agent_json(raw_answer, default_disclaimer)
            
            return {
                "draft_answer":      factual,
                "retrieved_chunks":  [f"[Document Context]\n{doc_context}"],
                "sources":           [{"source": "Uploaded Document", "score": 1.0}],
                "quiz":              state.get("quiz") or [],
                "domain_disclaimer": caution,
                "telemetry":         telemetry,
            }

        # Create the ReAct agent runner
        agent = create_react_agent(
            model=llm,
            tools=[agriculture_knowledge_base_retrieval, agriculture_web_search],
            prompt=AGRI_SYSTEM_PROMPT
        )

        # ── Setup Conversation Messages ──────────────────────────────────────
        messages = [
            HumanMessage(content=state["user_query"])
        ]

        critic_feedback = state.get("critic_feedback")
        retry_count = state.get("retry_count", 0)

        if retry_count > 0 and critic_feedback:
            print(f"  [Agri Agent] Retry #{retry_count} based on critic feedback...")
            messages.append(AIMessage(content=state.get("draft_answer", "")))
            messages.append(HumanMessage(content=f"""Our Quality Assurance Critic reviewed your answer and rejected it for the following reason:
{critic_feedback}

Please revise your response. Review the previous context, and use tools to re-query the database or search the web if you need more facts to satisfy the Critic's guidelines. Ensure every statement in your revised answer is directly and strictly supported by the retrieved context. Do NOT extrapolate."""))

        # ── Execute Agent ─────────────────────────────────────────────────────
        result = agent.invoke({"messages": messages})
        
        # Extract the final answer from the last message in history
        final_messages = result.get("messages", [])
        raw_answer = final_messages[-1].content if final_messages else ""

        print(f"  [Agri Agent] ✅ Done ({len(raw_answer)} chars)")

        default_disclaimer = "🌾 Always verify schemes and advisories on official government portals (e.g., pmkisan.gov.in) and follow local agriculture officer guidance."
        factual, caution = parse_agent_json(raw_answer, default_disclaimer)

        # Ensure we have default lists if no tool calls were triggered
        chunks_to_return = retrieved_chunks_list if retrieved_chunks_list else ["No context."]
        sources_to_return = retrieved_sources_list if retrieved_sources_list else [{"source": "system", "score": 1.0}]

        return {
            "draft_answer":      factual,
            "retrieved_chunks":  chunks_to_return,
            "sources":           sources_to_return,
            "quiz":              state.get("quiz") or [],
            "domain_disclaimer": caution,
            "telemetry":         telemetry,
        }

    except Exception as e:
        print(f"  [Agri Agent] Error: {str(e)[:120]}")
        error_msg = str(e)
        if "API_KEY" in error_msg or "credential" in error_msg or "Configur" in error_msg:
            return_msg = f"⚠️ Configuration Error: {error_msg}"
        else:
            return_msg = "I encountered an error processing your agriculture query. Please try again."
        default_disclaimer = "🌾 Always verify schemes and advisories on official government portals (e.g., pmkisan.gov.in) and follow local agriculture officer guidance."
        return {
            "draft_answer":      return_msg,
            "retrieved_chunks":  [],
            "sources":           [],
            "quiz":              state.get("quiz") or [],
            "domain_disclaimer": default_disclaimer,
            "telemetry":         telemetry,
        }
