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
from tools.rag_tool import retrieve
from tools.search_tool import web_search_tool

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = None

# ── System Prompt ─────────────────────────────────────────────────────────────
HEALTH_SYSTEM_PROMPT = """You are TriSeva's Healthcare Assistant — a knowledgeable,
empathetic medical information assistant for Indian patients.

Your task is to answer the patient's question.
You MUST use your healthcare database tool to retrieve grounded facts to answer the question.
If local retrieval returns no results or insufficient information, you may search the web.

Guidelines:
- Use the retrieved context to give accurate information.
- Do NOT include general definitions, background explanations, or medical theory unless explicitly written in the retrieved context.
- Never diagnose — provide information only.
- Be empathetic and clear.
- CRITICAL: Do NOT under any circumstances introduce any facts, medical advice, numbers, symptoms, or details that are not explicitly written in the retrieved context. Every single claim or precaution you mention must be directly supported by a source in the retrieved context. If the retrieved context does not contain enough information, state that the information is not available in the documents rather than extrapolating.
- Only include sections (like "Relevant medical context" or "What to watch out for") if the retrieved context explicitly contains that information. If not, omit those sections.

Format your response as:
1. Direct answer to the question
2. Relevant medical context (ONLY if explicitly in context; otherwise omit)
3. What to watch out for (ONLY if explicitly in context; otherwise omit)
4. Always end with: "Please consult a qualified doctor for personal medical advice." """


def health_agent_node(state: TriSevaState) -> dict:
    """Healthcare Specialist Node running a dynamic ReAct agent loop."""
    print(f"  [Health Agent] Processing: '{state['user_query'][:60]}'")

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

    retrieved_chunks_list = []
    retrieved_sources_list = []

    # ── Define Tools ──────────────────────────────────────────────────────────
    @tool
    def healthcare_knowledge_base_retrieval(query: str) -> str:
        """
        Query the healthcare database for medical guidelines, dosage, and diseases.
        Use this as your primary tool to retrieve grounded facts.
        """
        print(f"    [Health Agent Tool] Querying local KB: '{query}'")
        chunks = retrieve(query, domain="health", n_results=5)
        
        context = ""
        if chunks:
            for idx, c in enumerate(chunks, 1):
                retrieved_chunks_list.append(c["text"])
                retrieved_sources_list.append({"source": c["source"], "score": c["score"]})
                context += f"[Source {idx}: {c['source']}]\n{c['text']}\n\n"
        return context if context else "No relevant context found in healthcare database."

    @tool
    def healthcare_web_search(query: str) -> str:
        """
        Search the web for current medical information.
        Use this ONLY when the healthcare database does not contain the answer.
        """
        print(f"    [Health Agent Tool] Searching web: '{query}'")
        telemetry["is_fallback_retrieval"] = True
        res = web_search_tool.invoke(query)
        retrieved_sources_list.append({"source": "Web Search", "score": 1.0})
        return res

    try:
        global llm
        if llm is None:
            llm = get_llm(temperature=0.2, max_tokens=1024)

        # Create the ReAct agent runner
        agent = create_react_agent(
            model=llm,
            tools=[healthcare_knowledge_base_retrieval, healthcare_web_search],
            prompt=HEALTH_SYSTEM_PROMPT
        )

        # ── Setup Conversation Messages ──────────────────────────────────────
        messages = [
            HumanMessage(content=state["user_query"])
        ]

        critic_feedback = state.get("critic_feedback")
        retry_count = state.get("retry_count", 0)

        if retry_count > 0 and critic_feedback:
            print(f"  [Health Agent] Retry #{retry_count} based on critic feedback...")
            messages.append(AIMessage(content=state.get("draft_answer", "")))
            messages.append(HumanMessage(content=f"""Our Quality Assurance Critic reviewed your answer and rejected it for the following reason:
{critic_feedback}

Please revise your response. Review the previous context, and use tools to re-query the database or search the web if you need more facts to satisfy the Critic's guidelines. Ensure every statement in your revised answer is directly and strictly supported by the retrieved context. Do NOT extrapolate."""))

        # ── Execute Agent ─────────────────────────────────────────────────────
        result = agent.invoke({"messages": messages})
        
        # Extract the final answer from the last message in history
        final_messages = result.get("messages", [])
        answer = final_messages[-1].content if final_messages else ""

        print(f"  [Health Agent] ✅ Done ({len(answer)} chars)")

        # Ensure we have default lists if no tool calls were triggered
        chunks_to_return = retrieved_chunks_list if retrieved_chunks_list else ["No context."]
        sources_to_return = retrieved_sources_list if retrieved_sources_list else [{"source": "system", "score": 1.0}]

        return {
            "draft_answer":      answer,
            "retrieved_chunks":  chunks_to_return,
            "sources":           sources_to_return,
            "domain_disclaimer": "⚕️ This is informational only. Please consult a qualified doctor for personal medical advice.",
            "telemetry":         telemetry,
        }

    except Exception as e:
        print(f"  [Health Agent] Error: {str(e)[:120]}")
        error_msg = str(e)
        if "API_KEY" in error_msg or "credential" in error_msg or "Configur" in error_msg:
            return_msg = f"⚠️ Configuration Error: {error_msg}"
        else:
            return_msg = "I encountered an error processing your health query. Please try again."
        return {
            "draft_answer":      return_msg,
            "retrieved_chunks":  [],
            "sources":           [],
            "domain_disclaimer": "⚕️ Please consult a qualified doctor for personal medical advice.",
            "telemetry":         telemetry,
        }