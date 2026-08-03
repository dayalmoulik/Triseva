import warnings
warnings.filterwarnings("ignore")

import os
import sys
import time
from dotenv import load_dotenv
load_dotenv()

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')

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
- CRITICAL PARAMETRIC GUARDRAIL: You are strictly forbidden from answering using your own pre-trained external knowledge. If the local database context does not contain the specific facts needed to answer the user's query, you MUST use the `healthcare_web_search` tool to search the web for the necessary facts. Only if BOTH the local database and the web search fail to find the answer should you return a JSON object where "factual_response" is exactly "I cannot find the answer to this in the available database." and "caution_note" contains your standard disclaimer. Do not extrapolate, guess, or synthesize any answer.
- CRITICAL: Do NOT under any circumstances introduce any facts, medical advice, numbers, symptoms, or details that are not explicitly written in the retrieved context (from either database or web search). Every single claim or precaution you mention must be directly supported by a source in the retrieved context.
- Only include sections (like "Relevant medical context" or "What to watch out for") if the retrieved context explicitly contains that information. If not, omit those sections.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": The direct medical factual answer directly grounded in the context (including context-supported symptoms or precautions if present, formatted cleanly with markdown bullet points if appropriate). Do not include any warning or consultant recommendation here.
2. "caution_note": A safety note/disclaimer (e.g., 'Please consult a qualified doctor for personal medical advice.').

Example output format:
{
  "factual_response": "Your hemoglobin level is 10.2 g/dL. According to the reference guide, this is considered low for adults, which may indicate mild anemia.",
  "caution_note": "Please consult a qualified doctor for personal medical advice."
}

Do not include any text outside the JSON object."""


def health_agent_node(state: TriSevaState) -> dict:
    """Healthcare Specialist Node running a dynamic ReAct agent loop."""
    print(f"  [Health Agent] Processing: '{state['user_query'][:60]}'")

    # Load or initialize telemetry
    telemetry = initialize_telemetry(state)

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
        orig_q = state.get("original_query")
        chunks = retrieve(query, domain="health", n_results=7, native_query=orig_q)
        
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
        if "http://" in res or "https://" in res:
            import re
            urls = re.findall(r"https?://[^\s\)\"\']+", res)
            for u in urls:
                retrieved_sources_list.append({"source": u, "score": 1.0})
        else:
            retrieved_sources_list.append({"source": "Web Search", "score": 1.0})
        return res

    try:
        global llm
        if llm is None:
            llm = get_llm(temperature=0.2, max_tokens=1024)

        doc_context = get_document_context(state)
        if doc_context:
            # Direct LLM call to prevent tool-binding and avoid 403/Forbidden issues on model endpoints
            prompt = f"""You are TriSeva's Healthcare Assistant — a knowledgeable, empathetic medical information assistant for Indian patients.
Analyze the provided document context (e.g., doctor prescription, lab report, or clinical note) and fulfill the user's request.

[Document Context / Uploaded Prescription & Medical Report]
{doc_context}

User Query: {state['user_query']}

Guidelines:
- Ground your answer strictly in the provided [Document Context].
- If the user asks to summarize, explain, or transcribe the document (e.g., "summarize", "summarize this", "explain prescription", "what is written"), output a structured summary adhering strictly to this markdown structure:
  1. **Patient & Clinic Info**: Doctor name, clinic/hospital header, patient name, age/gender, and date (if present).
  2. **Prescribed Medications Table**: Create a clean GitHub Markdown table with these exact headers:
     | Medication | Dosage / Form | Frequency | Timing | Duration |
     | :--- | :--- | :--- | :--- | :--- |
     Fill in all prescribed drugs (brand & generic), dosage (e.g., 625mg), frequency (e.g. 1-0-1 or Twice Daily), administration timing (e.g. After Meals), and duration (e.g. 5 days).
  3. **Doctor Advice & Clinical Notes**: Bullet points of gargling, fluids, follow-up, or precautions written on the prescription.
- Never diagnose — provide an informational explanation of prescribed medications and clinical instructions.
- CRITICAL PARAMETRIC GUARDRAIL: If the provided [Document Context] is empty or completely unreadable and does not contain valid document text, return a JSON object where "factual_response" is exactly "I cannot find the answer to this in the available database." and "caution_note" contains your standard disclaimer.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": A clear, structured prescription summary containing the Markdown table and bulleted advice grounded in the document context. Do NOT include 'Source:' or 'स्रोत:' inline citations.
2. "caution_note": A safety note/disclaimer (e.g., 'Please consult a qualified doctor for personal medical advice.').

Do not include any text outside the JSON object."""
            
            response = llm.invoke(prompt)
            raw_answer = response.content
            print(f"  [Health Agent Direct] Done ({len(raw_answer)} chars)")
            
            default_disclaimer = "⚕️ This is informational only. Please consult a qualified doctor for personal medical advice."
            factual, caution = parse_agent_json(raw_answer, default_disclaimer)
            
            return {
                "draft_answer":      factual,
                "retrieved_chunks":  [f"[Document Context]\n{doc_context}"],
                "sources":           [{"source": "Uploaded Document", "score": 1.0}],
                "domain_disclaimer": caution,
                "telemetry":         telemetry,
            }

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
        raw_answer = final_messages[-1].content if final_messages else ""

        print(f"  [Health Agent] ✅ Done ({len(raw_answer)} chars)")

        default_disclaimer = "⚕️ This is informational only. Please consult a qualified doctor for personal medical advice."
        factual, caution = parse_agent_json(raw_answer, default_disclaimer)

        # Ensure we have default lists if no tool calls were triggered
        chunks_to_return = retrieved_chunks_list if retrieved_chunks_list else ["No context."]
        sources_to_return = retrieved_sources_list if retrieved_sources_list else [{"source": "system", "score": 1.0}]

        return {
            "draft_answer":      factual,
            "retrieved_chunks":  chunks_to_return,
            "sources":           sources_to_return,
            "domain_disclaimer": caution,
            "telemetry":         telemetry,
        }

    except Exception as e:
        print(f"  [Health Agent] Error: {str(e)[:120]}")
        error_msg = str(e)
        if "API_KEY" in error_msg or "credential" in error_msg or "Configur" in error_msg:
            return_msg = f"⚠️ Configuration Error: {error_msg}"
        else:
            return_msg = "I encountered an error processing your health query. Please try again."
        default_disclaimer = "⚕️ Please consult a qualified doctor for personal medical advice."
        return {
            "draft_answer":      return_msg,
            "retrieved_chunks":  [],
            "sources":           [],
            "domain_disclaimer": default_disclaimer,
            "telemetry":         telemetry,
        }