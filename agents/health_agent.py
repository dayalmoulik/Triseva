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
from agents.utils import parse_agent_json, initialize_telemetry, get_document_context, safe_llm_invoke
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
- FORMATTING RULE FOR NORMAL TEXT RESPONSES: For general Q&A queries without an uploaded image, structure your answer in clear, well-written conversational paragraphs with natural prose, bold key terms, and bullet points where helpful.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": The direct medical factual answer directly grounded in the context, written in natural conversational paragraphs. Do not include any warning or consultant recommendation here.
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
        llm = get_llm(temperature=0.2, max_tokens=1024)

        image_path = state.get("image_path")
        doc_context = get_document_context(state)

        if not doc_context and image_path:
            return {
                "draft_answer": "I was unable to extract readable text from the uploaded document image. Please ensure the document photo or scan is clear, well-lit, and legible, or try re-uploading a higher resolution image.",
                "retrieved_chunks": [],
                "sources": [],
                "domain_disclaimer": "⚕️ This is informational only. Please consult a qualified doctor for personal medical advice.",
                "telemetry": telemetry,
            }

        if doc_context:
            # Direct LLM call to prevent tool-binding and avoid 403/Forbidden issues on model endpoints
            prompt = f"""You are TriSeva's Healthcare Assistant — a knowledgeable, empathetic medical information assistant for Indian patients.
Analyze the provided document context (e.g., doctor prescription, lab report, or clinical note) and fulfill the user's request.

[Document Context / Uploaded Prescription & Medical Report]
{doc_context}

User Query: {state['user_query']}

Guidelines:
- If the provided [Document Context] is unrelated to healthcare (e.g. Soil Health Card, land document, or non-medical form), IGNORE the document context completely and answer the user's health question directly, accurately, and empathetically using standard clinical reference ranges.
- Otherwise, ground your answer strictly in the provided medical document context.
- Output a structured summary adhering strictly to this markdown structure (place EVERY item on a NEW LINE with a bullet dash `- `):
  1. **Patient & Clinic Info**: Clinic/hospital header, patient name, age/gender, and date (if present). If no doctor name is typed/printed on the prescription (only a signature appears), write "Doctor Name: Not explicitly printed (Signature present)". Do NOT invent doctor names.
  2. **Prescribed Medications**: Place EACH prescribed drug on its OWN NEW LINE starting with `- **[Medication Name]**:`
     - **[Medication 1]**: Dosage (e.g. 625mg) | Frequency (e.g. 1-0-1 or Twice Daily) | Administration Timing (e.g. After Meals) | Duration (e.g. 5 days)
     - **[Medication 2]**: Dosage | Frequency | Administration Timing | Duration
  3. **Doctor Advice & Clinical Notes**: Bullet points of gargling, fluids, gum paint massage, follow-up, or precautions on separate new lines.
- Never diagnose — provide an informational explanation of prescribed medications and clinical instructions.
- Faithfully transcribe all prescribed medications, dosages, timings, and clinical notes present in the document.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": A clear, structured prescription summary containing bulleted medication statements and advice grounded in the document context. Do NOT use pipe tables. Do NOT include 'Source:' or 'स्रोत:' inline citations.
2. "caution_note": A safety note/disclaimer (e.g., 'Please consult a qualified doctor for personal medical advice.').

Do not include any text outside the JSON object."""
            
            raw_answer = safe_llm_invoke(llm, prompt)
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