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
- FORMATTING RULE FOR NORMAL TEXT RESPONSES: For general agricultural Q&A queries without an uploaded image, structure your answer in clear, well-written conversational paragraphs with natural prose, bold key terms, and bullet points where helpful.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": The direct agricultural factual answer directly grounded in the context, written in natural conversational paragraphs. Do not include any caution or safety warning here.
2. "caution_note": A safety note/caution note (e.g., 'Always verify schemes on official government portals and follow local agriculture officer guidance.').

Example output format:
{
  "factual_response": "Under the PM-KISAN scheme, eligible farmers receive a financial benefit of Rs. 6,000 per year in three equal installments of Rs. 2,000.",
  "caution_note": "Always verify schemes on official government portals and follow local agriculture officer guidance."
}

Do not include any text outside the JSON object."""


def check_structured_agri_rules(query: str) -> str:
    """Matches structured agricultural advisories and scheme rules for common farming queries."""
    query_lower = query.strip().lower()
    matched_context = ""
    
    # 1. Wheat/Crop Yellowing (Chlorosis / Yellow Rust / Nitrogen Deficiency)
    if any(k in query_lower for k in ["yellow", "pila", "peela", "chlorosis", "rust", "gehun", "wheat", "dhan", "paddy"]):
        matched_context += "[STRUCTURED ICAR/KVK ADVISORY - Wheat & Crop Yellowing Management]\n"
        matched_context += "- Common Causes: Nitrogen or Zinc deficiency, waterlogging, or Yellow Rust fungal attack.\n"
        matched_context += "- Recommended Action Steps:\n"
        matched_context += "  * Nitrogen Deficiency: Spray 2% Urea solution (2 kg Urea in 100 Liters water per acre).\n"
        matched_context += "  * Zinc Deficiency: Spray 0.5% Zinc Sulphate + 0.25% Lime solution.\n"
        matched_context += "  * Drainage: Drain standing water from affected field patches.\n"
        matched_context += "  * Fungal/Yellow Rust: Apply Propiconazole 25% EC (1 ml/L water) if yellow powdery pustules appear.\n"
        matched_context += "- Precaution: Consult local Krishi Vigyan Kendra (KVK) or District Agriculture Officer before pesticide application.\n\n"

    # 2. PM Fasal Bima Yojana (PMFBY)
    if any(k in query_lower for k in ["fasal bima", "pmfby", "crop insurance", "bima", "crop loss", "nuksan", "insurance"]):
        matched_context += "[STRUCTURED SCHEME ADVISORY - Pradhan Mantri Fasal Bima Yojana (PMFBY)]\n"
        matched_context += "- Premium Rates: Kharif Crops (2.0%), Rabi Crops (1.5%), Commercial/Horticultural Crops (5.0%).\n"
        matched_context += "- Coverage: Prevented sowing, standing crop damage (drought, flood, pest attack), post-harvest losses.\n"
        matched_context += "- Claim Intimation: Report crop damage within 72 hours via PMFBY app or toll-free helpline 1800-180-1551.\n\n"

    # 3. PM KUSUM (Solar Pumps)
    if any(k in query_lower for k in ["kusum", "solar pump", "solar", "sinchai", "irrigation"]):
        matched_context += "[STRUCTURED SCHEME ADVISORY - PM KUSUM Solar Pump Scheme]\n"
        matched_context += "- Benefit: Up to 60% subsidy (30% Central + 30% State Govt) for standalone solar agriculture pumps up to 7.5 HP.\n"
        matched_context += "- Farmer Share: 40% (bank loan available for up to 30%).\n\n"

    # 4. Soil Health Card & NPK Balance
    if any(k in query_lower for k in ["soil health", "soil card", "npk", "khad", "fertilizer", "mitti"]):
        matched_context += "[STRUCTURED ADVISORY - Soil Health & Balanced Fertilizer Use]\n"
        matched_context += "- Recommended NPK Ratio: 4:2:1 (Nitrogen: Phosphorus: Potassium) for cereal crops.\n"
        matched_context += "- Soil Testing: Get soil samples tested every 2-3 years at nearest Soil Testing Laboratory or KVK.\n\n"

    return matched_context


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
        orig_q = state.get("original_query")
        chunks = retrieve(query, domain="agriculture", n_results=7, native_query=orig_q)
        
        context = ""
        struct_context = check_structured_agri_rules(query)
        if struct_context:
            print("      [Agri Agent Tool] Match found in structured agriculture advisories database.")
            retrieved_chunks_list.append(struct_context)
            retrieved_sources_list.append({"source": "ICAR/KVK Agriculture Advisory Database", "score": 1.0})
            context += struct_context

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
        if "http://" in res or "https://" in res:
            import re
            urls = re.findall(r"https?://[^\s\)\"\']+", res)
            for u in urls:
                retrieved_sources_list.append({"source": u, "score": 1.0})
        else:
            retrieved_sources_list.append({"source": "Web Search", "score": 1.0})
        return res

    try:
        llm = get_llm(temperature=0.3, max_tokens=1024)

        image_path = state.get("image_path")
        doc_context = get_document_context(state)

        if not doc_context and image_path:
            return {
                "draft_answer": "I was unable to extract readable text from the uploaded document image. Please ensure the document photo or scan is clear, well-lit, and legible, or try re-uploading a higher resolution image.",
                "retrieved_chunks": [],
                "sources": [],
                "quiz": state.get("quiz") or [],
                "domain_disclaimer": "🌾 Always verify schemes and advisories on official government portals (e.g., pmkisan.gov.in) and follow local agriculture officer guidance.",
                "telemetry": telemetry,
            }

        if doc_context:
            # Direct LLM call to prevent tool-binding and avoid 403/Forbidden issues on model endpoints
            prompt = f"""You are TriSeva's Agriculture Assistant for India.
Analyze the provided document context (e.g., Soil Health Card, crop advisory, or agricultural receipt) and fulfill the user's request.

[Document Context]
{doc_context}

User Query: {state['user_query']}

Guidelines:
- Ground your answer strictly in the provided [Document Context].
- Output a structured summary adhering strictly to this markdown structure (place EVERY parameter on a NEW LINE with a bullet dash `- `):
  1. **Farmer & Soil Card Details**: Farmer Name, Village/District, Location, Soil Type/Texture, and Crop info.
  2. **Soil Health Indicators & Nutrient Statements**: Place EACH parameter on its OWN NEW LINE starting with `- **[Parameter Name]**:`
     - **pH**: Value | Status Level (High/Medium/Low) | Recommended Action (e.g. Gypsum application)
     - **Electrical Conductivity**: Value | Status Level | Recommended Action
     - **Soil Organic Carbon**: Value | Status Level | Recommended Action (e.g. FYM application)
     - **Available Nitrogen / Phosphorus / Potassium**: Value | Status Level | Recommended Action
     Summarize ALL parameters present in the document context on separate new lines without markdown table pipes.
  3. **Fertilizer & Crop Recommendations**: Actionable dosage recommendations per acre/hectare (e.g. Urea, DAP, SSP, FYM) and crop advisories.
- Avoid unsafe chemical advice; recommend label and local agriculture officer guidance.
- Faithfully summarize all available soil parameters, readings, ratings, and fertilizer doses present in the document.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": A clear, structured Soil Health Card summary containing bulleted parameter statements and actionable advice grounded in the document context. Do NOT use pipe tables. Do NOT include 'Source:' or 'स्रोत:' inline citations.
2. "caution_note": A safety note/caution note (e.g., 'Always verify schemes on official government portals and follow local agriculture officer guidance.').

Do not include any text outside the JSON object."""
            
            raw_answer = safe_llm_invoke(llm, prompt)
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
