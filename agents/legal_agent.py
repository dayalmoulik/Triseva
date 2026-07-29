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
from tools.calculator_tool import calculator_tool

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = None

# ── System Prompt ─────────────────────────────────────────────────────────────
LEGAL_SYSTEM_PROMPT = """You are TriSeva's Legal and Government Schemes Assistant —
an expert on Indian government welfare schemes, citizen rights, and legal aid.

Your task is to answer the citizen's question.
You MUST use your legal database tool to retrieve grounded facts to answer the question.
If local retrieval returns no results or insufficient information, you may search the web.
If mathematical calculations (like annual income, land conversion, or thresholds) are needed to verify eligibility, you MUST use the calculator tool.

Guidelines:
- Be precise about eligibility — wrong information can harm citizens.
- Do NOT include general definitions, background explanations, or administrative details unless explicitly written in the retrieved context.
- Always cite the scheme name.
- Use calculator results if income thresholds are involved.
- CRITICAL PARAMETRIC GUARDRAIL: You are strictly forbidden from answering using your own pre-trained external knowledge. If the local database context does not contain the specific facts needed to answer the user's query, you MUST use the `legal_web_search` tool to search the web for the necessary facts. Only if BOTH the local database and the web search fail to find the answer should you return a JSON object where "factual_response" is exactly "I cannot find the answer to this in the available database." and "caution_note" contains your standard disclaimer. Do not extrapolate, guess, or synthesize any answer.
- CRITICAL: Do NOT assume, extrapolate, or introduce outside details about scheme eligibility, application steps, or required criteria. Every statement you make must be directly backed by the retrieved context (from either database or web search).
- Only include sections (like eligibility criteria, required documents, how to apply) if the retrieved context explicitly contains that information. If not, omit those sections.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": The direct legal/schemes factual answer directly grounded in the context (including context-supported eligibility criteria, required documents, or how to apply if present, formatted cleanly with markdown bullet points if appropriate) followed by the Source citation. Do not include any caution or safety warning here.
2. "caution_note": A safety note/disclaimer (e.g., 'This information is for guidance only. Consult a legal professional for specific advice.').

Example output format:
{
  "factual_response": "To be eligible for the scheme, the applicant must be a resident of India and have an annual income below Rs. 2 Lakhs. Source: National Welfare Scheme Guidelines.",
  "caution_note": "This information is for guidance only. Consult a legal professional for specific advice."
}

Do not include any text outside the JSON object."""


def legal_agent_node(state: TriSevaState) -> dict:
    """Legal & Government Schemes Specialist Node running a dynamic ReAct agent loop."""
    print(f"  [Legal Agent] Processing: '{state['user_query'][:60]}'")

    # Load or initialize telemetry
    telemetry = initialize_telemetry(state)

    retrieved_chunks_list = []
    retrieved_sources_list = []

    def check_structured_schemes(q_text: str) -> str:
        """Check query for scheme keywords and return their structured metadata context if matched."""
        try:
            import json
            import os
            schema_path = "data/legal_schemes.json"
            if not os.path.exists(schema_path):
                return ""
                
            with open(schema_path, "r", encoding="utf-8") as f_schema:
                schemes = json.load(f_schema)
                
            query_lower = q_text.lower()
            matched_context = ""
            
            # Check PM Kisan
            if "kisan" in query_lower or "pm-kisan" in query_lower or "pmkisan" in query_lower:
                pk = schemes.get("pm_kisan", {})
                matched_context += f"[STRUCTURED SCHEME RULES - {pk.get('name')}]\n"
                matched_context += f"- Benefit: {pk.get('benefit')}\n"
                matched_context += f"- Land Limit: Max {pk.get('land_limit_hectares')} hectares\n"
                matched_context += f"- Rules: {pk.get('eligibility_rules')}\n"
                matched_context += f"- Exclusions: {', '.join(pk.get('exclusions', []))}\n\n"
                
            # Check Ayushman Bharat
            if "ayushman" in query_lower or "pm-jay" in query_lower or "pmjay" in query_lower:
                ab = schemes.get("ayushman_bharat", {})
                matched_context += f"[STRUCTURED SCHEME RULES - {ab.get('name')}]\n"
                matched_context += f"- Benefit: {ab.get('benefit')}\n"
                matched_context += f"- Rules: {ab.get('eligibility_rules')}\n"
                matched_context += f"- Documents required: {', '.join(ab.get('documents_required', []))}\n"
                matched_context += f"- Exclusions: {', '.join(ab.get('exclusions', []))}\n\n"
                
            # Check MGNREGA
            if "mgnrega" in query_lower or "mnrega" in query_lower or "nrega" in query_lower or "muster roll" in query_lower or "musterroll" in query_lower:
                mn = schemes.get("mgnrega", {})
                matched_context += f"[STRUCTURED SCHEME RULES - {mn.get('name')}]\n"
                matched_context += f"- Benefit: {mn.get('benefit')}\n"
                matched_context += f"- Rules: {mn.get('eligibility_rules')}\n"
                matched_context += f"- Signatory Authority: {mn.get('signatory_authority')}\n"
                matched_context += f"- Work Allocation Time Limit: {mn.get('time_limit_days')} days\n"
                matched_context += f"- Unemployment Rule: {mn.get('unemployment_allowance_rule')}\n\n"
                
            # Check PMAY
            if "awas" in query_lower or "yojana" in query_lower or "pmay" in query_lower or "housing" in query_lower:
                pa = schemes.get("pmay", {})
                matched_context += f"[STRUCTURED SCHEME RULES - {pa.get('name')}]\n"
                matched_context += f"- Benefit: {pa.get('benefit')}\n"
                matched_context += "- Categories:\n"
                for cat in pa.get("categories", []):
                    matched_context += f"  * {cat.get('name')}: Income up to {cat.get('income_limit_lakhs')} lakhs, Subsidy {cat.get('subsidy_rate')}%, Max loan {cat.get('loan_limit_lakhs')} lakhs\n"
                matched_context += "\n"

            # Check Nagaland & Article 371A
            if "nagaland" in query_lower or "371a" in query_lower or "naga" in query_lower:
                ng = schemes.get("nagaland_special_provisions", {})
                matched_context += f"[STRUCTURED SCHEME RULES - {ng.get('name')}]\n"
                matched_context += f"- Benefit: {ng.get('benefit')}\n"
                matched_context += f"- Key Provisions: {ng.get('key_provisions')}\n"
                matched_context += f"- Land Transfer Rule: {ng.get('land_transfer_rule')}\n\n"

                nf = schemes.get("nagaland_focus_scheme", {})
                matched_context += f"[STRUCTURED SCHEME RULES - {nf.get('name')}]\n"
                matched_context += f"- Benefit: {nf.get('benefit')}\n"
                matched_context += f"- Eligibility: {nf.get('eligibility_rules')}\n"
                matched_context += f"- Components: {nf.get('key_components')}\n\n"

            return matched_context
        except Exception as e_schema:
            print(f"Error checking structured schemes: {e_schema}")
            return ""

    # ── Define Tools ──────────────────────────────────────────────────────────
    @tool
    def legal_knowledge_base_retrieval(query: str) -> str:
        """
        Query the legal and government schemes database for welfare details, laws, and eligibility rules.
        Use this as your primary tool to retrieve grounded facts.
        """
        print(f"    [Legal Agent Tool] Querying local KB: '{query}'")
        orig_q = state.get("original_query")
        chunks = retrieve(query, domain="legal", n_results=7, native_query=orig_q)
        
        context = ""
        struct_context = check_structured_schemes(query)
        if struct_context:
            print("      [Legal Agent Tool] Match found in structured schemes database.")
            retrieved_chunks_list.append(struct_context)
            retrieved_sources_list.append({"source": "Structured Schemes Database", "score": 1.0})
            context += struct_context

        if chunks:
            for idx, c in enumerate(chunks, 1):
                retrieved_chunks_list.append(c["text"])
                retrieved_sources_list.append({"source": c["source"], "score": c["score"]})
                context += f"[Source {idx}: {c['source']}]\n{c['text']}\n\n"
        return context if context else "No relevant context found in legal database."

    @tool
    def legal_web_search(query: str) -> str:
        """
        Search the web for current legal guidelines or government schemes.
        Use this ONLY when the legal database does not contain the answer.
        """
        print(f"    [Legal Agent Tool] Searching web: '{query}'")
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
            llm = get_llm(temperature=0.1, max_tokens=1024)

        doc_context = get_document_context(state)
        if doc_context:
            # Direct LLM call to prevent tool-binding and avoid 403/Forbidden issues on model endpoints
            prompt = f"""You are TriSeva's Legal and Government Schemes Assistant — an expert on Indian government welfare schemes, citizen rights, and legal aid.
Analyze the provided document context and answer the user's query.

[Document Context]
{doc_context}

User Query: {state['user_query']}

Guidelines:
- Ground your answer strictly in the provided [Document Context].
- CRITICAL PARAMETRIC GUARDRAIL: You are strictly forbidden from answering using your own pre-trained external knowledge. If the provided [Document Context] does not contain the specific facts needed to answer the query, you MUST return a JSON object where "factual_response" is exactly "I cannot find the answer to this in the available database." and "caution_note" contains your standard disclaimer. Do not extrapolate or guess.
- Be precise about eligibility — wrong information can harm citizens.

CRITICAL FORMATTING INSTRUCTION:
You MUST respond ONLY with a JSON object containing exactly two fields:
1. "factual_response": The direct legal/schemes factual answer directly grounded in the document context. Do not include any caution or safety warning here.
2. "caution_note": A safety note/disclaimer (e.g., 'This information is for guidance only. Consult a legal professional for specific advice.').

Do not include any text outside the JSON object."""
            
            response = llm.invoke(prompt)
            raw_answer = response.content
            print(f"  [Legal Agent Direct] Done ({len(raw_answer)} chars)")
            
            default_disclaimer = "⚖️ This information is for guidance only. Consult a legal professional for specific advice."
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
            tools=[legal_knowledge_base_retrieval, legal_web_search, calculator_tool],
            prompt=LEGAL_SYSTEM_PROMPT
        )

        # ── Setup Conversation Messages ──────────────────────────────────────
        messages = [
            HumanMessage(content=state["user_query"])
        ]

        critic_feedback = state.get("critic_feedback")
        retry_count = state.get("retry_count", 0)

        if retry_count > 0 and critic_feedback:
            print(f"  [Legal Agent] Retry #{retry_count} based on critic feedback...")
            messages.append(AIMessage(content=state.get("draft_answer", "")))
            messages.append(HumanMessage(content=f"""Our Quality Assurance Critic reviewed your answer and rejected it for the following reason:
{critic_feedback}

Please revise your response. Review the previous context, and use tools to re-query the database or search the web if you need more facts to satisfy the Critic's guidelines. Ensure every statement in your revised answer is directly and strictly supported by the retrieved context. Do NOT extrapolate."""))

        # ── Execute Agent ─────────────────────────────────────────────────────
        print(f"  [Legal Agent] About to call LLM...")
        sys.stdout.flush()
        
        result = agent.invoke({"messages": messages})
        
        print(f"  [Legal Agent] LLM call returned")
        sys.stdout.flush()

        # Extract the final answer from the last message in history
        final_messages = result.get("messages", [])
        raw_answer = final_messages[-1].content if final_messages else ""
        sys.stdout.flush()

        print(f"  [Legal Agent] ✅ Done ({len(raw_answer)} chars)")

        default_disclaimer = "⚖️ This information is for guidance only. Consult a legal professional for specific advice."
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
        print(f"  [Legal Agent] Error: {str(e)[:120]}")
        error_msg = str(e)
        if "API_KEY" in error_msg or "credential" in error_msg or "Configur" in error_msg:
            return_msg = f"⚠️ Configuration Error: {error_msg}"
        else:
            return_msg = "I encountered an error processing your legal query. Please try again."
        default_disclaimer = "⚖️ Consult a legal professional for specific advice."
        return {
            "draft_answer":      return_msg,
            "retrieved_chunks":  [],
            "sources":           [],
            "domain_disclaimer": default_disclaimer,
            "telemetry":         telemetry,
        }