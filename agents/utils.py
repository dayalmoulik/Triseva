"""
TriSeva Agent Utility Functions & Helper Modules.

Provides string cleaning, agent JSON response parsing, source relevance filtering,
official government deep-link mapping, telemetry initialization, and resilient primary LLM invocation fallbacks.
"""

import json
import re
import os
from typing import Optional, Tuple, List, Dict


def clean_inline_sources(text: str) -> str:
    """Strips inline 'Source: xyz' or 'स्रोत: xyz' citations from generated text.

    Args:
        text (str): Input text containing potential inline citations.

    Returns:
        str: Cleaned text string without inline citation strings.
    """
    if not text:
        return text
    text = re.sub(r'(?i)\n*\s*(?:source|sources|स्रोत|स्रोतः)\s*:.*$', '', text)
    text = re.sub(r'(?i)\s*\((?:source|sources|स्रोत|स्रोतः)\s*:.*?\)', '', text)
    return text.strip()


def parse_agent_json(text: str, default_disclaimer: str) -> Tuple[str, str]:
    """Parses JSON agent responses containing 'factual_response' and 'caution_note'.

    Attempts JSON deserialization, markdown code fence stripping, and multiline regex matching.

    Args:
        text (str): Raw string output from LLM.
        default_disclaimer (str): Fallback safety disclaimer if none parsed.

    Returns:
        Tuple[str, str]: Pair of (factual_response, caution_note).
    """
    if not text:
        return "", default_disclaimer

    text_clean = text.strip()

    # 1. Clean markdown code fence wrappers (```json ... ```)
    if "```" in text_clean:
        parts = text_clean.split("```")
        for p in parts:
            p_strip = p.strip()
            if p_strip.startswith("json"):
                p_strip = p_strip[4:].strip()
            if "{" in p_strip and "}" in p_strip:
                start = p_strip.find("{")
                end = p_strip.rfind("}") + 1
                text_clean = p_strip[start:end]
                break

    if not (text_clean.startswith("{") and text_clean.endswith("}")):
        if "{" in text_clean and "}" in text_clean:
            start = text_clean.find("{")
            end = text_clean.rfind("}") + 1
            text_clean = text_clean[start:end]

    # 2. Permissive json load with strict=False
    try:
        data = json.loads(text_clean, strict=False)
        factual = data.get("factual_response", "").strip()
        caution = data.get("caution_note", "").strip()
        if factual:
            return clean_inline_sources(factual), caution or default_disclaimer
    except Exception:
        pass

    # 3. Multiline Regex fallback for "factual_response" and "caution_note"
    factual_match = re.search(r'"factual_response"\s*:\s*"(.*?)"\s*,\s*"caution_note"', text_clean, re.DOTALL)
    if not factual_match:
        factual_match = re.search(r'"factual_response"\s*:\s*"(.*)"', text_clean, re.DOTALL)

    caution_match = re.search(r'"caution_note"\s*:\s*"(.*?)"\s*\}', text_clean, re.DOTALL)

    factual = ""
    caution = ""

    if factual_match:
        raw_f = factual_match.group(1)
        if raw_f.endswith('",'):
            raw_f = raw_f[:-2]
        elif raw_f.endswith('"'):
            raw_f = raw_f[:-1]
        factual = raw_f.replace('\\"', '"').replace('\\n', '\n').replace('\\t', '\t')

    if caution_match:
        raw_c = caution_match.group(1)
        if raw_c.endswith('"'):
            raw_c = raw_c[:-1]
        caution = raw_c.replace('\\"', '"').replace('\\n', '\n')

    if factual.strip():
        return clean_inline_sources(factual.strip()), caution.strip() or default_disclaimer

    # 4. Cleanup JSON syntax wrapper if regex fails
    cleaned_fallback = re.sub(r'^\s*\{\s*"factual_response"\s*:\s*"?', '', text_clean, flags=re.IGNORECASE)
    cleaned_fallback = re.sub(r'"\s*,\s*"caution_note"\s*:.*$', '', cleaned_fallback, flags=re.DOTALL | re.IGNORECASE)
    cleaned_fallback = re.sub(r'"\s*\}\s*$', '', cleaned_fallback).strip()

    return clean_inline_sources(cleaned_fallback), default_disclaimer


def initialize_telemetry(state: dict) -> dict:
    """Initializes or loads state telemetry dictionary.

    Args:
        state (dict): Pipeline state dictionary.

    Returns:
        dict: Populated telemetry dictionary.
    """
    return state.get("telemetry") or {
        "routing_hops": [],
        "retries": 0,
        "latency": 0.0,
        "is_fallback_routing": False,
        "fallback_routing_method": None,
        "is_fallback_critic": False,
        "is_fallback_retrieval": False,
    }


def get_document_context(state: dict, max_chars: int = 50000) -> Optional[str]:
    """Retrieves and defensively slices OCR document text if present in state.

    Args:
        state (dict): Pipeline state dictionary.
        max_chars (int, optional): Maximum character slice length. Defaults to 50000.

    Returns:
        Optional[str]: Extracted document text string or None.
    """
    image_text = state.get("image_text")
    if image_text:
        return image_text[:max_chars]
    return None


def is_answer_not_found(answer_text: str) -> bool:
    """Checks if generated answer text represents a fallback 'not found' or error message.

    Args:
        answer_text (str): Answer string to check.

    Returns:
        bool: True if answer indicates missing data or failure, False otherwise.
    """
    if not answer_text or not answer_text.strip():
        return True

    answer_lower = answer_text.lower().strip()

    not_found_phrases = [
        "cannot find the answer",
        "could not find",
        "cannot find",
        "no relevant context",
        "no relevant information",
        "no information available",
        "not found in the available",
        "unable to find",
        "unable to locate",
        "request timed out",
        "encountered an error",
        "no answer generated",
        "no relevant context found",
    ]

    for phrase in not_found_phrases:
        if phrase in answer_lower:
            return True

    return False


def filter_representative_sources(answer_text: str, sources: list, retrieved_chunks: list = None) -> list:
    """Filters retrieved sources to retain relevant items with positive confidence scores.

    Filters out system fallbacks, noise, sources with score <= 0, and items unrepresented in answer.

    Args:
        answer_text (str): Final generated answer text.
        sources (list): Raw list of source dictionaries or strings.
        retrieved_chunks (list, optional): Raw retrieved text passages. Defaults to None.

    Returns:
        list: Filtered list of representative source items.
    """
    if not sources or is_answer_not_found(answer_text):
        return []

    answer_lower = answer_text.lower()
    representative_sources = []
    seen_names = set()

    for s in sources:
        if not s:
            continue

        source_name = ""
        score = 1.0

        if isinstance(s, dict):
            source_name = s.get("source", "").strip()
            try:
                score = float(s.get("score", 1.0))
            except Exception:
                score = 1.0
        elif isinstance(s, str):
            source_name = s.strip()

        # Remove any source with confidence less than or equal to 0% (score <= 0)
        if score <= 0:
            continue

        if not source_name or source_name.lower() in ["system", "no context.", "no context", "unknown", "no web results found."]:
            continue

        if source_name in seen_names:
            continue

        # 1. Web URLs are always valid web references
        if source_name.startswith("http://") or source_name.startswith("https://"):
            representative_sources.append(s if isinstance(s, dict) else {"source": source_name, "score": score})
            seen_names.add(source_name)
            continue

        # 2. Document / PDF Sources: Check key word overlap with answer_text
        base_name = os.path.basename(source_name).lower()
        clean_base = base_name.replace(".pdf", "").replace(".txt", "").replace(".png", "").replace(".jpg", "").replace("_", " ").replace("-", " ")

        key_words = [w for w in clean_base.split() if len(w) > 3 and w not in ["data", "raw", "pdfs", "health", "legal", "agriculture", "report"]]

        is_relevant = False
        if not key_words:
            is_relevant = True
        else:
            for kw in key_words:
                if kw in answer_lower:
                    is_relevant = True
                    break

        if not is_relevant and score > 0.5:
            is_relevant = True

        if is_relevant:
            representative_sources.append(s if isinstance(s, dict) else {"source": source_name, "score": score})
            seen_names.add(source_name)

    # Fallback to top sources if no specific key words matched
    if not representative_sources and sources:
        for s in sources:
            name = s.get("source") if isinstance(s, dict) else str(s)
            if name and name.lower() not in ["system", "no context.", "no context", "no web results found."]:
                representative_sources.append(s if isinstance(s, dict) else {"source": name, "score": 1.0})
                if len(representative_sources) >= 3:
                    break

    return representative_sources


SPECIFIC_SOURCE_MAP = {
    # Healthcare Specific Links
    "medline": ("MedlinePlus Official Medical Database", "https://medlineplus.gov"),
    "health": ("Ministry of Health & Family Welfare Guidelines", "https://www.mohfw.gov.in"),
    "nih": ("National Institutes of Health (NCBI PubMed)", "https://www.ncbi.nlm.nih.gov"),
    "who": ("World Health Organization (WHO) Health Topics", "https://www.who.int"),

    # Legal & Scheme Specific Deep Links
    "mgnrega": ("myScheme Portal - MGNREGA Scheme Guidelines", "https://www.myscheme.gov.in/schemes/mgnrega"),
    "nrega": ("MGNREGA Official Ministry Portal (nrega.nic.in)", "https://nrega.nic.in"),
    "master_roll": ("MGNREGA Master Roll Operational Framework", "https://nrega.nic.in"),
    "constitution": ("Constitution of India Official Legislative Portal", "https://lddashboard.legislative.gov.in/sites/default/files/COI...pdf"),
    "article_371a": ("Article 371A Constitutional Provisions (Nagaland)", "https://www.india.gov.in/my-government/constitution-india"),
    "371a": ("Article 371A Constitutional Provisions for Nagaland", "https://www.india.gov.in/my-government/constitution-india"),
    "bns": ("Bharatiya Nyaya Sanhita (BNS Act 2023 Official PDF)", "https://www.mha.gov.in/sites/default/files/25072024_BNS_English.pdf"),
    "bnss": ("Bharatiya Nagarik Suraksha Sanhita (BNSS Act 2023 Official PDF)", "https://www.mha.gov.in/sites/default/files/25072024_BNSS_English.pdf"),
    "bsa": ("Bharatiya Sakshya Adhiniyam (BSA Act 2023 Official PDF)", "https://www.mha.gov.in/sites/default/files/25072024_BSA_English.pdf"),
    "dpdp": ("Digital Personal Data Protection Act 2023 Official PDF", "https://www.meity.gov.in/writereaddata/files/Digital%20Personal%20Data%20Protection%20Act%202023.pdf"),
    "rti": ("Right to Information Act 2005 Official Document", "https://rti.gov.in/webportal/RTIAct2005.pdf"),
    "nfsa": ("National Food Security Act (NFSA Official Guidelines)", "https://dfpd.gov.in"),
    "pmkisan": ("PM-KISAN Operational Guidelines Official Document", "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf"),
    "pm-kisan": ("PM-KISAN Operational Guidelines Official Document", "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf"),
    "ayushman": ("Ayushman Bharat PM-JAY Official Portal", "https://pmjay.gov.in/about/pmjay"),
    "structured schemes": ("myScheme National Official Government Schemes Portal", "https://www.myscheme.gov.in"),

    # Agriculture Specific Deep Links
    "pmfby": ("Pradhan Mantri Fasal Bima Yojana Operational Guidelines", "https://pmfby.gov.in/pdf/Revised_Operational_Guidelines.pdf"),
    "fasal_bima": ("PM Fasal Bima Yojana Official Guidelines", "https://pmfby.gov.in/pdf/Revised_Operational_Guidelines.pdf"),
    "kusum": ("PM-KUSUM Solar Pump Scheme Official Portal", "https://pmkusum.mnre.gov.in"),
    "soil": ("Soil Health Card National Scheme Portal", "https://soilhealth.dac.gov.in"),
    "aif": ("Agriculture Infrastructure Fund (AIF Official Portal)", "https://agriinfra.dac.gov.in"),
    "icar": ("ICAR National Agricultural Research & Advisory Network", "https://icar.org.in"),
    "kvk": ("Krishi Vigyan Kendra (KVK Advisory Network)", "https://icar.org.in"),
    "annual_report": ("Ministry of Agriculture Annual Reports & Policy Docs", "https://agricoop.nic.in"),
    "nfsm": ("National Food Security Mission (NFSM Portal)", "https://nfsm.gov.in"),
    "pm-rkvy": ("Rashtriya Krishi Vikas Yojana (RKVY Guidelines)", "https://rkvy.nic.in"),
}

def resolve_web_url(source_name: str) -> Optional[Tuple[str, str]]:
    """Maps local vector DB document paths to official web links and descriptive titles.

    Args:
        source_name (str): Document file path or web URL string.

    Returns:
        Optional[Tuple[str, str]]: Pair of (title, url) or None if unmapped.
    """
    if not source_name:
        return None
    source_lower = source_name.strip().lower()

    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        if "indiacode.nic.in" in source_lower:
            title = "India Code Official Legislative Repository"
        elif "nrega" in source_lower:
            title = "MGNREGA Official Ministry Portal & Document"
        elif "pmkisan" in source_lower:
            title = "PM-KISAN Operational Guidelines & Document"
        elif "mohfw" in source_lower:
            title = "Ministry of Health & Family Welfare Document"
        elif "myscheme" in source_lower:
            title = "myScheme National Official Portal Document"
        else:
            title = source_name.split("/")[2] if "//" in source_name else source_name
        return (title, source_name)

    for prefix, (title, web_url) in SPECIFIC_SOURCE_MAP.items():
        if prefix in source_lower:
            return (title, web_url)

    if any(ext in source_lower for ext in [".pdf", ".png", ".jpg", ".txt"]):
        if "health" in source_lower:
            return ("Ministry of Health & Family Welfare Document", "https://www.mohfw.gov.in")
        elif "legal" in source_lower:
            return ("India Code / Legislative Official Document", "https://www.indiacode.nic.in")
        elif "agri" in source_lower:
            return ("Department of Agriculture & Farmers Welfare", "https://agricoop.nic.in")

    return None

def append_source_links(answer_text: str, sources: list) -> str:
    """Appends clickable markdown reference links to answer text.

    Args:
        answer_text (str): Main answer content text.
        sources (list): List of representative sources.

    Returns:
        str: Answer text appended with reference links block.
    """
    clean_ans = clean_inline_sources(answer_text)
    if not sources or is_answer_not_found(clean_ans):
        return clean_ans

    rep_sources = filter_representative_sources(clean_ans, sources)
    if not rep_sources:
        return clean_ans

    resolved_links = []
    seen_urls = set()

    for s in rep_sources:
        name = s.get("source") if isinstance(s, dict) else str(s)
        resolved = resolve_web_url(name)
        if resolved:
            title, url = resolved
            if url not in seen_urls:
                resolved_links.append((title, url))
                seen_urls.add(url)

    if resolved_links:
        link_items = [f"- 📄 [{title}]({url})" for title, url in resolved_links]
        source_block = "\n\n**🔗 Reference Sources & Official Links:**\n" + "\n".join(link_items)
        if "Reference Sources & Official Links" not in clean_ans:
            return clean_ans + source_block

    return clean_ans

def safe_llm_invoke(llm, prompt: str, temperature: float = 0.3, max_tokens: int = 1024) -> str:
    """Invokes primary LLM with automated secondary cloud failover on empty model output.

    Args:
        llm: Primary LLM model instance.
        prompt (str): Prompt string to invoke.
        temperature (float, optional): Generation temperature. Defaults to 0.3.
        max_tokens (int, optional): Maximum tokens limit. Defaults to 1024.

    Returns:
        str: Non-empty generated response text string.
    """
    try:
        res = llm.invoke(prompt)
        content = res.content if res and hasattr(res, "content") else str(res)
        if content and len(content.strip()) > 10:
            return content
    except Exception as e:
        print(f"  [safe_llm_invoke] Primary LLM exception: {e}")

    # Fallback to Ollama (Gemma 4) / OpenAI / Claude if primary returned empty string or raised exception
    print("  [safe_llm_invoke] Primary LLM returned empty string or failed. Triggering fallback...")
    try:
        from langchain_ollama import ChatOllama
        ollama_base = os.getenv("OLLAMA_API_BASE", "http://localhost:11434")
        ollama_model = os.getenv("OLLAMA_MODEL", "gemma4")
        fallback = ChatOllama(model=ollama_model, base_url=ollama_base, temperature=temperature)
        res = fallback.invoke(prompt)
        if res and hasattr(res, "content") and res.content:
            return res.content
    except Exception as e:
        print(f"  [safe_llm_invoke] Ollama local fallback failed: {e}")

    openai_key = os.getenv("OPENAI_API_KEY")
    if openai_key:
        try:
            from langchain_openai import ChatOpenAI
            fallback = ChatOpenAI(model="gpt-4o-mini", api_key=openai_key, temperature=temperature, max_tokens=max_tokens)
            res = fallback.invoke(prompt)
            if res and hasattr(res, "content") and res.content:
                return res.content
        except Exception as e:
            print(f"  [safe_llm_invoke] OpenAI fallback failed: {e}")

    return ""
