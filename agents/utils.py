import json
import re
from typing import Optional, Tuple

def clean_inline_sources(text: str) -> str:
    """Strips inline 'Source: xyz' or 'स्रोत: xyz' or '(Source: xyz)' from the answer text."""
    if not text:
        return text
    # Strip inline Source / स्रोत citations appended to sentences
    text = re.sub(r'(?i)\n*\s*(?:source|sources|स्रोत|स्रोतः)\s*:.*$', '', text)
    text = re.sub(r'(?i)\s*\((?:source|sources|स्रोत|स्रोतः)\s*:.*?\)', '', text)
    return text.strip()

def parse_agent_json(text: str, default_disclaimer: str) -> Tuple[str, str]:
    """Parse JSON output containing 'factual_response' and 'caution_note' robustly."""
    text_clean = text.strip()

    # 1. Clean markdown wrappers
    if "```" in text_clean:
        parts = text_clean.split("```")
        for p in parts:
            p_strip = p.strip()
            if p_strip.startswith("json"):
                text_clean = p_strip[4:].strip()
                break
            elif p_strip.startswith("{") and p_strip.endswith("}"):
                text_clean = p_strip
                break

    # 2. Strict json load
    try:
        data = json.loads(text_clean)
        factual = data.get("factual_response", "").strip()
        caution = data.get("caution_note", "").strip()
        if factual:
            return clean_inline_sources(factual), caution or default_disclaimer
    except Exception:
        pass

    # 3. Regex fallback
    factual_match = re.search(r'"factual_response"\s*:\s*"((?:[^"\\]|\\.)*)"', text_clean, re.DOTALL)
    caution_match = re.search(r'"caution_note"\s*:\s*"((?:[^"\\]|\\.)*)"', text_clean, re.DOTALL)

    factual = ""
    caution = ""

    if factual_match:
        try:
            factual = json.loads('"' + factual_match.group(1) + '"')
        except Exception:
            factual = factual_match.group(1).replace('\\"', '"').replace('\\n', '\n')

    if caution_match:
        try:
            caution = json.loads('"' + caution_match.group(1) + '"')
        except Exception:
            caution = caution_match.group(1).replace('\\"', '"').replace('\\n', '\n')

    if factual.strip():
        return clean_inline_sources(factual.strip()), caution.strip() or default_disclaimer

    # 4. Final plain-text fallback
    return clean_inline_sources(text.strip()), default_disclaimer

def initialize_telemetry(state: dict) -> dict:
    """Initialize or load the telemetry metrics dictionary from State Graph state."""
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
    """Retrieve and defensively slice the uploaded document text if present."""
    image_text = state.get("image_text")
    if image_text:
        return image_text[:max_chars]
    return None


def is_answer_not_found(answer_text: str) -> bool:
    """Check if the answer indicates that information was not found or failed."""
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
    """
    Ensures retrieved sources are non-empty, representative, and relevant to the generated answer.
    Filters out system fallbacks, low-relevance noise, and sources not represented in the answer.
    Returns an empty list if the answer is a fallback 'not found' response.
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
            score = s.get("score", 1.0)
        elif isinstance(s, str):
            source_name = s.strip()

        if not source_name or source_name.lower() in ["system", "no context.", "no context", "unknown", "no web results found."]:
            continue

        if source_name in seen_names:
            continue

        # 1. Web URLs are always valid web references
        if source_name.startswith("http://") or source_name.startswith("https://"):
            representative_sources.append(s if isinstance(s, dict) else {"source": source_name, "score": score})
            seen_names.add(source_name)
            continue

        # 2. Document / PDF Sources: Check if filename/basename or key terms overlap with answer_text
        import os
        base_name = os.path.basename(source_name).lower()
        clean_base = base_name.replace(".pdf", "").replace(".txt", "").replace(".png", "").replace(".jpg", "").replace("_", " ").replace("-", " ")

        # Extract key words (>3 chars) from the clean base name
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

    # Fallback to top sources if no specific key words matched but valid sources exist
    if not representative_sources and sources:
        for s in sources:
            name = s.get("source") if isinstance(s, dict) else str(s)
            if name and name.lower() not in ["system", "no context.", "no context", "no web results found."]:
                representative_sources.append(s if isinstance(s, dict) else {"source": name, "score": 1.0})
                if len(representative_sources) >= 3:
                    break

    return representative_sources


LOCAL_TO_WEB_MAP = {
    # Healthcare
    "medline": "https://medlineplus.gov",
    "health": "https://www.mohfw.gov.in",
    "nih": "https://www.ncbi.nlm.nih.gov",
    "who": "https://www.who.int",
    # Legal & Government
    "bnss": "https://www.mha.gov.in",
    "bns": "https://www.mha.gov.in",
    "bsa": "https://www.mha.gov.in",
    "dpdp": "https://www.meity.gov.in",
    "rti": "https://rti.gov.in",
    "nfsa": "https://dfpd.gov.in",
    "pmkisan": "https://pmkisan.gov.in",
    "pm-kisan": "https://pmkisan.gov.in",
    "ayushman": "https://pmjay.gov.in",
    "legal": "https://www.india.gov.in",
    # Agriculture
    "annual_report": "https://agricoop.nic.in",
    "nfsm": "https://nfsm.gov.in",
    "pm-rkvy": "https://rkvy.nic.in",
    "pdmc": "https://pmksy.gov.in",
    "midh": "https://midh.gov.in",
    "atma": "https://agricoop.nic.in",
    "fpo": "https://sfacindia.com",
    "aif": "https://agriinfra.dac.gov.in",
    "soil": "https://soilhealth.dac.gov.in",
    "pm-aasha": "https://pmaasha.nic.in",
    "agriculture": "https://agricoop.nic.in",
    "agri": "https://agricoop.nic.in"
}

def resolve_web_url(source_name: str) -> Optional[str]:
    """Maps local vector DB document paths or web URLs to clickable official web links."""
    if not source_name:
        return None
    source_lower = source_name.strip().lower()
    
    if source_lower.startswith("http://") or source_lower.startswith("https://"):
        return source_name
        
    for prefix, web_url in LOCAL_TO_WEB_MAP.items():
        if prefix in source_lower:
            return web_url
            
    if any(ext in source_lower for ext in [".pdf", ".png", ".jpg", ".txt"]):
        if "health" in source_lower:
            return "https://www.mohfw.gov.in"
        elif "legal" in source_lower:
            return "https://www.india.gov.in"
        elif "agri" in source_lower:
            return "https://agricoop.nic.in"
            
    return None

def append_source_links(answer_text: str, sources: list) -> str:
    """Resolves reference URLs from sources list and appends clickable markdown links in a separate section."""
    clean_ans = clean_inline_sources(answer_text)
    if not sources or is_answer_not_found(clean_ans):
        return clean_ans

    rep_sources = filter_representative_sources(clean_ans, sources)
    if not rep_sources:
        return clean_ans

    web_urls = set()
    for s in rep_sources:
        name = s.get("source") if isinstance(s, dict) else str(s)
        url = resolve_web_url(name)
        if url:
            web_urls.add(url)

    if web_urls:
        source_block = "\n\n---\n### 🔗 Reference Sources & Official Links:\n" + "\n".join([f"- [{u}]({u})" for u in sorted(web_urls)])
        if "### 🔗 Reference Sources" not in clean_ans:
            return clean_ans + source_block

    return clean_ans
