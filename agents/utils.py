import json
import re
from typing import Optional, Tuple

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
            return factual, caution or default_disclaimer
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
        return factual.strip(), caution.strip() or default_disclaimer

    # 4. Final plain-text fallback
    return text.strip(), default_disclaimer

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
    """Resolves reference URLs from sources list and appends clickable markdown links at the end of the answer."""
    if not sources:
        return answer_text
        
    web_urls = set()
    for s in sources:
        name = s.get("source") if isinstance(s, dict) else str(s)
        url = resolve_web_url(name)
        if url:
            web_urls.add(url)
            
    if web_urls:
        source_block = "\n\n### 🔗 Reference Sources & Official Links:\n" + "\n".join([f"- [{u}]({u})" for u in sorted(web_urls)])
        if "### 🔗 Reference Sources" not in answer_text:
            return answer_text + source_block
            
    return answer_text
