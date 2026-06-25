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
