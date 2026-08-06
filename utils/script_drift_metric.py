"""
TriSeva Script Drift Quantitative Metric Module.

Evaluates script consistency and code-mixed alignment (Devanagari vs Latin Hinglish)
for multilingual responses, penalizing unwhitelisted script drift or pure monolingual style mismatches.
"""

import os
import re
import json
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

WORKSPACE_PATH = "c:/Users/Moulik/Agentic AI/Triseva"
sys.path.insert(0, WORKSPACE_PATH)

WHITELIST_FILE = os.path.join(WORKSPACE_PATH, "config/latin_whitelist.json")

_whitelist = None
_llm = None

def load_whitelist() -> set:
    """Loads whitelisted technical Latin terms (e.g. 'pH', 'g/dL', 'PM-KISAN') from JSON config.

    Returns:
        set: Lowercase set of whitelisted terms.
    """
    global _whitelist
    if _whitelist is None:
        if os.path.exists(WHITELIST_FILE):
            try:
                with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    _whitelist = {item.lower() for item in data.get("whitelist", [])}
            except Exception as e:
                print(f"[Script Drift] Error loading whitelist: {e}")
                _whitelist = set()
        else:
            print(f"[Script Drift] Warning: Whitelist file {WHITELIST_FILE} not found.")
            _whitelist = set()
    return _whitelist

def _get_llm():
    """Lazy initializes Sarvam LLM instance for script transliteration normalization.

    Returns:
        ChatOpenAI: Configured Sarvam model instance.
    """
    global _llm
    if _llm is None:
        api_key = os.getenv("SARVAM_API_KEY") or os.getenv("Sarvam_API_Key")
        if api_key:
            _llm = ChatOpenAI(
                model="sarvam-105b",
                openai_api_key=api_key,
                openai_api_base="https://api.sarvam.ai/v1",
                temperature=0.0,
                max_tokens=1536,
                timeout=45,
            )
    return _llm

def transliterate_to_devanagari(text: str) -> str:
    """Transliterates Latin Hinglish text to standard Devanagari Hindi while preserving technical terms.

    Args:
        text (str): Hinglish text string.

    Returns:
        str: Transliterated Devanagari text string.
    """
    llm = _get_llm()
    if not llm:
        return text
        
    prompt = f"""You are a professional Hindi translator and transliteration engine.
Transliterate the following Hinglish (Hindi written in Latin/English script) text into standard Devanagari script Hindi.
Keep all technical English words, proper nouns, and units (e.g. "Soil Health Card", "diabetes", "ha", "mg/dL") exactly in their original Latin script (English alphabets).

Text to Transliterate:
{text}

Devanagari Hindi Output:"""
    try:
        res = llm.invoke(prompt)
        return res.content.strip()
    except Exception as e:
        print(f"[Script Drift Normalizer] API call failed: {e}")
        return text

def is_actual_bilingual(text: str) -> bool:
    """Checks if text contains authentic Devanagari characters or Indic romanized stopwords.

    Args:
        text (str): Text string to analyze.

    Returns:
        bool: True if bilingual/Indic markers are present, False otherwise.
    """
    if re.search(r"[\u0900-\u097f]", text):
        return True
    hindi_stopwords = {
        "kya", "hai", "hain", "ko", "se", "ka", "ki", "ke", "mein", "par", 
        "bhi", "aur", "ya", "tha", "thi", "hoon", "pe", "ne", "karo", "karna", 
        "raha", "rahi", "toh", "ab", "kab", "jab", "tab", "bimar", "bimari", 
        "ilaaj", "dawa", "khet", "kheti", "mitti", "kisan", "samman", "gaane", "liye", "kaun"
    }
    words = re.findall(r"\b\w+\b", text.lower())
    match_count = sum(1 for w in words if w in hindi_stopwords)
    return match_count >= 1


def calculate_drift_score(text: str, script_hint: str, use_normalization: bool = False) -> float:
    """Calculates continuous script-drift metric score between 0.0 and 1.0.

    1.0 represents perfect script consistency, while 0.0 represents complete script drift.

    Args:
        text (str): Generated answer text string to evaluate.
        script_hint (str): Expected script style ('latin' or 'devanagari').
        use_normalization (bool, optional): Whether to run pre-transliteration normalization. Defaults to False.

    Returns:
        float: Calculated script drift score (0.000 to 1.000).
    """
    if not text or not text.strip():
        return 1.0
        
    whitelist = load_whitelist()
    cleaned_text = re.sub(r"\b\d+(?:\.\d+)?\b", "", text)
    
    if script_hint == "latin":
        if not is_actual_bilingual(text):
            return 0.0
            
        all_tokens = re.findall(r"\b\w+(?:-\w+)*\b", cleaned_text)
        if not all_tokens:
            return 1.0
            
        devanagari_tokens = [t for t in all_tokens if re.search(r"[\u0900-\u097f]", t)]
        error_count = len(devanagari_tokens)
        
        score = 1.0 - (error_count / len(all_tokens))
        return round(max(0.0, min(1.0, score)), 3)
        
    elif script_hint == "devanagari":
        if use_normalization:
            cleaned_text = transliterate_to_devanagari(cleaned_text)
            
        all_tokens = re.findall(r"\b[a-zA-Z\u0900-\u097f]+(?:-[a-zA-Z\u0900-\u097f]+)*\b", cleaned_text)
        if not all_tokens:
            return 1.0
            
        latin_tokens = [t for t in all_tokens if re.match(r"^[a-zA-Z-]+$", t)]
        
        non_whitelisted_latin = []
        for token in latin_tokens:
            low_token = token.lower()
            if low_token not in whitelist:
                non_whitelisted_latin.append(token)
                
        error_count = len(non_whitelisted_latin)
        score = 1.0 - (error_count / len(all_tokens))
        return round(max(0.0, min(1.0, score)), 3)
        
    return 1.0
