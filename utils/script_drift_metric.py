import os
import re
import json
import sys
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

WORKSPACE_PATH = "c:/Users/Moulik/Agentic AI/Triseva"
sys.path.insert(0, WORKSPACE_PATH)

WHITELIST_FILE = os.path.join(WORKSPACE_PATH, "config/latin_whitelist.json")

# Global whitelist cache
_whitelist = None
_llm = None

def load_whitelist():
    global _whitelist
    if _whitelist is None:
        if os.path.exists(WHITELIST_FILE):
            try:
                with open(WHITELIST_FILE, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    # Convert to lowercase set for fast lookup
                    _whitelist = {item.lower() for item in data.get("whitelist", [])}
            except Exception as e:
                print(f"[Script Drift] Error loading whitelist: {e}")
                _whitelist = set()
        else:
            print(f"[Script Drift] Warning: Whitelist file {WHITELIST_FILE} not found.")
            _whitelist = set()
    return _whitelist

def _get_llm():
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
    """Normalize Hinglish Latin text to standard Devanagari Hindi using Sarvam's IndicTrans2 model."""
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
    """Detect if text is Hindi or Romanized Hinglish by checking for Devanagari 
    characters or specific Hindi romanized stopwords, excluding common English overlaps."""
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
    """Calculate a continuous script-drift score between 0.0 and 1.0.
    1.0 means perfect script consistency, 0.0 means complete script drift.
    """
    if not text or not text.strip():
        return 1.0
        
    whitelist = load_whitelist()
    
    # Clean the text: remove numbers, percentages, and common units before analyzing script
    cleaned_text = re.sub(r"\b\d+(?:\.\d+)?\b", "", text)  # remove numbers
    
    if script_hint == "latin":
        # Language alignment check:
        # If the expected output script is Latin (Hinglish), but the output contains zero Hinglish words
        # and no Devanagari (meaning it is pure English), it's a complete style mismatch.
        if not is_actual_bilingual(text):
            return 0.0
            
        # Target is Latin Hinglish: Devanagari characters are drift errors
        # Find all Devanagari words using the unicode block range \u0900-\u097f
        all_tokens = re.findall(r"\b\w+(?:-\w+)*\b", cleaned_text)
        if not all_tokens:
            return 1.0
            
        devanagari_tokens = [t for t in all_tokens if re.search(r"[\u0900-\u097f]", t)]
        error_count = len(devanagari_tokens)
        
        score = 1.0 - (error_count / len(all_tokens))
        return round(max(0.0, min(1.0, score)), 3)
        
    elif script_hint == "devanagari":
        # Target is Devanagari: Latin characters are errors unless they are in the whitelist
        if use_normalization:
            # Transliterate first to normalize Hinglish to Devanagari
            cleaned_text = transliterate_to_devanagari(cleaned_text)
            
        all_tokens = re.findall(r"\b[a-zA-Z\u0900-\u097f]+(?:-[a-zA-Z\u0900-\u097f]+)*\b", cleaned_text)
        if not all_tokens:
            return 1.0
            
        latin_tokens = [t for t in all_tokens if re.match(r"^[a-zA-Z-]+$", t)]
        
        # Filter out whitelisted words (case-insensitive)
        non_whitelisted_latin = []
        for token in latin_tokens:
            low_token = token.lower()
            if low_token not in whitelist:
                non_whitelisted_latin.append(token)
                
        error_count = len(non_whitelisted_latin)
        score = 1.0 - (error_count / len(all_tokens))
        return round(max(0.0, min(1.0, score)), 3)
        
    return 1.0
