import os
import json
import requests
import re
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

load_dotenv()

CACHE_FILE = "data/translation_cache.json"
SARVAM_API_KEY = os.getenv("SARVAM_API_KEY") or os.getenv("Sarvam_API_Key")

_cache = None
_llm = None

def _get_llm():
    global _llm
    if _llm is None and SARVAM_API_KEY:
        _llm = ChatOpenAI(
            model="sarvam-105b",
            openai_api_key=SARVAM_API_KEY,
            openai_api_base="https://api.sarvam.ai/v1",
            temperature=0.0,
            max_tokens=1536,
            timeout=45,
        )
    return _llm

def _load_cache():
    global _cache
    if _cache is None:
        os.makedirs(os.path.dirname(CACHE_FILE), exist_ok=True)
        if os.path.exists(CACHE_FILE):
            try:
                with open(CACHE_FILE, "r", encoding="utf-8") as f:
                    _cache = json.load(f)
            except Exception as e:
                print(f"[Translation Cache] Error reading cache file: {e}")
                _cache = {}
        else:
            _cache = {}
    return _cache

def _save_cache():
    global _cache
    if _cache is not None:
        try:
            with open(CACHE_FILE, "w", encoding="utf-8") as f:
                json.dump(_cache, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"[Translation Cache] Error saving cache file: {e}")

def is_hindi_or_hinglish(text: str) -> bool:
    """Detect if the text is Hindi (contains Devanagari) or Hinglish (code-mixed)."""
    if not text or not text.strip():
        return False
        
    # 1. Direct Devanagari script check
    if re.search(r"[\u0900-\u097f]", text):
        return True
        
    # 2. Unambiguous Hinglish keywords (excluding common English words like 'is', 'to', 'me', 'he', 'the', 'crop')
    hinglish_keywords = {
        "kya", "hai", "hain", "ko", "se", "ka", "ki", "ke", "mein", 
        "par", "bhi", "aur", "ya", "tha", "thi", "theh", "hoon",
        "karo", "karna", "raha", "rahi", "rahe", "ga", "ge", "gi", "toh", 
        "kab", "jab", "tab", "bhai", "yaar", "bimar", "bimari", "ilaaj",
        "dawa", "khet", "kheti", "mitti", "chhoot", "fasal", "pila", "upay",
        "kisaan", "kisaano", "jameen", "jhum"
    }
    
    words = re.findall(r"\b\w+\b", text.lower())
    match_count = sum(1 for w in words if w in hinglish_keywords)
    
    if len(words) > 0 and (match_count >= 2 or (match_count / len(words)) >= 0.25):
        return True
        
    return False

def translate(text: str, source_lang: str, target_lang: str, script_hint: str = "latin") -> str:
    """Translate text using Sarvam's REST translation API (for hi->en and en->devanagari),
    falling back to Sarvam-105B LLM prompts for natural Latin Hinglish code-mixing."""
    if not text or not text.strip():
        return text
        
    if source_lang == target_lang:
        return text

    cache = _load_cache()
    # Cache key includes target script style hint
    cache_key = f"{text.strip()}|||{source_lang}|||{target_lang}|||{script_hint}"
    
    if cache_key in cache:
        return cache[cache_key]

    # Try high-speed REST translation API first if appropriate
    use_rest = False
    rest_src = None
    rest_tgt = None
    
    if target_lang == "en-IN":
        # Translating Hinglish/Hindi to English
        use_rest = True
        rest_src = "hi-IN"
        rest_tgt = "en-IN"
    elif target_lang == "hi-IN" and script_hint == "devanagari":
        # Translating English to Devanagari Hindi
        use_rest = True
        rest_src = "en-IN"
        rest_tgt = "hi-IN"
        
    if use_rest and SARVAM_API_KEY:
        try:
            url = "https://api.sarvam.ai/translate"
            headers = {
                "Content-Type": "application/json",
                "api-subscription-key": SARVAM_API_KEY
            }
            payload = {
                "input": text.strip(),
                "source_language_code": rest_src,
                "target_language_code": rest_tgt
            }
            print(f"  [Translation REST] Sending request to {url} ({rest_src} -> {rest_tgt})...")
            resp = requests.post(url, json=payload, headers=headers, timeout=10)
            if resp.status_code == 200:
                translated_text = resp.json().get("translated_text", "").strip()
                if translated_text:
                    cache[cache_key] = translated_text
                    _save_cache()
                    print(f"  [Translation REST] Succeeded: '{translated_text[:40]}...'")
                    return translated_text
            print(f"  [Translation REST] Failed with status {resp.status_code}. Falling back to LLM.")
        except Exception as e:
            print(f"  [Translation REST] Error: {e}. Falling back to LLM.")

    # Fallback: General LLM Prompt Translation
    llm = _get_llm()
    if not llm:
        print("[Translation LLM] Warning: LLM not initialized. Returning original text.")
        return text

    # Formulate prompts dynamically for natural Hinglish & Safety inclusion
    if target_lang == "en-IN":
        # Translate Hindi/Hinglish to English
        prompt = f"""You are a professional translator. Translate the following text to standard English. Keep the meaning and domain terms completely accurate.

Text:
{text}

English Translation:"""
    else:
        # Translate English to Hindi/Hinglish
        if script_hint == "devanagari":
            prompt = f"""You are an expert translator specializing in natural code-mixed Hinglish.
Translate the following English text to Hinglish, written in the Devanagari script.

Rules:
1. Use the Devanagari script (Hindi characters).
2. Keep technical domain-specific terms in English but write them phonetically in Devanagari script (e.g. write 'Soil Health Card' as 'सॉइल हेल्थ कार्ड', 'hemoglobin' as 'हीमोग्लोबिन', 'pesticide' as 'पेस्टीसाइड', 'doctor' as 'डॉक्टर', 'symptoms' as 'सिम्पटम्स').
3. Preserve Markdown Tables: Do NOT break or distort GitHub markdown tables (`| Column | Column |`). Keep table headers and numeric values (e.g., 625mg, 1-0-1, pH 7.2) intact.
4. Maintain Hindi grammar, sentence structure, and connectives (like 'है', 'को', 'से', 'होगा').
5. Do not use overly formal Sanskritized Hindi words like 'विषाणु', 'चिकित्सक', 'मृदा' (use 'मिट्टी' or 'सॉइल').
6. Ensure any warning disclaimers or caution notes are translated clearly and accurately.

English Text:
{text}

Devanagari Hinglish Translation:"""
        else:
            prompt = f"""You are an expert translator specializing in natural code-mixed Hinglish.
Translate the following English text to Hinglish, written in the Latin (English) script.

Rules:
1. Use the Latin (English) script.
2. Keep technical domain-specific terms exactly in English (e.g., 'Soil Health Card', 'fertilizer', 'hemoglobin', 'diabetes', 'pesticide', 'symptoms').
3. Preserve Markdown Tables: Maintain all GitHub markdown table boundaries (`| Column | Column |`) intact. Keep column header titles in English (e.g., `Medication`, `Dosage`, `Frequency`, `Timing`, `Duration`), while writing cell explanations in conversational Hinglish.
4. Use colloquial Hinglish phrasing and connectives (like 'hai', 'ko', 'se', 'karna hoga', 'raha hai').
5. Do not translate technical words into formal Hindi. Keep it matching how people naturally text/speak.
6. Ensure any warning disclaimers or caution notes are translated clearly and accurately.

English Text:
{text}

Latin Hinglish Translation:"""

    try:
        res = llm.invoke(prompt)
        translated_text = res.content.strip()
        
        # Clean up any potential markdown wrap by the LLM
        if translated_text.startswith("```"):
            lines = translated_text.split("\n")
            if len(lines) > 2:
                translated_text = "\n".join(lines[1:-1]).strip()

        if translated_text:
            cache[cache_key] = translated_text
            _save_cache()
            return translated_text
        return text
    except Exception as e:
        print(f"[Translation LLM] Request failed: {e}")
        return text
