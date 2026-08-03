from dotenv import load_dotenv
load_dotenv()

import warnings
warnings.filterwarnings("ignore")
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=PendingDeprecationWarning)

import os
import time
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from agents.state import TriSevaState

from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# ── Local / Remote Classifier loading ──────────────────────────────────────────
MODEL_PATH = "data/models/domain_router"
MODEL_HUB_ID = "maddy0494/triseva-router"
_tokenizer = None
_model = None

try:
    # Check if local model weights exist and are complete (>10MB)
    local_weights_path = os.path.join(MODEL_PATH, "model.safetensors")
    if os.path.exists(local_weights_path) and os.path.getsize(local_weights_path) > 10 * 1024 * 1024:
        print("  [Orchestrator] Loading classifier from local weights...")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        print("  [Orchestrator] Successfully loaded local DistilBERT classifier.")
    else:
        print(f"  [Orchestrator] Local weights not found. Loading from Hugging Face Hub '{MODEL_HUB_ID}'...")
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HF_Token")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_HUB_ID, token=hf_token)
        _model = AutoModelForSequenceClassification.from_pretrained(MODEL_HUB_ID, token=hf_token)
        print("  [Orchestrator] Successfully loaded classifier from Hugging Face Hub.")
except Exception as e:
    print(f"  [Orchestrator] Error loading classifier: {e}. Will use Groq fallback.")

# ── LLM — Groq for fast, high-limit routing (fallback) ─────────────────────────
llm = None

# ── Domain classification prompt ──────────────────────────────────────────────
ROUTER_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a domain classifier for a document QA system.
Classify the user query into exactly one of these domains:

- health     : medical reports, symptoms, medications, lab results, diseases, prescriptions
- legal      : government schemes, welfare benefits, RTI, legal rights, eligibility criteria
- agriculture: farming schemes, crop advisory, soil health, irrigation, mandi and MSP, Kisan services

Rules:
- Reply with ONLY one word: health, legal, or agriculture
- No explanation, no punctuation, just the single word
- If genuinely ambiguous, pick the closest match"""),
    ("human", "{query}")
])

router_chain = None


def regex_keyword_router(query: str) -> str:
    """Keyword-based regex fallback routing when both local model and LLM fail."""
    import re
    query_lower = query.lower()
    
    # Define regex patterns for each domain
    health_keywords = r"\b(haemoglobin|anemia|blood|sugar|diabetes|glucose|metformin|thyroid|tsh|cholesterol|kidney|creatinine|fever|covid|sore|throat|cough|dose|prescription|medical|health|patient|disease|treatment|doctor|clinical|tablet|vaccine|hospital|symptom|illness)\b"
    legal_keywords = r"\b(kisan|pmkisan|ayushman|pm-jay|housing|pmay|mgnrega|mnrega|nrega|scholarship|rti|legal|court|bpl|eligibility|scheme|subsidy|pension|tax|government|dlsa|slsa|citizen|act|section|welfare|rights|laws|tribunal|appeals)\b"
    agri_keywords = r"\b(crop|seed|soil|fertiliser|fertilizer|farming|harvest|kharif|rabi|pest|yield|cultivation|monsoon|irrigate|irrigation|drip|sprinkler|mandi|msp|procurement|krishi|kvk|agriculture|advisory|farmer|farmers|pesticide|sowing)\b"
    
    health_matches = len(re.findall(health_keywords, query_lower))
    legal_matches = len(re.findall(legal_keywords, query_lower))
    agri_matches = len(re.findall(agri_keywords, query_lower))
    
    print(f"  [Regex Fallback] Keyword matches - Health: {health_matches}, Legal: {legal_matches}, Agri: {agri_matches}")
    
    if health_matches > legal_matches and health_matches > agri_matches:
        return "health"
    elif legal_matches > health_matches and legal_matches > agri_matches:
        return "legal"
    elif agri_matches > health_matches and agri_matches > legal_matches:
        return "agriculture"
    
    # Check simple substrings
    if "kisan" in query_lower or "scheme" in query_lower or "pm" in query_lower or "yojana" in query_lower:
        if "crop" in query_lower or "soil" in query_lower or "fertilizer" in query_lower or "fertiliser" in query_lower:
            return "agriculture"
        return "legal"
    
    if "soil" in query_lower or "crop" in query_lower or "fertiliser" in query_lower or "fertilizer" in query_lower:
        return "agriculture"
        
    return "health"


# ── Orchestrator node ─────────────────────────────────────────────────────────
def orchestrator_node(state: TriSevaState) -> dict:
    """Classify the user query into a domain and set routing."""

    # On retry, refine the query with critic feedback
    query = state["user_query"]
    image_text = state.get("image_text") or ""
    retry_count = state.get("retry_count", 0)

    # For classification, combine user_query with extracted document image text if available
    classification_text = query
    if image_text:
        classification_text = f"{query} {image_text[:500]}"

    # If the critic has already evaluated an answer, this run is a retry
    if state.get("faithfulness_score") is not None:
        retry_count += 1

    if retry_count > 0:
        print(f"  [Orchestrator] Retry #{retry_count} — re-routing query")

    # Load or initialize telemetry
    telemetry = state.get("telemetry") or {
        "routing_hops": [],
        "retries": 0,
        "latency": 0.0,
        "is_fallback_routing": False,
        "fallback_routing_method": None,
        "is_fallback_critic": False,
        "is_fallback_retrieval": False,
    }

    query_lower = classification_text.strip().lower()

    # High-precision deterministic domain overrides for unambiguous domain markers
    if any(k in query_lower for k in ["prescription", "prescriptions", "rx", "medicine", "medication", "dosage", "tablet", "tablets", "syrup", "capsule", "capsules", "injection", "inj", "tab", "syp", "cap", "opd", "ipd", "clinic", "doctor", "clinical", "patient", "diagnosis", "ayushman", "pm-jay", "pmjay", "haemoglobin", "hemoglobin", "homeoglobin", "hospital", "medline", "pharma", "pharmacy", "medical", "consultant", "dental", "teeth", "white tusk", "augmentin", "enzoflam", "pan d", "hexigel"]):
        print("  [Orchestrator] Deterministic domain match -> HEALTH (confidence: 1.0)")
        telemetry["routing_hops"].append("health")
        return {
            "domain": "health",
            "routing_decision": "health",
            "retry_count": retry_count,
            "telemetry": telemetry,
        }

    if any(k in query_lower for k in ["soil health card", "soil health", "npk", "organic carbon", "fertilizer", "pesticide", "fungicide", "soil test", "krishi", "kvk", "dap", "urea", "crop loss", "farm holding"]):
        print("  [Orchestrator] Deterministic domain match -> AGRICULTURE (confidence: 1.0)")
        telemetry["routing_hops"].append("agriculture")
        return {
            "domain": "agriculture",
            "routing_decision": "agriculture",
            "retry_count": retry_count,
            "telemetry": telemetry,
        }

    if any(k in query_lower for k in ["pm kisan", "pm-kisan", "mgnrega", "mnrega", "nrega", "pmay", "rti", "dpdp", "bns", "bnss", "bsa", "ration card", "aadhaar", "land record", "khasra", "khatauni"]):
        print("  [Orchestrator] Deterministic domain match -> LEGAL (confidence: 1.0)")
        telemetry["routing_hops"].append("legal")
        return {
            "domain": "legal",
            "routing_decision": "legal",
            "retry_count": retry_count,
            "telemetry": telemetry,
        }

    # 1. User Domain Override check
    override = state.get("domain_override")
    if override in ["health", "legal", "agriculture"]:
        print(f"  [Orchestrator] Using user domain override: {override.upper()}")
        telemetry["routing_hops"].append(override)
        return {
            "domain": override,
            "domain_confidence": 1.0,
            "retry_count": retry_count,
            "telemetry": telemetry,
        }

    domain = None
    confidence = 1.0

    # 2. Local DistilBERT Classification on combined query + image text
    if _model is not None and _tokenizer is not None:
        try:
            inputs = _tokenizer(classification_text, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = _model(**inputs)
            
            probs = F.softmax(outputs.logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            confidence = round(probs[pred_idx].item(), 4)
            
            domain_map = {0: "health", 1: "legal", 2: "agriculture"}
            domain = domain_map.get(pred_idx, "health")
            print(f"  [Orchestrator] Local classifier predicted: {domain.upper()} (confidence: {confidence})")
        except Exception as e:
            print(f"  [Orchestrator] Local inference error: {e}. Falling back to Groq.")
            domain = None

    # 3. Fallback: LLM Classification with Regex Fallback on Failure
    if domain is None:
        try:
            global llm, router_chain
            if llm is None:
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    api_key=os.getenv("GROQ_API_KEY") or os.getenv("Groq_API_Key"),
                    temperature=0,
                )
                router_chain = ROUTER_PROMPT | llm

            # Set timeout to 10 seconds for the orchestrator routing LLM
            response = router_chain.invoke({"query": classification_text}, config={"timeout": 10})
            domain = response.content.strip().lower()
            confidence = 1.0

            if domain not in ["health", "legal", "agriculture"]:
                print(f"  [Orchestrator] Unexpected domain '{domain}' — defaulting via regex router")
                domain = regex_keyword_router(classification_text)
                confidence = 0.5
                telemetry["is_fallback_routing"] = True
                telemetry["fallback_routing_method"] = "regex"
            else:
                print(f"  [Orchestrator] Groq fallback predicted: {domain.upper()}")
                telemetry["is_fallback_routing"] = True
                telemetry["fallback_routing_method"] = "groq"
        except Exception as e:
            print(f"  [Orchestrator] Groq fallback failed: {e}. Falling back to Regex Keyword router.")
            domain = regex_keyword_router(query)
            confidence = 0.5
            telemetry["is_fallback_routing"] = True
            telemetry["fallback_routing_method"] = "regex"

    if domain:
        telemetry["routing_hops"].append(domain)

    return {
        "domain": domain,
        "domain_confidence": confidence,
        "retry_count": retry_count,
        "telemetry": telemetry,
    }


# ── Routing function ──────────────────────────────────────────────────────────
def route_to_agent(state: TriSevaState) -> str:
    """Return routing key for conditional edge."""
    domain = state.get("domain", "health")
    if domain in ["health", "legal", "agriculture"]:
        return domain
    return "health"