import warnings
warnings.filterwarnings("ignore")

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["LANGCHAIN_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

from dotenv import load_dotenv
load_dotenv()

# --- Monkeypatch MAX_RETRIES to 0 to disable retry loop while keeping the 0.7 threshold constant ---
import agents.critic
agents.critic.MAX_RETRIES = 0

from main import ask

def ask_b2_nocritic(query: str, session_id: str = "eval-b2") -> dict:
    """B2 Baseline: Run TriSeva LangGraph workflow but with critic retries disabled (MAX_RETRIES = 0)."""
    return ask(query=query, session_id=session_id)

if __name__ == "__main__":
    test_query = "What is the annual benefit amount under PM Kisan scheme?"
    print(f"Testing B2 Baseline (No Critic Loop - Retries = 0) on: '{test_query}'")
    res = ask_b2_nocritic(test_query)
    print(f"\nDomain: {res['domain']}")
    print(f"Critic Retries Run: {res['telemetry'].get('retries', 0)}")
    print(f"Answer:\n{res['answer']}")
