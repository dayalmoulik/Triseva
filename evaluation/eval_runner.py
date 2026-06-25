import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
import argparse
import pandas as pd
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

# Turn off LangChain tracing to prevent rate limits and console pollution
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_PROJECT"] = ""

WORKSPACE_PATH = "c:/Users/Moulik/Agentic AI/Triseva"
sys.path.insert(0, WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)

from dotenv import load_dotenv
load_dotenv()

# Import RAG tools and baseline functions
from tools.rag_tool import retrieve
from main import ask
from evaluation.baseline_b1_naive import ask_b1_naive, retrieve_cross_domain
from evaluation.baseline_b3_keyword import ask_b3_keyword, retrieve_bm25

DATASET_FILE = "evaluation/results/triseva_qa_dataset.csv"

def get_sampled_questions(size: int):
    """Load and sample N questions with equal domain representation from the dataset CSV."""
    if not os.path.exists(DATASET_FILE):
        raise FileNotFoundError(f"Dataset file {DATASET_FILE} not found.")

    df = pd.read_csv(DATASET_FILE, encoding="utf-8")
    
    # Calculate counts per domain
    n_health = size // 3
    n_legal = size // 3
    n_agri = size - (n_health + n_legal)
    
    print(f"Sampling {size} questions:")
    print(f"  Health:      {n_health}")
    print(f"  Legal:       {n_legal}")
    print(f"  Agriculture: {n_agri}")
    
    # Sample from each domain
    df_health = df[df["domain"] == "health"].sample(n=n_health, random_state=42)
    df_legal = df[df["domain"] == "legal"].sample(n=n_legal, random_state=42)
    df_agri = df[df["domain"] == "agriculture"].sample(n=n_agri, random_state=42)
    
    combined = pd.concat([df_health, df_legal, df_agri])
    combined = combined.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    
    questions = []
    for idx, row in combined.iterrows():
        questions.append({
            "id": row.get("id", f"q_{idx}"),
            "question": row["question"],
            "domain": row["domain"],
            "ground_truth": row["answer"]
        })
    return questions

def run_with_backoff(func, *args, max_retries=5, initial_delay=2.0, **kwargs):
    """Run an API function with exponential backoff on rate limits or timeouts."""
    delay = initial_delay
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            err_msg = str(e).lower()
            if any(term in err_msg for term in ["rate", "limit", "overloaded", "timeout"]):
                print(f"     ⚠️ API issue hit: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                raise e
    return func(*args, **kwargs)

def process_single_question(record, idx, size):
    q = record["question"]
    domain = record["domain"]
    
    # 1. TriSeva (Full Multi-Agent + Critic)
    try:
        print(f"  🤖 [Q{idx}/{size}] Running TriSeva (Full)...")
        import agents.critic
        agents.critic.FAITHFULNESS_THRESHOLD = 0.7
        agents.critic.MAX_RETRIES = 2
        
        res = run_with_backoff(ask, query=q, session_id=f"eval-{size}-ts-{idx}")
        record["triseva_answer"] = res.get("answer", "")
        
        chunks = retrieve(q, domain=domain, n_results=3)
        record["triseva_contexts"] = [c["text"] for c in chunks] if chunks else ["No context."]
        record["triseva_latency"] = res.get("telemetry", {}).get("latency", 0.0)
        record["triseva_retries"] = res.get("telemetry", {}).get("retries", 0)
    except Exception as e:
        print(f"     ❌ TriSeva Error Q{idx}: {e}")
        record["triseva_answer"] = f"Error: {e}"
        record["triseva_contexts"] = ["No context."]
        record["triseva_latency"] = 0.0
        record["triseva_retries"] = 0

    # 2. B1 — Naive RAG (Cross-Domain)
    try:
        print(f"  📄 [Q{idx}/{size}] Running B1 (Naive Cross-Domain)...")
        res = run_with_backoff(ask_b1_naive, q)
        record["b1_answer"] = res.get("answer", "")
        
        chunks = retrieve_cross_domain(q, n_results=3)
        record["b1_contexts"] = [c["text"] for c in chunks] if chunks else ["No context."]
    except Exception as e:
        print(f"     ❌ B1 Error Q{idx}: {e}")
        record["b1_answer"] = f"Error: {e}"
        record["b1_contexts"] = ["No context."]

    # 3. B2 — No Critic Loop (MAX_RETRIES = 0)
    try:
        print(f"  🛑 [Q{idx}/{size}] Running B2 (No Critic)...")
        import agents.critic
        agents.critic.MAX_RETRIES = 0
        
        res = run_with_backoff(ask, query=q, session_id=f"eval-{size}-b2-{idx}")
        record["b2_answer"] = res.get("answer", "")
        
        chunks = retrieve(q, domain=domain, n_results=3)
        record["b2_contexts"] = [c["text"] for c in chunks] if chunks else ["No context."]
    except Exception as e:
        print(f"     ❌ B2 Error Q{idx}: {e}")
        record["b2_answer"] = f"Error: {e}"
        record["b2_contexts"] = ["No context."]

    # 4. B3 — Keyword Search (BM25)
    try:
        print(f"  🔍 [Q{idx}/{size}] Running B3 (BM25 Keyword Search)...")
        res = run_with_backoff(ask_b3_keyword, q, domain=domain)
        record["b3_answer"] = res.get("answer", "")
        
        chunks = retrieve_bm25(q, domain=domain, n_results=3)
        record["b3_contexts"] = [c["text"] for c in chunks] if chunks else ["No context."]
    except Exception as e:
        print(f"     ❌ B3 Error Q{idx}: {e}")
        record["b3_answer"] = f"Error: {e}"
        record["b3_contexts"] = ["No context."]

    return record

def main():
    parser = argparse.ArgumentParser(description="TriSeva Baseline Response Collector")
    parser.add_argument("--size", type=int, default=50, help="Number of questions to sample and process")
    args = parser.parse_args()
    
    size = args.size
    progress_file = f"evaluation/results/answered_questions_{size}.jsonl"
    os.makedirs(os.path.dirname(progress_file), exist_ok=True)
    
    # 1. Sample questions
    questions = get_sampled_questions(size)
    
    # 2. Check resume state
    processed_questions = {}
    if os.path.exists(progress_file):
        with open(progress_file, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_questions[item["question"]] = item
        print(f"🔄 Resuming collection: {len(processed_questions)} / {size} questions already processed.")
        
    to_process = [q for q in questions if q["question"] not in processed_questions]
    print(f"Remaining questions to process: {len(to_process)}")
    
    if not to_process:
        print("✅ All questions are already processed!")
        return

    # 3. Process in parallel
    num_workers = 3
    print(f"\nProcessing {len(to_process)} questions using {num_workers} parallel workers...")
    
    with open(progress_file, "a", encoding="utf-8") as out_f:
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            future_to_record = {
                executor.submit(process_single_question, dict(record), idx, size): (record, idx)
                for idx, record in enumerate(to_process, len(processed_questions) + 1)
            }
            
            completed = 0
            for future in as_completed(future_to_record):
                record, idx = future_to_record[future]
                try:
                    updated_record = future.result()
                    out_f.write(json.dumps(updated_record, ensure_ascii=False) + "\n")
                    out_f.flush()
                    completed += 1
                    print(f"[{completed}/{len(to_process)}] Completed Q{idx}: {updated_record['question'][:50]}...")
                except Exception as e:
                    print(f"🔴 Error processing Q{idx}: {e}")

    print(f"🎉 Answer collection complete! Saved to {progress_file}")

if __name__ == "__main__":
    main()
