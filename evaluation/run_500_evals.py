import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
import random
import pandas as pd
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["LANGCHAIN_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

# Import RAG tools and baseline functions
from tools.rag_tool import retrieve
from main import ask
from evaluation.baseline_b1_naive import ask_b1_naive, retrieve_cross_domain
from evaluation.baseline_b3_keyword import ask_b3_keyword, retrieve_bm25

# Ragas Evaluators
eval_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="sarvam-105b",
        openai_api_key=os.getenv("SARVAM_API_KEY"),
        openai_api_base="https://api.sarvam.ai/v1",
        temperature=0.0,
        max_tokens=4096,
        timeout=60,
    ),
    is_finished_parser=lambda x: True
)

eval_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
))

PROGRESS_FILE = "evaluation/results/eval_500_progress.jsonl"
DATASET_FILE = "evaluation/results/triseva_qa_dataset.csv"

def get_500_questions():
    """Load and sample 500 questions with equal domain representation."""
    df = pd.read_csv(DATASET_FILE, encoding="utf-8")
    
    # 166 Health, 166 Legal, 168 Agriculture = 500
    df_health = df[df["domain"] == "health"].sample(n=166, random_state=42)
    df_legal = df[df["domain"] == "legal"].sample(n=166, random_state=42)
    df_agri = df[df["domain"] == "agriculture"].sample(n=168, random_state=42)
    
    combined = pd.concat([df_health, df_legal, df_agri])
    combined = combined.sample(frac=1.0, random_state=42) # shuffle
    
    questions = []
    for _, row in combined.iterrows():
        questions.append({
            "id": row.get("id", ""),
            "question": row["question"],
            "domain": row["domain"],
            "ground_truth": row["answer"] # ground truth
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
            if "rate" in err_msg or "limit" in err_msg or "overloaded" in err_msg or "timeout" in err_msg:
                print(f"     ⚠️ Rate limit/timeout hit: {e}. Retrying in {delay:.1f}s...")
                time.sleep(delay)
                delay *= 2.0
            else:
                # Raise other exceptions directly
                raise e
    # Final try
    return func(*args, **kwargs)

def collect_responses(questions: list):
    """Collect outputs for all 4 pipelines with progress saving and resuming."""
    os.makedirs(os.path.dirname(PROGRESS_FILE), exist_ok=True)
    
    # Load existing progress to support resume
    processed_questions = {}
    if os.path.exists(PROGRESS_FILE):
        with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    item = json.loads(line)
                    processed_questions[item["question"]] = item
        print(f"🔄 Resuming benchmark run: {len(processed_questions)} / {len(questions)} questions already processed.")

    # Open progress file in append mode
    with open(PROGRESS_FILE, "a", encoding="utf-8") as out_f:
        for idx, item in enumerate(questions, 1):
            q = item["question"]
            domain = item["domain"]
            gt = item["ground_truth"]
            
            if q in processed_questions:
                continue

            print(f"\n[{idx:03d}/{len(questions)}] Domain: {domain.upper()} | Question: {q[:60]}...")
            record = {
                "question": q,
                "domain": domain,
                "ground_truth": gt,
                "triseva_answer": "", "triseva_contexts": [], "triseva_latency": 0.0, "triseva_retries": 0,
                "b1_answer": "", "b1_contexts": [],
                "b2_answer": "", "b2_contexts": [],
                "b3_answer": "", "b3_contexts": []
            }

            # --- 1. TriSeva (Full Multi-Agent + Critic) ---
            try:
                print("  🤖 Running TriSeva (Full)...")
                import agents.critic
                agents.critic.FAITHFULNESS_THRESHOLD = 0.7
                agents.critic.MAX_RETRIES = 2
                
                res = run_with_backoff(ask, query=q, session_id=f"eval-500-ts-{idx}")
                record["triseva_answer"] = res.get("answer", "")
                
                chunks = retrieve(q, domain=domain, n_results=3)
                record["triseva_contexts"] = [c["text"] for c in chunks] if chunks else ["No context."]
                record["triseva_latency"] = res.get("telemetry", {}).get("latency", 0.0)
                record["triseva_retries"] = res.get("telemetry", {}).get("retries", 0)
            except Exception as e:
                print(f"     ❌ TriSeva Error: {e}")
                record["triseva_answer"] = f"Error: {e}"

            # --- 2. B1 — Naive RAG (Cross-Domain) ---
            try:
                print("  📄 Running B1 (Naive Cross-Domain)...")
                res = run_with_backoff(ask_b1_naive, q)
                record["b1_answer"] = res.get("answer", "")
                
                chunks = retrieve_cross_domain(q, n_results=3)
                record["b1_contexts"] = [c["text"] for c in chunks] if chunks else ["No context."]
            except Exception as e:
                print(f"     ❌ B1 Error: {e}")
                record["b1_answer"] = f"Error: {e}"

            # --- 3. B2 — No Critic Loop (MAX_RETRIES = 0) ---
            try:
                print("  🛑 Running B2 (No Critic)...")
                import agents.critic
                agents.critic.MAX_RETRIES = 0
                
                res = run_with_backoff(ask, query=q, session_id=f"eval-500-b2-{idx}")
                record["b2_answer"] = res.get("answer", "")
                
                chunks = retrieve(q, domain=domain, n_results=3)
                record["b2_contexts"] = [c["text"] for c in chunks] if chunks else ["No context."]
            except Exception as e:
                print(f"     ❌ B2 Error: {e}")
                record["b2_answer"] = f"Error: {e}"

            # --- 4. B3 — Keyword Search (BM25) ---
            try:
                print("  🔍 Running B3 (BM25 Keyword Search)...")
                res = run_with_backoff(ask_b3_keyword, q, domain=domain)
                record["b3_answer"] = res.get("answer", "")
                
                chunks = retrieve_bm25(q, domain=domain, n_results=3)
                record["b3_contexts"] = [c["text"] for c in chunks] if chunks else ["No context."]
            except Exception as e:
                print(f"     ❌ B3 Error: {e}")
                record["b3_answer"] = f"Error: {e}"

            # Save line progress
            out_f.write(json.dumps(record, ensure_ascii=False) + "\n")
            out_f.flush()
            
            # Rate limit mitigation sleep
            time.sleep(1.2)

def evaluate_progress():
    """Load all records from progress file and run Ragas evaluations."""
    print(f"\n{'='*60}")
    print("RUNNING RAGAS BENCHMARK EVALUATIONS ON COLLECTED DATA")
    print(f"{'='*60}\n")
    
    records = []
    with open(PROGRESS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    if len(records) < 500:
        print(f"⚠️ Warning: Evaluating on partial progress ({len(records)} / 500 questions).")
        
    datasets = {
        "triseva": {"question": [], "answer": [], "contexts": [], "ground_truth": []},
        "b1_naive": {"question": [], "answer": [], "contexts": [], "ground_truth": []},
        "b2_nocritic": {"question": [], "answer": [], "contexts": [], "ground_truth": []},
        "b3_keyword": {"question": [], "answer": [], "contexts": [], "ground_truth": []},
    }
    
    latencies = []
    retries = []
    
    for r in records:
        q = r["question"]
        gt = r["ground_truth"]
        
        datasets["triseva"]["question"].append(q)
        datasets["triseva"]["answer"].append(r["triseva_answer"])
        datasets["triseva"]["contexts"].append(r["triseva_contexts"])
        datasets["triseva"]["ground_truth"].append(gt)
        
        datasets["b1_naive"]["question"].append(q)
        datasets["b1_naive"]["answer"].append(r["b1_answer"])
        datasets["b1_naive"]["contexts"].append(r["b1_contexts"])
        datasets["b1_naive"]["ground_truth"].append(gt)
        
        datasets["b2_nocritic"]["question"].append(q)
        datasets["b2_nocritic"]["answer"].append(r["b2_answer"])
        datasets["b2_nocritic"]["contexts"].append(r["b2_contexts"])
        datasets["b2_nocritic"]["ground_truth"].append(gt)
        
        datasets["b3_keyword"]["question"].append(q)
        datasets["b3_keyword"]["answer"].append(r["b3_answer"])
        datasets["b3_keyword"]["contexts"].append(r["b3_contexts"])
        datasets["b3_keyword"]["ground_truth"].append(gt)
        
        latencies.append(r.get("triseva_latency", 0.0))
        retries.append(r.get("triseva_retries", 0))

    scores = {}
    for name, data in datasets.items():
        print(f"\nEvaluating {name} dataset ({len(data['question'])} questions)...")
        try:
            dataset = Dataset.from_dict(data)
            score_results = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=eval_llm,
                embeddings=eval_embeddings,
                raise_exceptions=False,
            )
            df = score_results.to_pandas()
            scores[name] = {
                "faithfulness": round(float(df["faithfulness"].mean()), 3),
                "answer_relevancy": round(float(df["answer_relevancy"].mean()), 3),
            }
            # Save details
            df.to_csv(f"evaluation/results/eval_500_{name}_details_sarvam.csv", index=False)
        except Exception as e:
            print(f"  [Error evaluating {name}]: {e}")
            scores[name] = {"faithfulness": 0.0, "answer_relevancy": 0.0}

    # Telemetries summary
    num_runs = len(latencies)
    avg_latency = sum(latencies) / num_runs if num_runs > 0 else 0.0
    avg_retries = sum(retries) / num_runs if num_runs > 0 else 0.0

    # Save final scores
    summary = {
        "scores": scores,
        "telemetry": {
            "avg_latency_triseva_sec": round(avg_latency, 2),
            "avg_retries_triseva": round(avg_retries, 2),
            "num_evaluated_questions": num_runs
        }
    }
    with open("evaluation/results/all_baselines_500_results_sarvam.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Print comparison table
    print(f"\n{'='*80}")
    print("500 QUESTION COMPARATIVE BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Configuration                       | Faithfulness | Answer Relevancy | Status")
    print(f"{'-'*80}")
    for name, metrics in scores.items():
        print(f"{name:<35} | {metrics['faithfulness']:.3f}        | {metrics['answer_relevancy']:.3f}          | Success")
    print(f"TriSeva Avg Latency: {avg_latency:.2f} seconds | Critic Avg Retries: {avg_retries:.2f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    # Sample questions
    questions = get_500_questions()
    
    # Collect answers
    collect_responses(questions)
    
    # Evaluate
    evaluate_progress()
