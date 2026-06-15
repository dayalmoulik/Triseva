import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
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

# Import questions
from evaluation.ragas_eval import TEST_QUESTIONS

# Setup Ragas LLM & Embeddings
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

def collect_all_data():
    print(f"\n{'='*60}")
    print(f"COLLECTING BENCHMARK DATA FOR ALL PIPELINES — {len(TEST_QUESTIONS)} questions")
    print(f"{'='*60}\n")

    datasets = {
        "triseva": {"question": [], "answer": [], "contexts": [], "ground_truth": []},
        "b1_naive": {"question": [], "answer": [], "contexts": [], "ground_truth": []},
        "b2_nocritic": {"question": [], "answer": [], "contexts": [], "ground_truth": []},
        "b3_keyword": {"question": [], "answer": [], "contexts": [], "ground_truth": []},
    }
    
    telemetries = []

    for i, item in enumerate(TEST_QUESTIONS, 1):
        q = item["question"]
        domain = item["domain"]
        gt = item["ground_truth"]

        print(f"\n[{i:02d}/{len(TEST_QUESTIONS)}] Domain: {domain.upper()} | Question: {q[:60]}...")
        
        # --- 1. TriSeva (Full Multi-Agent + Critic) ---
        try:
            print("  🤖 Running TriSeva (Full)...")
            import agents.critic
            agents.critic.FAITHFULNESS_THRESHOLD = 0.7
            agents.critic.MAX_RETRIES = 2
            
            res = ask(query=q, session_id=f"eval-all-ts-{i}")
            datasets["triseva"]["question"].append(q)
            datasets["triseva"]["answer"].append(res.get("answer", ""))
            
            chunks = retrieve(q, domain=domain, n_results=3)
            datasets["triseva"]["contexts"].append([c["text"] for c in chunks] if chunks else ["No context."])
            datasets["triseva"]["ground_truth"].append(gt)
            telemetries.append(res.get("telemetry", {}))
        except Exception as e:
            print(f"     ❌ TriSeva Error: {e}")
            
        # --- 2. B1 — Naive RAG (Cross-Domain) ---
        try:
            print("  📄 Running B1 (Naive Cross-Domain)...")
            res = ask_b1_naive(q)
            datasets["b1_naive"]["question"].append(q)
            datasets["b1_naive"]["answer"].append(res.get("answer", ""))
            
            chunks = retrieve_cross_domain(q, n_results=3)
            datasets["b1_naive"]["contexts"].append([c["text"] for c in chunks] if chunks else ["No context."])
            datasets["b1_naive"]["ground_truth"].append(gt)
        except Exception as e:
            print(f"     ❌ B1 Error: {e}")

        # --- 3. B2 — No Critic Loop (MAX_RETRIES = 0) ---
        try:
            print("  🛑 Running B2 (No Critic Loop)...")
            import agents.critic
            agents.critic.MAX_RETRIES = 0
            
            res = ask(query=q, session_id=f"eval-all-b2-{i}")
            datasets["b2_nocritic"]["question"].append(q)
            datasets["b2_nocritic"]["answer"].append(res.get("answer", ""))
            
            chunks = retrieve(q, domain=domain, n_results=3)
            datasets["b2_nocritic"]["contexts"].append([c["text"] for c in chunks] if chunks else ["No context."])
            datasets["b2_nocritic"]["ground_truth"].append(gt)
        except Exception as e:
            print(f"     ❌ B2 Error: {e}")

        # --- 4. B3 — Keyword Search (BM25) ---
        try:
            print("  🔍 Running B3 (BM25 Keyword Search)...")
            res = ask_b3_keyword(q, domain=domain)
            datasets["b3_keyword"]["question"].append(q)
            datasets["b3_keyword"]["answer"].append(res.get("answer", ""))
            
            chunks = retrieve_bm25(q, domain=domain, n_results=3)
            datasets["b3_keyword"]["contexts"].append([c["text"] for c in chunks] if chunks else ["No context."])
            datasets["b3_keyword"]["ground_truth"].append(gt)
        except Exception as e:
            print(f"     ❌ B3 Error: {e}")

        time.sleep(1.0) # sleep between questions to respect rate limits
        
    return datasets, telemetries

def evaluate_all(datasets: dict):
    print(f"\n{'='*60}")
    print("EVALUATING DATASETS WITH RAGAS")
    print(f"{'='*60}\n")
    
    scores = {}
    for name, data in datasets.items():
        print(f"Evaluating {name}...")
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
        except Exception as e:
            print(f"  [Error evaluating {name}]: {e}")
            scores[name] = {"faithfulness": 0.0, "answer_relevancy": 0.0}
            
    return scores

if __name__ == "__main__":
    start_time = time.time()
    datasets, telemetries = collect_all_data()
    scores = evaluate_all(datasets)
    
    # Calculate telemetry metrics
    num_runs = len(telemetries)
    avg_latency = sum(t.get("latency", 0.0) for t in telemetries) / num_runs if num_runs > 0 else 0.0
    
    # Save results
    os.makedirs("evaluation/results", exist_ok=True)
    out_path = "evaluation/results/all_baselines_results.json"
    with open(out_path, "w") as f:
        json.dump({
            "scores": scores,
            "avg_latency_triseva": avg_latency,
            "runtime_sec": round(time.time() - start_time, 2)
        }, f, indent=2)
        
    # Print comparison table
    print(f"\n{'='*80}")
    print("COMPARATIVE EVALUATION SUMMARY — ALL BASELINES")
    print(f"{'='*80}")
    print(f"Configuration                       | Faithfulness | Answer Relevancy | Status")
    print(f"{'-'*80}")
    for name, metrics in scores.items():
        print(f"{name:<35} | {metrics['faithfulness']:.3f}        | {metrics['answer_relevancy']:.3f}          | Success")
    print(f"{'='*80}\n")
