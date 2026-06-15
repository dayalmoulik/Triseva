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
from ragas.metrics import faithfulness, answer_relevancy, context_precision
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

from tools.rag_tool import retrieve
from main import ask

# ── LLM and embeddings for RAGAS ──────────────────────────────────────────────
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

# ── Test dataset ──────────────────────────────────────────────────────────────
TEST_QUESTIONS = [
    # Health
    {
        "question":    "What is the normal haemoglobin level for adult men?",
        "domain":      "health",
        "ground_truth": "The normal haemoglobin level for adult men is 13.5 to 17.5 g/dL.",
    },
    {
        "question":    "What are the symptoms of anaemia?",
        "domain":      "health",
        "ground_truth": "Symptoms of anaemia include fatigue, weakness, pale skin, and shortness of breath.",
    },
    {
        "question":    "What is the normal blood pressure range?",
        "domain":      "health",
        "ground_truth": "Normal blood pressure is below 120/80 mmHg.",
    },
    {
        "question":    "What does a high TSH level indicate?",
        "domain":      "health",
        "ground_truth": "A TSH level above 4.0 mIU/L suggests hypothyroidism, meaning the thyroid gland is underactive.",
    },
    {
        "question":    "What is the normal fasting blood sugar range?",
        "domain":      "health",
        "ground_truth": "Normal fasting blood sugar is 70 to 100 mg/dL.",
    },

    # Legal
    {
        "question":    "What is the annual benefit amount under PM Kisan scheme?",
        "domain":      "legal",
        "ground_truth": "PM Kisan provides Rs 6000 per year in three equal instalments of Rs 2000 each.",
    },
    {
        "question":    "How many days of employment does MGNREGA guarantee?",
        "domain":      "legal",
        "ground_truth": "MGNREGA guarantees 100 days of wage employment per year to rural households.",
    },
    {
        "question":    "What documents are needed for Ayushman Bharat?",
        "domain":      "legal",
        "ground_truth": "Documents required for Ayushman Bharat include Aadhaar card, ration card, and income certificate.",
    },
    {
        "question":    "What is the health cover amount under PM-JAY?",
        "domain":      "legal",
        "ground_truth": "Ayushman Bharat PM-JAY provides health cover of Rs 5 lakh per family per year.",
    },
    {
        "question":    "What is the fee to file an RTI application?",
        "domain":      "legal",
        "ground_truth": "The fee to file an RTI application is Rs 10 by postal order or demand draft.",
    },

    # Agriculture
    {
        "question":    "How much annual support is provided under PM-KISAN?",
        "domain":      "agriculture",
        "ground_truth": "PM-KISAN provides Rs 6000 per year in three instalments of Rs 2000 each to eligible farmer families.",
    },
    {
        "question":    "What is the farmer premium for Kharif crops under PMFBY?",
        "domain":      "agriculture",
        "ground_truth": "Under PMFBY, farmer premium for Kharif crops is generally 2 percent.",
    },
    {
        "question":    "What is the role of a Soil Health Card?",
        "domain":      "agriculture",
        "ground_truth": "Soil Health Card provides soil nutrient status and fertilizer recommendations for balanced nutrient application.",
    },
    {
        "question":    "What is one benefit of micro-irrigation for farmers?",
        "domain":      "agriculture",
        "ground_truth": "Micro-irrigation such as drip or sprinkler improves water-use efficiency and can reduce input costs.",
    },
    {
        "question":    "Why should farmers use KVK advisories?",
        "domain":      "agriculture",
        "ground_truth": "KVK advisories provide district-level, weather-based guidance on crop practices, nutrients, and pest management.",
    },
]


def collect_comparison_data(questions: list, delay: float = 2.0) -> tuple:
    """Collect evaluation data for both TriSeva (Multi-Agent) and Naive RAG (Single-Agent)."""
    triseva_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    naive_data = {
        "question": [],
        "answer": [],
        "contexts": [],
        "ground_truth": []
    }
    telemetries = []

    print(f"\n{'='*60}")
    print(f"COLLECTING COMPARATIVE DATA — {len(questions)} questions")
    print(f"{'='*60}\n")

    for i, item in enumerate(questions, 1):
        q = item["question"]
        domain = item["domain"]
        gt = item["ground_truth"]

        print(f"[{i:02d}/{len(questions)}] Domain: {domain.upper()} | Question: {q[:50]}...")

        # --- TriSeva Multi-Agent ---
        try:
            print("  🤖 Running TriSeva Multi-Agent...")
            res_ma = ask(query=q, session_id=f"eval-ma-{i}")
            ans_ma = res_ma.get("answer", "")
            
            # Retrieve chunks directly for the evaluation context
            chunks_ma = retrieve(q, domain=domain, n_results=3)
            context_ma = [c["text"] for c in chunks_ma] if chunks_ma else ["No context retrieved."]
            
            triseva_data["question"].append(q)
            triseva_data["answer"].append(ans_ma)
            triseva_data["contexts"].append(context_ma)
            triseva_data["ground_truth"].append(gt)
            
            # Collect telemetry
            telemetry = res_ma.get("telemetry", {})
            telemetries.append(telemetry)
            
            score = res_ma.get("score", 0)
            print(f"     ✅ TriSeva Answer: {len(ans_ma)} chars | Critic score: {score:.2f}")
        except Exception as e:
            print(f"     ❌ TriSeva Error: {e}")
            triseva_data["question"].append(q)
            triseva_data["answer"].append("Error during processing.")
            triseva_data["contexts"].append(["No context retrieved."])
            triseva_data["ground_truth"].append(gt)

        # --- Naive RAG Single-Agent ---
        try:
            print("  📄 Running Naive RAG...")
            from evaluation.baseline_rag import ask_naive_rag
            res_sr = ask_naive_rag(query=q, domain=domain)
            ans_sr = res_sr.get("answer", "")
            context_sr = res_sr.get("retrieved_chunks", ["No context retrieved."])
            
            naive_data["question"].append(q)
            naive_data["answer"].append(ans_sr)
            naive_data["contexts"].append(context_sr)
            naive_data["ground_truth"].append(gt)
            print(f"     ✅ Naive RAG Answer: {len(ans_sr)} chars")
        except Exception as e:
            print(f"     ❌ Naive RAG Error: {e}")
            naive_data["question"].append(q)
            naive_data["answer"].append("Error during processing.")
            naive_data["contexts"].append(["No context retrieved."])
            naive_data["ground_truth"].append(gt)

        if i < len(questions):
            time.sleep(delay)

    return triseva_data, naive_data, telemetries


def run_comparative_evaluation(triseva_data: dict, naive_data: dict):
    """Run RAGAS evaluation on both datasets and compare them."""
    print(f"\n{'='*60}")
    print("RUNNING RAGAS COMPARATIVE EVALUATION")
    print(f"{'='*60}\n")

    # Evaluate TriSeva
    print("Evaluating TriSeva Multi-Agent...")
    ts_dataset = Dataset.from_dict(triseva_data)
    ts_scores = evaluate(
        dataset=ts_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False,
    )

    # Evaluate Naive RAG
    print("Evaluating Naive RAG Single-Agent...")
    nr_dataset = Dataset.from_dict(naive_data)
    nr_scores = evaluate(
        dataset=nr_dataset,
        metrics=[faithfulness, answer_relevancy],
        llm=eval_llm,
        embeddings=eval_embeddings,
        raise_exceptions=False,
    )

    return ts_scores, nr_scores


def save_comparative_results(triseva_data: dict, ts_scores, naive_data: dict, nr_scores, telemetries: list):
    os.makedirs("evaluation/results", exist_ok=True)

    ts_df = ts_scores.to_pandas()
    nr_df = nr_scores.to_pandas()

    ts_summary = {
        "faithfulness": round(float(ts_df["faithfulness"].mean()), 3),
        "answer_relevancy": round(float(ts_df["answer_relevancy"].mean()), 3),
    }

    nr_summary = {
        "faithfulness": round(float(nr_df["faithfulness"].mean()), 3),
        "answer_relevancy": round(float(nr_df["answer_relevancy"].mean()), 3),
    }

    # Aggregate telemetry metrics
    num_runs = len(telemetries)
    if num_runs > 0:
        avg_latency = sum(t.get("latency", 0.0) for t in telemetries) / num_runs
        avg_retries = sum(t.get("retries", 0) for t in telemetries) / num_runs
        
        fallback_routing = sum(1 for t in telemetries if t.get("is_fallback_routing", False))
        fallback_critic = sum(1 for t in telemetries if t.get("is_fallback_critic", False))
        fallback_retrieval = sum(1 for t in telemetries if t.get("is_fallback_retrieval", False))
        
        routing_fallback_pct = (fallback_routing / num_runs) * 100
        critic_fallback_pct = (fallback_critic / num_runs) * 100
        retrieval_fallback_pct = (fallback_retrieval / num_runs) * 100
    else:
        avg_latency = 0.0
        avg_retries = 0.0
        routing_fallback_pct = 0.0
        critic_fallback_pct = 0.0
        retrieval_fallback_pct = 0.0

    comparison = {
        "triseva_multi_agent": ts_summary,
        "naive_rag_single_agent": nr_summary,
        "telemetry_summary": {
            "avg_latency_sec": round(avg_latency, 2),
            "avg_retries": round(avg_retries, 2),
            "routing_fallback_rate_pct": round(routing_fallback_pct, 2),
            "critic_fallback_rate_pct": round(critic_fallback_pct, 2),
            "retrieval_fallback_rate_pct": round(retrieval_fallback_pct, 2),
        }
    }

    # Save to JSON
    with open("evaluation/results/ragas_comparison_results.json", "w") as f:
        json.dump(comparison, f, indent=2)

    # Save details to CSV
    ts_df.to_csv("evaluation/results/ragas_triseva_details.csv", index=False)
    nr_df.to_csv("evaluation/results/ragas_naive_details.csv", index=False)

    # Print summary table
    print(f"\n{'='*60}")
    print("COMPARATIVE EVALUATION SUMMARY")
    print(f"{'='*60}")
    print(f"Metric            | TriSeva (Multi-Agent) | Naive RAG (Single-Agent) | Difference")
    print(f"{'-'*75}")
    
    f_diff = ts_summary["faithfulness"] - nr_summary["faithfulness"]
    r_diff = ts_summary["answer_relevancy"] - nr_summary["answer_relevancy"]
    
    print(f"Faithfulness      | {ts_summary['faithfulness']:.3f}                 | {nr_summary['faithfulness']:.3f}                   | {f_diff:+.3f}")
    print(f"Answer Relevancy  | {ts_summary['answer_relevancy']:.3f}                 | {nr_summary['answer_relevancy']:.3f}                   | {r_diff:+.3f}")
    print(f"{'='*60}\n")

    # Print Multi-Agent Specific Metrics table
    print(f"{'='*60}")
    print("MULTI-AGENT COORDINATION & TELEMETRY METRICS")
    print(f"{'='*60}")
    print(f"Metric                           | Value")
    print(f"{'-'*45}")
    print(f"Average Execution Latency        | {avg_latency:.2f} seconds")
    print(f"Average Negotiation Retries      | {avg_retries:.2f} retries")
    print(f"Orchestrator Fallback Rate       | {routing_fallback_pct:.1f}%")
    print(f"Critic Fallback Rate             | {critic_fallback_pct:.1f}%")
    print(f"Retrieval Fallback Rate          | {retrieval_fallback_pct:.1f}%")
    print(f"{'='*60}\n")

    return comparison


if __name__ == "__main__":
    # Step 1 — collect comparison data
    ts_data, nr_data, telemetries = collect_comparison_data(TEST_QUESTIONS, delay=3.0)

    # Step 2 — run RAGAS evaluation
    ts_scores, nr_scores = run_comparative_evaluation(ts_data, nr_data)

    # Step 3 — save and display comparative results
    save_comparative_results(ts_data, ts_scores, nr_data, nr_scores, telemetries)