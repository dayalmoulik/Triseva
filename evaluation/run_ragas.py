import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import argparse
from pathlib import Path
import pandas as pd

WORKSPACE_PATH = "c:/Users/Moulik/Agentic AI/Triseva"
sys.path.insert(0, WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)

from dotenv import load_dotenv
load_dotenv()

from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings

def get_evaluator_llm():
    provider = os.getenv("EVAL_LLM_PROVIDER", "openai").lower()
    print(f"Initializing Ragas evaluation LLM using provider: {provider}")
    
    if provider == "groq":
        from langchain_groq import ChatGroq
        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY is not set in the environment.")
        model = ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0.0,
            max_tokens=2048,
            timeout=60,
        )
    elif provider == "anthropic":
        from langchain_anthropic import ChatAnthropic
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set in the environment.")
        model = ChatAnthropic(
            model="claude-3-5-haiku-latest",
            api_key=api_key,
            temperature=0.0,
            max_tokens=2048,
            timeout=60,
        )
    else:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY is not set in the environment.")
        model = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=api_key,
            temperature=0.0,
            max_tokens=2048,
            timeout=60,
        )
    return LangchainLLMWrapper(model, is_finished_parser=lambda x: True)

eval_llm = get_evaluator_llm()

eval_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
))

def main():
    parser = argparse.ArgumentParser(description="TriSeva Ragas Evaluator")
    parser.add_argument("--size", type=int, default=50, help="Size of the evaluated dataset")
    args = parser.parse_args()
    
    size = args.size
    progress_file = f"evaluation/results/answered_questions_{size}.jsonl"
    summary_file = f"evaluation/results/ragas_summary_{size}.json"
    
    if not os.path.exists(progress_file):
        print(f"Error: Answered questions file {progress_file} not found. Please run eval_runner.py first.")
        sys.exit(1)
        
    records = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"Loaded {len(records)} answered questions from {progress_file}.")
    
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
        
        # For TriSeva and B2, evaluate the intermediate English draft against the English query
        triseva_q = r.get("triseva_english_query") or q
        triseva_ans = r.get("triseva_draft_answer") or r.get("triseva_answer", "")
        datasets["triseva"]["question"].append(triseva_q)
        datasets["triseva"]["answer"].append(triseva_ans)
        datasets["triseva"]["contexts"].append(r.get("triseva_contexts", ["No context."]))
        datasets["triseva"]["ground_truth"].append(gt)
        
        datasets["b1_naive"]["question"].append(q)
        datasets["b1_naive"]["answer"].append(r.get("b1_answer", ""))
        datasets["b1_naive"]["contexts"].append(r.get("b1_contexts", ["No context."]))
        datasets["b1_naive"]["ground_truth"].append(gt)
        
        b2_q = r.get("b2_english_query") or q
        b2_ans = r.get("b2_draft_answer") or r.get("b2_answer", "")
        datasets["b2_nocritic"]["question"].append(b2_q)
        datasets["b2_nocritic"]["answer"].append(b2_ans)
        datasets["b2_nocritic"]["contexts"].append(r.get("b2_contexts", ["No context."]))
        datasets["b2_nocritic"]["ground_truth"].append(gt)
        
        datasets["b3_keyword"]["question"].append(q)
        datasets["b3_keyword"]["answer"].append(r.get("b3_answer", ""))
        datasets["b3_keyword"]["contexts"].append(r.get("b3_contexts", ["No context."]))
        datasets["b3_keyword"]["ground_truth"].append(gt)
        
        latencies.append(r.get("triseva_latency", 0.0))
        retries.append(r.get("triseva_retries", 0))

    scores = {}
    for name, data in datasets.items():
        print(f"\nEvaluating {name} dataset using Ragas...")
        try:
            # Map standard keys for Ragas compatibility
            ragas_data = {
                "user_input": data["question"],
                "response": data["answer"],
                "retrieved_contexts": data["contexts"],
                "reference": data["ground_truth"]
            }
            dataset = Dataset.from_dict(ragas_data)
            score_results = evaluate(
                dataset=dataset,
                metrics=[faithfulness, answer_relevancy],
                llm=eval_llm,
                embeddings=eval_embeddings,
                raise_exceptions=False,
            )
            df = score_results.to_pandas()
            
            faith_val = 0.0
            relev_val = 0.0
            if 'faithfulness' in df.columns:
                faith_val = round(float(df["faithfulness"].dropna().mean()), 3)
            if 'answer_relevancy' in df.columns:
                relev_val = round(float(df["answer_relevancy"].dropna().mean()), 3)
                
            scores[name] = {
                "faithfulness": faith_val,
                "answer_relevancy": relev_val,
            }
            
            # Save details to CSV
            details_csv = f"evaluation/results/ragas_details_{name}_{size}.csv"
            df.to_csv(details_csv, index=False)
            print(f"  Saved details to {details_csv}")
        except Exception as e:
            print(f"  [Error evaluating {name}]: {e}")
            scores[name] = {"faithfulness": 0.0, "answer_relevancy": 0.0}

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
        },
        "metadata": {
            "evaluator": "ragas",
            "model": "gpt-4o-mini"
        }
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary results saved to {summary_file}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{size} QUESTION COMPARATIVE RAGAS BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Configuration                       | Faithfulness | Answer Relevancy | Status")
    print(f"{'-'*80}")
    for name, metrics in scores.items():
        print(f"{name:<35} | {metrics['faithfulness']:.3f}        | {metrics['answer_relevancy']:.3f}          | Success")
    print(f"TriSeva Avg Latency: {avg_latency:.2f} seconds | Critic Avg Retries: {avg_retries:.2f}")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
