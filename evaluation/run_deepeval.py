import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import pandas as pd

# Opt out of DeepEval telemetry/Confident AI platform logging to avoid credentials prompts
os.environ["DEEPEVAL_TELEMETRY_OPT_OUT"] = "YES"

WORKSPACE_PATH = "c:/Users/Moulik/Agentic AI/Triseva"
sys.path.insert(0, WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)

from dotenv import load_dotenv
load_dotenv()

from deepeval.metrics import FaithfulnessMetric, AnswerRelevancyMetric
from deepeval.test_case import LLMTestCase

def evaluate_single_case(q, answer, contexts, gt, name, idx):
    """Evaluate a single test case for a specific configuration using DeepEval."""
    # Build test case
    test_case = LLMTestCase(
        input=q,
        actual_output=answer or "No answer.",
        retrieval_context=contexts or ["No context."],
        expected_output=gt
    )
    
    # Initialize metrics
    # Force model="gpt-4o-mini"
    faithfulness_metric = FaithfulnessMetric(threshold=0.7, model="gpt-4o-mini", async_mode=False)
    relevancy_metric = AnswerRelevancyMetric(threshold=0.7, model="gpt-4o-mini", async_mode=False)
    
    try:
        faithfulness_metric.measure(test_case)
        f_score = faithfulness_metric.score
        f_reason = faithfulness_metric.reason
    except Exception as e:
        print(f"  [Q{idx} {name}] Faithfulness Evaluation Error: {e}")
        f_score = 0.0
        f_reason = str(e)
        
    try:
        relevancy_metric.measure(test_case)
        r_score = relevancy_metric.score
        r_reason = relevancy_metric.reason
    except Exception as e:
        print(f"  [Q{idx} {name}] Relevancy Evaluation Error: {e}")
        r_score = 0.0
        r_reason = str(e)
        
    return {
        "configuration": name,
        "index": idx,
        "faithfulness": round(float(f_score), 3),
        "faithfulness_reason": f_reason,
        "answer_relevancy": round(float(r_score), 3),
        "relevancy_reason": r_reason
    }

def main():
    parser = argparse.ArgumentParser(description="TriSeva DeepEval Evaluator")
    parser.add_argument("--size", type=int, default=50, help="Size of the evaluated dataset")
    parser.add_argument("--workers", type=int, default=5, help="Number of parallel evaluation threads")
    args = parser.parse_args()
    
    size = args.size
    num_workers = args.workers
    progress_file = f"evaluation/results/answered_questions_{size}.jsonl"
    summary_file = f"evaluation/results/deepeval_summary_{size}.json"
    
    if not os.path.exists(progress_file):
        print(f"Error: Answered questions file {progress_file} not found. Please run eval_runner.py first.")
        sys.exit(1)
        
    records = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"Loaded {len(records)} answered questions from {progress_file}.")
    
    # We will evaluate all 4 configurations: triseva, b1_naive, b2_nocritic, b3_keyword
    eval_tasks = []
    for idx, r in enumerate(records, 1):
        q = r["question"]
        gt = r["ground_truth"]
        
        eval_tasks.append((q, r.get("triseva_answer", ""), r.get("triseva_contexts", []), gt, "triseva", idx))
        eval_tasks.append((q, r.get("b1_answer", ""), r.get("b1_contexts", []), gt, "b1_naive", idx))
        eval_tasks.append((q, r.get("b2_answer", ""), r.get("b2_contexts", []), gt, "b2_nocritic", idx))
        eval_tasks.append((q, r.get("b3_answer", ""), r.get("b3_contexts", []), gt, "b3_keyword", idx))
        
    print(f"Total metrics to compute: {len(eval_tasks)} test cases (Faithfulness & Relevancy each).")
    
    results = {
        "triseva": [],
        "b1_naive": [],
        "b2_nocritic": [],
        "b3_keyword": []
    }
    
    completed = 0
    print(f"\nRunning evaluations in parallel via DeepEval (gpt-4o-mini) using {num_workers} workers...")
    
    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        future_to_task = {
            executor.submit(evaluate_single_case, q, ans, ctx, gt, name, idx): (name, idx)
            for q, ans, ctx, gt, name, idx in eval_tasks
        }
        
        for future in as_completed(future_to_task):
            name, idx = future_to_task[future]
            try:
                res = future.result()
                results[res["configuration"]].append(res)
                completed += 1
                if completed % 10 == 0 or completed == len(eval_tasks):
                    print(f"[{completed}/{len(eval_tasks)}] Completed test case evaluations.")
            except Exception as e:
                print(f"Error evaluating task {idx} for {name}: {e}")
                
    # Calculate average scores
    scores = {}
    for name, list_res in results.items():
        avg_f = sum(r["faithfulness"] for r in list_res) / len(list_res) if list_res else 0.0
        avg_r = sum(r["answer_relevancy"] for r in list_res) / len(list_res) if list_res else 0.0
        scores[name] = {
            "faithfulness": round(avg_f, 3),
            "answer_relevancy": round(avg_r, 3)
        }
        
        # Save details to CSV
        df = pd.DataFrame(list_res).sort_values(by="index")
        details_csv = f"evaluation/results/deepeval_details_{name}_{size}.csv"
        df.to_csv(details_csv, index=False)
        print(f"  Saved details for {name} to {details_csv}")
        
    # Save summary results
    summary = {
        "scores": scores,
        "metadata": {
            "evaluator": "deepeval",
            "model": "gpt-4o-mini",
            "num_evaluated_questions": len(records)
        }
    }
    with open(summary_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary results saved to {summary_file}")

    # Print comparison table
    print(f"\n{'='*80}")
    print(f"{size} QUESTION DEEPEVAL COMPARATIVE BENCHMARK SUMMARY")
    print(f"{'='*80}")
    print(f"Configuration                       | Faithfulness | Answer Relevancy | Status")
    print(f"{'-'*80}")
    for name, metrics in scores.items():
        print(f"{name:<35} | {metrics['faithfulness']:.3f}        | {metrics['answer_relevancy']:.3f}          | Success")
    print(f"{'='*80}\n")

if __name__ == "__main__":
    main()
