import sys
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

import os
import json
import argparse
import pandas as pd
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import context_precision, context_recall
from langchain_openai import ChatOpenAI
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from langchain_community.embeddings import HuggingFaceEmbeddings
from bert_score import score as bert_score_fn
from dotenv import load_dotenv

load_dotenv()

# Fixed globals
DATASET_JSON = "evaluation/results/triseva_qa_dataset.json"
MODEL_PATH = "data/models/domain_router"

# Ragas Evaluators Setup (defaulting to Sarvam-105B API)
eval_llm = LangchainLLMWrapper(
    ChatOpenAI(
        model="sarvam-105b",
        openai_api_key=os.getenv("SARVAM_API_KEY"),
        openai_api_base="https://api.sarvam.ai/v1",
        temperature=0.0,
        max_tokens=2048,
        timeout=60,
    ),
    is_finished_parser=lambda x: True
)

eval_embeddings = LangchainEmbeddingsWrapper(HuggingFaceEmbeddings(
    model_name="all-MiniLM-L6-v2"
))


def load_progress_data(progress_file):
    """Load the progress file."""
    if not os.path.exists(progress_file):
        raise FileNotFoundError(f"Progress file {progress_file} does not exist.")
    
    print(f"📂 Loading progress data from: {progress_file}")
    records = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def run_domain_routing_metrics(records, cm_plot_path):
    """Run local DistilBERT classifier on questions to evaluate F1 and confusion matrix."""
    print("\n🔍 Running Domain Routing Evaluation...")
    if not os.path.exists(MODEL_PATH):
        print("  ⚠️ Local classifier model not found. Skipping classifier evaluation.")
        return {}, None
        
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
        model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH)
        
        questions = [r["question"] for r in records]
        true_domains = [r["domain"] for r in records]
        
        domain_map = {0: "health", 1: "legal", 2: "agriculture"}
        domain_to_lbl = {v: k for k, v in domain_map.items()}
        
        pred_domains = []
        for i, q in enumerate(questions):
            inputs = tokenizer(q, return_tensors="pt", truncation=True, padding=True, max_length=128)
            with torch.no_grad():
                outputs = model(**inputs)
            probs = F.softmax(outputs.logits, dim=-1)[0]
            pred_idx = torch.argmax(probs).item()
            pred_domains.append(domain_map.get(pred_idx, "health"))
            
        # Metrics
        y_true = [domain_to_lbl[d] for d in true_domains]
        y_pred = [domain_to_lbl[d] for d in pred_domains]
        
        report = classification_report(y_true, y_pred, target_names=list(domain_map.values()), output_dict=True)
        print("  ✅ Domain Router Classification Report completed.")
        
        # Plot Confusion Matrix
        cm = confusion_matrix(true_domains, pred_domains, labels=["health", "legal", "agriculture"])
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=["health", "legal", "agriculture"], 
                    yticklabels=["health", "legal", "agriculture"])
        plt.title("Orchestrator Routing Confusion Matrix")
        plt.ylabel("Actual Domain")
        plt.xlabel("Predicted Domain")
        plt.tight_layout()
        plt.savefig(cm_plot_path)
        plt.close()
        print(f"  🖼️ Confusion matrix plot saved to: {cm_plot_path}")
        
        return report, cm.tolist()
        
    except Exception as e:
        print(f"  ❌ Error running domain routing metrics: {e}")
        return {}, None


def run_ragas_context_metrics(records):
    """Run Ragas evaluations for Context Precision and Context Recall on all 500 questions."""
    print("\n🔍 Running Ragas Context Metrics (Precision & Recall) on all 500 questions...")
    
    data = {
        "question": [],
        "contexts": [],
        "ground_truth": []
    }
    
    for r in records:
        # Check triseva contexts
        contexts = r.get("triseva_contexts", [])
        if not contexts or contexts == ["No context."]:
            contexts = ["No reference context retrieved."]
            
        data["question"].append(r["question"])
        data["contexts"].append(contexts)
        data["ground_truth"].append(r["ground_truth"])
        
    try:
        dataset = Dataset.from_dict(data)
        print(f"  🚀 Evaluating Context Precision & Recall on all {len(data['question'])} questions...")
        score_results = evaluate(
            dataset=dataset,
            metrics=[context_precision, context_recall],
            llm=eval_llm,
            embeddings=eval_embeddings,
            raise_exceptions=False,
        )
        
        df = score_results.to_pandas()
        precision = round(float(df["context_precision"].mean()), 3)
        recall = round(float(df["context_recall"].mean()), 3)
        print(f"  ✅ Ragas results: Context Precision = {precision:.3f} | Context Recall = {recall:.3f}")
        return {"context_precision": precision, "context_recall": recall}
    except Exception as e:
        print(f"  ❌ Ragas evaluation failed: {e}")
        return {"context_precision": 0.0, "context_recall": 0.0}


def run_language_stratification(ts_csv):
    """Group faithfulness and relevancy by query language (EN / HI / Hinglish)."""
    print("\n🔍 Running Language Stratification...")
    if not os.path.exists(ts_csv) or not os.path.exists(DATASET_JSON):
        print(f"  ⚠️ Required files ({ts_csv} or {DATASET_JSON}) missing. Skipping language stratification.")
        return {}
        
    try:
        # Load triseva scores
        ts_df = pd.read_csv(ts_csv)
        
        # Load dataset JSON to map questions to language
        with open(DATASET_JSON, "r", encoding="utf-8") as f:
            dataset_pairs = json.load(f)
            
        q_to_lang = {p["question"].strip().lower(): p["language"] for p in dataset_pairs}
        
        langs = []
        for q in ts_df["user_input"]:
            q_clean = q.strip().lower()
            langs.append(q_to_lang.get(q_clean, "en")) # default to en if missing
            
        ts_df["language"] = langs
        
        summary = ts_df.groupby("language")[["faithfulness", "answer_relevancy"]].mean().round(3).to_dict(orient="index")
        
        print("  ✅ Language-stratified performance metrics:")
        for lang, metrics in summary.items():
            print(f"     - {lang:<8}: Faithfulness = {metrics['faithfulness']:.3f} | Relevancy = {metrics['answer_relevancy']:.3f}")
        return summary
        
    except Exception as e:
        print(f"  ❌ Error running language stratification: {e}")
        return {}


def run_critic_retry_metrics(records, ts_csv, b2_csv):
    """Compute overall Retry Rate and Faithfulness Delta (TriSeva vs. B2 No-Critic)."""
    print("\n🔍 Running Critic Retry & Faithfulness Delta Evaluation...")
    
    # 1. Retry Rate
    retries = [r.get("triseva_retries", 0) for r in records]
    num_runs = len(retries)
    retry_runs = sum(1 for r in retries if r > 0)
    retry_rate = (retry_runs / num_runs) * 100 if num_runs > 0 else 0.0
    avg_retries = sum(retries) / num_runs if num_runs > 0 else 0.0
    
    print(f"  ✅ Retry Rate: {retry_rate:.1f}% | Avg Retries: {avg_retries:.2f}")
    
    # 2. Faithfulness Delta (TriSeva vs. B2)
    f_delta = 0.0
    if os.path.exists(ts_csv) and os.path.exists(b2_csv):
        try:
            ts_df = pd.read_csv(ts_csv)
            b2_df = pd.read_csv(b2_csv)
            
            # Match questions
            merged = pd.merge(ts_df, b2_df, on="user_input", suffixes=("_ts", "_b2"))
            
            # Calculate overall delta
            avg_f_ts = merged["faithfulness_ts"].mean()
            avg_f_b2 = merged["faithfulness_b2"].mean()
            f_delta = avg_f_ts - avg_f_b2
            
            # Filter to retried questions to see the exact improvement due to revision
            retried_qs = {r["question"].strip().lower() for r in records if r.get("triseva_retries", 0) > 0}
            merged["is_retried"] = merged["user_input"].apply(lambda q: q.strip().lower() in retried_qs)
            
            retried_merged = merged[merged["is_retried"]]
            if not retried_merged.empty:
                delta_retried = retried_merged["faithfulness_ts"].mean() - retried_merged["faithfulness_b2"].mean()
                print(f"  ✅ Faithfulness Delta (All): {f_delta:+.3f}")
                print(f"  ✅ Faithfulness Delta (Retried Only): {delta_retried:+.3f}")
            else:
                print(f"  ✅ Faithfulness Delta (All): {f_delta:+.3f}")
        except Exception as e:
            print(f"  ❌ Error matching TS and B2 CSVs: {e}")
            
    return {
        "retry_rate_pct": round(retry_rate, 2),
        "avg_retries": round(avg_retries, 2),
        "faithfulness_delta": round(f_delta, 3)
    }


def run_bertscore_metrics(records):
    """Compute multilingual BERTScore comparing triseva_answer to ground_truth."""
    print("\n🔍 Running Multilingual BERTScore Evaluation...")
    
    answers = [r["triseva_answer"] for r in records if r.get("triseva_answer")]
    ground_truths = [r["ground_truth"] for r in records if r.get("triseva_answer")]
    
    if not answers:
        print("  ⚠️ No triseva answers found. Skipping BERTScore.")
        return {}
        
    try:
        print("  🚀 Computing BERTScore using xlm-roberta-base...")
        P, R, F1 = bert_score_fn(answers, ground_truths, model_type="xlm-roberta-base", lang="multilingual")
        
        avg_p = round(float(P.mean().item()), 3)
        avg_r = round(float(R.mean().item()), 3)
        avg_f1 = round(float(F1.mean().item()), 3)
        
        print(f"  ✅ BERTScore: Precision = {avg_p:.3f} | Recall = {avg_r:.3f} | F1 = {avg_f1:.3f}")
        return {
            "bertscore_precision": avg_p,
            "bertscore_recall": avg_r,
            "bertscore_f1": avg_f1
        }
    except Exception as e:
        print(f"  ❌ BERTScore calculation failed: {e}")
        return {}


def main():
    parser = argparse.ArgumentParser(description="Run advanced evaluations.")
    parser.add_argument("--provider", type=str, choices=["claude", "sarvam"], default="claude", help="Log provider (claude or sarvam)")
    args = parser.parse_args()
    
    provider = args.provider
    print("=" * 60)
    print(f"STARTING ADVANCED EVALUATIONS FOR: {provider.upper()} ON ALL 500 QA")
    print("=" * 60)
    
    # Path configuration
    if provider == "claude":
        progress_file = "evaluation/results/eval_500_progress_claude.jsonl"
        ts_csv = "evaluation/results/eval_500_triseva_details_claude.csv"
        b2_csv = "evaluation/results/eval_500_b2_nocritic_details_claude.csv"
        output_summary = "evaluation/results/advanced_evals_summary_claude.json"
        cm_plot_path = "evaluation/results/routing_confusion_matrix_claude.png"
    else:
        progress_file = "evaluation/results/eval_500_progress.jsonl"
        ts_csv = "evaluation/results/eval_500_triseva_details_sarvam.csv"
        b2_csv = "evaluation/results/eval_500_b2_nocritic_details_sarvam.csv"
        output_summary = "evaluation/results/advanced_evals_summary_sarvam.json"
        cm_plot_path = "evaluation/results/routing_confusion_matrix_sarvam.png"
    
    # 1. Load progress log data
    try:
        records = load_progress_data(progress_file)
    except FileNotFoundError as e:
        print(f"❌ {e}")
        return
    
    # 2. Run metrics
    routing_report, cm = run_domain_routing_metrics(records, cm_plot_path)
    context_metrics = run_ragas_context_metrics(records)
    lang_metrics = run_language_stratification(ts_csv)
    critic_metrics = run_critic_retry_metrics(records, ts_csv, b2_csv)
    bertscore_metrics = run_bertscore_metrics(records)
    
    # 3. Save output
    summary = {
        "provider": provider,
        "progress_file_evaluated": progress_file,
        "domain_routing": {
            "classification_report": routing_report,
            "confusion_matrix": cm
        },
        "ragas_context_metrics": context_metrics,
        "language_stratified": lang_metrics,
        "critic_retry_analysis": critic_metrics,
        "bertscore": bertscore_metrics
    }
    
    os.makedirs(os.path.dirname(output_summary), exist_ok=True)
    with open(output_summary, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
        
    print("\n" + "=" * 60)
    print(f"✅ Advanced Evaluations Complete! Summary saved to: {output_summary}")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
