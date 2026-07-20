import os
import sys
import json
import re
import random
import pandas as pd
from scipy.stats import pearsonr, spearmanr
from langchain_openai import ChatOpenAI

# Load environment variables
WORKSPACE_PATH = "c:/Users/Moulik/Agentic AI/Triseva"
sys.path.insert(0, WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)

from dotenv import load_dotenv
load_dotenv()

from utils.script_drift_metric import calculate_drift_score, is_actual_bilingual

# Initialize an independent judge LLM representing "human native speakers"
# We use OpenAI GPT-4o-mini to get stable, human-like scoring alignment
def get_human_judge_llm():
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY is not set in the environment or .env file.")
        sys.exit(1)
    return ChatOpenAI(
        model="gpt-4o-mini",
        openai_api_key=api_key,
        temperature=0.0,
        max_tokens=10,
        timeout=30
    )

def get_human_rating(llm, query: str, response: str) -> float:
    """Query the LLM to act as a human annotator rating the script consistency on a scale from 1 to 5."""
    prompt = f"""You are a native Hindi and Hinglish speaker from India.
Rate the script consistency of the following assistant response relative to the user query on a scale of 1 to 5.

User Query (defines expected script style):
{query}

Assistant Response:
{response}

Guidelines:
- 5: Perfect script consistency. The assistant replied in the same script style as the user (Latin Hinglish to Latin Hinglish, Devanagari Hindi to Devanagari Hindi). Proper technical English terms (like "Soil Health Card", "diabetes", etc.) are allowed to stay in English letters and should not be penalized.
- 4: Mostly consistent, but has 1 or 2 minor script leaks (e.g. writing standard words like "khet" or "dawa" in English letters inside a Devanagari answer, or vice-versa).
- 3: Moderately consistent, quite a few script leaks, feels like a random mix of both scripts.
- 2: Poor script consistency, mostly written in the wrong script.
- 1: Completely incorrect script. The user wrote in Devanagari Hindi and the assistant replied entirely in Latin script, or vice-versa.

Output ONLY a single integer between 1 and 5. Do not write any other text."""
    try:
        res = llm.invoke(prompt)
        match = re.search(r"\b[1-5]\b", res.content.strip())
        if match:
            return float(match.group(0))
        return 3.0
    except Exception as e:
        print(f"Warning: Human judge API call failed: {e}")
        return 3.0



def main():
    import argparse
    parser = argparse.ArgumentParser(description="TriSeva Script-Drift Metric Validator")
    parser.add_argument("--size", type=int, default=50, help="Number of samples to validate (default: 50)")
    args = parser.parse_args()
    
    sample_size = args.size
    progress_file = f"evaluation/results/answered_questions_{sample_size}.jsonl"
    
    if not os.path.exists(progress_file):
        print(f"Warning: {progress_file} not found. Falling back to answered_questions_500.jsonl")
        progress_file = "evaluation/results/answered_questions_500.jsonl"
        
    if not os.path.exists(progress_file):
        print(f"Error: Neither answered_questions_{sample_size}.jsonl nor answered_questions_500.jsonl found.")
        sys.exit(1)
        
    # 1. Load execution records
    records = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
                
    print(f"Loaded {len(records)} records from benchmark results.")
    
    # 2. Filter for queries that are Hindi or Hinglish (ignoring pure English questions)
    bilingual_records = [
        r for r in records 
        if is_actual_bilingual(r["question"]) and r.get("triseva_answer")
    ]
    print(f"Found {len(bilingual_records)} bilingual (Hindi/Hinglish) records.")
    
    if len(bilingual_records) < sample_size:
        print(f"Warning: Only {len(bilingual_records)} bilingual records available. Downscaling sample size.")
        sample_size = len(bilingual_records)
        
    # 3. Random sample for validation
    random.seed(42)
    sample_records = random.sample(bilingual_records, sample_size)
    print(f"Sampling {sample_size} records for human vs. metric correlation validation...")
    
    human_judge = get_human_judge_llm()
    
    results = []
    
    for idx, r in enumerate(sample_records, 1):
        query = r["question"]
        answer = r["triseva_answer"]
        
        # Detect script hint from query
        is_devanagari = bool(re.search(r"[\u0900-\u097f]", query))
        script_hint = "devanagari" if is_devanagari else "latin"
        
        # Calculate metric drift score (continuous: 0.0 to 1.0)
        metric_score = calculate_drift_score(answer, script_hint, use_normalization=False)
        
        # Get simulated human rating (1.0 to 5.0)
        human_rating = get_human_rating(human_judge, query, answer)
        
        results.append({
            "query": query,
            "answer": answer,
            "script_hint": script_hint,
            "metric_score": metric_score,
            "human_rating": human_rating
        })
        
        if idx % 10 == 0 or idx == sample_size:
            print(f"Processed {idx}/{sample_size} validation samples.")
            
    # 4. Perform correlation analysis
    df = pd.DataFrame(results)
    
    # Convert human rating from [1, 5] scale to [0, 1] scale for direct comparison
    df["human_score_normalized"] = (df["human_rating"] - 1.0) / 4.0
    
    # Calculate correlation coefficients
    p_corr, p_pval = pearsonr(df["metric_score"], df["human_rating"])
    s_corr, s_pval = spearmanr(df["metric_score"], df["human_rating"])
    
    # Save validation details
    validation_csv = f"evaluation/results/script_drift_validation_{sample_size}.csv"
    df.to_csv(validation_csv, index=False)
    
    print("\n" + "=" * 80)
    print(f"SCRIPT-DRIFT SENSITIVITY METRIC VALIDATION REPORT (N={sample_size})")
    print(f"Saved details to {validation_csv}")
    print("=" * 80)
    print(f"Pearson Correlation Coefficient (r):  {p_corr:.4f} (p-value: {p_pval:.4e})")
    print(f"Spearman's Rank Correlation (rho):   {s_corr:.4f} (p-value: {s_pval:.4e})")
    print(f"Mean Metric Consistency Score:       {df['metric_score'].mean():.3f}")
    print(f"Mean Human Normalized Score:         {df['human_score_normalized'].mean():.3f}")
    print("-" * 80)
    
    # Qualitative analysis: print top 3 matches and mismatches
    df["abs_diff"] = (df["metric_score"] - df["human_score_normalized"]).abs()
    
    def clean_print(s):
        return str(s).encode('ascii', 'ignore').decode('ascii')
        
    print("\nTop 3 Closest Alignments (Metric matches Human):")
    for idx, row in df.sort_values(by="abs_diff").head(3).iterrows():
        print(f"  - Q: {clean_print(row['query'])[:60]}...")
        print(f"    Metric: {row['metric_score']:.3f} | Human (Norm): {row['human_score_normalized']:.3f} (Raw: {row['human_rating']})")
        
    print("\nTop 3 Largest Discrepancies (Metric differs from Human):")
    for idx, row in df.sort_values(by="abs_diff", ascending=False).head(3).iterrows():
        print(f"  - Q: {clean_print(row['query'])[:60]}...")
        print(f"    Metric: {row['metric_score']:.3f} | Human (Norm): {row['human_score_normalized']:.3f} (Raw: {row['human_rating']})")
    print("=" * 80 + "\n")

if __name__ == "__main__":
    main()
