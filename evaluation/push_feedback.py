import os
import sys
import json
import argparse
import pandas as pd
from langsmith import Client

# Load environment variables
WORKSPACE_PATH = "c:/Users/Moulik/Agentic AI/Triseva"
sys.path.insert(0, WORKSPACE_PATH)
os.chdir(WORKSPACE_PATH)

from dotenv import load_dotenv
load_dotenv()

def main():
    parser = argparse.ArgumentParser(description="TriSeva LangSmith Feedback Uploader")
    parser.add_argument("--size", type=int, default=50, help="Size of the evaluated dataset")
    args = parser.parse_args()
    
    size = args.size
    progress_file = f"evaluation/results/answered_questions_{size}.jsonl"
    
    if not os.path.exists(progress_file):
        print(f"Error: Answered questions file {progress_file} not found. Please run eval_runner.py first.")
        sys.exit(1)
        
    # Initialize LangSmith client
    api_key = os.getenv("LANGCHAIN_API_KEY")
    if not api_key:
        print("Error: LANGCHAIN_API_KEY is not set in the environment or .env file.")
        sys.exit(1)
        
    print(f"Initializing LangSmith client...")
    client = Client(api_key=api_key)
    
    # Load execution records
    records = []
    with open(progress_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    print(f"Loaded {len(records)} execution records.")
    
    # Load Ragas CSVs if they exist
    ragas_triseva_df = None
    ragas_b2_df = None
    
    ragas_triseva_path = f"evaluation/results/ragas_details_triseva_{size}.csv"
    ragas_b2_path = f"evaluation/results/ragas_details_b2_nocritic_{size}.csv"
    
    if os.path.exists(ragas_triseva_path):
        ragas_triseva_df = pd.read_csv(ragas_triseva_path)
        print(f"Loaded Ragas details for TriSeva from {ragas_triseva_path}")
    if os.path.exists(ragas_b2_path):
        ragas_b2_df = pd.read_csv(ragas_b2_path)
        print(f"Loaded Ragas details for B2 from {ragas_b2_path}")
        
    # Load DeepEval CSVs if they exist
    deepeval_triseva_df = None
    deepeval_b2_df = None
    
    deepeval_triseva_path = f"evaluation/results/deepeval_details_triseva_{size}.csv"
    deepeval_b2_path = f"evaluation/results/deepeval_details_b2_nocritic_{size}.csv"
    
    if os.path.exists(deepeval_triseva_path):
        deepeval_triseva_df = pd.read_csv(deepeval_triseva_path)
        print(f"Loaded DeepEval details for TriSeva from {deepeval_triseva_path}")
    if os.path.exists(deepeval_b2_path):
        deepeval_b2_df = pd.read_csv(deepeval_b2_path)
        print(f"Loaded DeepEval details for B2 from {deepeval_b2_path}")
        
    if not any([ragas_triseva_df is not None, ragas_b2_df is not None, deepeval_triseva_df is not None, deepeval_b2_df is not None]):
        print("Warning: No Ragas or DeepEval details CSV files found. Nothing to upload.")
        sys.exit(0)
        
    uploaded_count = 0
    errors_count = 0
    
    print("\nUploading feedback scores to LangSmith...")
    
    for idx, r in enumerate(records):
        ts_run_id = r.get("triseva_run_id")
        b2_run_id = r.get("b2_run_id")
        
        # 1. Upload TriSeva Scores
        if ts_run_id and ts_run_id != "None":
            # Ragas
            if ragas_triseva_df is not None and idx < len(ragas_triseva_df):
                row = ragas_triseva_df.iloc[idx]
                # Try to map columns. Ragas df uses standard metric names
                for col in ["faithfulness", "answer_relevancy"]:
                    if col in ragas_triseva_df.columns:
                        val = row[col]
                        if pd.notna(val):
                            try:
                                client.create_feedback(
                                    run_id=ts_run_id,
                                    key=f"ragas_{col}",
                                    score=round(float(val), 3),
                                    comment="Offline Ragas evaluation"
                                )
                                uploaded_count += 1
                            except Exception as e:
                                errors_count += 1
                                
            # DeepEval
            if deepeval_triseva_df is not None and idx < len(deepeval_triseva_df):
                row = deepeval_triseva_df.iloc[idx]
                for col in ["faithfulness", "answer_relevancy"]:
                    if col in deepeval_triseva_df.columns:
                        val = row[col]
                        reason_col = "faithfulness_reason" if col == "faithfulness" else "relevancy_reason"
                        comment = str(row[reason_col])[:500] if reason_col in deepeval_triseva_df.columns and pd.notna(row[reason_col]) else "Offline DeepEval evaluation"
                        if pd.notna(val):
                            try:
                                client.create_feedback(
                                    run_id=ts_run_id,
                                    key=f"deepeval_{col}",
                                    score=round(float(val), 3),
                                    comment=comment
                                )
                                uploaded_count += 1
                            except Exception as e:
                                errors_count += 1
                                
        # 2. Upload B2 (No Critic) Scores
        if b2_run_id and b2_run_id != "None":
            # Ragas
            if ragas_b2_df is not None and idx < len(ragas_b2_df):
                row = ragas_b2_df.iloc[idx]
                for col in ["faithfulness", "answer_relevancy"]:
                    if col in ragas_b2_df.columns:
                        val = row[col]
                        if pd.notna(val):
                            try:
                                client.create_feedback(
                                    run_id=b2_run_id,
                                    key=f"ragas_{col}",
                                    score=round(float(val), 3),
                                    comment="Offline Ragas evaluation"
                                )
                                uploaded_count += 1
                            except Exception as e:
                                errors_count += 1
                                
            # DeepEval
            if deepeval_b2_df is not None and idx < len(deepeval_b2_df):
                row = deepeval_b2_df.iloc[idx]
                for col in ["faithfulness", "answer_relevancy"]:
                    if col in deepeval_b2_df.columns:
                        val = row[col]
                        reason_col = "faithfulness_reason" if col == "faithfulness" else "relevancy_reason"
                        comment = str(row[reason_col])[:500] if reason_col in deepeval_b2_df.columns and pd.notna(row[reason_col]) else "Offline DeepEval evaluation"
                        if pd.notna(val):
                            try:
                                client.create_feedback(
                                    run_id=b2_run_id,
                                    key=f"deepeval_{col}",
                                    score=round(float(val), 3),
                                    comment=comment
                                )
                                uploaded_count += 1
                            except Exception as e:
                                errors_count += 1

        if (idx + 1) % 10 == 0 or (idx + 1) == len(records):
            print(f"Processed {idx + 1}/{len(records)} records. Uploaded {uploaded_count} scores.")
            
    print(f"\n[SUCCESS] Feedback upload complete!")
    print(f"Successfully uploaded: {uploaded_count} scores")
    if errors_count > 0:
        print(f"Errors encountered:  {errors_count} (some runs might not have finished or traces weren't found)")

if __name__ == "__main__":
    main()
