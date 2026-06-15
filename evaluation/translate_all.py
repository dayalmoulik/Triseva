import warnings
warnings.filterwarnings("ignore")

import os
import sys

# Disable LangChain tracing to prevent LangSmith 429 rate limit warnings
os.environ["LANGCHAIN_TRACING_V2"] = "false"

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

import json
import time
import csv
import requests
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["LANGCHAIN_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from agents.llm_factory import get_llm
from langchain_core.prompts import ChatPromptTemplate

# ── Initialize LLM (Sarvam-105B) ──────────────────────────────────────────────
print("Initializing Sarvam-105B LLM for Hinglish pair-by-pair translation...")
llm = get_llm(temperature=0.2, max_tokens=512)

# ── Hinglish Prompt (Single Pair) ──────────────────────────────────────────────
HINGLISH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a translator. Translate the following English question-answer pair to Hinglish (Hindi language written in Roman/Latin script, using common English words mixed in as spoken in daily life in India).
Return ONLY a JSON object with keys "hinglish_question" and "hinglish_answer". Do not include any other text or markdown code fences:
{{
  "hinglish_question": "Hinglish translation of question",
  "hinglish_answer": "Hinglish translation of answer"
}}"""),
    ("human", "English Pair:\nQuestion: {question}\nAnswer: {answer}\n\nTranslate to Hinglish:"),
])

hinglish_chain = HINGLISH_PROMPT | llm

# Helper to clean LLM response content and extract JSON block robustly
def clean_json_content(content):
    if isinstance(content, list):
        content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
    content = content.strip()
    
    # Try to find JSON block enclosed in triple backticks
    if "```" in content:
        parts = content.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{") and part.endswith("}"):
                return part
            if part.startswith("[") and part.endswith("]"):
                return part
                
    # Fallback: Find the first '{' and last '}'
    start = content.find("{")
    end = content.rfind("}")
    if start != -1 and end != -1 and end > start:
        return content[start:end+1].strip()

    return content.strip()

# ── Hindi Translation using Sarvam REST API (Single Pair with Retry) ──────────
def translate_single_hindi_sarvam(item: dict, max_retries: int = 3) -> dict:
    url = "https://api.sarvam.ai/translate"
    headers = {
        "Content-Type": "application/json",
        "api-subscription-key": os.getenv("SARVAM_API_KEY")
    }
    
    for attempt in range(max_retries):
        try:
            # 1. Translate question
            q_payload = {
                "input": item["question"],
                "source_language_code": "en-IN",
                "target_language_code": "hi-IN"
            }
            q_resp = requests.post(url, json=q_payload, headers=headers, timeout=15)
            q_text = ""
            if q_resp.status_code == 200:
                q_text = q_resp.json().get("translated_text", "")
            
            # 2. Translate answer
            a_payload = {
                "input": item["answer"],
                "source_language_code": "en-IN",
                "target_language_code": "hi-IN"
            }
            a_resp = requests.post(url, json=a_payload, headers=headers, timeout=15)
            a_text = ""
            if a_resp.status_code == 200:
                a_text = a_resp.json().get("translated_text", "")
            
            if q_text and a_text:
                new_pair = item.copy()
                new_pair["question"] = q_text
                new_pair["answer"] = a_text
                new_pair["language"] = "hi"
                return new_pair
                
            print(f"      ⚠️ Attempt {attempt+1}/{max_retries} failed for Hindi. Retrying...")
            time.sleep(1.0)
        except Exception as e:
            print(f"      ⚠️ Exception on Hindi attempt {attempt+1}: {e}")
            time.sleep(1.0)
    return None

# ── Hinglish Translation using Sarvam-105B (Single Pair with Retry) ───────────
def translate_single_hinglish_sarvam(item: dict, max_retries: int = 3) -> dict:
    for attempt in range(max_retries):
        try:
            response = hinglish_chain.invoke({
                "question": item["question"],
                "answer": item["answer"]
            })
            cleaned = clean_json_content(response.content)
            t = json.loads(cleaned)
            
            new_pair = item.copy()
            new_pair["question"] = t.get("hinglish_question", item["question"])
            new_pair["answer"] = t.get("hinglish_answer", item["answer"])
            new_pair["language"] = "hinglish"
            return new_pair
        except Exception as e:
            print(f"      ⚠️ Attempt {attempt+1}/{max_retries} failed for Hinglish: {str(e)[:100]}. Retrying...")
            time.sleep(1.0)
    return None

# ── HuggingFace Upload ────────────────────────────────────────────────────────
def push_to_huggingface(pairs: list, repo_name: str):
    try:
        from datasets import Dataset as HFDataset
        import huggingface_hub as hf

        token = os.getenv("HF_TOKEN")
        if not token:
            print("⚠️  HF_TOKEN not set — skipping HuggingFace upload")
            return

        hf.login(token=token, add_to_git_credential=False)
        hf_username = hf.whoami()["name"]
        repo_id     = f"{hf_username}/{repo_name}"

        print(f"\n📤 Pushing to HuggingFace Hub: {repo_id}")
        dataset = HFDataset.from_list(pairs)
        dataset.push_to_hub(
            repo_id,
            token=token,
            commit_message="Publish complete parallel multilingual dataset (970 pairs per language)",
        )
        print(f"✅ Dataset pushed successfully: https://huggingface.co/datasets/{repo_id}")
    except Exception as e:
        print(f"⚠️  HuggingFace upload failed: {str(e)[:150]}")

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    dataset_path = "evaluation/results/triseva_qa_dataset.json"
    temp_path = "evaluation/results/triseva_qa_dataset_complete_temp.json"
    
    if not os.path.exists(dataset_path):
        print(f"❌ Error: Baseline dataset file not found at {dataset_path}")
        sys.exit(1)

    print(f"Loading English baseline dataset from {dataset_path}...")
    with open(dataset_path, "r", encoding="utf-8") as f:
        all_pairs = json.load(f)

    # Filter only English pairs
    english_pairs = [p for p in all_pairs if p.get("language") == "en"]
    print(f"Loaded {len(english_pairs)} English pairs.")

    # Re-assign sequential IDs to English pairs (triseva-0001 to triseva-0970)
    for idx, p in enumerate(english_pairs, 1):
        p["id"] = f"triseva-{idx:04d}"

    # Load existing temp progress if it exists
    progress_pairs = []
    if os.path.exists(temp_path):
        print(f"Checking existing temp checkpoint file at {temp_path}...")
        try:
            with open(temp_path, "r", encoding="utf-8") as f:
                progress_pairs = json.load(f)
            print(f"Found checkpoint containing {len(progress_pairs)} completed translations.")
        except Exception as e:
            print(f"⚠️ Checkpoint file corrupt or empty: {e}. Starting clean.")
            progress_pairs = []

    # Map existing translated IDs to skip duplication
    translated_ids = {p["id"] for p in progress_pairs if "id" in p}

    translated_hindi = [p for p in progress_pairs if p.get("language") == "hi"]
    translated_hinglish = [p for p in progress_pairs if p.get("language") == "hinglish"]

    print("\n" + "=" * 60)
    print("STARTING COMPLETE PARALLEL TRANSLATIONS (RESUMABLE)")
    print(f"Target: Translating {len(english_pairs)} pairs to Hindi and Hinglish...")
    print(f"Current Progress: Hindi {len(translated_hindi)}/{len(english_pairs)}, Hinglish {len(translated_hinglish)}/{len(english_pairs)}")
    print("=" * 60 + "\n")

    start_time = time.time()
    processed_count = max(len(translated_hindi), len(translated_hinglish))

    for idx, item in enumerate(english_pairs, 1):
        # Determine expected IDs
        hi_id = f"triseva-{idx + 970:04d}"
        he_id = f"triseva-{idx + 1940:04d}"
        
        # Check if already completed
        hi_done = hi_id in translated_ids
        he_done = he_id in translated_ids
        
        if hi_done and he_done:
            # Skip since both are done
            continue

        elapsed = time.time() - start_time
        processed_count += 1
        avg_time = elapsed / processed_count if processed_count > 1 else 0.0
        est_rem = avg_time * (len(english_pairs) - idx)
        
        print(f"[{idx}/{len(english_pairs)}] (Elapsed: {elapsed:.1f}s, Est. Remaining: {est_rem:.1f}s)")
        print(f"  English: {item['question'][:60]}...")

        progress_changed = False

        # 1. Translate to Hindi if missing
        if not hi_done:
            h_pair = translate_single_hindi_sarvam(item, max_retries=3)
            if h_pair:
                h_pair["id"] = hi_id
                translated_hindi.append(h_pair)
                progress_pairs.append(h_pair)
                translated_ids.add(hi_id)
                progress_changed = True
                print(f"    🇮🇳 Hindi:  {h_pair['question'][:60]}...")
            else:
                print("    ❌ Hindi translation failed after retries.")
        else:
            print("    🇮🇳 Hindi:  (Already Translated)")

        # 2. Translate to Hinglish if missing
        if not he_done:
            he_pair = translate_single_hinglish_sarvam(item, max_retries=3)
            if he_pair:
                he_pair["id"] = he_id
                translated_hinglish.append(he_pair)
                progress_pairs.append(he_pair)
                translated_ids.add(he_id)
                progress_changed = True
                print(f"    🗣️ Hinglish: {he_pair['question'][:60]}...")
            else:
                print("    ❌ Hinglish translation failed after retries.")
        else:
            print("    🗣️ Hinglish: (Already Translated)")

        # Save checkpoint to disk
        if progress_changed:
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(progress_pairs, f, indent=2, ensure_ascii=False)

        print("-" * 50)
        time.sleep(0.3)  # Small pause to avoid rate limits

    print("\n" + "=" * 60)
    print("TRANSLATION SEQUENCE COMPLETE")
    print(f"English pairs:  {len(english_pairs)}")
    print(f"Hindi pairs:    {len(translated_hindi)}/{len(english_pairs)}")
    print(f"Hinglish pairs: {len(translated_hinglish)}/{len(english_pairs)}")
    print("=" * 60 + "\n")

    # Combine all pairs
    complete_dataset = []
    complete_dataset.extend(english_pairs)
    complete_dataset.extend(translated_hindi)
    complete_dataset.extend(translated_hinglish)

    # Save locally
    out_json = "evaluation/results/triseva_qa_dataset_complete.json"
    out_csv = "evaluation/results/triseva_qa_dataset_complete.csv"

    print(f"Saving complete dataset locally to {out_json}...")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(complete_dataset, f, indent=2, ensure_ascii=False)

    print(f"Saving complete dataset locally to {out_csv}...")
    if complete_dataset:
        with open(out_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=complete_dataset[0].keys())
            writer.writeheader()
            writer.writerows(complete_dataset)

    # Delete temp progress file upon success
    if os.path.exists(temp_path):
        os.remove(temp_path)
        print("Removed temp progress checkpoint file.")

    # Push to HuggingFace
    push_to_huggingface(complete_dataset, repo_name="triseva-qa-multilingual-complete")

if __name__ == "__main__":
    main()
