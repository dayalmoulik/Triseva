import warnings
warnings.filterwarnings("ignore")

import os
import sys
import json
import time
import csv
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
os.environ["LANGCHAIN_SUPPRESS_DEPRECATION_WARNINGS"] = "1"

from dotenv import load_dotenv
load_dotenv()

from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from tools.rag_tool import retrieve

# ── LLM ───────────────────────────────────────────────────────────────────────
llm = ChatAnthropic(
    model="claude-haiku-4-5",
    api_key=os.getenv("ANTHROPIC_API_KEY"),
    temperature=0.7,
    max_tokens=2048,
    max_retries=2,
    timeout=30,
)

# ── Prompts ───────────────────────────────────────────────────────────────────
QA_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a dataset curator for an Indian document QA system.
Given a text passage, generate {n} diverse question-answer pairs.

Rules:
- Questions must be answerable ONLY from the given passage
- Vary difficulty: {difficulty_mix}
- Include factual, inferential, and application questions
- Answers must be concise (1-3 sentences)
- Make questions realistic — things an Indian citizen would actually ask

Return ONLY a JSON array, no other text:
[
  {{
    "question": "question text",
    "answer": "answer text",
    "difficulty": "easy|medium|hard",
    "question_type": "factual|inferential|application"
  }}
]"""),
    ("human", "Passage:\n{passage}\n\nGenerate {n} QA pairs:"),
])

HINDI_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a translator. Translate the following English question-answer pairs to Hindi.
Both the translated question and the translated answer must be in Hindi script (Devanagari).
Return ONLY a JSON array of objects:
[
  {{
    "hindi_question": "Hindi translation of question",
    "hindi_answer": "Hindi translation of answer"
  }}
]"""),
    ("human", "English Pairs:\n{pairs}\n\nTranslate to Hindi:"),
])

HINGLISH_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are a translator. Translate the following English question-answer pairs to Hinglish (Hindi language written in Roman/Latin script, using common English words mixed in as spoken in daily life in India).
Both the translated question and the translated answer must be in Roman-script Hinglish.
Return ONLY a JSON array of objects:
[
  {{
    "hinglish_question": "Hinglish translation of question",
    "hinglish_answer": "Hinglish translation of answer"
  }}
]"""),
    ("human", "English Pairs:\n{pairs}\n\nTranslate to Hinglish:"),
])

qa_chain    = QA_PROMPT | llm
hindi_chain = HINDI_PROMPT | llm
hinglish_chain = HINGLISH_PROMPT | llm

# ── Seed passages ──────────────────────────────────────────────────────────────
SEED_DATA = {
    "health": [
        {"source": "medline_haemoglobin",    "text": "Normal haemoglobin levels: Men 13.5-17.5 g/dL, Women 12.0-15.5 g/dL, Children 11.0-16.0 g/dL. Low haemoglobin is called anaemia. Common causes include iron deficiency, vitamin B12 deficiency, and chronic disease. Symptoms include fatigue, weakness, pale skin, and shortness of breath."},
        {"source": "medline_diabetes",       "text": "Metformin is a medication used to treat type 2 diabetes. It works by reducing glucose production in the liver and improving insulin sensitivity. Common side effects include nausea, diarrhoea, and stomach upset. It should be taken with meals. Normal fasting blood sugar: 70-100 mg/dL. Diabetic range: above 126 mg/dL fasting."},
        {"source": "medline_blood_pressure", "text": "Normal blood pressure is below 120/80 mmHg. High blood pressure (hypertension) is 130/80 mmHg or above. Low blood pressure (hypotension) is below 90/60 mmHg. High blood pressure increases risk of heart disease, stroke, and kidney failure. Lifestyle changes include reducing salt, exercising regularly, and limiting alcohol."},
        {"source": "medline_cholesterol",    "text": "Total cholesterol should be below 200 mg/dL. LDL (bad) cholesterol should be below 100 mg/dL. HDL (good) cholesterol should be above 40 mg/dL for men and 50 mg/dL for women. Triglycerides should be below 150 mg/dL. High cholesterol increases risk of heart disease and stroke."},
        {"source": "medline_thyroid",        "text": "Normal TSH range is 0.4 to 4.0 mIU/L. TSH above 4.0 suggests hypothyroidism (underactive thyroid). TSH below 0.4 suggests hyperthyroidism (overactive thyroid). Common hypothyroid symptoms: fatigue, weight gain, cold intolerance, constipation. Common hyperthyroid symptoms: weight loss, rapid heartbeat, anxiety, heat intolerance."},
        {"source": "medline_kidney",         "text": "Creatinine is a waste product filtered by kidneys. Normal creatinine levels: Men 0.74-1.35 mg/dL, Women 0.59-1.04 mg/dL. High creatinine indicates reduced kidney function. eGFR above 60 is considered normal kidney function. Below 60 for 3 months indicates chronic kidney disease."},
        {"source": "medline_fever",          "text": "A fever is a body temperature above 38°C (100.4°F). Common causes include viral infections, bacterial infections, and inflammation. Treatment includes rest, hydration, and paracetamol or ibuprofen for fever above 38.5°C. Seek immediate medical care if fever exceeds 40°C, or is accompanied by severe headache, stiff neck, or difficulty breathing."},
        {"source": "nhp_covid",              "text": "COVID-19 symptoms include fever, cough, fatigue, loss of taste or smell, sore throat, headache, and difficulty breathing. Isolate if symptomatic. Seek emergency care for severe breathing difficulty, persistent chest pain, confusion, or bluish lips. Vaccination is the best protection. Wear masks in crowded areas and wash hands frequently."},
    ],
    "legal": [
        {"source": "myscheme_pmkisan",    "text": "PM Kisan Samman Nidhi provides income support of Rs 6000 per year to farmer families in three equal instalments. Eligibility: small and marginal farmer families with combined landholding up to 2 hectares. Exclusions: institutional landholders, farmer families holding constitutional posts, serving or retired government employees with monthly pension above Rs 10000, income tax payers."},
        {"source": "myscheme_ayushman",   "text": "Ayushman Bharat PM-JAY provides health cover of Rs 5 lakh per family per year for secondary and tertiary care hospitalisation. Eligible families are based on SECC 2011 database. Documents required: Aadhaar card, ration card, income certificate. Over 1700 treatment packages covered. No premium to be paid by beneficiary."},
        {"source": "myscheme_pmay",       "text": "Pradhan Mantri Awas Yojana aims to provide affordable housing for all. EWS category (income up to Rs 3 lakh): subsidy of 6.5% for loan up to Rs 6 lakh. LIG category (income Rs 3-6 lakh): subsidy of 6.5% for loan up to Rs 6 lakh. MIG-I (income Rs 6-12 lakh): 4% subsidy. MIG-II (income Rs 12-18 lakh): 3% subsidy."},
        {"source": "rti_guide",           "text": "Right to Information Act 2005 allows Indian citizens to request information from public authorities. To file RTI: write application on plain paper addressed to Public Information Officer. Include name, address, and specific information requested. Pay fee of Rs 10 by postal order or demand draft. PIO must respond within 30 days. If unsatisfied, appeal to First Appellate Authority within 30 days."},
        {"source": "myscheme_mgnrega",    "text": "MGNREGA guarantees 100 days of wage employment per year to rural households. Eligibility: adult members of rural households willing to do unskilled manual work. Wage rate varies by state. Work must be provided within 15 days of application, failing which unemployment allowance is paid. Apply at local Gram Panchayat."},
        {"source": "myscheme_scholarship","text": "National Scholarship Portal provides various central scholarships. Pre-matric scholarships for SC/ST/OBC students from class 1 to 10. Post-matric scholarships for students above class 10. Merit-cum-means scholarships for minority students. Documents required: income certificate, caste certificate, previous marksheets, bank account details, Aadhaar."},
        {"source": "legal_aid",           "text": "Free legal aid is available to women, SC/ST communities, persons with disabilities, victims of trafficking, children in conflict with law, persons in custody, and persons with annual income below Rs 1 lakh. Apply at District Legal Services Authority (DLSA) or State Legal Services Authority (SLSA). Services include legal advice, court representation, and mediation."},
        {"source": "myscheme_ujjwala",    "text": "Pradhan Mantri Ujjwala Yojana provides free LPG connections to women from BPL households. Eligibility: women from BPL families not already having LPG connection. Documents: BPL ration card, Aadhaar card, bank account details. The scheme covers cost of security deposit, regulator, and first refill."},
    ],
    "agriculture": [
        {"source": "pmkisan_benefit",      "text": "PM-KISAN provides income support of Rs 6000 per year to eligible farmer families in three equal instalments of Rs 2000 each. The amount is transferred directly to bank accounts through DBT. Farmers should complete eKYC, Aadhaar-bank seeding, and land record verification to avoid payment delays."},
        {"source": "soil_health_card",     "text": "Soil Health Card scheme provides farmers with soil nutrient status and fertilizer recommendations. Key parameters include pH, organic carbon, nitrogen, phosphorus, potassium, and micronutrients. Farmers should apply balanced fertilizers based on SHC advice rather than using only urea, to improve yield and soil health."},
        {"source": "pmfby_crop_insurance", "text": "Pradhan Mantri Fasal Bima Yojana offers crop insurance against yield losses due to natural calamities, pests, and diseases. Farmer premium is typically 2 percent for Kharif crops, 1.5 percent for Rabi crops, and 5 percent for commercial or horticulture crops, with remaining premium subsidized by governments."},
        {"source": "msp_procurement",      "text": "Minimum Support Price is announced by the Government of India for selected crops to protect farmers from distress sales. MSP procurement operations are season and state dependent. Farmers should check procurement centers, quality norms, and registration timelines with state agencies before bringing produce for sale."},
        {"source": "kharif_rabi_planning", "text": "Kharif crops are generally sown with monsoon onset around June to July and harvested around September to October. Rabi crops are usually sown from October to December and harvested around March to April. Crop planning should consider rainfall, irrigation availability, local advisories, and market demand."},
        {"source": "integrated_pest_management", "text": "Integrated Pest Management promotes preventive and need-based control methods. Farmers should monitor fields regularly, use pest thresholds, prefer biological and mechanical controls first, and use chemical pesticides only when needed. Always follow label dose, waiting period, and safety precautions to reduce residue and resistance risk."},
        {"source": "micro_irrigation",     "text": "Micro-irrigation systems such as drip and sprinkler can improve water-use efficiency and reduce input cost. Drip is suitable for row crops and horticulture, while sprinkler can support wider field coverage. States often provide subsidies under PMKSY components; farmers can apply through agriculture department portals."},
        {"source": "kvk_extension_support","text": "Krishi Vigyan Kendras provide district-level support including soil testing, crop advisories, demonstration trials, and farmer training. Farmers should use KVK and state agriculture university advisories for local weather-based recommendations on seed variety, nutrient management, and pest control."},
    ],
}

# ── Generator ──────────────────────────────────────────────────────────────────
def generate_qa_for_passage(passage: dict, domain: str, n: int = 8) -> list:
    """Generate n QA pairs from a single passage."""
    try:
        response = qa_chain.invoke({
            "passage":        passage["text"],
            "n":              n,
            "difficulty_mix": "3 easy, 3 medium, 2 hard",
        })
        content = response.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        content = content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        pairs = json.loads(content)

        # Add metadata
        for p in pairs:
            p["domain"]   = domain
            p["source"]   = passage["source"]
            p["language"] = "en"
            p["context"]  = passage["text"]

        return pairs

    except Exception as e:
        print(f"    ⚠️  Error: {str(e)[:80]}")
        return []


def translate_to_hindi(qa_pairs: list, sample_size: int = 3) -> list:
    """Translate a sample of QA pairs (both questions and answers) to Hindi and return new pairs."""
    sample = qa_pairs[:sample_size]
    pairs_text = json.dumps([{"question": p["question"], "answer": p["answer"]} for p in sample])

    try:
        response = hindi_chain.invoke({"pairs": pairs_text})
        content  = response.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        content = content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        translations = json.loads(content)
        hindi_pairs  = []

        for i, t in enumerate(translations):
            if i < len(sample):
                new_pair = sample[i].copy()
                new_pair["question"] = t.get("hindi_question", sample[i]["question"])
                new_pair["answer"] = t.get("hindi_answer", sample[i]["answer"])
                new_pair["language"] = "hi"
                hindi_pairs.append(new_pair)

        return hindi_pairs

    except Exception as e:
        print(f"    ⚠️  Hindi translation error: {str(e)[:60]}")
        return []


def translate_to_hinglish(qa_pairs: list, sample_size: int = 3) -> list:
    """Translate a sample of QA pairs (both questions and answers) to Hinglish and return new pairs."""
    sample = qa_pairs[:sample_size]
    pairs_text = json.dumps([{"question": p["question"], "answer": p["answer"]} for p in sample])

    try:
        response = hinglish_chain.invoke({"pairs": pairs_text})
        content  = response.content
        if isinstance(content, list):
            content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
        content = content.strip()
        if "```" in content:
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]

        translations = json.loads(content)
        hinglish_pairs  = []

        for i, t in enumerate(translations):
            if i < len(sample):
                new_pair = sample[i].copy()
                new_pair["question"] = t.get("hinglish_question", sample[i]["question"])
                new_pair["answer"] = t.get("hinglish_answer", sample[i]["answer"])
                new_pair["language"] = "hinglish"
                hinglish_pairs.append(new_pair)

        return hinglish_pairs

    except Exception as e:
        print(f"    ⚠️  Hinglish translation error: {str(e)[:60]}")
        return []


def chunk_text_into_passages(text: str, source_name: str, max_words: int = 180) -> list:
    """Chunk a large text file into semantic passages of approx max_words words."""
    import re
    # Replace multiple spaces/newlines with single spaces
    text = re.sub(r'\s+', ' ', text).strip()
    words = text.split()
    
    chunks = []
    current_chunk = []
    word_count = 0
    
    for word in words:
        current_chunk.append(word)
        word_count += 1
        # Split on sentence boundary when we have enough words
        if word_count >= max_words and word.endswith(('.', '?', '!', '।')):
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            word_count = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    passages = []
    for idx, text_chunk in enumerate(chunks):
        if len(text_chunk.split()) > 40:
            passages.append({
                "source": f"{source_name}_chunk{idx+1}",
                "text": text_chunk
            })
    return passages


def load_passages_from_raw() -> dict:
    """Load passages dynamically from the actual scraped and downloaded files."""
    from pathlib import Path
    passages = {
        "health": [],
        "legal": [],
        "agriculture": []
    }
    
    # 1. Health
    health_path = Path("data/raw/health/medlineplus_topics.jsonl")
    if health_path.exists():
        with open(health_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("text") and len(record["text"].split()) > 40:
                        passages["health"].append({
                            "source": record.get("title") or record.get("url") or "medlineplus_topics",
                            "text": record["text"]
                        })
    # 2. Legal
    legal_path = Path("data/raw/legal/india_code_acts.jsonl")
    if legal_path.exists():
        with open(legal_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    record = json.loads(line)
                    if record.get("text") and len(record["text"].split()) > 40:
                        passages["legal"].append({
                            "source": f"{record.get('title')}_p{record.get('page', 1)}",
                            "text": record["text"]
                        })
    # 3. Agriculture
    agri_paths = [
        Path("data/raw/agriculture/icar_advisories.jsonl"),
        Path("data/raw/agriculture/soil_health.jsonl")
    ]
    for agri_path in agri_paths:
        if agri_path.exists():
            with open(agri_path, "r", encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        record = json.loads(line)
                        if record.get("text") and len(record["text"].split()) > 40:
                            passages["agriculture"].append({
                                "source": record.get("title") or record.get("crop") or record.get("url") or agri_path.name,
                                "text": record["text"]
                            })
                            
    # Also load from raw text files in agriculture directory (since JSONL files are empty)
    agri_raw_path = Path("data/raw/agriculture")
    if agri_raw_path.exists():
        txt_files = list(agri_raw_path.glob("**/*.txt"))
        for txt_file in txt_files:
            if txt_file.name == "index.txt":
                continue
            try:
                content = txt_file.read_text(encoding="utf-8", errors="ignore")
                chunks = chunk_text_into_passages(content, txt_file.name, max_words=180)
                passages["agriculture"].extend(chunks)
            except Exception as e:
                print(f"Failed parsing {txt_file.name}: {e}")
                        
    # Fallback to hardcoded seed data if files are missing or empty
    for domain in ["health", "legal", "agriculture"]:
        if not passages[domain]:
            print(f"⚠️  No raw files found for {domain} — falling back to hardcoded seed data")
            passages[domain] = SEED_DATA[domain]
            
    # Sample up to 25 passages per domain to keep dataset curation concise and diverse
    import random
    random.seed(42)  # for reproducibility
    for domain in ["health", "legal", "agriculture"]:
        if len(passages[domain]) > 25:
            passages[domain] = random.sample(passages[domain], 25)
            
    return passages


def curate_dataset(target_per_domain: int = 60) -> list:
    """Curate the full TriSeva-QA dataset from actual raw passages."""
    all_pairs = []

    print(f"\n{'='*60}")
    print(f"TRISEVA-QA DATASET CURATION")
    print(f"Target: ~{target_per_domain * 3} total QA pairs")
    print(f"{'='*60}\n")

    passages_dict = load_passages_from_raw()

    for domain, passages in passages_dict.items():
        print(f"\n📂 Domain: {domain.upper()}")
        print(f"   Passages: {len(passages)}")

        domain_pairs = []
        n_per_passage = target_per_domain // len(passages)
        n_per_passage = min(15, max(4, n_per_passage))

        for j, passage in enumerate(passages, 1):
            print(f"   [{j}/{len(passages)}] {passage['source']}...")

            pairs = generate_qa_for_passage(passage, domain, n=n_per_passage)
            print(f"          ✅ Generated {len(pairs)} QA pairs")

            # Add Hindi and Hinglish for first 5 passages per domain
            if j <= 5 and pairs:
                hindi = translate_to_hindi(pairs, sample_size=2)
                if hindi:
                    print(f"          🇮🇳 Added {len(hindi)} Hindi pairs")
                    domain_pairs.extend(hindi)
                
                hinglish = translate_to_hinglish(pairs, sample_size=2)
                if hinglish:
                    print(f"          🗣️  Added {len(hinglish)} Hinglish pairs")
                    domain_pairs.extend(hinglish)

            domain_pairs.extend(pairs)
            time.sleep(2)  # rate limit

        print(f"\n   📊 {domain.upper()} total: {len(domain_pairs)} pairs")
        all_pairs.extend(domain_pairs)

    return all_pairs


def save_dataset(pairs: list):
    """Save dataset as JSON, CSV, and HuggingFace format."""
    os.makedirs("evaluation/results", exist_ok=True)

    # Assign IDs
    for i, p in enumerate(pairs):
        p["id"] = f"triseva-{i+1:04d}"

    # Save JSON
    with open("evaluation/results/triseva_qa_dataset.json", "w", encoding="utf-8") as f:
        json.dump(pairs, f, indent=2, ensure_ascii=False)

    # Save CSV
    if pairs:
        with open("evaluation/results/triseva_qa_dataset.csv", "w",
                  newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=pairs[0].keys())
            writer.writeheader()
            writer.writerows(pairs)

    # Print stats
    domains   = {}
    languages = {}
    for p in pairs:
        domains[p["domain"]]     = domains.get(p["domain"], 0)     + 1
        languages[p["language"]] = languages.get(p["language"], 0) + 1

    difficulties = {}
    for p in pairs:
        d = p.get("difficulty", "unknown")
        difficulties[d] = difficulties.get(d, 0) + 1

    print(f"\n{'='*60}")
    print(f"DATASET STATISTICS")
    print(f"{'='*60}")
    print(f"Total QA pairs    : {len(pairs)}")
    print(f"\nBy domain:")
    for d, n in sorted(domains.items()):
        print(f"  {d:<12} : {n}")
    print(f"\nBy language:")
    for l, n in sorted(languages.items()):
        print(f"  {l:<12} : {n}")
    print(f"\nBy difficulty:")
    for d, n in sorted(difficulties.items()):
        print(f"  {d:<12} : {n}")
    print(f"\nSaved to evaluation/results/triseva_qa_dataset.json")
    print(f"         evaluation/results/triseva_qa_dataset.csv")
    print(f"{'='*60}\n")

    return pairs


def push_to_huggingface(pairs: list, repo_name: str = "triseva-qa"):
    """Push dataset to HuggingFace Hub."""
    try:
        from datasets import Dataset as HFDataset
        import huggingface_hub as hf

        token = os.getenv("HF_TOKEN")
        if not token:
            print("⚠️  HF_TOKEN not set — skipping HuggingFace upload")
            print("   Set HF_TOKEN in .env and re-run to upload")
            return

        hf.login(token=token, add_to_git_credential=False)

        hf_username = hf.whoami()["name"]
        repo_id     = f"{hf_username}/{repo_name}"

        print(f"\n📤 Pushing to HuggingFace Hub: {repo_id}")

        dataset = HFDataset.from_list(pairs)
        dataset.push_to_hub(
            repo_id,
            token=token,
            commit_message="Initial release of TriSeva-QA dataset",
        )

        print(f"✅ Dataset pushed to: https://huggingface.co/datasets/{repo_id}")

    except Exception as e:
        print(f"⚠️  HuggingFace upload failed: {str(e)[:150]}")
        print("   Dataset saved locally — upload manually later")


if __name__ == "__main__":
    # Curate
    pairs = curate_dataset(target_per_domain=350)

    # Save locally
    pairs = save_dataset(pairs)

    # Push to HuggingFace
    push_to_huggingface(pairs, repo_name="triseva-qa")