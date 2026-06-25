import warnings
warnings.filterwarnings("ignore")

import os
from dotenv import load_dotenv
load_dotenv()

# Authenticate with HuggingFace to suppress unauthenticated warning
import huggingface_hub
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    huggingface_hub.login(token=hf_token, add_to_git_credential=False)
    
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding='utf-8')

import chromadb
from chromadb.utils import embedding_functions
from knowledge_base.chunker import chunk_pdf, chunk_raw_text
from typing import List, Dict
from pathlib import Path

# ── ChromaDB setup ────────────────────────────────────────────────────────────
CHROMA_PATH = "data/chromadb"
EMBED_MODEL  = "intfloat/multilingual-e5-base"

def get_chroma_client():
    return chromadb.PersistentClient(path=CHROMA_PATH)

def get_embedding_function():
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBED_MODEL
    )

def get_or_create_collection(client, domain: str):
    """Get or create a domain-specific collection."""
    return client.get_or_create_collection(
        name=f"triseva_{domain}",
        embedding_function=get_embedding_function(),
        metadata={"domain": domain}
    )


# ── Add chunks to collection ──────────────────────────────────────────────────
def add_chunks_to_collection(collection, chunks: List[Dict]):
    """Add chunks to ChromaDB with e5 passage prefix for cross-lingual alignment."""
    if not chunks:
        print("  No chunks to add.")
        return

    # multilingual-e5 requires "passage: " prefix at index time
    documents = [f"passage: {c['text']}" for c in chunks]
    ids       = [c["chunk_id"] for c in chunks]
    metadatas = [{"source": c["source"], "domain": c["domain"]} for c in chunks]

    batch_size = 100
    for i in range(0, len(documents), batch_size):
        collection.add(
            documents=documents[i:i+batch_size],
            ids=ids[i:i+batch_size],
            metadatas=metadatas[i:i+batch_size],
        )

    print(f"  ✅ Added {len(chunks)} chunks to '{collection.name}'")


# ── Domain-specific chunk configurations ──────────────────────────────────────
DOMAIN_CONFIGS = {
    "health":      {"chunk_size": 1000, "overlap": 150},
    "legal":       {"chunk_size": 600,  "overlap": 100},
    "agriculture": {"chunk_size": 600,  "overlap": 100},
}


# ── Seed data — starter knowledge for each domain ────────────────────────────
# This gets you running immediately without needing PDFs
# You will add real PDFs on top of this

HEALTH_SEED = [
    {"source": "medline_haemoglobin", "text": "Normal haemoglobin levels: Men 13.5-17.5 g/dL, Women 12.0-15.5 g/dL, Children 11.0-16.0 g/dL. Low haemoglobin is called anaemia. Common causes include iron deficiency, vitamin B12 deficiency, and chronic disease. Symptoms include fatigue, weakness, pale skin, and shortness of breath."},
    {"source": "medline_diabetes", "text": "Metformin is a medication used to treat type 2 diabetes. It works by reducing glucose production in the liver and improving insulin sensitivity. Common side effects include nausea, diarrhoea, and stomach upset. It should be taken with meals. Normal fasting blood sugar: 70-100 mg/dL. Diabetic range: above 126 mg/dL fasting."},
    {"source": "medline_fever", "text": "A fever is a body temperature above 38°C (100.4°F). Common causes include viral infections, bacterial infections, and inflammation. Treatment includes rest, hydration, and paracetamol or ibuprofen for fever above 38.5°C. Seek immediate medical care if fever exceeds 40°C, or is accompanied by severe headache, stiff neck, or difficulty breathing."},
    {"source": "medline_blood_pressure", "text": "Normal blood pressure is below 120/80 mmHg. High blood pressure (hypertension) is 130/80 mmHg or above. Low blood pressure (hypotension) is below 90/60 mmHg. High blood pressure increases risk of heart disease, stroke, and kidney failure. Lifestyle changes include reducing salt, exercising regularly, and limiting alcohol."},
    {"source": "medline_cholesterol", "text": "Total cholesterol should be below 200 mg/dL. LDL (bad) cholesterol should be below 100 mg/dL. HDL (good) cholesterol should be above 40 mg/dL for men and 50 mg/dL for women. Triglycerides should be below 150 mg/dL. High cholesterol increases risk of heart disease and stroke."},
    {"source": "medline_thyroid", "text": "Normal TSH (thyroid stimulating hormone) range is 0.4 to 4.0 mIU/L. TSH above 4.0 suggests hypothyroidism (underactive thyroid). TSH below 0.4 suggests hyperthyroidism (overactive thyroid). Common hypothyroid symptoms: fatigue, weight gain, cold intolerance, constipation. Common hyperthyroid symptoms: weight loss, rapid heartbeat, anxiety, heat intolerance."},
    {"source": "medline_kidney", "text": "Creatinine is a waste product filtered by kidneys. Normal creatinine levels: Men 0.74-1.35 mg/dL, Women 0.59-1.04 mg/dL. High creatinine indicates reduced kidney function. eGFR (estimated glomerular filtration rate) above 60 is considered normal kidney function. Below 60 for 3 months indicates chronic kidney disease."},
    {"source": "nhp_covid", "text": "COVID-19 symptoms include fever, cough, fatigue, loss of taste or smell, sore throat, headache, and difficulty breathing. Isolate if symptomatic. Seek emergency care for severe breathing difficulty, persistent chest pain, confusion, or bluish lips. Vaccination is the best protection. Wear masks in crowded areas and wash hands frequently."},
]

LEGAL_SEED = [
    {"source": "myscheme_pmkisan", "text": "PM Kisan Samman Nidhi (PM-KISAN) provides income support of Rs 6000 per year to farmer families in three equal instalments. Eligibility: small and marginal farmer families with combined landholding up to 2 hectares. Exclusions: institutional landholders, farmer families holding constitutional posts, serving or retired government employees with monthly pension above Rs 10000, income tax payers."},
    {"source": "myscheme_ayushman", "text": "Ayushman Bharat Pradhan Mantri Jan Arogya Yojana (PM-JAY) provides health cover of Rs 5 lakh per family per year for secondary and tertiary care hospitalisation. Eligible families are based on SECC 2011 database. Documents required: Aadhaar card, ration card, income certificate. Over 1700 treatment packages covered. No premium to be paid by beneficiary."},
    {"source": "myscheme_pmay", "text": "Pradhan Mantri Awas Yojana (PMAY) aims to provide affordable housing for all. Urban component provides interest subsidy on home loans. EWS category (income up to Rs 3 lakh): subsidy of 6.5% for loan up to Rs 6 lakh. LIG category (income Rs 3-6 lakh): subsidy of 6.5% for loan up to Rs 6 lakh. MIG-I (income Rs 6-12 lakh): 4% subsidy. MIG-II (income Rs 12-18 lakh): 3% subsidy."},
    {"source": "rti_guide", "text": "Right to Information (RTI) Act 2005 allows Indian citizens to request information from public authorities. To file RTI: write application on plain paper addressed to Public Information Officer (PIO) of concerned department. Include name, address, and specific information requested. Pay fee of Rs 10 by postal order or demand draft. PIO must respond within 30 days. If unsatisfied, appeal to First Appellate Authority within 30 days."},
    {"source": "myscheme_mgnrega", "text": "Mahatma Gandhi National Rural Employment Guarantee Act (MGNREGA) guarantees 100 days of wage employment per year to rural households. Eligibility: adult members of rural households willing to do unskilled manual work. Wage rate varies by state. Work must be provided within 15 days of application, failing which unemployment allowance is paid. Apply at local Gram Panchayat."},
    {"source": "myscheme_scholarship", "text": "National Scholarship Portal (NSP) provides various central scholarships for students. Pre-matric scholarships for SC/ST/OBC students from class 1 to 10. Post-matric scholarships for students above class 10. Merit-cum-means scholarships for minority students in technical and professional courses. Documents required: income certificate, caste certificate, previous marksheets, bank account details, Aadhaar."},
    {"source": "legal_aid", "text": "Free legal aid is available to women, SC/ST communities, persons with disabilities, victims of trafficking, children in conflict with law, persons in custody, and persons with annual income below Rs 1 lakh. Apply at District Legal Services Authority (DLSA) or State Legal Services Authority (SLSA). Services include legal advice, court representation, and mediation."},
    {"source": "myscheme_ujjwala", "text": "Pradhan Mantri Ujjwala Yojana provides free LPG connections to women from BPL households. Eligibility: women from BPL families not already having LPG connection. Documents: BPL ration card, Aadhaar card, bank account details. The scheme covers cost of security deposit, regulator, and first refill. Apply at nearest LPG distributor."},
]

AGRICULTURE_SEED = [
    {"source": "pmkisan_benefit", "text": "PM-KISAN provides income support of Rs 6000 per year to eligible farmer families in three equal instalments of Rs 2000 each. The amount is transferred directly to bank accounts through DBT. Farmers should complete eKYC, Aadhaar-bank seeding, and land record verification to avoid payment delays."},
    {"source": "soil_health_card", "text": "Soil Health Card scheme provides farmers with soil nutrient status and fertilizer recommendations. Key parameters include pH, organic carbon, nitrogen, phosphorus, potassium, and micronutrients. Farmers should apply balanced fertilizers based on SHC advice rather than using only urea, to improve yield and soil health."},
    {"source": "pmfby_crop_insurance", "text": "Pradhan Mantri Fasal Bima Yojana offers crop insurance against yield losses due to natural calamities, pests, and diseases. Farmer premium is typically 2 percent for Kharif crops, 1.5 percent for Rabi crops, and 5 percent for commercial or horticulture crops, with remaining premium subsidized by governments."},
    {"source": "msp_procurement", "text": "Minimum Support Price is announced by the Government of India for selected crops to protect farmers from distress sales. MSP procurement operations are season and state dependent. Farmers should check procurement centers, quality norms, and registration timelines with state agencies before bringing produce for sale."},
    {"source": "kharif_rabi_planning", "text": "Kharif crops are generally sown with monsoon onset around June to July and harvested around September to October. Rabi crops are usually sown from October to December and harvested around March to April. Crop planning should consider rainfall, irrigation availability, local advisories, and market demand."},
    {"source": "integrated_pest_management", "text": "Integrated Pest Management promotes preventive and need-based control methods. Farmers should monitor fields regularly, use pest thresholds, prefer biological and mechanical controls first, and use chemical pesticides only when needed. Always follow label dose, waiting period, and safety precautions to reduce residue and resistance risk."},
    {"source": "micro_irrigation", "text": "Micro-irrigation systems such as drip and sprinkler can improve water-use efficiency and reduce input cost. Drip is suitable for row crops and horticulture, while sprinkler can support wider field coverage. States often provide subsidies under PMKSY components; farmers can apply through agriculture department portals."},
    {"source": "kvk_extension_support", "text": "Krishi Vigyan Kendras provide district-level support including soil testing, crop advisories, demonstration trials, and farmer training. Farmers should use KVK and state agriculture university advisories for local weather-based recommendations on seed variety, nutrient management, and pest control."},
]


# ── Main build function ───────────────────────────────────────────────────────
def build_knowledge_base():
    print("\n🔨 Building TriSeva Knowledge Base...")
    print("=" * 60)

    client = get_chroma_client()

    domains_seed = {
        "health":    HEALTH_SEED,
        "legal":     LEGAL_SEED,
        "agriculture": AGRICULTURE_SEED,
    }

    for domain, seed_data in domains_seed.items():
        print(f"\n📂 Processing domain: {domain.upper()}")
        
        # Reset collection to avoid duplicates or orphaned old data
        collection_name = f"triseva_{domain}"
        try:
            client.delete_collection(collection_name)
            print(f"  🗑️ Deleted existing collection '{collection_name}'")
        except Exception:
            pass

        collection = get_or_create_collection(client, domain)

        # Get config for chunking
        config = DOMAIN_CONFIGS.get(domain, {"chunk_size": 1000, "overlap": 150})
        chunk_sz = config["chunk_size"]
        overlap = config["overlap"]
        print(f"  🔧 Using chunking config: size={chunk_sz}, overlap={overlap}")

        # Add seed data
        chunks = []
        for i, item in enumerate(seed_data):
            chunks.append({
                "text":     item["text"],
                "source":   item["source"],
                "domain":   domain,
                "chunk_id": f"{domain}_seed_{i:04d}",
            })
        add_chunks_to_collection(collection, chunks)

        # Also load any PDFs, JSONLs, and clean.txt files recursively from data/raw/{domain}/
        raw_dir = f"data/raw/{domain}"
        raw_path = Path(raw_dir)
        if raw_path.exists():
            # 1. Recursive PDFs
            pdf_files = list(raw_path.glob("**/*.pdf"))
            if pdf_files:
                print(f"  Found {len(pdf_files)} PDF(s) in {raw_dir}")
                for pdf_file in pdf_files:
                    pdf_chunks = chunk_pdf(str(pdf_file), domain=domain, chunk_size=chunk_sz, overlap=overlap)
                    add_chunks_to_collection(collection, pdf_chunks)
            else:
                print(f"  No PDFs found in {raw_dir}")

            # 2. JSONL files (scraped datasets)
            jsonl_files = list(raw_path.glob("**/*.jsonl"))
            if jsonl_files:
                print(f"  Found {len(jsonl_files)} JSONL file(s) in {raw_dir}")
                import json
                for jsonl_file in jsonl_files:
                    print(f"  Processing JSONL: {jsonl_file.name}")
                    jsonl_chunks = []
                    try:
                        with open(jsonl_file, "r", encoding="utf-8") as f:
                            for i, line in enumerate(f):
                                line = line.strip()
                                if not line:
                                    continue
                                record = json.loads(line)
                                text = record.get("text", "")
                                if not text:
                                    continue
                                source = record.get("url") or record.get("source") or jsonl_file.name
                                # Chunk this text
                                record_chunks = chunk_raw_text(text, source=source, domain=domain, chunk_size=chunk_sz, overlap=overlap)
                                for rc in record_chunks:
                                    rc["chunk_id"] = f"{rc['chunk_id']}_rec{i}"
                                jsonl_chunks.extend(record_chunks)
                        add_chunks_to_collection(collection, jsonl_chunks)
                    except Exception as e:
                        print(f"  Failed parsing JSONL {jsonl_file.name}: {e}")

            # 3. Cleaned text files
            clean_txt_files = list(raw_path.glob("**/*.clean.txt"))
            if clean_txt_files:
                print(f"  Found {len(clean_txt_files)} clean.txt file(s) in {raw_dir}")
                for txt_file in clean_txt_files:
                    print(f"  Processing clean txt: {txt_file.name}")
                    try:
                        text = txt_file.read_text(encoding="utf-8", errors="ignore")
                        txt_chunks = chunk_raw_text(text, source=str(txt_file), domain=domain, chunk_size=chunk_sz, overlap=overlap)
                        add_chunks_to_collection(collection, txt_chunks)
                    except Exception as e:
                        print(f"  Failed parsing text file {txt_file.name}: {e}")

        # Print collection stats
        count = collection.count()
        print(f"  📊 Total chunks in '{collection.name}': {count}")

    print("\n" + "=" * 60)
    print("✅ Knowledge base built successfully")
    print(f"📁 Stored at: {os.path.abspath(CHROMA_PATH)}")


if __name__ == "__main__":
    build_knowledge_base()