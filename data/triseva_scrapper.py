"""
TriSeva Knowledge Base Scraper (Optimized with Concurrency and Delays)
======================================================================
Scrapes Health, Legal/Government, and Agriculture domains.
Features:
- ThreadPoolExecutor for parallel crawls.
- Polite delays between requests inside each worker thread to prevent blocks.
- Expanded crawl caps to target ~45,000 vector chunks when built.

Sources:
  Health        : MedlinePlus (drug info + health topics), NHP India
  Legal/Govt    : MyScheme.gov.in (web scraping), India Code (RTI, acts)
  Agriculture   : ICAR advisories, PM-KISAN guidelines, Kisan Call Centre FAQs, Soil Health Portal

Usage:
  conda run -n Triseva python triseva_scrapper.py --domain all
  conda run -n Triseva python triseva_scrapper.py --domain health
  conda run -n Triseva python triseva_scrapper.py --domain legal
  conda run -n Triseva python triseva_scrapper.py --domain agriculture
"""

import os
import re
import sys
import json
import time
import random
import logging
import argparse
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from bs4 import BeautifulSoup
from tqdm import tqdm

# ─────────────────────────── Config ────────────────────────────

BASE_DIR   = Path("./data/raw")
LOG_DIR    = Path("./logs")
DELAY_MIN  = 2.0   # Polite sleep range per request per thread
DELAY_MAX  = 4.5
MAX_RETRY  = 3
NUM_WORKERS = 8     # Concurrency limit (keeps crawl polite but fast)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.5",
}

for d in ["health", "legal", "agriculture"]:
    (BASE_DIR / d).mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

# Force UTF-8 logging stream handler
_stream_handler = logging.StreamHandler(
    stream=open(sys.stdout.fileno(), mode="w", encoding="utf-8", buffering=1)
    if hasattr(sys.stdout, "fileno") else sys.stdout
)
_stream_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_DIR / "scrape.log", encoding="utf-8"),
        _stream_handler,
    ],
)
log = logging.getLogger(__name__)


# ─────────────────────── Utilities ─────────────────────────────

def polite_sleep():
    time.sleep(random.uniform(DELAY_MIN, DELAY_MAX))


def safe_get(url: str, params: dict = None, retries: int = MAX_RETRY) -> Optional[requests.Response]:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, params=params, timeout=25)
            resp.raise_for_status()
            return resp
        except requests.RequestException as e:
            wait = 2 ** attempt * 3
            log.warning(f"Attempt {attempt+1} failed for {url}: {e}. Retrying in {wait}s…")
            time.sleep(wait)
    log.error(f"All retries exhausted for: {url}")
    return None


def doc_id(text: str) -> str:
    return hashlib.md5(text.encode("utf-8", errors="ignore")).hexdigest()[:10]


def save_jsonl(records: list, path: Path):
    if not records:
        return
    with open(path, "w", encoding="utf-8") as f:  # Overwrite with updated clean list
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info(f"  Saved {len(records)} records → {path}")


def load_pdf_reader():
    try:
        from pypdf import PdfReader as _PdfReader
        def make_reader(path):
            return _PdfReader(str(path), strict=False)
        return make_reader
    except ImportError:
        try:
            import PyPDF2
            def make_reader(path):
                return PyPDF2.PdfReader(str(path), strict=False)
            return make_reader
        except ImportError:
            log.warning("Neither pypdf nor PyPDF2 installed. PDF parsing will be skipped.")
            return None


# ══════════════════════════════════════════════════════════════
#  HEALTHCARE  ─  MedlinePlus + NHP India
# ══════════════════════════════════════════════════════════════

MEDLINEPLUS_DRUG_INDEX = "https://medlineplus.gov/druginformation.html"

# Health topic URLs targeted to index in depth
MEDLINEPLUS_TOPICS = [
    ("Diabetes",          "diabetes.html"),
    ("High Blood Pressure","highbloodpressure.html"),
    ("Tuberculosis",      "tuberculosis.html"),
    ("Malaria",           "malaria.html"),
    ("Dengue",            "dengue.html"),
    ("Anemia",            "anemia.html"),
    ("Cancer",            "cancer.html"),
    ("Heart Disease",     "heartdiseases.html"),
    ("Asthma",            "asthma.html"),
    ("COVID-19",          "covid19.html"),
    ("Kidney Disease",    "kidneydiseases.html"),
    ("Liver Disease",     "liverdiseases.html"),
    ("Thyroid",           "thyroiddiseases.html"),
    ("Arthritis",         "arthritis.html"),
    ("Stroke",            "stroke.html"),
    ("Infectious Diseases","infectiousdiseases.html"),
    ("Depression",        "depression.html"),
    ("Obesity",           "obesity.html"),
]


def scrape_single_drug(drug_info: Tuple[str, str]) -> Optional[dict]:
    name, url = drug_info
    polite_sleep()
    resp = safe_get(url)
    if not resp:
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    sections = {}
    for section in soup.select("div#drug-info section, div.section"):
        h2 = section.find("h2")
        if h2:
            title = h2.get_text(strip=True)
            body  = section.get_text(separator=" ", strip=True)
            body  = re.sub(r'\s+', ' ', body)
            sections[title] = body

    full_text = " ".join(sections.values())
    if len(full_text) < 100:
        return None

    return {
        "id":       doc_id(url),
        "domain":   "health",
        "source":   "medlineplus_drugs",
        "title":    name,
        "url":      url,
        "sections": sections,
        "text":     full_text,
        "language": "en",
        "scraped_at": datetime.utcnow().isoformat(),
    }


def scrape_medlineplus_drugs(max_drugs: int = 1200):
    """Scrape drug monographs from MedlinePlus concurrently."""
    out = BASE_DIR / "health" / "medlineplus_drugs.jsonl"
    log.info("── MedlinePlus Drugs (Multithreaded) ──")

    # Fetch alphabetical index pages first
    letters = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    urls = [f"https://medlineplus.gov/druginfo/drug_{l}a.html" for l in letters] + ["https://medlineplus.gov/druginfo/drug_00.html"]
    
    drug_links = []
    log.info("  Fetching alphabetical drug index pages...")
    
    # Inner helper to fetch individual alphabet pages
    def fetch_alphabet_page(url):
        resp = safe_get(url)
        if not resp:
            return []
        soup = BeautifulSoup(resp.text, "html.parser")
        links = []
        for a in soup.find_all("a"):
            href = a.get("href", "")
            if href.startswith(("./meds/", "/meds/")):
                from urllib.parse import urljoin
                full_url = urljoin(url, href)
                links.append((a.text.strip(), full_url))
        return links

    with ThreadPoolExecutor(max_workers=8) as executor:
        results = list(executor.map(fetch_alphabet_page, urls))
        for r in results:
            drug_links.extend(r)

    # De-duplicate
    drug_links = list(dict.fromkeys(drug_links))[:max_drugs]
    log.info(f"  Found {len(drug_links)} unique drug links. Processing up to {max_drugs}...")

    records = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(scrape_single_drug, link): link for link in drug_links}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping Drugs"):
            res = future.result()
            if res:
                records.append(res)

    save_jsonl(records, out)


def scrape_medlineplus_topics():
    """Scrape health topic pages (diseases, conditions)."""
    out = BASE_DIR / "health" / "medlineplus_topics.jsonl"
    log.info("── MedlinePlus Health Topics ──")

    records = []
    for topic_name, slug in tqdm(MEDLINEPLUS_TOPICS, desc="MedlinePlus topics"):
        url  = f"https://medlineplus.gov/{slug}"
        resp = safe_get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")

        content = soup.find("div", {"id": "ency_summary"}) or soup.find("article")
        if not content:
            content = soup.find("div", class_=re.compile(r"health-topic|topic-summary"))

        text = content.get_text(separator=" ", strip=True) if content else ""
        text = re.sub(r'\s+', ' ', text)
        if len(text) < 100:
            continue

        records.append({
            "id":       doc_id(url),
            "domain":   "health",
            "source":   "medlineplus_topics",
            "title":    topic_name,
            "url":      url,
            "text":     text,
            "language": "en",
            "scraped_at": datetime.utcnow().isoformat(),
        })
        polite_sleep()

    save_jsonl(records, out)


def scrape_single_nhp_disease(url: str) -> Optional[dict]:
    polite_sleep()
    resp = safe_get(url)
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")

    title   = soup.find("h1")
    content = soup.find("div", class_=re.compile(r"field-item|body|content"))
    if not content:
        return None

    text = re.sub(r'\s+', ' ', content.get_text(separator=" ", strip=True))
    if len(text) < 100:
        return None

    return {
        "id":       doc_id(url),
        "domain":   "health",
        "source":   "nhp_india",
        "title":    title.get_text(strip=True) if title else url.split("/")[-1],
        "url":      url,
        "text":     text,
        "language": "en",
        "scraped_at": datetime.utcnow().isoformat(),
    }


def scrape_nhp_india(max_pages: int = 500):
    """Scrape NHP India disease fact sheets concurrently."""
    out = BASE_DIR / "health" / "nhp_india.jsonl"
    log.info("── NHP India (Multithreaded) ──")

    resp = safe_get("https://www.nhp.gov.in/disease-a-z")
    if not resp:
        log.warning("  NHP index unreachable — skipping")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    links = []
    for a in soup.select("a[href*='/disease/']"):
        href = a.get("href", "")
        full = href if href.startswith("http") else "https://www.nhp.gov.in" + href
        links.append(full)
    links = list(set(links))[:max_pages]
    log.info(f"  Found {len(links)} NHP disease fact sheets to process...")

    records = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(scrape_single_nhp_disease, url): url for url in links}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping NHP Diseases"):
            res = future.result()
            if res:
                records.append(res)

    save_jsonl(records, out)


# ══════════════════════════════════════════════════════════════
#  LEGAL / GOVERNMENT  ─  MyScheme.gov.in + India Code
# ══════════════════════════════════════════════════════════════

INDIA_CODE_ACTS = [
    ("RTI Act 2005",        "https://www.indiacode.nic.in/bitstream/123456789/2065/1/aa2005.pdf"),
    ("PM-KISAN Guidelines", "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf"),
    ("MGNREGA Act",         "https://www.indiacode.nic.in/bitstream/123456789/2014/5/a2005-42.pdf"),
    ("Consumer Protection", "https://www.indiacode.nic.in/bitstream/123456789/15256/1/eng201935.pdf"),
    ("Aadhaar Act",         "https://www.indiacode.nic.in/bitstream/123456789/2160/1/engaadhaar.pdf"),
    ("National Food Security Act", "https://prsindia.org/files/bills_acts/acts_parliament/2013/national-food-security-act-2013.pdf"),
    ("Digital Personal Data Protection Act", "https://www.meity.gov.in/static/uploads/2024/06/2bf1f0e9f04e6fb4f8fef35e82c42aa5.pdf"),
    ("Bharatiya Nyaya Sanhita", "https://prsindia.org/files/bills_acts/bills_parliament/2023/Bharatiya_Nyaya_Sanhita,_2023.pdf"),
]


def scrape_single_scheme(url: str) -> Optional[dict]:
    polite_sleep()
    resp = safe_get(url)
    if not resp:
        return None
    soup = BeautifulSoup(resp.text, "html.parser")

    title_el = soup.find("h1") or soup.find("h2")
    title = title_el.get_text(strip=True) if title_el else url.split("/")[-1]

    sections = {}
    for heading in soup.find_all(re.compile(r"h[2-5]")):
        label = heading.get_text(strip=True)
        content_parts = []
        for sib in heading.find_next_siblings():
            if sib.name and re.match(r"h[2-5]", sib.name):
                break
            content_parts.append(sib.get_text(separator=" ", strip=True))
        if content_parts:
            sections[label] = " ".join(content_parts)

    text = " | ".join(f"{k}: {v}" for k, v in sections.items())
    if not text:
        text = re.sub(r'\s+', ' ', soup.get_text(separator=" ", strip=True))[:3000]

    if len(text) < 100:
        return None

    return {
        "id":       doc_id(url),
        "domain":   "legal",
        "source":   "myscheme",
        "title":    title,
        "url":      url,
        "sections": sections,
        "text":     text,
        "language": "en",
        "scraped_at": datetime.utcnow().isoformat(),
    }


def scrape_myscheme_api(max_pages: int = 50):
    """Scrape MyScheme category pages and scheme detail pages concurrently."""
    out = BASE_DIR / "legal" / "myscheme.jsonl"
    log.info("── MyScheme Scraper (Multithreaded) ──")

    CATEGORIES = [
        "agriculture-rural-environment",
        "banking-financial-services-insurance",
        "education-learning",
        "health-wellness",
        "housing-shelter",
        "public-safety-law-justice",
        "science-it-communications",
        "skills-employment",
        "social-welfare-empowerment",
        "sports-culture",
        "women-child",
    ]

    scheme_urls = []
    for cat in CATEGORIES:
        url  = f"https://www.myscheme.gov.in/search?category={cat}"
        resp = safe_get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")
        for a in soup.select("a[href^='/schemes/']"):
            href = a["href"]
            full = "https://www.myscheme.gov.in" + href
            scheme_urls.append(full)
        polite_sleep()

    scheme_urls = list(dict.fromkeys(scheme_urls))
    log.info(f"  Found {len(scheme_urls)} total scheme pages. Crawling them concurrently...")

    records = []
    with ThreadPoolExecutor(max_workers=NUM_WORKERS) as executor:
        futures = {executor.submit(scrape_single_scheme, url): url for url in scheme_urls}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping MyScheme"):
            res = future.result()
            if res:
                records.append(res)

    save_jsonl(records, out)


def scrape_india_code_acts():
    """Download key legislation PDFs and extract their pages."""
    reader_builder = load_pdf_reader()
    if not reader_builder:
        return

    out   = BASE_DIR / "legal" / "india_code_acts.jsonl"
    pdfs  = BASE_DIR / "legal" / "pdfs"
    pdfs.mkdir(exist_ok=True)
    log.info("── India Code Central Legislation Acts ──")

    records = []
    for act_name, url in tqdm(INDIA_CODE_ACTS, desc="Central Acts"):
        fname = pdfs / (act_name.replace(" ", "_") + ".pdf")

        if not fname.exists():
            log.info(f"  Downloading PDF: {act_name}")
            resp = safe_get(url)
            if not resp:
                continue
            fname.write_bytes(resp.content)
            polite_sleep()

        try:
            reader = reader_builder(fname)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) < 100:
                    continue
                records.append({
                    "id":       doc_id(f"{act_name}_p{i}"),
                    "domain":   "legal",
                    "source":   "india_code",
                    "title":     act_name,
                    "page":      i + 1,
                    "url":       url,
                    "text":      text,
                    "language":  "en",
                    "scraped_at": datetime.utcnow().isoformat(),
                })
        except Exception as e:
            log.warning(f"  Failed parsing PDF {act_name}: {e}")

    save_jsonl(records, out)


# ══════════════════════════════════════════════════════════════
#  AGRICULTURE  ─  ICAR + KCC FAQs + PM-KISAN + Soil Health
# ══════════════════════════════════════════════════════════════

ICAR_CROP_ADVISORIES = [
    ("Wheat",      "https://www.icar.org.in/index.php/en/component/content/article?id=219"),
    ("Rice",       "https://www.icar.org.in/index.php/en/component/content/article?id=220"),
    ("Maize",      "https://www.icar.org.in/index.php/en/component/content/article?id=223"),
    ("Cotton",     "https://www.icar.org.in/index.php/en/component/content/article?id=226"),
    ("Sugarcane",  "https://www.icar.org.in/index.php/en/component/content/article?id=224"),
    ("Soybean",    "https://farmer.gov.in/cropadvise.aspx?ID=8"),
    ("Groundnut",  "https://farmer.gov.in/cropadvise.aspx?ID=5"),
    ("Tomato",     "https://farmer.gov.in/cropadvise.aspx?ID=24"),
    ("Potato",     "https://farmer.gov.in/cropadvise.aspx?ID=19"),
    ("Chickpea",   "https://farmer.gov.in/cropadvise.aspx?ID=11"),
    ("Mustard",    "https://farmer.gov.in/cropadvise.aspx?ID=18"),
    ("Onion",      "https://farmer.gov.in/cropadvise.aspx?ID=22"),
]

KCC_FAQ_URL = "https://farmer.gov.in/faq.aspx"

PMKISAN_DOCS = [
    ("PM-KISAN Operational Guidelines (EN)",
     "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf"),
    ("PM-KISAN Operational Guidelines (HI)",
     "https://pmkisan.gov.in/Documents/%E0%A4%AA%E0%A5%8D%E0%A4%B0%E0%A4%A7%E0%A4%BE%E0%A4%A8%E0%A4%AE%E0%A4%82%E0%A4%A4%E0%A5%8D%E0%A4%B0%E0%A5%80%20%E0%A4%95%E0%A4%BF%E0%A4%B8%E0%A4%BE%E0%A4%A8%20%E0%A4%B8%E0%A4%AE%E0%A5%8D%E0%A4%AE%E0%A4%BE%E0%A4%A8%20%E0%A4%A8%E0%A4%BF%E0%A4%A7%E0%A4%BF.pdf"),
    ("RAD-NMSA Guidelines",
     "https://keralaagriculture.gov.in/wp-content/uploads/2024/11/Working-Instructions-RADNMSA-2024-25.pdf"),
]

SOIL_HEALTH_URLS = [
    "https://soilhealth.dac.gov.in/content/crop-specific-fertilizer-recommendation",
    "https://soilhealth.dac.gov.in/content/about-soil-health-card",
    "https://soilhealth.dac.gov.in/content/soil-health-parameters",
]


def scrape_icar_advisories():
    """Scrape ICAR and Farmer Portal crop advisory pages."""
    out = BASE_DIR / "agriculture" / "icar_advisories.jsonl"
    log.info("── Agriculture Crop Advisories ──")

    records = []
    for crop, url in tqdm(ICAR_CROP_ADVISORIES, desc="Crop advisories"):
        resp = safe_get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")

        content = (
            soup.find("div", class_="field-body")
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|main|body"))
        )
        if not content:
            content = soup

        full_text = re.sub(r'\s+', ' ', content.get_text(separator=" ", strip=True))
        if len(full_text) < 100:
            continue

        records.append({
            "id":       doc_id(url),
            "domain":   "agriculture",
            "source":   "icar",
            "crop":     crop,
            "title":    f"Advisory: {crop}",
            "url":      url,
            "text":     full_text,
            "language": "en",
            "scraped_at": datetime.utcnow().isoformat(),
        })
        polite_sleep()

    save_jsonl(records, out)


def scrape_kisan_call_centre():
    """Scrape Kisan Call Centre FAQ pages."""
    out = BASE_DIR / "agriculture" / "kcc_faqs.jsonl"
    log.info("── Kisan Call Centre FAQs ──")

    resp = safe_get(KCC_FAQ_URL)
    if not resp:
        log.warning("  KCC FAQ page unreachable")
        return

    soup = BeautifulSoup(resp.text, "html.parser")
    records = []
    qa_pairs = []

    for panel in soup.select("div.panel, div.accordion-item, div.faq-item"):
        q_el = panel.find(class_=re.compile(r"panel-title|question|accordion-header"))
        a_el = panel.find(class_=re.compile(r"panel-body|answer|accordion-body"))
        if q_el and a_el:
            qa_pairs.append((q_el.get_text(strip=True), a_el.get_text(strip=True)))

    if not qa_pairs:
        for dt in soup.find_all("dt"):
            dd = dt.find_next_sibling("dd")
            if dd:
                qa_pairs.append((dt.get_text(strip=True), dd.get_text(strip=True)))

    log.info(f"  Found {len(qa_pairs)} FAQ pairs")

    for q, a in qa_pairs:
        text = f"Q: {q} A: {a}"
        records.append({
            "id":       doc_id(text),
            "domain":   "agriculture",
            "source":   "kcc_faq",
            "title":    q[:100],
            "url":      KCC_FAQ_URL,
            "question": q,
            "answer":   a,
            "text":     text,
            "language": "en",
            "scraped_at": datetime.utcnow().isoformat(),
        })

    save_jsonl(records, out)


def scrape_pmkisan_docs():
    """Download and extract PM-KISAN guideline PDFs."""
    reader_builder = load_pdf_reader()
    if not reader_builder:
        return

    out  = BASE_DIR / "agriculture" / "pmkisan_docs.jsonl"
    pdfs = BASE_DIR / "agriculture" / "pdfs"
    pdfs.mkdir(exist_ok=True)
    log.info("── PM-KISAN Operational Guidelines ──")

    records = []
    for title, url in tqdm(PMKISAN_DOCS, desc="PM-KISAN Guidelines"):
        lang  = "hi" if "Hindi" in title else "en"
        fname = pdfs / (title.replace(" ", "_").replace("(", "").replace(")", "") + ".pdf")

        if not fname.exists():
            log.info(f"  Downloading PDF: {title}")
            resp = safe_get(url)
            if not resp:
                continue
            fname.write_bytes(resp.content)
            polite_sleep()

        try:
            reader = reader_builder(fname)
            for i, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                text = re.sub(r'\s+', ' ', text).strip()
                if len(text) < 80:
                    continue
                records.append({
                    "id":       doc_id(f"{title}_p{i}"),
                    "domain":   "agriculture",
                    "source":   "pmkisan",
                    "title":    title,
                    "page":     i + 1,
                    "url":      url,
                    "text":     text,
                    "language": lang,
                    "scraped_at": datetime.utcnow().isoformat(),
                })
        except Exception as e:
            log.warning(f"  Failed parsing PDF {title}: {e}")

    save_jsonl(records, out)


def scrape_soil_health_portal():
    """Scrape Soil Health Card portal content."""
    out = BASE_DIR / "agriculture" / "soil_health.jsonl"
    log.info("── Soil Health Card Portal ──")

    records = []
    for url in tqdm(SOIL_HEALTH_URLS, desc="Soil health portal"):
        resp = safe_get(url)
        if not resp:
            continue
        soup = BeautifulSoup(resp.text, "html.parser")

        content = (
            soup.find("div", class_=re.compile(r"field-body|content|main-content"))
            or soup.find("main")
            or soup.find("article")
        )
        if not content:
            content = soup

        tables = []
        for table in content.find_all("table"):
            rows = []
            for tr in table.find_all("tr"):
                row = [td.get_text(strip=True) for td in tr.find_all(["td", "th"])]
                rows.append(" | ".join(row))
            tables.append("\n".join(rows))

        text = re.sub(r'\s+', ' ', content.get_text(separator=" ", strip=True))
        if tables:
            text += " TABLE_DATA: " + " || ".join(tables)

        records.append({
            "id":       doc_id(url),
            "domain":   "agriculture",
            "source":   "soil_health_portal",
            "title":    soup.title.get_text(strip=True) if soup.title else url,
            "url":      url,
            "text":     text,
            "language": "en",
            "scraped_at": datetime.utcnow().isoformat(),
        })
        polite_sleep()

    save_jsonl(records, out)


# ══════════════════════════════════════════════════════════════
#  STATS REPORTER & MAIN RUNNER
# ══════════════════════════════════════════════════════════════

def print_stats():
    print("\n" + "="*55)
    print("  TriSeva Scrape Summary")
    print("="*55)
    total = 0
    for domain in ["health", "legal", "agriculture"]:
        domain_total = 0
        for jl in (BASE_DIR / domain).glob("*.jsonl"):
            count = sum(1 for _ in open(jl, encoding="utf-8"))
            domain_total += count
            print(f"  {domain:<14} {jl.name:<35} {count:>6} docs")
        total += domain_total
        print(f"  {'':14} {'SUBTOTAL':<35} {domain_total:>6}")
        print()
    print(f"  {'TOTAL':>51} {total:>6} docs")
    print("="*55)


def run_health():
    log.info("╔══ HEALTH ══╗")
    scrape_medlineplus_drugs(max_drugs=1400)
    scrape_medlineplus_topics()
    scrape_nhp_india(max_pages=300)


def run_legal():
    log.info("╔══ LEGAL / GOVERNMENT ══╗")
    scrape_myscheme_api()
    scrape_india_code_acts()


def run_agriculture():
    log.info("╔══ AGRICULTURE ══╗")
    scrape_icar_advisories()
    scrape_kisan_call_centre()
    scrape_pmkisan_docs()
    scrape_soil_health_portal()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="TriSeva KB Scraper")
    parser.add_argument(
        "--domain",
        choices=["health", "legal", "agriculture", "all"],
        default="all",
        help="Which domain to scrape",
    )
    args = parser.parse_args()

    start = time.time()
    if args.domain in ("health", "all"):
        run_health()
    if args.domain in ("legal", "all"):
        run_legal()
    if args.domain in ("agriculture", "all"):
        run_agriculture()

    elapsed = time.time() - start
    log.info(f"\nDone in {elapsed:.1f}s")
    print_stats()