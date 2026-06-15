import os
import re
import time
import requests
from pathlib import Path
from bs4 import BeautifulSoup
from urllib.parse import urlparse

BASE_DIR = Path(__file__).resolve().parent / "raw"
BASE_DIR.mkdir(parents=True, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/125.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
    "Accept-Language": "en-IN,en;q=0.9",
    "Connection": "keep-alive",
    "Upgrade-Insecure-Requests": "1",
}

session = requests.Session()
session.headers.update(HEADERS)

URLS = {
    "health": [
        "https://medlineplus.gov/",
        "https://datahub.hhs.gov/NIH/MedlinePlus/afxd-vfgu",
    ],
    "legal": [
        "https://www.myscheme.gov.in/",
        "https://lawmin.gov.in/acts/acts-rules",
        "https://services.india.gov.in/service/detail/india-code-digital-repository-of-all-central-and-state-acts",
        "https://www.mha.gov.in/en/acts",
    ],
    "agriculture": [
        "https://pmkisan.gov.in/",
        "https://pmkisan.gov.in/Documents/RevisedPM-KISANOperationalGuidelines(English).pdf",
        "https://icar.org.in/en/weather-based-crop-advisory",
        "https://icar.org.in/weather-based-crop-advisory",
        "https://www.pib.gov.in/FactsheetDetails.aspx?Id=148602",
    ],
}

def safe_name(url):
    path = urlparse(url).path.strip("/")
    query = urlparse(url).query
    name = path if path else "index"
    if query:
        name += "_" + query.replace("=", "_").replace("&", "_")
    name = re.sub(r"[^a-zA-Z0-9._-]+", "_", name)
    return name[:150]

def ensure_dir(domain):
    d = BASE_DIR / domain
    d.mkdir(parents=True, exist_ok=True)
    return d

def is_pdf(resp, url):
    ctype = resp.headers.get("Content-Type", "").lower()
    return "application/pdf" in ctype or url.lower().endswith(".pdf")

def fetch(url, retries=3, delay=2):
    for attempt in range(retries):
        try:
            resp = session.get(url, timeout=40, allow_redirects=True)
            resp.raise_for_status()
            return resp
        except Exception as e:
            if attempt == retries - 1:
                raise
            time.sleep(delay)

def html_to_text(html):
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript"]):
        tag.decompose()
    blocks = []
    for tag in soup.find_all(["h1", "h2", "h3", "p", "li", "td", "th"]):
        text = tag.get_text(" ", strip=True)
        if text:
            blocks.append(text)
    return "\n".join(blocks)

def save_response(domain, url, resp):
    out_dir = ensure_dir(domain)
    base = safe_name(str(resp.url))

    if is_pdf(resp, str(resp.url)):
        out_name = base if base.lower().endswith(".pdf") else f"{base}.pdf"
        out = out_dir / out_name
        out.write_bytes(resp.content)
        print(f"Saved PDF: {out}")
    else:
        text = html_to_text(resp.text)
        out = out_dir / f"{base}.txt"
        out.write_text(text, encoding="utf-8", errors="ignore")
        print(f"Saved text: {out}")

for domain, urls in URLS.items():
    print(f"\n=== Downloading {domain} ===")
    for url in urls:
        try:
            resp = fetch(url)
            save_response(domain, url, resp)
        except Exception as e:
            print(f"Failed {url}: {e}")

print("\nDone.")