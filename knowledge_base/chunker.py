import os
import pymupdf as fitz
from typing import List, Dict


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract all text from a PDF file."""
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    doc.close()
    return text


def chunk_text(
    text: str,
    source: str,
    domain: str,
    chunk_size: int = 250,
    overlap: int = 30,
) -> List[Dict]:
    """
    Split text into overlapping chunks.
    Returns list of dicts with text, source, domain, chunk_id.
    """
    words = text.split()
    chunks = []
    chunk_id = 0

    i = 0
    while i < len(words):
        chunk_words = words[i : i + chunk_size]
        chunk_text = " ".join(chunk_words).strip()

        if len(chunk_text) > 100:  # skip tiny chunks
            chunks.append({
                "text":     chunk_text,
                "source":   source,
                "domain":   domain,
                "chunk_id": f"{domain}_{os.path.basename(source)}_{chunk_id}",
            })
            chunk_id += 1

        i += chunk_size - overlap  # overlap between chunks

    return chunks


def chunk_pdf(pdf_path: str, domain: str, chunk_size: int = 250, overlap: int = 30) -> List[Dict]:
    """Extract and chunk a PDF file."""
    print(f"  Chunking: {os.path.basename(pdf_path)}")
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text, source=pdf_path, domain=domain, chunk_size=chunk_size, overlap=overlap)


def chunk_raw_text(text: str, source: str, domain: str, chunk_size: int = 250, overlap: int = 30) -> List[Dict]:
    """Chunk a raw text string directly."""
    return chunk_text(text, source=source, domain=domain, chunk_size=chunk_size, overlap=overlap)