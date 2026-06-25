import os
import pymupdf as fitz
from typing import List, Dict
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
    chunk_size: int = 1000,
    chunk_overlap: int = 150,
) -> List[Dict]:
    """
    Split text into overlapping chunks using RecursiveCharacterTextSplitter.
    Preserves paragraph and sentence integrity.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    splits = splitter.split_text(text)
    
    chunks = []
    for chunk_id, split_text in enumerate(splits):
        cleaned_text = split_text.strip()
        if len(cleaned_text) > 100:  # Skip tiny structural artifacts
            chunks.append({
                "text":     cleaned_text,
                "source":   source,
                "domain":   domain,
                "chunk_id": f"{domain}_{os.path.basename(source)}_{chunk_id:04d}",
            })
            
    return chunks


def chunk_pdf(pdf_path: str, domain: str, chunk_size: int = 1000, overlap: int = 150) -> List[Dict]:
    """Extract and chunk a PDF file."""
    print(f"  Chunking: {os.path.basename(pdf_path)}")
    text = extract_text_from_pdf(pdf_path)
    return chunk_text(text, source=pdf_path, domain=domain, chunk_size=chunk_size, chunk_overlap=overlap)


def chunk_raw_text(text: str, source: str, domain: str, chunk_size: int = 1000, overlap: int = 150) -> List[Dict]:
    """Chunk a raw text string directly."""
    return chunk_text(text, source=source, domain=domain, chunk_size=chunk_size, chunk_overlap=overlap)