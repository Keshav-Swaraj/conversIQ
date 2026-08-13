"""Document Processing Module - Modern Fast-Path with PyMuPDF & RecursiveCharacterTextSplitter."""

import os
import re
import time
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class HierarchicalChunk:
    chunk_id: str
    section: str
    paragraph_index: int
    content: str
    sentences: List[str]
    word_count: int


class PDFDocumentProcessor:
    """Ultra-fast document loader and hierarchical splitter using PyMuPDF and RecursiveCharacterTextSplitter."""

    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " "]
        )

    def extract_raw_pages(self, pdf_path: str) -> List[Dict[str, Any]]:
        """Extract text using ultra-fast PyMuPDF C-engine."""
        abs_path = os.path.abspath(pdf_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"PDF file not found at: {abs_path}")
            
        loader = PyMuPDFLoader(abs_path)
        docs = loader.load()
        return [{"page_number": doc.metadata.get("page", 0) + 1, "raw_text": doc.page_content} for doc in docs]

    @staticmethod
    def clean_text(text: str) -> str:
        """Normalize page numbers, whitespace, and formatting."""
        cleaned = re.sub(r'(?i)page\s+\d+(\s+of\s+\d+)?', '', text)
        paragraphs = [re.sub(r'\s+', ' ', p).strip() for p in cleaned.split('\n\n') if p.strip()]
        return '\n\n'.join(paragraphs)

    def process(self, pdf_path: str) -> List[HierarchicalChunk]:
        """Ultra-fast document ingestion."""
        chunks, _ = self.process_with_timings(pdf_path)
        return chunks

    def process_with_timings(self, pdf_path: str) -> Tuple[List[HierarchicalChunk], Dict[str, float]]:
        """Document ingestion with detailed time.perf_counter() micro-profiling metrics."""
        abs_path = os.path.abspath(pdf_path)
        if not os.path.exists(abs_path):
            raise FileNotFoundError(f"PDF file not found at: {abs_path}")

        timings: Dict[str, float] = {}

        # 1. PDF Loading via PyMuPDF C-engine
        t0 = time.perf_counter()
        loader = PyMuPDFLoader(abs_path)
        raw_docs = loader.load()
        timings["pdf_loading_ms"] = (time.perf_counter() - t0) * 1000.0

        # 2. Text Extraction & Cleaning
        t1 = time.perf_counter()
        cleaned_docs = []
        for doc in raw_docs:
            c_text = self.clean_text(doc.page_content)
            if c_text.strip():
                doc.page_content = c_text
                cleaned_docs.append(doc)
        timings["text_cleaning_ms"] = (time.perf_counter() - t1) * 1000.0

        # 3. Recursive Text Chunking
        t2 = time.perf_counter()
        split_docs = self.text_splitter.split_documents(cleaned_docs)
        timings["text_chunking_ms"] = (time.perf_counter() - t2) * 1000.0

        # 4. Chunk Hierarchy Formatting
        t3 = time.perf_counter()
        chunks: List[HierarchicalChunk] = []
        for idx, doc in enumerate(split_docs):
            content = doc.page_content.strip()
            if not content.endswith(('.', '!', '?', ':', ';')):
                content += '.'
            
            lines = content.split('\n')
            first_line = lines[0].strip()
            sec_title = first_line if re.match(r'^\d+\.\s+[A-Z]', first_line) else f"Page {doc.metadata.get('page', 0) + 1} Context"
            sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', content) if s.strip()]

            chunks.append(HierarchicalChunk(
                chunk_id=f"chunk_{idx+1:03d}",
                section=sec_title,
                paragraph_index=idx + 1,
                content=content,
                sentences=sentences,
                word_count=len(content.split())
            ))
        timings["hierarchy_formatting_ms"] = (time.perf_counter() - t3) * 1000.0
        timings["total_ingestion_ms"] = sum(timings.values())

        return chunks, timings

