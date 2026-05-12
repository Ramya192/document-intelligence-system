# tools/pdf_extractor.py
# Extracts text from a PDF file and splits it into chunks for embedding.
# Supports both file path (main.py) and raw bytes (Streamlit upload).

import pdfplumber
import io
from langchain_text_splitters import RecursiveCharacterTextSplitter
from config.settings import CHUNK_SIZE, CHUNK_OVERLAP

class PDFExtractor:
    def __init_self(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size = CHUNK_SIZE,
            chunk_overlap = CHUNK_OVERLAP,
            separators = ["/n/n","/n"," ",""]
        )

    def extract_from_path(self, pdf_path: str) -> str:
        with pdfplumber.open(pdf_path) as pdf:
            return self._extract_pages(pdf)
        
    def extract_from_bytes(self, pdf_bytes: bytes) -> str:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return self._extract_pages(self)
        
    def _extract_pages(self, pdf) -> str:
        pages = []
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append(f"[Page {i+1}]\n{text.strip()}")
        return "\n\n".join(pages)
    
    def chunk(self, text:str) -> list[str]:
        return self.splitter.split_text(text)
    
    def extract_and_chunk(self, pdf_path: str) -> tuple[str, list[str]]:
        full_text = self.extract_from_path(pdf_path)
        chunks = self.chunk(full_text)
        return full_text, chunks
    
    def extract_and_chunk_bytes(self, pdf_bytes: bytes) -> tuple[str, list[str]]:
        full_text = self.extract_from_bytes(pdf_bytes)
        chunks = self.chunk(full_text)
        return full_text, chunks