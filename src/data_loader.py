"""
Data loading utilities.
"""

import pandas as pd
from PyPDF2 import PdfReader
from config import NON_GENERIC_PARTS_PATH, DOWNTIME_HISTORY_PATH, PDF_FILES


def load_csvs():
    """Load the spare parts and downtime history CSVs."""
    non_generic_parts = pd.read_csv(NON_GENERIC_PARTS_PATH)
    downtime_history = pd.read_csv(DOWNTIME_HISTORY_PATH, encoding="cp1252", low_memory=False)
    return non_generic_parts, downtime_history


def extract_text_from_pdf(pdf_path):
    """Extract text from a PDF file."""
    text = ""
    reader = PdfReader(pdf_path)
    for page in reader.pages:
        text += page.extract_text() or ""
    return text


def load_pdfs():
    """Load and extract text from PDF files."""
    pdf_texts = {name: extract_text_from_pdf(path) for name, path in PDF_FILES.items()}
    return pdf_texts
