"""
Configuration file for the project.
Customize these settings for your environment.
"""

import os

# OpenAI API Key - set your key here or load from .env
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_openai_api_key_here")

# File paths - update to your local or cloud paths
NON_GENERIC_PARTS_PATH = r"C:/Users/lgian/OneDrive - James Cook University/Data/Sample_Spare_Parts2.csv"
DOWNTIME_HISTORY_PATH = r"C:/Users/lgian/OneDrive - James Cook University/Data/Downtime_History_Data.csv"
PDF_FILES = {
    "KXE_brochure": r"C:/Users/lgian/OneDrive - James Cook University/Data/998KXE_Brochure.pdf",
    "XE_brochure": r"C:/Users/lgian/OneDrive - James Cook University/Data/998XE_Brochure.pdf",
    "XE_specs": r"C:/Users/lgian/OneDrive - James Cook University/Data/998XE_Specifications.pdf",
}
ORIGINAL_CRITICAL_SPARES_PATH = r"C:/Users/lgian/OneDrive - James Cook University/Data/Original_Critical_Spare_Parts_List.csv"
ORIGINAL_F1_PATH = r"C:/Users/lgian/OneDrive - James Cook University/Data/f1_scores/original_500.csv"
PREDICTED_F1_PATH = r"C:/Users/lgian/OneDrive - James Cook University/Data/f1_scores/new_500.csv"
OUTPUT_MAPPING_SIMILARITY = r"C:/Users/lgian/OneDrive - James Cook University/Data/Mapping_Similarity.csv"
OUTPUT_MAPPING_SUPERVISED = r"C:/Users/lgian/OneDrive - James Cook University/Data/Mapping_Supervised.csv"

# Model settings
LLM_MODEL = "o3-mini"  # or "o4-mini"
EMBEDDING_MODEL = "text-embedding-3-small"
K_CLUSTERS = 10  # for FAISS KMeans
MAX_FAILURES_TO_PROCESS = 100  # limit for loops

# Paths for saved models/vectorstores
VECTORSTORE_PARTS_PATH = "models/parts_vectorstore"
VECTORSTORE_FAILURES_PATH = "models/failures_vectorstore"
VECTORSTORE_PDF_PATH = "models/pdf_vectorstore"
KNN_MODEL_PATH = "models/knn_classifier.pkl"
