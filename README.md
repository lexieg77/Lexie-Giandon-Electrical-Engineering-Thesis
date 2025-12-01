# Critical Spares Project

This project maps equipment failures to corresponding spare parts using FAISS for similarity search, key-value pairs, unsupervised clustering, and supervised classification. It also generates critical spare parts lists and calculates F1 scores for performance evaluation.

## Features

- Load and preprocess CSV data (spare parts, downtime history) and PDF documents.
- Build FAISS vectorstores for efficient similarity search.
- Train a KNN classifier for supervised PDF-based retrieval.
- Perform inference: Map failures to parts using similarity or supervised methods.
- Generate critical spare parts lists with criticality scoring.
- Calculate F1 scores to evaluate prediction accuracy.
- Save results to CSV for analysis.

## Installation

1. Clone the repository.
2. Install dependencies: `pip install -r requirements.txt`
3. Set your OpenAI API key in `src/config.py` or via environment variable `OPENAI_API_KEY`.

## Usage

1. Customize paths in `src/config.py`.
2. Run the main script: `python src/main.py`

This will build models (if not present) and perform inference, saving outputs to configured paths.

## Project Structure

- `src/`: Source code
  - `config.py`: Configuration settings
  - `data_loader.py`: Data loading utilities
  - `preprocessor.py`: Chunking, embeddings, vectorstores, KNN training
  - `inference.py`: Inference functions
  - `main.py`: Entry point
- `models/`: Saved models and vectorstores
- `requirements.txt`: Python dependencies
- `README.md`: This file
