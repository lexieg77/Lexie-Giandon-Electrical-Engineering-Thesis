"""
Main entry point for inference.
Run this script to perform failure-to-part mapping.
"""

import pandas as pd
from data_loader import load_csvs, load_pdfs
from preprocessor import create_chunks, build_vectorstores, train_knn
from inference import similarity_inference, supervised_inference
from config import OUTPUT_MAPPING_SIMILARITY, OUTPUT_MAPPING_SUPERVISED


def main():
    # Load data
    non_generic_parts, downtime_history = load_csvs()
    pdf_texts = load_pdfs()

    # Prepare lists
    parts_list = non_generic_parts.apply(lambda row: f"{row.get('CAT Part', '')}: {row.get('SAP Material Description', '')}", axis=1).tolist()
    failure_list = downtime_history.apply(
        lambda row: f"{row.get('Order', '')}: {row.get('Notification', '')}: {row.get('Order Type', '')}: {row.get('Order Long Text Description', '')}: {row.get('Notification Long Text Description', '')}: {row.get('Sort Field', '')}: {row.get('Total Costs', '')}: {row.get('Total Work Hours ', '')}",
        axis=1,
    ).tolist()

    # Chunk PDFs
    pdf_chunks = {}
    for name, text in pdf_texts.items():
        chunks = create_chunks([text])  # Assuming create_chunks takes list
        pdf_chunks[name] = chunks

    # Build and save models (run once for training)
    print("Building vectorstores and training KNN...")
    build_vectorstores(parts_list, failure_list, pdf_chunks)
    train_knn(pdf_chunks)
    print("Models saved.")

    # Perform inference
    print("Running similarity inference...")
    similarity_results = similarity_inference(failure_list, parts_list)
    pd.DataFrame(similarity_results).to_csv(OUTPUT_MAPPING_SIMILARITY, index=False)
    print(f"Saved similarity results to {OUTPUT_MAPPING_SIMILARITY}")

    print("Running supervised inference...")
    supervised_results = supervised_inference(failure_list, parts_list, pdf_chunks)
    pd.DataFrame(supervised_results).to_csv(OUTPUT_MAPPING_SUPERVISED, index=False)
    print(f"Saved supervised results to {OUTPUT_MAPPING_SUPERVISED}")


if __name__ == "__main__":
    main()
