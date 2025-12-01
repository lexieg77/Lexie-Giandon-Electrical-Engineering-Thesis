"""
Preprocessing utilities: chunking, embeddings, vectorstores.
"""

import numpy as np
from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from sklearn.neighbors import KNeighborsClassifier
import joblib
from config import EMBEDDING_MODEL, VECTORSTORE_PARTS_PATH, VECTORSTORE_FAILURES_PATH, VECTORSTORE_PDF_PATH, KNN_MODEL_PATH


def create_chunks(texts, chunk_size=1000, overlap=100):
    """Create chunks from a list of texts."""
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=overlap)
    chunks = []
    for text in texts:
        chunks.extend(splitter.split_text(text))
    return chunks


def build_vectorstores(parts_list, failure_list, pdf_chunks):
    """Build FAISS vectorstores for parts, failures, and PDFs."""
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)

    # Parts
    parts_chunks = create_chunks(parts_list)
    parts_vectorstore = FAISS.from_texts(parts_chunks, embedding_model)
    parts_vectorstore.save_local(VECTORSTORE_PARTS_PATH)

    # Failures
    all_failure_text = "\n".join(failure_list)
    failure_chunks = create_chunks([all_failure_text])
    failure_vectorstore = FAISS.from_texts(failure_chunks, embedding_model)
    failure_vectorstore.save_local(VECTORSTORE_FAILURES_PATH)

    # PDFs
    pdf_all_chunks = []
    for name, chunks in pdf_chunks.items():
        pdf_all_chunks.extend([f"{name}: {chunk}" for chunk in chunks])
    pdf_vectorstore = FAISS.from_texts(pdf_all_chunks, embedding_model)
    pdf_vectorstore.save_local(VECTORSTORE_PDF_PATH)

    return parts_vectorstore, failure_vectorstore, pdf_vectorstore


def load_vectorstores():
    """Load pre-built vectorstores."""
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    parts_vectorstore = FAISS.load_local(VECTORSTORE_PARTS_PATH, embedding_model, allow_dangerous_deserialization=True)
    failure_vectorstore = FAISS.load_local(VECTORSTORE_FAILURES_PATH, embedding_model, allow_dangerous_deserialization=True)
    pdf_vectorstore = FAISS.load_local(VECTORSTORE_PDF_PATH, embedding_model, allow_dangerous_deserialization=True)
    return parts_vectorstore, failure_vectorstore, pdf_vectorstore


def train_knn(pdf_chunks):
    """Train KNN classifier on PDF chunks."""
    embedding_model = OpenAIEmbeddings(model=EMBEDDING_MODEL)
    texts = []
    labels = []
    for label, chunks in pdf_chunks.items():
        texts.extend(chunks)
        labels.extend([label] * len(chunks))

    X = [embedding_model.embed_query(t) for t in texts]
    X = np.array(X)
    y = np.array(labels)
    clf = KNeighborsClassifier(n_neighbors=3)
    clf.fit(X, y)
    joblib.dump(clf, KNN_MODEL_PATH)
    return clf


def load_knn():
    """Load pre-trained KNN classifier."""
    return joblib.load(KNN_MODEL_PATH)
