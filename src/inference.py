"""
Inference utilities for mapping failures to parts.
"""

from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from config import LLM_MODEL, MAX_FAILURES_TO_PROCESS
from preprocessor import load_vectorstores, load_knn


# Prompt template
PROMPT_TEMPLATE = ChatPromptTemplate.from_template(
    """
    You are a reliability engineer.
    You have the following list of part numbers and descriptions:
    {parts_context}
    Task: For the given failure description, return the most relevant PartNumber.
    If none of the part numbers seem to fit, return "Unknown".
    Use this extra context to make informed decisions: {KXE_brochure_context}, {XE_brochure_context}, {XE_specs_context}
    Failure: {failure_context}
    Answer with ONLY the PartNumber or "Unknown".
    Please output your thinking process as well as your final answer.
    """
)


def load_models():
    """Load pre-trained models and vectorstores."""
    parts_vectorstore, failure_vectorstore, pdf_vectorstore = load_vectorstores()
    clf = load_knn()
    llm = ChatOpenAI(model=LLM_MODEL)
    return parts_vectorstore, pdf_vectorstore, clf, llm


def similarity_inference(failure_list, parts_list):
    """Perform similarity-based mapping."""
    parts_vectorstore, pdf_vectorstore, clf, llm = load_models()
    parts_retriever = parts_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    pdf_retriever = pdf_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 4})

    failure_to_part = []
    for text in failure_list[:MAX_FAILURES_TO_PROCESS]:
        top_parts = parts_retriever.get_relevant_documents(text)
        parts_context = "\n".join([doc.page_content for doc in top_parts])
        top_pdf_chunks = pdf_retriever.get_relevant_documents(text)
        pdf_context = "\n".join([doc.page_content for doc in top_pdf_chunks])

        prompt = PROMPT_TEMPLATE.format(
            parts_context=parts_context,
            failure_context=text,
            KXE_brochure_context=pdf_context,
            XE_brochure_context=pdf_context,
            XE_specs_context=pdf_context,
        )
        response = llm.predict(prompt)
        failure_to_part.append({"Failure": text, "Response": response})

    return failure_to_part


def supervised_inference(failure_list, parts_list, pdf_chunks):
    """Perform supervised mapping using KNN."""
    parts_vectorstore, pdf_vectorstore, clf, llm = load_models()
    embedding_model = parts_vectorstore.embedding_function  # Assuming same

    failure_to_part = []
    for failure in failure_list[:MAX_FAILURES_TO_PROCESS]:
        failure_emb = embedding_model.embed_query(failure)
        predicted_label = clf.predict([failure_emb])[0]
        if predicted_label in pdf_chunks:
            retriever = pdf_vectorstore.as_retriever(search_type="similarity", search_kwargs={"k": 3})
            relevant_docs = retriever.get_relevant_documents(failure)
            context_snippets = "\n".join([doc.page_content for doc in relevant_docs])
        else:
            context_snippets = ""

        prompt = PROMPT_TEMPLATE.format(
            parts_context="\n".join(parts_list[:20]),
            failure_context=failure,
            KXE_brochure_context=context_snippets,
            XE_brochure_context=context_snippets,
            XE_specs_context=context_snippets,
        )
        response = llm.predict(prompt)
        failure_to_part.append({"Failure": failure, "Response": response})

    return failure_to_part
