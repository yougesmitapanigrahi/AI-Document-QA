"""Streamlit interface for asking questions about uploaded PDF documents."""

import hashlib
from io import BytesIO
from typing import Any

import streamlit as st

from src.document_processor import extract_text_from_pdf, split_text_into_chunks
from src.qa_engine import generate_answer, load_qa_model
from src.question_analyzer import analyze_question, get_invalid_page_references
from src.query_router import retrieve_chunks_for_analysis
from src.vector_store import (
    create_vector_store,
    load_embedding_model,
    search_vector_store,
)


APP_TITLE = "AI-Powered Document Question Answering System"


def clear_document_state() -> None:
    """Clear document-specific data while keeping loaded models available."""
    for key in (
        "document_hash",
        "document_name",
        "document_bytes",
        "document_pages",
        "document_chunks",
        "vector_index",
        "last_question",
        "last_result",
    ):
        st.session_state[key] = None


st.set_page_config(page_title=APP_TITLE, layout="wide")

state_defaults = {
    "document_hash": None,
    "document_name": None,
    "document_bytes": None,
    "document_pages": None,
    "document_chunks": None,
    "vector_index": None,
    "embedding_model": None,
    "qa_tokenizer": None,
    "qa_model": None,
    "last_question": None,
    "last_result": None,
}
for state_key, default_value in state_defaults.items():
    st.session_state.setdefault(state_key, default_value)

st.title(APP_TITLE)
st.write(
    "Upload a PDF, then ask questions using information from its pages. "
    "Answers are generated from the document excerpts retrieved for each question."
)

uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type=["pdf"],
    help="The document is processed locally by the configured Python models.",
)

if uploaded_file is None:
    clear_document_state()
    st.info("Upload a PDF to get started.")
    st.stop()

uploaded_bytes = uploaded_file.getvalue()
uploaded_hash = hashlib.sha256(uploaded_bytes).hexdigest()

# Reuse extracted text and embeddings when the same file is uploaded again.
if uploaded_hash != st.session_state.document_hash:
    clear_document_state()
    try:
        with st.spinner("Reading the PDF and extracting page text..."):
            pages = extract_text_from_pdf(BytesIO(uploaded_bytes))

        if not pages:
            st.error("This PDF contains no pages.")
            st.stop()
        if not any(page["text"].strip() for page in pages):
            st.error("This PDF has no extractable text. Scanned PDFs need OCR.")
            st.stop()

        with st.spinner("Splitting document text into searchable chunks..."):
            chunks = split_text_into_chunks(pages)
        if not chunks:
            st.error("No text chunks could be created from this PDF.")
            st.stop()

        # Save this stage so a model-loading failure does not reprocess the PDF.
        st.session_state.document_hash = uploaded_hash
        st.session_state.document_name = uploaded_file.name
        st.session_state.document_bytes = uploaded_bytes
        st.session_state.document_pages = pages
        st.session_state.document_chunks = chunks
    except (OSError, RuntimeError, ValueError) as exc:
        st.error(f"Could not process this PDF: {exc}")
        st.stop()

# The same PDF contents may be uploaded under a different filename.
st.session_state.document_name = uploaded_file.name
st.session_state.document_bytes = uploaded_bytes

st.subheader("Uploaded document")
document_column, page_column = st.columns(2)
document_column.metric("Document", st.session_state.document_name)
page_column.metric("Total pages", len(st.session_state.document_pages))

st.subheader("Ask a question")
question = st.text_input(
    "Question about this PDF",
    placeholder="For example: What are the main findings?",
    key="question",
)


def run_semantic_search() -> list[dict[str, Any]]:
    """Load cached search resources on demand and search the current document."""
    if st.session_state.embedding_model is None:
        with st.spinner("Loading the document embedding model..."):
            st.session_state.embedding_model = load_embedding_model()

    if st.session_state.vector_index is None:
        with st.spinner("Building the document search index..."):
            st.session_state.vector_index = create_vector_store(
                st.session_state.document_chunks,
                st.session_state.embedding_model,
            )

    with st.spinner("Searching the document for relevant excerpts..."):
        return search_vector_store(
            question,
            st.session_state.embedding_model,
            st.session_state.vector_index,
            st.session_state.document_chunks,
        )


if st.button("Get Answer", type="primary"):
    if not question.strip():
        st.warning("Enter a question before requesting an answer.")
    else:
        st.session_state.last_result = None
        try:
            total_pages = len(st.session_state.document_pages)
            analysis = analyze_question(
                question,
                total_pages=total_pages,
            )
            if analysis["category"] == "page_count":
                result = {
                    "answer": f"This PDF contains {total_pages} pages.",
                    "source_pages": [],
                }
            else:
                page_reference_issues = None
                if analysis["category"] == "page_lookup":
                    page_reference_issues = get_invalid_page_references(
                        question,
                        total_pages,
                    )

                if page_reference_issues:
                    requested_pages = ", ".join(
                        str(page_number) for page_number in page_reference_issues
                    )
                    st.error(
                        "The requested page(s) do not exist in this PDF: "
                        f"{requested_pages}."
                    )
                    st.session_state.last_question = question
                    st.stop()

                if (
                    analysis["category"] == "page_lookup"
                    and page_reference_issues is not None
                    and not analysis["page_numbers"]
                ):
                    st.error(
                        "The requested page range is too large to process safely."
                    )
                    st.session_state.last_question = question
                    st.stop()

                retrieved_chunks = retrieve_chunks_for_analysis(
                    analysis,
                    st.session_state.document_chunks,
                    run_semantic_search,
                )
                if retrieved_chunks and (
                    st.session_state.qa_tokenizer is None
                    or st.session_state.qa_model is None
                ):
                    with st.spinner("Loading the question-answering model..."):
                        tokenizer, qa_model = load_qa_model()
                        st.session_state.qa_tokenizer = tokenizer
                        st.session_state.qa_model = qa_model

                with st.spinner("Generating an answer from the document excerpts..."):
                    result = generate_answer(
                        question,
                        retrieved_chunks,
                        st.session_state.qa_tokenizer,
                        st.session_state.qa_model,
                        total_pages=len(st.session_state.document_pages),
                    )

            st.session_state.last_question = question
            st.session_state.last_result = result
        except (ImportError, OSError, RuntimeError, TypeError, ValueError) as exc:
            st.error(f"Could not generate an answer: {exc}")

if (
    st.session_state.last_result is not None
    and st.session_state.last_question == question
):
    st.markdown("### Answer")
    st.write(st.session_state.last_result["answer"])

    source_pages = st.session_state.last_result["source_pages"]
    if source_pages:
        page_list = ", ".join(str(page_number) for page_number in source_pages)
        st.markdown(f"**Source pages:** {page_list}")
    else:
        st.caption("No source pages were available for this answer.")
