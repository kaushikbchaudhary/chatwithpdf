"""Streamlit UI for chatting with uploaded PDFs."""

from __future__ import annotations

import streamlit as st

from rag import AppConfig, build_conversational_chain, build_document_chunks, build_vectorstore
from rag.embeddings import build_embeddings


st.set_page_config(page_title="Chat with your PDFs", layout="wide")
st.title("📚 Chat with your PDFs")

if "chain" not in st.session_state:
    st.session_state.chain = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "messages" not in st.session_state:
    st.session_state.messages = []


@st.cache_resource(show_spinner=False)
def load_config() -> AppConfig:
    return AppConfig.from_env()


try:
    config = load_config()
except RuntimeError as exc:  # Missing config, show guidance
    st.error(str(exc))
    st.stop()


st.sidebar.header("Document setup")
uploaded = st.sidebar.file_uploader(
    "Upload PDF files",
    accept_multiple_files=True,
    type=["pdf"],
)
process_clicked = st.sidebar.button("Build knowledge base", use_container_width=True)

if process_clicked:
    if not uploaded:
        st.sidebar.warning("Please upload at least one PDF before processing.")
    else:
        with st.spinner("Processing documents..."):
            blobs = [(file.name, file.read()) for file in uploaded]
            documents = build_document_chunks(blobs, config)
            if not documents:
                st.warning("No readable text was found in the uploaded PDFs.")
            else:
                try:
                    embeddings = build_embeddings(config)
                    vector_store = build_vectorstore(documents, embeddings)
                    st.session_state.chain = build_conversational_chain(
                        vector_store, config
                    )
                    st.session_state.chat_history = []
                    st.session_state.messages = []
                    st.success("Knowledge base ready. Ask away!")
                except Exception as exc:  # Catch API issues so the app keeps running
                    st.error(
                        "Failed to build the knowledge base. "
                        "Check your API credentials, network connection, "
                        "and provider quota.\n\nDetails: "
                        f"{exc}"
                    )

if st.session_state.chain is None:
    st.info("Upload PDFs and click 'Build knowledge base' to start chatting.")
    st.stop()

for role, content in st.session_state.messages:
    with st.chat_message(role):
        st.markdown(content)

prompt = st.chat_input("Ask a question about the uploaded PDFs")

if prompt:
    st.session_state.messages.append(("user", prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    chain = st.session_state.chain

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = chain.invoke(
                {"question": prompt, "chat_history": st.session_state.chat_history}
            )
            answer: str = response.get("answer", "I couldn't find an answer.")
            sources = response.get("source_documents", [])
            st.markdown(answer)
            if sources:
                with st.expander("Sources"):
                    for doc in sources:
                        meta = doc.metadata or {}
                        source = meta.get("source", "Unknown")
                        page = meta.get("page")
                        label = f"{source}"
                        if page is not None:
                            label += f" — page {page}"
                        st.write(f"- {label}")

    st.session_state.chat_history.append((prompt, answer))
    st.session_state.messages.append(("assistant", answer))
