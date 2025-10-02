"""PDF ingestion and chunking utilities."""

from __future__ import annotations

import io
from typing import Iterable, List, Sequence, Tuple

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader

from .config import AppConfig


UploadedBlob = Tuple[str, bytes]


def _pdf_bytes_to_documents(name: str, data: bytes) -> List[Document]:
    reader = PdfReader(io.BytesIO(data))
    documents: List[Document] = []

    for page_number, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        text = text.strip()
        if not text:
            continue
        documents.append(
            Document(
                page_content=text,
                metadata={"source": name, "page": page_number},
            )
        )

    return documents


def build_document_chunks(
    uploaded_files: Sequence[UploadedBlob],
    config: AppConfig,
) -> List[Document]:
    """Turn uploaded PDFs into chunked LangChain documents."""

    raw_documents: List[Document] = []
    for name, data in uploaded_files:
        raw_documents.extend(_pdf_bytes_to_documents(name, data))

    if not raw_documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.chunk_size,
        chunk_overlap=config.chunk_overlap,
        separators=["\n\n", "\n", " ", ""],
    )
    return splitter.split_documents(raw_documents)
