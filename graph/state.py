
from typing import TypedDict, List
from langchain_core.documents import Document


class RAGState(TypedDict):
    query: str
    retrieved_docs: List[Document]
    source: List[Document]
    answer: str
    grounded: bool