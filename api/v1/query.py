from fastapi import APIRouter
from pydantic import BaseModel
from graph.graph import build_multirag_graph

router = APIRouter(prefix="/api/v1")

class QueryRequest(BaseModel):
    query: str

@router.post("/query")
def query(req: QueryRequest):
    query=req.query
    return run_rag(query)
   
   
def run_rag(query):
    graph = build_multirag_graph()

    state = {
        "query": query,
    }

    result = graph.invoke(state)

    [(doc.page_content) for doc in result["retrieved_docs"]]
        
    return {
        "answer":result["answer"],
        "source":[(doc.page_content) for doc in result["retrieved_docs"]]
    }


