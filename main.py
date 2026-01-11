from fastapi import FastAPI
from api.v1 import upload
from api.v1 import query

app = FastAPI(title="Agentic RAG API")

app.include_router(upload.router)
app.include_router(query.router)

@app.get("/")
def health():
    return {"status": "ok"}
