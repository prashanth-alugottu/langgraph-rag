import os
import uuid
import pickle
from langchain_community.document_loaders import PyPDFLoader, TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from utils.config import *

import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS

# from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings


def upload_file(file_bytes: bytes, filename: str):
    # 1️⃣ Save file temporarily
    os.makedirs(config.upload_dir, exist_ok=True)
    os.makedirs(config.persist_directory, exist_ok=True)

    temp_path = os.path.join(
        config.upload_dir,
        f"{uuid.uuid4()}_{filename}"
    )

    with open(temp_path, "wb") as f:
        f.write(file_bytes)

    # 2️⃣ Load file based on extension
    ext = filename.lower().split(".")[-1]

    if ext == "pdf":
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

    elif ext == "txt":
        loader = TextLoader(temp_path, encoding="utf-8")
        docs = loader.load()

    else:
        return {
            "filename": filename,
            "error": "Unsupported file type"
        }

    # 3️⃣ Split documents
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    chunks = splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings(
        model=config.embedding_model_name,
        api_key=config.open_api_key
    )
    
    # embeddings = AzureOpenAIEmbeddings(
    # azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    # api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    # azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    # api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # )


    # Load or create FAISS safely
    faiss_index_path = config.persist_directory
    if os.path.exists(os.path.join(faiss_index_path, "index.faiss")):
        vector_store = get_vector_db()
    else:
        vector_store = FAISS.from_documents(chunks, embeddings)

        

    # Add documents ONLY after vector_store exists
    vector_store.add_documents(chunks)

    # Persist FAISS
    vector_store.save_local(faiss_index_path)
    return {
        "filename": filename,
        "chunks_added": len(chunks)
    }
    
def get_vector_db():
    embeddings = OpenAIEmbeddings(
        model=config.embedding_model_name,
        api_key=config.open_api_key
    )
    
    # embeddings = AzureOpenAIEmbeddings(
    # azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT"),
    # api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    # azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    # api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    # )
     
    return FAISS.load_local(
            config.persist_directory,
            embeddings,
            allow_dangerous_deserialization=True
        )
    