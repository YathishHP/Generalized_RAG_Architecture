# main.py

import os
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from dotenv import load_dotenv

from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI(
    title="DocuMind AI API",
    description="Upload PDFs and ask questions via REST",
    version="1.0.0"
)

# —————————————— Global Components ——————————————

# 1. Embedding model & in-memory vector store
EMBEDDING_MODEL = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-small-en-v1.5",
    model_kwargs={"device": "cpu"},
    encode_kwargs={"normalize_embeddings": True}
)
VECTOR_DB = InMemoryVectorStore(EMBEDDING_MODEL)

# 2. LLM & Prompt Template for QA
LANGUAGE_MODEL = ChatOpenAI(
    base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
    api_key=os.getenv("OPENROUTER_API_KEY"),
    model="deepseek/deepseek-chat-v3-0324:free",
    temperature=0.7
)
PROMPT_TEMPLATE = """
You are an expert research assistant. Use the provided context to answer the query.
If unsure, state that you don't know. Be concise and factual (max 3 sentences).

Query: {user_query}
Context: {document_context}
Answer:
"""
CONVERSATION_CHAIN = (
    ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    | LANGUAGE_MODEL
    | StrOutputParser()
)

# —————————————— Helper Functions ——————————————

def save_pdf(upload: UploadFile) -> str:
    """Save uploaded PDF to disk and return file path."""
    os.makedirs("document_store/pdfs", exist_ok=True)
    file_path = os.path.join("document_store/pdfs", upload.filename)
    with open(file_path, "wb") as f:
        f.write(upload.file.read())
    return file_path

def load_and_index_pdf(file_path: str):
    """Load, split, and index PDF contents into VECTOR_DB."""
    loader = PDFPlumberLoader(file_path)
    raw_docs = loader.load()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, add_start_index=True
    )
    chunks = splitter.split_documents(raw_docs)
    VECTOR_DB.add_documents(chunks)

def retrieve_relevant_docs(query: str, k: int = 4):
    """Return top-k similar document chunks."""
    return VECTOR_DB.similarity_search(query, k=k)

def generate_answer(user_query: str, docs: list) -> str:
    """Build context string and invoke the QA chain."""
    context = "\n\n".join(doc.page_content for doc in docs)
    return CONVERSATION_CHAIN.invoke({
        "user_query": user_query,
        "document_context": context
    })

# —————————————— API Models ——————————————

class QueryRequest(BaseModel):
    question: str

# —————————————— Endpoints ——————————————

@app.post("/upload-pdf", summary="Upload a PDF and index its contents")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")
    path = save_pdf(file)
    load_and_index_pdf(path)
    return {"detail": f"Indexed '{file.filename}' successfully."}

@app.post("/query", summary="Ask a question about uploaded documents")
async def query_docs(payload: QueryRequest):
    if VECTOR_DB._collection.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No documents indexed. Upload a PDF first via /upload-pdf."
        )
    docs = retrieve_relevant_docs(payload.question)
    answer = generate_answer(payload.question, docs)
    return {"answer": answer}

@app.get("/health", summary="Health check")
async def health():
    return {"status": "ok"}

# —————————————— Run with Uvicorn ——————————————
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
