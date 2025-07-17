# main.py

import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_openai import ChatOpenAI

load_dotenv()

app = FastAPI(
    title="QuadCore Code Ally API",
    description="RESTful API for your multi-mode coding assistant",
    version="1.0.0"
)

# Map the human-friendly names to model IDs
MODEL_MAP = {
    "Gemma 2B": "google/gemma-2-9b-it:free",
    "DeepSeek Chat": "deepseek/deepseek-chat-v3-0324:free",
    "Mistral Instruct": "cognitivecomputations/dolphin3.0-mistral-24b:free"
}

# Reuse the same prompt template from your Streamlit app
PROMPT_TEMPLATE = """
You are a {mode}.
Provide your response in English.
Be concise, correct, and—where applicable—show code examples or debugging print statements.

Question:
{question}
"""

class QueryRequest(BaseModel):
    mode: str      # one of ["Python Expert","Debugging Assistant","Code Documentation","Solution Design"]
    model: str     # one of the keys in MODEL_MAP
    question: str  # the user’s prompt

@app.post("/chat")
async def chat_endpoint(payload: QueryRequest):
    # Validate model choice
    if payload.model not in MODEL_MAP:
        raise HTTPException(status_code=400, detail=f"Unsupported model '{payload.model}'")
    
    # Initialize the LLM
    llm = ChatOpenAI(
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=os.getenv("OPENROUTER_API_KEY"),
        model=MODEL_MAP[payload.model],
        temperature=0.3
    )
    
    # Build the prompt‐>LLM‐>parser pipeline
    prompt_chain = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    pipeline = prompt_chain | llm | StrOutputParser()
    
    # Invoke the pipeline with mode & question
    answer = pipeline.invoke({
        "mode": payload.mode,
        "question": payload.question
    })
    
    return {"response": answer}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=int(os.getenv("PORT", 8000)),
        reload=True
    )
