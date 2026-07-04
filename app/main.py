"""
RAG Document Chatbot - FastAPI Backend (Ollama version)
"""
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from rag_pipeline import process_pdf, ask_question, get_chat_history, clear_session

app = FastAPI(
    title="RAG Document Chatbot",
    description="Upload a PDF and chat with it using local AI (Ollama).",
    version="2.0.0"
)


class QuestionInput(BaseModel):
    question: str

    class Config:
        json_schema_extra = {"example": {"question": "What is this document about?"}}


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max 10MB.")

    result = process_pdf(contents)

    if not result["success"]:
        raise HTTPException(status_code=422, detail=result["error"])

    return {
        "message": "PDF processed successfully.",
        "filename": file.filename,
        "chunks": result["chunks"],
        "characters": result["characters"]
    }


@app.post("/chat")
async def chat(input: QuestionInput):
    if not input.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    result = ask_question(input.question)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["error"])

    return {"answer": result["answer"]}


@app.get("/history")
def history():
    return {"history": get_chat_history()}


@app.post("/clear")
def clear():
    clear_session()
    return {"message": "Session cleared."}
