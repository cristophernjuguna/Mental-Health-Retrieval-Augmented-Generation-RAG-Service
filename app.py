from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_service import ask

app = FastAPI(title="Mental health RAG API")

class QuestionRequest(BaseModel):
    question: str
    
class QuestionResponse(BaseModel):
    answer: str 
    sources: list

@app.get("/")
def root():
    return {"message": "RAG API is running"}

@app.post("/ask", response_model=QuestionResponse)
def ask_question(request: QuestionRequest):
    try:
        answer, sources = ask(request.question)
        
        return QuestionResponse(
            answer=answer,
            sources=sources
            )
        
    except Exception as e:
            raise HTTPException(status_code=500, detail=str(e)) 