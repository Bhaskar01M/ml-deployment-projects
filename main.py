from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import os
from dotenv import load_dotenv
from rag_chatbot_demo import RAGChatbotDemo
import tempfile
from pathlib import Path

# Load environment
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="RAG Chatbot API",
    description="Retrieval-Augmented Generation Chatbot",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize chatbot
chatbot = RAGChatbotDemo()

# Root endpoint
@app.get("/")
def read_root():
    """Welcome endpoint"""
    return {
        "message": "Welcome to RAG Chatbot API!",
        "endpoints": {
            "ask": "/ask",
            "docs": "/docs",
            "health": "/health"
        }
    }

# Health check
@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "RAG Chatbot API"
    }

# Main endpoint: Ask a question
@app.post("/ask")
def ask_question(question: str, k: int = 5):
    """
    Ask the RAG chatbot a question.
    
    - **question**: Your question (required)
    - **k**: Number of chunks to retrieve (default: 5)
    
    Returns: Answer with retrieved passages
    """
    if not question or not question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        answer, chunks = chatbot.answer_question_demo(question, k=k)
        
        # Format response
        passages = [
            {
                "id": chunk["chunk_id"],
                "text": chunk["text"],
                "similarity": round(chunk["similarity"], 4)
            }
            for chunk in chunks
        ]
        
        return {
            "question": question,
            "answer": answer,
            "passages": passages,
            "total_passages_retrieved": len(passages)
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint: Get chatbot info
@app.get("/info")
def get_info():
    """Get information about loaded documents"""
    return {
        "total_embeddings": len(chatbot.retriever.embeddings),
        "embedding_dimension": 1536,
        "model": "text-embedding-3-small",
        "documents_loaded": 3,
        "chunks": len(chatbot.retriever.embeddings)
    }

# Endpoint: Interactive chat (streaming-like)
@app.post("/chat")
def chat(message: str):
    """
    Chat endpoint - same as /ask but different naming
    """
    return ask_question(message)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)