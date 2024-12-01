from fastapi import APIRouter, HTTPException
from fastapi.responses import JSONResponse
from app.models.chat import ChatRequest, ChatResponse

router = APIRouter()

@router.get("/health")
async def health_check():
    return JSONResponse(content={
        "status": "healthy",
        "message": "Service is running smoothly."
    })

@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    # For demonstration, echo the message back as the response
    return ChatResponse(response=f"Echo: {request.message}")
