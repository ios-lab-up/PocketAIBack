from pydantic import BaseModel

class ChatRequest(BaseModel):
    message: str

class ChatResponse(BaseModel):
    response: str

class PredictResponse(BaseModel):
    intent: str  # Predicted intent
    confidence: float
