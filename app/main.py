from fastapi import FastAPI
from app.services.api_client import router as api_client_router

app = FastAPI()

app.include_router(api_client_router, prefix="/api/v1")
