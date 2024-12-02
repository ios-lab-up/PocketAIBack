from fastapi import FastAPI
from app.router import router as api_client_router

app = FastAPI()

app.include_router(api_client_router, prefix="/api/v1")
