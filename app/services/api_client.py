import logging
from fastapi import APIRouter, HTTPException, Response
from fastapi.responses import JSONResponse
from app.models.chat import ChatRequest, ChatResponse
import time
import requests

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

router = APIRouter()
@router.get("/health")
def healthcheck(response: Response):
    """
    Check the health of the API, provide server time, a random server ID, and the client's IP address.
    """
    server_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    server_id = 1
    version = "1.0.1"


    # Attempt to fetch the public IP address
    try:
        public_ip = requests.get("https://api.ipify.org?format=json", timeout=10).json()
    except requests.RequestException:
        public_ip = 'Unavailable'

    logging.info(f"Health check accessed at {server_time} with server ID {server_id} from IP {public_ip}")


    return {
        "status": "ok",
        "server_time": server_time,
        "server_id": server_id,
        **public_ip,
        "version": version
    }


@router.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    logger.info(f"Received message: {request.message}")
    # For demonstration, echo the message back as the response
    return ChatResponse(response=f"Echo: {request.message}")
