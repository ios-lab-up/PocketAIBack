from fastapi.testclient import TestClient
from app.main import app
from app.agents.intent_agent import IntentBasedAgent
import time
import requests

client= TestClient(app)

def test_helthcheck ():
    response=client.get("/api/v1/health")
    assert response.status_code == 200
    server_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    server_id = 1
    version = "1.0.1"

    # Attempt to fetch the public IP address
    try:
        public_ip = requests.get("https://api.ipify.org?format=json", timeout=10).json()
    except requests.RequestException:
        public_ip = 'Unavailable'
    assert response.json() == {
        "status": "ok",
        "server_time": server_time,
        "server_id": server_id,
        **public_ip,
        "version": version
    }

def test_chat():
    response=client.post("/api/v1/chat?user_id=0267688&term=1252", json={"message": "Hello"} )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)
    assert "Hola" in data["response"]


