from fastapi.testclient import TestClient
from app.main import app
import time
import requests
import re

client= TestClient(app)

def test_helthcheck ():
    response=client.get("/api/v1/health")
    assert response.status_code == 200

    server_time = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
    server_id = 1
    version = "1.0.1"
    try:
        public_ip = requests.get("https://api.ipify.org?format=json", timeout=10).json()
    except requests.RequestException:
        public_ip = 'Unavailable'
    
    time_pattern = r"^\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}$"
    assert re.match(time_pattern, response.json()["server_time"]), f"Invalid server_time format: {response.json()['server_time']}"

    assert response.json() == {
        "status": "ok",
        "server_time": response.json()["server_time"],
        "server_id": server_id,
        **public_ip,
        "version": version
    }

def test_post_healthcheck():
    response = client.post("/api/v1/health")
    assert response.status_code == 405

def test_put_healthcheck():
    response = client.put("/api/v1/health")
    assert response.status_code == 405

def test_delete_healthcheck():
    response = client.delete("/api/v1/health")
    assert response.status_code == 405

def test_chat():
    response=client.post("/api/v1/chat?user_id=0267688", json={"message": "Hello"} )
    assert response.status_code == 200
    data = response.json()
    assert "response" in data
    assert isinstance(data["response"], str)

def test_get_chat():
    response = client.get("/api/v1/chat?user_id=0267688")
    assert response.status_code == 405

def test_put_chat():
    response = client.put("/api/v1/chat?user_id=0267688")
    assert response.status_code == 405

def test_delete_chat():
    response = client.delete("/api/v1/chat?user_id=0267688")
    assert response.status_code == 405

def test_wrongid_chat():
    response=client.post("/api/v1/chat?user_id=01", json={"message": "Hello"} )
    assert response.status_code == 422

def test_wrongterm_chat():
    response=client.post("/api/v1/chat?user_id=0267688&term=a", json={"message": "Hello"} )
    assert response.status_code == 422


