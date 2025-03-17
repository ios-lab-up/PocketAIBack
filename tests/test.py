import requests

BASE_URL = "http://127.0.0.1:8000/api/v1"

# Health check test
health_response = requests.get(f"{BASE_URL}/health")
print("Health Check Response:", health_response.json())

# Chat test
chat_data = {"message": "Hello, how are you?"}
chat_response = requests.post(f"{BASE_URL}/chat?user_id=12345&term=test", json=chat_data)
print("Chat Response:", chat_response.json())
