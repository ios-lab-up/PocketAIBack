import sys
import os
# Add the project root to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests
from app.agents.intent_agent import IntentBasedAgent

# Base URL for API endpoints
BASE_URL = "http://127.0.0.1:8000/api/v1"

# Update the base URL in the IntentBasedAgent class to match the test environment
IntentBasedAgent._base_url = BASE_URL
# Also update the LLM API URL
IntentBasedAgent._llm_api_url = f"{BASE_URL}/api/chat/completions"

# Health check test
try:
    health_response = requests.get(f"{BASE_URL}/health")
    print("Health Check Response:", health_response.json())
except Exception as e:
    print(f"Health check failed: {e}")

# Chat test
try:
    chat_data = {"message": "Hello, how are you?"}
    chat_response = requests.post(f"{BASE_URL}/chat?user_id=12345&term=test", json=chat_data)
    print("Chat Response:", chat_response.json())
except Exception as e:
    print(f"Chat test failed: {e}")

# Handle a user query using IntentBasedAgent
try:
    response = IntentBasedAgent.handle_query(
        user_query="¿Cuáles son mis calificaciones del semestre pasado?",
        user_id="0260694",
        term="1252"
    )
    print("Agent Response:", response)
except Exception as e:
    print(f"Agent test failed: {str(e)}")