from typing import Optional
import requests
import logging
import json
import joblib
import urllib.parse
from app.utils.preprocess import clean_text
from app.core.settings import settings

logger = logging.getLogger(__name__)

class IntentBasedAgent:
    """
    A utility-based agent that uses an intent model to classify user queries,
    performs an HTTP request, and generates a response using direct API calls.
    """

    # Model paths and configuration
    _vectorizer = joblib.load("../data/models/vectorizer.pkl")
    _classifier = joblib.load("../data/models/intent_classifier.pkl")
    _label_encoder = joblib.load("../data/models/label_encoder.pkl")
    _base_url = settings.API_BASE_URL
    _timeout = 30
    _llm_model = "llama3.2:3b"
    _llm_api_url = f"{settings.API_BASE_URL}/api/chat/completions"
    _llm_api_key = settings.API_KEY
    _llm_headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.API_KEY}"
    }

    @classmethod
    def classify_intent(cls, message: str):
        """
        Classify the intent of a message using the intent classifier.

        Args:
            message (str): The user query.

        Returns:
            tuple: (predicted_intent, confidence)
        """
        try:
            clean_query = clean_text(message)
            query_vectorized = cls._vectorizer.transform([clean_query])
            predicted_label = cls._classifier.predict(query_vectorized)[0]
            predicted_intent = cls._label_encoder.inverse_transform([predicted_label])[0]
            probabilities = cls._classifier.predict_proba(query_vectorized)[0]
            confidence = max(probabilities)

            logger.info(f"Predicted intent: {predicted_intent}, confidence: {confidence}")
            return predicted_intent, confidence
        except Exception as e:
            logger.error(f"Error classifying intent: {e}")
            raise RuntimeError("Failed to classify intent.")

    @classmethod
    def make_request(cls, intent: str, user_id: str, term: Optional[str] = None) -> dict:
        try:
            # Build query parameters
            params = {"action": intent, "user_id": user_id}
            if term:
                params["term"] = term

            # Build the complete URL with parameters
            base_url = f"{cls._base_url}/student-data"
            full_url = f"{base_url}?{urllib.parse.urlencode(params)}"

            logger.info(f"Making request to: {full_url}")

            response = requests.get(full_url, timeout=cls._timeout)
            
            # Log response details for debugging
            logger.info(f"Response status code: {response.status_code}")
            logger.info(f"Response headers: {dict(response.headers)}")
            content_preview = response.text[:100] if response.text else "EMPTY RESPONSE"
            logger.info(f"Response content preview: {content_preview}")
            
            response.raise_for_status()

            # Handle empty response
            if not response.text or not response.text.strip():
                logger.error("Received empty response from API")
                return {"error": "Empty response from API"}
                
            try:
                return response.json()
            except json.JSONDecodeError as e:
                logger.error(f"JSON parse error: {e}, Raw content: {response.text[:200]}")
                return {"error": f"Invalid JSON response: {str(e)}"}
                
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            return {"error": f"Request failed: {str(e)}"}
            
    @classmethod
    def generate_response(cls, user_query: str, api_data: dict) -> str:
        """
        Generate a response using direct API call to LLM service based on the API data.

        Args:
            user_query (str): The original user query.
            api_data (dict): The data retrieved from the API.

        Returns:
            str: The generated response from the LLM.
        """
        try:
            # Create the prompt for the model
            prompt = (
                "Eres un asistente que responde en español basado en datos de una API.\n"
                f"Consulta del usuario: {user_query}\n\n"
                f"Datos de la API:\n{json.dumps(api_data, indent=2)}\n\n"
                "Genera una respuesta clara y útil para el usuario basada en estos datos."
            )
            
            logger.info(f"Prompt for LLM: {prompt}")

            # Prepare the payload for the API call
            payload = {
                "model": cls._llm_model,
                "messages": [
                    {
                        "role": "system",
                        "content": prompt
                    }
                ]
            }
            
            # Make the API call
            response = requests.post(
                cls._llm_api_url,
                headers=cls._llm_headers,
                json=payload,
                timeout=cls._timeout
            )
            
            response.raise_for_status()
            result = response.json()
            
            # Extract content from the API response
            if "choices" in result and len(result["choices"]) > 0:
                content = result["choices"][0]["message"]["content"]
                logger.info(f"LLM API response content: {content}")
                return content
            else:
                logger.error("Unexpected API response format")
                return "Lo siento, no pude generar una respuesta válida en este momento."
                
        except requests.RequestException as e:
            logger.error(f"Error calling LLM API: {e}")
            return f"Lo siento, no pude generar una respuesta en este momento. Error: {str(e)}"
        except Exception as e:
            logger.error(f"Failed to generate response with LLM: {e}")
            return "Lo siento, no pude generar una respuesta en este momento."

    @classmethod
    def handle_query(cls, user_query: str, user_id: str, term: Optional[str] = None) -> str:
        """
        Handle a user query by classifying the intent, making an API request, and generating a response.

        Args:
            user_query (str): The user's query.
            user_id (str): The user identifier.
            term (str, optional): Optional term for term-specific actions.

        Returns:
            str: The generated response for the user.
        """
        try:
            # Classify the intent of the user's query
            intent, confidence = cls.classify_intent(user_query)

            # Perform the API request
            api_response = cls.make_request(intent, user_id, term)
            if "error" in api_response:
                return f"Error al realizar la consulta: {api_response['error']}"

            # Generate a response based on the API data
            return cls.generate_response(user_query, api_response)
        except Exception as e:
            logger.error(f"Error handling query: {e}")
            return "Lo siento, no pude procesar tu consulta en este momento."